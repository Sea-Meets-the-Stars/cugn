"""
    qg_drifters.jl

Advect virtual Lagrangian drifters through QG velocity fields using Drifters.jl.

Depends on: qg_grid.jl (must be included first)

Usage:
    include("qg_grid.jl")
    include("qg_drifters.jl")
    results = run_drifters(; t_start=5001, n_days=30, n_drifters_per_side=64)
"""

using Drifters
using Drifters: DataFrames, DataFrame, groupby, combine, nrow

"""
    load_uv_arrays(nc_path::String, t_idx::Int, dx::Float64, dy::Float64; lev::Int=1)

Load a single velocity snapshot from the NetCDF file, normalize to grid units/s,
and apply C-grid staggering shift.

Returns: (u_grid::Matrix{Float64}, v_grid::Matrix{Float64})
"""
function load_uv_arrays(nc_path::String, t_idx::Int, dx::Float64, dy::Float64; lev::Int=1)
    ds = NCDataset(nc_path, "r")
    u_phys = ds["u"][:, :, lev, t_idx]  # m/s
    v_phys = ds["v"][:, :, lev, t_idx]  # m/s
    close(ds)

    # Normalize to grid units per second
    u_grid = u_phys ./ dx
    v_grid = v_phys ./ dy

    # Apply C-grid shift: pyqg velocities are at cell centers,
    # but Drifters.jl expects C-grid staggering where u[i,j] is at (i-1, j-1/2)
    # and v[i,j] is at (i-1/2, j-1)
    to_C_grid!(u_grid, dims=1)
    to_C_grid!(v_grid, dims=2)

    return u_grid, v_grid
end

"""
    deploy_drifters(n_per_side::Int, nx::Int)

Create initial drifter positions on a regular sub-grid.
Positions are in grid-index coordinates (0 to nx).

Returns: (x0::Vector{Float64}, y0::Vector{Float64})
"""
function deploy_drifters(n_per_side::Int, nx::Int)
    spacing = nx / n_per_side
    offsets = range(spacing/2, stop=nx - spacing/2, length=n_per_side)

    x0 = Float64[]
    y0 = Float64[]
    for yy in offsets, xx in offsets
        push!(x0, xx)
        push!(y0, yy)
    end
    return x0, y0
end

"""
    run_drifters(; nc_path=nothing, t_start=5001, n_days=30,
                   n_per_side=64, lev=1, record_interval=1)

Run drifter advection through the QG flow field.

Arguments:
- nc_path: path to QG NetCDF file (defaults to \$OS_DATA/QG/...)
- t_start: starting time index in the NetCDF (1-based)
- n_days: number of days to advect
- n_per_side: drifters per side of the deployment grid (total = n_per_side^2)
- lev: vertical level (1=upper, 2=lower)
- record_interval: record positions every N days

Returns: NamedTuple with fields:
- trajectories: DataFrame with (ID, x, y, t, x_m, y_m) columns
- dx: grid spacing in meters
- nx: grid size
- n_drifters: total number of drifters
"""
function run_drifters(; nc_path=nothing, t_start::Int=5001, n_days::Int=30,
                       n_per_side::Int=64, lev::Int=1, record_interval::Int=1)
    if isnothing(nc_path)
        nc_path = joinpath(ENV["OS_DATA"], "QG", "QGModelOutput20years.nc")
    end

    # Get grid info
    ds = NCDataset(nc_path, "r")
    x_coord = ds["x"][:]
    nx = length(x_coord)
    dx = x_coord[2] - x_coord[1]
    dy = dx  # square grid
    close(ds)

    dt_seconds = 86400.0  # 1 day between snapshots

    # Deploy drifters
    x0, y0 = deploy_drifters(n_per_side, nx)
    n_drifters = length(x0)
    println("Deploying $n_drifters drifters on $(n_per_side)x$(n_per_side) grid")

    # Initialize trajectory storage
    traj = DataFrame(ID=Int[], x=Float64[], y=Float64[], t=Float64[])

    # Record initial positions
    t_abs = 0.0
    for i in 1:n_drifters
        push!(traj, (ID=i, x=x0[i], y=y0[i], t=t_abs))
    end

    # Load first velocity field
    println("Loading initial velocity field (t_idx=$t_start)...")
    u0, v0 = load_uv_arrays(nc_path, t_start, dx, dy; lev=lev)

    # Time-stepping loop
    for day in 1:n_days
        t_idx_next = t_start + day

        # Load next velocity field
        u1, v1 = load_uv_arrays(nc_path, t_idx_next, dx, dy; lev=lev)

        # Build FlowFields (time spans one day in seconds)
        F = FlowFields(u0, u1, v0, v1, [0.0, dt_seconds])

        # Build Individuals
        I = Individuals(F, x0, y0)

        # Integrate one day
        ∫!(I)

        # Update positions (with periodic wrapping)
        # 📌 is Matrix{Vector{Float64}} of size (1, n_drifters)
        # Each I.📌[1,i] is a [x, y] vector
        for i in 1:n_drifters
            pos = I.📌[1, i]
            x0[i] = mod(pos[1], nx)
            y0[i] = mod(pos[2], nx)
        end

        # Record positions
        t_abs += dt_seconds
        if day % record_interval == 0
            for i in 1:n_drifters
                push!(traj, (ID=i, x=x0[i], y=y0[i], t=t_abs))
            end
        end

        # Advance velocity fields
        u0, v0 = u1, v1

        if day % 10 == 0
            println("  Day $day/$n_days complete")
        end
    end

    # Add physical coordinates
    traj.x_m = traj.x .* dx
    traj.y_m = traj.y .* dy

    println("Done. $(nrow(traj)) trajectory records for $n_drifters drifters over $n_days days.")

    return (trajectories=traj, dx=dx, nx=nx, n_drifters=n_drifters)
end
