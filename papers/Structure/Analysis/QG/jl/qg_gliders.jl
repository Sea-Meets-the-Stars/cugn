"""
    qg_gliders.jl

Sample the QG velocity field at prescribed glider positions.

Unlike qg_drifters.jl (which advects particles through the flow),
this script evaluates u and v at externally-specified positions ---
mimicking how real gliders sample the ocean velocity field.

Depends on: qg_grid.jl (must be included first)

Usage:
    include("qg_grid.jl")
    include("qg_gliders.jl")
    result = interpolate_velocity(nc_path, glider_df, 5001; lev=1)
"""

using NCDatasets
using Drifters: DataFrames, DataFrame, nrow
using DelimitedFiles


"""
    load_glider_trajectories(csv_path::String)

Read a glider trajectory CSV file into a DataFrame.
Expected columns: x, y, time, missid.

Returns: DataFrame with columns x, y, time, missid.
"""
function load_glider_trajectories(csv_path::String)
    # Read CSV manually (no CSV.jl dependency)
    data, header = readdlm(csv_path, ',', Any; header=true)
    col_names = vec(strip.(string.(header)))

    expected = ["x", "y", "time", "missid"]
    for col in expected
        if !(col in col_names)
            error("Missing expected column '$col' in $csv_path. Found: $col_names")
        end
    end

    df = DataFrame(
        x = Float64.(data[:, findfirst(==("x"), col_names)]),
        y = Float64.(data[:, findfirst(==("y"), col_names)]),
        time = Float64.(data[:, findfirst(==("time"), col_names)]),
        missid = Int.(data[:, findfirst(==("missid"), col_names)]),
    )
    return df
end


"""
    bilinear_periodic(field::Matrix{Float64}, x::Float64, y::Float64, nx::Int)

Bilinear interpolation on a doubly-periodic grid.

Arguments:
- field: 2D array of size (nx, nx) — the velocity field at cell centers.
- x, y: position in grid-index units (0-based, fractional).
         E.g., x=0.0 is the center of the first cell, x=255.0 the last.
- nx: grid size (field is nx × nx).

Returns: interpolated value (Float64).
"""
function bilinear_periodic(field::Matrix{Float64}, x::Float64, y::Float64, nx::Int)
    # Wrap to [0, nx) for periodicity
    xw = mod(x, nx)
    yw = mod(y, nx)

    # Integer indices (0-based) of the lower-left cell
    i0 = Int(floor(xw))
    j0 = Int(floor(yw))

    # Fractional offsets within the cell
    fx = xw - i0
    fy = yw - j0

    # Four corner indices (1-based for Julia array access, periodic wrap)
    i1 = mod(i0, nx) + 1
    i2 = mod(i0 + 1, nx) + 1
    j1 = mod(j0, nx) + 1
    j2 = mod(j0 + 1, nx) + 1

    # Bilinear interpolation
    val = (1 - fx) * (1 - fy) * field[i1, j1] +
          fx       * (1 - fy) * field[i2, j1] +
          (1 - fx) * fy       * field[i1, j2] +
          fx       * fy       * field[i2, j2]

    return val
end


"""
    load_uv_physical(nc_path::String, t_idx::Int; lev::Int=1)

Load a single velocity snapshot from the NetCDF file in physical units (m/s).
No C-grid staggering or grid-unit normalization is applied.

Returns: (u::Matrix{Float64}, v::Matrix{Float64})
"""
function load_uv_physical(nc_path::String, t_idx::Int; lev::Int=1)
    ds = NCDataset(nc_path, "r")
    # NCDatasets transposes NetCDF (time, lev, y, x) → Julia (x, y, lev, time)
    # Convert from Union{Missing, Float64} to Float64
    u = Float64.(ds["u"][:, :, lev, t_idx])
    v = Float64.(ds["v"][:, :, lev, t_idx])
    close(ds)
    return u, v
end


"""
    interpolate_velocity(nc_path::String, glider_df::DataFrame, t_start::Int;
                         lev::Int=1, offset_x::Float64=0.0, offset_y::Float64=0.0)

Interpolate QG velocity field (u, v) at each glider position.

For each glider record, maps glider time (seconds) onto the QG daily snapshot
axis, then performs bilinear spatial interpolation and linear temporal
interpolation to obtain the velocity in m/s.

Arguments:
- nc_path: path to QGModelOutput20years.nc
- glider_df: DataFrame with columns x, y, time, missid (x, y in grid units)
- t_start: QG time index at which glider time=0 begins (1-based)
- lev: vertical level (1=upper, 2=lower)
- offset_x, offset_y: translation of glider positions in grid units

Returns: DataFrame with columns x, y, time, missid, x_m, y_m, u_qg, v_qg
"""
function interpolate_velocity(nc_path::String, glider_df::DataFrame, t_start::Int;
                              lev::Int=1, offset_x::Float64=0.0, offset_y::Float64=0.0)
    # Get grid info
    ds = NCDataset(nc_path, "r")
    x_coord = ds["x"][:]
    nx = length(x_coord)
    dx = x_coord[2] - x_coord[1]
    nt_total = size(ds["u"], 4)  # total number of time snapshots
    close(ds)

    n_records = nrow(glider_df)
    println("Interpolating velocity for $n_records glider records...")
    println("  t_start=$t_start, lev=$lev, offset=($offset_x, $offset_y)")

    # Sort by time for efficient snapshot loading
    sorted_idx = sortperm(glider_df.time)

    # Output arrays
    u_qg = Vector{Float64}(undef, n_records)
    v_qg = Vector{Float64}(undef, n_records)

    # Track which snapshot pair is currently loaded
    current_t_lo = -1
    u_lo = Matrix{Float64}(undef, 0, 0)
    v_lo = Matrix{Float64}(undef, 0, 0)
    u_hi = Matrix{Float64}(undef, 0, 0)
    v_hi = Matrix{Float64}(undef, 0, 0)

    for idx in sorted_idx
        # Map glider time (seconds) to QG day index (fractional)
        glider_time_s = glider_df.time[idx]
        t_qg_frac = t_start + glider_time_s / 86400.0  # fractional QG day index (1-based)

        # Bounding integer day indices
        t_lo = Int(floor(t_qg_frac))
        t_hi = t_lo + 1

        # Clamp to valid range
        t_lo = clamp(t_lo, 1, nt_total)
        t_hi = clamp(t_hi, 1, nt_total)

        # Fractional weight for temporal interpolation
        if t_hi > t_lo
            w = (t_qg_frac - t_lo) / (t_hi - t_lo)
        else
            w = 0.0  # at the boundary, just use t_lo
        end

        # Load snapshots if needed (only when the window advances)
        if t_lo != current_t_lo
            if t_lo == current_t_lo + 1 && current_t_lo > 0
                # Shift: the old "hi" becomes the new "lo"
                u_lo = u_hi
                v_lo = v_hi
                u_hi, v_hi = load_uv_physical(nc_path, t_hi; lev=lev)
            else
                # Fresh load of both snapshots
                u_lo, v_lo = load_uv_physical(nc_path, t_lo; lev=lev)
                if t_hi != t_lo
                    u_hi, v_hi = load_uv_physical(nc_path, t_hi; lev=lev)
                else
                    u_hi, v_hi = u_lo, v_lo
                end
            end
            current_t_lo = t_lo
        end

        # Apply spatial offset and periodic wrap
        xg = mod(glider_df.x[idx] + offset_x, nx)
        yg = mod(glider_df.y[idx] + offset_y, nx)

        # Bilinear interpolation at both time steps
        u_at_lo = bilinear_periodic(u_lo, xg, yg, nx)
        v_at_lo = bilinear_periodic(v_lo, xg, yg, nx)
        u_at_hi = bilinear_periodic(u_hi, xg, yg, nx)
        v_at_hi = bilinear_periodic(v_hi, xg, yg, nx)

        # Linear temporal interpolation
        u_qg[idx] = (1.0 - w) * u_at_lo + w * u_at_hi
        v_qg[idx] = (1.0 - w) * v_at_lo + w * v_at_hi
    end

    # Build output DataFrame
    # Apply offset to positions for output
    x_out = mod.(glider_df.x .+ offset_x, nx)
    y_out = mod.(glider_df.y .+ offset_y, nx)

    result = DataFrame(
        x = x_out,
        y = y_out,
        time = glider_df.time,
        missid = glider_df.missid,
        x_m = x_out .* dx,
        y_m = y_out .* dx,
        u_qg = u_qg,
        v_qg = v_qg,
    )

    n_gliders = length(unique(glider_df.missid))
    println("Done. $n_records records for $n_gliders gliders.")

    return result, dx, nx, n_gliders
end
