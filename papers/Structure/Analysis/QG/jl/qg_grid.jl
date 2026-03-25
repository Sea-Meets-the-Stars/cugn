"""
    qg_grid.jl

Load QG model output into MeshArrays.jl grid structures.

Usage:
    include("qg_grid.jl")
    γ, Γ, u_fields, v_fields = load_qg_mesharrays(; lev=1, trange=1:100)
"""

using MeshArrays
using NCDatasets

"""
    qg_gcmgrid(nc_path::String)

Create a MeshArrays gcmgrid for the pyqg doubly-periodic domain
by reading grid metadata from the NetCDF file.

Returns: (γ::gcmgrid, dx::Float64, dy::Float64, x::Vector, y::Vector)
"""
function qg_gcmgrid(nc_path::String)
    ds = NCDataset(nc_path, "r")
    x = ds["x"][:]
    y = ds["y"][:]
    close(ds)

    nx = length(x)
    ny = length(y)
    dx = x[2] - x[1]
    dy = y[2] - y[1]

    # Single-face periodic domain
    nFaces = 1
    ioSize = [nx ny]
    facesSize = [(nx, ny)]
    ioPrec = Float64

    γ = gcmgrid("", "PeriodicDomain", nFaces, facesSize, ioSize, ioPrec, read, write)

    return γ, dx, dy, x, y
end

"""
    qg_grid_variables(γ::gcmgrid, x::Vector, y::Vector, dx::Float64, dy::Float64)

Build grid variable NamedTuple (XC, YC, DXC, DYC, RAC) for the QG domain.
Coordinates are in meters.
"""
function qg_grid_variables(γ::gcmgrid, x::Vector, y::Vector, dx::Float64, dy::Float64)
    nx = length(x)
    ny = length(y)

    # Cell center coordinates
    XC_arr = [x[i] for i in 1:nx, j in 1:ny]
    YC_arr = [y[j] for i in 1:nx, j in 1:ny]

    XC = MeshArray(γ, Float64)
    XC[1] = XC_arr
    YC = MeshArray(γ, Float64)
    YC[1] = YC_arr

    # Grid spacings (uniform)
    DXC = MeshArray(γ, Float64)
    DXC[1] = fill(dx, nx, ny)
    DYC = MeshArray(γ, Float64)
    DYC[1] = fill(dy, nx, ny)

    # Cell area
    RAC = MeshArray(γ, Float64)
    RAC[1] = fill(dx * dy, nx, ny)

    return (XC=XC, YC=YC, DXC=DXC, DYC=DYC, RAC=RAC)
end

"""
    load_velocity_fields(nc_path::String, γ::gcmgrid; lev::Int=1, trange=1:100)

Load u and v velocity snapshots from the QG NetCDF file into vectors of MeshArrays.

Arguments:
- nc_path: path to QGModelOutput20years.nc
- γ: gcmgrid from qg_gcmgrid()
- lev: vertical level (1=upper, 2=lower)
- trange: range of time indices to load

Returns: (u_fields::Vector{MeshArray}, v_fields::Vector{MeshArray}, times::Vector{Float64})
"""
function load_velocity_fields(nc_path::String, γ::gcmgrid; lev::Int=1, trange=1:100)
    ds = NCDataset(nc_path, "r")
    times = ds["time"][trange]

    nt = length(trange)
    u_fields = Vector{typeof(MeshArray(γ, Float64))}(undef, nt)
    v_fields = Vector{typeof(MeshArray(γ, Float64))}(undef, nt)

    for (k, t) in enumerate(trange)
        # NCDatasets uses 1-based indexing; dimensions are (x, y, lev, time) in Julia
        # NetCDF variable order is (time, lev, y, x) but NCDatasets transposes to (x, y, lev, time)
        u_snap = ds["u"][:, :, lev, t]
        v_snap = ds["v"][:, :, lev, t]

        u_ma = MeshArray(γ, Float64)
        u_ma[1] = u_snap
        u_fields[k] = u_ma

        v_ma = MeshArray(γ, Float64)
        v_ma[1] = v_snap
        v_fields[k] = v_ma
    end

    close(ds)
    return u_fields, v_fields, times
end

"""
    load_qg_mesharrays(; nc_path=nothing, lev=1, trange=1:100)

Convenience function: build grid + load velocity fields in one call.

Returns: (γ, Γ, u_fields, v_fields, times)
"""
function load_qg_mesharrays(; nc_path=nothing, lev::Int=1, trange=1:100)
    if isnothing(nc_path)
        nc_path = joinpath(ENV["OS_DATA"], "QG", "QGModelOutput20years.nc")
    end

    γ, dx, dy, x, y = qg_gcmgrid(nc_path)
    Γ = qg_grid_variables(γ, x, y, dx, dy)
    u_fields, v_fields, times = load_velocity_fields(nc_path, γ; lev=lev, trange=trange)

    return γ, Γ, u_fields, v_fields, times
end
