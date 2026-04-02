"""
CLI entry point for sampling QG velocity at glider positions.

Usage:
    julia run_gliders_cli.jl --glider_csv path/to/gliders.csv --t_start 5001 --output /tmp/result.csv

Required arguments:
    --glider_csv    Path to the input glider trajectory CSV file.

Optional arguments:
    --t_start       QG time index at which glider time=0 begins (default: 5001)
    --lev           Vertical level, 1=upper, 2=lower (default: 1)
    --offset_x      Translation in grid units, x-direction (default: 0.0)
    --offset_y      Translation in grid units, y-direction (default: 0.0)
    --output        Output CSV path (default: /tmp/qg_glider_velocities.csv)
"""

# Parse command-line arguments (reuse pattern from run_drifters_cli.jl)
function parse_args(args)
    params = Dict{String,String}()
    i = 1
    while i <= length(args)
        if startswith(args[i], "--")
            key = args[i][3:end]
            if i + 1 <= length(args) && !startswith(args[i+1], "--")
                params[key] = args[i+1]
                i += 2
            else
                params[key] = "true"
                i += 1
            end
        else
            i += 1
        end
    end
    return params
end

params = parse_args(ARGS)

# Required argument
if !haskey(params, "glider_csv")
    error("Missing required argument: --glider_csv <path>")
end
glider_csv = params["glider_csv"]

# Optional arguments with defaults
t_start = parse(Int, get(params, "t_start", "5001"))
lev = parse(Int, get(params, "lev", "1"))
offset_x = parse(Float64, get(params, "offset_x", "0.0"))
offset_y = parse(Float64, get(params, "offset_y", "0.0"))
coords_km = parse(Bool, get(params, "coords_km", "true"))
output_path = get(params, "output", "/tmp/qg_glider_velocities.csv")

# Include the glider modules (qg_grid.jl is a dependency of qg_gliders.jl)
script_dir = @__DIR__
include(joinpath(script_dir, "qg_grid.jl"))
include(joinpath(script_dir, "qg_gliders.jl"))

# Print parameters
println("Parameters:")
println("  glider_csv=$glider_csv")
println("  t_start=$t_start, lev=$lev")
println("  offset=($offset_x, $offset_y), coords_km=$coords_km")
println("  output=$output_path")

# Default NetCDF path
nc_path = joinpath(ENV["OS_DATA"], "QG", "QGModelOutput20years.nc")

# Load glider trajectories
glider_df = load_glider_trajectories(glider_csv)

# Run interpolation
result, dx, nx, n_gliders = interpolate_velocity(
    nc_path, glider_df, t_start;
    lev=lev, offset_x=offset_x, offset_y=offset_y, coords_km=coords_km
)

# Write output CSV (manual to avoid CSV.jl write overhead)
open(output_path, "w") do f
    println(f, "x,y,time,missid,x_m,y_m,u_qg,v_qg")
    for row in eachrow(result)
        # Format missid as integer (join promotes mixed Int/Float vectors to Float)
        println(f, "$(row.x),$(row.y),$(row.time),$(Int(row.missid)),$(row.x_m),$(row.y_m),$(row.u_qg),$(row.v_qg)")
    end
end
println("Output written to $output_path")

# Write metadata JSON sidecar
meta_path = output_path * ".meta.json"
open(meta_path, "w") do f
    write(f, """{"dx": $dx, "nx": $nx, "t_start": $t_start, "lev": $lev, "n_gliders": $n_gliders, "offset_x": $offset_x, "offset_y": $offset_y, "coords_km": $coords_km, "glider_csv": "$glider_csv"}""")
end
println("Metadata written to $meta_path")
