"""
CLI entry point for running QG drifter simulations.

Usage:
    julia run_drifters_cli.jl --t_start 5001 --n_days 30 --n_per_side 16 --output /tmp/traj.csv

All arguments are optional with sensible defaults.
"""

# Parse command-line arguments
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

t_start = parse(Int, get(params, "t_start", "5001"))
n_days = parse(Int, get(params, "n_days", "30"))
n_per_side = parse(Int, get(params, "n_per_side", "16"))
lev = parse(Int, get(params, "lev", "1"))
record_interval = parse(Int, get(params, "record_interval", "1"))
output_path = get(params, "output", "/tmp/qg_drifter_traj.csv")
box_center_x = parse(Float64, get(params, "box_center_x", "NaN"))
box_center_y = parse(Float64, get(params, "box_center_y", "NaN"))
box_size_km = parse(Float64, get(params, "box_size_km", "NaN"))
drifter_spacing_km = parse(Float64, get(params, "drifter_spacing_km", "NaN"))

# Include the drifter modules
script_dir = @__DIR__
include(joinpath(script_dir, "qg_grid.jl"))
include(joinpath(script_dir, "qg_drifters.jl"))

# Run the simulation
println("Parameters:")
println("  t_start=$t_start, n_days=$n_days, n_per_side=$n_per_side, lev=$lev")
println("  record_interval=$record_interval, output=$output_path")
if !isnan(box_size_km)
    println("  box: $(box_size_km)km at center ($box_center_x, $box_center_y), spacing=$(drifter_spacing_km)km")
end

results = run_drifters(;
    t_start=t_start,
    n_days=n_days,
    n_per_side=n_per_side,
    lev=lev,
    record_interval=record_interval,
    box_center_x=box_center_x,
    box_center_y=box_center_y,
    box_size_km=box_size_km,
    drifter_spacing_km=drifter_spacing_km
)

# Write trajectories to CSV (manual to avoid CSV.jl dependency)
traj = results.trajectories
open(output_path, "w") do f
    println(f, join(names(traj), ","))
    for row in eachrow(traj)
        println(f, join([row.ID, row.x, row.y, row.t, row.x_m, row.y_m], ","))
    end
end
println("Trajectories written to $output_path")

# Write metadata as JSON sidecar
meta_path = output_path * ".meta.json"
open(meta_path, "w") do f
    meta_str = """{"dx": $(results.dx), "nx": $(results.nx), "n_drifters": $(results.n_drifters), "t_start": $t_start, "n_days": $n_days, "n_per_side": $n_per_side, "lev": $lev"""
    if !isnan(box_size_km)
        meta_str *= """, "box_center_x": $box_center_x, "box_center_y": $box_center_y, "box_size_km": $box_size_km, "drifter_spacing_km": $drifter_spacing_km"""
    end
    meta_str *= "}"
    write(f, meta_str)
end
println("Metadata written to $meta_path")
