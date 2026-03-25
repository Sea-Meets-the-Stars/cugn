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

# Include the drifter modules
script_dir = @__DIR__
include(joinpath(script_dir, "qg_grid.jl"))
include(joinpath(script_dir, "qg_drifters.jl"))

# Run the simulation
println("Parameters:")
println("  t_start=$t_start, n_days=$n_days, n_per_side=$n_per_side, lev=$lev")
println("  record_interval=$record_interval, output=$output_path")

results = run_drifters(;
    t_start=t_start,
    n_days=n_days,
    n_per_side=n_per_side,
    lev=lev,
    record_interval=record_interval
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
    write(f, """{"dx": $(results.dx), "nx": $(results.nx), "n_drifters": $(results.n_drifters), "t_start": $t_start, "n_days": $n_days, "n_per_side": $n_per_side, "lev": $lev}""")
end
println("Metadata written to $meta_path")
