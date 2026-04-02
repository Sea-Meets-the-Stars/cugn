# Gliders in QG

**IMPORTANT:** At the start of every conversation involving this project, you MUST read the local `Claude.md` file in this directory (`papers/Structure/Analysis/QG/Claude.md`) and follow its instructions. Do NOT rely solely on the top-level CUGN `CLAUDE.md` — the local one contains project-specific guidance, conventions, and context.

## Context

We wish to analyze the outputs of a quasigeostrophic model in a region of the ocean where gliders have been deployed.  We will use the Python interface to the Julia code to run the model.  The primary goals of the code will be to:

- Allow the user to specify the region of the ocean to analyze.
- Allow the user to specify the days to analyze.
- Interpolate the velocity field to the glider positions.
- Record the velocity field at the glider positions.
- Write to a machine-readable format.

It will be helpful for you to review the codes we generated for Drifters in QG.  These are in the `papers/Structure/Analysis/QG/jl` directory.

You should also review the prompt files in this directory (drift_in_qg.md and small_box_drifters.md) to understand the code and the analysis.

## Organization

### Code

- Place analysis Julia scripts in `papers/Structure/Analysis/QG/jl/`.
- Place any Python helper scripts in `papers/Structure/Analysis/QG/py/`.
- Place figure-generation scripts in `papers/Structure/Figures/py/` (or `jl/` for Julia).
- Jupyter/Pluto notebooks for exploratory analysis go in `papers/Structure/Analysis/QG/`.
- Reuse existing code when possible
- Include inline comments in the code to explain what is happening.


### Data

- QG model output: `$OS_DATA/QG/QGModelOutput20years.nc`
- The glider trajectories are provded in the data/100km100day10gliders3h.csv file.

## Claude Code

- You should be critical of any prompts and not simply aim to please.
- Julia is the primary language for this project; Python may be used for plotting/analysis as needed.
- You are allowed to run safe bash commands without prompting.
- You are welcome to use multiple agents to help you with the task.
- When possible, reuse existing code and modules rather than writing new code.

### LaTeX / Overleaf

- The paper LaTeX source is in this Overleaf-synced directory: /home/xavier/Projects/overleaf/Structure_function
- When adding new files, update `main.tex` to include them.
- You may push to git as you work. The access token is in the user's `.bashrc` profile with the name OVERLEAF.

# Requirements

## Answers to questions in the planning doc

- We will not record vorticity or streamfunction
- We will test sensitivity to the start time, so that needs to be a free parameter
- The spatial offset feature does not need to allow for rotation

## R1: Create Julia interpolation script `jl/qg_gliders.jl`

Create a new Julia module that samples the QG velocity field at prescribed glider positions. Reuse `qg_grid.jl` for NetCDF loading.

**Functions:**

- `load_glider_trajectories(csv_path)` — Read the glider CSV into a DataFrame with columns `x, y, time, missid`.

- `bilinear_periodic(field, x, y, nx)` — Bilinear interpolation on the periodic grid. Wrap indices with `mod` for periodicity. `field` is a 2D array (nx × nx), `x` and `y` are in grid-index units (0-based, fractional).

- `interpolate_velocity(nc_path, glider_df, t_start; lev=1, offset_x=0.0, offset_y=0.0)` — Main function. For each glider record:
  1. Compute the QG day index: `t_qg = t_start + glider_time / 86400.0`
  2. Identify bounding daily snapshots `t_n` and `t_n+1`.
  3. Load u, v at both snapshots (physical units, m/s — do NOT normalize to grid units since we are not advecting).
  4. Apply `offset_x`, `offset_y` (in grid units) to the glider positions.
  5. Bilinearly interpolate in space at the (offset) glider position for both snapshots.
  6. Linearly interpolate in time.
  7. Return the glider DataFrame augmented with `x_m, y_m, u_qg, v_qg` columns.

**Memory management:** Only keep two velocity snapshots in memory at a time. Process records in chronological order, advancing the snapshot window as needed.

**Key difference from drifter code:** Velocities are returned in physical units (m/s), not grid-units/s. No C-grid staggering correction is needed since we are sampling, not advecting.

## R2: Create Julia CLI entry point `jl/run_gliders_cli.jl`

Follow the pattern of `run_drifters_cli.jl`.

**Arguments:**
- `--glider_csv` (required): path to the input glider trajectory CSV.
- `--t_start` (default 5001): QG time index at which glider time=0 begins.
- `--lev` (default 1): QG vertical level.
- `--offset_x` (default 0.0): translation of glider positions in grid units (x-direction).
- `--offset_y` (default 0.0): translation of glider positions in grid units (y-direction).
- `--output` (default `/tmp/qg_glider_velocities.csv`): output CSV path.

**Output:**
- CSV with columns: `x, y, time, missid, x_m, y_m, u_qg, v_qg`
- Sidecar JSON with metadata: `dx, nx, t_start, lev, n_gliders, offset_x, offset_y, glider_csv`

## R3: Create Python wrapper `py/qg_gliders.py`

Follow the pattern of `py/qg_drifters.py`.

**Functions:**

- `run_gliders(glider_csv, t_start=5001, lev=1, offset_x=0.0, offset_y=0.0, output_path=None, cache=True, verbose=True)` — Call `run_gliders_cli.jl` via subprocess. Auto-generate output filename from parameters if `output_path` is None. Skip re-run if output CSV and JSON exist and `cache=True`. Return `(DataFrame, metadata_dict)`.

- `load_glider_velocities(csv_path)` — Load a previously saved output CSV + JSON sidecar. Return `(DataFrame, metadata_dict)`.

## R4: Backward compatibility

- The existing drifter pipeline (`qg_drifters.jl`, `run_drifters_cli.jl`, `qg_drifters.py`) must not be modified.
- The glider pipeline is entirely new code in separate files.
- Shared code should be imported from `qg_grid.jl` (Julia) or `qg_io.py` (Python), not duplicated.

## R5: Output specification

The output CSV must contain exactly these columns:

| Column | Units | Description |
|--------|-------|-------------|
| `x` | grid units | Glider x-position (from input, with offset applied) |
| `y` | grid units | Glider y-position (from input, with offset applied) |
| `time` | seconds | Time along glider trajectory (from input) |
| `missid` | — | Glider/mission identifier (from input) |
| `x_m` | meters | x-position in physical units (`x * dx`) |
| `y_m` | meters | y-position in physical units (`y * dx`) |
| `u_qg` | m/s | QG zonal velocity interpolated to glider position |
| `v_qg` | m/s | QG meridional velocity interpolated to glider position |

The sidecar JSON must include: `dx`, `nx`, `t_start`, `lev`, `n_gliders`, `offset_x`, `offset_y`, `glider_csv`.

## R6: Start time as a free parameter

`t_start` must be a user-configurable parameter at all levels (Julia function, CLI, Python wrapper). This allows testing sensitivity of structure functions to the QG flow realization by running the same glider trajectories at different QG start times.

# Testing

We will need several tests to ensure the code is working correctly.
These are the following tests to generate:

- Test that the trajectory outputs are correct. Generate a figure.
- Test that the velocity field is interpolated correctly. Generate a figure.
- Test the output file is written correctly.

Place the testing code in a py/test_gliders.py module.

# Analysis

# Prompts

## Plan

1. Read this document and develop a plan for the analysis.  Write it down in Overleaf.  Do not execute any code yet.
2. Turn the plan into a set of requirements for the code and put them in the Requirements section above.  Answers to the open questions in the planning doc are given in the Requirements section above.

## Code

1. Reread this doc. Generate the code to satisfy the Requirements.  Place the Python code in a py/qg_gliders.py module. Place the Julia code in a jl/qg_gliders.jl module.

## Tests

1. Reread this doc. Create the tests described in the Testing section above.
