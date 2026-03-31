# Structure function analysis

**IMPORTANT:** At the start of every conversation involving this project, you MUST read the local `Claude.md` file in this directory (`papers/Structure/Analysis/QG/Claude.md`) and follow its instructions. Do NOT rely solely on the top-level CUGN `CLAUDE.md` — the local one contains project-specific guidance, conventions, and context.

## Context

We wish to calculate the structure function of the velocity field in the QG model from the drifter trajectories.  The drifter trajectories are saved in a CSV file.  We will use the profiler package to load the drifter trajectories and calculate the structure function.  This will require new methods in the profiler package. If you haven't already, inspect that package and understand its structure and methods.

## Code

- Use code in the analysis.py module to calculate the structure function of the velocity field for a drifter as one approach.
- Modify the profiler package to add a method to calculate the structure function of the velocity field for a drifter as the second approach.
- Use Python exclusively
- Include inline comments in the code to explain what is happening.

## Claude Code

- You should be critical of any prompts and not simply aim to please.
- Julia is the primary language for this project; Python may be used for plotting/analysis as needed.
- You are allowed to run safe bash commands without prompting.
- You are welcome to use multiple agents to help you with the task.
- When possible, reuse existing code and modules rather than writing new code.

### LaTeX / Overleaf

- The paper LaTeX source is in this Overleaf-synced directory: /home/xavier/Projects/overleaf/Structure_function
- You may push to git as you work. The access token is in the user's `.bashrc` profile with the name OVERLEAF.
- For this analysis, document your work in the Overleaf project in a new file named structure_function.tex.  Make it a standalone LateX file.

# Requirements

## R1: DrifterData class in the profiler package

Create a `DrifterData` class in a new module `profiler/drifterdata.py` that subclasses `ProfilerData` and adapts QG drifter trajectory data to the profiler interface.

### R1.1: Data mapping

Each drifter becomes a separate `DrifterData` instance (one per drifter ID), with:
- `time`: 1D array of timestamps (seconds) from the trajectory CSV — one entry per recorded time step (i.e., each time step is treated as a "profile").
- `lat`, `lon`: Synthetic lat/lon derived from the QG model's `x_m`, `y_m` positions. Since the QG domain has no real geography, use a simple meter-to-degree conversion centered at (0, 0): `lat = y_m / 111_000`, `lon = x_m / (111_000 * cos(0)) = x_m / 111_000`.
- `udop`, `vdop`: 2D arrays of shape `(Ntime, 1)` holding the u, v Lagrangian velocities (m/s) derived from centered finite differences of the trajectory positions (reuse logic from `analysis.py::_compute_drifter_velocities`). The single "depth" dimension represents the surface layer.
- `depth`: 1D array `[0.]` (single depth level).
- `missid`: Set to the integer drifter ID, so `ProfilerPairs` can use `avoid_same_glider=True` to prevent self-pairing.
- `dataset`: A descriptive string, e.g., `'QG_drifter_<ID>'`.
- `distE`, `distN`: Will be set by `ProfilerPairs.calc_dist()` during initialization. The synthetic lat/lon must be compatible with the `offsets.calc_dist_offset()` function.

### R1.2: Constructor

Provide a class method `DrifterData.from_trajectory(traj_df, meta, drifter_id)` that:
1. Extracts the subset of `traj_df` for the given `drifter_id`.
2. Computes velocities via centered finite differences (no periodic wrapping — per decision above).
3. Populates all required ProfilerData attributes.
4. Sets `profile_arrays = ['lat', 'lon', 'time']`, `depth_arrays = ['depth']`, `profile_depth_arrays = ['udop', 'vdop']`, `scalar_keys = []`.

### R1.3: Batch constructor

Provide a class method `DrifterData.all_from_trajectory(traj_df, meta)` that returns a list of `DrifterData` objects, one per drifter ID.

## R2: Time-offset handling for ProfilerPairs compatibility

`ProfilerPairs.generate_pairs()` uses a strict `dt > 0` filter, which excludes pairs at the exact same time. Since all drifters share identical time arrays, cross-drifter pairs at the same time step would have `dt = 0` and be discarded.

**Solution**: In `DrifterData.from_trajectory()`, add a tiny per-drifter time offset: `time += drifter_id * 1e-3` (1 ms). This is negligible relative to the 86400 s time step but ensures all cross-drifter pairs have `dt != 0`. Set `max_time` in `ProfilerPairs` to a small value (e.g., 1 hour) so only "same time step" pairs are included.

## R3: structure_function.py analysis module

Create `papers/Structure/Analysis/QG/py/structure_function.py` that:

### R3.1: Approach 1 — analysis.py (existing)

Provide a function `run_analysis_approach(traj, meta)` that:
1. Calls `analysis.compute_structure_functions(traj, meta)`.
2. Returns the resulting DataFrame with `D_LL`, `D_TT`, `D_2` vs `r_mid`.

### R3.2: Approach 2 — profiler/ProfilerPairs

Provide a function `run_profiler_approach(traj, meta)` that:
1. Creates a list of `DrifterData` objects via `DrifterData.all_from_trajectory(traj, meta)`.
2. Instantiates `ProfilerPairs(drifter_list, max_time=1.0)` with `avoid_same_glider=True`.
3. Calls `calc_delta(iz=0, variables='duLduLduL')` to compute velocity differences at the single depth level.
4. Calls `calc_Sn(variables='duLduLduL')` and then `calc_Sn_vs_r(rbins)` with the same log-spaced bins as Approach 1.
5. Repeats for `'duTduTduT'`.
6. Returns the Sn_dict results.

### R3.3: Comparison

Provide a function `compare_approaches(traj, meta)` that runs both approaches and returns the results for plotting/comparison.

## R4: Constraints

- Do **not** modify `analysis.py` — use it as-is for Approach 1.
- Ignore periodic boundary conditions in the profiler approach (per decision).
- Use daily-snapshot velocities as-is (no sub-daily interpolation).
- Include inline comments explaining what is happening.
- Put import statements at the top of the file.

# Testing

Tests should be in `profiler/profiler/tests/test_drifter.py` and use pytest. Use the test dataset at `papers/Structure/Analysis/QG/data/test_small_box_drifters.csv` (121 drifters, 10 days, 11 time steps).

## T1: DrifterData construction

### T1.1: `test_single_drifter`
Load the test CSV via `qg_io.load_trajectories()`. Construct a single `DrifterData` instance via `DrifterData.from_trajectory(traj, meta, drifter_id=1.0)`. Assert:
- `time` is a 1D numpy array with 11 elements (one per time step).
- `lat` and `lon` are 1D arrays with 11 elements and contain finite values.
- `udop` and `vdop` are 2D arrays of shape `(11, 1)` with finite values.
- `depth` is `[0.]`.
- `missid` equals `1` (integer drifter ID).
- `dataset` is a non-empty string.

### T1.2: `test_all_drifters`
Call `DrifterData.all_from_trajectory(traj, meta)`. Assert:
- Returns a list of length 121.
- Each element is a `DrifterData` instance.
- All `missid` values are unique.

## T2: Time-offset handling

### T2.1: `test_time_offsets_unique`
Construct all 121 `DrifterData` objects. Assert that no two drifters share the exact same time value at any nominal time step (i.e., the per-drifter offsets produce distinct timestamps).

### T2.2: `test_time_offsets_small`
Assert that the maximum time offset across all drifters is less than 1 second (negligible compared to the 86400 s recording interval).

## T3: ProfilerPairs compatibility

### T3.1: `test_profilepairs_construction`
Construct `DrifterData` objects for the first 5 drifters only (to keep runtime manageable). Instantiate `ProfilerPairs(drifter_list, max_time=1.0, avoid_same_glider=True)`. Assert:
- `ProfilerPairs` object is created without error.
- `npairs > 0` — pairs were successfully generated.
- No self-pairs: for every pair index, the two profiles come from different drifters.

### T3.2: `test_calc_delta_and_Sn`
Using the 5-drifter `ProfilerPairs` from T3.1:
1. Call `calc_delta(iz=0, variables='duLduLduL')`.
2. Assert `duL` and `duT` arrays exist and have length `npairs`.
3. Call `calc_Sn(variables='duLduLduL')`.
4. Assert `S1`, `S2`, `S3` arrays exist and have length `npairs`.
5. Call `calc_Sn_vs_r(rbins)` with 10 log-spaced bins from 5 km to 200 km.
6. Assert the returned dict contains keys `'r'`, `'S2_duL**2'` and that `'r'` has 10 elements.

## T4: Velocity sanity check

### T4.1: `test_velocity_magnitude`
For any single `DrifterData` object, assert that all velocity values (`udop`, `vdop`) have magnitude less than 1 m/s. The QG model has typical velocities of O(0.1) m/s; values exceeding 1 m/s would indicate a finite-differencing or unit error.

## T5: Structure function comparison (approach 1 vs approach 2)

### T5.1: `test_approaches_qualitative_agreement`
Run both approaches on the test dataset (or a small subset). Assert that D_LL from Approach 2 (`S2_duL**2`) and D_LL from Approach 1 are within a factor of 3 at overlapping separation bins. Exact agreement is not expected because Approach 1 uses periodic BCs and Approach 2 does not, but they should be in the same ballpark.

# Prompts

## Plan

1. Read this document and develop a plan for the analysis.  Write it down in Overleaf.  Do not execute any code yet.
2. Re-read this doc. Turn the plan you created into a set of requirements for the code and put them in the Requirements section above.  Below are answers to the Key Decisions and other questions in the Planning document.

- The code for the first approach should be in the analysis.py module.  The code for the second approach should be in the profiler package.
- Keep analysis.py as is.  Only use it for comparisons and testing.
- Ignore the periodic boundary conditions 
- We are stuck with daily snapshots 
- Add drifter support for the profiler package as a DrifterData object (Option A).  We will then use the ProfilerPairs class to calculate the structure function of the velocity field for a drifter.

3. Add to the Testing section a set of tests to verify that the code satisfies the Requirements.

## Code

1. Generate the code to satisfy the Requirements and perform the Tests.


## Tests

1. Reread this doc. Perform the first step under Testing above.

## Analysis

1. Reread this doc. Perform the first step under Analysis above.