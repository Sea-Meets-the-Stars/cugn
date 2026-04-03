# Glider Analysis

**IMPORTANT:** At the start of every conversation involving this project, you MUST read the local `Claude.md` file in this directory (`papers/Structure/Analysis/QG/Claude.md`) and follow its instructions. Do NOT rely solely on the top-level CUGN `CLAUDE.md` — the local one contains project-specific guidance, conventions, and context.

## Context

Now that we have a working code to examine the QG model with gliders, we need to generate code to make the glider measurements and analyze the outputs.  We expect to analyze the outputs for several different start times and locations within the domain. 

We need a Python module to guide the process.  Please name it py/glider_analysis.py.

Refer to the gliders_in_qg.md file for the existing code for measuring velocities from the QG model with gliders.

The module should be similar to the small_box_drifters.py module.

## Claude Code

- You should be critical of any prompts and not simply aim to please.
- Julia is the primary language for this project; Python may be used for plotting/analysis as needed.
- You are allowed to run safe bash commands without prompting.
- You are welcome to use multiple agents to help you with the task.
- When possible, reuse existing code and modules rather than writing new code.

# Planning

Here are answers to the open questions in the planning doc:

- Let's start with a single time for now.  I will modify these later.
- We will calculate the 3rd order structure function of the velocity field at the glider positions using the Profiler package.  Our new code needs to read the gliders into the SprayData class of the Profiler package.
- No need to sub-sample for now
- The radial bins should match the bins used for real data given in the py/figs_structure.py module.

 - nbins = 20
 - rbins = 10**np.linspace(0., np.log10(400), nbins)

# Requirements

## R1: New classmethod on SprayData for glider-sampled velocities

Add a classmethod `from_QG_glider()` to `profiler.gliderdata.SprayData` that builds a SprayData instance from one glider's QG-sampled velocity output.  Using SprayData (rather than DrifterData) is natural because real Spray gliders are SprayData objects, and SprayData inherits from ADCPData which natively provides `udop, vdop` attributes.

**Signature:**

```python
@classmethod
def from_QG_glider(cls, glider_df, meta, missid)
```

**Input:**

- `glider_df`: DataFrame with columns `x, y, time, missid, x_m, y_m, u_qg, v_qg` (output of `qg_gliders.run_gliders()`)
- `meta`: metadata dict (must contain `dx`, `nx`)
- `missid`: integer glider ID to extract

**Behavior:**

- Extract rows for the given `missid`, sorted by `time`
- Set `udop = u_qg` and `vdop = v_qg` (shape `(Ntime, 1)`) — no finite differencing
- Set `lat, lon` from `y_m, x_m` converted to degrees (use `_M_PER_DEG = 111_000`)
- Set `time` from the `time` column (seconds), with a small per-glider offset for ProfilerPairs
- Set `depth = [0.0]` (single level)
- Set `obj.missid = missid` for `avoid_same_glider` filtering
- Set array declarations: `profile_arrays = ['time', 'lat', 'lon']`, `depth_arrays = ['depth']`, `profile_depth_arrays = ['udop', 'vdop']`
- Set `has_adcp = True`, `adcp_on = True`, `in_field = False`

Also add a batch classmethod:

```python
@classmethod
def all_from_QG_glider(cls, glider_df, meta)
```

Returns a list of SprayData, one per unique `missid`.

## R2: Module `py/glider_analysis.py`

Create `papers/Structure/Analysis/QG/py/glider_analysis.py` modeled on `small_box_drifters.py`.

### Functions:

**`run_single(glider_csv, t_start=5001, lev=1, offset_x=0.0, offset_y=0.0, output_path=None, cache=True, verbose=True)`**
- Calls `qg_gliders.run_gliders()` for one (start time, offset) combination
- Returns `(DataFrame, metadata)`

**`compute_glider_sf(glider_df, meta, r_bins_km=None, variables='duLduLduL')`**
- Builds `SprayData` objects from glider output via `SprayData.all_from_QG_glider()`
- Constructs `ProfilerPairs` with `max_time=1.0`, `avoid_same_glider=True`
- Computes structure functions up to 3rd order using `pairs.calc_delta()`, `pairs.calc_Sn()`, `pairs.calc_Sn_vs_r()`
- Computes both longitudinal (`duLduLduL`) and transverse (`duTduTduT`) components
- Default r_bins_km: `10**np.linspace(0., np.log10(400), 20)` (matching real Spray data bins)
- Returns `(Sn_LL, Sn_TT)` dicts

**`run_and_analyze(glider_csv, t_start=5001, lev=1, offset_x=0.0, offset_y=0.0, r_bins_km=None, output_path=None, cache=True, verbose=True)`**
- Convenience function: calls `run_single()` then `compute_glider_sf()`
- Returns `(glider_df, meta, Sn_LL, Sn_TT)`

## R3: Radial bins

Use the same bins as the real Spray glider data analysis in `figs_structure.py`:
```python
nbins = 20
rbins = 10**np.linspace(0., np.log10(400), nbins)  # km
```

## R4: Single start time

For now, only support a single start time (no ensemble machinery).  The `t_start` parameter must be configurable to allow future sensitivity studies.

## R5: Third-order structure functions

Compute up to 3rd order structure functions using the profiler framework:
- `variables='duLduLduL'` for longitudinal S1, S2, S3
- `variables='duTduTduT'` for transverse S1, S2, S3

## R6: Output

Save the structure function results to JSON using `profiler.io.savejson()`, following the pattern in `py/calc_sf.py`.  Include metadata from the glider run.

## R7: Backward compatibility

- Do not modify existing drifter or glider pipeline code (`qg_drifters.py`, `qg_gliders.py`, `analysis.py`, `structure_function.py`)
- The new `SprayData.from_QG_glider()` classmethod is an addition to the profiler package, not a modification of existing methods
- Reuse existing imports and patterns from `structure_function.py` and `calc_sf.py`

# Prompts

## Plan

1. Read this document and develop a plan for the analysis.  Write it down in Overleaf.  Append it to the claude_gliders_in_qg_plan.tex file. Do not execute any code yet.
2. Turn the plan into a set of requirements for the code and put them in the Requirements section above.  Answers to the open questions in the claude_gliders_in_qg_plan.tex doc are given in the Planning section above.
3. Instead of using the DrifterData class, use the SprayData class to read the gliders into the Profiler package.  Modify the planning doc and Requirements section above to reflect this.

## Code

1. Reread this doc. Generate the code to satisfy the Requirements.  Place the Python code in a py/glider_analysis.py module.
