# Generate a time series plot of the relative vorticity at a specific location of the QG outputs

# Code

Follow these guidelines:

- Use inline comments to explain the code.
- Use the modules in Analysis/py to load any data
- Use matplotlib
- Code in Python
- Place imports at the top of modules
- Reuse any code in the relvor_movie.py module to help you

# Overview

Generate a new module named time_series.py in Figures/py that generates a time_series plot of the relative vorticity at a grid point near the center of the region of the QG outputs.  Use matplotlib.

The time step and duration will be an input and defaults to 1 day and 5 years. 

Have the figure start 10 years before the end of the time series.

You may wish to review the relvor_movie.md file

## Data

The velocities are provided in the file $OS_DATA/QG/QGModelOutput20years.nc.  
The grid spacing is 1000km/256 = 3.9 km.

The relative vorticity is computed as zeta = dv/dx - du/dy.

# Plan

## Module

`papers/Structure/Figures/py/time_series.py`

## Reuse

- Import `relative_vorticity` and the `DX_M` / `DX_KM` constants from
  `relvor_movie.py` so the differencing stencil and grid spacing match.
- Use `qg_utils.load_qg()` from `Analysis/py` for I/O.

## Inputs (function args + CLI)

- `region_size_km` : float (default 100.0) — region used to anchor the
  "near-center" grid point.
- `x0_km`, `y0_km` : float (default 400.0, 400.0) — lower-left corner of
  the region (matches `relvor_movie.py` defaults).
- `dt_days` : int (default 1) — stride between samples in days.
- `duration_days` : int (default 365 * 5 = 1825) — span of the time
  series.
- `start_before_end_days` : int (default 365 * 10 = 3650) — the figure
  starts this many days before the end of the model run. (See
  clarifications: combined with `duration_days=5y` this yields a window
  of `[end-10y, end-5y]`.)
- `lev` : int (default 1) — vertical level (top layer).
- `out_path` : str (default `time_series_relvor.png`).

## Steps

1. **Imports** at top of module: `os`, `argparse`, `sys`, `numpy`,
   `matplotlib.pyplot`, plus `relative_vorticity`, `DX_M` from
   `relvor_movie`, and `qg_utils`.
2. **Load** the QG dataset with `qg_utils.load_qg()`; pick `lev` with
   `qg.sel(lev=lev)`.
3. **Locate grid point**: convert `x_center = x0_km + region_size_km/2`,
   `y_center = y0_km + region_size_km/2` into grid indices using
   `np.argmin(|x_m - x_center_m|)`.
4. **Build time window**: with `n_time = qg.sizes['time']`,
   `t_start = max(0, n_time - start_before_end_days)`,
   `t_end = min(n_time, t_start + duration_days)`,
   `time_idx = np.arange(t_start, t_end, dt_days)`.
5. **Load u, v** for the selected frames over the full level, compute
   ζ via `relative_vorticity(u, v)`, then index at the chosen
   (iy_center, ix_center) → 1-D time series. Loading the full level is
   needed so the periodic centered difference is correct, but only the
   selected frames are pulled from disk.
6. **Time axis** in days = `qg.time.isel(time=time_idx).values / 86400`.
7. **Plot**: `fig, ax = plt.subplots()`, plot ζ(t), label x-axis as
   `time [days]`, y-axis as `ζ [s⁻¹]`, title showing the location in
   km, draw a horizontal `0` reference line, `tight_layout`, save to
   `out_path`.
8. **CLI** with `argparse` mirroring the keyword args, plus a
   `main(flg)`-style harness consistent with how `relvor_movie.py` is
   currently invoked.

## Open clarifications

(See the Clarifications section below.)

# Clarifications

1. Yes I meant from end-10y to end-5y.
2. The center is the closest to 500km, 500km
3. Single grid point is fine, no average.
4. Yes, one PNG figure with dpi=300.  You may review figs_structure.py for my standard approach to coding figures.
5. dt_days = 1 day is correct

# Modifications

## Set 1

1. Indicate time in weeks on the x-axis.
2. Add grid lines

## Set 2

1. Modify the code so that one can plot horizontal velocity (u or v) instead of relative vorticity.

# Prompts

1. Read this file.  Generate a plan for the code and put it in the Plan section above.  Ask me any clarifications you need.
2. Read this file.  I have answered your questions in the Clarifications section above.  Generate the code and put it in the py/time_series.py module.
3. Perform the first set of modifications to the code.
4. Perform the 2nd set of modifications to the code.