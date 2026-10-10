# Generate a move of the relative vorticity in a region of the QG outputs

# Code

Follow these guidelines:

- Use inline comments to explain the code.
- Use the modules in Analysis/py to load any data
- Use matplotlib
- Code in Python
- Place imports at the top of modules

# Overview

Generate a new module named relvor_movie.py in Figures/py that generates a movie of the relative vorticity in a region of the QG outputs.  Use matplotlib.
The size of the region will be an input and defaults to 100km x 100km.
The time step and duration will be an input and defaults to 30 days and 5 years.

Have the movie start 5 years before the end of the time series.

## Other features

- Label the axes in km
- Include a color bar
- Use a divergent colormap that is appropriate for the relative vorticity, such as 'RdBu_r'

## Data

The velocities are provided in the file $OS_DATA/QG/QGModelOutput20years.nc.  
The grid spacing is 1000km/256 = 3.9 km.

The relative vorticity is computed as zeta = dv/dx - du/dy.
Assume f corresponds to a deformation radius of 15 km/rad^-1.

# Plan

## Module

`papers/Structure/Figures/py/relvor_movie.py`

## Inputs (CLI-style with defaults)

- `region_size_km` : float (default 100.0) — side length of the square region in km
- `x0_km`, `y0_km` : float (default tbd, see clarifications) — lower-left corner of the region in km
- `dt_days` : int (default 30) — interval between movie frames in days
- `duration_days` : int (default 365 * 5 = 1825) — total time spanned by the movie
- `out_path` : str (default `Figures/relvor_movie.mp4`) — output movie path
- `lev` : int (default 0, surface) — vertical level (see clarifications)
- `fps` : int (default 6)

## Steps

1. **Imports** at top of module: `os`, `argparse`, `numpy`, `xarray`, `matplotlib`, `matplotlib.animation`, plus `qg_utils.load_qg` from `Analysis/py`.
2. **Load data** with `qg_utils.load_qg()` (uses `$OS_DATA/QG/QGModelOutput20years.nc`).
3. **Select time window**: pick frames in the last `duration_days` of the time series, sub-sampled at `dt_days` intervals (last frame is the end of the run).
4. **Select region**: convert `x0_km`, `y0_km`, `region_size_km` to grid indices using `dx = 1000.0 / 256` km and slice `u`, `v` at the requested `lev`.
5. **Compute relative vorticity** ζ for each selected frame using centered differences on `u`, `v`:
   - `ζ = ∂v/∂x − ∂u/∂y`, with periodic boundaries (the QG model is doubly periodic) so edges of the region are differentiated using neighboring cells from the full domain (i.e., compute ζ on the full level then slice the region).
   - Add the constant Coriolis term `f` if the user wants q (the prompt says `q = f + del2(u,v)`); see clarifications.
   - `f` from deformation radius Ld = 15 km, with QG nondimensional N·H = 1 → `f = 1/Ld` in rad/km, i.e. `1/15` rad/km. (This is actually a clarification item.)
6. **Set color limits** symmetrically from a robust percentile (e.g. ±99th percentile of |ζ| over all selected frames) so the color scale is consistent across frames.
7. **Build the movie** with `matplotlib.animation.FuncAnimation`:
   - axes labeled `x [km]`, `y [km]` (region-local coordinates starting at `x0_km`, `y0_km`)
   - `pcolormesh`/`imshow` with `cmap='RdBu_r'` and the symmetric vmin/vmax
   - colorbar labeled with vorticity units
   - frame title shows current model day
8. **Write** the movie to `out_path` (mp4 via ffmpeg writer; fall back to gif if ffmpeg is missing).
9. **CLI entry point** `if __name__ == '__main__':` parses defaults and calls a top-level `make_movie(...)` function.

## Clarifications

1. The relative vorticity should be computed as zeta = dv/dx - du/dy. I have fixed the doc above.
2. Take f from the model attributes.
3. Have the default grid placement be at x0=400km, y0=400km.
4. Use vertical level 1 (top layer).
5. Yes, .mp4 is fine
6. Yes, every 30th frame is fine for dt=30 days.  I may choose to use a finer time step


# Prompts

1. Read this file.  Generate a plan for the code and put it in the Plan section above.  Ask me any clarifications you need.
2. Read this file.  I have answered your questions in the Clarifications section above.  Generate the code and put it in the py/relvor_movie.py module.