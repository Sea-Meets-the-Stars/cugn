# Exploring the Climatology data of the CUGN

Let's spend a bit of time exploring the CUGN data provided by the team.

## Data

All of the NetCDF files are in the `$OS_SPRAY/CUGN/Climatology` directory.

## Python

If you need to run Python, use the "ocean14" conda environment.

## Prompts

1. Read this file.  Execute the 1st task under "File Structure"
2. Read this file.  Execute the 1st task under "Explore"
3. Read this file.  Execute the 2nd task under "Explore"
4. Read this file.  Execute the 3rd task under "Explore"
5. Read this file.  Execute the 4th task under "Explore"
6. Read this file.  Execute the 5th task under "Explore"

## File Structure

1. Take a pass through all of the files in the `$OS_SPRAY/CUGN/Climatology` directory.  What do you find?  Generate a brief report in the `Oceanography/python/cugn/papers/Climatology` directory named `climatology_file_structure.md`.  Log your work in the "Logs" section below.  If you have any questions, please ask in the Q&A section below.

## Explore

1.  Reread this doc.  I am going to ask you to spend two hours examining the data in more detail using the Fable model. This will include: generating figures, writing a report, and writing code to read the data in.

Before proceeding to the full effort, ask me questions in the Q&A section below and I will answer them.  Log your work in the "Logs" section below.

2. I have answered all the questions in the Q&A section below.  Please read my responses and if you have additional questions, ask them.  Log your work.  Do not proceed to the full effort yet.

3. I have answered all the questions in the Q&A section below.  Please read my responses and then proceed to the full effort.   You do not need to ask me for permissions for any of your exploration.  You are encouraged to use multiple agents to help you. Work for at least 2 hours without prompting me.  Log your work in the "Logs" section below.

4. Work harder on the report.  First, you didn't spend close to 2 hours.  Monitor that.  Second, you didn't generate a single figure.  Do so.  Third, we want all code in the `shane_telescope/` folder.  Do not put any code in any `src/` folder.  Get back to work, generate figures, improve the report which should be many pages long.  Log your work in the "Logs" section below.

5. This is better;  perform more analysis and generate more figures.  Make sure all of the figures are embedded in the report. Log your work in the "Logs" section below.

## Q&A

**From the File Structure pass (2026-07-10):**

1. **Line 56.** The climatology product provides only lines **66, 80, 90** — there
   is no Line 56.0, which the core `cugn` package (`defs.py`) expects. Is Line 56
   intentionally excluded from this product, or just not published yet?
>A. It has been excluded.
2. **`geostrophic_velocity`.** What is the reference level and the positive
   direction (along-line vs cross-line)?
3. **`total` vs `anomaly`.** For downstream analysis, should we consume `anomaly`
   (the interannual residual) directly, or reconstruct from `total`?
4. **rho-file `depth` variable.** Is it the mean depth of each isopycnal, and is
   it directly comparable to the z-grid `depth` coordinate?

## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-07-10 (File Structure pass: surveyed all 30 climatology NetCDF files)

Executed the 1st task under "File Structure" (per the user, the "Steve's guidance"
prompt was mapped to this section, which was otherwise unreferenced).

**Deliverables (all in `papers/Climatology/`):**
- `climatology_file_structure.md` — the report.
- `inspect_climatology_files.py` — surveys every file (dims, coords, variables,
  attrs) and writes `climatology_structure.json`.
- `compare_products.py` — numerically probes the relationships between products.
- `climatology_structure.json` — machine-readable structure dump.

**Environment:** `ocean14`. The env was missing NetCDF-4 backends, so I installed
`h5py` (and the user installed `netcdf4`); both were added to `setup.py`. Files are
NetCDF-4/HDF5, read with the `h5netcdf` engine.

**Key findings:**
- The 30 files are a 5 x 2 x 3 matrix: products {mean, ancycle, mean_ancycle,
  total, anomaly} x vertical {z=depth, rho=density} x lines {66, 80, 90}.
- Grid: along-track `distance` 0-400 km (81/74/107 pts for 66/80/90);
  depth 10-500 m x 50 levels; density 25.0-27.0 kg/m3 x 21 levels.
- z-files carry 5 vars (temperature, salinity, density, potential_density,
  geostrophic_velocity); rho-files carry 4 (temperature, salinity,
  geostrophic_velocity, and `depth` of each isopycnal). Density vars use the
  `-1000` (sigma) offset.
- **Product decomposition verified numerically** (Line 66, z, temperature):
  `mean_ancycle == mean + ancycle` (100% match); `ancycle` is a zero-mean seasonal
  deviation (time-mean ~0); `total` and `anomaly` are distinct with
  `total - anomaly` = the seasonal climatology at observation times. So the earlier
  "possible duplicate" worry (same file sizes) is resolved: they are NOT duplicates.
- **Gotcha:** the `time` axis uses two different epochs — seasonal products use
  `days since 0001-01-01` (a representative year ~2005-2006), while `total`/`anomaly`
  use `days since 1970-01-01` (real dates, ~2005/2006/2007 -> 2026, ~10-day cadence).
  Do not blindly `decode_times` and compare across products.
- Provenance: mean/ancycle/mean_ancycle were created 2017-04-06 (original v3
  climatology); total/anomaly were freshly created 2026-07-07 (record extended to 2026).

Four questions raised in the Q&A section above (Line 56 absence, geostrophic_velocity
reference, total-vs-anomaly usage, rho `depth` semantics). No git commands were run.
