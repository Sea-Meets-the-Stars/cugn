# Prep for Science

This prompt doc will get us ready to set Claude Science on its way.

## Process

If you need to run Python, use the "ocean14" conda environment.

## Prompts

1, Execute the first task under Grab the data.

## Grab the data

1. This webpage `https://spraydata.ucsd.edu/products/cugn-climatology/` under Data Access has a list of NetCDF files.  Please download the files to the `$OS_SPRAY/CUGN/Climatology` directory.  If you have any questions, please ask in the Q&A section below.  Log your work.

## Logging

Append an entry to the **Logs** section of this file using the format:

```
### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>
```

## Q&A

- **Lines covered.** The climatology set covers only lines **66, 80, 90** — there
  is **no line 56**, and the files are labeled `66/80/90` rather than the
  `56.0/66.7/80.0/90.0` strings the core package (`defs.py`) uses. Is a line-label
  mapping expected when we wire these in, and is line 56 simply not part of the
  climatology product?
- **Possible duplicate products.** The `mean_ancycle_*` files are byte-for-byte the
  same size as the corresponding `ancycle_*` files, and `total_*` matches
  `anomaly_*` in size. Are these genuinely distinct products, or overlapping?

## Logs

### 2026-07-09 (Downloaded the 30 CUGN climatology NetCDF files)

Downloaded all NetCDF files listed under "Data Access" on
`https://spraydata.ucsd.edu/products/cugn-climatology/` into
`$OS_SPRAY/CUGN/Climatology` (= `/Users/xavier/Projects/Oceanography/data/Spray/CUGN/Climatology`).

**Result:** 30 files, ~1.6 GB total, all verified as valid HDF5/NetCDF-4
(magic bytes `\x89HDF`); no HTML/error pages among them.

**What's in the set.** The files form a 5 x 2 x 3 grid:
- 5 products: `mean`, `ancycle` (annual cycle), `mean_ancycle`, `total`, `anomaly`
- 2 vertical coordinates: `z` (depth) and `rho` (density)
- 3 lines: `66`, `80`, `90`

Filename pattern: `<product>_<zORrho>_<line>.nc` (e.g. `anomaly_z_90.nc`).
Base download URL: `https://spraydata.ucsd.edu/data/cugn/clim/netcdf/<file>`.

**How it was done.** Scraped the page HTML for `*.nc` hrefs (30 unique), then
`curl -fsSL` each into the destination directory. No git commands were run
(per CLAUDE.md). Two questions raised in the Q&A section above.