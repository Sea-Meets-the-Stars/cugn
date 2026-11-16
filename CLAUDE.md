# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CUGN (California Underwater Glider Network) is a Python package for analyzing oceanographic data from Spray underwater gliders operating along the California coast. The project processes glider profiles to study oxygen dynamics, water mass properties, and ocean structure.

## Working Conventions

- **Git:** The user (not Claude) will perform all git commands (add, commit, push, etc.). Do not run git commands unless explicitly asked.
- **Calculations:** If you do any calculation, generate it as a Python script and write it to disk so that it can be added to the repository. Do not perform one-off calculations only in memory or in the chat.
- **Python environment:** If you need to run Python, use the `ocean14` conda environment.

## Environment Setup

**Required environment variable:**
- `OS_SPRAY`: Path to Spray glider data directory (data lives in `$OS_SPRAY/CUGN/`)
- `OS_CCS`: Path to California Current System data (for upwelling indices)

**Installation:**
```bash
pip install -e .
```

## Dependency: profiler Package

CUGN depends on the `profiler` package (located at `../profiler/` in this workspace) for base profiler data classes and pair analysis. Key imports from profiler:

- `profiler.gliderdata.SprayData`: Base class for Spray glider data
- `profiler.floatdata`: Float data classes (Solo, Flip, etc.)
- `profiler.profilerpairs.ProfilerPairs`: Profile pair analysis for structure functions

Install profiler alongside CUGN:
```bash
pip install -e ../profiler
```

## Running Tests

```bash
pytest cugn/tests/
pytest cugn/tests/test_spray.py  # single test file
pytest cugn/tests/test_spray.py::test_single_spray  # single test
```

Note: Tests require actual glider data files and may not pass without the proper data setup.

## Key Architecture

### Data Flow
1. Raw Spray glider NetCDF files (`CUGN_line_*.nc`) are processed by `process.py` to add derived quantities (potential density, oxygen saturation, etc.) producing `CUGN_potential_line_*.nc`
2. Grid tables (`.parquet`) are built that bin data by salinity (SA) and density (sigma0), enabling water mass analysis
3. Analysis modules work with these grid tables to identify outliers and cluster them

### Core Modules

- **defs.py**: Global constants including CUGN lines (56.0, 66.7, 80.0, 90.0), thresholds for oxygen extrema (SO_hyper=1.1, AOU_hyper=25)
- **io.py**: Data loading functions; `load_line()` returns dict with xarray Dataset, grid table DataFrame, and bin edges
- **process.py**: Main processing script; run as `python process.py <flag>` where flag 1=add GSW quantities, 2=build full grids, 3=build control grids
- **grid_utils.py**: 2D binning in SA-sigma0 space, outlier detection by percentile, grid cell statistics
- **highres.py**: High-resolution profile analysis for MLD (mixed layer depth) and N (buoyancy frequency)
- **clusters.py**: DBSCAN clustering of oxygen outliers in time-distance-depth space

### Data Structures

Grid tables (pandas DataFrames) contain:
- `depth`, `profile`: indices into the xarray Dataset
- `row`, `col`: indices into the SA-sigma0 grid
- `doxy`, `doxy_p`: dissolved oxygen and its percentile within grid cell
- `MLD`, `N`, `zNpeak`: derived physical quantities

### Line Identifiers
Lines are identified by strings: '56.0', '66.7', '80.0', '90.0' corresponding to CalCOFI line numbers along the California coast.

## Papers Directory

Contains analysis scripts and figures for specific publications (ARCTERX, SO, Structure). These are research-specific and not part of the core library.
