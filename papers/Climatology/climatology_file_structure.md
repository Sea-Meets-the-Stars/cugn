# CUGN Climatology — File Structure Report

**Date:** 2026-07-10
**Author:** Claude (Claude Code)
**Data location:** `$OS_SPRAY/CUGN/Climatology/` (30 NetCDF files, ~1.6 GB)
**Source:** https://spraydata.ucsd.edu/products/cugn-climatology/
**Reference:** Rudnick, D. L., K. D. Zaba, R. E. Todd, and R. E. Davis, 2016,
*A climatology of the California Current System from a network of underwater
gliders*, Prog. Oceanogr. — DOI `10.21238/S8SPRAY7292`, product version **v3**.

This report is generated from two scripts in this directory:
`inspect_climatology_files.py` (writes `climatology_structure.json`) and
`compare_products.py` (probes the product relationships). Both run under the
`ocean14` conda environment with the `h5netcdf` engine (files are NetCDF‑4/HDF5).

---

## 1. Overview: a 5 × 2 × 3 matrix

The 30 files are the full cross-product of **5 product types × 2 vertical
coordinates × 3 CUGN lines**. Filename pattern:

```
<product>_<vcoord>_<line>.nc
```

| Axis | Values |
|------|--------|
| **Product** | `mean`, `ancycle`, `mean_ancycle`, `total`, `anomaly` |
| **Vertical coord** | `z` (depth, m) · `rho` (potential density, kg m⁻³) |
| **Line** | `66`, `80`, `90` (CalCOFI lines 66.7, 80.0, 90.0) |

> **Note — no Line 56.** The core `cugn` package (`defs.py`) works with lines
> `56.0, 66.7, 80.0, 90.0`, but the climatology product provides only **66, 80,
> 90**. Line 56.0 is absent. The line labels here are the bare integers `66/80/90`.

---

## 2. Dimensions

Every file is organized on an along-track **`distance`** axis (0–400 km offshore
from the coast) plus a vertical axis, and — for all products except `mean` — a
**`time`** axis.

| Dim | Meaning | Line 66 | Line 80 | Line 90 |
|-----|---------|:-------:|:-------:|:-------:|
| `distance` | km offshore along the line (0–400 km) | 81 | 74 | 107 |
| `depth` | z-files only: 10–500 m, 50 levels (10 m steps) | 50 | 50 | 50 |
| `density` | rho-files only: 25.0–27.0 kg m⁻³, 21 levels (0.1 steps) | 21 | 21 | 21 |
| `time` | see §5 | 365 / 740 | 365 / 814 | 365 / 777 |

The `distance` count differs per line (different line lengths); depth (50) and
density (21) levels are identical across lines.

---

## 3. Coordinates

All files carry these coordinate variables:

- **`distance`** (km) — along-track distance offshore, 0–400 km.
- **`latitude`**, **`longitude`** — 1-D along `distance` (the nominal line track,
  e.g. Line 66: 35.09–36.89 °N, −125.69 to −121.84 °E).
- **`depth`** (m, z-files) *or* **`density`** (kg m⁻³, rho-files) — the vertical axis.
- **`time`** (all except `mean`) — **see the epoch warning in §5.**

Global attributes are rich and CF-1.7 / ACDD-1.3 compliant (institution = Scripps
IDG, `processing_level` = "Level 4", geospatial bounds polygon, DOI, contributor
list led by D. Rudnick, etc.).

---

## 4. Data variables

The variable set depends on the vertical coordinate:

**z-files (depth coordinate) — 5 variables:**

| Variable | Units | Long name |
|----------|-------|-----------|
| `temperature` | °C | Sea Water Temperature [ITS-90] |
| `salinity` | 1 | Practical Salinity |
| `density` | kg m⁻³ | Density − 1000 |
| `potential_density` | kg m⁻³ | Potential density − 1000 |
| `geostrophic_velocity` | m s⁻¹ | Geostrophic Velocity |

**rho-files (density coordinate) — 4 variables:**

| Variable | Units | Long name |
|----------|-------|-----------|
| `temperature` | °C | Sea Water Temperature [ITS-90] |
| `salinity` | 1 | Practical Salinity |
| `geostrophic_velocity` | m s⁻¹ | Geostrophic Velocity |
| `depth` | m | Depth (of each density surface) |

In density coordinates the two density variables are replaced by a single
`depth` **variable** (the depth of each isopycnal), which is the natural
diagnostic on a density surface. Variables carry the `-1000` offset convention
for (potential) density (i.e. σ, not full ρ).

---

## 5. The product decomposition (verified numerically)

The five products form a **mean + seasonal-cycle + anomaly** decomposition of
the fields. Using Line 66, depth, `temperature`, `compare_products.py` confirms:

| Product | Time dim | What it is |
|---------|:--------:|------------|
| **`mean`** | *none* | Time-independent spatial mean field, `(distance, vcoord)`. |
| **`ancycle`** | 365 | Annual (seasonal) cycle as a **zero-mean deviation** from `mean`, on a representative-year daily grid. Verified: time-mean(`ancycle`) ≈ 0 (±0.001 °C). |
| **`mean_ancycle`** | 365 | Full seasonal **climatology** = `mean + ancycle`. **Verified to 100%** (`mean_ancycle == mean + ancycle`, atol 1e-4). |
| **`total`** | 740* | Full reconstructed field on the **actual observation times**. Equals seasonal climatology(t) + `anomaly`. |
| **`anomaly`** | 740* | Interannual **residual** = `total − climatology(t)`, on the actual observation times. |

*Time length is per-line (66→740, 80→814, 90→777).

Evidence that `total` and `anomaly` are distinct and related by the climatology:
`total` and `anomaly` share zero identical points; `(total − anomaly)` has range
[5.24, 17.97] °C and mean 8.21 °C — matching the `mean_ancycle` range
[5.24, 17.99] and the `mean` value 8.21. So `total − anomaly` **is** the seasonal
climatology evaluated at the observation times.

This resolves the earlier question about the same-sized file pairs: the pairs are
**not** duplicates. `ancycle`/`mean_ancycle` differ by the additive `mean`; they
happen to be the same on-disk size because they have identical shape and similar
entropy. `total`/`anomaly` likewise share shape but hold different fields.

### ⚠️ Two different time epochs (read carefully)

The `time` coordinate uses **different epochs** across products:

| Products | `time:units` | Interpretation |
|----------|--------------|----------------|
| `ancycle`, `mean_ancycle` | `days since 0001-01-01` | Representative year (≈ 2005‑09 → 2006‑09), i.e. a day-of-year seasonal axis. |
| `total`, `anomaly` | `days since 1970-01-01` | Real calendar dates of the observation record. |

Decoded calendar spans (z-files):

| Line | `total`/`anomaly` span | Steps | Cadence |
|------|------------------------|:-----:|:-------:|
| 66 | 2007‑01‑01 → 2026‑12‑27 | 740 | ~10 days |
| 80 | 2005‑01‑01 → 2026‑12‑27 | 814 | ~10 days |
| 90 | 2006‑01‑01 → 2026‑12‑27 | 777 | ~10 days |

Because the epochs differ, **do not** let a reader blindly `decode_times=True`
across products and then compare time axes — the seasonal products would decode
to years ~2005 and the observational products to their true dates. Read with
`decode_times=False` and convert explicitly, or handle each epoch separately.

Provenance note: the `mean`/`ancycle`/`mean_ancycle` files were `date_created`
2017‑04‑06 (the original v3 climatology), whereas `total`/`anomaly` were freshly
created 2026‑07‑07 — consistent with the anomalies being an extension of the
record through 2026.

---

## 6. Practical notes for reading the data

- **Engine:** open with `xarray.open_dataset(path, engine="h5netcdf")` (files are
  NetCDF‑4/HDF5; the `h5py`/`netcdf4` backends were added to the env + `setup.py`).
- **Time:** prefer `decode_times=False` and convert with the per-product epoch
  (§5), or decode per file — never mix epochs.
- **Density offset:** `(potential_)density` are σ values (true value − 1000).
- **Missing data:** fields are heavily masked with NaN where the glider grid has
  no coverage (e.g. `mean` temperature Line 66 has 3,901 finite of 81×50 = 4,050).

---

## 7. Sizes at a glance

| Product | z-file size (per line) | rho-file size |
|---------|:----------------------:|:-------------:|
| `mean` | 0.18–0.25 MB | 0.08–0.10 MB |
| `ancycle` = `mean_ancycle` | 54–78 MB | 18–26 MB |
| `total` = `anomaly` | 120–166 MB | 40–56 MB |

(z-files are larger than rho-files: 50 depth levels vs 21 density levels, and
carry one extra variable.)

---

## 8. Open questions (also logged to the Q&A of `explore_prompts.md`)

1. **Line 56.** Is Line 56.0 intentionally excluded from the climatology product,
   or is it simply not published yet? The core package expects it.
2. **`geostrophic_velocity` sign/reference.** What is the reference level and
   positive direction (along-line vs cross-line)?
3. **`total` vs `anomaly` intended use.** Should downstream analysis consume
   `anomaly` (interannual signal) directly, or reconstruct from `total`?
4. **rho `depth` variable.** Is it the mean depth of each isopycnal, and is it
   directly comparable to the z-grid `depth` coordinate?
