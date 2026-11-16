"""Probe the relationships between the CUGN climatology product types.

The five products (mean, ancycle, mean_ancycle, total, anomaly) share the same
(distance, depth|density [, time]) grid per line. Their filenames suggest a
decomposition of each observed field into a spatial mean, a seasonal
(annual-cycle) part, and a residual anomaly. This script tests those
relationships numerically for line 66 in depth (z) coordinates, using the
``temperature`` variable as a representative field.

Hypotheses tested:
  H1  mean_ancycle == mean + ancycle           (full seasonal climatology)
  H2  ancycle is a zero-mean-over-year deviation (annual mean of ancycle ~ 0)
  H3  total and anomaly are numerically distinct products
  H4  total == (seasonal climatology at obs times) + anomaly
      i.e. anomaly == total - climatology(day-of-year)

Nothing is written to disk; results print to stdout so they can be quoted in
the report. Run per CLAUDE.md:

    conda run -n ocean14 python compare_products.py
"""

import os

import numpy as np
import xarray as xr

VAR = "temperature"
ENG = "h5netcdf"


def clim_dir():
    os_spray = os.environ["OS_SPRAY"]
    return os.path.join(os_spray, "CUGN", "Climatology")


def load(name):
    return xr.open_dataset(os.path.join(clim_dir(), name), engine=ENG,
                           decode_times=False)


def summarize(label, arr):
    finite = arr[np.isfinite(arr)]
    print(f"  {label:32s} n_finite={finite.size:>9d}  "
          f"min={np.min(finite):+.4f}  max={np.max(finite):+.4f}  "
          f"mean={np.mean(finite):+.4f}")


def close_fraction(a, b, atol=1e-6):
    """Fraction of positions where both are finite and |a-b| <= atol."""
    both = np.isfinite(a) & np.isfinite(b)
    if both.sum() == 0:
        return float("nan"), 0
    diff = np.abs(a[both] - b[both])
    return float((diff <= atol).mean()), int(both.sum())


def main():
    mean = load("mean_z_66.nc")[VAR]                 # (distance, depth)
    ancycle = load("ancycle_z_66.nc")[VAR]           # (distance, depth, time=365)
    mean_ancycle = load("mean_ancycle_z_66.nc")[VAR] # (distance, depth, time=365)
    total = load("total_z_66.nc")[VAR]               # (distance, depth, time=740)
    anomaly = load("anomaly_z_66.nc")[VAR]           # (distance, depth, time=740)

    print(f"Variable: {VAR}  (line 66, depth coords)\n")
    print("Raw field summaries:")
    summarize("mean", mean.values)
    summarize("ancycle", ancycle.values)
    summarize("mean_ancycle", mean_ancycle.values)
    summarize("total", total.values)
    summarize("anomaly", anomaly.values)
    print()

    # H1: mean_ancycle == mean + ancycle  (broadcast mean over the 365 days)
    recon = mean.values[:, :, None] + ancycle.values
    frac, n = close_fraction(recon, mean_ancycle.values, atol=1e-4)
    print(f"H1  mean_ancycle == mean + ancycle : "
          f"{frac*100:.2f}% of {n} points match (atol=1e-4)")

    # H2: annual mean of `ancycle` ~ 0 (it is a deviation from the mean)
    ann_mean = np.nanmean(ancycle.values, axis=2)
    summarize("H2  time-mean(ancycle) [~0?]", ann_mean)

    # also: annual mean of mean_ancycle ~ mean ?
    ann_mean_ma = np.nanmean(mean_ancycle.values, axis=2)
    frac, n = close_fraction(ann_mean_ma, mean.values, atol=1e-3)
    print(f"    time-mean(mean_ancycle) == mean : "
          f"{frac*100:.2f}% of {n} points match (atol=1e-3)")
    print()

    # H3: are total and anomaly distinct?
    frac, n = close_fraction(total.values, anomaly.values, atol=1e-6)
    print(f"H3  total == anomaly (identical?) : "
          f"{frac*100:.2f}% of {n} points match  -> "
          f"{'IDENTICAL' if frac > 0.999 else 'DISTINCT'}")
    summarize("    (total - anomaly)", (total.values - anomaly.values))
    print()

    # H4: anomaly == total - seasonal climatology(day-of-year)?
    # Build day-of-year for the total/anomaly time axis and index the seasonal
    # climatology (mean_ancycle, 365 days) accordingly.
    t = load("total_z_66.nc")["time"]  # days since 1970-01-01
    # day-of-year 0..364 (approx; leap handling is not critical for a check)
    doy = (np.floor(t.values).astype(int) % 365)
    clim = mean_ancycle.values[:, :, doy]  # (distance, depth, time=740)
    recon_anom = total.values - clim
    frac, n = close_fraction(recon_anom, anomaly.values, atol=1e-2)
    print(f"H4  anomaly == total - clim(doy) : "
          f"{frac*100:.2f}% of {n} points match (atol=1e-2, rough doy index)")


if __name__ == "__main__":
    main()
