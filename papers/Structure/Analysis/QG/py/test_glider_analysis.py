"""
Test the glider analysis pipeline with stationary gliders on the QG grid.

Deploys stationary virtual gliders at every grid point in a box,
reads velocities directly from the QG NetCDF (no interpolation),
and produces a DataFrame compatible with glider_analysis.compute_glider_sf().

Usage:
    from test_glider_analysis import build_stationary_glider_df, run_stationary_test

    # Build the DataFrame
    df, meta = build_stationary_glider_df(t_start=5001, n_days=10)

    # Or run end-to-end (builds df + computes SF + saves JSON)
    Sn_LL, Sn_TT = run_stationary_test(t_start=5001, n_days=10)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

from IPython import embed

# Default QG NetCDF path
_NC_PATH = os.path.join(os.environ.get('OS_DATA', ''), 'QG', 'QGModelOutput20years.nc')


def build_stationary_glider_df(
    t_start=5001,
    n_days=10,
    box_center_idx=(128, 128),
    box_size_km=100.0,
    lev=1,
    nc_path=None,
):
    """Build a DataFrame of stationary glider velocities from the QG NetCDF.

    Places one glider at every QG grid point within a box.  Reads u, v
    directly at integer grid indices (no interpolation).

    Parameters
    ----------
    t_start : int
        QG time index at which to begin (1-based, as in the Julia code).
        Converted to 0-based for xarray indexing.
    n_days : int
        Number of days to sample (reads t_start through t_start + n_days).
    box_center_idx : tuple of int
        (ix, iy) center of the box in grid-index units.
    box_size_km : float
        Side length of the box in km.
    lev : int
        Vertical level (1=upper, 2=lower; converted to 0-based for xarray).
    nc_path : str, optional
        Path to the QG NetCDF file.

    Returns
    -------
    df : pd.DataFrame
        Columns: x, y, time, missid, x_m, y_m, u_qg, v_qg
        x, y are in grid-index units.
    meta : dict
        Keys: dx, nx, t_start, n_days, lev, n_gliders, box_center_idx, box_size_km
    """
    if nc_path is None:
        nc_path = _NC_PATH

    ds = xr.open_dataset(nc_path)
    x_coord = ds['x'].values
    nx = len(x_coord)
    dx = float(x_coord[1] - x_coord[0])  # meters
    dx_km = dx / 1000.0

    # Determine box extent in grid indices
    half_n = int(np.floor(box_size_km / (2.0 * dx_km)))
    cx, cy = box_center_idx
    i_lo = cx - half_n
    i_hi = cx + half_n
    j_lo = cy - half_n
    j_hi = cy + half_n

    # Grid point indices within the box
    ix_arr = np.arange(i_lo, i_hi + 1)
    iy_arr = np.arange(j_lo, j_hi + 1)
    n_per_side = len(ix_arr)
    n_gliders = n_per_side * n_per_side

    # Time indices (convert 1-based t_start to 0-based for xarray)
    t_indices = np.arange(t_start - 1, t_start - 1 + n_days + 1)
    n_times = len(t_indices)

    # Convert lev to 0-based
    lev_idx = lev - 1

    print(f"Stationary glider grid: {n_per_side}x{n_per_side} = {n_gliders} gliders")
    print(f"  Box: [{i_lo}, {i_hi}] x [{j_lo}, {j_hi}] (grid indices)")
    print(f"  Time: {n_times} snapshots (t_start={t_start}, n_days={n_days})")
    print(f"  Level: {lev} (0-based index: {lev_idx})")

    # Read velocity slices for all times at once
    # u, v shape: (n_times, ny_box, nx_box) after slicing
    u_all = ds['u'].isel(
        time=t_indices, lev=lev_idx,
        y=slice(j_lo, j_hi + 1), x=slice(i_lo, i_hi + 1)
    ).values  # shape: (n_times, n_per_side, n_per_side)
    v_all = ds['v'].isel(
        time=t_indices, lev=lev_idx,
        y=slice(j_lo, j_hi + 1), x=slice(i_lo, i_hi + 1)
    ).values

    ds.close()

    # Build the DataFrame
    # For each time step and each grid point, create a row
    records = []
    missid = 0
    # Map (ix, iy) -> missid (fixed across time)
    missid_map = {}
    for jj, iy in enumerate(iy_arr):
        for ii, ix in enumerate(ix_arr):
            missid_map[(ix, iy)] = missid
            missid += 1

    embed(header='123 of build_stationary_glider_df')
    for t_idx_local, t_idx_global in enumerate(t_indices):
        # Time in seconds from t_start
        time_s = float(t_idx_local) * 86400.0
        for jj, iy in enumerate(iy_arr):
            for ii, ix in enumerate(ix_arr):
                mid = missid_map[(ix, iy)]
                u_val = float(u_all[t_idx_local, jj, ii])
                v_val = float(v_all[t_idx_local, jj, ii])
                records.append((
                    float(ix), float(iy), time_s, mid,
                    float(ix) * dx, float(iy) * dx,
                    u_val, v_val,
                ))

    df = pd.DataFrame(records, columns=[
        'x', 'y', 'time', 'missid', 'x_m', 'y_m', 'u_qg', 'v_qg'
    ])

    meta = {
        'dx': dx,
        'nx': nx,
        't_start': t_start,
        'n_days': n_days,
        'lev': lev,
        'n_gliders': n_gliders,
        'box_center_idx': list(box_center_idx),
        'box_size_km': box_size_km,
    }

    print(f"Built DataFrame: {len(df)} rows ({n_gliders} gliders x {n_times} times)")
    print(f"  u range: [{df.u_qg.min():.4f}, {df.u_qg.max():.4f}] m/s")
    print(f"  v range: [{df.v_qg.min():.4f}, {df.v_qg.max():.4f}] m/s")

    return df, meta


def run_stationary(
    t_start=5001,
    n_days=10,
    box_center_idx=(128, 128),
    box_size_km=100.0,
    lev=1,
    outfile_df:str=None,
    outfile_LL:str=None,
    outfile_TT:str=None):
    """End-to-end test: build stationary glider DataFrame, compute SF, save.

    Parameters
    ----------
    t_start : int
        QG start time (1-based).
    n_days : int
        Number of days.
    box_center_idx : tuple of int
        Center of the box in grid-index units.
    box_size_km : float
        Side length in km.
    lev : int
        Vertical level (1=upper).
    outfile_LL, outfile_TT : str
        Output JSON paths for longitudinal and transverse SFs.

    Returns
    -------
    Sn_LL : dict
        Longitudinal structure function dict.
    Sn_TT : dict
        Transverse structure function dict.
    """
    from glider_analysis import compute_glider_sf, save_sf

    # Build the stationary glider DataFrame
    df, meta = build_stationary_glider_df(
        t_start=t_start, n_days=n_days,
        box_center_idx=box_center_idx, box_size_km=box_size_km,
        lev=lev,
    )

    # Compute structure functions
    dr = 5 # meters
    rbins = np.arange(0, 1.3e2, dr) # 130 km
    Sn_LL, Sn_TT = compute_glider_sf(df, meta, r_bins_km=rbins)

    # Save
    if outfile_df is not None:
        df.to_csv(outfile_df, index=False)
    if outfile_LL is not None:  
        save_sf(Sn_LL, outfile_LL)
    if outfile_TT is not None:
        save_sf(Sn_TT, outfile_TT)

    return Sn_LL, Sn_TT


if __name__ == "__main__":
    # Test
    #run_stationary(t_start=5001, n_days=10)

    # Full
    run_stationary(t_start=5001, n_days=100,
        outfile_df='data/stationary_gliders_ts5001_nd100_df.csv',
        outfile_LL='data/stationary_gliders_ts5001_nd100_sf_LL.json',
        outfile_TT='data/stationary_gliders_ts5001_nd100_sf_TT.json')