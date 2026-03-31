"""
Small-box drifter analysis for the QG model.

Deploys drifters in a localized box (default 100 km x 100 km) and provides
analysis functions for pair separations and structure functions.

Usage:
    from py.small_box_drifters import run_small_box, compute_pair_separations

    traj, meta = run_small_box(t_start=5001, n_days=100)
    pairs = compute_pair_separations(traj, meta)
"""

import numpy as np
import pandas as pd
import xarray as xr
import os

from qg_drifters import run_drifters, load_trajectories


def run_small_box(
    t_start=5001,
    n_days=100,
    box_size_km=100.0,
    drifter_spacing_km=10.0,
    box_center_x=128.0,
    box_center_y=128.0,
    lev=1,
    record_interval=1,
    output_path=None,
    cache=True,
    verbose=True,
):
    """Run a small-box drifter simulation.

    Thin wrapper around qg_drifters.run_drifters() with box-specific defaults.

    Parameters
    ----------
    t_start : int
        Starting time index (1-based). Default 5001 (day ~5000).
    n_days : int
        Number of days to advect. Default 100.
    box_size_km : float
        Side length of the deployment box in km. Default 100.
    drifter_spacing_km : float
        Spacing between drifters in km. Default 10.
    box_center_x, box_center_y : float
        Center of the box in grid-index units. Default (128, 128).
    lev : int
        Vertical level. Default 1 (upper).
    record_interval : int
        Record positions every N days.
    output_path : str or Path, optional
        Output CSV path.
    cache : bool
        Skip re-running if output exists.
    verbose : bool
        Stream Julia output.

    Returns
    -------
    traj : pd.DataFrame
        Trajectory data with columns: ID, x, y, t, x_m, y_m
    meta : dict
        Metadata including box parameters.
    """
    return run_drifters(
        t_start=t_start,
        n_days=n_days,
        n_per_side=1,  # ignored when box_size_km is set
        lev=lev,
        record_interval=record_interval,
        box_center_x=box_center_x,
        box_center_y=box_center_y,
        box_size_km=box_size_km,
        drifter_spacing_km=drifter_spacing_km,
        output_path=output_path,
        cache=cache,
        verbose=verbose,
    )


def compute_pair_separations(traj, meta):
    """Compute pair separations at each recorded time step.

    Uses the periodic minimum-image convention.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory data with columns: ID, x_m, y_m, t
    meta : dict
        Must contain 'dx' and 'nx'.

    Returns
    -------
    pairs : pd.DataFrame
        Columns: ID_i, ID_j, t, r_m, r0_m
        r_m: separation in meters at time t
        r0_m: initial separation in meters
    """
    dx = meta["dx"]
    nx = meta["nx"]
    L = dx * nx  # domain size in meters

    ids = sorted(traj.ID.unique())
    n = len(ids)
    times = np.sort(traj.t.unique())

    # Build initial separation lookup
    init = traj[traj.t == times[0]].set_index("ID")

    # Generate all unique pairs
    pair_list = []
    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            pair_list.append((ids[i_idx], ids[j_idx]))

    # Compute initial separations
    r0_dict = {}
    for (i, j) in pair_list:
        ddx = init.loc[i, "x_m"] - init.loc[j, "x_m"]
        ddy = init.loc[i, "y_m"] - init.loc[j, "y_m"]
        ddx -= L * round(ddx / L)
        ddy -= L * round(ddy / L)
        r0_dict[(i, j)] = np.sqrt(ddx**2 + ddy**2)

    # Compute separations at all times
    records = []
    for t_val in times:
        sub = traj[traj.t == t_val].set_index("ID")
        for (i, j) in pair_list:
            ddx = sub.loc[i, "x_m"] - sub.loc[j, "x_m"]
            ddy = sub.loc[i, "y_m"] - sub.loc[j, "y_m"]
            ddx -= L * round(ddx / L)
            ddy -= L * round(ddy / L)
            r = np.sqrt(ddx**2 + ddy**2)
            records.append((i, j, t_val, r, r0_dict[(i, j)]))

    pairs = pd.DataFrame(records, columns=["ID_i", "ID_j", "t", "r_m", "r0_m"])
    return pairs


def compute_structure_functions(traj, meta, nc_path=None, r_bins=None):
    """Compute second-order Lagrangian velocity structure functions.

    Interpolates gridded velocity to drifter positions, computes velocity
    differences for all pairs, and bins by separation distance.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory data with columns: ID, x, y, t, x_m, y_m
    meta : dict
        Must contain 'dx', 'nx', 't_start'.
    nc_path : str, optional
        Path to QG NetCDF file. Defaults to $OS_DATA/QG/QGModelOutput20years.nc.
    r_bins : array-like, optional
        Bin edges for separation distance in meters.
        Default: 20 log-spaced bins from 5 km to 500 km.

    Returns
    -------
    sf : pd.DataFrame
        Columns: r_mid (bin center, m), D_LL, D_TT, D_2, n_pairs
    """
    if nc_path is None:
        nc_path = os.path.join(os.environ["OS_DATA"], "QG", "QGModelOutput20years.nc")

    dx = meta["dx"]
    nx = meta["nx"]
    L = dx * nx
    t_start = meta["t_start"]

    if r_bins is None:
        r_bins = np.logspace(np.log10(5e3), np.log10(500e3), 21)

    ds = xr.open_dataset(nc_path)

    ids = sorted(traj.ID.unique())
    n = len(ids)
    times = np.sort(traj.t.unique())

    # Accumulate structure function data
    n_bins = len(r_bins) - 1
    dll_sum = np.zeros(n_bins)
    dtt_sum = np.zeros(n_bins)
    d2_sum = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for t_val in times:
        # Get time index in NetCDF
        day_idx = int(t_val / 86400.0)
        t_nc = t_start - 1 + day_idx  # 0-based for xarray

        # Load velocity field
        u_field = ds.u.isel(lev=0, time=t_nc).values  # (y, x) in xarray
        v_field = ds.v.isel(lev=0, time=t_nc).values

        # Get drifter positions at this time
        sub = traj[traj.t == t_val].set_index("ID")

        # Interpolate velocity to drifter positions (bilinear)
        u_at_drifter = {}
        v_at_drifter = {}
        for did in ids:
            # Grid-index positions
            gx = sub.loc[did, "x"]
            gy = sub.loc[did, "y"]

            # Bilinear interpolation with periodic wrapping
            ix = int(np.floor(gx)) % nx
            iy = int(np.floor(gy)) % nx
            ix1 = (ix + 1) % nx
            iy1 = (iy + 1) % nx
            fx = gx - np.floor(gx)
            fy = gy - np.floor(gy)

            u_at_drifter[did] = (
                (1 - fx) * (1 - fy) * u_field[iy, ix]
                + fx * (1 - fy) * u_field[iy, ix1]
                + (1 - fx) * fy * u_field[iy1, ix]
                + fx * fy * u_field[iy1, ix1]
            )
            v_at_drifter[did] = (
                (1 - fx) * (1 - fy) * v_field[iy, ix]
                + fx * (1 - fy) * v_field[iy, ix1]
                + (1 - fx) * fy * v_field[iy1, ix]
                + fx * fy * v_field[iy1, ix1]
            )

        # Compute structure functions for all pairs
        for i_idx in range(n):
            for j_idx in range(i_idx + 1, n):
                i, j = ids[i_idx], ids[j_idx]

                # Separation vector (periodic)
                ddx = sub.loc[i, "x_m"] - sub.loc[j, "x_m"]
                ddy = sub.loc[i, "y_m"] - sub.loc[j, "y_m"]
                ddx -= L * round(ddx / L)
                ddy -= L * round(ddy / L)
                r = np.sqrt(ddx**2 + ddy**2)

                if r < r_bins[0] or r >= r_bins[-1]:
                    continue

                # Unit separation vector
                rx, ry = ddx / r, ddy / r

                # Velocity difference
                du = u_at_drifter[i] - u_at_drifter[j]
                dv = v_at_drifter[i] - v_at_drifter[j]

                # Longitudinal and transverse components
                du_L = du * rx + dv * ry
                du_T = -du * ry + dv * rx

                # Bin
                b = np.searchsorted(r_bins, r) - 1
                if 0 <= b < n_bins:
                    dll_sum[b] += du_L**2
                    dtt_sum[b] += du_T**2
                    d2_sum[b] += du**2 + dv**2
                    counts[b] += 1

    ds.close()

    # Average
    mask = counts > 0
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    dll = np.where(mask, dll_sum / counts, np.nan)
    dtt = np.where(mask, dtt_sum / counts, np.nan)
    d2 = np.where(mask, d2_sum / counts, np.nan)

    sf = pd.DataFrame({
        "r_mid": r_mid,
        "D_LL": dll,
        "D_TT": dtt,
        "D_2": d2,
        "n_pairs": counts,
    })
    return sf
