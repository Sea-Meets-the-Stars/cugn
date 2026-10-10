"""
Analysis functions for QG drifter trajectories.

Pair separations and Lagrangian velocity structure functions,
with velocities derived from drifter trajectories (finite differences).
"""

import numpy as np
import pandas as pd


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
    L = dx * nx

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


def _compute_drifter_velocities(traj, meta):
    """Compute Lagrangian velocities from drifter trajectories via centered finite differences.

    At interior time steps, uses centered differences. At endpoints, uses
    forward/backward differences. Applies periodic minimum-image for displacements.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory data with columns: ID, x_m, y_m, t
    meta : dict
        Must contain 'dx' and 'nx'.

    Returns
    -------
    vel : pd.DataFrame
        Same as traj with added columns: u_m (m/s), v_m (m/s)
    """
    dx = meta["dx"]
    nx = meta["nx"]
    L = dx * nx

    ids = sorted(traj.ID.unique())
    times = np.sort(traj.t.unique())
    nt = len(times)

    # Build lookup: id -> array of (x_m, y_m) indexed by time position
    pos_x = {}
    pos_y = {}
    for did in ids:
        sub = traj[traj.ID == did].sort_values("t")
        pos_x[did] = sub.x_m.values
        pos_y[did] = sub.y_m.values

    records = []
    for did in ids:
        x_arr = pos_x[did]
        y_arr = pos_y[did]
        for k in range(nt):
            if k == 0:
                # Forward difference
                dt = times[1] - times[0]
                dxm = x_arr[1] - x_arr[0]
                dym = y_arr[1] - y_arr[0]
            elif k == nt - 1:
                # Backward difference
                dt = times[-1] - times[-2]
                dxm = x_arr[-1] - x_arr[-2]
                dym = y_arr[-1] - y_arr[-2]
            else:
                # Centered difference
                dt = times[k + 1] - times[k - 1]
                dxm = x_arr[k + 1] - x_arr[k - 1]
                dym = y_arr[k + 1] - y_arr[k - 1]

            # Periodic minimum-image for displacement
            dxm -= L * round(dxm / L)
            dym -= L * round(dym / L)

            records.append((did, times[k], x_arr[k], y_arr[k], dxm / dt, dym / dt))

    vel = pd.DataFrame(records, columns=["ID", "t", "x_m", "y_m", "u_m", "v_m"])
    return vel


def compute_structure_functions(traj, meta, r_bins=None):
    """Compute second-order Lagrangian velocity structure functions.

    Velocities are derived from drifter trajectories via finite differences
    (not from the QG model output).

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory data with columns: ID, x, y, t, x_m, y_m
    meta : dict
        Must contain 'dx' and 'nx'.
    r_bins : array-like, optional
        Bin edges for separation distance in meters.
        Default: 20 log-spaced bins from 5 km to 500 km.

    Returns
    -------
    sf : pd.DataFrame
        Columns: r_mid (bin center, m), D_LL, D_TT, D_2, n_pairs
    """
    dx = meta["dx"]
    nx = meta["nx"]
    L = dx * nx

    if r_bins is None:
        r_bins = np.logspace(np.log10(5e3), np.log10(500e3), 21)

    # Compute velocities from trajectories
    vel = _compute_drifter_velocities(traj, meta)

    ids = sorted(vel.ID.unique())
    n = len(ids)
    times = np.sort(vel.t.unique())

    n_bins = len(r_bins) - 1
    dll_sum = np.zeros(n_bins)
    dtt_sum = np.zeros(n_bins)
    d2_sum = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for t_val in times:
        sub = vel[vel.t == t_val].set_index("ID")

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

                # Velocity difference (from trajectory-derived velocities)
                du = sub.loc[i, "u_m"] - sub.loc[j, "u_m"]
                dv = sub.loc[i, "v_m"] - sub.loc[j, "v_m"]

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

    # Average
    mask = counts > 0
    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
    dll = np.full(n_bins, np.nan)
    dtt = np.full(n_bins, np.nan)
    d2 = np.full(n_bins, np.nan)
    dll[mask] = dll_sum[mask] / counts[mask]
    dtt[mask] = dtt_sum[mask] / counts[mask]
    d2[mask] = d2_sum[mask] / counts[mask]

    sf = pd.DataFrame({
        "r_mid": r_mid,
        "D_LL": dll,
        "D_TT": dtt,
        "D_2": d2,
        "n_pairs": counts,
    })
    return sf
