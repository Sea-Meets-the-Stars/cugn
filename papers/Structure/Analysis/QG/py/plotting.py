"""
Plotting utilities for QG drifter analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os


def plot_trajectories_over_speed(traj, meta, nc_path=None, max_drifters=None,
                                  ax=None, figsize=(8, 7), title=None):
    """Plot drifter trajectories over the QG speed field.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory data with columns: ID, x_m, y_m, t
    meta : dict
        Must contain 'dx', 'nx', 't_start'.
    nc_path : str, optional
        Path to QG NetCDF file. Defaults to $OS_DATA/QG/QGModelOutput20years.nc.
    max_drifters : int, optional
        Max number of drifter trajectories to plot. Default: all.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    figsize : tuple
        Figure size (only used if ax is None).
    title : str, optional
        Plot title. Default: auto-generated.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if nc_path is None:
        nc_path = os.path.join(os.environ["OS_DATA"], "QG", "QGModelOutput20years.nc")

    t_start = meta["t_start"]
    dx = meta["dx"]
    nx = meta["nx"]

    # Load speed field at start time
    qg = xr.open_dataset(nc_path)
    u0 = qg.u.isel(lev=0, time=t_start - 1).values
    v0 = qg.v.isel(lev=0, time=t_start - 1).values
    speed = np.sqrt(u0**2 + v0**2)
    x_km = qg.x.values / 1000
    y_km = qg.y.values / 1000
    qg.close()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.pcolormesh(x_km, y_km, speed.T, cmap="viridis", alpha=0.3, shading="auto")

    ids = sorted(traj.ID.unique())
    if max_drifters is not None:
        ids = ids[:max_drifters]

    for did in ids:
        sub = traj[traj.ID == did]
        ax.plot(sub.x_m / 1000, sub.y_m / 1000, "w-", lw=0.5, alpha=0.7)
        ax.plot(sub.x_m.iloc[0] / 1000, sub.y_m.iloc[0] / 1000, "r.", ms=3)
        ax.plot(sub.x_m.iloc[-1] / 1000, sub.y_m.iloc[-1] / 1000, "k.", ms=3)

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")

    if title is None:
        n_days = int(traj.t.max() / 86400)
        n_drifters = traj.ID.nunique()
        box_km = meta.get("box_size_km")
        if box_km:
            title = f"Small-box drifters ({n_drifters}, {box_km:.0f} km box, {n_days} days)"
        else:
            title = f"Drifter trajectories ({n_drifters} drifters, {n_days} days)"
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax
