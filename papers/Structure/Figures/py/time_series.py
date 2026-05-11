"""Time series at a single QG grid point.

Plots one of:
  - relative vorticity zeta = dv/dx - du/dy  [s^-1]
  - zonal velocity      u                    [m s^-1]
  - meridional velocity v                    [m s^-1]

Reuses the differencing stencil from `relvor_movie.py` and the dataset
loader from `Analysis/py/qg_utils.py`.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Reuse the relative-vorticity stencil and grid spacing from the movie module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from relvor_movie import relative_vorticity  # noqa: E402

# QG dataset loader
_ANALYSIS_PY = os.path.abspath(os.path.join(_HERE, '..', '..', 'Analysis', 'py'))
if _ANALYSIS_PY not in sys.path:
    sys.path.insert(0, _ANALYSIS_PY)
import qg_utils  # noqa: E402


# Allowed time-series quantities and their plot metadata.
# Keys:
#   ylabel  -- y-axis label (LaTeX)
#   title   -- title prefix shown in the figure title
#   default_outfile -- default output filename if the caller doesn't pass one
_QUANTITIES = {
    'zeta': {
        'ylabel': r'$\zeta = \partial_x v - \partial_y u$  [s$^{-1}$]',
        'title': 'Relative vorticity',
        'default_outfile': 'fig_time_series_relvor.png',
        'color': 'k',
    },
    'u': {
        'ylabel': r'$u$  [m s$^{-1}$]',
        'title': 'Zonal velocity $u$',
        'default_outfile': 'fig_time_series_u.png',
        'color': 'r',
    },
    'v': {
        'ylabel': r'$v$  [m s$^{-1}$]',
        'title': 'Meridional velocity $v$',
        'default_outfile': 'fig_time_series_v.png',
        'color': 'g',
    },
}


def fig_time_series(
    x_center_km: float = 500.0,
    y_center_km: float = 500.0,
    dt_days: int = 1,
    duration_days: int = 365 * 5,
    start_before_end_days: int = 365 * 10,
    lev: int = 1,
    quantity: str = 'zeta',
    outfile: str = None,
):
    """Plot a 1-D time series at the QG grid cell closest to
    (x_center_km, y_center_km).

    The time window starts `start_before_end_days` before the end of the run
    and spans `duration_days`, sampled every `dt_days`.  With the defaults
    this is [end - 10y, end - 5y], daily.

    Parameters
    ----------
    x_center_km, y_center_km : float
        Target point (km).  The nearest grid cell is used.
    dt_days : int
        Stride between samples in days.
    duration_days : int
        Total span of the figure in days.
    start_before_end_days : int
        Offset of the window's start from the end of the model run.
    lev : int
        Vertical level (dataset has lev=[1, 2]; 1 is the top).
    quantity : {'zeta', 'u', 'v'}
        Which field to plot:
          - 'zeta' (default): relative vorticity dv/dx - du/dy [s^-1]
          - 'u'              : zonal velocity      [m s^-1]
          - 'v'              : meridional velocity [m s^-1]
    outfile : str, optional
        Output PNG path.  If None, falls back to the per-quantity default.
    """
    # Validate the quantity selector up front so a typo fails fast
    if quantity not in _QUANTITIES:
        raise ValueError(
            f"quantity must be one of {sorted(_QUANTITIES)}; got {quantity!r}")
    qmeta = _QUANTITIES[quantity]
    if outfile is None:
        outfile = qmeta['default_outfile']

    # Load the QG dataset and pick the requested vertical level
    qg, _ = qg_utils.load_qg()
    qg_lev = qg.sel(lev=lev)

    # Build the time-index window: start `start_before_end_days` before the
    # end of the run, span `duration_days`, sub-sample by `dt_days`.
    n_time = qg_lev.sizes['time']
    t_start = max(0, n_time - start_before_end_days)
    t_end = min(n_time, t_start + duration_days)
    time_idx = np.arange(t_start, t_end, dt_days)
    print(f'Time series ({quantity}): idx [{time_idx[0]}, {time_idx[-1]}], '
          f'{time_idx.size} samples (dt={dt_days} days)')

    # Locate the grid cell closest to the requested centre (coords in metres)
    x_m = qg.x.values
    y_m = qg.y.values
    ix = int(np.argmin(np.abs(x_m - x_center_km * 1e3)))
    iy = int(np.argmin(np.abs(y_m - y_center_km * 1e3)))
    print(f'Sampling grid cell (ix={ix}, iy={iy}) at '
          f'({x_m[ix] / 1e3:.2f} km, {y_m[iy] / 1e3:.2f} km)')

    # Pull the requested field at the single grid cell.  For zeta we still
    # need u, v across the full level so the periodic centered difference
    # is correct, but for plain u or v we can index the cell directly,
    # which avoids loading the whole level from disk.
    if quantity == 'zeta':
        u_all = qg_lev.u.isel(time=time_idx).values  # (nt, ny, nx)
        v_all = qg_lev.v.isel(time=time_idx).values
        series = relative_vorticity(u_all, v_all)[:, iy, ix]
    elif quantity == 'u':
        series = qg_lev.u.isel(time=time_idx, y=iy, x=ix).values
    else:  # 'v'
        series = qg_lev.v.isel(time=time_idx, y=iy, x=ix).values

    # Time axis in weeks since the start of the run
    weeks = qg.time.isel(time=time_idx).values / (86400.0 * 7.0)

    # Plot
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.plot(weeks, series, color=qmeta['color'], lw=0.8)
    # Zero reference line (useful for both zeta and signed velocities)
    ax.axhline(0.0, color='k', lw=0.5, ls='--')

    ax.set_xlabel('time [weeks]')
    ax.set_ylabel(qmeta['ylabel'])
    ax.set_title(
        f"{qmeta['title']} at "
        f"({x_m[ix] / 1e3:.1f}, {y_m[iy] / 1e3:.1f}) km, lev={lev}"
    )
    # Light grid for easier reading of the time series
    ax.grid(True, which='both', ls=':', lw=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f'Saved: {outfile}')


def main(flg):
    if flg == 'all':
        flg = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg = int(flg)

    # Default: zeta(t) at (500, 500) km, [end-10y, end-5y], daily, top layer.
    if flg == 1:
        fig_time_series(
            x_center_km=500.0,
            y_center_km=500.0,
            dt_days=1,
            duration_days=365 * 5,
            start_before_end_days=365 * 10,
            quantity='zeta',
            outfile='fig_time_series_relvor.png',
        )

    # Same window/location, but plot zonal velocity u
    if flg == 2:
        fig_time_series(
            x_center_km=500.0,
            y_center_km=500.0,
            dt_days=1,
            duration_days=365 * 5,
            start_before_end_days=365 * 10,
            quantity='u',
            outfile='fig_time_series_u.png',
        )

    # Same window/location, but plot meridional velocity v
    if flg == 3:
        fig_time_series(
            x_center_km=500.0,
            y_center_km=500.0,
            dt_days=1,
            duration_days=365 * 5,
            start_before_end_days=365 * 10,
            quantity='v',
            outfile='fig_time_series_v.png',
        )


if __name__ == '__main__':
    if len(sys.argv) == 1:
        flg = 0
    else:
        flg = sys.argv[1]
    main(flg)
