"""Movie of relative vorticity in a region of the QG model output.

Computes zeta = dv/dx - du/dy from the QG velocity fields and renders an
mp4 animation of a square sub-region over a configurable time window.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Pull in the existing QG loader from Analysis/py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS_PY = os.path.abspath(os.path.join(_HERE, '..', '..', 'Analysis', 'py'))
if _ANALYSIS_PY not in sys.path:
    sys.path.insert(0, _ANALYSIS_PY)

import qg_utils  # noqa: E402

# Grid spacing of the QG run: 1000 km domain on 256 cells.
DX_KM = 1000.0 / 256.0
DX_M = DX_KM * 1000.0


def relative_vorticity(u, v, dx_m=DX_M):
    """Compute zeta = dv/dx - du/dy with centered differences and periodic BCs.

    The QG model is doubly periodic, so np.roll is used so that points near
    the region edges still get a correct centered-difference stencil.
    """
    # Last two axes are (y, x); axis=-1 -> x, axis=-2 -> y.
    dv_dx = (np.roll(v, -1, axis=-1) - np.roll(v, 1, axis=-1)) / (2.0 * dx_m)
    du_dy = (np.roll(u, -1, axis=-2) - np.roll(u, 1, axis=-2)) / (2.0 * dx_m)
    return dv_dx - du_dy


def make_movie(
    region_size_km: float = 100.0,
    x0_km: float = 400.0,
    y0_km: float = 400.0,
    dt_days: int = 30,
    duration_days: int = 365 * 5,
    out_path: str = 'relvor_movie.mp4',
    lev: int = 1,
    fps: int = 6,
):
    """Render a movie of relative vorticity over a square QG sub-region.

    Parameters
    ----------
    region_size_km : float
        Side length of the square region (km).
    x0_km, y0_km : float
        Lower-left corner of the region (km).
    dt_days : int
        Stride between rendered frames (days). The model output is saved
        daily, so a stride of 30 means every 30th saved frame.
    duration_days : int
        Total time spanned by the movie. The window ends on the last frame
        of the model output -- i.e. the movie starts `duration_days` before
        the end of the run.
    out_path : str
        Path to the output .mp4 file.
    lev : int
        Vertical level coordinate (dataset has lev=[1, 2]); 1 is the top.
    fps : int
        Frames per second for the rendered movie.
    """
    # Load the full QG model output via the Analysis/py loader.
    qg, _ = qg_utils.load_qg()

    # Pick the requested vertical level (label-based: lev=1 means top).
    qg_lev = qg.sel(lev=lev)

    # Build the time-index window: last `duration_days`, every `dt_days`.
    n_time = qg_lev.sizes['time']
    t_start = max(0, n_time - duration_days)
    time_idx = np.arange(t_start, n_time, dt_days)

    # Region indices from the physical x/y coordinates (stored in meters).
    x_m = qg.x.values
    y_m = qg.y.values
    x0_m, x1_m = x0_km * 1e3, (x0_km + region_size_km) * 1e3
    y0_m, y1_m = y0_km * 1e3, (y0_km + region_size_km) * 1e3
    ix = np.where((x_m >= x0_m) & (x_m < x1_m))[0]
    iy = np.where((y_m >= y0_m) & (y_m < y1_m))[0]
    if ix.size == 0 or iy.size == 0:
        raise ValueError(
            f'Empty region for x0={x0_km}km y0={y0_km}km size={region_size_km}km')

    # Load u, v over the FULL level for the selected frames so that the
    # periodic stencil is correct at every cell of the region.
    print(f'Loading {time_idx.size} frames '
          f'(time index {time_idx[0]}..{time_idx[-1]} step {dt_days})')
    u_all = qg_lev.u.isel(time=time_idx).values  # (n_frames, ny, nx)
    v_all = qg_lev.v.isel(time=time_idx).values

    # Vorticity for every frame, then slice down to the region.
    zeta_full = relative_vorticity(u_all, v_all)
    zeta = zeta_full[:, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]

    # Symmetric color limits from a robust percentile across all frames so
    # the diverging colormap stays centered at zero.
    vmax = float(np.nanpercentile(np.abs(zeta), 99))
    vmin = -vmax

    # Region-local axes in km, using cell-edge extents for imshow.
    x_lo = (x_m[ix[0]] - 0.5 * DX_M) / 1e3 - x0_km
    x_hi = (x_m[ix[-1]] + 0.5 * DX_M) / 1e3 - x0_km
    y_lo = (y_m[iy[0]] - 0.5 * DX_M) / 1e3 - y0_km
    y_hi = (y_m[iy[-1]] + 0.5 * DX_M) / 1e3 - y0_km

    # qg.time is in seconds since the start of the run.
    time_days = (qg.time.isel(time=time_idx).values / 86400.0).astype(int)

    # Build the figure.
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    im = ax.imshow(
        zeta[0],
        origin='lower',
        extent=[x_lo, x_hi, y_lo, y_hi],
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
    )
    ax.set_xlabel(f'x - {x0_km:g} km  [km]')
    ax.set_ylabel(f'y - {y0_km:g} km  [km]')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$\zeta = \partial_x v - \partial_y u$  [s$^{-1}$]')
    title = ax.set_title(f'Day {time_days[0]}')
    fig.tight_layout()

    def update(i):
        im.set_data(zeta[i])
        title.set_text(f'Day {time_days[i]}')
        return im, title

    anim = animation.FuncAnimation(
        fig, update, frames=zeta.shape[0],
        interval=1000.0 / fps, blit=False)

    # Write the .mp4 (requires ffmpeg).
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    print(f'Writing movie to {out_path}')
    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    print('Done.')


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region-size', type=float, default=100.0,
                   help='Region side length in km (default 100)')
    p.add_argument('--x0', type=float, default=400.0,
                   help='Region lower-left x in km (default 400)')
    p.add_argument('--y0', type=float, default=400.0,
                   help='Region lower-left y in km (default 400)')
    p.add_argument('--dt', type=int, default=30,
                   help='Frame stride in days (default 30)')
    p.add_argument('--duration', type=int, default=365 * 5,
                   help='Movie duration in days (default 1825 = 5 years)')
    p.add_argument('--lev', type=int, default=1,
                   help='Vertical level (1 = top, 2 = bottom; default 1)')
    p.add_argument('--fps', type=int, default=6, help='Frames per second')
    p.add_argument('--out', type=str, default='relvor_movie.mp4',
                   help='Output mp4 path')
    return p.parse_args()

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # 300km x 300km region at x0=500km, y0=300km
    if flg == 1:
        make_movie(
            region_size_km=300.0,
            x0_km=300.0,
            y0_km=300.0,
            dt_days=10,
            duration_days=365 * 5,
            out_path='relvor_movie_300km_300_300.mp4',
        )


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)

'''
    args = _parse_args()
    make_movie(
        region_size_km=args.region_size,
        x0_km=args.x0,
        y0_km=args.y0,
        dt_days=args.dt,
        duration_days=args.duration,
        out_path=args.out,
        lev=args.lev,
        fps=args.fps,
    )
'''