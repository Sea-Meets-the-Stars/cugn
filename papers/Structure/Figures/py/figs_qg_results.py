""" QG Results figures for the energy-cascade paper (Section 4c).

Two figures:
  fig_qg_time_evolution  -- convergence of the dense-region S1, S3 with
                            averaging time (1..1825 days).
  fig_qg_sampling        -- Eulerian (in-box truth) vs glider vs drifter
                            centered S3, at two QG start times, showing the
                            realization-dependence of sparse sampling.

Run:  python figs_qg_results.py <flag>      (flag 1 = time evo, 2 = sampling)
"""
import os
import sys
import json

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.append(os.path.join('..', '..', 'Analysis', 'py'))
import qg_utils

OUTPUT = os.path.join('..', '..', 'Analysis', 'Output')


def _set_fontsize(ax, sz):
    """Set font size on all tick labels / axis labels of an axis."""
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(sz)


def _centered_S3(S1, S2, S3):
    """Centered (cumulant) third moment: S3 - 3 S1 S2 + 2 S1^3."""
    return S3 - 3 * S1 * S2 + 2 * S1**3


# #####################################################################
def fig_qg_time_evolution(x0=300, y0=400,
                          outfile='fig_qg_time_evolution.png'):
    """Convergence of the dense-region S1 and centered S3 with averaging time.

    The region SF (computed over all grid pairs inside a 100 km box) is
    averaged over the first N days for increasing N, and compared with the
    exact full-domain structure functions.
    """
    # Full-domain truth (5-yr mean)
    qg, mSF = qg_utils.load_qg(use_SFduL=True)
    full = qg_utils.calc_dus(qg, mSF)
    r_full = full['rr1'] * 1e-3          # km
    s1_full = full['dull_mn'].values
    s3_full = full['du3_mn'].values

    # Dense 100 km region, per-day SF
    reg = xr.load_dataset(os.path.join(
        OUTPUT, f'SF_region_x{x0}_y{y0}_5years.nc'))
    r_reg = reg.dr.mean('time').values * 1e-3   # km

    windows = [30, 90, 180, 365, 1825]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(windows)))

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2)

    ax1 = plt.subplot(gs[0])   # S1
    ax3 = plt.subplot(gs[1])   # centered S3

    for nday, clr in zip(windows, colors):
        sub = reg.isel(time=slice(0, nday))
        S1 = sub.ulls.mean('time').values
        S2 = sub.du2.mean('time').values
        S3 = sub.du3.mean('time').values
        S3c = _centered_S3(S1, S2, S3)
        lbl = f'{nday} d' if nday < 365 else (
            '1 yr' if nday == 365 else '5 yr')
        ax1.semilogx(r_reg, S1 * 1e3, '-', color=clr, lw=1.5, label=lbl)
        ax3.semilogx(r_reg, S3c, '-', color=clr, lw=1.5, label=lbl)

    # Full-domain reference
    ax1.semilogx(r_full, s1_full * 1e3, 'k--', lw=1.2, label='Full domain')
    ax3.semilogx(r_full, s3_full, 'k--', lw=1.2, label='Full domain')

    ax1.axhline(0., color='gray', ls=':', lw=0.8)
    ax3.axhline(0., color='gray', ls=':', lw=0.8)

    ax1.set_xlabel(r'$r$ [km]')
    ax1.set_ylabel(r'$\langle \delta u_L \rangle \;\; 10^{-3}$ [m s$^{-1}$]')
    ax3.set_xlabel(r'$r$ [km]')
    ax3.set_ylabel(r'$\langle \delta u_L^3 \rangle_{\rm c}$  [m$^3$ s$^{-3}$]')

    ax1.set_xlim(4, 100)
    ax1.set_ylim(-18, 3)
    ax3.set_xlim(4, 100)
    ax3.legend(fontsize=8, loc='upper left', ncol=2)

    for ax in (ax1, ax3):
        _set_fontsize(ax, 12)
        ax.grid(which='major', lw=0.8, alpha=0.6)
        ax.grid(which='minor', lw=0.5, alpha=0.25)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f'Saved: {outfile}')


# #####################################################################
def fig_qg_sampling(x0=450, y0=450, nd=100, dx=100, rmax=55.,
                    outfile='fig_qg_sampling.png'):
    """Eulerian vs glider vs drifter centered S3 at two QG start times.

    Shows that sparse (glider, drifter) sampling over 100 days recovers the
    true (positive) cascade unreliably -- both the in-box truth and the
    recovered estimate vary between flow realizations -- while the dense
    Eulerian estimate is robustly positive.  Panels are scaled independently
    because the in-box truth itself differs between the two windows.
    """
    start_times = [5001, 6001]

    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2)

    for ip, ts in enumerate(start_times):
        ax = plt.subplot(gs[ip])

        tag = f'ts{ts}_nd{nd}_x{x0}_y{y0}_dx{dx}'

        # Eulerian in-box truth (.nc), time-averaged
        eu = xr.load_dataset(os.path.join(OUTPUT, f'qg_eulerian_SF_{tag}.nc'))
        r_eu = eu.dr.mean('time').values * 1e-3
        S1 = eu.ulls.mean('time').values
        S2 = eu.du2.mean('time').values
        S3 = eu.du3.mean('time').values
        S3c_eu = _centered_S3(S1, S2, S3)
        m = r_eu <= rmax
        ax.semilogx(r_eu[m], S3c_eu[m] * 1e5, 'k-', lw=1.8,
                    label='Eulerian (truth)', zorder=1)

        # Sparse samplings (JSON)
        styles = {'glider': ('o', 'C0', 'Glider (10)'),
                  'drifter': ('s', 'C3', 'Drifter (121)')}
        for kind, (mk, clr, lbl) in styles.items():
            with open(os.path.join(OUTPUT, f'qg_{kind}_SF_{tag}.json')) as f:
                d = json.load(f)
            r = np.array(d['r'])
            S1 = np.array(d['S1_duL'])
            S2 = np.array(d['S2_duL**2'])
            S3 = np.array(d['S3_duLduLduL'])
            err = np.array(d['err_S3_duLduLduL'])
            S3c = _centered_S3(S1, S2, S3)
            m = r <= rmax
            ax.errorbar(r[m], S3c[m] * 1e5, yerr=err[m] * 1e5, fmt=mk,
                        color=clr, ms=4, capsize=2, lw=1, label=lbl, zorder=3,
                        markerfacecolor='none' if kind == 'drifter' else clr)

        ax.axhline(0., color='gray', ls='--', lw=0.9)
        ax.set_xlabel(r'$r$ [km]')
        if ip == 0:
            ax.set_ylabel(r'$\langle \delta u_L^3 \rangle_{\rm c}$  '
                          r'$10^{-5}$ [m$^3$ s$^{-3}$]')
        ax.set_xlim(4, rmax)
        ax.set_title(f'Realization {"AB"[ip]}  (start $t={ts}$ d)',
                     fontsize=12)
        ax.legend(fontsize=8, loc='upper left')
        _set_fontsize(ax, 12)
        ax.grid(which='major', lw=0.8, alpha=0.6)
        ax.grid(which='minor', lw=0.5, alpha=0.25)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f'Saved: {outfile}')


if __name__ == '__main__':
    flg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if flg in (0, 1):
        fig_qg_time_evolution()
    if flg in (0, 2):
        fig_qg_sampling()
