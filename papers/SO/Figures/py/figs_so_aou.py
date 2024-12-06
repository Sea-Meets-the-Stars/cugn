""" AOU figures for the paper 

"""

# imports
import os, sys

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d 

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator 

mpl.rcParams['font.family'] = 'stixgeneral'

from ocpy.utils import plotting

import seaborn as sns

from gsw import conversions, density
import gsw

from cugn import grid_utils
from cugn import defs as cugn_defs
from cugn import io as cugn_io
from cugn import annualcycle

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import so_analysis

from IPython import embed

lines = cugn_defs.lines
line_colors = cugn_defs.line_colors
line_cmaps = cugn_defs.line_cmaps

labels = dict(
    SA='Absolute Salinity (g/kg)',
    sigma0='Potential Density (kg/m$^3$)',
    CT='Conservative Temperature (deg C)',
    SO='Oxygen Saturation',
    AOU='Apparent Oxygen Utilization '+r'$(\mu$'+'mol/kg)',
    N='Buoyancy Frequency (cycles/hour)',
    DO='Dissolved Oxygen '+r'$(\mu$'+'mol/kg)',
    chla='Chl-a (mg/m'+r'$^3$'+')',
)
labels['doxy'] = labels['DO']

short_lbl = {'doxy': 'DO ('+r'$\mu$'+'mol/kg)', 
                 'T': 'T (deg C)',
                 'CT': 'T (deg C)',
                 'SA': 'SA (g/kg)',
                 'SO': 'SO',
                 'N': 'N (cycles/hr)',
                 'dist': 'Distance from shore (km)',
                 'chla': 'Chl-a (mg/m'+r'$^3$'+')'}


def gen_cb(img, lbl, csz = 17.):
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=csz)
    cbaxes.ax.tick_params(labelsize=csz)

def fig_joint_pdf_NAOU(line:str, max_depth:int=30):

    def gen_cb(img, lbl, csz = 17.):
        cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
        cbaxes.set_label(lbl, fontsize=csz)
        cbaxes.ax.tick_params(labelsize=csz)

    xvar = 'AOU'
    yvar = 'N'
    outfile = f'fig_jointPDF_{line}_AOU_N.png'

    # Load
    items = cugn_io.load_line(line)
    ds = items['ds']

    # PDF
    _, xedges, yedges, counts, grid_indices, _, _ = grid_utils.gen_grid(
        ds, axes=(xvar, yvar), stat='mean', variable='doxy', 
        max_depth=max_depth)

    # PDF
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    p_norm = np.sum(counts) * (dx * dy)
    consv_pdf = counts / p_norm
    #embed(header='764 of figs_so')

    fig = plt.figure(figsize=(12,10))
    plt.clf()
    ax = plt.gca()

    # #####################################################
    # PDF
    img = ax.pcolormesh(xedges, yedges, np.log10(consv_pdf.T), 
                            cmap='Blues')
    gen_cb(img, r'$\log_{10} \, p('+f'{xvar},{yvar})$',
           csz=19.)

    # ##########################################################
    tsz = 25.
    ax.text(0.05, 0.9, f'Line: {line}',
                transform=ax.transAxes,
                fontsize=tsz, ha='left', color='k')
    ax.text(0.05, 0.8, f'z <= {max_depth}m',
                transform=ax.transAxes,
                fontsize=tsz, ha='left', color='k')

    ax.set_xlabel(labels[xvar])
    ax.set_ylabel(labels[yvar])

    #ax.set_xlim(0.4, 1.6)
    ax.set_ylim(0., 22.)
    # Set x-axis interval to 0.5
    #ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # 
    fsz = 27.
    plotting.set_fontsize(ax, fsz)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_AOU_cdf(outfile:str='fig_AOU_cdf.png', 
                use_full:bool=False):

    # Figure
    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    for clr, line in zip(line_colors, lines):
        # Load
        items = cugn_io.load_up(line, use_full=use_full)#, skip_dist=True)
        grid_tbl = items[3]

        for ss, depth in enumerate([0,1]):
            ax = plt.subplot(gs[ss])
            cut_depth = grid_tbl.depth == depth
            grid_plt = grid_tbl[cut_depth]

            # Plot CDF
            if ss == 0:
                lbl = f'Line {line}'
            else:
                lbl = None
            sns.ecdfplot(x=grid_plt.AOU, ax=ax, label=lbl, color=clr)

            # Stats
            srt = np.argsort(grid_plt.AOU.values)
            AOUvals = grid_plt.AOU.values[srt]
            cdf = np.arange(len(grid_plt.AOU))/len(grid_plt.AOU)
            idx = np.argmin(np.abs(cdf-0.95))
            print(f'95% for {line} {(depth+1)*10}m: {grid_plt.AOU.values[srt][idx]}')

            # Percent satisfying the criterion
            high = grid_plt.AOU > cugn_defs.AOU_hyper
            print(f'Percent AOU > {cugn_defs.AOU_hyper} for Line={line} {10*(depth+1)}m: {100.*np.sum(high)/len(AOUvals):.1f}%')
            

    # Finish
    lsz = 17.
    for ss, depth in enumerate([0,1]):
        ax = plt.subplot(gs[ss])
        ax.axvline(1., color='black', linestyle='-')
        ax.axvline(cugn_defs.AOU_hyper, color='black', linestyle=':')

        #ax.set_xlim(0.5, 1.4)
        ax.set_xlabel(labels['AOU'])
        ax.set_ylabel('CDF')
                 #label=f'SO > {SO_cut}', log_scale=log_scale)
        ax.text(0.95, 0.05, f'z={(depth+1)*10}m',
                transform=ax.transAxes,
                fontsize=lsz, ha='right', color='k')
        plotting.set_fontsize(ax, lsz)

    ax = plt.subplot(gs[0])
    ax.legend(fontsize=15., loc='upper left')

    plt.tight_layout(pad=0.5)#, h_pad=0.1, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_extrema_cdfs(outfile:str='fig_N_AOU_cdfs.png', metric:str='N',
                     xyLine:tuple=(0.05, 0.90),
                     leg_loc:str='lower right'):

    # CDFs
    fig = plt.figure(figsize=(7,7))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    for ss, line in enumerate(cugn_defs.lines):
        # Load
        items = cugn_io.load_up(line, gextrem='highAOU')
        grid_extrem = items[0]
        ds = items[1]
        times = items[2]
        grid_tbl = items[3]

        cut_grid = (grid_tbl.depth <= 5) & np.isfinite(grid_tbl[metric])

        ctrl = grid_utils.grab_control_values(
            grid_extrem, grid_tbl[cut_grid], metric, boost=5)


        ax = plt.subplot(gs[ss])

        sns.ecdfplot(x=grid_extrem[metric], ax=ax, label='Extrema', 
                     color=cugn_defs.line_colors[ss])
        sns.ecdfplot(x=ctrl, ax=ax, label='Control', color='k', ls='--')


        # Finish
        #ax.axvline(1., color='black', linestyle='--')
        #ax.axvline(1.1, color='black', linestyle=':')
        lsz = 12.
        ax.legend(fontsize=lsz, loc=leg_loc)

        #ax.set_xlim(0.5, 1.4)
        ax.set_xlabel(labels[metric])
        ax.set_ylabel('CDF')
        ax.text(xyLine[0], xyLine[1], f'Line: {line}', 
                transform=ax.transAxes,
                fontsize=lsz, ha='left', color='k')
        plotting.set_fontsize(ax, 13)

        # Stats
        # Percentile of the extrema
        val = np.nanpercentile(grid_extrem[metric], (10,90))
        print(f'Line: {line} -- percentiles={val}')

    plt.tight_layout(pad=0.8)#, w_pad=2.0)#, w_pad=0.8)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)


    # Figure 1 -- Joint PDFs N vs. AOU
    if flg == 1:
        fig_joint_pdf_NAOU('90.0')

    # Figure 1 -- CDFs of AOU
    if flg == 2:
        fig_AOU_cdf()

    # Extrema CDFs
    if flg == 3:
        # N
        fig_extrema_cdfs()
        # Chla
        fig_extrema_cdfs('fig_chla_AOU_cdfs.png', metric='chla',
                         xyLine=(0.7, 0.4))
        # DO
        fig_extrema_cdfs('fig_doxy_AOU_cdfs.png', metric='doxy',
                         xyLine=(0.7, 0.4), leg_loc='upper left')

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg = 1 
    else:
        flg = sys.argv[1]

    main(flg)