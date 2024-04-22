""" Final figures for the paper 

Other figures are in figs_so.py

"""

# imports
from importlib import reload
import os
import xarray

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d 

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator 

mpl.rcParams['font.family'] = 'stixgeneral'


import seaborn as sns

import pandas

from cugn import grid_utils
from cugn import defs as cugn_defs
from cugn import io as cugn_io
from siosandbox import plot_utils


from IPython import embed

lines = cugn_defs.lines
line_colors = cugn_defs.line_colors
line_cmaps = cugn_defs.line_cmaps

labels = dict(
    SA='Absolute Salinity (g/kg)',
    sigma0='Potential Density (kg/m$^3$)',
    CT='Conservative Temperature (C)',
    DO='Dissolved Oxygen (umol/kg)',
)

def gen_cb(img, lbl, csz = 17.):
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=csz)
    cbaxes.ax.tick_params(labelsize=csz)


def fig_joint_pdfs(use_density:bool=False):


    if use_density:
        axes=('SA', 'sigma0')
        outfile = 'fig_paper_jointPDFs_density.png'
        lbl = r'$\log_{10} \, p(S_A,\sigma)$'
        ylbl = 'Potential Density (kg/m$^3$)'
        ypos = 0.9
    else:
        axes=('SA', 'CT')
        outfile = 'fig_paper_jointPDFs.png'
        lbl = r'$\log_{10} \, p(S_A,\theta)$'
        ylbl = 'Conservative Temperature (C)'
        ypos = 0.1


    fig = plt.figure(figsize=(12,10))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    all_ax = []
    for ss, cmap, line in zip(range(4), line_cmaps, lines):

        # Load
        items = cugn_io.load_line(line)
        ds = items['ds']

        # Oxygen
        mean_oxy, xedges, yedges, counts, grid_indices, _, _ = grid_utils.gen_grid(
            ds, axes=axes, stat='mean', variable='doxy')

        # PDF
        dx = xedges[1] - xedges[0]
        dy = yedges[1] - yedges[0]

        p_norm = np.sum(counts) * (dx * dy)
        consv_pdf = counts / p_norm

        # #####################################################
        # PDF
        ax_pdf = plt.subplot(gs[ss])
        img = ax_pdf.pcolormesh(xedges, yedges, np.log10(consv_pdf.T),
                                cmap=cmap)
        gen_cb(img, lbl)
        all_ax.append(ax_pdf)

    for ss, line in enumerate(lines):
        ax = all_ax[ss]
        fsz = 17.
        ax.set_xlabel('Absolute Salinity (g/kg)')                    
        ax.set_ylabel(ylbl)
        # Set x-axis interval to 0.5
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        # 
        plot_utils.set_fontsize(ax, fsz)
        ax.text(0.05, ypos, f'Line={line}',
                transform=ax.transAxes,
                fontsize=fsz, ha='left', color='k')
        # Grid lines
        ax.grid()
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# ##########################################################
def fig_mean_DO_SO(line, outfile:str=None):

    def gen_cb(img, lbl, csz = 17.):
        cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
        cbaxes.set_label(lbl, fontsize=csz)
        cbaxes.ax.tick_params(labelsize=csz)

    outfile = f'fig_mean_DO_SO_{line}.png'
    # Load
    items = cugn_io.load_line(line)
    ds = items['ds']

    # PDF
    axes=('SA', 'sigma0')
    mean_oxy, xedges, yedges, counts, grid_indices, _, _ = grid_utils.gen_grid(
        ds, axes=axes, stat='mean', variable='doxy')
    mean_SO, xedges, yedges, counts, grid_indices, _, _ = grid_utils.gen_grid(
        ds, axes=axes, stat='mean', variable='SO')

    # Figure
    fig = plt.figure(figsize=(12,10))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    # ##########################################################
    # DO
    axes = []
    for kk, mean, lbl, cmap in zip(np.arange(2), [mean_oxy, mean_SO], ['DO', 'SO'],
                                   ['Purples', 'jet']):
        for ss in range(2):
            ax= plt.subplot(gs[ss+kk*2])

            if ss == 0:
                vmin,vmax = None, None
                xmin,xmax = 32.5, 35.0
                ymin,ymax = None, None
            else:
                if lbl == 'DO':
                    vmin,vmax = 200., None
                else:
                    vmin,vmax = 0.5, None
                xmin,xmax = 32.5, 34.2
                ymin,ymax = 22.8, 25.5

            axes.append(ax)

            img = ax.pcolormesh(xedges, yedges, mean.T, cmap=cmap,
                                vmin=vmin, vmax=vmax)
            gen_cb(img, labels['DO'])

            # ##########################################################
            tsz = 19.
            ax.text(0.05, 0.9, f'Line={line}',
                        transform=ax.transAxes,
                        fontsize=tsz, ha='left', color='k')

            ax.set_xlabel(labels['SA'])
            ax.set_ylabel(labels['sigma0'])

            ax.set_xlim(xmin, xmax)
            if ymin is not None:
                ax.set_ylim(ymin, ymax)

    # Set x-axis interval to 0.5
    #ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # 
    fsz = 17.
    for ax in axes:
        plot_utils.set_fontsize(ax, fsz)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_SO_cdf(outfile:str):

    # Figure
    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    for clr, line in zip(line_colors, lines):
        # Load
        items = cugn_io.load_up(line)#, skip_dist=True)
        grid_extrem = items[0]
        ds = items[1]
        times = items[2]
        grid_tbl = items[3]

        for ss, depth in enumerate([0,1]):
            ax = plt.subplot(gs[ss])
            # Cut
            cut_depth = grid_tbl.depth == depth
            grid_plt = grid_tbl[cut_depth]

            # Plot CDF
            if ss == 0:
                lbl = f'Line {line}'
            else:
                lbl = None
            sns.ecdfplot(x=grid_plt.SO, ax=ax, label=lbl, color=clr)

            # Stats
            srt = np.argsort(grid_plt.SO.values)
            SOvals = grid_plt.SO.values[srt]
            cdf = np.arange(len(grid_plt.SO))/len(grid_plt.SO)
            idx = np.argmin(np.abs(cdf-0.95))
            print(f'95% for {line} {(depth+1)*10}m: {grid_plt.SO.values[srt][idx]}')

            # Percent satisfying the criterion
            high = grid_plt.SO > 1.1
            print(f'Percent SO > 1.1 for Line={line} {10*(depth+1)}m: {100.*np.sum(high)/len(SOvals):.1f}%')
            

    # Finish
    for ss, depth in enumerate([0,1]):
        ax = plt.subplot(gs[ss])
        ax.axvline(1., color='black', linestyle='--')
        ax.axvline(1.1, color='black', linestyle=':')

        ax.set_xlim(0.5, 1.4)
        ax.set_xlabel('Oxygen Saturation')
        ax.set_ylabel('CDF')
                 #label=f'SO > {SO_cut}', log_scale=log_scale)
        ax.text(0.95, 0.05, f'depth={(depth+1)*10}m',
                transform=ax.transAxes,
                fontsize=15, ha='right', color='k')
        plot_utils.set_fontsize(ax, 15)

    ax = plt.subplot(gs[0])
    ax.legend(fontsize=15., loc='upper left')

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_dist_doy(outfile:str, line:str, color:str,
                 gextrem:str='high',
                 show_legend:bool=False, 
                 clr_by_depth:bool=False):

    # Figure
    #sns.set()
    fig = plt.figure(figsize=(12,12))
    plt.clf()
    #ax = plt.gca()

    # Load
    items = cugn_io.load_up(line, gextrem=gextrem)
    grid_extrem = items[0]
    #ds = items[1]
    #times = items[2]
    #grid_tbl = items[3]

    jg = sns.jointplot(data=grid_extrem, x='dist', 
                    y='doy', color=color,
                    marginal_kws=dict(fill=False, color='black', 
                                        bins=100)) 


    # Scatter plot time
    markers = ['o','x','v','s','*']
    clrs = ['k', 'r','b','g','purple']
    jg.ax_joint.cla()

    for depth in range(5):
        if clr_by_depth:
            clr = clrs[depth]
        else:
            clr = color
        at_depth = grid_extrem.depth == depth
        # Scatter plot
        if show_legend:
            label=f'z={(depth+1)*10}m'
        else:
            label = None
        if depth != 1:
            fc = 'none'
        else:
            fc = clr
        jg.ax_joint.scatter(
            grid_extrem[at_depth].dist, 
            grid_extrem[at_depth].doy, 
            marker=markers[depth], label=label, facecolors=fc,
            edgecolors=clr)#, s=50.)
    
    # Axes                                 
    jg.ax_joint.set_ylabel('DOY')
    jg.ax_joint.set_xlabel('Distance from shore (km)')
    jg.ax_joint.set_ylim(0., 365.)

    xmin = -20. if line != '80' else -60.
    xmax = max(500., grid_extrem.dist.max())
    jg.ax_joint.set_xlim(xmin, xmax)

    fsz = 14.
    jg.ax_joint.text(0.95, 0.95, f'Line {line}',
                transform=jg.ax_joint.transAxes,
                fontsize=fsz, ha='right', color='k')
    plot_utils.set_fontsize(jg.ax_joint, 14)
    if show_legend:
        jg.ax_joint.legend(fontsize=13., loc='lower right')

    
    #gs.tight_layout(fig)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# ######################################################
def fig_SO_vs_N_zoom():
    #def fig_joint_pdf(line:str, xvar:str, yvar:str):

    default_bins = dict(SA=np.linspace(32.1, 34.8, 50),
                sigma0=np.linspace(22.8, 27.2, 50),
                SO=np.linspace(1.05, 1.5, 50),
                z=np.linspace(0., 500, 50),
                N=np.linspace(1., 25, 50),
                CT=np.linspace(4, 22.5, 50))

    line = '90'
    xvar = 'SO'
    yvar = 'N'


    outfile = f'fig_jointPDF_{line}_{xvar}_{yvar}_zoom.png'

    # Load
    items = cugn_io.load_line(line)
    ds = items['ds']

    # PDF
    _, xedges, yedges, counts, grid_indices, _, _ = grid_utils.gen_grid(
        ds, axes=(xvar, yvar), stat='mean', variable='doxy', bins=default_bins
        )

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
    show_log = False

    if show_log:
    #    img = ax.pcolormesh(xedges, yedges, np.log10(consv_pdf.T), 
        img = ax.pcolormesh(xedges, yedges, np.log10(counts.T), 
                            cmap='autumn')
        gen_cb(img, r'$\log_{10} \, counts('+f'{xvar},{yvar})$')
    else:
    #    img = ax.pcolormesh(xedges, yedges, consv_pdf.T, 
        img = ax.pcolormesh(xedges, yedges, counts.T, 
                            cmap='jet')#, vmin=0.01)    
        gen_cb(img, r'$counts('+f'{xvar},{yvar})$')


    # ##########################################################
    tsz = 19.
    ax.text(0.05, 0.9, f'Line={line}',
                transform=ax.transAxes,
                fontsize=tsz, ha='left', color='k')

    fsz = 17.
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    # Set x-axis interval to 0.5
    #ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # 
    plot_utils.set_fontsize(ax, fsz)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)


    # Figure 1 -- Joint PDFs
    if flg & (2**0):
        fig_joint_pdfs()
        fig_joint_pdfs(use_density=True)

    # Figure 2 -- average DO, SO 
    if flg & (2**1):
        line = '90'
        fig_mean_DO_SO(line)

    # Figure 3 -- SO CDFs
    if flg & (2**2):
        fig_SO_cdf('fig_SO_cdf.png')

    # Figure 3 -- DOY vs Offshore distance
    if flg & (2**3):
        for line, clr in zip(lines, line_colors):
            # Skip for now
            #if line == '56':
            #    continue
            if line == '56.0':
                show_legend = True
            else:
                show_legend = False
            # High
            #fig_dist_doy(f'fig_dist_doy_{line}.png', 
            #             line, clr, show_legend=show_legend,
            #             clr_by_depth=True)
            # Low
            fig_dist_doy(f'fig_dist_doy_low_{line}.png', 
                         line, clr, 
                         gextrem='low',
                         show_legend=show_legend,
                         clr_by_depth=True)

    # Figure 4 -- SO vs. N
    if flg & (2**4):
        fig_SO_vs_N_zoom()

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Joint PDFs of all 4 lines
        #flg += 2 ** 1  # 2 -- 
        #flg += 2 ** 2  # 4 -- SO CDF
        #flg += 2 ** 3  # 8 -- DOY vs. offshore distance
        #flg += 2 ** 4  # 16 -- SO vs N zoom
    else:
        flg = sys.argv[1]

    main(flg)