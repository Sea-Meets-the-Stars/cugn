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
from matplotlib.ticker import MultipleLocator 

mpl.rcParams['font.family'] = 'stixgeneral'


import seaborn as sns

from gsw import conversions, density
import gsw

from cugn import grid_utils
from cugn import defs as cugn_defs
from cugn import io as cugn_io
from siosandbox import plot_utils
from cugn import annualcycle


from IPython import embed

lines = cugn_defs.lines
line_colors = cugn_defs.line_colors
line_cmaps = cugn_defs.line_cmaps

labels = dict(
    SA='Absolute Salinity (g/kg)',
    sigma0='Potential Density (kg/m$^3$)',
    CT='Conservative Temperature (deg C)',
    SO='Oxygen Saturation',
    N='Buoyancy (cycles/hour)',
    DO='Dissolved Oxygen '+r'$(\mu$'+'mol/kg)',
)
labels['doxy'] = labels['DO']

def gen_cb(img, lbl, csz = 17.):
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=csz)
    cbaxes.ax.tick_params(labelsize=csz)


def fig_joint_pdfs(use_density:bool=False, use_DO:bool=False):


    if use_density:
        axes=('SA', 'sigma0')
        outfile = 'fig_paper_jointPDFs_density.png'
        lbl = r'$\log_{10} \, p(S_A,\sigma)$'
        xlbl = 'Absolute Salinity (g/kg)'
        ylbl = 'Potential Density (kg/m$^3$)'
        ypos = 0.9
    elif use_DO:
        #axes=('SA', 'doxy')
        #xlbl = 'Absolute Salinity (g/kg)'
        axes=('sigma0', 'doxy')
        xlbl = labels['sigma0']
        #axes=('CT', 'doxy')
        #xlbl = 'Conservative Temperature (C)'
        outfile = 'fig_paper_jointPDFs_DO.png'
        lbl = r'$\log_{10} \, p(\sigma,DO)$'
        ylbl = labels['DO']
        ypos = 0.1
    else:
        axes=('SA', 'CT')
        outfile = 'fig_paper_jointPDFs.png'
        lbl = r'$\log_{10} \, p(S_A,\theta)$'
        xlbl = 'Absolute Salinity (g/kg)'
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
        ax.set_xlabel(xlbl)
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


def fig_SO_cdf(outfile:str, use_full:bool=False):

    # Figure
    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    for clr, line in zip(line_colors, lines):
        # Load
        items = cugn_io.load_up(line, use_full=use_full)#, skip_dist=True)
        grid_extrem = items[0]
        ds = items[1]
        times = items[2]
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
    lsz = 17.
    for ss, depth in enumerate([0,1]):
        ax = plt.subplot(gs[ss])
        ax.axvline(1., color='black', linestyle='-')
        ax.axvline(1.1, color='black', linestyle=':')

        ax.set_xlim(0.5, 1.4)
        ax.set_xlabel('Oxygen Saturation')
        ax.set_ylabel('CDF')
                 #label=f'SO > {SO_cut}', log_scale=log_scale)
        ax.text(0.95, 0.05, f'z={(depth+1)*10}m',
                transform=ax.transAxes,
                fontsize=lsz, ha='right', color='k')
        plot_utils.set_fontsize(ax, lsz)

    ax = plt.subplot(gs[0])
    ax.legend(fontsize=15., loc='upper left')

    plt.tight_layout(pad=0.5)#, h_pad=0.1, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_dist_doy(outfile:str, line:str, color:str,
                 gextrem:str='hi_noperc',
                 show_legend:bool=False, 
                 clr_by_depth:bool=False):

    # Figure
    #sns.set()
    fig = plt.figure(figsize=(12,12))
    plt.clf()
    #ax = plt.gca()

    # Load
    items = cugn_io.load_up(line, gextrem=gextrem, use_full=True)
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
            marker=markers[depth], label=label, 
            facecolors=fc,
            s=15., 
            edgecolors=clr)#, s=50.)
        
    # Add clusters
    uni_cluster = np.unique(grid_extrem.cluster)
    for cluster in uni_cluster:
        if cluster < 0:
            continue
        #
        in_cluster = grid_extrem.cluster == cluster
        med_idx = np.where(in_cluster)[0][np.sum(in_cluster)//2]
        jg.ax_joint.scatter(
            grid_extrem.dist.values[med_idx], 
            grid_extrem.doy.values[med_idx], 
            marker='s', 
            facecolors='none',
            s=80., 
            edgecolors='cyan')
    
    # Axes                                 
    jg.ax_joint.set_ylabel('DOY')
    jg.ax_joint.set_xlabel('Distance from shore (km)')
    jg.ax_joint.set_ylim(0., 365.)

    xmin = -20. if line != '80' else -60.
    xmax = max(500., grid_extrem.dist.max())
    jg.ax_joint.set_xlim(xmin, xmax)

    fsz = 17.
    jg.ax_joint.text(0.95, 0.95, f'Line {line}',
                transform=jg.ax_joint.transAxes,
                fontsize=fsz, ha='right', color='k')
    plot_utils.set_fontsize(jg.ax_joint, 19)
    if show_legend:
        jg.ax_joint.legend(fontsize=13., loc='lower right')

    
    #plt.tight_layout(h_pad=0.3, w_pad=10.3)
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

    # Stats
    not_winter = (grid_extrem.doy > 50) & (grid_extrem.doy < 300)
    print(f'Percent of profiles not in winter [50-300]: {100.*np.sum(not_winter)/len(grid_extrem):.1f}%')

def fig_dist_doy_low(outfile:str='fig_dist_doy_low.png', 
                 gextrem:str='low_noperc'):


    # Start the figure
    fig = plt.figure(figsize=(10,12))
    plt.clf()
    gs = gridspec.GridSpec(4,2)

    # DOY

    # Load
    for ss, line in enumerate(cugn_defs.lines):
        items = cugn_io.load_up(line, gextrem=gextrem)
        grid_extrem = items[0]

        # DOY
        ax_doy = plt.subplot(gs[ss, 0])
        ax_doy.hist(grid_extrem.doy, bins=20,# histtype='step',
                    color=cugn_defs.line_colors[ss], label=f'Line {line}', lw=2)
        #
        ax_doy.grid(True)
        ax_doy.set_xlim(0., 366.)
        if ss < 3:
            ax_doy.set_xticklabels([])
        else:
            ax_doy.set_xlabel('DOY')
        plot_utils.set_fontsize(ax_doy, 17)
        ax_doy.set_ylabel('Count')

        # Stats
        in_spring = (grid_extrem.doy > 50) & (grid_extrem.doy < 200)
        print(f'Line={line}: Percent of profiles with DOY=50-200: {100.*np.sum(in_spring)/len(grid_extrem):.1f}%')

        # ##########################################################
        # Distance offshore
        ax_doff = plt.subplot(gs[ss, 1])
        ax_doff.hist(grid_extrem.dist, bins=20,# histtype='step',
                    color=cugn_defs.line_colors[ss], label=f'Line {line}', lw=2)
        #
        ax_doff.grid(True)
        ax_doff.set_xlim(-50., 250.)
        if ss < 3:
            ax_doff.set_xticklabels([])
        else:
            ax_doff.set_xlabel('Distance Offshore (km)')
        plot_utils.set_fontsize(ax_doff, 17)
        ax_doff.set_ylabel('Count')

        # Stats
        in_shore = grid_extrem.dist < 100.
        print(f'Line={line}: Percent of profiles within 100km of shore: {100.*np.sum(in_shore)/len(grid_extrem):.1f}%')

        # Text label
        ax_doff.text(0.8, 0.9, f'Line {line}',
                transform=ax_doff.transAxes,
                fontsize=17., ha='right', color='k')
                    


    #plt.tight_layout(h_pad=0.3, w_pad=10.3)
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.3)
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

    cmap = 'Blues'
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
                            cmap=cmap)#, vmin=0.01)    
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



def fig_joint_line90(outfile:str='fig_joint_TDO_line90.png', 
                     line:str='90.0', metric='CT',
                     xmetric:str='doxy',
                     max_depth:int=30):
    # Figure
    #sns.set()
    fig = plt.figure(figsize=(12,12))
    plt.clf()

    if metric == 'N':
        cmap = 'Blues'
    elif metric == 'chla':
        cmap = 'Greens'
    elif metric == 'CT':
        cmap = 'Oranges'
    elif metric == 'SA':
        cmap = 'Greys'
    
    # Load
    items = cugn_io.load_up(line)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]

    # Cut on depth
    grid_tbl = grid_tbl[grid_tbl.depth <= (max_depth//10 - 1)]

    # SO calculation

    # Canonical values
    z=20. # m
    SA = 33.7  
    DO = 260.

    lat = np.nanmedian(ds.lat.data)
    lon = np.nanmedian(ds.lon.data)
    p = conversions.p_from_z(-z, lat)

    # Interpolators
    CTs = np.linspace(12., 22., 100)
    OCs = gsw.O2sol(SA, CTs, p, lon, lat)

    jg = sns.jointplot(data=grid_tbl, x=xmetric,
                    y=metric,
                    kind='hex', bins='log', # gridsize=250, #xscale='log',
                    # mincnt=1,
                    cmap=cmap,
                    marginal_kws=dict(fill=False, color='black', 
                                        bins=100)) 

    # Axes                                 
    plot_utils.set_fontsize(jg.ax_joint, 14)

    # SO
    if metric == 'CT':
        jg.ax_joint.plot(OCs, CTs, 'k:', lw=1)


    # Labels
    #lbl = r'$z \le $'+f'{max_depth}m'
    jg.ax_joint.text(0.95, 0.05, f'z <= {max_depth}m',
                transform=jg.ax_joint.transAxes,
                fontsize=14., ha='right', color='k')
    jg.ax_joint.text(0.05, 0.95, f'Line: {line}',
                transform=jg.ax_joint.transAxes,
                fontsize=14., ha='left', color='k')
    jg.ax_joint.set_xlabel(labels['DO'])
    jg.ax_joint.set_ylabel(labels[metric])

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_joint_pdf_NSO(line:str, max_depth:int=30):

    def gen_cb(img, lbl, csz = 17.):
        cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
        cbaxes.set_label(lbl, fontsize=csz)
        cbaxes.ax.tick_params(labelsize=csz)

    xvar = 'SO'
    yvar = 'N'
    outfile = f'fig_jointPDF_{line}_SO_N.png'

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

    ax.set_xlim(0.4, 1.6)
    ax.set_ylim(0., 22.)
    # Set x-axis interval to 0.5
    #ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # 
    fsz = 27.
    plot_utils.set_fontsize(ax, fsz)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_extrema_cdfs(outfile:str='fig_N_cdfs.png', metric:str='N',
                     xyLine:tuple=(0.05, 0.90),
                     leg_loc:str='lower right'):

    # CDFs
    fig = plt.figure(figsize=(7,7))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    for ss, line in enumerate(cugn_defs.lines):
        # Load
        items = cugn_io.load_up(line)
        grid_extrem = items[0]
        ds = items[1]
        times = items[2]
        grid_tbl = items[3]

        cut_grid = (grid_tbl.depth <= 5) & np.isfinite(grid_tbl[metric])

        ctrl = grid_utils.grab_control_values(grid_extrem, grid_tbl[cut_grid], metric, boost=5)


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
        plot_utils.set_fontsize(ax, 13)

        # Stats
        # Percentile of the extrema
        val = np.nanpercentile(grid_extrem[metric], (10,90))
        print(f'Line: {line} -- percentiles={val}')

    plt.tight_layout(pad=0.8)#, w_pad=2.0)#, w_pad=0.8)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_annual(outfile:str, line:str, metric='N',
                 max_depth:int=20):

    # Figure
    #sns.set()
    fig = plt.figure(figsize=(12,12))
    plt.clf()

    if metric == 'N':
        cmap = 'Blues'
    elif metric == 'chla':
        cmap = 'Greens'
    elif metric == 'T':
        cmap = 'Oranges'
        ann_var = 't'
        ylbl = 'Temperature Anomaly (deg C)'
    elif metric == 'SA':
        cmap = 'Greys'
    

    # Load
    items = cugn_io.load_up(line, use_full=True,
                            gextrem='hi_noperc')
    low_items = cugn_io.load_up(line, use_full=True,
                            gextrem='low_noperc')
    # Unpack
    low_extrem = low_items[0]
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]

    # Cut on depth
    grid_tbl = grid_tbl[grid_tbl.depth <= (max_depth//10 - 1)]

    # Annual T
    annual = annualcycle.calc_for_grid(grid_tbl, line, ann_var)
    grid_tbl[f'D{metric}'] = grid_tbl[metric] - annual

    jg = sns.jointplot(data=grid_tbl, x='doxy', 
                    y=f'D{metric}',
                    kind='hex', bins='log', # gridsize=250, #xscale='log',
                    # mincnt=1,
                    cmap=cmap,
                    marginal_kws=dict(fill=False, color='orange', 
                                        bins=100)) 

    # Axes                                 
    jg.ax_joint.set_ylabel(ylbl)
    jg.ax_joint.set_xlabel(labels['doxy'])
    plot_utils.set_fontsize(jg.ax_joint, 14)

    # Extrema
    ex_clr = 'gray'
    annual = annualcycle.calc_for_grid(grid_extrem, line, ann_var)
    grid_extrem[f'D{metric}'] = grid_extrem[metric] - annual

    jg.ax_joint.plot(grid_extrem.doxy, 
                     grid_extrem[f'D{metric}'],  'o',
                     color=ex_clr, ms=0.5)

    # Low extreme
    annual = annualcycle.calc_for_grid(low_extrem, line, ann_var)
    low_extrem[f'D{metric}'] = low_extrem[metric] - annual
    jg.ax_joint.plot(low_extrem.doxy, 
                     low_extrem[f'D{metric}'],  'x',
                     color='blue', ms=1)

    #jg.ax_joint.text(0.95, 0.05, r'$z \le $'+f'{max_depth}m',
    #            transform=jg.ax_joint.transAxes,
    #            fontsize=14., ha='right', color='k')
    # Label
    jg.ax_joint.text(0.05, 0.95, f'Line: {line}',
                transform=jg.ax_joint.transAxes,
                fontsize=14., ha='left', color='k')
    jg.ax_joint.text(0.05, 0.88, f'z <= {max_depth}m',
                transform=jg.ax_joint.transAxes,
                fontsize=14., ha='left', color='k')

    # Add another histogram?
    #jg.ax_marg_y.cla()
    jg.ax_marg_y.hist(grid_extrem[f'D{metric}'], color=ex_clr, #alpha=0.5, 
                      bins=20, fill=False, edgecolor=ex_clr,
                      range=(-5., 5.), orientation='horizontal')

    # Label
    
    #gs.tight_layout(fig)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)


    # Figure 1 -- Joint PDFs
    if flg & (2**0):
        #fig_joint_pdfs()
        #fig_joint_pdfs(use_density=True)
        fig_joint_pdfs(use_DO=True)

    # Joint PDF: T, DO on Line 90
    if flg & (2**1):
        fig_joint_line90()

    # Figure 3  Joint PDF: T, DO on Line 90
    if flg & (2**2):
        line = '90'
        fig_joint_pdf_NSO(line)

    # Figure 4 -- SO CDFs
    if flg & (2**3):
        fig_SO_cdf('fig_SO_cdf.png', use_full=True)


    # Figure 5 -- DOY vs Offshore distance
    if flg & (2**4):
        for line, clr in zip(lines, line_colors):
            # Skip for now
            #if line == '56':
            #    continue
            if line == '56.0':
                show_legend = True
            else:
                show_legend = False
            # High
            fig_dist_doy(f'fig_dist_doy_{line}.png', 
                         line, clr, show_legend=show_legend,
                         clr_by_depth=True)
            # Low
            #fig_dist_doy(f'fig_dist_doy_low_{line}.png', 
            #             line, clr, 
            #             gextrem='low_noperc',
            #             show_legend=show_legend,
            #             clr_by_depth=True)

    # Figure 4 -- SO vs. N
    if flg & (2**5):
        fig_SO_vs_N_zoom()

    # Figure 9 -- doy, distance for the low extrema
    if flg & (2**8):
        fig_dist_doy_low()


    # Figure 2 -- average DO, SO 
    if flg & (2**10):
        line = '90'
        fig_mean_DO_SO(line)

    # Figure 2 -- T vs. DO
    if flg & (2**11):
        line = '90'
        fig_mean_DO_SO(line)

    # Extrema CDFs
    if flg & (2**18):
        # N
        #fig_extrema_cdfs()
        # Chla
        #fig_extrema_cdfs('fig_chla_cdfs.png', metric='chla',
        #                 xyLine=(0.7, 0.4))
        # DO
        fig_extrema_cdfs('fig_doxy_cdfs.png', metric='doxy',
                         xyLine=(0.7, 0.4), leg_loc='upper left')

    # Annual cycle
    if flg & (2**19):
        fig_annual('fig_annual_TDO.png', line='90.0', metric='T')

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Joint PDFs of all 4 lines
        #flg += 2 ** 1  # 2 -- Joint PDF, T vs DO
        #flg += 2 ** 2  # 4 -- Joint PDF, N vs SO
        #flg += 2 ** 3  # 8 -- SO CDFs
        #flg += 2 ** 4  # 16 -- Figure 5: DOY vs. offshore distance

        flg += 2 ** 8  # 256 -- Figure 9: DOY vs. offshore distance for low

        #flg += 2 ** 11  
        #flg += 2 ** 12  # Low histograms

        #flg += 2 ** 18  # # Extreme CDFs
        #flg += 2 ** 19  # T anomaly vs. DO
    else:
        flg = sys.argv[1]

    main(flg)