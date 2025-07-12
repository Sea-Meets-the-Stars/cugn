""" Final figures for the paper 

Other figures are in figs_so.py
  and figs_so_aou.py

"""

# imports
import os, sys

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d 

import pandas

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator 
from matplotlib.patches import Ellipse
import matplotlib.dates as mdates
import matplotlib.image as mpimg

import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

mpl.rcParams['font.family'] = 'stixgeneral'

from ocpy.utils import plotting

import seaborn as sns

from gsw import conversions, density
import gsw

from cugn import grid_utils
from cugn import defs as cugn_defs
from cugn import io as cugn_io
from cugn import annualcycle
from cugn import utils as cugn_utils
from cugn import clusters

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import so_analysis

from IPython import embed

tformM = ccrs.Mollweide()
tformP = ccrs.PlateCarree()  

lines = cugn_defs.lines
line_colors = cugn_defs.line_colors
line_cmaps = cugn_defs.line_cmaps

labels = dict(
    SA='Absolute Salinity (g/kg)',
    sigma0='Potential Density (kg/m$^3$)',
    CT='Conservative Temperature (deg C)',
    SO='Oxygen Saturation',
    N='Buoyancy Frequency (cycles/hour)',
    u='Eastward Velocity (m/s)',
    v='Northward Velocity (m/s)',
    vel='Total Velocity (m/s)',
    cuti='CUTI',
    beuti='BEUTI',
    dsigma0=r'$\sigma_0 - \sigma_0(z=0)$',
    dsigma=r'$\sigma_0 - \sigma_0(z=0)$',
    MLD='Mixed Layer Depth (m)',
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


def fig_cugn(outfile:str='fig_cugn.png', debug:bool=False,
             use_density:bool=False, use_DO:bool=False):

    # Labels
    lon_lbl = [-125.9, -125.5, -123.5, -121.8]
    lat_lbl = [36.3, 34.55, 32.0, 31.0]

    # Start the figure
    fig = plt.figure(figsize=(7,6))
    plt.clf()
    ax = plt.subplot(projection=tformP)

    for ss, cmap, line in zip(range(4), line_cmaps, lines):
        if ss > 0 and debug:
            continue
        # Load
        items = cugn_io.load_up(line, use_full=True, kludge_MLDN=False)
        #grid_extrem = items[0]
        #ds = items[1]
        #times = items[2]
        grid_tbl = items[3]

        #ds = items['ds']
        #grid_tbl = items['grid_tbl']

        # Trim (remove this)
        dist, offset = cugn_utils.calc_dist_offset(
            line, grid_tbl.lon.values, grid_tbl.lat.values)
        grid_tbl['dist'] = dist
        ok_off = np.abs(offset) < 90.
        if np.any(~ok_off):
            embed(header='100 of figs')
            raise ValueError("Bad offset")

        grid_tbl = grid_tbl[ok_off]

        #embed(header='83 of figs')

        img = plt.scatter(x=grid_tbl.lon.values,
            y=grid_tbl.lat.values, color=line_colors[ss],# cmap=cmap,
                #vmin=0.,
                #vmax=vmax, 
            s=0.5,
            transform=tformP, label=f'Line {line}')

        # Add a star at each start point
        lons, lats = cugn_utils.line_endpoints(line)
        plt.scatter(x=lons, y=lats,
            color='k', transform=tformP, s=50.0,
            marker='*', zorder=10, facecolors='none')

        # Color bar
        #cbaxes = plt.colorbar(img, pad=0., fraction=0.030, orientation='horizontal') #location='left')
        #cbaxes.set_label(param, fontsize=17.)
        #cbaxes.ax.tick_params(labelsize=15)

        # Label
        lat_off = 0.2

        ax.text(lon_lbl[ss], lat_lbl[ss],
                f'Line {line}', fontsize=15, 
                ha='left', va='bottom', 
                transform=ccrs.PlateCarree(),
                color=line_colors[ss])
    # Coast lines
    ax.coastlines(zorder=10)
    ax.add_feature(cartopy.feature.LAND, 
            facecolor='lightgray', edgecolor='black')
        #ax.set_global()

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, 
        color='black', alpha=0.5, linestyle=':', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right=False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black', 'weight': 'bold'}

    # Define cities
    cities = [
        ('Los Angeles', -118.2437, 34.0522),
        ('San Francisco', -122.4194, 37.7749),
        ('Santa Barbara', -119.6982, 34.4208),
        ('San Diego', -117.1611, 32.7157),
        ('Santa Cruz', -122.0308, 36.9741),
        #('Sacramento', -121.4944, 38.5816)
    ]

    # Add cities to the map
    for city, lon, lat in cities:
        ax.plot(lon, lat, 'ko', markersize=5, transform=ccrs.PlateCarree())
        if city == 'San Diego':
            lat -= 0.45
            lon += 0.05
        elif city == 'San Francisco':
            lat -= 0.2
            lon += 0.25
        ax.text(lon, lat, city, fontsize=15, ha='left', va='bottom', transform=ccrs.PlateCarree())


    # Label the axes
    #ax.set_title('UG2', fontsize=17.)
    #ax.set_extent([-100, -30, -70, 50])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    #lon_min, lon_max, lat_min, lat_max = -100, 30, -70, 50
    #ax.set_extent([lon_min, lon_max, lat_min, lat_max])

    plotting.set_fontsize(ax, 17)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

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
        plotting.set_fontsize(ax, fsz)
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
        plotting.set_fontsize(ax, fsz)
    
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
        plotting.set_fontsize(ax, lsz)

    ax = plt.subplot(gs[0])
    ax.legend(fontsize=15., loc='upper left')

    plt.tight_layout(pad=0.5)#, h_pad=0.1, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_dist_doy(outfile:str, line:str, color:str,
                 gextrem:str='hi_noperc',
                 show_legend:bool=False, 
                 clr_by_depth:bool=False,
                 cluster_only:bool=False,
                 kludge_MLDN:bool=False,
                 show_clusters:bool=False):

    # Figure
    #sns.set()
    fig = plt.figure(figsize=(12,12))
    plt.clf()
    #ax = plt.gca()

    # Load
    items = cugn_io.load_up(line, gextrem=gextrem, use_full=True,
                            kludge_MLDN=kludge_MLDN)
    grid_extrem = items[0]
    #ds = items[1]
    #times = items[2]
    #grid_tbl = items[3]
    #embed(header='450 of figs')

    jg = sns.jointplot(data=grid_extrem, x='dist', 
                    y='doy', color=color,
                    marginal_ticks=True,
                    marginal_kws=dict(fill=False, color='black', 
                                        bins=100)) 


    # Scatter plot time
    markers = ['o','x','v','s','*']
    clrs = ['k', 'r','b','g','purple']
    jg.ax_joint.cla()
    jg.ax_joint.grid(visible=True)

    for depth in range(5):
        if clr_by_depth:
            clr = clrs[depth]
        else:
            clr = color
        #
        show_these = grid_extrem.depth == depth
        if cluster_only:
            in_cluster = grid_extrem.cluster >= 0
            show_these &= in_cluster

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
            grid_extrem[show_these].dist, 
            grid_extrem[show_these].doy, 
            marker=markers[depth], label=label, 
            facecolors=fc,
            s=15., 
            edgecolors=clr)#, s=50.)
        
    # Add clusters
    if show_clusters:
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
    plotting.set_fontsize(jg.ax_joint, 19)
    if show_legend:
        jg.ax_joint.legend(fontsize=13., loc='lower right')

    
    #plt.tight_layout(h_pad=0.3, w_pad=10.3)
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

    # Stats
    not_winter = (grid_extrem.doy > 50) & (grid_extrem.doy < 300)
    print(f'Percent of profiles not in winter [50-300]: {100.*np.sum(not_winter)/len(grid_extrem):.1f}%')

def fig_combine_dist_doy(img_files:list,
                         outfile='fig_combine_dist_doy.png'):


    # Read the PNG files
    imgs = []
    for img_file in img_files:
        img = mpimg.imread(img_file)
        imgs.append(img)

    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2,2)

    # Display images in each subplot
    for ss in range(4):
        ax = plt.subplot(gs[ss])
        img = imgs[ss]
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

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
        plotting.set_fontsize(ax_doy, 17)
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
        plotting.set_fontsize(ax_doff, 17)
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
    plotting.set_fontsize(ax, fsz)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")



def fig_joint_line90(outfile:str='fig_joint_TDO_line90.png', 
                     line:str='90.0', xmetric='CT',
                     metric:str='doxy',
                     max_depth:int=20,
                     kludge_MLDN:bool=False):
    # Figure
    #sns.set()
    fig = plt.figure(figsize=(12,12))
    plt.clf()

    if metric == 'N':
        cmap = 'Blues'
    elif metric == 'chla':
        cmap = 'Greens'
    elif metric in ['CT', 'doxy']:
        cmap = 'Oranges'
    elif metric == 'SA':
        cmap = 'Greys'
    
    # Load
    items = cugn_io.load_up(line, kludge_MLDN=kludge_MLDN)
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
                    marginal_ticks=True,
                    marginal_kws=dict(fill=False, color='black', 
                                        bins=100)) 

    # Axes                                 
    plotting.set_fontsize(jg.ax_joint, 14)

    # SO
    if metric == 'CT':
        jg.ax_joint.plot(OCs, CTs, 'k:', lw=1)
    elif xmetric == 'CT' and metric == 'doxy':
        jg.ax_joint.plot(CTs, OCs, 'k:', lw=1)


    # Labels
    #lbl = r'$z \le $'+f'{max_depth}m'
    jg.ax_joint.text(0.95, 0.05, f'z <= {max_depth}m',
                transform=jg.ax_joint.transAxes,
                fontsize=14., ha='right', color='k')
    jg.ax_joint.text(0.05, 0.95, f'Line: {line}',
                transform=jg.ax_joint.transAxes,
                fontsize=14., ha='left', color='k')
    jg.ax_joint.set_xlabel(labels[xmetric])
    jg.ax_joint.set_ylabel(labels[metric])

    # y-axis of the marginal plots
    #embed(header='716 of figs')
    #for tick in jg.ax_marg_x.get_yticklabels():
    #    tick.set_visible(True)
    #jg.ax_marg_x.yaxis.set_visible(True)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_joint_pdf_NSO(line:str, max_depth:int=20):

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
    plotting.set_fontsize(ax, fsz)

    # Vertical line at hyperoxic
    ax.axvline(1.1, color='black', linestyle=':')
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_joint_pdf_dsigmaSO(line:str):#, max_depth:int=20):

    def gen_cb(img, lbl, csz = 17.):
        cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
        cbaxes.set_label(lbl, fontsize=csz)
        cbaxes.ax.tick_params(labelsize=csz)

    xvar = 'SO'
    yvar = 'dsigma'
    outfile = f'fig_jointPDF_{line}_SO_dsigma.png'

    # Load
    items = cugn_io.load_line(line, add_fullres=True)
    full_res = items['full_res']
    #embed(header='838 of figs_so')

    # Grid
    SObins=np.linspace(0.4, 1.5, 100)
    dsigbins = np.linspace(-0.2, 3.0, 100)

    gd = np.isfinite(full_res[xvar].values) & np.isfinite(full_res[yvar].values)

    # Counts
    counts, xedges, yedges = np.histogram2d(
                full_res[xvar].values[gd], 
                full_res[yvar].values[gd], 
                bins=[SObins, dsigbins])

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
                            cmap='Reds')
    gen_cb(img, r'$\log_{10} \, p('+f'{xvar},{yvar})$',
           csz=19.)

    # ##########################################################
    tsz = 25.
    ax.text(0.95, 0.9, f'Line: {line}',
                transform=ax.transAxes,
                fontsize=tsz, ha='right', color='k')
    ax.text(0.95, 0.8, f'z <= 100m',
                transform=ax.transAxes,
                fontsize=tsz, ha='right', color='k')

    ax.set_xlabel(labels[xvar])
    ax.set_ylabel(labels[yvar])

    #ax.set_xlim(0.4, 1.6)
    #ax.set_ylim(0., 22.)
    # Set x-axis interval to 0.5
    #ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # 
    fsz = 27.
    plotting.set_fontsize(ax, fsz)

    # Vertical line at hyperoxic
    ax.axvline(1.1, color='black', linestyle=':')
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_extrema_cdfs(outfile:str='fig_N_cdfs.png', metric:str='N',
                     xyLine:tuple=(0.05, 0.90),
                     leg_loc:str='lower right', kludge_MLDN:bool=False):

    # CDFs
    fig = plt.figure(figsize=(7,7))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    for ss, line in enumerate(cugn_defs.lines):
        # Load
        items = cugn_io.load_up(line, kludge_MLDN=kludge_MLDN)
        grid_extrem = items[0]
        ds = items[1]
        times = items[2]
        grid_tbl = items[3]

        cut_grid = (grid_tbl.depth <= 5) & np.isfinite(grid_tbl[metric])

        ctrl = grid_utils.grab_control_values(grid_extrem, grid_tbl[cut_grid], metric, boost=5)

        # Avoid NaN
        ok_extrem = np.isfinite(grid_extrem[metric])
        grid_extrem = grid_extrem[ok_extrem]

        ax = plt.subplot(gs[ss])

        sns.ecdfplot(x=grid_extrem[metric], ax=ax, label='Extrema', 
                     color=cugn_defs.line_colors[ss])
        sns.ecdfplot(x=ctrl, ax=ax, label='Control', color='k', ls='--')

        # KS test
        ks_eval = stats.ks_2samp(grid_extrem[metric], ctrl)
        print(f'KS test: {ks_eval.statistic:.3f}, {ks_eval.pvalue}, {ks_eval.statistic_location}')
        #if line == '90.0':
        #    embed(header='960 of figs')


        # Finish
        #ax.axvline(1., color='black', linestyle='--')
        #ax.axvline(1.1, color='black', linestyle=':')
        lsz = 12.
        ax.legend(fontsize=lsz, loc=leg_loc)

        #ax.set_xlim(-20, 100.)
        ax.set_xlabel(labels[metric])
        ax.set_ylabel('CDF')
        ax.text(xyLine[0], xyLine[1], f'Line: {line}', 
                transform=ax.transAxes,
                fontsize=lsz, ha='left', color='k')
        ax.grid()
        plotting.set_fontsize(ax, 13)

        # Stats
        # Percentile of the extrema
        val = np.nanpercentile(grid_extrem[metric], (10,90))
        print(f'Line: {line} -- percentiles={val}')

        # KS tests
        ks_eval = stats.ks_2samp(grid_extrem[metric], ctrl)
        #embed(header='716 of figs')
        print(f'KS test: {ks_eval.statistic:.3f}, {ks_eval.pvalue:.3f}, {ks_eval.statistic_location}')

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
    plotting.set_fontsize(jg.ax_joint, 14)

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


def fig_multi_scatter_event(outfile:str, line:str, 
                      event:str, t_off, gextrem:str='high',
                      ):

    # Load
    items = cugn_io.load_up(line, gextrem=gextrem, use_full=True)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]

    # Grid extrem
    #uni, n_uni = np.unique(grid_extrem.cluster, return_counts=True)

    # Grab event
    tevent = pandas.Timestamp(event)
    tmin = tevent - pandas.Timedelta(t_off)
    tmax = tevent + pandas.Timedelta(t_off)

    # In event
    in_event = (grid_extrem.time >= tmin) & (grid_extrem.time <= tmax)
    ds_in_event = (ds.time >= tmin) & (ds.time <= tmax)
    grid_in_event = (grid_tbl.time >= tmin) & (grid_tbl.time <= tmax)

    # Mission
    missions = np.unique(ds.mission[ds_in_event].data)
    mission_profiles = np.unique(ds.mission_profile[ds_in_event].data)
    print(f'Missions: {missions}')
    #print(f'Mission Profiles: {mission_profiles}')

    dist, _ = cugn_utils.calc_dist_offset(
                    line, ds.lon[ds_in_event].values, 
                    ds.lat[ds_in_event].values)
    imin = np.argmin(dist)

    # Range of dist
    print(f"Range of dist: {dist[0]} to {dist[-1]}")
    print(f"Minimum dist = {dist.min()} at {ds.time[ds_in_event][imin]}")

    fig = plt.figure(figsize=(10,8))
    plt.clf()

    gs = gridspec.GridSpec(5,2)

    cnt = 0
    #metrics = ['SO', 'doxy', 'N', 'CT', 'dist']
    metrics = ['SO', 'doxy', 'N', 'CT', 'chla']
    #metrics = ['SO', 'doxy', 'N', 'CT', 'SA']
    clrs = ['gray', 'purple', 'blue', 'red', 'green']
    nsub = len(metrics)
    #for clr, z in zip(['b', 'g', 'r'], [10, 20, 30]):
    for col, z in enumerate([10, 20]):
        depth = z//10 - 1

        grid_depth = grid_tbl[grid_tbl.depth == depth].copy()

        # Axis
        for ii, clr, metric in zip(
            np.arange(nsub), clrs, metrics):

            #row = col*3 + ii
            row = ii
            ax = plt.subplot(gs[row:row+1, col:col+1])
                                     #'dist']):
            if metric == 'T':
                ds_metric = 'CT'
            elif metric == 'chla':
                ds_metric = 'chlorophyll_a'
            else:
                ds_metric = metric


            # Save the date
            if ii == 0:
                #sv_dates = ds.time[ds_in_event][srt_ds]
                sv_dates = grid_depth.time[grid_in_event].values
            plt_depth = depth
            if metric in ['dist']:
                yvals = dist
            else:
                yvals = grid_depth[metric][grid_in_event].values
                #yvals = ds[ds_metric][plt_depth,ds_in_event][srt_ds].values

            # Twin?
            axi = ax
            #if metric == 'T':
            #    ax2 = ax.twinx()
            #    ax2.set_ylim(yvals.min(), yvals.max())
            #    axi = ax2
            #elif metric == 'N':
            #    ax3 = ax2.twinx()
            #    ax3.set_ylim(yvals.min(), yvals.max())
            #    axi = ax3


            # Plot all
            #axi.scatter(ds.time[ds_in_event][srt], 
            axi.scatter(grid_depth.time[grid_in_event].values, 
                    yvals, edgecolor=clr,
                    facecolor='none', alpha=0.5, zorder=1)

            # Plot extrema
            extrem = grid_depth.SO[grid_in_event].values > 1.1
            axi.scatter(grid_depth.time[grid_in_event].values[extrem],
                    yvals[extrem], color=clr,
                    zorder=10)
            #at_d = grid_extrem.depth[in_event] == depth
            #axi.scatter(grid_extrem.time[in_event][at_d], 
            #           grid_extrem[metric][in_event][at_d], 
            #           color=clr, zorder=10)

            # Debug
            #embed(header='fig_multi_scatter_event 965')

            if metric == 'N':
                ax.set_ylim(bottom=0.)
                ax.text(0.95, 0.05, f'z={z} m',
                    transform=ax.transAxes,
                    fontsize=20, ha='right', color='k')
            elif metric == 'SO':
                # Horizontal line at 1.1
                ax.axhline(1.1, color='gray', linestyle='--')

            # Axes
            ax.set_ylabel(short_lbl[metric])
            plotting.set_fontsize(ax, 14.)
            #if z < 20 or ii < 3:
            if ii < (nsub-1):
                ax.set_xticklabels([])
                ax.tick_params(bottom=False)
            else:
                #plt.locator_params(axis='x', nbins=5)  # Show at most 5 ticks
                #locator = mdates.MultipleLocator(2)
                #ax.xaxis.set_major_locator(locator)
                ax.tick_params(axis='x', rotation=45)

            ax.set_xlim(sv_dates[0], sv_dates[-1])
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

            # Add grid
            ax.grid(True)

            cnt += 1

    #gs.tight_layout(fig)
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_SOa_pdfs(line:str, zmax:int=4, 
                 dmax:float=50.,
                 variable:str='SO'):
    """
    Generate a figure showing the anomalies of a given variable (default: 'doxy') for a specific line.
    Parameters:
        line (str): The line for which the figure is generated.
        zmax (int): The maximum depth for the figure in meters (default: 9).
        dmax (float): The maximum distance for the figure in kilometers (default: 100.0).
        variable (str): The variable to plot the anomalies for (default: 'doxy').
    Returns:
        None
    """

    outfile = f'fig_{variable}a_{line}_{int(dmax)}_z{10*(zmax+1)}.png' 
    iline = lines.index(line)
    clr = line_colors[iline]

    # Load
    grid = so_analysis.load_annual(line)

    # Calculate
    grid['doxya'] = grid.doxy - grid.ann_doxy
    grid['SOa'] = grid.SO - grid.ann_SO


    # Histogram me
    fig = plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(2,2)
    plt.clf()

    if variable == 'doxy':
        bins = np.linspace(-150.,150.,50)
    elif variable == 'SO':
        bins = np.linspace(-0.5, 0.5, 50)

    for ss in range(4):
        ax = plt.subplot(gs[ss])

        # Cut down on season
        if ss == 0: # Winter
            in_season = (grid.time.dt.month >= 12) | (grid.time.dt.month <= 2)
            lbl = 'Winter'
        elif ss == 1: # Spring
            in_season = (grid.time.dt.month >= 3) & (grid.time.dt.month <= 5)
            lbl = 'Spring'
        elif ss == 2: # Summer
            in_season = (grid.time.dt.month >= 6) & (grid.time.dt.month <= 8)
            lbl = 'Summer'
        elif ss == 3: # Fall
            in_season = (grid.time.dt.month >= 9) & (grid.time.dt.month <= 11)
            lbl = 'Fall'

        if variable == 'doxy':
            anom = grid.doxya[in_season]
        elif variable == 'SO':
            anom = grid.SOa[in_season]
        

        ax.hist(anom, bins=bins, color=clr, 
                 fill=True, edgecolor=clr, label=lbl) 
                 #log_scale=(False,True))#, label='Extrema')

        # Gaussian
        mean = np.mean(anom)
        std = np.std(anom)
        skew = stats.skew(anom)

        '''
        # Add a cDF in an inset
        axin = ax.inset_axes([0.67, 0.6, 0.3, 0.3])

        sorted_data = np.sort(anom)

        # Calculate the proportional values of samples
        y = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)


        # Create the CDF plot
        axin.plot(sorted_data, y, color=clr)

        # Overplot the Gaussian
        x = np.linspace(-5*std, 5*std, 10000)
        y = stats.norm.cdf(x/std)
        axin.plot(x, y, 'k:')

        axin.set_xlabel('Value')
        axin.set_ylabel('CDF')
        axin.set_xlim(-0.5, 0.5)
        '''

        #ax.plot(x, y*len(grid.doxya)*std/50, 'k-', lw=2)                

        xlbl = short_lbl[variable]
        if variable == 'doxy':
            xlbl = xlbl.replace('DO','DOa')
        elif variable == 'SO':
            xlbl = xlbl.replace('SO','SOa')
        ax.set_xlabel(xlbl)
        ax.set_ylabel('Count')

        # Describe
        #if ss == 0:
        ax.text(0.05, 0.90, 
                lbl+'\n\n'+f'N={len(anom)}\n'+f'$\mu$={mean:.2f}\n$\sigma$={std:.2f}\nskew={skew:.2f}',
                transform=ax.transAxes, 
                fontsize=18., ha='left', color='k',
                va='top')

        fsz = 19.
        plotting.set_fontsize(ax, fsz)

        #ax.legend(fontsize=fsz)

    # Title
    fig.suptitle(f'Line={line}, '+r'$z \leq $'+f'{10*(zmax+1)}m, '+r'$d \leq $'+f'{int(dmax)}km', 
                 fontsize=29)


    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_diurnal(line:str, kludge_MLDN:bool=False,
                iz:int=0):

    bins_SO = np.linspace(0.8, 1.3, 50)
    bins_hour = np.linspace(0, 24, 25)

    # Load
    items = cugn_io.load_up(line, kludge_MLDN=kludge_MLDN)
    grid_extrem = items[0]
    ds = items[1]

    # Time
    ns = ds.time.data.astype('datetime64[ns]').astype('int64') % (24 * 60 * 60 * 1_000_000_000)
    decimal_hours = ns / (60 * 60 * 1_000_000_000) 
    # Offset to CA
    decimal_hours -= 8
    next_day = decimal_hours < 0
    decimal_hours[next_day] += 24

    gd = np.isfinite(ds.SO.data[iz,:])

    counts, xedges, yedges = np.histogram2d(
                decimal_hours[gd],
                ds.SO.data[iz,gd], 
                bins=[bins_hour, bins_SO])

    # PDF
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    p_norm = np.sum(counts) * (dx * dy)
    consv_pdf = counts / p_norm

    # Average
    consv_pdf.shape
    SO_vals = yedges[:-1] + (yedges[1]-yedges[0])/2.
    avg_SOs = []
    for hour in range(consv_pdf.shape[0]):
        avg_SO = np.sum(SO_vals * consv_pdf[hour,:]) / np.sum(consv_pdf[hour,:])
        avg_SOs.append(avg_SO)
    #
    avg_SOs = np.array(avg_SOs)

    # Stats
    mean_SO = np.mean(avg_SOs)
    median_SO = np.median(avg_SOs)
    peak_SO = np.max(avg_SOs)
    print(f'Line={line}: Mean SO={mean_SO:.3f}, Peak SO={peak_SO:.3f}, Peak hour={np.argmax(avg_SOs)}')
    print(f'Median SO={median_SO:.3f}')
    print(f'Percent increase = {(peak_SO/mean_SO-1)*100:.3f}')

    cmap = 'Purples'

    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    #
    img = ax.pcolormesh(xedges, yedges, np.log10(consv_pdf.T),
                                    cmap=cmap)
    # Averages
    ax.plot(np.arange(24)+0.5, avg_SOs, 'ro')
    #
    gen_cb(img, 'Log10 Counts')
    #
    ax.set_xlabel('Hour (Relative to GMT-8)')
    ax.set_ylabel(f'SO at z={(iz+1)*10}m')
    #
    plotting.set_fontsize(ax, 15.)
    #
    ax.text(0.05, 0.05, f'Line={line}',
                transform=ax.transAxes,
                fontsize=15., ha='left', color='k')
    plt.tight_layout()
    outfile = f'fig_diurnal_{line}_iz{iz}.png' 

    plt.savefig(outfile, dpi=300) 
    print(f'Saved: {outfile}')


def fig_di_extrem(line:str, kludge_MLDN:bool=False,
                iz:int=0):

    lidx = lines.index(line)
    outfile = f'fig_di_extrem_{line}.png' 

    bins_SO = np.linspace(0.8, 1.3, 50)
    bins_hour = np.linspace(0, 24, 25)

    # Load
    items = cugn_io.load_up(line, kludge_MLDN=kludge_MLDN)
    grid_extrem = items[0]
    ds = items[1]

    # Time
    ns = grid_extrem.time.values.astype('datetime64[ns]').astype('int64') % (24 * 60 * 60 * 1_000_000_000)
    decimal_hours = ns / (60 * 60 * 1_000_000_000) 
    # Offset to CA
    decimal_hours -= 8
    next_day = decimal_hours < 0
    decimal_hours[next_day] += 24

    #gd = np.isfinite(ds.SO.data[iz,:])

    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    #
    _ = sns.histplot(decimal_hours, bins=bins_hour, color=line_colors[lidx], ax=ax)
    #
    ax.set_xlabel('Hour (Relative to GMT-8)')
    ax.set_ylabel('Counts')
    #
    plotting.set_fontsize(ax, 15.)
    #
    ax.text(0.05, 0.95, f'Line={line}\nHyperoxic Extrema',
                transform=ax.transAxes,
                fontsize=15., va='top', ha='left', color='k')
    plt.tight_layout()

    plt.savefig(outfile, dpi=300) 
    print(f'Saved: {outfile}')


def fig_cluster_date_vs_loc(outfile='fig_cluster_date_vs_loc.png', 
                            use_full:bool=False, kludge_MLDN:bool=False,
                            debug:bool=False):

    day_ns = 24 * 60 * 60 * 1_000_000_000
    
    # Figure
    fig = plt.figure(figsize=(6,6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])

    markers = ['o', 's', 'd', '^']
    min_dur, max_dur = 9999., -1
    for tt, clr, line in zip(np.arange(4), line_colors, lines):
        if tt != 0 and debug:
            continue

        # Load
        items = cugn_io.load_up(line, use_full=use_full, kludge_MLDN=kludge_MLDN)#, skip_dist=True)
        grid_extrem = items[0]
        ds = items[1]
        times = items[2]
        grid_tbl = items[3]

        cluster_stats = clusters.cluster_stats(grid_extrem)
        dur_days = cluster_stats.Dtime.values.astype('float') / day_ns
        #embed(header='fig_cluster_char 1622')

        # Duration
        min_dur = min(min_dur, np.min(dur_days))
        max_dur = max(max_dur, np.max(dur_days))

        # Duration, size
        #embed(header='1689 of figs')
        ax.scatter(cluster_stats['Cdist'].values, 
                   cluster_stats['time'].values, 
                marker=markers[tt], edgecolor=clr, 
                facecolor='None',
                s=1., #cluster_stats['Ddist'].values, 
                label=f'Line {line}')

        # Loop over the clusters
        #embed(header='1739 of figs')
        for jj in range(len(cluster_stats)):
            iobj = cluster_stats.iloc[jj]
            #
            iE = Ellipse((iobj.Cdist,iobj.time),
                         iobj.Ddist, 3*dur_days[jj],
                         color=clr)
            ax.add_patch(iE)

    plotting.set_fontsize(ax, 17.)
    ax.yaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # Add a horizontal line on January 1 of each year
    for year in range(2017,2024):
        ax.axhline(pandas.Timestamp(f'{year}-01-01').to_datetime64(), color='gray', linestyle='--')

    ax.legend(fontsize=13., loc='lower right')

    #ax.legend(fontsize=15.)

    # Labels
    ax.set_xlabel('Distance Offshore (km)')
    ax.set_ylabel('Date')
    
    plt.tight_layout(pad=0.5)#, h_pad=0.1, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

    # Duration
    print(f'Min duration: {min_dur:.2f} days')
    print(f'Max duration: {max_dur:.2f} days')


def fig_SO_ex_below(Nval, outroot:str='fig_SO_ex_below_N'):

    outfile = f'{outroot}{Nval}.png'


    # Start the figure
    fig = plt.figure(figsize=(10,12))
    plt.clf()
    gs = gridspec.GridSpec(2,2)
    #embed(header='fig_SO_ex_below 1792')

    # Load
    for ss, line in enumerate(cugn_defs.lines):
        items = cugn_io.load_up(line)
        grid_extrem = items[0]
        profiles = np.unique(grid_extrem.profile)

        # Stats
        f5s, f10s = [], []
        tot_SO, tot_N5, tot_N10 = 0, 0, 0
        for profile in profiles:
            idx = grid_extrem.profile == profile
            idx0 = int(np.where(idx)[0][0])
            if np.isnan(grid_extrem.iloc[idx0].NSO):
                continue
            #
            f5s.append(grid_extrem.iloc[idx0].Nf5 /  grid_extrem.iloc[idx0].NSO)
            f10s.append(grid_extrem.iloc[idx0].Nf10 /  grid_extrem.iloc[idx0].NSO)
            #if profile == 32965:
            #    import pdb
            #    pdb.set_trace()
            #
            tot_SO += grid_extrem.iloc[idx0].NSO
            tot_N5 += grid_extrem.iloc[idx0].Nf5
            tot_N10 += grid_extrem.iloc[idx0].Nf10
        # Recast
        f5s = np.array(f5s)
        f10s = np.array(f10s)

        if Nval == 5:
            fvals = f5s
            print(f'Line {line}: N5/Ntot = {tot_N5/tot_SO:.3f}')
        else:
            fvals = f10s
            print(f'Line {line}: N10/Ntot = {tot_N10/tot_SO:.3f}')

        # DOY
        ax= plt.subplot(gs[ss])
        ax.hist(fvals, bins=20,# histtype='step',
                    color=cugn_defs.line_colors[ss], label=f'Line {line}', lw=2)
        #
        ax.grid(True)
        ax.set_xlim(0., 1.)
        if ss < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(f'Fraction of Hyperoxic Below N={Nval} (All profiles)')
        plotting.set_fontsize(ax, 17)
        ax.set_ylabel('Count')

    #plt.tight_layout(h_pad=0.3, w_pad=10.3)
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # MLD
    if flg & (2**30):
        fig_cugn()

    # Figure 2 -- Joint PDFs
    if flg & (2**0):
        #fig_joint_pdfs()
        #fig_joint_pdfs(use_density=True)
        fig_joint_pdfs(use_DO=True)

    # Figure 3 -- Joint PDF: T, DO on Line 90
    if flg & (2**2):
        fig_joint_line90(kludge_MLDN=False)

    # 
    if flg & (2**3):
        line = '90'
        fig_joint_pdf_NSO(line)

    # Figure 4 -- SO CDFs
    if flg & (2**4):
        fig_SO_cdf('fig_SO_cdf.png', use_full=True)


    # Figure 5 -- DOY vs Offshore distance
    if flg & (2**5):
        img_files = []
        for line, clr in zip(lines, line_colors):
            # Skip for now
            if line != '90.0':
                continue
            if line == '56.0':
                show_legend = True
            else:
                show_legend = False
            # High
            img_file = f'fig_dist_doy_{line}.png' 
            fig_dist_doy(img_file,
                         line, clr, show_legend=show_legend,
                         gextrem='high',
                         clr_by_depth=True,
                         cluster_only=False,
                         kludge_MLDN=False)
            img_files.append(img_file)
            # Low
            #fig_dist_doy(f'fig_dist_doy_low_{line}.png', 
            #             line, clr, 
            #             gextrem='low_noperc',
            #             show_legend=show_legend,
            #             clr_by_depth=True)
        # Combine them
        #fig_combine_dist_doy(img_files)

    # Figure ? -- SO vs. N
    if flg & (2**6):
        fig_SO_vs_N_zoom()

    # Figure 7 -- multi-scatter
    if flg & (2**7):
        line = '90.0'
        eventA = ('2020-09-01', '10D') # Sub-surface

        event, t_off = eventA
        fig_multi_scatter_event(
            f'fig_multi_scatter_event_{line}_{event}.png', 
            line, event, t_off)#, gextrem='hi_noperc')
        

    # Figure 10 -- doy, distance for the low extrema
    if flg & (2**8):
        fig_dist_doy_low()

    # Figure 9 -- doy, distance for the low extrema
    if flg & (2**9):
        for line in ['90.0', '56.0']:
            fig_SOa_pdfs(line)

    # Figure 2 -- average DO, SO 
    if flg & (2**10):
        line = '90'
        fig_mean_DO_SO(line)


    # Extrema CDFs
    if flg & (2**18):
        kludge_MLDN = False
        # drho
        #fig_extrema_cdfs('fig_dsigma0_cdfs.png', metric='dsigma0',
        #                 xyLine=(0.7, 0.4), kludge_MLDN=kludge_MLDN)
        # N
        fig_extrema_cdfs(kludge_MLDN=kludge_MLDN)
        '''
        # Chla
        fig_extrema_cdfs('fig_chla_cdfs.png', metric='chla',
                         xyLine=(0.7, 0.4), kludge_MLDN=kludge_MLDN)
        # DO
        fig_extrema_cdfs('fig_doxy_cdfs.png', metric='doxy',
                         xyLine=(0.7, 0.4), leg_loc='upper left', 
                         kludge_MLDN=kludge_MLDN)
        # MLD
        fig_extrema_cdfs('fig_mld_cdfs.png', metric='MLD',
                         xyLine=(0.7, 0.4), leg_loc='lower right',
                         kludge_MLDN=kludge_MLDN)
        '''

    # Annual cycle
    if flg & (2**19):
        fig_annual('fig_annual_TDO.png', line='90.0', metric='T')

    # Velocities
    if flg & (2**25):
        # Total velocity
        fig_extrema_cdfs('fig_vel_cdfs.png', metric='vel',
                         xyLine=(0.7, 0.4), leg_loc='upper left')
        # U, V
        fig_extrema_cdfs('fig_u_cdfs.png', metric='u',
                         xyLine=(0.7, 0.4), leg_loc='upper left')
        fig_extrema_cdfs('fig_v_cdfs.png', metric='v',
                         xyLine=(0.7, 0.4), leg_loc='upper left')

    # Upwelling
    if flg & (2**26):
        # CUTI
        fig_extrema_cdfs('fig_cuti_cdfs.png', metric='cuti',
                         xyLine=(0.7, 0.4), leg_loc='upper left')
        # BEUTI
        fig_extrema_cdfs('fig_beuti_cdfs.png', metric='beuti',
                         xyLine=(0.7, 0.4), leg_loc='lower right')

    # Diurnal
    if flg & (2**31):
        #fig_diurnal('90.0', kludge_MLDN=True)
        fig_di_extrem('90.0')

    # Fraction of SO below N threshold
    if flg & (2**32):
        fig_SO_ex_below(5)
        #fig_SO_ex_below(10)

    # Cluster date vs location/size
    if flg & (2**37):
        fig_cluster_date_vs_loc(kludge_MLDN=False)#debug=True)

    if flg & (2**40):
        #line = '90'
        line = '56'
        fig_joint_pdf_dsigmaSO(line)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 30  # Figure 1 -- CUGN 
        #flg += 2 ** 0  # 1 -- Figure 2 Joint PDFs of all 4 lines
        #flg += 2 ** 2  # 4 Figure 3 Joint PDF of DO vs. T on Line 90 
        #flg += 2 ** 4  # Figure 4: SO CDFs
        flg += 2 ** 5  # Figure 5: DOY vs. offshore distance
        #flg += 2 ** 37  # Figure 8: Clusters
        #flg += 2 ** 8  # 256 -- Figure 10: DOY vs. offshore distance for low
        #flg += 2 ** 18  # # Extreme CDFs; Figures 11, 12, 13
        #flg += 2 ** 6  # 
        #flg += 2 ** 7  # 
        #flg += 2 ** 3  # N vs. SO

        #flg += 2 ** 9  # SOa PDFs

        # Appenedix
        #flg += 2 ** 31  # Diurnal figs
        #flg += 2 ** 32  # SO below N threshold (and MOD)

        #flg += 2 ** 11  
        #flg += 2 ** 12  # Low histograms

        #flg += 2 ** 19  # T anomaly vs. DO

        #flg += 2 ** 25  # 
        #flg += 2 ** 26  # Upwelling

        #flg += 2 ** 40  # Joint PDF dsigma, SO

    else:
        flg = sys.argv[1]

    main(flg)