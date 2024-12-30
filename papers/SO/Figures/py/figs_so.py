# imports
import os, sys

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d 

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator 
from matplotlib.patches import Ellipse

mpl.rcParams['font.family'] = 'stixgeneral'


import seaborn as sns

import pandas

from ocpy.utils import plotting

from cugn import grid_utils
from cugn import utils as cugn_utils
from cugn import io as cugn_io
from cugn import defs as cugn_defs
from cugn import clusters

lines = cugn_defs.lines
line_colors = cugn_defs.line_colors
line_cmaps = cugn_defs.line_cmaps

from gsw import conversions, density
import gsw

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import so_analysis

from IPython import embed

def gen_cb(img, lbl, csz = 17.):
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=csz)
    cbaxes.ax.tick_params(labelsize=csz)

ylbl_dict = {'doxy': 'DO ('+r'$\mu$'+'mol/kg)', 
                 'T': 'Temperature (deg C)',
                 'CT': 'Temperature (deg C)',
                 'SA': 'Salinity (g/kg)',
                 'N': 'Buoyancy frequency (cycles/hr)', 
                 'SO':'Oxygen Saturation', 
                 'dist': 'Distance from shore (km)', 
                 'MLD': 'Mixed Layer Depth (m)',
                 'chla': 'Chl-a (mg/m'+r'$^3$'+')'}

short_lbl = {'doxy': 'DO ('+r'$\mu$'+'mol/kg)', 
                 'T': 'T (deg C)',
                 'CT': 'T (deg C)',
                 'SA': 'SA (g/kg)',
                 'SO': 'SO',
                 'N': 'N (cycles/hr)',
                 'MLD': 'MLD (m)',
                 'dist': 'Distance from shore (km)',
                 'chla': 'Chl-a (mg/m'+r'$^3$'+')'}


class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(
            r+1,r+1, subplot_spec=self.subplot)
        #print(f'r={r}')

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

'''
def load_up(line, get_extrem:str='high'):
    # Load
    items = cugn_io.load_line(line, gextrem)
    grid_tbl = items['grid_tbl']
    ds = items['ds']

    # Fill
    grid_utils.fill_in_grid(grid_tbl, ds)

    # Cluters 
    perc = 80.  # Low enough to grab them all
    grid_outliers, _, _ = grid_utils.gen_outliers(line, perc)

    extrem = grid_outliers.SO > 1.1
    grid_extrem = grid_outliers[extrem].copy()
    times = pandas.to_datetime(grid_extrem.time.values)

    # Fill in N_p, chla_p
    grid_utils.find_perc(grid_tbl, 'N')
    grid_utils.find_perc(grid_tbl, 'chla')

    dp_gt = grid_tbl.depth*100000 + grid_tbl.profile
    dp_ge = grid_extrem.depth*100000 + grid_extrem.profile
    ids = cat_utils.match_ids(dp_ge, dp_gt, require_in_match=True)
    assert len(np.unique(ids)) == len(ids)

    grid_extrem['N_p'] = grid_tbl.N_p.values[ids]
    grid_extrem['chla_p'] = grid_tbl.chla_p.values[ids]

    # Add to df
    grid_extrem['year'] = times.year
    grid_extrem['doy'] = times.dayofyear

    # Add distance from shore

    dist, _ = cugn_utils.calc_dist_offset(
        line, grid_extrem.lon.values, grid_extrem.lat.values)
    grid_extrem['dist'] = dist

    # Cluster me
    clusters.generate_clusters(grid_extrem)
    cluster_stats = clusters.cluster_stats(grid_extrem)

    return grid_extrem, ds, times, grid_tbl
'''

def fig_pdf_cdf(outfile:str, line, SO_cut:float=1.1):

    # Load
    items = cugn_io.load_line(line)
    grid_tbl = items['grid_tbl']
    ds = items['ds']

    # Fill
    grid_utils.fill_in_grid(grid_tbl, ds)

    # Cuts
    highSO = grid_tbl.SO > SO_cut
    highSO_tbl = grid_tbl[highSO]

    # FIGURE
    fig = plt.figure(figsize=(12,12))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    axes = []
    metrics = ['N', 'z', 'chla', 'T']
    labels = ['N (cycles/hour)', 'z (m)', 'Chl-a', 'T (deg C)']
    for ss, metric, label in zip(np.arange(len(metrics)), metrics, labels):
        # Build control
        control = grid_utils.grab_control_values(
            highSO_tbl, grid_tbl, metric)
        # Nan
        control = control[np.isfinite(control)]

        # Plot
        ax = plt.subplot(gs[ss])

        if metric in ['chla']:
            log_scale = (True, False)
        else:
            log_scale = (False, False)

        sns.ecdfplot(x=highSO_tbl[metric], ax=ax, label=f'SO > {SO_cut}', log_scale=log_scale)
        sns.ecdfplot(x=control, ax=ax, label='Control', color='k', log_scale=log_scale)

        if ss == 0:
            ax.legend(fontsize=15.)

        # Label
        ax.set_xlabel(label)
        ax.set_ylabel('CDF')

        axes.append(ax)

    # Pretty up
    for ax in axes:
        plotting.set_fontsize(ax, 17)

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_varySO_pdf_cdf(outfile:str, line):
    # Load
    items = cugn_io.load_line(line)
    grid_tbl = items['grid_tbl']
    ds = items['ds']

    # Fill
    grid_utils.fill_in_grid(grid_tbl, ds)

    # Figure
    fig = plt.figure(figsize=(12,12))
    plt.clf()
    ax = plt.gca()

    metric = 'N'
    for SO_cut in [1., 1.05, 1.1, 1.2, 1.3]:
        highSO = grid_tbl.SO > SO_cut
        highSO_tbl = grid_tbl[highSO]

        if SO_cut == 1.:
            label = 'Control'
            control = grid_utils.grab_control_values(
                highSO_tbl, grid_tbl, metric)
            control = control[np.isfinite(control)]
            sns.ecdfplot(x=control, ax=ax, label='Control', color='k')

        sns.ecdfplot(x=highSO_tbl[metric], ax=ax, label=f'SO > {SO_cut}')


    # Finish
    ax.legend(fontsize=15.)
    plotting.set_fontsize(ax, 17)

    ax.set_xlabel('N (cycles/hour)')
    ax.set_ylabel('CDF')

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_timeseries(outfile:str, line, vmax=1.3):

    # Do it
    items = load_up(line)
    grid_extrem = items[0]

    # Figure
    # https://stackoverflow.com/questions/57292022/day-of-year-format-x-axis-matplotlib
    fig = plt.figure(figsize=(12,12))
    _, axs = plt.subplots(1,6,sharey=True,gridspec_kw = {'wspace':0, 'hspace':0})


    # Loop in the years
    for ss, year in enumerate([2017, 2018, 2019, 2020, 2021, 2022]):
        in_year = grid_extrem.year == year
        grid_year = grid_extrem[in_year]

        #if year == 2020:
        #    embed(header='185 of figs_so')

        ax = axs[ss]
        sc = ax.scatter(grid_year.dist, grid_year.doy,
                c=grid_year.SO, cmap='jet', s=1, vmin=1.1,
                vmax=vmax)

        if ss == 1:
            ax.set_xlabel('Distance from shore (km)')
        if ss == 0:
            #ax.set_ylabel('Date')
            ax.set_ylim(0., 366.)
            major_format = mdates.DateFormatter('%b-%d')
            ax.yaxis.set_major_formatter(major_format)
        # Title
        if ss == 3:
            ax.set_title(f'Line {line}', fontsize=15.)

        fsz = 13.
        ax.text(0.95, 0.9, f'{year}',
                transform=ax.transAxes,
                fontsize=fsz, ha='right', color='k')
        if line == '90':
            ax.set_xlim(-20., 399.)
        elif line == '80':
            ax.set_xlim(-150., 236.)
        elif line == '66':
            ax.set_xlim(-10., 320.)
        

        # Finish
        plotting.set_fontsize(ax, 13)

    cax,kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
    cbaxes = plt.colorbar(sc, cax=cax, **kw)
    cbaxes.set_label('Saturated Oxygen', fontsize=13.)
    cbaxes.ax.tick_params(labelsize=13)



    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_event(outfile:str, line:str, event:str, t_off,
    max_depth=10):

    # Load
    items = load_up(line)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]


    # Grab event
    tevent = pandas.Timestamp(event)
    tmin = tevent - pandas.Timedelta(t_off)
    tmax = tevent + pandas.Timedelta(t_off)

    # DOY
    otimes = pandas.to_datetime(ds.time.data)
    in_event = (ds.time >= tmin) & (ds.time <= tmax)
    p_min = ds.profile[in_event].values.min() 
    p_max = ds.profile[in_event].values.max() 

    x_lims = mdates.date2num([otimes[p_min], otimes[p_max]])


    csz = 13.
    # Figure
    fig = plt.figure(figsize=(12,10))
    gs = gridspec.GridSpec(2,2)

    # #########################################################3
    # DO
    ax_DO = plt.subplot(gs[0])
    # Contours from SO
    im = ax_DO.imshow(ds.doxy.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='Purples', vmin=200., aspect='auto')
    #im = ax.imshow(ds.SO.data[0:max_depth, p_min:p_max],
    #    extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
    #               cmap='jet', vmin=0.9, aspect='auto')
    cax,kw = mpl.colorbar.make_axes([ax_DO])
    cbaxes = plt.colorbar(im, cax=cax, **kw)
    cbaxes.set_label('Dissolved Oxygen', fontsize=csz)

    # #########################################################3
    # SO contour
    SOs = ds.SO.data[0:max_depth, p_min:p_max]
    Np = p_max-p_min
    X = np.outer(np.ones(max_depth), 
        np.linspace(x_lims[0], x_lims[1], Np))
    Y = np.outer(np.linspace(0., max_depth*10., max_depth), 
                 np.ones(Np))

    ax_DO.contour(X, Y, SOs, levels=[1., 1.1, 1.2],
               colors=['white', 'gray', 'black'], linewidths=1.5)

    '''
    # #########################################################3
    # DO percentile contour
    in_view = (grid_tbl.profile >= p_min) & (grid_tbl.profile <= p_max) & (
        grid_tbl.depth < max_depth)
    doxy_p_grid = np.zeros_like(ds.N.data)
    for _, row in grid_tbl[in_view].iterrows():
        doxy_p_grid[row.depth, row.profile] = row.doxy_p
    ax_DO.contour(X, Y, doxy_p_grid[0:max_depth, p_min:p_max], 
                  levels=[90., 95.],
               colors=['white', 'black'], linewidths=1.5)
    '''

    # #########################################################3
    # N
    ax_N = plt.subplot(gs[1])
    im_N = ax_N.imshow(ds.N.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='Blues', aspect='auto')

    # N percentile
    grid_utils.find_perc(grid_tbl, 'N')
    Np_grid = np.zeros_like(ds.N.data)

    in_view = (grid_tbl.profile >= p_min) & (grid_tbl.profile <= p_max) & (
        grid_tbl.depth < max_depth)
    # Painful loop, but do it
    #embed(header='214 of figs_so')
    for _, row in grid_tbl[in_view].iterrows():
        Np_grid[row.depth, row.profile] = row.N_p

    ax_N.contour(X, Y, Np_grid[0:max_depth, p_min:p_max], 
                  levels=[90., 95.],
               colors=['gray', 'black'], linewidths=1.5)
    

    cax,kw = mpl.colorbar.make_axes([ax_N])
    cbaxes = plt.colorbar(im_N, cax=cax, **kw)
    cbaxes.set_label('Buoyancy (cycles/hour)', fontsize=csz)

    # T
    ax_T = plt.subplot(gs[2])
    im_T = ax_T.imshow(ds.temperature.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='jet', aspect='auto')
    cax,kw = mpl.colorbar.make_axes([ax_T])
    cbaxes = plt.colorbar(im_T, cax=cax, **kw)
    cbaxes.set_label('Temperature (deg C)', fontsize=csz)

    # T
    ax_C = plt.subplot(gs[3])
    im_C = ax_C.imshow(ds.chlorophyll_a.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='Greens', aspect='auto')
    cax,kw = mpl.colorbar.make_axes([ax_C])
    cbaxes = plt.colorbar(im_C, cax=cax, **kw)
    cbaxes.set_label('Chla', fontsize=csz)


    # Finish
    for ss in range(4):
        ax = plt.subplot(gs[ss])
        #
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Date')
        # Axes
        ax.xaxis_date()
        #major_format = mdates.DateFormatter('%b')
        #ax.xaxis.set_major_formatter(major_format)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))

        ax.tick_params(axis='x', rotation=10)
        #ax.autofmt_xdate()
        plotting.set_fontsize(ax, 14)

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_pivot_event(outfile:str, line:str, event:str, t_off,
    max_depth=8):

    # Load
    items = load_up(line)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]


    # Grab event
    tevent = pandas.Timestamp(event)
    tmin = tevent - pandas.Timedelta(t_off)
    tmax = tevent + pandas.Timedelta(t_off)

    # DOY
    otimes = pandas.to_datetime(ds.time.data)
    in_event = (ds.time >= tmin) & (ds.time <= tmax)
    p_min = ds.profile[in_event].values.min() 
    p_max = ds.profile[in_event].values.max() 

    x_lims = mdates.date2num([otimes[p_min], otimes[p_max]])


    csz = 13.
    # Figure
    fig = plt.figure(figsize=(12,4))
    #gs = gridspec.GridSpec(1,9)

    all_ax = []
    ypos = 0.15
    axw = 0.27
    axh = 0.8

    # #########################################################3
    # DO
    #ax_DO = plt.subplot(gs[0:2])
    ax_DO = fig.add_axes([0.05, ypos, axw, axh])
    all_ax.append(ax_DO)
    # Contours from SO
    im = ax_DO.imshow(ds.doxy.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='Purples', vmin=200., aspect='auto')
    #im = ax.imshow(ds.SO.data[0:max_depth, p_min:p_max],
    #    extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
    #               cmap='jet', vmin=0.9, aspect='auto')
    #cax,kw = mpl.colorbar.make_axes([ax_DO])
    cbaxes = plt.colorbar(im, orientation='horizontal', location='top')
    #cbaxes = plt.colorbar(im, pad=0., fraction=0.030)
    cbaxes.set_label('Dissolved Oxygen', fontsize=csz)

    # #########################################################3
    # SO contour
    SOs = ds.SO.data[0:max_depth, p_min:p_max]
    Np = p_max-p_min
    X = np.outer(np.ones(max_depth), 
        np.linspace(x_lims[0], x_lims[1], Np))
    Y = np.outer(np.linspace(0., max_depth*10., max_depth), 
                 np.ones(Np))

    #ax_DO.contour(X, Y, SOs, levels=[1., 1.1],
    #           colors=['white', 'black'], linewidths=1.5)
    ax_DO.contour(X, Y, SOs, levels=[1.],
               colors=['black'], linewidths=1.5)

    '''
    # #########################################################3
    # DO percentile contour
    in_view = (grid_tbl.profile >= p_min) & (grid_tbl.profile <= p_max) & (
        grid_tbl.depth < max_depth)
    doxy_p_grid = np.zeros_like(ds.N.data)
    for _, row in grid_tbl[in_view].iterrows():
        doxy_p_grid[row.depth, row.profile] = row.doxy_p
    ax_DO.contour(X, Y, doxy_p_grid[0:max_depth, p_min:p_max], 
                  levels=[90., 95.],
               colors=['white', 'black'], linewidths=1.5)
    '''

    # #########################################################3
    # N
    #ax_N = plt.subplot(gs[3:5])
    ax_N = fig.add_axes([0.37, ypos, axw, axh])
    all_ax.append(ax_N)
    im_N = ax_N.imshow(ds.N.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='Blues', aspect='auto')

    # N percentile
    grid_utils.find_perc(grid_tbl, 'N')
    Np_grid = np.zeros_like(ds.N.data)

    in_view = (grid_tbl.profile >= p_min) & (grid_tbl.profile <= p_max) & (
        grid_tbl.depth < max_depth)
    # Painful loop, but do it
    #embed(header='214 of figs_so')
    for _, row in grid_tbl[in_view].iterrows():
        Np_grid[row.depth, row.profile] = row.N_p

    #ax_N.contour(X, Y, Np_grid[0:max_depth, p_min:p_max], 
    #              levels=[90., 95.],
    #           colors=['gray', 'black'], linewidths=1.5)
    

    #cax,kw = mpl.colorbar.make_axes([ax_N])
    #cbaxes = plt.colorbar(im_N, cax=cax, **kw)
    cbaxes = plt.colorbar(im_N, orientation='horizontal', location='top')
    cbaxes.set_label('Buoyancy (cycles/hour)', fontsize=csz)

    # T
    #ax_T = plt.subplot(gs[6:])
    ax_T = fig.add_axes([0.70, ypos, axw, axh])
    all_ax.append(ax_T)
    im_T = ax_T.imshow(ds.temperature.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='jet', aspect='auto')
    #cax,kw = mpl.colorbar.make_axes([ax_T])
    cbaxes = plt.colorbar(im_T, orientation='horizontal', location='top')
    #cbaxes = plt.colorbar(im_T, cax=cax, **kw)
    cbaxes.set_label('Temperature (deg C)', fontsize=csz)

    '''
    # Chl
    ax_C = plt.subplot(gs[3])
    im_C = ax_C.imshow(ds.chlorophyll_a.data[0:max_depth, p_min:p_max],
        extent=[x_lims[0], x_lims[1], max_depth*10, 0.],
                   cmap='Greens', aspect='auto')
    cax,kw = mpl.colorbar.make_axes([ax_C])
    cbaxes = plt.colorbar(im_C, cax=cax, **kw)
    cbaxes.set_label('Chla', fontsize=csz)
    '''


    # Finish
    for ss, ax in enumerate(all_ax):
        #
        if ss == 0:
            ax.set_ylabel('Depth (m)')
        #ax.set_xlabel('Date')
        # Axes
        ax.xaxis_date()
        #major_format = mdates.DateFormatter('%b')
        #ax.xaxis.set_major_formatter(major_format)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))

        ax.tick_params(axis='x', rotation=20)
        #ax.autofmt_xdate()
        plotting.set_fontsize(ax, 14)

    #plt.tight_layout(h_pad=0.3, w_pad=10.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_percentiles(outfile:str, line:str, metric='N',
                    xlabel:str=None, ylabel:str=None):

    # Figure
    #sns.set()
    fig = plt.figure(figsize=(13,12))
    plt.clf()

    if metric == 'N':
        cmap = 'Blues'
    elif metric == 'chla':
        cmap = 'Greens'
    

    # Load
    items = load_up(line)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]

        #sns.histplot(data=grid_extrem, x='doxy_p', y='N_p', ax=ax)

    jg = sns.jointplot(data=grid_extrem, x='doxy_p', 
                    y=f'{metric}_p',
                    kind='hex', bins='log', # gridsize=250, #xscale='log',
                    # mincnt=1,
                    cmap=cmap,
                    marginal_kws=dict(fill=False, color='black', 
                                        bins=100)) 

    # Axes                                 
    if ylabel is not None:
        jg.ax_joint.set_ylabel(ylabel)
    else:
        jg.ax_joint.set_ylabel(f'{metric} Percentile')
    if xlabel is None:
        jg.ax_joint.set_xlabel('DO Percentile')
    else:
        jg.ax_joint.set_xlabel(xlabel)
    plotting.set_fontsize(jg.ax_joint, 14)

        # Submplit
        #mg0 = SeabornFig2Grid(jg, fig, gs[ss])
        #subfigs[ss] = jg
        #fig.add_subfigure(jg)

    
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_scatter_event(outfile:str, line:str, 
                      event:str, t_off):

    # Load
    items = cugn_io.load_up(line)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]

    # Grab event
    tevent = pandas.Timestamp(event)
    tmin = tevent - pandas.Timedelta(t_off)
    tmax = tevent + pandas.Timedelta(t_off)

    # In event
    in_event = (grid_extrem.time >= tmin) & (grid_extrem.time <= tmax)
    ds_in_event = (ds.time >= tmin) & (ds.time <= tmax)

    # Mission
    missions = np.unique(ds.mission[ds_in_event].data)
    mission_profiles = np.unique(ds.mission_profile[ds_in_event].data)
    print(f'Missions: {missions}')
    #print(f'Mission Profiles: {mission_profiles}')


    fig = plt.figure(figsize=(12,10))
    plt.clf()

    gs = gridspec.GridSpec(2,3)

    for ss, metric in enumerate(['SO', 'doxy', 'N', 'T', 'chla', 'lon']):
        ax = plt.subplot(gs[ss])

        if metric == 'T':
            ds_metric = 'temperature'
        elif metric == 'chla':
            ds_metric = 'chlorophyll_a'
        else:
            ds_metric = metric

        # Plot all
        srt = np.argsort(ds.time[ds_in_event].values)
        plt_depth = 0
        if metric in ['lon']:
            ax.plot(ds.time[ds_in_event][srt], ds[ds_metric][ds_in_event][srt], 'k-')
        else:
            ax.plot(ds.time[ds_in_event][srt], ds[ds_metric][plt_depth,ds_in_event][srt], 'k-')

        for depth, clr in zip(np.arange(3), ['b', 'g', 'r']):
            at_d = grid_extrem.depth[in_event] == depth
            ax.scatter(grid_extrem.time[in_event][at_d], grid_extrem[metric][in_event][at_d], color=clr)

        ax.set_ylabel(metric)

        plotting.set_fontsize(ax, 13.)

    #gs.tight_layout(fig)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_multi_z_event(outfile:str, line:str, 
                      event:str, t_off):

    # Load
    items = cugn_io.load_up(line)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]

    # Grab event
    tevent = pandas.Timestamp(event)
    tmin = tevent - pandas.Timedelta(t_off)
    tmax = tevent + pandas.Timedelta(t_off)

    # In event
    in_event = (grid_extrem.time >= tmin) & (grid_extrem.time <= tmax)
    ds_in_event = (ds.time >= tmin) & (ds.time <= tmax)

    # Mission
    missions = np.unique(ds.mission[ds_in_event].data)
    mission_profiles = np.unique(ds.mission_profile[ds_in_event].data)
    print(f'Missions: {missions}')
    #print(f'Mission Profiles: {mission_profiles}')


    fig = plt.figure(figsize=(17,10))
    plt.clf()

    gs = gridspec.GridSpec(3,5)

    cnt = 0
    for clr, z in zip(['b', 'g', 'r'], [10, 20, 30]):
        depth = z//10 - 1

        for ss, metric in enumerate(['doxy', 'T', 'N', 'chla', 
                                     'dist']):
            ax = plt.subplot(gs[cnt])

            if metric == 'T':
                ds_metric = 'temperature'
            elif metric == 'chla':
                ds_metric = 'chlorophyll_a'
            else:
                ds_metric = metric

            # Plot all
            srt = np.argsort(ds.time[ds_in_event].values)
            plt_depth = depth
            if metric in ['dist']:
                # Convert to distance
                dist, _ = cugn_utils.calc_dist_offset(
                    line, ds.lon[ds_in_event].values, 
                    ds.lat[ds_in_event].values)
                ax.plot(ds.time[ds_in_event][srt], 
                        #ds[ds_metric][ds_in_event][srt], 
                        dist[srt],
                        '-', color='gray', zorder=1)
            else:
                ax.plot(ds.time[ds_in_event][srt], 
                        ds[ds_metric][plt_depth,ds_in_event][srt], 
                        '-', color='gray', zorder=1)

            #for depth, clr in zip(np.arange(3), ['b', 'g', 'r']):
            at_d = grid_extrem.depth[in_event] == depth
            ax.scatter(grid_extrem.time[in_event][at_d], 
                       grid_extrem[metric][in_event][at_d], 
                       color=clr, zorder=10)


            # Axes
            ax.set_ylabel(ylbl_dict[metric])
            plotting.set_fontsize(ax, 13.)
            if z < 30:
                ax.set_xticklabels([])
            else:
                #plt.locator_params(axis='x', nbins=5)  # Show at most 5 ticks
                ax.tick_params(axis='x', rotation=45)


            cnt += 1

    #gs.tight_layout(fig)
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.3)
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
    mission_names = np.unique(ds.mission_name[missions].data)
    print(f'Missions: {missions}')
    print(f'Mission Names: {mission_names}')
    print(f'Profiles: {mission_profiles}')

    dist, _ = cugn_utils.calc_dist_offset(
                    line, ds.lon[ds_in_event].values, 
                    ds.lat[ds_in_event].values)
    imin = np.argmin(dist)

    # Range of dist
    print(f"Range of dist: {dist[0]} to {dist[-1]}")
    print(f"Minimum dist = {dist.min()} at {ds.time[ds_in_event][imin]}")

    fig = plt.figure(figsize=(10,8))
    plt.clf()


    cnt = 0
    #metrics = ['SO', 'doxy', 'N', 'CT', 'dist']
    metrics = ['SO', 'doxy', 'N', 'MLD', 'CT', 'chla']

    gs = gridspec.GridSpec(len(metrics),2)
    #metrics = ['SO', 'doxy', 'N', 'CT', 'SA']
    clrs = ['gray', 'purple', 'blue', 'orange', 'red', 'green']
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

            #if metric == 'N':
            #    embed(header='fig_multi_scatter_event 905')
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

def fig_dSO_dT():
    outfile = 'fig_dSO_dT.png'

    # Load (just for lon, lat)
    line = '90'
    items = cugn_io.load_line(line)
    grid_tbl = items['grid_tbl']
    ds = items['ds']
    z=10. # m

    SA = 33.7  # a canonical value
    lat = np.nanmedian(ds.lat.data)
    lon = np.nanmedian(ds.lon.data)
    p = conversions.p_from_z(-z, lat)
    DO = 260.

    # Interpolators
    CTs = np.linspace(12., 20., 100)
    OCs = gsw.O2sol(SA, CTs, p, lon, lat)

    f_T_OC = interp1d(CTs, OCs)
    f_OC_T = interp1d(OCs, CTs)

    DO_SO1 = OCs
    DO_SO105 = 1.05 * OCs

    dOC_dT = np.gradient(DO_SO1, CTs[1]-CTs[0])
    dSO_dT = -1*DO_SO1 / OCs**2 * dOC_dT

    dSO105_dT = -1*DO_SO105 / OCs**2 * dOC_dT

    fig = plt.figure(figsize=(12,10))
    plt.clf()
    ax = plt.gca()

    ax.plot(CTs, dSO_dT, 'k-', label='SO=1')
    ax.plot(CTs, dSO105_dT, 'b-', label='SO=1.05')

    ax.set_ylim(0., 0.025)

    # Label
    ax.set_xlabel('Temperature (deg C)')
    ax.set_ylabel('dSO/dT (1/deg C)')

    fsz = 21.
    plotting.set_fontsize(ax, fsz)

    ax.legend(fontsize=fsz)

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_T_fluctuations(outfile:str, line:str, debug:bool=False):


    # Load
    items = load_up(line)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]

    # Grab event
    current_time = pandas.Timestamp('2000-01-01')
    later = grid_extrem.time > current_time
    ds_mjd = pandas.to_datetime(ds.time.data).to_julian_date()

    rms_T = []
    while(np.sum(later) > 0):
        # Find the next one
        next_idx = np.where(later)[0][0]
        event = grid_extrem.iloc[next_idx]
        tevent = event.time

        # Mission number too
        mn = ds.mission.data[event.profile]

        # t interval
        tmin = tevent - pandas.Timedelta('5D')
        tmax = tevent + pandas.Timedelta('20D')

        in_event = ((ds.time >= tmin) & (ds.time <= tmax) & (
           ds.mission == mn)).data

        # Fit a line
        in_times = ds_mjd[in_event] - ds_mjd[in_event][0]
        Ts = ds.temperature[0, in_event].data
        # Keep the good ones
        good = np.isfinite(Ts)
        in_times = in_times[good]
        Ts = Ts[good]
        fit = np.polyfit(in_times, Ts, 1)
        #except:
        #    embed(header='fig_T_fluctuations')
        if debug:
            plt.clf()
            ax = plt.gca()
            ax.scatter(in_times, Ts)
            ax.plot(in_times, fit[1] + fit[0]*in_times, 'k-')
            plt.show()
        # RMS
        res = Ts - (fit[1] + fit[0]*in_times)
        rms_T.append(np.sqrt(np.mean(res**2)))

        # Update
        later = grid_extrem.time > tmax


    fig = plt.figure(figsize=(12,10))
    plt.clf()
    ax = plt.gca()

    sns.histplot(rms_T, ax=ax, bins=15)

    #ax.set_ylim(0., 0.025)

    # Label
    ax.set_xlabel('RMS Temperature (deg C)')
    #ax.set_ylabel('dSO/dT (1/deg C)')

    fsz = 21.
    plotting.set_fontsize(ax, fsz)

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_joint_pdfs(line:str):

    def gen_cb(img, lbl, csz = 17.):
        cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
        cbaxes.set_label(lbl, fontsize=csz)
        cbaxes.ax.tick_params(labelsize=csz)

    outfile = f'fig_jointPDFs_{line}.png'

    # Load
    items = cugn_io.load_line(line)
    ds = items['ds']

    # Grids

    # Oxygen
    mean_oxy, xedges, yedges, counts, grid_indices, _, _ = grid_utils.gen_grid(
        ds, axes=('SA', 'CT'), stat='mean', variable='doxy')

    # PDF
    dSA = xedges[1] - xedges[0]
    dsigma = yedges[1] - yedges[0]

    p_norm = np.sum(counts) * (dSA * dsigma)
    consv_pdf = counts / p_norm
    #embed(header='764 of figs_so')

    # z
    mean_z, xedges, yedges, counts, grid_indices, _, _ = grid_utils.gen_grid(
        ds, axes=('SA', 'CT'), stat='mean', variable='depth')

    # chlorophyll
    mean_chl, xedges, yedges, counts, grid_indices, _, _ = grid_utils.gen_grid(
        ds, axes=('SA', 'CT'), stat='nanmean', variable='chlorophyll_a')

    fig = plt.figure(figsize=(12,10))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    # #####################################################
    # PDF
    ax_pdf = plt.subplot(gs[0])
    img = ax_pdf.pcolormesh(xedges, yedges, np.log10(consv_pdf.T), 
                            cmap='Blues')
    gen_cb(img, r'$\log_{10} \, p(S_A,\sigma)$')

    # #####################################################
    # DO
    ax_DO = plt.subplot(gs[1])
    img = ax_DO.pcolormesh(xedges, yedges, mean_oxy.T, cmap='Purples')
    gen_cb(img, 'DO (umol/kg)')

    # #####################################################
    # z
    ax_z = plt.subplot(gs[2])
    img = ax_z.pcolormesh(xedges, yedges, mean_z.T, cmap='inferno_r')
    gen_cb(img, 'z (m)')

    # #####################################################
    # z
    ax_chla = plt.subplot(gs[3])
    img = ax_chla.pcolormesh(xedges, yedges, 
                             np.log10(mean_chl.T), 
                             cmap='Greens')
    gen_cb(img, r'$\log_{10}$ Chla (mg/m^3)')

    # ##########################################################

    fsz = 17.
    for ax in [ax_pdf, ax_DO, ax_z, ax_chla]:
        ax.set_xlabel('Absolute Salinity (g/kg)')                    
        ax.set_ylabel('Conservative Temperature (C)')
        # Set x-axis interval to 0.5
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        # 
        plotting.set_fontsize(ax, fsz)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_joint_pdf(line:str, xvar:str, yvar:str):

    def gen_cb(img, lbl, csz = 17.):
        cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
        cbaxes.set_label(lbl, fontsize=csz)
        cbaxes.ax.tick_params(labelsize=csz)

    outfile = f'fig_jointPDF_{line}_{xvar}_{yvar}.png'

    # Load
    items = cugn_io.load_line(line)
    ds = items['ds']

    # PDF
    _, xedges, yedges, counts, grid_indices, _, _ = grid_utils.gen_grid(
        ds, axes=(xvar, yvar), stat='mean', variable='doxy')

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
                            cmap='autumn')
    gen_cb(img, r'$\log_{10} \, p('+f'{xvar},{yvar})$')

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


def fig_avgSO_zd(line:str):


    outfile = f'fig_avgSO_zd_{line}.png'

    # Load
    items = cugn_io.load_line(line)
    ds = items['ds']

    # Grid
    zbins= np.linspace(0., 500, 50)
    dbins = np.linspace(0., 500., 50)

    xdata = np.outer(ds.depth.data, np.ones_like(ds.profile))

    dist, _ = cugn_utils.calc_dist_offset(
        line, ds.lon.values, ds.lat.values)
    ydata = np.outer(np.ones_like(ds.depth.data), dist)

    gd = np.isfinite(ds.SO.data)

    measure, xedges, yedges, grid_indices =\
            stats.binned_statistic_2d(
                xdata[gd].astype(float),
                ydata[gd],
                ds.SO.data[gd],
                statistic='mean',
                bins=[zbins, dbins],
                expand_binnumbers=True)



    fig = plt.figure(figsize=(12,10))
    plt.clf()
    ax = plt.gca()

    # #####################################################
    # PDF
    img = ax.pcolormesh(xedges, yedges, measure,
                            cmap='Reds')
    gen_cb(img, 'Mean SO') 

    # ##########################################################
    tsz = 19.
    ax.text(0.05, 0.1, f'Line={line}',
                transform=ax.transAxes,
                fontsize=tsz, ha='left', color='k')

    fsz = 17.
    ax.set_xlabel('Offshore Distance (km)')
    ax.set_ylabel('z (m)')
    ax.invert_yaxis()
    ax.invert_xaxis()
    # Set x-axis interval to 0.5
    #ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # 
    plotting.set_fontsize(ax, fsz)
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_absolute(outfile:str, line:str, metric='N',
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
    elif metric == 'SA':
        cmap = 'Greys'
    

    # Load
    items = load_up(line)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]

    # Cut on depth
    grid_tbl = grid_tbl[grid_tbl.depth <= (max_depth//10 - 1)]

        #sns.histplot(data=grid_extrem, x='doxy_p', y='N_p', ax=ax)

    jg = sns.jointplot(data=grid_tbl, x='doxy', 
                    y=f'{metric}',
                    kind='hex', bins='log', # gridsize=250, #xscale='log',
                    # mincnt=1,
                    cmap=cmap,
                    marginal_kws=dict(fill=False, color='black', 
                                        bins=100)) 

    # Axes                                 
    jg.ax_joint.set_ylabel(f'{metric}')
    jg.ax_joint.set_xlabel('DO')
    plotting.set_fontsize(jg.ax_joint, 14)

    # Extrema
    jg.ax_joint.plot(grid_extrem.doxy, grid_extrem[metric], 
                     'ro', ms=1)
    jg.ax_joint.text(0.95, 0.05, f'depth <= {max_depth}m',
                transform=jg.ax_joint.transAxes,
                fontsize=14., ha='right', color='k')
    jg.ax_joint.text(0.05, 0.95, f'Line: {line}',
                transform=jg.ax_joint.transAxes,
                fontsize=14., ha='left', color='k')

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")



def fig_lowSO_jointDO(outfile:str, line:str='56.0', 
              metric='N', max_depth:int=10,
              lowSO_cut:float=0.8):

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
    elif metric == 'SA':
        cmap = 'Greys'
    

    # Load
    items = load_up(line)
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]

    # Cut on depth
    grid_tbl = grid_tbl[grid_tbl.depth <= (max_depth//10 - 1)]

    # Low SO
    lowSO = grid_tbl.SO < lowSO_cut
    grid_extrem = grid_tbl[lowSO].copy()

    jg = sns.jointplot(data=grid_tbl, x='doxy', 
                    y=f'{metric}',
                    kind='hex', bins='log', # gridsize=250, #xscale='log',
                    # mincnt=1,
                    cmap=cmap,
                    marginal_kws=dict(fill=False, color='black', 
                                        bins=100)) 

    # Axes                                 
    jg.ax_joint.set_ylabel(f'{metric}')
    jg.ax_joint.set_xlabel('DO')
    plotting.set_fontsize(jg.ax_joint, 14)

    # Extrema
    jg.ax_joint.plot(grid_extrem.doxy, grid_extrem[metric], 
                     'ro', ms=1, label=f'SO < {lowSO_cut}')
    jg.ax_joint.text(0.95, 0.05, f'depth <= {max_depth}m',
                transform=jg.ax_joint.transAxes,
                fontsize=14., ha='right', color='k')
    jg.ax_joint.text(0.05, 0.95, f'Line: {line}',
                transform=jg.ax_joint.transAxes,
                fontsize=14., ha='left', color='k')
    jg.ax_joint.legend(loc='upper right', fontsize=14.)

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_upwell_search(line:str, gextrem:str='high',
                      minDT:int=20):

    outfile = f'fig_upwell_search_{line}.png'
    # Load
    items = cugn_io.load_up(line, gextrem=gextrem)
    grid_extrem = items[0]
    ds = items[1]
    times = items[2]
    grid_tbl = items[3]

    # Loop on the clusters 
    uni_clusters = np.unique(grid_extrem.cluster)

    dTs_ex = []  # Delta T for the extrema
    dTs_sat = [] # Delta T for the saturated parels (not extreme)

    for cluster in uni_clusters:
        if cluster == -1:
            continue

        in_cluster = grid_extrem.cluster == cluster
        g_cluster = grid_extrem[in_cluster].copy()

        tmin = g_cluster.time.min()
        tmax = g_cluster.time.max()
        dt = (tmax - tmin).days

        # Extend?
        if dt < minDT:
            nadd = (minDT - dt) / 2 + 1
            tmin = tmin - pandas.Timedelta(f'{nadd}D')
            tmax = tmax + pandas.Timedelta(f'{nadd}D')

        # Unique depths
        uni_depths = np.unique(g_cluster.depth)
        for depth in uni_depths:
            ds_in_event = (ds.time >= tmin) & (ds.time <= tmax) 

            in_depth = g_cluster.depth == depth
            if np.sum(in_depth) < 2:
                continue
            g_depth = g_cluster[in_depth]

            # Calcualte the temperature
            Ts = ds['CT'][depth, ds_in_event].values
            # Reject NaNs
            median_Ts = np.nanmedian(Ts)

            dT = g_depth.CT.values - median_Ts
            dTs_ex += list(dT)

            # Saturation
            SOs = ds['SO'][depth, ds_in_event]
            sat = (SOs > 1.0) & (SOs < 1.1)
            dT = Ts - median_Ts
            dTs_sat += list(dT[sat])

    # Histogram me
    fig = plt.figure(figsize=(12,10))
    plt.clf()
    ax = plt.gca()

    #embed(header='fig_upwell_search 1571')
    sns.histplot(dTs_ex, ax=ax, bins=20, color='r', 
                 fill=False, edgecolor='r', label='Extrema')
    sns.histplot(dTs_sat, ax=ax, bins=20, color='b',
                    fill=False, edgecolor='b', label='Saturated')

    ax.set_xlabel('Temperature Anomaly (deg C)')
    ax.set_ylabel('Count')

    fsz = 17.
    plotting.set_fontsize(ax, fsz)

    ax.legend(fontsize=fsz)

    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_sodo_anomalies(line:str, zmax:int=9, 
                       dmax:float=100.,
                       variable:str='doxy'):
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

    outfile = f'fig_{variable}_anomalies_{line}_{int(dmax)}_z{10*(zmax+1)}.png' 
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

        # Add a cDF in an inset
        axin = ax.inset_axes([0.65, 0.6, 0.3, 0.3])

        sorted_data = np.sort(anom)

        # Calculate the proportional values of samples
        y = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        # Gaussian
        mean = np.mean(anom)
        std = np.std(anom)
        skew = stats.skew(anom)

        # Create the CDF plot
        axin.plot(sorted_data, y, color=clr)

        # Overplot the Gaussian
        x = np.linspace(-5*std, 5*std, 10000)
        y = stats.norm.cdf(x/std)
        axin.plot(x, y, 'k:')

        axin.set_xlabel('Value')
        axin.set_ylabel('CDF')

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
                lbl+'\n\n'+f'N={len(anom)}\n'+f'$\mu$={mean:.1f}\n$\sigma$={std:.1f}\nskew={skew:.2f}',
                transform=ax.transAxes, 
                fontsize=18., ha='left', color='k',
                va='top')

        fsz = 19.
        plotting.set_fontsize(ax, fsz)

        #ax.legend(fontsize=fsz)

    # Title
    fig.suptitle(f'Line={line}, zmax={10*(zmax+1)}m, dmax={dmax}km', 
                 fontsize=29)


    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_cluster_char(outfile='fig_cluster_char.png', use_full:bool=False):

    day_ns = 24 * 60 * 60 * 1_000_000_000
    
    # Figure
    fig = plt.figure(figsize=(6,6))
    plt.clf()
    gs = gridspec.GridSpec(4,2)

    for tt, clr, line in zip(np.arange(4), line_colors, lines):

        # Load
        items = cugn_io.load_up(line, use_full=use_full)#, skip_dist=True)
        grid_extrem = items[0]
        ds = items[1]
        times = items[2]
        grid_tbl = items[3]

        cluster_stats = clusters.cluster_stats(grid_extrem)
        dur_days = cluster_stats.Dtime.values.astype('float') / day_ns
        #embed(header='fig_cluster_char 1622')

        # Duration, size
        for ss in range(2):
            ax = plt.subplot(gs[tt,ss])

            # Duration
            if ss == 0:
                sns.histplot(dur_days, ax=ax, color=clr,
                             binrange=[0.,15.],
                             binwidth=1.)
            else: # size
                sns.histplot(cluster_stats['Ddist'].values, ax=ax, color=clr,
                             binrange=[0.,150.],
                             binwidth=10.)
                             #binrange=[0.,15.],
                             #binwidth=1.)
                print(f'Line={line}, max={cluster_stats.Ddist.max()}')
            
            if tt < 3:
                ax.set_xticklabels([])
                ax.tick_params(bottom=False)
            else:
                ax.set_xlabel(['Duration (days)', 'Size (km)'][ss])
            # Font size
            plotting.set_fontsize(ax, 15.)
    
    plt.tight_layout(pad=0.5)#, h_pad=0.1, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_cluster_size_vs_age(outfile='fig_cluster_size_vs_age.png', use_full:bool=False):

    day_ns = 24 * 60 * 60 * 1_000_000_000
    
    # Figure
    fig = plt.figure(figsize=(6,6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])

    markers = ['o', 's', 'd', '^']
    for tt, clr, line in zip(np.arange(4), line_colors, lines):

        # Load
        items = cugn_io.load_up(line, use_full=use_full)#, skip_dist=True)
        grid_extrem = items[0]
        ds = items[1]
        times = items[2]
        grid_tbl = items[3]

        cluster_stats = clusters.cluster_stats(grid_extrem)
        dur_days = cluster_stats.Dtime.values.astype('float') / day_ns
        #embed(header='fig_cluster_char 1622')

        # Duration, size
        #embed(header='1689 of figs')
        ax.scatter(dur_days, cluster_stats['Ddist'].values, 
                marker=markers[tt], edgecolor=clr, 
                facecolor='None',
                s=cluster_stats['N'].values/2., 
                label=f'Line {line}')

    plotting.set_fontsize(ax, 17.)
    ax.legend(fontsize=15.)

    # Labels
    ax.set_xlabel('Duration (days)')
    ax.set_ylabel('Size (km)')
    
    plt.tight_layout(pad=0.5)#, h_pad=0.1, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_cluster_date_vs_loc(outfile='fig_cluster_date_vs_loc.png', 
                            use_full:bool=False, debug:bool=False):

    day_ns = 24 * 60 * 60 * 1_000_000_000
    
    # Figure
    fig = plt.figure(figsize=(6,6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])

    markers = ['o', 's', 'd', '^']
    for tt, clr, line in zip(np.arange(4), line_colors, lines):
        if tt != 0 and debug:
            continue

        # Load
        items = cugn_io.load_up(line, use_full=use_full)#, skip_dist=True)
        grid_extrem = items[0]
        ds = items[1]
        times = items[2]
        grid_tbl = items[3]

        cluster_stats = clusters.cluster_stats(grid_extrem)
        dur_days = cluster_stats.Dtime.values.astype('float') / day_ns
        #embed(header='fig_cluster_char 1622')

        # Duration, size
        #embed(header='1689 of figs')
        ax.scatter(cluster_stats['Cdist'].values, 
                   cluster_stats['time'].values, 
                marker=markers[tt], edgecolor=clr, 
                facecolor='None',
                s=0.01, #cluster_stats['Ddist'].values, 
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

    #ax.legend(fontsize=15.)

    # Labels
    ax.set_xlabel('Distance Offshore (km)')
    ax.set_ylabel('Date')
    
    plt.tight_layout(pad=0.5)#, h_pad=0.1, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_joint_pdf_MLDSO(line:str, iz:int=0):

    def gen_cb(img, lbl, csz = 17.):
        cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
        cbaxes.set_label(lbl, fontsize=csz)
        cbaxes.ax.tick_params(labelsize=csz)

    xvar = 'SO'
    yvar = 'MLD'
    outfile = f'fig_jointPDF_{line}_SO_MLD.png'

    # Load
    items = cugn_io.load_line(line)
    ds = items['ds']

    #
    bins_SO = np.linspace(0.8, 1.3, 50)
    bins_MLD = np.linspace(0, 100, 50)

    gd = np.isfinite(ds.SO.data[iz,:]) & np.isfinite(ds.MLD.data)

    # Counts
    counts, xedges, yedges = np.histogram2d(
                ds.SO.data[iz,gd], 
                ds.MLD.data[gd],
                bins=[bins_SO, bins_MLD])

    # PDF
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    p_norm = np.sum(counts) * (dx * dy)
    consv_pdf = counts / p_norm


    fig = plt.figure(figsize=(12,10))
    plt.clf()
    ax = plt.gca()

    # #####################################################
    # PDF
    img = ax.pcolormesh(xedges, yedges, np.log10(consv_pdf.T), 
                            cmap='Greens')
    gen_cb(img, r'$\log_{10} \, p('+f'{xvar},{yvar})$',
           csz=19.)

    # ##########################################################
    tsz = 25.
    ax.text(0.05, 0.9, f'Line: {line}',
                transform=ax.transAxes,
                fontsize=tsz, ha='left', color='k')
    ax.text(0.05, 0.8, f'z = {(iz+1)*10}m',
                transform=ax.transAxes,
                fontsize=tsz, ha='left', color='k')

    ax.set_xlabel(ylbl_dict[xvar])
    ax.set_ylabel(ylbl_dict[yvar])

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

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)


    # PDF CDFs
    if flg & (2**0):
        line = '90'
        #fig_pdf_cdf(f'fig_pdf_cdf_{line}.png', line)
        fig_pdf_cdf(f'fig_pdf_cdf_{line}_105.png', line, SO_cut=1.05)

    # PDF CDF vary SO_cut
    if flg & (2**1):
        line = '90'
        fig_varySO_pdf_cdf(f'fig_varySO_pdf_cdf_{line}.png', line)

    # Time-series
    if flg & (2**2):
        line = '66'
        line = '90'
        line = '80'
        fig_timeseries(f'fig_timeseries_{line}.png', line)

    # Events
    if flg & (2**3):
        line = '90.0'
        eventA = ('2020-09-01', '2W') # Sub-surface
        eventB = ('2019-08-15', '3W')
        eventC = ('2019-03-02', '1W') # Spring
        eventD = ('2021-08-10', '3W') # Surface but abrupt start

        #line = '80'
        #eventA = ('2020-08-11', '1W') # 
        #eventB = ('2022-02-15', '2W') # 


        #for event in [eventA, eventB, eventC]:
        for event, t_off in [eventA]:
            fig_event(f'fig_event_{line}_{event}.png', line, event, t_off)

    # Percentiles of DO and N
    if flg & (2**4):
        #line = '80'
        line = '90.0'
        metric = 'N'
        #metric = 'chla'
        fig_percentiles(f'fig_percentiles_{line}_{metric}.png', 
                        line, metric=metric)


    # SO CDF
    if flg & (2**5):
        pass

    # dist vs DOY
    if flg & (2**6):
        line = '90'
        line = '80'
        fig_dist_doy(f'fig_dist_doy_{line}.png', line)

    # Scatter event
    if flg & (2**7):
        line = '90.0'
        eventA = ('2020-09-01', '10D') # Sub-surface
        eventB = ('2019-08-15', '3W') # Surface but abrupt start
        eventC = ('2019-03-02', '1W') # Spring
        eventD = ('2021-08-13', '15D') # Intense, near shore episode

        # Bad
        eventN = ('2020-05-10', '2W') # 
        eventO = ('2022-03-01', '2W') # 

        line = '80.0'
        #eventA = ('2020-08-11', '1W') # NO GOOD 
        #eventB = ('2022-02-15', '10D') # 
        eventC = ('2022-06-25', '10D') # 

        event, t_off = eventC
        # Original
        #fig_scatter_event(f'fig_scatter_event_{line}_{event}.png', 
        #             line, event, t_off)
        # Variation #2
        #fig_multi_z_event(f'fig_multi_z_event_{line}_{event}.png', 
        #             line, event, t_off)
        # Variation #3
        fig_multi_scatter_event(
            f'fig_multi_scatter_event_{line}_{event}.png', 
            line, event, t_off)#, gextrem='hi_noperc')

    # Scatter event
    if flg & (2**8):
        fig_dSO_dT()

    # T fluctuations
    if flg & (2**9):
        line = '90'
        fig_T_fluctuations(f'fig_T_fluctuations_{line}.png', line)

    # Joint PDFs
    if flg & (2**10):
        line = '90'
        fig_joint_pdfs(line)

    # Joint PDF
    if flg & (2**11):
        line = '90'
        #line = '80'
        fig_joint_pdf(line, 'SO', 'N')

    # <SO>(z,d)
    if flg & (2**12):
        line = '90'
        fig_avgSO_zd(line)
        
    # Absolute N, DO, Chl
    if flg & (2**13):
        line = '90.0'
        #line = '80.0'
        #metric = 'chla'
        metric = 'SA'
        metric = 'N'
        metric = 'T'
        fig_absolute(f'fig_absolute_{line}_{metric}.png', 
                        line, metric=metric)

    # Relative N, DO, Chl
    #if flg & (2**14):
    #    line = '80.0'
    #    line = '90.0'
    #    #metric = 'chla'
    #    #metric = 'N'
    #    metric = 'T'
    #    fig_relative(f'fig_relative_{line}_{metric}.png', 
    #                    line, metric=metric)

    # Line 56
    if flg & (2**15):
        line = '56.0'
        #line = '90.0'
        #metric = 'chla'
        #metric = 'N'
        metric = 'T'
        fig_lowSO_jointDO(f'fig_lowSO_{line}_{metric}.png', 
                        line, metric=metric)


    # Joint PDF: N, SO on Line 90
    if flg & (2**17):
        fig_joint_line90(outfile='fig_joint_NSO_line90.png',
                         metric='N', xmetric='SO')



    # Scatter events for Sub-SO
    if flg & (2**19):
        line = '56.0'
        eventA = ('2020-09-12', '2W') 
        eventB = ('2021-05-20', '2W') 

        event, t_off = eventB

        # Original
        #fig_scatter_event(f'fig_scatter_event_{line}_{event}.png', 
        #             line, event, t_off)
        # Variation #2
        #fig_multi_z_event(f'fig_multi_z_event_{line}_{event}.png', 
        #             line, event, t_off)
        # Variation #2
        fig_multi_scatter_event(
            f'fig_multi_scatter_event_low_{line}_{event}.png', 
            line, event, t_off, gextrem='low_noperc')


    # Joint PDF: T, DO on Line 90
    if flg & (2**31):
        #line = '90.0'
        line = '90.0'
        eventA = ('2020-09-01', '2W') # Sub-surface
        event, t_off = eventA
        fig_pivot_event('fig_pivot_event.png', 
                        line, event, t_off)

    if flg & (2**32):
        # For Pivot
        line = '90.0'
        metric = 'N'
        fig_percentiles(f'fig_pivot_percentiles_{line}_{metric}.png', 
                        line, metric=metric,
                        xlabel=r'percentile(DO$_i$ | $T_i,S_i$)',
                        ylabel=r'percentile(Buoyancy$_i$ | $T_i,S_i$)')

    # Upwelling search
    if flg & (2**33):
        #line = '90.0'
        line = '90.0'
        fig_upwell_search(line, gextrem='high')

    # Upwelling search
    if flg & (2**34):
        for zmax in [4, 9]:
            for variable in ['SO', 'doxy']:
                for line in lines:
                    fig_sodo_anomalies(
                        line, dmax=50., variable=variable,
                        zmax=zmax)

    # Cluster char
    if flg & (2**35):
        fig_cluster_char()

    # Cluster size vs age
    if flg & (2**36):
        fig_cluster_size_vs_age()

    # Cluster date vs location/size
    if flg & (2**37):
        fig_cluster_date_vs_loc()#debug=True)

    if flg & (2**38):
        line = '80' # '90'
        fig_joint_pdf_MLDSO(line)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- PDF CDF
        #flg += 2 ** 1  # 2 -- Vary SO cut
        #flg += 2 ** 2  # 4 -- Interannual time-series of outliers
        #flg += 2 ** 3  # 8 -- Show individual events
        #flg += 2 ** 4  # 16 -- Percentiles
        #flg += 2 ** 5  # 32 -- 
        #flg += 2 ** 6  # 64 -- dist vs DOY

        flg += 2 ** 7  # 128 -- scatter event
        #flg += 2 ** 8  # 256 -- dSO/dT
        #flg += 2 ** 9  # 512 -- T fluctuations
        #flg += 2 ** 10  # 1024 -- joint PDFs
        #flg += 2 ** 11  # 2048 -- joint PDF of X,Y
        #flg += 2 ** 12  # 4096 -- SO(z,d)
        #flg += 2 ** 13  # 8192 -- Absolute N, DO, Chl, T
        #flg += 2 ** 14  # -- Relative to interannual
        #flg += 2 ** 15  # Low SO, joint DO
        #flg += 2 ** 16  # Joint PDF: T, DO on Line 90
        #flg += 2 ** 17  # Joint PDF: N, SO on Line 90, z<=30m

        #flg += 2 ** 18  # N CDF
        #flg += 2 ** 19  # Sub-SO events


        #flg += 2 ** 31  # Pivot event figure
        #flg += 2 ** 32  # Pivot percentile
        #flg += 2 ** 33  # Search for upwelling

        # Anamolies
        #flg += 2 ** 34  # DO

        # Clusters
        #flg += 2 ** 35  # clusters
        #flg += 2 ** 36  # size vs age
        #flg += 2 ** 37  # date vs location/size

        # MLD
        #flg += 2 ** 38  # date vs location/size

    else:
        flg = sys.argv[1]

    main(flg)