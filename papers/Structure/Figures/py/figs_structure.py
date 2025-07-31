""" Figures for the structure function paper. """


# imports
import os
import sys
import glob
from importlib import resources

import numpy as np
import xarray

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec

import seaborn as sns

from ocpy.utils import plotting

from profiler import gliderdata
from profiler import profilerpairs
from cugn import io as cugn_io
from cugn import utils as cugn_utils
from cugn import plotting as cugn_plotting

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import qg_utils
import data_utils
import glider_io

Sn_lbls = cugn_plotting.Sn_lbls

def fig_separations(dataset:str, outroot='fig_sep', max_time:float=10.):
    outfile = f'{outroot}_{dataset}.png'

    # Load dataset
    profilers = glider_io.load_dataset(dataset)
    

    # Generate pairs
    #gPairs = gliderpairs.GliderPairs(gData, max_time=max_time)
    gPairs = profilerpairs.ProfilerPairs(profilers, 
                                          max_time=max_time,
                                          debug=False,
                                          randomize=True)


    # Start the figure
    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)


    # Lat/lon
    ax_ll = plt.subplot(gs[0])

    for mid in np.unique(gPairs.data('missida',2).astype(int)):
        marker = '+'
        idx = gPairs.data('missida',2) == mid
        ax_ll.scatter(gPairs.data('lon', 2)[idx], 
            gPairs.data('lat', 2)[idx], s=2, label=f'MID={mid}',
            marker=marker)

    ax_ll.set_xlabel('Longitude [deg]')
    ax_ll.set_ylabel('Latitude [deg]')
    ax_ll.legend(fontsize=4, ncol=2,
                 loc='upper left')

    ax_ll.grid()
    ax_ll.xaxis.set_major_locator(MultipleLocator(1.0))  # Major ticks every 2 units


    # Separations
    ax_r = plt.subplot(gs[1])
    _ = sns.histplot(gPairs.r, bins=50, log_scale=True, ax=ax_r)
    # Label
    ax_r.set_xlabel('Separation [km]')
    ax_r.set_ylabel('Count')

    # Add dataset
    lsz = 16.
    ax_r.text(0.1, 0.9, dataset, transform=ax_r.transAxes, fontsize=lsz)
    # Label time separation
    ax_r.text(0.1, 0.8, f't < {max_time} hours', transform=ax_r.transAxes, fontsize=15)

    for ax in [ax_ll, ax_r]:
        plotting.set_fontsize(ax, 15) 
        
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_dtimes(dataset:str, outroot='fig_dtime', max_time:float=10.):
    outfile = f'{outroot}_{dataset}.png'

    # Load dataset
    gData = glider_io.load_dataset(dataset)
    
    # Cut on valid velocity data 
    gData = gData.cut_on_good_velocity()

    # Generate pairs
    gPairs = gliderpairs.GliderPairs(gData, max_time=max_time)

    # Start the figure
    fig = plt.figure(figsize=(8,8))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lat/lon
    ax_t = plt.subplot(gs[0])

    _ = sns.histplot(x=gPairs.dtime, bins=50, log_scale=True, ax=ax_t,
                     color='green')
    # Label
    ax_t.set_xlabel(r'$\Delta t$ [hr]')
    ax_t.set_ylabel('Count')

    # Add dataset
    lsz = 18.
    ax_t.text(0.1, 0.9, dataset, transform=ax_t.transAxes, fontsize=lsz)
    # Label time separation
    #ax_t.text(0.1, 0.8, f't < {max_time} hours', transform=ax_r.transAxes, fontsize=15)

    # Log scale
    ax_t.set_yscale('log')

    plotting.set_fontsize(ax_t, 15) 
        
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_dus(dataset:str, outroot='fig_du', max_time:float=10., iz:int=4):
    # Outfile
    outfile = f'{outroot}_z{(iz+1)*10}_{dataset}.png'

    # Load dataset
    gData = glider_io.load_dataset(dataset)
    
    # Cut on valid velocity data 
    gData = gData.cut_on_good_velocity()

    # Generate pairs
    gPairs = gliderpairs.GliderPairs(gData, max_time=max_time)

    # Velocity
    #gPairs.calc_velocity(iz)
    gPairs.calc_delta(iz, '')

    # Start the figure
    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # |u|
    ax_umag = plt.subplot(gs[0])

    _ = sns.histplot(x=gPairs.umag, bins=50, log_scale=True, ax=ax_umag,
                     color='red')
    # Label
    ax_umag.set_xlabel(r'$|u|$ [m/s]')
    ax_umag.set_ylabel('Count')

    # Add dataset
    lsz = 18.
    ax_umag.text(0.1, 0.9, dataset, transform=ax_umag.transAxes, fontsize=lsz)

    # |u|
    ax_duL = plt.subplot(gs[1])

    _ = sns.histplot(x=gPairs.duL, bins=50, ax=ax_duL, color='purple')
    # Label
    ax_duL.set_xlabel(r'$\delta u_L$ [m/s]')
    ax_duL.set_ylabel('Count')

    # Label time separation
    ax_duL.text(0.1, 0.8, f'depth = {(iz+1)*10} m', 
                transform=ax_duL.transAxes, fontsize=15, ha='left')
    # Log scale
    #ax_umag.set_yscale('log')

    for ax in [ax_umag, ax_duL]:
        plotting.set_fontsize(ax, 15) 
        
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_structure(dataset:str, outroot='fig_structure',
                  variables = 'duLduLduL',
                  iz:int=5, tcut:tuple=None,
                  skip_vel:bool=False,
                  stretch:bool=False,
                  gpair_file:str=None,
                  use_xlim:tuple=None,
                  use_ylim:tuple=None,
                  minN:int=10, avoid_same_glider:bool=True,
                  show_correct:bool=True):

    # Set in_field=True to load in-field data
    #kwargs = {}
    #if variables in ['duLduLduL']:
    #    kwargs['in_field'] = True
    #    kwargs['adcp_on'] = True
    #    skip_vel = False

    # Load dataset
    profilers = glider_io.load_dataset(dataset)

    # Outfile
    if iz >= 0:
        outfile = f'{outroot}_z{(iz+1)*10}_{dataset}_{variables}.png'
    else:
        outfile = f'{outroot}_iso{np.abs(iz)}_{dataset}_{variables}.png'
    if stretch:
        outfile = outfile.replace('.png', '_stretch.png')

    # Load
    #gpair_file = cugn_io.gpair_filename(
    #    dataset, iz, not avoid_same_glider)
    #gpair_file = os.path.join('..', 'Analysis', 'Outputs', gpair_file)

    if gpair_file is not None:
        Sn_dict = gliderpairs.load_Sndict(gpair_file)
        print(f'Loaded: {gpair_file}')
    else:
        if variables != 'duLduLduL':
            raise NotImplementedError('Not ready for these variablaes')
        # Cut on valid velocity data 
        nbins = 20
        rbins = 10**np.linspace(0., np.log10(400), nbins) # km
        # Generate pairs
        #gData = gliderdata.load_dataset(dataset)
        gPairs = profilerpairs.ProfilerPairs(
            profilers, max_time=10.,
            avoid_same_glider=avoid_same_glider,
            remove_nans=True,
            debug=False, 
            randomize=False)
        # Isopycnals?
        if iz < 0:
            gPairs.prep_isopycnals('t')
        #gData = gData.cut_on_good_velocity()
        #gData = gData.cut_on_reltime(tcut)

        gPairs.calc_delta(iz, variables, skip_velocity=skip_vel)
        gPairs.calc_Sn(variables)

        Sn_dict = gPairs.calc_Sn_vs_r(rbins, nboot=100)
        gPairs.calc_corr_Sn(Sn_dict)
        gPairs.add_meta(Sn_dict)

    #embed(header='fig_structure: 215')

    # Start the figure
    if stretch:
        fig = plt.figure(figsize=(19,4))
    else:
        fig = plt.figure(figsize=(19,6))
    plt.clf()
    gs = gridspec.GridSpec(1,3)

    goodN = np.array(Sn_dict['config']['N']) > minN
    

    # Generate the keys
    if variables == 'duLduLduL':
        Skeys = ['S1_duL', 'S2_duL**2', 'S3_'+variables]
    elif variables == 'duTduTduT':
        Skeys = ['S1_duT', 'S2_duT**2', 'S3_'+variables]
    elif variables == 'duLTduLTduLT':
        Skeys = ['S1_duLT', 'S2_duLT**2', 'S3_'+variables]
    elif variables == 'duLdSdS':
        Skeys = ['S1_duL', 'S2_dS**2', 'S3_'+variables]
    elif variables == 'duLdTdT':
        Skeys = ['S1_duL', 'S2_dT**2', 'S3_'+variables]
    elif variables == 'duLduTduT':
        Skeys = ['S1_duL', 'S2_duT**2', 'S3_'+variables]
    else:
        raise IOError("Bad variables")


    for n, clr in enumerate('krb'):
        ax = plt.subplot(gs[n])
        Skey = Skeys[n] 
        ax.errorbar(Sn_dict['r'][goodN], 
                    Sn_dict[Skey][goodN], 
                    yerr=Sn_dict['err_'+Skey][goodN],
                    color=clr,
                    fmt='o', capsize=5)  # fmt defines marker style, capsize sets error bar cap length

        # Corrected
        if n > 0 and show_correct:
            corr_key = Skey[0:2]+'corr'+Skey[2:]
            ax.plot(Sn_dict['r'][goodN], 
                    Sn_dict[corr_key][goodN],  
                    'x',
                    color=clr)
        elif 'med_S1' in Sn_dict.keys():
            ax.plot(Sn_dict['r'][goodN], Sn_dict['med_S1'][goodN],  
                    'x', color=clr)


        ax.set_xscale('log')
    #
        ax.set_xlabel('Separation (km)')
        ax.set_ylabel(Sn_lbls[Skey])

        # Label time separation
        if n == 2:
            same_glider = 'True' if avoid_same_glider else 'False'
            if stretch:
                text = f'{dataset}'
                ytxt = 0.9
                tsz = 18.
            else:
                text = f'{dataset}\n depth = {(iz+1)*10} m, t<{int(Sn_dict['config']['max_time'])} hr\nAvoid same glider? {same_glider}\n {variables}' 
                ytxt = 0.8
                tsz = 16.
            ax.text(0.1, ytxt, text,
                transform=ax.transAxes, fontsize=tsz, ha='left')
        # 0 line
        ax.axhline(0., color='red', linestyle='--')

        plotting.set_fontsize(ax, 19) 
        ax.grid()
        if use_xlim:
            ax.set_xlim(use_xlim)
        if n == 2 and use_ylim is not None:
            ax.set_ylim(use_ylim)
        
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_Sn_depth(dataset:str, outroot='fig_Sn_depth',
    variables = 'duLduLduL', minN:int=10,
    nz:int=50):

    outfile = f'{outroot}_{dataset}_{variables}'
    varlbl = variables
    
    
    # Load and parse
    all_N = []
    all_S1 = []
    all_S2 = []
    all_S3 = []
    for iz in range(nz):
        # Load
        gpair_file = cugn_io.gpair_filename(dataset, iz, variables)
        gpair_file = os.path.join('..', 'Analysis', 'Outputs', gpair_file)

        Sn_dict = gliderpairs.load_Sndict(gpair_file)
        # Grab em
        all_N.append(Sn_dict['N'])
        all_S1.append(Sn_dict['S1'])
        all_S2.append(Sn_dict['S2corr'])
        all_S3.append(Sn_dict['S3corr'])
        # r
        if iz == 0:
            r = Sn_dict['r']

    all_N = np.array(all_N)
    all_S1 = np.array(all_S1)
    all_S2 = np.array(all_S2)
    all_S3 = np.array(all_S3)

    goodN = all_N[0] > minN

    # Median
    med_S1 = np.median(all_S1, axis=0)
    med_S2 = np.median(all_S2, axis=0)
    med_S3 = np.median(all_S3, axis=0)

    std_S1 = np.std(all_S1, axis=0)
    std_S2 = np.std(all_S2, axis=0)
    std_S3 = np.std(all_S3, axis=0)

    #embed(header='297 of figs')

    # Start the figure
    fig = plt.figure(figsize=(19,6))
    plt.clf()
    gs = gridspec.GridSpec(1,3)

    for n, clr in enumerate(['g','r','b']):
        ax = plt.subplot(gs[n])
        Skey = f'S{n+1}'
        if n == 0:
            Sn = all_S1
            medSn = med_S1
        elif n == 1:
            Sn = all_S2
            medSn = med_S2
        else:
            Sn = all_S3
            medSn = med_S3

        for iz in range(nz):
            #a = 0.9 - 0.8*iz/nz
            a = 0.1 + 0.8*iz/nz
            ax.plot(r[goodN], Sn[iz][goodN], color=clr, alpha=a)

        # Median
        ax.plot(r[goodN], medSn[goodN], color='k')

        ax.set_xscale('log')
        ax.set_xlabel('Separation (km)')
        ax.set_ylabel(r'$S_'+str(n+1)+r'$ Corrected')
        # 0 line
        ax.axhline(0., color='k', linestyle='--')

        plotting.set_fontsize(ax, 19) 
        ax.grid()

        ax.set_xlim(1., 100.)
        ax.minorticks_on()

        # Label time separation
        if n == 2:
            ax.text(0.9, 0.1, f'{dataset}\n  {varlbl}', 
                transform=ax.transAxes, fontsize=18, ha='right')
        

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
    
def fig_neg_mean_spatial(max_time=10., avoid_same_glider=True, nbins=20, 
                 iz = 5, dataset = 'ARCTERX', minN:int=10,
                 variables = 'duLduLduL', connect_dots:bool=False,
                 show_zero:bool=False,
                 outfile='fig_ARCTERX_negmean_spatial.png'):

    # ARCTERX
    rbins = 10**np.linspace(0., np.log10(400), nbins) # km

    
    # Load dataset
    gData = gliderdata.load_dataset(dataset)

    # Cut on valid velocity data 
    gData = gData.cut_on_good_velocity()

    # Generate pairs
    gPairs = gliderpairs.GliderPairs(
        gData, max_time=max_time, 
        avoid_same_glider=avoid_same_glider)
    gPairs.calc_delta(iz, variables)
    gPairs.calc_Sn(variables)

    #embed(header='fig_neg_mean_spatial: 366')    

    #Sn_dict = gPairs.calc_Sn_vs_r(rbins, nboot=10000)
    #gPairs.calc_corr_Sn(Sn_dict) 
    

    #goodN = Sn_dict['config']['N'] > minN

    # Grab them
    neg_idx = np.arange(11,15)

    # Maybe loop over these?




    fig = plt.figure(figsize=(7,6))
    plt.clf()
    ax_ll = plt.gca()

    # 
    for ridx in [1]:#,1,2]:
        ss = neg_idx[ridx]
        if show_zero:
            ss = neg_idx[0]-1
            if ridx > 0:
                break
        #print(f"rbin: {rbins[ss],rbins[ss+1]}, Stat: {Sn_dict['S1_duL'][ss]}")

        in_r = (gPairs.r > rbins[ss]) & (gPairs.r <= rbins[ss+1])
        for tt in [0,1]:
            m = 'o' if tt == 0 else 's'
            ec = None #'k' if tt == 0 else None
            scatter = ax_ll.scatter(gPairs.data('lon', tt)[in_r],  
                    gPairs.data('lat', tt)[in_r], 
                    c=gPairs.S1[in_r],
                                    marker=m,
                    edgecolor=ec,
                    cmap='seismic',
                    vmin=-1., vmax=1.,
                    s=5)#, label=f'MID={mid}')

        # Connect with a line
        if connect_dots:
            for tt in np.where(in_r)[0]:
                ax_ll.plot(
                    [gPairs.data('lon', 0)[tt], gPairs.data('lon', 1)[tt]],  
                    [gPairs.data('lat', 0)[tt], gPairs.data('lat', 1)[tt]],
                    color='gray', ls=':', zorder=10, lw=0.3)  
        # Add text
        ax_ll.text(0.1, 0.2, f'{dataset}\n depth = {(iz+1)*10} m \n r={int(np.round(rbins[ss])),int(np.round(rbins[ss+1]))} km',
                transform=ax_ll.transAxes, fontsize=16, ha='left')

    # Color
    cb = plt.colorbar(scatter, pad=0., fraction=0.030)
    cb.set_label(r'$\delta u_L$ (m/s)', fontsize=14)


    ax_ll.set_xlabel('Longitude [deg]')
    ax_ll.set_ylabel('Latitude [deg]')
    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_Sn_distribution(dataset:str, outfile:str,
    variables:str, Sn:str, rval:float,
    iz:int=5, tcut=None,
    avoid_same_glider=True):

    #outfile = f'{outroot}_{dataset}_{variables}'
    #varlbl = variables
    
    # Cut on valid velocity data 
    nbins = 20
    rbins = 10**np.linspace(0., np.log10(400), nbins) # km

    # Pick out the radial bin
    rbool = (rval> rbins) & (rval < np.roll(rbins,-1))
    ir = np.where(rbool)[0][0]

    # Load dataset
    gData = gliderdata.load_dataset(dataset)
    gData = gData.cut_on_good_velocity()
    if tcut is not None:
        gData = gData.cut_on_reltime(tcut)

    # Generate pairs
    gPairs = gliderpairs.GliderPairs(
        gData, max_time=10., 
        avoid_same_glider=avoid_same_glider)
    gPairs.calc_delta(iz, variables)
    gPairs.calc_Sn(variables)

    Sn_dict = gPairs.calc_Sn_vs_r(rbins, nboot=10000)
    gPairs.calc_corr_Sn(Sn_dict) 
    gPairs.add_meta(Sn_dict)

    all_vals = getattr(gPairs, Sn)
    in_r = (gPairs.r > rbins[ir]) & (gPairs.r <= rbins[ir+1])
    vals = all_vals[in_r]

    # Normalize by the error
    err_key = 'err_'+Sn+f'_{variables}'
    vals = vals/(Sn_dict[err_key][ir]*np.sqrt(np.sum(in_r)))
    #embed(header='fig_Sn_distribution: 493')
    
    # Start the figure
    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    hist, bins = np.histogram(vals, bins=100)#, density=True)
    # Plot
    ax.hist(vals, bins=bins, color='green')
    #sns.histplot(vals, bins=100, ax=ax, color='green')
                 #log_scale=(False,True))

    ax.set_xlabel(Sn+r'/$\sigma('+f'{Sn}'+r')$')
    #ax.set_ylabel(r'$S_'+str(n+1)+r'$ Corrected')
    # 0 line
    #ax.axhline(0., color='k', linestyle='--')

    # Overlay a Gaussian
    area = np.sum(hist)*(bins[1]-bins[0])

    #embed(header='fig_Sn_distribution: 513')

    x = np.linspace(-5., 5., 100)
    y = np.exp(-0.5*x**2)/np.sqrt(2*np.pi)
    a_y = np.sum(y)*(x[1]-x[0])
    ax.plot(x, y*area/a_y, color='red', linestyle='--')

    plotting.set_fontsize(ax, 19) 
    ax.grid()

    ax.text(0.1, 0.8, f'{dataset}\n depth = {(iz+1)*10} m, \nr={int(np.round(rbins[ir])),int(np.round(rbins[ir+1]))} km',
                transform=ax.transAxes, fontsize=16, ha='left')

    #ax.set_xlim(1., 100.)
    ax.minorticks_on()

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_region_dus(outroot:str='fig_qg_region_dul', 
                   calc_du3:bool=False,
                   Ndays:int=None,
                   tot_time='5years'):
    """
    Examine the QG structure function
    """

    # Grab files
    output_files = glob.glob(f'../Analysis/Output/SF_region_*_{tot_time}.nc')
    output_files.sort()

    # Loop on em
    for ss, output_file in enumerate(output_files):
        # Parse for x,y
        ix = output_file.find('_x')
        xval = int(output_file[ix+2:ix+5])
        iy = output_file.find('_y')
        yval = int(output_file[iy+2:iy+5])
        # Outfile
        outfile = f'{outroot}_x{xval}_y{yval}.png'
        # Do it
        if calc_du3:
            fig_region_dul3(output_file, outfile,
                       title=f'QG: x={xval}-{xval+100}km, y={yval}-{yval+100}km',
                       Ndays=Ndays)
        else:
            fig_region_dul(output_file, outfile,
                       title=f'QG: x={xval}-{xval+100}km, y={yval}-{yval+100}km',
                       Ndays=Ndays)

def fig_region_dul(output_file:str, outfile:str, title:str=None,
                   Ndays:int=None):
    """
    Generate and save a plot of the <du_L> in the structure function region
    for daily and total time series (5 years).

    Parameters:
    -----------
    output_file : str
        Path to the input NetCDF file containing the dataset.
    outfile : str
        Path to save the generated figure.
    title : str, optional
        Title of the plot. If None, no title will be displayed.

    Description:
    ------------
    This function loads a dataset from the specified NetCDF file, 
    generates a plot of the structure function region, and saves 
    the figure to the specified output file. The plot includes 
    individual time series as well as the mean over time. The x-axis 
    represents the radial distance in kilometers, and the y-axis 
    represents the longitudinal velocity difference.

    Notes:
    ------
    - The function uses the `xarray` library to load the dataset.
    - The `cugn_plotting.set_fontsize` function is used to adjust 
        the font size of the plot.
    - The figure is saved with a resolution of 300 DPI.

    Returns:
    --------
    None
    """
    if Ndays is None:
        Ndays = 60
    # Load
    SFds = xarray.load_dataset(output_file)
    #embed(header='fig_region_dul: 648')

    # Cut on time
    i1 = -1*Ndays
    times = np.arange(i1, i1 + Ndays)
    SFds = SFds.isel(time=times)

    # Start the figure
    fig = plt.figure(figsize=(10,10))
    plt.clf()
    ax = plt.gca()

    ax.plot(SFds.dr.values[0,:]*1e-3, SFds.ulls.T, '-k', linewidth=0.5, alpha=0.3)
    ax.plot(SFds.dr.mean('time')*1e-3, SFds.ulls.T.mean('time'), '-r', linewidth=1.5)
    ax.set_xlabel(r'$r$ [km]')
    ax.set_ylabel(r'$\delta u_L(r, t)$ [m s$^{-1}$]')

    if title is not None:
        ax.set_title(title, fontsize=23.)

    cugn_plotting.set_fontsize(ax, 20)

    ax.minorticks_on()
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_region_dul3(output_file:str, outfile:str,
                   title:str=None, Ndays:int=None):
    if Ndays is None:
        Ndays = 90
    # Load
    SFds = xarray.load_dataset(output_file)
    qg, mSF_15_duL = qg_utils.load_qg(use_SFduL=True)
    SF_dict_duL = qg_utils.calc_dus(qg, mSF_15_duL)
    du2_mn_duL = SF_dict_duL['du2_mn']
    du3_mn_duL = SF_dict_duL['du3_mn']
    dull_mn = SF_dict_duL['dull_mn']
    rr1 = SF_dict_duL['rr1']

    # Cut on time
    i1 = -1*Ndays
    times = np.arange(i1, i1 + Ndays)
    SFds = SFds.isel(time=times)

    # Correct the du3
    du1 = SFds.ulls.T.mean('time')
    du2 = SFds.du2.T.mean('time')
    du3 = SFds.du3.T.mean('time')
    du3_corr = du3 - 3*du1*du2 + 2*du1**3
    
    rrr1 = SFds.dr.mean('time')*1e-3 


    # Start the figure
    fig = plt.figure(figsize=(10,3))
    plt.clf()
    gs = gridspec.GridSpec(1,3)

    # ################################################3
    # du
    ax1 = plt.subplot(gs[0])

    ax1.semilogx(rrr1, du1, 'k', linewidth=1)
    ax1.semilogx(rr1*1e-3, dull_mn, 'r', linewidth=1, 
                label=r'Full grid $<\delta u_L>$')
    ax1.set_xlabel(r'$r$ [km]')
    ax1.set_ylabel(r'$<\delta u_L>$ [m/s]')
    ax1.text(0.1, 0.1, f'{Ndays} days', transform=ax1.transAxes, fontsize=12.)

    # ################################################3
    # du2
    ax2 = plt.subplot(gs[1])
    ax2.loglog(rr1*1e-3, du2_mn_duL, 'r', linewidth=1, 
                label=r'Full grid $<\delta u_L^2>$')
    ax2.loglog(SFds.dr.mean('time')*1e-3, du2, '-k', linewidth=1.5, 
                label=r'Region $<\delta u_L^2>$')
    lsz = 10.
    ax2.legend(fontsize=lsz, loc='lower right')
    ax2.set_xlabel(r'$r$ [km]')
    ax2.set_ylabel(r'$<\delta u^2> \, {\rm [m/s]^2}$')

    # ################################################3
    # du3
    ax3 = plt.subplot(gs[2])
    ax3.semilogx(SFds.dr.mean('time')*1e-3, du3, '-k', linewidth=1.5, 
                label=r'$<\delta u_L^3>$')
    ax3.semilogx(SFds.dr.mean('time')*1e-3, du3_corr, 'x', color='b', 
                label=r'Corrected $<\delta u_L^3>$')
    ax3.semilogx(rr1*1e-3, du3_mn_duL, '-r', linewidth=1, 
                label=r'Full grid $<\delta u_L^3>$')
    ax3.set_xlabel(r'$r$ [km]')
    ax3.set_ylabel(r'$<\delta u_L^3(r)> \, \rm [m^{3} \, s^{-3}]$')

    if title is not None:
        ax2.set_title(title, fontsize=15.)

    for ax in [ax1, ax2, ax3]:
        cugn_plotting.set_fontsize(ax, 11)

    ax.minorticks_on()
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_qg_duL_vs_time(x0:int,y0:int, outroot:str='fig_qg_duL_vs_time',
                   title:str=None):
    outfile = f'{outroot}_x{x0}_y{y0}.png'
    output_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_5years.nc' 

    # Load
    SFds = xarray.load_dataset(output_file)

    # Start the figure
    fig = plt.figure(figsize=(10,10))
    plt.clf()
    ax = plt.gca()

    ndays = [1, 60, 180, 365, 2*365, 5*365]
    for nday in ndays:
        ax.plot(SFds.dr.mean('time')*1e-3, 
                SFds.ulls.isel(time=np.arange(nday)).T.mean('time'), 
                '-', linewidth=1.5,
                label=f'ndays={nday}')
    ax.set_xlabel('$r$ [km]')
    ax.set_ylabel('$\\delta u_L(r, t)$ [m s$^{-1}$]')

    title=f'QG: x={x0}-{x0+100}km, y={x0}-{x0+100}km'
    ax.set_title(title, fontsize=23.)

    cugn_plotting.set_fontsize(ax, 20)

    ax.axhline(0., color='gray', linestyle='--')
    ax.legend(fontsize=20.)
    ax.minorticks_on()
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_qg_duL_by_year(x0:int,y0:int, outroot:str='fig_qg_duL_yearly',
                   title:str=None):
    outfile = f'{outroot}_x{x0}_y{y0}.png'
    output_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_5years.nc' 

    # Load
    SFds = xarray.load_dataset(output_file)

    # Start the figure
    fig = plt.figure(figsize=(10,10))
    plt.clf()
    ax = plt.gca()

    #ndays = [1, 60, 180, 365, 2*365, 5*365]
    #for nday in ndays:
    #    ax.plot(SFds.dr.mean('time')*1e-3, 
    #            SFds.ulls.isel(time=np.arange(nday)).T.mean('time'), 
    #            '-', linewidth=1.5,
    #            label=f'ndays={nday}')
    for ss in range(5):
        ax.plot(SFds.dr.mean('time')*1e-3, 
                SFds.ulls.isel(time=np.arange(ss*365, (ss+1)*365)).T.mean('time'), 
                '-', linewidth=1.5,
                label=f'year={ss+1}')
    ax.set_xlabel('$r$ [km]')
    ax.set_ylabel('$\\delta u_L(r, t)$ [m s$^{-1}$]')

    title=f'QG: x={x0}-{x0+100}km, y={x0}-{x0+100}km'
    if title is not None:
        ax.set_title(title, fontsize=23.)

    cugn_plotting.set_fontsize(ax, 20)

    ax.axhline(0., color='gray', linestyle='--')
    ax.legend(fontsize=20.)
    ax.minorticks_on()
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_compare_duL_qg_data(dataset, outroot:str='fig_duL_QG_vs',
                   title:str=None):
    # Load
    outfile = f'{outroot}_{dataset}.png'

    # Data
    rdict = data_utils.load_SF(dataset)
    Sn_dict = rdict['Sn_dict']
    gPairs = rdict['gPairs']
    Skeys = rdict['Skeys']
    goodN = rdict['goodN']

    # Start the figure
    fig = plt.figure(figsize=(10,7))
    plt.clf()
    ax = plt.gca()

    # Dataset
    Skey = Skeys[0]
    ax.errorbar(Sn_dict['r'][goodN], 
                Sn_dict[Skey][goodN], 
                yerr=Sn_dict['err_'+Skey][goodN],
                color='k', 
                fmt='o', capsize=5,
                label=dataset)
    ax.axhline(0., color='gray', linestyle='--')

    # Grab QG files
    output_files = glob.glob('../Analysis/Output/SF_region_*')
    output_files.sort()

    for output_file in output_files:
        SFds = xarray.load_dataset(output_file)
        ax.plot(SFds.dr.mean('time')*1e-3, SFds.ulls.T.mean('time'), '-', linewidth=1.5)

    ax.set_xlabel('$r$ [km]')
    ax.set_ylabel('$\\delta u_L(r, t)$ [m s$^{-1}$]')
    ax.set_xscale('log')

    cugn_plotting.set_fontsize(ax, 20)

    ax.minorticks_on()
    ax.legend(fontsize=20.)
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_examine_qg(outfile:str='fig_examine_qg.png'):
    """
    Compare <duL> with sqrt<du^2>
    """
    # Load the data
    qg, mSF_15 = qg_utils.load_qg()

    # Calculate the first order structure function
    rr1, du1, du1LL, dull_mn, dull_25, dull_50, du2_mn, du3_mn, du3_corr = \
        qg_utils.calc_dus(qg, mSF_15)

    rms = np.sqrt(du2_mn)


    # Start the figure
    fig = plt.figure(figsize=(10,6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])


    #ax.set_xlim(1., 100.)
    #ax.minorticks_on()

    # Plot the QG RMS from <delta u^2>
    ax.semilogx(rr1*1e-3, du2_mn*1e3, 'b', linewidth=1, 
                label=r'$\sqrt{<\delta u^2>}$')

    # Plot a random set of 1 day averages
    c1 = 'red'
    #embed(header='fig_examine_qg: 617')
    randi = np.random.choice(np.arange(0, du1LL.data.shape[0]), size=100, replace=False)

    ax.semilogx(rr1*1e-3, du1LL.data.T[:,randi]*1e3, '-', color=c1,
            linewidth=0.5, alpha=0.1)
    ax.semilogx(0, 0, '-', color=c1, linewidth=0.5, alpha=0.8, 
                label='Daily $\\overline{\\delta u1_{L}}(r, t)$')


    # Plot the 2-month averages
    c50 = 'green'
    ax.semilogx(rr1*1e-3, dull_50.T*1e3, '-', color=c50,
                linewidth=0.5, alpha=0.3)
    ax.semilogx(0, 0, '-', color=c50, linewidth=0.5, alpha=0.8, 
                label='50 days $\\overline{\\delta u1_L}(r, t)$')

    ax.legend(fontsize=15, loc='upper left')

    ax.set_xlabel(r'$r$ [km]')
    ax.set_ylabel(r'Comparison of $\delta u$ and RMS')

    cugn_plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
        
def fig_qg_SF(outfile:str='fig_qg_SF.png'):
    """
    QG Structure Function, duL and total
    """
    # Load the data
    qg, mSF_15 = qg_utils.load_qg()
    _, mSF_15_duL = qg_utils.load_qg(use_SFduL=True)

    # Calculate the first order structure function
    SF_dict = qg_utils.calc_dus(qg, mSF_15)
    SF_dict_duL = qg_utils.calc_dus(qg, mSF_15_duL)

    # Unpack a bit
    rr1 = SF_dict['rr1']
    du1 = SF_dict['du1']
    du1LL = SF_dict['du1LL']
    du1_mn = SF_dict['du1_mn']
    dull_mn = SF_dict['dull_mn']
    dutt_mn = SF_dict['dutt_mn']
    du2_mn = SF_dict['du2_mn']
    du3_mn = SF_dict['du3_mn']

    du2_mn_duL = SF_dict_duL['du2_mn']
    du3_mn_duL = SF_dict_duL['du3_mn']


    # Start the figure
    fig = plt.figure(figsize=(10,4))
    plt.clf()
    gs = gridspec.GridSpec(1,3)

    # ################################################3
    # du
    ax0 = plt.subplot(gs[0])

    ax0.semilogx(rr1*1e-3, du1_mn*1e3, 'k', linewidth=1, 
                label=r'$<\delta u>$')
    ax0.semilogx(rr1*1e-3, dull_mn*1e3, 'b', linewidth=1, 
                label=r'$<\delta u_L>$')
    ax0.semilogx(rr1*1e-3, dutt_mn*1e3, 'r', linewidth=1, 
                label=r'$<\delta u_T>$')

    lsz = 13.
    ax0.legend(fontsize=lsz, loc='lower left')
    ax0.set_xlabel(r'$r$ [km]')
    ax0.set_ylabel(r'$<\delta u> \, 10^{-3}$ [m/s]')

    # ################################################3
    # du2
    ax2 = plt.subplot(gs[1])

    ax2.loglog(rr1*1e-3, du2_mn, 'k', linewidth=1, 
                label=r'$<\delta u_L^2 + \delta u_T^2>$')
    ax2.loglog(rr1*1e-3, du2_mn_duL, 'b', linewidth=1, 
                label=r'$<\delta u_L^2>$')
    ax2.legend(fontsize=lsz, loc='lower right')
    ax2.set_xlabel(r'$r$ [km]')
    ax2.set_ylabel(r'$<\delta u^2> \, {\rm [m/s]^2}$')

    # ################################################3
    # du3
    ax3 = plt.subplot(gs[2])

    ax3.semilogx(rr1*1e-3, du3_mn, 'k', linewidth=1, 
                label=r'$<\delta u_L(\delta u_L^2 + \delta u_T^2)>$')
    ax3.semilogx(rr1*1e-3, du3_mn_duL, 'b', linewidth=1, 
                label=r'$<\delta u_L^3>$')
    ax3.legend(fontsize=lsz, loc='lower left')
    ax3.set_xlabel(r'$r$ [km]')
    ax3.set_ylabel(r'$<\delta u^3> \, {\rm [m/s]^2}$')

    for ax in [ax0, ax2, ax3]:
        cugn_plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_compare_dus(dataset:str, outroot:str='fig_comp_dus',
                  variables = 'duLduLduL',
                  iz:int=5): 
    """
    Compare dus
    """
    outfile = f'{outroot}_z{(iz+1)*10}_{dataset}.png'

    # Data
    rdict = data_utils.load_SF(dataset, variables=variables, iz=iz)
    Sn_dict = rdict['Sn_dict']
    gPairs = rdict['gPairs']
    Skeys = rdict['Skeys']
    goodN = rdict['goodN']

    u_rms = np.sqrt(Sn_dict[Skeys[1]][goodN])

    # Start the figure
    fig = plt.figure(figsize=(10,6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    ax = plt.subplot(gs[0])

    # S1
    Skey = Skeys[0]
    ax.errorbar(Sn_dict['r'][goodN], 
                Sn_dict[Skey][goodN], 
                yerr=Sn_dict['err_'+Skey][goodN],
                color='k', label=Sn_lbls[Skey],
                fmt='o', capsize=5)  # fmt defines marker style, capsize sets error bar cap length

    ax.plot(Sn_dict['r'][goodN], u_rms, color='b', 
                label=r'$\sqrt{<\delta u^2>}$')

    ax.axhline(0., color='gray', linestyle='--')

    ax.legend(fontsize=15, loc='lower right')
    ax.set_xscale('log')
    ax.grid()

    ax.set_xlabel(r'$r$ [km]')
    ax.set_ylabel(r'Comparison of $\delta u$ and RMS [m/s]')

    cugn_plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Separations
    if flg == 1:
        fig_separations('ARCTERX')
        fig_separations('Calypso2019', max_time=10.)

    # Delta times
    if flg == 2:
        fig_dtimes('ARCTERX')
        fig_dtimes('Calypso2019')

    # velocities
    if flg == 3:
        fig_dus('ARCTERX')
        fig_dus('Calypso2019')

    # Sn
    if flg == 4:
        #fig_structure('ARCTERX')
        #fig_structure('ARCTERX', avoid_same_glider=False)
        #fig_structure('Calypso2019')
        fig_structure('Calypso2022')
        #fig_structure('Calypso2022', variables='duLdSdS')
        #fig_structure('Calypso2022', variables='duLdTdT', iz=3)

    # Full run on a daatset
    if flg == 5:
        #dataset = 'Calypso2019'
        dataset = 'Calypso2022'
        #dataset = 'ARCTERX'
        avoid_same_glider = True
        #fig_separations(dataset)
        #fig_dtimes(dataset)
        #fig_dus(dataset)

        fig_structure(dataset, avoid_same_glider=avoid_same_glider)
        #fig_structure(dataset, avoid_same_glider=avoid_same_glider,
        #              variables='duTduTduT')#, iz=5)
        #fig_structure(dataset, avoid_same_glider=avoid_same_glider,
        #              variables='duLTduLTduLT', show_correct=False)#, iz=5)
        #fig_structure(dataset, avoid_same_glider=avoid_same_glider, iz=10)

    # Sn with depth
    if flg == 6:
        fig_Sn_depth('Calypso2022')

    # Explore negative mean
    if flg == 7:
        dataset = 'ARCTERX'
        avoid_same_glider = True
        fig_neg_mean_spatial(connect_dots=True)
        #fig_neg_mean_spatial(outfile='fig_ARCTERX_zeromean_spatial.png',
        #                     show_zero=True)#, connect_dots=True)

        #fig_structure(dataset, avoid_same_glider=avoid_same_glider,
        #              tcut=(0.5,1.0), outroot='fig_structre_tcut51')
                      #tcut=(0.,0.5), outroot='fig_structre_tcut05')

    # Sn Distributions
    if flg == 8:
        fig_Sn_distribution('ARCTERX', 'fig_Sn_distrib_ARCTERX.png',
            'duLduLduL', 'S3', rval=70., iz=5)

    # QG uL and uL^2
    if flg == 9:
        fig_examine_qg()

    # Compare RMS with uL
    if flg == 10:
        fig_compare_dus('Calypso2022')

    # Generate du_L figures for QG
    if flg == 11:
        fig_region_dus()
        #fig_region_duls(tot_time='60days')

    # Compare duL QG vs. dataset``
    if flg == 12:
        fig_compare_duL_qg_data('Calypso2022')
    
    # Compare duL QG vs. dataset
    if flg == 13:
        fig_qg_duL_vs_time(300,300)
        fig_qg_duL_by_year(300,300)

    # Compare duL vs. total QG SF
    if flg == 14:
        fig_qg_SF()

    # Generate du_L figures for QG
    if flg == 15:
        #fig_region_dul3('../Analysis/Output/SF_region_x300_y300_5years.nc',
        #                'fig_region_dul3_x300_y300.png',
        #                title='QG duL3: x=300-400km, y=300-400km')
        #fig_region_dul3('../Analysis/Output/SF_region_x400_y400_5years.nc',
        #                'fig_region_dul3_x400_y400.png',
        #                title='QG duL3: x=400-500km, y=400-500km')

        #fig_region_dus(outroot='fig_qg_region_du3', 
        #               calc_du3=True)
        fig_region_dus(outroot='fig_qg_region_du3_300days', 
                       calc_du3=True, Ndays=300)

    # Figs for Pitch slide
    if flg == 16:
        avoid_same_glider = True
        #fig_separations(dataset)
        #fig_dtimes(dataset)
        #fig_dus(dataset)

        datasets = ['Calypso2019', 'Calypso2022', 'ARCTERX-2023']
        #datasets = ['ARCTERX-2023']
        for dataset in datasets:
            if dataset == 'Calypso2019':
                use_ylim = (-0.003,0.003)
            else:
                use_ylim=None
            fig_structure(dataset, avoid_same_glider=avoid_same_glider,
                      stretch=True, use_xlim=(1.,400.), use_ylim=use_ylim)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)