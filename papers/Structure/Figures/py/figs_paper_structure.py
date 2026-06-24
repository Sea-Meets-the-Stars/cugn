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
import struct_defs
import qg_uL_SF
import qg_utils

# Globals
Sn_lbls = cugn_plotting.Sn_lbls
datasets = ['Calypso2019', 'Calypso2022', 'ARCTERX-2023']#, 'ARCTERX-2025']
focus_dataset = 'Calypso2022'

def set_xmax(counts, rbins, minxmax:float=100):
    non_zero = counts > 0
    xmax = rbins[1:][non_zero][-1]
    xmax = max(xmax, minxmax)
    return xmax

def plot_single_order(order:int, ax, rdict,
                      use_xlim:tuple=None, use_ylim:tuple=None,
                      corrected:bool=False):

    color = 'krb'[order-1]
    ## Unpack
    gPairs = rdict['gPairs']
    Sn_dict = rdict['Sn_dict']
    goodN = rdict['goodN']
    Skeys = rdict['Skeys']
    rbins = rdict['rbins']
    Skey = Skeys[order-1]

    # Plot main data
    corr_key = Skey[0:2]+'corr'+Skey[2:]
    key = corr_key if corrected else Skey
    ax.errorbar(Sn_dict['r'][goodN], 
                Sn_dict[key][goodN], 
                yerr=Sn_dict['err_'+Skey][goodN],
                color=color,
                fmt='o', capsize=5, label='Corrected')  # fmt defines marker style, capsize sets error bar cap length

    if corrected:
        ax.plot(Sn_dict['r'][goodN], 
                Sn_dict[Skey][goodN], 'x', color=color, label='Raw')
        ax.legend(fontsize=17, loc='upper left')

    if struct_defs.btype == 'log':
        ax.set_xscale('log')
#
    ax.set_xlabel('Separation (km)')
    ax.set_ylabel(Sn_lbls[Skey])

    # 0 line
    ax.axhline(0., color='g', linestyle='--')

    plotting.set_fontsize(ax, 19) 
    #ax.grid()
    ax.grid(which='major', linewidth=0.8, alpha=0.7)
    ax.grid(which='minor', linewidth=0.5, alpha=0.3)
    if use_xlim:
        ax.set_xlim(use_xlim)
    else:
        xmax = set_xmax(np.array(Sn_dict['config']['N']), rbins)
        ax.set_xlim(None, xmax)
    if use_ylim is not None:
        ax.set_ylim(use_ylim)
    

def fig_experiments(outfile='fig_experiments.png', 
                    max_time:float=10., show_legend:bool=False):

    # Start the figure
    ndata = len(datasets)
    if ndata == 3:
        fig = plt.figure(figsize=(9,3))
        gs = gridspec.GridSpec(1,3)
        fsz = 12.
        tsz = 14.
    elif ndata == 4:
        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(2,2)
        fsz = 15.
        tsz = 17.
    else:
        raise ValueError(f"Bad number of datasets: {len(datasets)}")
    plt.clf()

    for ss, dataset in enumerate(datasets):

        # Axis
        ax_ll = plt.subplot(gs[ss])

        # Load dataset
        profilers = glider_io.load_dataset(dataset)
    
        # Generate pairs
        gPairs = profilerpairs.ProfilerPairs(profilers, 
                                          max_time=max_time,
                                          debug=False,
                                          randomize=True)
        for mid in np.unique(gPairs.data('missida',2).astype(int)):
            marker = '+'
            idx = gPairs.data('missida',2) == mid
            ax_ll.scatter(gPairs.data('lon', 2)[idx], 
                gPairs.data('lat', 2)[idx], s=2, label=f'MID={mid}',
                marker=marker)

        # Dataset
        #  Make it an axis title
        ax_ll.set_title(dataset, fontsize=tsz)

        ax_ll.set_xlabel('Longitude [deg]')
        ax_ll.set_ylabel('Latitude [deg]')
        if show_legend:
            ax_ll.legend(fontsize=12, #ncol=2,
                        loc='upper left')
        ax_ll.grid()
        if dataset == 'ARCTERX-2023':
            ssz = 1.0
        elif dataset == 'ARCTERX-2025':
            ssz = 0.25
        else: 
            ssz = 0.5
        ax_ll.xaxis.set_major_locator(MultipleLocator(ssz))  # Major ticks every 2 units
        ax_ll.yaxis.set_major_locator(MultipleLocator(ssz))  # Major ticks every 2 units
        plotting.set_fontsize(ax_ll, fsz) 

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_separations(dataset:str, outroot='fig_separations', 
                    fsz:float=10., xmax:float=None):

    outfile = f'{outroot}_{dataset}.png'
    clr = struct_defs.dataset_clrs[dataset]

    # Load dataset
    #profilers = glider_io.load_dataset(dataset)
    #embed(header='45 of figs_structure')

    rdict = data_utils.load_SF(
        dataset=dataset, iz=struct_defs.iz, 
        minN=struct_defs.minN, btype=struct_defs.btype, 
        max_time=struct_defs.max_time, 
        avoid_same_glider=struct_defs.avoid_same_glider)
    
    ## Unpack
    gPairs = rdict['gPairs']
    Sn_dict = rdict['Sn_dict']
    goodN = rdict['goodN']
    Skeys = rdict['Skeys']
    rbins = rdict['rbins']

    #embed(header='129 of figs_paper_structure')

    # Generate pairs
    #gPairs = gliderpairs.GliderPairs(gData, max_time=max_time)
    #gPairs = profilerpairs.ProfilerPairs(profilers, 
    #                                      max_time=struct_defs.max_time,
    #                                      debug=False,
    #                                      randomize=False)


    # Start the figure
    fig = plt.figure(figsize=(6,6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)


    # Log Separations
    rbins = data_utils.rbinning(struct_defs.btype, dataset)

    ax_r = plt.subplot(gs[0])
    # Plot a bar chart histogram from the counts
    counts = np.array(Sn_dict['config']['N'])
    ax_r.stairs(counts, rbins, color=clr, fill=True)
    #_ = sns.histplot(gPairs.r, bins=rbins, ax=ax_r, color=clr)#, log_scale=True)
    # Label
    ax_r.set_xlabel('Separation [km]')
    ax_r.set_ylabel('Number of pairs')

    # Add dataset
    lsz = 16.
    ax_r.text(0.1, 0.9, dataset, transform=ax_r.transAxes, fontsize=lsz)
    # Label time separation
    #ax_r.text(0.1, 0.8, f't < {struct_defs.max_time} hours', transform=ax_r.transAxes, fontsize=15)
    if struct_defs.btype == 'log':
        ax_r.set_xscale('log')

    plotting.set_fontsize(ax_r, 15) 
    if xmax is None:
        xmax = set_xmax(counts, rbins)

    ax_r.set_xlim(None, xmax)
        #ax.set_yscale('log')

    # Add horizontal line at minN, labeled
    ax_r.axhline(struct_defs.minN, color='gray', linestyle='--')
    ax_r.text(0.97, struct_defs.minN, rf'$N_{{\rm min}}={struct_defs.minN}$',
              transform=ax_r.get_yaxis_transform(),
              ha='right', va='bottom', color='gray', fontsize=13)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_single_order(dataset:str, order:int, outroot='fig_duL', 
                    use_xlim:float=None, use_ylim:float=None):

    # Order specific
    if order > 1:
        outroot = outroot + f'{order}'

    outfile = f'{outroot}_{dataset}.png'
    clr = struct_defs.dataset_clrs[dataset]

    # Run
    rdict = data_utils.load_SF(
        dataset=dataset, iz=struct_defs.iz, 
        minN=struct_defs.minN, btype=struct_defs.btype, 
        max_time=struct_defs.max_time, 
        avoid_same_glider=struct_defs.avoid_same_glider)

    # Start the figure
    fig = plt.figure(figsize=(7,6))
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    n=order-1
    ax = plt.subplot(gs[0])
    plot_single_order(order, ax, rdict)
        
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_corrected_s3(dataset:str, outfile:str, use_xlim:tuple=None):

    rdict = data_utils.load_SF(dataset, variables='duLduLduL')
    # Unpack
    Sn_dict = rdict['Sn_dict']
    goodN = rdict['goodN']
    rbins = rdict['rbins']

    # Trim Calypso2019
    #if dataset == 'Calypso2019':
    #    goodN[-4] = False
    
    Skey = rdict['Skeys'][2]

    # Combine the figures
    fig = plt.figure(figsize=(10, 6))
    plt.clf()
    ax = plt.gca()

    # Corrected
    plot_single_order(3, ax, rdict, corrected=True)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_S1S3_other(datasets:str, outfile:str='fig_S1S3_other.png', use_xlim:tuple=None):

    # Combine the figures
    fig = plt.figure(figsize=(12, 12))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    for row, dataset in enumerate(datasets):
        rdict = data_utils.load_SF(dataset, variables='duLduLduL')
        for col, order in enumerate([1, 3]):
            ax = plt.subplot(gs[row, col])
            #
            correct = True if order == 3 else False
            plot_single_order(order, ax, rdict, corrected=correct)
            # Label by dataset
            if order == 1:
                ax.text(0.05, 0.95, dataset, transform=ax.transAxes,
                         fontsize=19, ha='left', va='top')

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# #########################################################
# QG figs
# #########################################################


def fig_full_qg_SF(outfile:str='fig_full_qg_SF.png'):
    """
    QG Structure Function, duL and total
    """
    # Load the data
    qg, mSF_15_duL = qg_utils.load_qg(use_SFduL=True)

    # Calculate the first order structure function
    SF_dict_duL = qg_utils.calc_dus(qg, mSF_15_duL)

    # Unpack a bit
    rr1 = SF_dict_duL['rr1']
    # du1
    dull_mn = SF_dict_duL['dull_mn']
    # du2
    du2_mn_duL = SF_dict_duL['du2_mn']
    # du3
    du3_mn_duL = SF_dict_duL['du3_mn']

    # Start the figure
    fig = plt.figure(figsize=(10,3))
    plt.clf()
    gs = gridspec.GridSpec(1,3)

    # ################################################3
    # du
    ax0 = plt.subplot(gs[0])

    ols = ':'
    clr = 'k'

    ax0.semilogx(rr1[:-1]*1e-3, dull_mn[:-1]*1e0, 'o', color=clr, markersize=3, #linewidth=1, 
                label=r'$<\delta u_L>$')

    lsz = 7.
    #ax0.legend(fontsize=lsz, loc='lower left')
    ax0.set_xlabel(r'$r$ [km]')
    ax0.set_ylabel(r'$<\delta u_L>$   [m/s]')
    #ax0.set_ylabel(r'$<\delta u> \, 10^{-3}$ [m/s]')

    # ################################################3
    # du2
    ax2 = plt.subplot(gs[1])

    clr = 'r'
    ax2.semilogx(rr1*1e-3, du2_mn_duL, 'o', color=clr, markersize=3, #linewidth=1, 
                label=r'New $<\delta u_L^2>$')
    #ax2.legend(fontsize=lsz, loc='lower right')
    ax2.set_xlabel(r'$r$ [km]')
    ax2.set_ylabel(r'$<\delta u_L^2> \;\; {\rm [m/s]^2}$')


    # ################################################3
    # du3
    ax3 = plt.subplot(gs[2])

    clr = 'b'
    ax3.semilogx(rr1*1e-3, du3_mn_duL, 'o', color=clr, 
                 markersize=3, #linewidth=1, 
                label=r'New $<\delta u_L^3>$')
    #ax3.legend(fontsize=lsz, loc='upper left')
    ax3.set_xlabel(r'$r$ [km]')
    ax3.set_ylabel(r'$<\delta u_L^3> \;\; {\rm [m/s]^3}$')
    ax3.axvline(92.3, color='k', linestyle='--')

    for ax in [ax0, ax2, ax3]:
        cugn_plotting.set_fontsize(ax, 13)
        #
        ax.grid(which='major', linewidth=0.8, alpha=0.7)
        ax.grid(which='minor', linewidth=0.5, alpha=0.3)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_qg_subregion_vs_full(x0:int=400, y0:int=400, dx:int=100,
                          outfile:str='fig_qg_100km_vs_full.png',
                          Ndays:int=1825, llocs=(None,None,None)):
    """
    Compare QG Ndays structure functions: 100km region vs full box.
    Plots S1 (duL), S2 (du^2), and S3 (du^3) side by side.
    """
    # Use parse_SF() for both full box and 100km region calculations
    if dx == 100:
        region_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_5years.nc'
    elif dx == 200:
        region_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_200km_5years.nc'
    elif dx == 300:
        region_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_300km_5years.nc'
    elif dx == 500:
        region_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_500km_5years.nc'
    else:
        raise ValueError(f'Invalid dx: {dx}')

    rr1_full, rrr1_region, du1_region, du2_region, du3_region, \
        du3_corr_region, dull_mn_full, du2_mn_full, du3_mn_full = \
        qg_uL_SF.parse_SF(region_file, Ndays)

    # Cut on r
    if dx == 100:
        rcut = rrr1_region <= 100.
    elif dx == 200:
        rcut = rrr1_region <= 200.
    elif dx == 300:
        rcut = rrr1_region <= 300.
    elif dx == 500:
        rcut = rrr1_region <= 500.
    else:
        raise ValueError(f'Invalid dx for cut: {dx}')

    rrr1_region = rrr1_region[rcut]
    du1_region = du1_region[rcut]
    du2_region = du2_region[rcut]
    du3_region = du3_region[rcut]
    du3_corr_region = du3_corr_region[rcut]

    # Start the figure (3-panel layout like fig_full_qg_SF)
    fig = plt.figure(figsize=(10,3))
    plt.clf()
    gs = gridspec.GridSpec(1,3)

    lsz = 9.
    cfull = 'gray'

    # ################################################
    # du (first-order, longitudinal)
    ax0 = plt.subplot(gs[0])
    ax0.semilogx(rr1_full*1e-3, dull_mn_full*1e3, cfull, linewidth=1,
                label='Full box')
    ax0.semilogx(rrr1_region, du1_region*1e3, 'ko', markersize=3,
                label=f'{dx}km region')
    ax0.legend(fontsize=lsz, loc=llocs[0])
    ax0.set_xlabel(r'$r$ [km]')
    ax0.set_ylabel(r'$<\delta u> \, 10^{-3}$ [m/s]')

    # ################################################
    # du2 (second-order)
    ax2 = plt.subplot(gs[1])
    ax2.semilogx(rr1_full*1e-3, du2_mn_full, cfull, linewidth=1,
                label=r'Full box') 
    ax2.semilogx(rrr1_region, du2_region, 'ro', markersize=3,
                label=f'{dx}km region')
    ax2.legend(fontsize=lsz, loc=llocs[1])
    ax2.set_xlabel(r'$r$ [km]')
    ax2.set_ylabel(r'$<\delta u^2> \;\; {\rm [m/s]^2}$')

    # ################################################
    # du3 (third-order, scaled by 1e-3)
    ax3 = plt.subplot(gs[2])
    scl3 = 1
    ax3.semilogx(rr1_full*1e-3, du3_mn_full*scl3, cfull, linewidth=1,
                label='Full box') 
    ax3.semilogx(rrr1_region, du3_region*scl3, 'bo', markersize=3,
                label=f'{dx}km region',
                markerfacecolor='none') 
    # Corrected du3 for 100km region (open red circles)
    ax3.semilogx(rrr1_region, du3_corr_region*scl3, 'bo', markersize=3,
                label=r'Corrected')
    ax3.legend(fontsize=lsz, loc=llocs[2])
    ax3.set_xlabel(r'$r$ [km]')
    ax3.set_ylabel(r'$<\delta u^3>  \;\; {\rm [m/s]^3}$')
    # Horizontal line at 0
    ax3.axhline(0., color='gray', linestyle='--')

    for ax in [ax0, ax2, ax3]:
        cugn_plotting.set_fontsize(ax, 13)
        # Grid
        ax.grid(which='major', linewidth=0.8, alpha=0.7)
        ax.grid(which='minor', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# #########################################################
# APPENDIX
# #########################################################


def fig_histogram_dr(outfile='fig_histogram_dr.png', 
                    max_time:float=10., log_rbins:bool=False):

    # Start the figure
    fig = plt.figure(figsize=(10,10))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    #datasets = ['Calypso2019', 'Calypso2022', 'ARCTERX-2023',
    #            'ARCTERX-2025']
    for ss, dataset in enumerate(datasets):
        clr = struct_defs.dataset_clrs[dataset]

        # Axis
        ax_r = plt.subplot(gs[ss])

        # Load dataset
        profilers = glider_io.load_dataset(dataset)
    
        # Generate pairs
        gPairs = profilerpairs.ProfilerPairs(profilers, 
                                          max_time=max_time,
                                          debug=False,
                                          randomize=True)
        _ = sns.histplot(gPairs.r, bins=20, 
                         log_scale=log_rbins, 
                         ax=ax_r, color=clr)

        ax_r.text(0.95, 0.95, dataset, transform=ax_r.transAxes,
                     fontsize=17, ha='right', va='top')
        ax_r.set_xlabel('Separation [km]')
        ax_r.set_ylabel('Count')
        #if dataset == 'ARCTERX-2023':
        #    ssz = 1.0
        #elif dataset == 'ARCTERX-2025':
        #    ssz = 0.25
        #else: 
        #    ssz = 0.5
        plotting.set_fontsize(ax_r, 15) 

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_structure(dataset:str, outroot='fig_structure',
                  variables = 'duLduLduL',
                  iz:int=5,
                  skip_vel:bool=False,
                  stretch:bool=False,
                  use_xlim:tuple=None,
                  use_ylim:tuple=None,
                  max_time=7.,
                  minN:int=50, avoid_same_glider:bool=True,
                  show_correct:bool=True,
                  btype:str='log'):

    # Set in_field=True to load in-field data
    #kwargs = {}
    #if variables in ['duLduLduL']:
    #    kwargs['in_field'] = True
    #    kwargs['adcp_on'] = True
    #    skip_vel = False

    # Load dataset
    #profilers = glider_io.load_dataset(dataset)

    # Outfile
    if iz >= 0:
        outfile = f'{outroot}_z{(iz+1)*10}_{dataset}_{variables}.png'
    else:
        outfile = f'{outroot}_iso{np.abs(iz)}_{dataset}_{variables}.png'
    if stretch:
        outfile = outfile.replace('.png', '_stretch.png')

    # Load
    if variables != 'duLduLduL':
        raise NotImplementedError('Not ready for these variablaes')

    '''
    # Cut on valid velocity data 
    #nbins = 20
    #rbins = 10**np.linspace(0., np.log10(400), nbins) # km
    rbins = data_utils.rbinning(binning, dataset)
    if binning == 'log':
        rbins = data_utils.log_binning(dataset)
    elif binning == 'linear':
        rbins = data_utils.linear_binning(dataset)
    else:
        raise ValueError(f'Bad binning style: {binning}')

    # Generate pairs
    #gData = gliderdata.load_dataset(dataset)
    gPairs = profilerpairs.ProfilerPairs(
        profilers, 
        max_time=max_time,
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
    '''

    rdict = data_utils.load_SF(
        dataset=dataset, variables=variables, iz=iz, 
        minN=minN, btype=btype, max_time=max_time, 
        avoid_same_glider=avoid_same_glider)
    # Unpack
    Sn_dict = rdict['Sn_dict']
    goodN = rdict['goodN']
    Skeys = rdict['Skeys']
    rbins = rdict['rbins']

    #embed(header='fig_structure: 215')

    # Start the figure
    if stretch:
        fig = plt.figure(figsize=(19,4))
    else:
        fig = plt.figure(figsize=(19,6))
    plt.clf()
    gs = gridspec.GridSpec(1,3)

    goodN = np.array(Sn_dict['config']['N']) > minN
    

    '''
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
    '''


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


        if btype == 'log':
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
        #ax.grid()
        ax.grid(which='major', linewidth=0.8, alpha=0.7)
        ax.grid(which='minor', linewidth=0.5, alpha=0.3)
        if use_xlim:
            ax.set_xlim(use_xlim)
        if n == 2 and use_ylim is not None:
            ax.set_ylim(use_ylim)
        
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Figure 1  (Profile tracks)
    if flg == 1:
        fig_experiments()

    # Figure 1  (Profile tracks)
    if flg == 2:
        fig_separations(focus_dataset)

    # Figure 1  (First moment)
    if flg == 3:
        fig_single_order(focus_dataset, order=1)

    # Figure 4  (Second moment)
    if flg == 4:
        fig_single_order(focus_dataset, order=2)

    # Figure 5  (Third moment)
    if flg == 5:
        fig_single_order(focus_dataset, order=3)

    # Figure 6  (Corrected third moment)
    if flg == 6:
        fig_corrected_s3(focus_dataset,
                         f'fig_corrected_s3_{focus_dataset}.png')

    # Figure 7  (Other 1st/3rd)
    if flg == 7:
        other_datasets = datasets.copy()
        other_datasets.remove(focus_dataset)
        fig_S1S3_other(other_datasets)

    # Figure 8  (QG full grid)
    if flg == 8:
        fig_full_qg_SF()

    # Figure 9  (An example 100km, 100 day region)
    if flg == 9:
        #fig_qg_subregion_vs_full(x0=500, y0=500)  # Negative duL
        #fig_qg_subregion_vs_full(x0=300, y0=500)  # Negative duL
        fig_qg_subregion_vs_full(x0=300, y0=400)  # Negative duL


    # Figure ??
    if flg == 20:
        #dataset = 'ARCTERX-2025'
        dataset = 'Calypso2022'
        fig_structure(dataset, avoid_same_glider=True,
                      btype='log', show_correct=False,
                      use_xlim=(None, 100))
        fig_structure(dataset, avoid_same_glider=True,
                      outroot='fig_structure_lin',
                      btype='lin', show_correct=False)
                      #use_xlim=(None, 100))

    # #########################################################
    # APPENDIX
    # #########################################################

    # Figure 100  (Separation histograms for the others)
    if flg == 100:
        fig_histogram_dr(log_rbins=False)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)