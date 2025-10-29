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

def fig_experiments(outfile='fig_experiments.png', 
                    max_time:float=10.):

    # Start the figure
    fig = plt.figure(figsize=(10,10))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    datasets = ['Calypso2019', 'Calypso2022', 'ARCTERX-2023',
                'ARCTERX-2025']
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

        ax_ll.text(0.95, 0.02, dataset, transform=ax_ll.transAxes,
                     fontsize=17, ha='right', va='bottom')
        ax_ll.set_xlabel('Longitude [deg]')
        ax_ll.set_ylabel('Latitude [deg]')
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
        plotting.set_fontsize(ax_ll, 15) 

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_histogram_dr(outfile='fig_histogram_dr.png', 
                    max_time:float=10., log_rbins:bool=False):

    # Start the figure
    fig = plt.figure(figsize=(10,10))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    datasets = ['Calypso2019', 'Calypso2022', 'ARCTERX-2023',
                'ARCTERX-2025']
    clrs = ['blue', 'orange', 'green', 'red']
    for ss, dataset in enumerate(datasets):

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
                         ax=ax_r, color=clrs[ss])

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
    if variables != 'duLduLduL':
        raise NotImplementedError('Not ready for these variablaes')
    # Cut on valid velocity data 
    nbins = 20
    rbins = 10**np.linspace(0., np.log10(400), nbins) # km
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

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Figure 1  (Profile tracks)
    if flg == 1:
        fig_experiments()

    # Figure 2  (Separation histogram)
    if flg == 2:
        fig_histogram_dr(log_rbins=False)


    # Figure ??
    if flg == 20:
        dataset = 'ARCTERX-2025'
        fig_structure(dataset, avoid_same_glider=True)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)