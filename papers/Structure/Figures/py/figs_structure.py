""" Figures for the structure function paper. """


# imports
import os
import sys
from importlib import resources

import numpy as np

import torch
import corner
import xarray

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import seaborn as sns

from ocpy.utils import plotting

from cugn import gliderdata
from cugn import gliderpairs
from cugn import io as cugn_io
from cugn import utils as cugn_utils

from IPython import embed

Sn_lbls = dict(
    S1=r'$<\delta u_L> \;\; \rm [m/s]$',
    S1_duL=r'$<\delta u_L> \;\; \rm [m/s]$',
    S2=r'$<\delta u_L^2> \;\; \rm [m/s]^2$',
    S3=r'$<\delta u_L^3> \;\; \rm [m/s]^3$',
)
Sn_lbls['S2_dS**2'] = r'$<\delta S^2> \;\; \rm [m/s]^2$'
Sn_lbls['S3_duLdSdS'] = r'$<\delta u_L \delta S^2> \;\; \rm [m/s]^2$'

def fig_separations(dataset:str, outroot='fig_sep', max_time:float=10.):
    outfile = f'{outroot}_{dataset}.png'

    # Load dataset
    gData = gliderdata.load_dataset(dataset)
    
    # Cut on valid velocity data 
    gData = gData.cut_on_good_velocity()

    # Generate pairs
    gPairs = gliderpairs.GliderPairs(gData, max_time=max_time)

    # Start the figure
    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # Lat/lon
    ax_ll = plt.subplot(gs[0])

    for mid in np.unique(gPairs.data('missid',2)):
        idx = gPairs.data('missid',2) == mid
        ax_ll.scatter(gPairs.data('lon', 2)[idx], 
            gPairs.data('lat', 2)[idx], s=1, label=f'MID={mid}')

    ax_ll.set_xlabel('Longitude [deg]')
    ax_ll.set_ylabel('Latitude [deg]')
    ax_ll.legend(fontsize=15)

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
    gData = gliderdata.load_dataset(dataset)
    
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
    gData = gliderdata.load_dataset(dataset)
    
    # Cut on valid velocity data 
    gData = gData.cut_on_good_velocity()

    # Generate pairs
    gPairs = gliderpairs.GliderPairs(gData, max_time=max_time)

    # Velocity
    gPairs.calc_velocity(iz)

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
                  iz:int=5, 
                  minN:int=10, avoid_same_glider:bool=True):

    # Outfile
    outfile = f'{outroot}_z{(iz+1)*10}_{dataset}_{variables}.png'

    '''
    # Load dataset
    gData = gliderdata.load_dataset(dataset)
    
    # Cut on valid velocity data 
    gData = gData.cut_on_good_velocity()

    # Generate pairs
    gPairs = gliderpairs.GliderPairs(
        gData, max_time=max_time, 
        avoid_same_glider=avoid_same_glider)

    # Velocity
    gPairs.calc_velocity(iz)

    rbins = 10**np.linspace(0., np.log10(400), nbins) # km
    Sn_dict = gPairs.calc_Sn_vs_r(rbins, nboot=10000)
    gPairs.calc_corr_Sn(Sn_dict)
    '''

    # Load
    gpair_file = cugn_io.gpair_filename(dataset, iz, variables)
    gpair_file = os.path.join('..', 'Analysis', 'Outputs', gpair_file)

    Sn_dict = gliderpairs.load_Sndict(gpair_file)

    #embed(header='fig_structure: 215')

    # Start the figure
    fig = plt.figure(figsize=(19,6))
    plt.clf()
    gs = gridspec.GridSpec(1,3)

    goodN = Sn_dict['N'] > minN
    

    # Generate the keys
    if variables == 'duLduLduL':
        Skeys = ['S1', 'S2', 'S3']
    elif variables == 'duLdSdS':
        Skeys = ['S1_duL', 'S2_dS**2', 'S3_'+variables]
    elif variables == 'duLdTdT':
        Skeys = ['S1_duL', 'S2_dT**2', 'S3_'+variables]


    for n, clr in enumerate('krb'):
        ax = plt.subplot(gs[n])
        Skey = Skeys[n] 
        ax.errorbar(Sn_dict['r'][goodN], 
                    Sn_dict[Skey][goodN], 
                    yerr=Sn_dict['err_'+Skey][goodN],
                    color=clr,
                    fmt='o', capsize=5)  # fmt defines marker style, capsize sets error bar cap length

        # Corrected
        if n > 0:
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
            ax.text(0.1, 0.8, 
                    f'{dataset}\n depth = {(iz+1)*10} m\nAvoid same glider? {same_glider}\n {variables}', 
                transform=ax.transAxes, fontsize=16, ha='left')
        # 0 line
        ax.axhline(0., color='red', linestyle='--')

        plotting.set_fontsize(ax, 19) 
        ax.grid()
        
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
        #fig_structure('Calypso2022')
        fig_structure('Calypso2022', variables='duLdSdS')

    # Calypso 2022
    if flg == 5:
        dataset = 'Calypso2022'
        #fig_separations(dataset)
        #fig_dtimes(dataset)
        #fig_dus(dataset)
        fig_structure(dataset, avoid_same_glider=False)
        fig_structure(dataset, avoid_same_glider=False, iz=10)

    # Sn with depth
    if flg == 6:
        fig_Sn_depth('Calypso2022')

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)