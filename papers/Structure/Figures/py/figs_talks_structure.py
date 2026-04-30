""" Figures for development of the structure function paper. """


# imports
import os
import sys
import glob
import numpy as np
from importlib import resources

from matplotlib import pyplot as plt

from ocpy.utils import plotting

from IPython import embed

from cugn import plotting as cugn_plotting

# Local
import figs_structure

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import data_utils

Sn_lbls = cugn_plotting.Sn_lbls

def fig_all_s3(datasets:list, outfile:str):

    # Combine the figures
    fig = plt.figure(figsize=(12, 6))
    plt.clf()
    ax = plt.gca()

    for ss, dataset in enumerate(datasets):
        rdict = data_utils.load_SF(dataset, variables='duLduLduL')
        # Unpack
        Sn_dict = rdict['Sn_dict']
        goodN = rdict['goodN']

        # Trim Calypso2019
        if dataset == 'Calypso2019':
            goodN[-4] = False
        
        Skey = rdict['Skeys'][2]
        ax.plot(Sn_dict['r'][goodN], 
                    Sn_dict[Skey][goodN], 
                label=dataset)
                    #yerr=Sn_dict['err_'+Skey][goodN],
                    #color='k', fmt='o', capsize=5)
    ax.legend(fontsize=12)
    ax.set_ylabel(Sn_lbls[Skey])
    ax.grid()
    ax.set_xscale('log')
    ax.set_xlabel('Separation (km)')
    plotting.set_fontsize(ax, 19) 
    ax.axhline(0., color='k', linestyle='--')

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_corrected_s3(dataset:str, outfile:str):

    rdict = data_utils.load_SF(dataset, variables='duLduLduL')
    # Unpack
    Sn_dict = rdict['Sn_dict']
    goodN = rdict['goodN']

    # Trim Calypso2019
    if dataset == 'Calypso2019':
        goodN[-4] = False
    
    Skey = rdict['Skeys'][2]

    # Combine the figures
    fig = plt.figure(figsize=(12, 6))
    plt.clf()
    ax = plt.gca()

    # Corrected
    clr = 'b'
    corr_key = Skey[0:2]+'corr'+Skey[2:]
    ax.errorbar(Sn_dict['r'][goodN], 
            Sn_dict[corr_key][goodN],  
            yerr=Sn_dict['err_'+Skey][goodN],
            fmt='o',
            color=clr, label='Corrected')

    # Original
    ax.plot(Sn_dict['r'][goodN], 
                Sn_dict[Skey][goodN], 'x', color=clr, label='Original')
            #label=dataset)
                    #yerr=Sn_dict['err_'+Skey][goodN],
                    #color='k', fmt='o', capsize=5)
    ax.legend(fontsize=17, loc='upper left')
    ax.set_ylabel(Sn_lbls[Skey])
    ax.grid()
    ax.set_xscale('log')
    ax.set_xlabel('Separation (km)')
    plotting.set_fontsize(ax, 19) 
    ax.axhline(0., color='k', linestyle='--')

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Tracks and separations
    if flg == 1:
        #fig_separations('ARCTERX-2023', fsz=12, ncol=1)
        #fig_separations('Calypso2019', max_time=10.)
        figs_structure.fig_separations('Calypso2022', max_time=10.)
    
    # Structure functions
    if flg == 2:
        #dataset = 'Calypso2019'
        #dataset = 'ARCTERX'
        #dataset = 'ARCTERX-2023'
        avoid_same_glider = True

        dataset = 'Calypso2022'
        figs_structure.fig_structure(dataset, avoid_same_glider=avoid_same_glider,
            show_correct=False, outfile=f'fig_structure_{dataset}_no_correct.png',
            no_labeling=True, use_xlim=(1.7, 1e2))

    # All s3
    if flg == 3:
        datasets = ['Calypso2019', 'ARCTERX-2023', 'Calypso2022']
        fig_all_s3(datasets, 'fig_all_s3.png')

    # Corrected s3
    if flg == 4:
        fig_corrected_s3('ARCTERX-2023', 'fig_ARCTERX-2023_corrected_s3.png')

    # 100 days, 100km
    if flg == 5:
        # 100 days, 100km
        #figs_structure.fig_qg_subregion_vs_full(Ndays=100,
        #    outfile='fig_qg_100days_100km_400_400.png',
        #    llocs=('lower left', None, 'lower left'))

        # 1825 days, 100km
        #figs_structure.fig_qg_subregion_vs_full(Ndays=5*365,
        #    x0=300, y0=300,
        #    outfile='fig_qg_5years_100km_300_300.png',
        #    llocs=('lower left', None, 'lower left'))

        # 100 days, 200km
        #figs_structure.fig_qg_subregion_vs_full(Ndays=100, dx=200,
        #    outfile='fig_qg_100days_200km_400_400.png',
        #    llocs=('lower left', None, 'lower left'))

        # 1000 days, 200km 
        #figs_structure.fig_qg_subregion_vs_full(Ndays=1000, dx=200,
        #    outfile='fig_qg_1000days_200km_400_400.png',
        #    llocs=('lower left', None, 'lower left'))

        # 5 years, 200km 
        #figs_structure.fig_qg_subregion_vs_full(Ndays=5*365, dx=200,
        #    x0=200, y0=200,
        #    outfile='fig_qg_5years_200km_200_200.png',
        #    llocs=('lower left', None, 'lower left'))
        #figs_structure.fig_qg_subregion_vs_full(Ndays=5*365, dx=200,
        #    x0=400, y0=400,
        #    outfile='fig_qg_5years_200km_400_400.png',
        #    llocs=('lower left', None, 'lower left'))

        # 1000 days, 200km; 400, 600
        #figs_structure.fig_qg_subregion_vs_full(Ndays=1000, dx=200,
        #    y0=600,
        #    outfile='fig_qg_1000days_200km_400_600.png',
        #    llocs=('lower left', None, 'lower left'))

        # 1000 days, 300km; 
        #figs_structure.fig_qg_subregion_vs_full(Ndays=1000, dx=300,
        #    x0=200, y0=200,
        #    outfile='fig_qg_1000days_300km_200_200.png',
        #    llocs=('lower left', None, 'lower left'))

        # 5 years, 300km; 300km
        figs_structure.fig_qg_subregion_vs_full(Ndays=5*365, dx=300,
            x0=200, y0=200,
            outfile='fig_qg_5years_300km_200_200.png',
            llocs=('lower left', None, 'lower left'))

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)