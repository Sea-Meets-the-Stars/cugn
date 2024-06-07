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

from oceancolor.utils import plotting

from cugn import gliderdata
from cugn import gliderpairs

from IPython import embed

def fig_separations(dataset:str, outroot='fig_sep'):
    outfile = f'{outroot}_{dataset}.png'

    # Load dataset
    gData = gliderdata.load_dataset(dataset)
    
    # Cut on valid velocity data 
    gData = gData.cut_on_good_velocity()

    # Generate pairs
    gPairs = gliderpairs.GliderPairs(gData, max_time=5.)

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
    ax_r.text(0.1, 0.9, dataset, transform=ax_r.transAxes, fontsize=15)

    for ax in [ax_ll, ax_r]:
        plotting.set_fontsize(ax, 15) 
        
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Indiv
    if flg == 1:
        fig_separations('ARCTERX')

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)