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


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Figure 1  (Profile tracks)
    if flg == 1:
        fig_experiments()


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0

        #flg = 1
        
    else:
        flg = sys.argv[1]

    main(flg)