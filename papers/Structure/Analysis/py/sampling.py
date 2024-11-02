""" Codes to sample ARCTERX with gliders"""
import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator 

import seaborn as sns

from cugn import plotting

from IPython import embed

def circle_sep(used:int, xy_rand:np.ndarray, npoints:int):
    # Grab em
    rand1 = xy_rand[used:used+npoints]
    used += npoints
    rand2 = xy_rand[used:used+npoints]

    # Calc
    sep2 = (rand1[:,0]-rand2[:,0])**2 + (
        rand1[:,1]-rand2[:,1])**2

    # Return
    return used, np.sqrt(sep2)

def wedding_cake(ngliders:tuple=(4,4), 
                 outer:float=50., inner:float=20,
                 ndives=500, check:bool=True):

    ntot = np.sum(ngliders)

    # Draw random points in the outer circle
    xy_rand = 2*np.random.uniform(size=(ndives*ntot*1000, 2)) - 1.

    # Parse down to in circle
    in_circ = (xy_rand[:,0]**2 + xy_rand[:,1]**2) < 1.
    xy_rand = xy_rand[in_circ]
    used = 0

    # Generate outer circle
    nouter = ngliders[0]**2 * ndives
    used, outer_sep = circle_sep(used, xy_rand, nouter)
    outer_sep *= outer


    # Inner circle
    ninner = ngliders[1]**2 * ndives
    used, inner_sep = circle_sep(used, xy_rand, ninner)
    inner_sep *= inner

    # Now intermediate
    ninter = ngliders[1]*ngliders[0] * ndives
    rand1 = xy_rand[used:used+ninter] * outer
    used += ninter
    rand2 = xy_rand[used:used+ninter] * inner

    sep2 = (rand1[:,0]-rand2[:,0])**2 + (
        rand1[:,1]-rand2[:,1])**2
    inter_sep = np.sqrt(sep2)

    # Check
    if False:
        plt.clf()
        ax = plt.gca()
        sns.histplot(inter_sep, ax=ax)
        plotting.set_fontsize(ax, 16)
        plt.show()

    all_seps = np.concatenate([outer_sep,inter_sep,inner_sep])

    # Figure time
    fig = plt.figure(figsize=(10,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # Linear
    ax_linear = plt.subplot(gs[0])
    sns.histplot(all_seps, ax=ax_linear)

    # Log
    ax_log = plt.subplot(gs[1])
    sns.histplot(all_seps, ax=ax_log, log_scale=(True,False),
                 color='green')
    ax_log.set_xlim(1., 100.)

    for ax in [ax_linear, ax_log]:
        ax.set_xlabel('Separation (km)')
        plotting.set_fontsize(ax, 18)


    outfile = 'fig_wedding.png'
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

    embed(header='33 of wedding')
    


if __name__ == '__main__':
    wedding_cake()