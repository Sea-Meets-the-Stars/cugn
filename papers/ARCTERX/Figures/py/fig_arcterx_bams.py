""" Figures for the ARCTERX BAMS paper
    Temperature structure functions for Spray, Slocum, and Seaglider data.
"""

# imports
import os
import sys

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from profiler import profilerpairs

from cugn import plotting as cugn_plotting

# Local imports
sys.path.append(os.path.abspath("../Analysis/py"))
from load_profilers import load_by_asset
import arcterx_utils

from IPython import embed

Sn_lbls = cugn_plotting.Sn_lbls


def calc_temperature_structure(assets: list,
                               iz: int = 5,
                               max_time: float = 10.,
                               log_rbins: bool = True,
                               avoid_same_glider: bool = True,
                               restrict: bool = True,
                               debug: bool = False):
    """
    Calculate temperature structure functions for the given assets.

    Args:
        assets (list): List of asset names to load (e.g., ['Spray', 'Slocum', 'Seaglider'])
        iz (int): Depth index (depth = (iz+1)*10 meters). Default 5 = 60m.
        max_time (float): Maximum time separation in hours. Default 10.
        log_rbins (bool): Use logarithmic radial bins. Default True.
        avoid_same_glider (bool): Avoid pairing same glider with itself. Default True.
        restrict (bool): Restrict the gliders to lie in 100km box
        debug (bool): Enable debug mode. Default False.

    Returns:
        tuple: (profilers, Sn_dict, gPairs, rbins)
    """
    variables = 'dTdTdT'

    # Load profilers
    profilers = load_by_asset(assets)

    # Restrict in box
    if restrict:
        arcterx_utils.restrict_to_arcterx_box(profilers)

    # Set up radial bins
    nbins = 20
    if log_rbins:
        rbins = 10**np.linspace(0., np.log10(400), nbins)  # km
    else:
        rbins = np.linspace(0, 100*np.sqrt(2), nbins)

    # Generate pairs
    gPairs = profilerpairs.ProfilerPairs(
        profilers, max_time=max_time,
        avoid_same_glider=avoid_same_glider,
        cen_latlon=(arcterx_utils.Leg2_lat_box, arcterx_utils.Leg2_lon_box),
        remove_nans=True, randomize=True,
        debug=debug)

    # Calculate delta T
    gPairs.calc_delta(iz, variables, skip_velocity=True)
    gPairs.calc_Sn(variables)

    # Calculate structure functions vs separation
    Sn_dict = gPairs.calc_Sn_vs_r(rbins)
    gPairs.calc_corr_Sn(Sn_dict)
    gPairs.add_meta(Sn_dict)

    return profilers, Sn_dict, gPairs, rbins


def fig_temperature_structure(outfile: str = 'fig_bams_temp_structure.png',
                              assets: list = None,
                              iz: int = 5,
                              max_time: float = 10.,
                              log_rbins: bool = True,
                              avoid_same_glider: bool = True,
                              minN: int = 10,
                              ylog: bool = False,
                              debug: bool = False):
    """
    Plot temperature structure functions (S1, S2, S3) for the given assets.

    Args:
        outfile (str): Output filename for the figure.
        assets (list): List of assets to use. Default ['Spray', 'Slocum', 'Seaglider'].
        iz (int): Depth index (depth = (iz+1)*10 meters). Default 5 = 60m.
        max_time (float): Maximum time separation in hours. Default 10.
        log_rbins (bool): Use logarithmic radial bins. Default True.
        avoid_same_glider (bool): Avoid pairing same glider with itself. Default True.
        minN (int): Minimum number of pairs per bin for plotting. Default 10.
        ylog (bool): Use log10 on y-axis
        debug (bool): Enable debug mode. Default False.
    """
    if assets is None:
        assets = ['Spray', 'Slocum', 'Seaglider']

    # Calculate structure functions
    profilers, Sn_dict, gPairs, rbins = calc_temperature_structure(
        assets, iz=iz, max_time=max_time, log_rbins=log_rbins,
        avoid_same_glider=avoid_same_glider, debug=debug)

    # Define structure function keys
    Skeys = ['S1_dT', 'S2_dT**2', 'S3_dTdTdT']

    # Start the figure
    fig = plt.figure(figsize=(19, 6))
    plt.clf()
    gs = gridspec.GridSpec(1, 3)

    goodN = np.array(Sn_dict['config']['N']) > minN

    for n, clr in enumerate('krb'):
        ax = plt.subplot(gs[n])
        Skey = Skeys[n]

        # Plot with error bars
        ax.errorbar(Sn_dict['r'][goodN],
                    Sn_dict[Skey][goodN],
                    yerr=Sn_dict['err_'+Skey][goodN],
                    color=clr,
                    fmt='o', capsize=5)

        if log_rbins:
            ax.set_xscale('log')

        ax.set_xlabel('Separation (km)')
        ax.set_ylabel(Sn_lbls[Skey])

        # Add label on third panel
        if n == 2:
            same_lbl = 'True' if avoid_same_glider else 'False'
            lbl = (f'Assets: {", ".join(assets)}\n'
                   f'depth = {(iz+1)*10} m\n'
                   f't < {int(max_time)} hr\n'
                   f'Avoid same glider? {same_lbl}')
            ax.text(0.1, 0.75, lbl,
                    transform=ax.transAxes, fontsize=14, ha='left')

        # Reference line at zero
        ax.axhline(0., color='red', linestyle='--')

        cugn_plotting.set_fontsize(ax, 19)
        ax.grid()

        # Log scale for S2
        if n == 1 and ylog:
            ax.set_yscale('log')
            ax.set_ylim(1e-3, 1.)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_temperature_structure_by_asset(outroot: str = 'fig_bams_temp_struct',
                                       iz: int = 5,
                                       max_time: float = 10.,
                                       log_rbins: bool = True,
                                       avoid_same_glider: bool = True,
                                       minN: int = 10,
                                       debug: bool = False):
    """
    Plot temperature structure functions separately for each asset type
    (Spray, Slocum, Seaglider) and combined.

    Args:
        outroot (str): Root name for output files.
        iz (int): Depth index (depth = (iz+1)*10 meters). Default 5 = 60m.
        max_time (float): Maximum time separation in hours. Default 10.
        log_rbins (bool): Use logarithmic radial bins. Default True.
        avoid_same_glider (bool): Avoid pairing same glider with itself. Default True.
        minN (int): Minimum number of pairs per bin for plotting. Default 10.
        debug (bool): Enable debug mode. Default False.
    """
    asset_groups = [
        ['Spray'],
        ['Slocum'],
        ['Seaglider'],
        ['Spray', 'Slocum', 'Seaglider'],
    ]

    for assets in asset_groups:
        asset_str = '_'.join(assets)
        outfile = f'{outroot}_{asset_str}_z{(iz+1)*10}.png'

        fig_temperature_structure(
            outfile=outfile,
            assets=assets,
            iz=iz,
            max_time=max_time,
            log_rbins=log_rbins,
            avoid_same_glider=avoid_same_glider,
            minN=minN,
            debug=debug)


def fig_compare_assets(outfile: str = 'fig_bams_compare_assets.png',
                       iz: int = 5,
                       max_time: float = 10.,
                       log_rbins: bool = True,
                       avoid_same_glider: bool = True,
                       minN: int = 10,
                       debug: bool = False):
    """
    Compare temperature structure functions across different assets on a single plot.

    Args:
        outfile (str): Output filename for the figure.
        iz (int): Depth index (depth = (iz+1)*10 meters). Default 5 = 60m.
        max_time (float): Maximum time separation in hours. Default 10.
        log_rbins (bool): Use logarithmic radial bins. Default True.
        avoid_same_glider (bool): Avoid pairing same glider with itself. Default True.
        minN (int): Minimum number of pairs per bin for plotting. Default 10.
        debug (bool): Enable debug mode. Default False.
    """
    asset_groups = {
        'Spray': 'blue',
        'Slocum': 'green',
        'Seaglider': 'orange',
    }

    # Start the figure
    fig = plt.figure(figsize=(19, 6))
    plt.clf()
    gs = gridspec.GridSpec(1, 3)

    Skeys = ['S1_dT', 'S2_dT**2', 'S3_dTdTdT']
    axes = [plt.subplot(gs[n]) for n in range(3)]

    for asset_name, color in asset_groups.items():
        try:
            profilers, Sn_dict, gPairs, rbins = calc_temperature_structure(
                [asset_name], iz=iz, max_time=max_time, log_rbins=log_rbins,
                avoid_same_glider=avoid_same_glider, debug=debug)

            goodN = np.array(Sn_dict['config']['N']) > minN

            for n, ax in enumerate(axes):
                Skey = Skeys[n]
                ax.errorbar(Sn_dict['r'][goodN],
                            Sn_dict[Skey][goodN],
                            yerr=Sn_dict['err_'+Skey][goodN],
                            color=color,
                            fmt='o', capsize=3, label=asset_name)
        except Exception as e:
            print(f"Warning: Could not process {asset_name}: {e}")
            continue

    # Format axes
    for n, ax in enumerate(axes):
        Skey = Skeys[n]
        if log_rbins:
            ax.set_xscale('log')
        ax.set_xlabel('Separation (km)')
        ax.set_ylabel(Sn_lbls[Skey])
        ax.axhline(0., color='red', linestyle='--')
        cugn_plotting.set_fontsize(ax, 19)
        ax.grid()

        if n == 0:
            ax.legend(fontsize=14, loc='upper left')

        # Log scale for S2
        if n == 1:
            ax.set_yscale('log')
            ax.set_ylim(1e-3, 1.)

        # Add label on third panel
        if n == 2:
            same_lbl = 'True' if avoid_same_glider else 'False'
            lbl = (f'depth = {(iz+1)*10} m\n'
                   f't < {int(max_time)} hr\n'
                   f'Avoid same glider? {same_lbl}')
            ax.text(0.1, 0.75, lbl,
                    transform=ax.transAxes, fontsize=14, ha='left')

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def main(flg):
    if flg == 'all':
        flg = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg = int(flg)

    # Combined Spray, Slocum, Seaglider temperature structure
    if flg == 0:
        fig_temperature_structure(log_rbins=False)

    # Individual asset temperature structure functions
    if flg == 1:
        fig_temperature_structure_by_asset()

    # Compare assets on single plot
    if flg == 2:
        fig_compare_assets()

    # Spray only
    if flg == 3:
        fig_temperature_structure(
            outfile='fig_bams_temp_struct_Spray.png',
            assets=['Spray'])

    # Slocum only
    if flg == 4:
        fig_temperature_structure(
            outfile='fig_bams_temp_struct_Slocum.png',
            assets=['Slocum'])

    # Seaglider only
    if flg == 5:
        fig_temperature_structure(
            outfile='fig_bams_temp_struct_Seaglider.png',
            assets=['Seaglider'])

    # Different depth (100m)
    if flg == 6:
        fig_temperature_structure(
            outfile='fig_bams_temp_struct_z100.png',
            iz=9)  # 100m

    # Linear r bins
    if flg == 7:
        fig_temperature_structure(
            outfile='fig_bams_temp_struct_linr.png',
            log_rbins=False)


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg = 0
    else:
        flg = sys.argv[1]

    main(flg)
