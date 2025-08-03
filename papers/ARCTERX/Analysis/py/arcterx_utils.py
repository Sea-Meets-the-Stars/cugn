""" Routines for ARCTERX analysis."""

import os
import numpy as np

from profiler.utils import offsets
from profiler import profilerpairs

from cugn import io as cugn_io

from load_profilers import load_by_asset

from IPython import embed

# Center of the box
lat_box=20.3333 
lon_box=129.9167

def calc_structure(dataset, variables:str, assets:list,
                   iz:int, max_time:float,
                   log_rbins:bool=False,
                   avoid_same_glider:bool=True,
                   skip_vel:bool=True, debug:bool=False):
    """ Calculate structure functions for ARCTERX data.
    """

    # Set in_field=True to load in-field data
    kwargs = {}
    if variables in ['duLduLduL']:
        kwargs['in_field'] = True
        kwargs['adcp_on'] = True
        skip_vel = False
    profilers = load_by_asset(assets, **kwargs)

    # Load
    if iz >= 0:
        gpair_file = cugn_io.gpair_filename(
            dataset, iz, not avoid_same_glider)
        gpair_file = os.path.join('..', 'Analysis', 'Outputs', gpair_file)

    if variables not in ['duLduLduL', 'dTdTdT']:
        raise NotImplementedError('Not ready for these variablaes')

    # Cut on valid velocity data 
    nbins = 20
    if log_rbins:
        rbins = 10**np.linspace(0., np.log10(400), nbins) # km
    else:
        rbins = np.linspace(0,100*np.sqrt(2),nbins);
        #rbins = np.linspace(0., 200, nbins) # km
        #embed(header='257 of figs')

    #embed(header='fig_structure: 253')
    gPairs = profilerpairs.ProfilerPairs(
        profilers, max_time=max_time, 
        avoid_same_glider=avoid_same_glider,
        cen_latlon=(lat_box, lon_box),
        remove_nans=True, #randomize=False,
        debug=debug)
    # Isopycnals?
    if iz < 0:
        gPairs.prep_isopycnals('t')
    gPairs.calc_delta(iz, variables, skip_velocity=skip_vel)
    gPairs.calc_Sn(variables)

    Sn_dict = gPairs.calc_Sn_vs_r(rbins)#, nboot=100)
    gPairs.calc_corr_Sn(Sn_dict) 
    gPairs.add_meta(Sn_dict)

    # Return
    return profilers, Sn_dict, gPairs, rbins

def restrict_to_arcterx_box(profilers:list, 
    in_latlon:tuple=None, boxsize:float=50.):

    if in_latlon is not None:
        lat_box, lon_box = in_latlon

    # Restrict to a box
    for profiler in profilers:
        # Calcualte dist, offset
        dist, offset = offsets.calc_dist_offset(
            profiler.lon, profiler.lat,
            endpoints=((lon_box, lon_box), (lat_box, lat_box+1)),
            debug=False)

        # Lie within the box
        good = ((dist < boxsize) & (offset < boxsize) &
                (dist > -boxsize) & (offset > -boxsize))
        #if np.any(~good):
        #    print(f"Restricting {profiler.__class__.__name__} to ARCTERX box")
        #    embed(header='Restricting to ARCTERX box')
        
        # Restrict
        profiler.profile_subset(good, init=False)

        # Report max dist, offset
        print(f'Maximum dist: {np.max(dist[good]):.2f} km, '
              f'maximum offset: {np.max(offset[good]):.2f} km')
        # Report number of profiles
        print(f'Number of profiles after restriction: {np.sum(good)}')