""" Routines for ARCTERX analysis."""

import numpy as np

from profiler.utils import offsets
from IPython import embed


def restrict_to_arcterx_box(profilers:list, 
                            boxsize:float=50.,
                 lat_box:float=20.3333, 
                 lon_box:float=129.9167):

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