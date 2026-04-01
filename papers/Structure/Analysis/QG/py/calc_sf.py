# Calculate the structure function of the velocity field for a drifter using the profiler package.  Up to third order.

import numpy as np
import json
import qg_io
import structure_function

from profiler import io as p_io

from IPython import embed

nbins = 20
rbins = 10**np.linspace(0., np.log10(400), nbins) # km

def calc_drifter_sf(drifter_file:str, outfile:str):

    # Load up
    traj, meta = qg_io.load_trajectories(drifter_file)

    # Calculate the structure function
    Sn_LL, Sn_TT = structure_function.run_profiler_approach(
        traj, meta, r_bins_km=rbins)


    # Write to JSON
    jdict = p_io.jsonify(Sn_LL)
    p_io.savejson(outfile, jdict, easy_to_read=True, overwrite=True)
    print(f'Saved: {outfile}')

# Command line execution
if __name__ == "__main__":
    calc_drifter_sf(drifter_file='data/small_box_drifters_ts5001_nd100.csv',
        outfile='data/small_box_drifters_ts5001_nd100_sf.json')