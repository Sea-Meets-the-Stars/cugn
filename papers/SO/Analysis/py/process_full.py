""" Process full resolution data """

import os
import numpy as np
from glob import glob
import xarray

from cugn import defs as cugn_defs
from cugn import io as cugn_io
from cugn import highres as cugn_highres
from cugn import utils as cugn_utils

from IPython import embed

high_path = os.path.join(os.getenv('OS_SPRAY'), 'CUGN', 'HighRes')

def calc_mission(line:str, min_depth:float=2.0,
                  max_offset:float=90., warn_highres:bool=False,
               max_depth:float=100., debug:bool=False):
    """ Calculate MLD and N for a mission

    Args:
        min_depth (float, optional): Minimum depth. Defaults to 2.0.
        max_depth (float, optional): Maximum depth. Defaults to 100..

    Returns:
        np.array: MLDs
        np.array: Ns
        np.array: zNs
    """
    line_files = cugn_io.line_files(line)
    ds = xarray.load_dataset(line_files['datafile'])

    # Cut on offset
    dist, offset = cugn_utils.calc_dist_offset(
            line, ds.lon.values, ds.lat.values)
    ok_off = (np.abs(offset) < max_offset) & np.isfinite(offset)
    ds = ds.isel(profile=ok_off)

    # Missions
    uni_missions = np.unique(ds.mission_name.values.astype(str))

    for mission in uni_missions:
        gfiles = glob(os.path.join(high_path, f'SPRAY-FRSQ-{mission}-*.nc'))
        if len(gfiles) != 1:
            if warn_highres:
                print(f"Missing high res data for {mission}")
                continue
            else:
                raise ValueError(f"Missing high res data for {mission}")

        # Indices

        # High res
        ds_high = xarray.open_dataset(gfiles[0])

        lat = np.nanmedian(ds_high.latitude.data)
        lon = np.nanmedian(ds_high.longitude.data)

        salinity = ds_high.salinity.values
        temperature = ds_high.temperature.values
        oxygen = ds_high.doxy.values

        # QC
        good_sal = ds_high.salinity_qc.values.astype(int) == 1
        good_sal |= ds_high.salinity_qc.values.astype(int) == 3
        good_temp = ds_high.temperature_qc.values.astype(int) == 1
        good_temp |= ds_high.temperature_qc.values.astype(int) == 3


        mission_idx = ds.mission_name.values.astype(str).tolist().index(mission)
        gm_idx = ds.mission.values == mission_idx
        mission_profiles = np.unique(ds.mission_profile[gm_idx])

        for mission_profile in mission_profiles:
            #print(f'Working on {mission_name} {mission_profile}')
    
            # Find the obs
            my_obs = ds_high.profile_obs_index.values == mission_profile
            my_obs &= ds_high.depth.values > min_depth

            # Require finite
            my_obs &= np.isfinite(salinity) & np.isfinite(temperature)

            # Deal with QC
            my_obs &= good_sal & good_temp

            # Calculate 
            MLD, bin_means, z_Npeak, zN5, zN10, \
                Nf5, Nf10, NSO, extras = cugn_highres.calc_mld_N(
                ds_high.depth.values[my_obs],
                salinity[my_obs],
                temperature[my_obs],
                oxygen[my_obs],
                lat, lon, max_depth, debug=debug, return_extras=True)

            embed(header='cugn/highres.py: 88')


def main(flg, debug=False):

    if debug:
        calc_mission('90.0', debug=True, warn_highres=True)


    if flg == 1:
        items = cugn_defs.lines
        n_cores = 4
        map_fn = partial(calc_mission, warn_highres=True)
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
            answers = list(tqdm(executor.map(map_fn, items,
                                            chunksize=chunksize), total=len(items)))

# Command line execution
if __name__ == '__main__':
    import sys
    flg = int(sys.argv[1])

    main(flg, debug=True)
