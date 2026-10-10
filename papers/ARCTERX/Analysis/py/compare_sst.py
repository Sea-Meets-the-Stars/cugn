""" Compare SST between satellite and ship """

import os
import glob
from importlib import reload

import numpy as np

import xarray
import pandas
import healpy
from datetime import datetime, timedelta, timezone

from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
import seaborn as sns

from remote_sensing.healpix import rs_healpix
from remote_sensing.healpix import utils as rshp_utils 
from remote_sensing.plotting import utils as plot_utils
from remote_sensing import io as rs_io

import ship

from IPython import embed

lon_lim = (127.,134)
lat_lim = (18.,23)

def explore_microwave(outfile:str='Compare_SST_AMSR2_ship.json',
                      t_near:pandas.Timedelta=pandas.Timedelta('2 hours'),
                      save_ship:str=None,
                      load_ship_file:str=None,
                      debug:bool=False):

    if load_ship_file is not None:
        print(f"Loading the ship data from {load_ship_file}")
        ship_data = np.load(load_ship_file)
        ship_t = ship_data['ship_t']
        # Convert to pandas
        ship_t = pandas.to_datetime(ship_t)
        #
        ship_lons, ship_lats = ship_data['ship_lons'], ship_data['ship_lats']
        Tinlet = ship_data['Tinlet']
    else:
        print("Reading the ship data from the shared drive")
        # Load up the ship
        ship_data = ship.load_ship()

        # Unfurl
        ship_t = pandas.to_datetime(ship_data.time.values)
        ship_lons, ship_lats = ship_data.lon.values, ship_data.lat.values
        Tinlet = ship_data.temperatureInlet.values

    # Nans
    ok_lons = np.isfinite(ship_lons)
    ok_lats = np.isfinite(ship_lats)
    ok_times = np.isfinite(ship_t)

    if save_ship is not None:
        np.savez(save_ship, ship_t=ship_t, ship_lons=ship_lons, ship_lats=ship_lats, Tinlet=Tinlet)

    # Load up the satellite files
    amsr2_files = glob.glob(os.path.join(
        os.getenv('OS_RS'), 'PODAAC', 
        'AMSR2-REMSS-L2P_RT-v8.2', 
        '2025*AMSR2-L2B_*'))

    # Loop
    sv_dict = {}
    for amsr2_file in amsr2_files:
        print(f"Working on: {os.path.basename(amsr2_file)}")

        # Load me
        amsr2 = rs_healpix.RS_Healpix.from_dataset_file(
            amsr2_file,  'sea_surface_temperature',
            time_isel=0, resol_km=11., 
            lat_slice=lat_lim, lon_slice=lon_lim)

        # Find overlap in time
        dt = ship_t - amsr2.time
        close_t = (np.abs(dt) < t_near) & ok_lons & ok_lats & ok_times

        # Cut on space
        ok_latlon = np.invert(rshp_utils.check_masked(
            amsr2.hp, ship_lons[close_t], ship_lats[close_t]))

        # Ok T?
        okT = np.isfinite(Tinlet[close_t])
        all_ok = ok_latlon & okT

        if not np.any(all_ok):
            print(f'No overlap for {amsr2_file}')
            continue

        # Compare
        amsr2T_at_ship = healpy.pixelfunc.get_interp_val(
            amsr2.hp, ship_lons[close_t][all_ok], 
            ship_lats[close_t][all_ok], lonlat=True)


        # DT
        DT = Tinlet[close_t][all_ok] - amsr2T_at_ship

        # Hour
        local_time = amsr2.time.tz_localize('UTC').tz_convert('Asia/Tokyo')
        hour = local_time.hour + local_time.minute/60 + local_time.second/3600

        #
        if debug:
            # Plot amsr2
            amsr2.plot(figsize=(10.,6), cmap='jet', 
                    lon_lim=lon_lim, lat_lim=lat_lim, 
                    projection='platecarree', ssize=40., 
                    vmin=23.7, vmax=27., show=True)            
            embed(header='87 of compare_sst')

        # Save
        sv_dict[os.path.basename(amsr2_file)] = {
            'datetime': str(amsr2.time),
            'hour': hour,
            'DT': DT, 
            'medianDT': np.median(DT), 
            'meanDT': np.mean(DT),
            'stdDT': np.std(DT),
            'median_lat': np.median(ship_lats[close_t][all_ok]),
            'median_lon': np.median(ship_lons[close_t][all_ok]),
            'Dlat': np.max(ship_lats[close_t][all_ok]) - np.min(ship_lats[close_t][ok_latlon]),
            'Dlon': np.max(ship_lons[close_t][all_ok]) - np.min(ship_lons[close_t][ok_latlon])
        }

    # Write
    jdict = rs_io.jsonify(sv_dict)
    rs_io.savejson(outfile, jdict)
    print(f'Wrote: {outfile}')

if __name__ == '__main__':

    # Grab time
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    #explore_microwave(save_ship=f'ship_data_{now}')
    explore_microwave(load_ship_file='ship_data_2025-02-16T232403Z.npz',
                      debug=False)