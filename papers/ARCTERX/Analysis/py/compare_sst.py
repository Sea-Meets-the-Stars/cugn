""" Compare SST between satellite and ship """

import os
import glob
from importlib import reload

import numpy as np

import xarray
import pandas
import healpy

from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
import seaborn as sns

from remote_sensing.healpix import rs_healpix
from remote_sensing.plotting import utils as plot_utils

import ship

def explore_microwave(t_near:pandas.Timedelta('2 hours')):

    # Load up the ship
    ship_data = ship.load_ship()

    # Unfurl
    ship_t = pandas.to_datetime(ship_data.time.values)
    ship_lons, ship_lats = ship_data.lon.values[close_t], ship_data.lat.values[close_t] 

    # Load up the satellite files
    amsr2_files = glob.glob(os.path.join(
        os.getenv('OS_RS'), 'PODAAC', 
        'AMSR2-REMSS-L2P_RT-v8.2', 
        '20250210161056-REMSS-L2P_GHRSST-SSTsubskin-AMSR2-L2B_*'))

    # Loop
    for amsr2_file in amsr2_files:

        # Load me
        amsr2 = rs_healpix.RS_Healpix.from_dataset_file(
            amsr2_file,  'sea_surface_temperature',
            time_isel=0, resol_km=11., 
            lat_slice=(18,23.),  lon_slice=(127., 134.))

        # Find overlap in time
        dt = ship_t - amsr2.time
        close_t = np.abs(dt) < t_near

        # Cut on space