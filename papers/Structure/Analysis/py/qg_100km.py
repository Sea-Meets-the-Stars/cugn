""" Calculations on a ~100 km region """

from importlib import reload
import os

import xarray

import numpy as np
# import fsspec
import matplotlib
import matplotlib.pyplot as plt
import gsw_xarray as gsw
from xhistogram.xarray import histogram

from dask.diagnostics import ProgressBar

from strucFunct2_ai import timescale

import qg_utils
import strucFunct2_ai

# Calculates structure functions
shiftdim = 'x','y'
grid = 'm'

def test_full(ndays=15, maxcorr=60):
    # Load
    qg, _ = qg_utils.load_qg()

    # Gets last 6 months of data
    nmonths = 6
    month = 30
    #yr = 365

    i1 = -2-month*nmonths
    time6m = np.arange(i1, i1 + month*nmonths)

    # Chunks data
    chx = len(qg.x)
    chy = len(qg.y)
    cht = len(qg.time)
    chunks = {'x': chx, 'y': chy, 'time': cht}

    # Selects the first level (surface)
    # Last 6 months
    Udsn = qg.isel(lev=0, time=time6m).chunk(chunks)

    # Grab the last 15 days
    SFtest = strucFunct2_ai.calculateSF_2(Udsn.isel(
        time=np.arange(0, ndays)), maxcorr, shiftdim, grid)

    # Higher order
    SF2, SF3 = strucFunct2_ai.SF2_3_ul(SFtest.ulls)

    # Slice the data to include the current chunk
    data_slice = SFtest.isel(time=slice(0,ndays))
        
    # Calculates du1, du2 and du3
    sf2, sf3 = strucFunct2_ai.SF2_3_ul(data_slice.ulls)#, data_slice.dut)
    data_slice['du2'] = sf2
    data_slice['du3'] = sf3
        
    # Averages over all $s$ positions
    with ProgressBar():
        data_avers = data_slice.mean(dim=('x','y'), skipna=True).compute()

    embed(header='67 of qg_100km')
    # Defines distance bins 
    dr = 5000 # meters
    rbins = np.arange(0, 1.8e5, dr) # 180 km
    mid_rbins = 0.5*(rbins[:-1] + rbins[1:])

    # Average over orientation
    dudlt_aver_angl = strucFunct2_ai.process_SF_samples(data_avers, rbins, mid_rbins)

    # Save
    outfile = 'test_full_grid_15days.nc'
    dudlt_aver_angl.to_netcdf(outfile)
    print(f'Saved: {outfile}')

def run_one_region():
    pass

if __name__ == '__main__':
    test_full()