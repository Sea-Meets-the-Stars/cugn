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

from IPython import embed

# Calculates structure functions
shiftdim = 'x','y'
grid = 'm'

def load_last_time(ndays=6*30):
    # Load
    qg, _ = qg_utils.load_qg()

    # Gets last 6 months of data
    #nmonths = 6
    #month = 30
    #yr = 365

    i1 = -2-ndays
    all_time = np.arange(i1, i1 + ndays)

    # Chunks data
    chx = len(qg.x)
    chy = len(qg.y)
    cht = len(qg.time)
    chunks = {'x': chx, 'y': chy, 'time': cht}

    # Selects the first level (surface)
    # Last 6 months
    Udsn = qg.isel(lev=0, time=all_time).chunk(chunks)

    # Return
    return qg, Udsn

def test_full(ndays=15, maxcorr=60):

    qg, Udsn = load_last_time()

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

    # Defines distance bins 
    dr = 5000 # meters
    rbins = np.arange(0, 3.e5, dr) # 300 km (same as Miguel)
    mid_rbins = 0.5*(rbins[:-1] + rbins[1:])

    # Average over orientation
    dudlt_aver_angl = strucFunct2_ai.process_SF_samples(data_avers, rbins, mid_rbins)

    # Save
    outfile = 'test_full_grid_15days.nc'
    dudlt_aver_angl.to_netcdf(outfile)
    print(f'Saved: {outfile}')

def run_one_region(xlim:tuple, ylim:tuple, outfile:str,
                   timelast=180,
                   ndays:int=60, maxcorr:int=30):

    # Load
    qg, Udsn = load_last_time(ndays=timelast)

    iregion_x = np.where((qg.x >= xlim[0]*1e3) & (qg.x < xlim[1]*1e3))[0]
    iregion_y = np.where((qg.y >= ylim[0]*1e3) & (qg.y < ylim[1]*1e3))[0]

    # Cut down Usdn
    Udsn = Udsn.isel(x=iregion_x, y=iregion_y, time=np.arange(0, ndays))

    # Grab the last 15 days
    SFtest = strucFunct2_ai.calculateSF_2(Udsn, maxcorr, shiftdim, grid)

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

    # Defines distance bins
    dr = 5000 # meters
    rbins = np.arange(0, 1.3e5, dr) # 130 km
    mid_rbins = 0.5*(rbins[:-1] + rbins[1:])

    # Average over orientation
    dudlt_aver_angl = strucFunct2_ai.process_SF_samples(data_avers, rbins, mid_rbins)

    # Save
    dudlt_aver_angl.to_netcdf(outfile)
    print(f'Saved: {outfile}')

if __name__ == '__main__':

    # Full
    #test_full()

    # Regions for 60 days
    if False:
        for x0 in [300., 400, 500.]:
            for y0 in [300., 400, 500.]:
                run_one_region((x0, x0+100.), (y0, y0+100.), 
                            f'Output/SF_region_x{int(x0)}_y{int(y0)}_60days.nc', 
                            ndays=60, maxcorr=30)

    # Regions for 5 years
    if True:
        for x0 in [300., 400, 500.]:
            for y0 in [300., 400, 500.]:
                run_one_region((x0, x0+100.), (y0, y0+100.), 
                            f'Output/SF_region_x{int(x0)}_y{int(y0)}_5years.nc', 
                            timelast=int(365*5.1),
                            ndays=365*5, maxcorr=30)