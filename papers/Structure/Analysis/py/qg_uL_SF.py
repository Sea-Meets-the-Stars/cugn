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
from tqdm.notebook import tqdm

from strucFunct2_ai import timescale

import qg_utils
import strucFunct2_ai

from IPython import embed

# Calculates structure functions
shiftdim = 'x','y'
grid = 'm'

qg_path = os.path.join(os.getenv('OS_DATA'), 'QG')
raw_path = os.path.join(qg_path, 'rawduLT')
SFavg_path = os.path.join(qg_path, 'SF_spatialav')

def calc_rawduLT(nyears=5, maxcorr=60, clobber:bool=False):

    ndays = nyears * 365
    qg, Udsn = qg_utils.load_last_time(ndays=ndays)

    # Runs code for all datasets
    time_indices = np.arange(len(Udsn.time))  # All time indices

    # Define the chunk size
    chunk_size = 15
    chx = len(qg.x)
    chy = len(qg.y)

    # Loop over the time indices in chunks of 15
    for start in tqdm(range(0, len(time_indices), chunk_size), desc="Processing Chunks: "):
        filessv = os.path.join(raw_path, str(start)+'.nc')
        if os.path.exists(filessv) and not clobber:
            print(f'File {filessv} already exists, skipping...')
            continue
        # 
        end = start + chunk_size
        
        # Ensure that the 'end' index doesn't exceed the total number of time indices
        if end > len(time_indices):
            end = len(time_indices)
        
        # Slice the time indices for the current chunk
        indx_time = time_indices[start:end]
        data = Udsn.isel(time=indx_time).chunk({'x': chx, 'y': chy, 'time': chunk_size})
        
        # Runs code
        SFQG = strucFunct2_ai.calculateSF_2(data, maxcorr, shiftdim, grid)
        print('Save {}.nc file'.format(start))
        SFQG.to_netcdf(filessv)


def calc_SF(dcorr=3599, chkx=256, chky=256):
    # Open the NetCDF files using xarray's open_mfdataset (multi-file dataset)
    nc_files = os.path.join(raw_path, '*.nc')  #
    dult = xarray.open_mfdataset(nc_files, engine='netcdf4', combine='by_coords', 
                          chunks={'time': 100, 'x': chkx, 'y': chky, 'dcorr': 10}, 
                          parallel=False) # Was True, but seg faulting
    dult = dult.sortby('time')

    tchunk_size = 15 # time slice length
    Ntot = len(dult.time)
    chunk_slic = {'time': tchunk_size, 'x': len(dult.x), 'y': len(dult.y), 'dcorr': dcorr}
    dult = dult.chunk(chunk_slic)

    ii = 0
    for start_time in tqdm(range(0, Ntot, tchunk_size), desc="Time slice", position=2):
        fileSp = os.path.join(SFavg_path, str(ii)+'.nc')
        if os.path.exists(fileSp):
            print(f'File {fileSp} already exists, skipping...')
            ii += 1
            continue
        # Ensure the end time does not exceed the total length of the 'time' dimension
        end_time = min(start_time + tchunk_size, len(dult['time']))
        
        # Slice the data to include the current chunk
        data_slice = dult.isel(time=slice(start_time, end_time))
        
        # Calculates du1, du2 and du3
        sf2, sf3 = strucFunct2_ai.SF2_3_ul(data_slice.ulls)
        data_slice['du2'] = sf2
        data_slice['du3'] = sf3
        
        # Averages over all $s$ positions
        with ProgressBar():
            data_avers = data_slice.mean(dim=('x','y'), skipna=True).compute()
        
        print('Save SF_spatialaver_{}.nc file'.format(ii))
        data_avers.to_netcdf(fileSp)
        
        ii = ii + 1

def calc_SF_5years():

    # Open the NetCDF files using xarray's open_mfdataset (multi-file dataset)
    nc_files3 = os.path.join(SFavg_path,'*.nc')  #
    dult_aver = xarray.open_mfdataset(nc_files3, engine='netcdf4', combine='by_coords')

    dult_aver = dult_aver.sortby('time').chunk({'time': 1825, 'dcorr': 2}).load()

    # Defines distance bins
    dr = 5000 # meters
    rbins = np.arange(0, 3e5, dr)
    mid_rbins = 0.5*(rbins[:-1] + rbins[1:])

    # Average over orientation
    dudlt_aver_angl = strucFunct2_ai.process_SF_samples(dult_aver, rbins, mid_rbins)
    outfile = os.path.join(qg_path, 'SFQG_aver_pos_orien_5yearb_duL.nc')
    dudlt_aver_angl.to_netcdf(outfile)
    print(f'Saved: {outfile}')



if __name__ == '__main__':

    # raw dULT
    calc_rawduLT()#clobber=True)

    # SF
    #calc_SF()

    # Lastly
    #calc_SF_5years()
