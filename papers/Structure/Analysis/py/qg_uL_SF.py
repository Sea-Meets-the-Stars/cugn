""" Calculate QG SF on the full grid """

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

def grab_path(use_duL:bool):
    """Return the directory path for spatially averaged SF files.

    Args:
        use_duL: If True, return path for longitudinal-only (duL) SF averages;
            otherwise return the standard SF spatial average path.

    Returns:
        str: Absolute directory path.
    """
    if use_duL:
        return os.path.join(qg_path, 'SF_spatialav_duL')
    else:
        return os.path.join(qg_path, 'SF_spatialav')

def calc_rawduLT(nyears=5, maxcorr=90, clobber:bool=False):
    """Compute raw duL/duT structure functions in 15-day chunks and save to disk.

    Processes the last nyears of QG model data, computing velocity structure
    functions for each 15-day chunk and writing individual NetCDF files.

    Args:
        nyears: Number of years of data to process from the end of the time series.
        maxcorr: Maximum number of grid shifts for pair separations.
            90 gives good stats to ~300km
        clobber: If True, overwrite existing output files.
    """
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


def calc_SF(dcorr=3599, chkx=256, chky=256, clobber:bool=False, use_dLT:bool=True):
    """Compute spatially averaged 2nd and 3rd order structure functions.

    Reads raw duLT files, computes du2 and du3, averages over spatial
    dimensions (x, y), and saves per-chunk NetCDF files.

    Args:
        dcorr: Number of correlation shifts for chunking.
        chkx: Chunk size in x dimension.
        chky: Chunk size in y dimension.
        clobber: If True, overwrite existing output files.
        use_dLT: If True, compute SF using longitudinal component only (duL);
            otherwise use both longitudinal and transverse components.
    """
    # duL?
    SFavg_path = grab_path(use_dLT)
    
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
        if os.path.exists(fileSp) and not clobber:
            print(f'File {fileSp} already exists, skipping...')
            ii += 1
            continue
        # Ensure the end time does not exceed the total length of the 'time' dimension
        end_time = min(start_time + tchunk_size, len(dult['time']))
        
        # Slice the data to include the current chunk
        data_slice = dult.isel(time=slice(start_time, end_time))
        
        # Calculates du1, du2 and du3
        if use_dLT:
            sf2, sf3 = strucFunct2_ai.SF2_3_ul(data_slice.ulls)
        else:
            sf2, sf3 = strucFunct2_ai.SF2_3(data_slice.ulls, data_slice.utts)
        data_slice['du2'] = sf2
        data_slice['du3'] = sf3
        
        # Averages over all $s$ positions
        with ProgressBar():
            data_avers = data_slice.mean(dim=('x','y'), skipna=True).compute()
        
        print('Save SF_spatialaver_{}.nc file'.format(ii))
        data_avers.to_netcdf(fileSp)
        
        ii = ii + 1

def calc_SF_5years(use_dLT:bool=True):
    """Combine spatially averaged SF files into a single 5-year dataset.

    Loads all per-chunk spatial average files, bins by separation distance,
    averages over orientation, and saves as a single NetCDF file.

    Args:
        use_dLT: If True, use longitudinal-only SF files and save with '_duL' suffix;
            otherwise use standard SF files and save with '_new' suffix.
    """
    SFavg_path = grab_path(use_dLT)
    if use_dLT:
        outfile = os.path.join(qg_path, 'SFQG_aver_pos_orien_5yearb_duL.nc')
    else:
        outfile = os.path.join(qg_path, 'SFQG_aver_pos_orien_5yearb_new.nc')

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
    outfile = os.path.join(qg_path, outfile)
    dudlt_aver_angl.to_netcdf(outfile)
    print(f'Saved: {outfile}')


def parse_SF(SF_file:str, Ndays:int):
    """Load and parse a regional SF file, applying third-order cumulant correction.

    Args:
        SF_file: Path to a regional structure function NetCDF file.
        Ndays: Number of days from the end to use for time-averaging.

    Returns:
        tuple: (rr1, rrr1, du1, du2, du3, du3_corr, dull_mn, du2_mn_duL, du3_mn_duL)
            where rr1 is from the 5-year duL reference, rrr1 is from the regional file
            (in km), du1/du2/du3 are time-mean SFs from the regional file, du3_corr is
            the corrected third-order SF, and the _duL quantities are from the 5-year
            duL reference dataset.
    """
    # Load
    SFds = xarray.load_dataset(SF_file)
    qg, mSF_15_duL = qg_utils.load_qg(use_SFduL=True)
    SF_dict_duL = qg_utils.calc_dus(qg, mSF_15_duL)
    du2_mn_duL = SF_dict_duL['du2_mn']
    du3_mn_duL = SF_dict_duL['du3_mn']
    dull_mn = SF_dict_duL['dull_mn']
    rr1 = SF_dict_duL['rr1']

    # Cut on time
    i1 = -1*Ndays
    times = np.arange(i1, i1 + Ndays)
    SFds = SFds.isel(time=times)

    # Correct the du3
    du1 = SFds.ulls.T.mean('time')
    du2 = SFds.du2.T.mean('time')
    du3 = SFds.du3.T.mean('time')
    du3_corr = du3 - 3*du1*du2 + 2*du1**3
    
    rrr1 = SFds.dr.mean('time')*1e-3 

    # Return
    return rr1, rrr1, du1, du2, du3, du3_corr, dull_mn, du2_mn_duL, du3_mn_duL


if __name__ == '__main__':

    # raw dULT
    calc_rawduLT(clobber=True)

    # SF per time step
    calc_SF(clobber=True)
    #calc_SF(use_dLT=False, clobber=True)

    # Lastly
    calc_SF_5years(use_dLT=True)
    #calc_SF_5years(use_dLT=False)
