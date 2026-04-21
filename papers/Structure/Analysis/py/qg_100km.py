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

def test_full(ndays=15, maxcorr=60):
    """Compute and save structure functions over the full QG domain for a short period.

    Args:
        ndays: Number of days to process from the end of the time series.
        maxcorr: Maximum number of grid shifts for pair separations.
    """
    qg, Udsn = qg_utils.load_last_time()

    # Grab the last ndays
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
                   timelast=180, clobber:bool=False,
                   ndays:int=60, maxcorr:int=30,
                   time_batch:int=365,
                   reduce_xy:bool=False,
                   rbins:np.ndarray=None):
    """ 
    Calculate the structure function for a region of the QG model output.

    The SF pipeline is run on consecutive time slabs of length `time_batch`
    (default one year) and the per-batch, orientation-averaged results are
    concatenated along `time`. This keeps peak memory bounded by a single
    batch rather than the full `ndays` window.

    Args:
        xlim: tuple of (xmin, xmax) in km
        ylim: tuple of (ymin, ymax) in km
        outfile: path to save the structure function
        timelast: last time index to load
        ndays: number of days to calculate the structure function
        maxcorr: maximum correlation distance
        time_batch: number of time steps processed per batch
        reduce_xy: If True, take the (x, y) spatial mean inside each
            per-shift task in calculateSF_2 rather than after concat.
            Shrinks the dask graph from ~maxcorr^2 full-grid nodes to
            ~maxcorr^2 (time,) nodes. Big win on large regions; results
            are mathematically equivalent.
        rbins: array of distance bins in meters
    """

    # Clobber?
    if os.path.exists(outfile) and not clobber:
        print(f'File {outfile} exists. Skipping')
        return

    # Load
    qg, Udsn = qg_utils.load_last_time(ndays=timelast, use_SFduL=True)

    iregion_x = np.where((qg.x >= xlim[0]*1e3) & (qg.x < xlim[1]*1e3))[0]
    iregion_y = np.where((qg.y >= ylim[0]*1e3) & (qg.y < ylim[1]*1e3))[0]

    # Cut down Usdn (spatial only; time is batched below)
    print(f'Cutting down Usdn to region {xlim} {ylim}')
    print(f'Calculating structure function for {ndays} days '
          f'in batches of {time_batch}')
    Udsn = Udsn.isel(x=iregion_x, y=iregion_y, time=np.arange(0, ndays))

    # Defines distance bins (shared across batches)
    if rbins is None:
        dr = 5000 # meters
        rbins = np.arange(0, 1.3e5, dr) # 130 km

    # Calculate structure function
    SFtest = strucFunct2_ai.calculateSF_2(Udsn, maxcorr, shiftdim, grid)

    #embed(header='103 of run_one_region')

    # Higher order;  Use duL only
    #SF2, SF3 = strucFunct2_ai.SF2_3_ul(SFtest.ulls)

    # Slice the data to include the current chunk
    data_slice = SFtest.isel(time=slice(0,ndays))
        
    # Calculates du1, du2 and du3
    print(f'Calculating du2 and du3 for {outfile}')
    sf2, sf3 = strucFunct2_ai.SF2_3_ul(data_slice.ulls)#, data_slice.dut)
    data_slice['du2'] = sf2
    data_slice['du3'] = sf3
        
    # Averages over all $s$ positions
    print(f'Averaging over all $s$ positions for {outfile}')
    with ProgressBar():
        data_avers = data_slice.mean(dim=('x','y'), skipna=True).compute()

    mid_rbins = 0.5*(rbins[:-1] + rbins[1:])

    # Loop over time batches
    n_batches = int(np.ceil(ndays / time_batch))
    batch_results = []
    for ibatch in range(n_batches):
        t0 = ibatch * time_batch
        t1 = min(t0 + time_batch, ndays)
        print(f'--- Batch {ibatch+1}/{n_batches}: time [{t0}, {t1}) ---')

        Udsn_batch = Udsn.isel(time=np.arange(t0, t1))

        # Calculate structure function for this batch
        SFtest = strucFunct2_ai.calculateSF_2(
            Udsn_batch, maxcorr, shiftdim, grid, reduce_xy=reduce_xy)

        data_slice = SFtest.isel(time=slice(0, t1 - t0))

        if reduce_xy:
            # du2, du3 and the spatial mean are already done inside
            # calculateSF_2; just materialize.
            print(f'Materializing reduced SF (batch {ibatch+1})')
            with ProgressBar():
                data_avers = data_slice.compute()
        else:
            # Calculates du2 and du3
            print(f'Calculating du2 and du3 (batch {ibatch+1})')
            sf2, sf3 = strucFunct2_ai.SF2_3_ul(data_slice.ulls)
            data_slice['du2'] = sf2
            data_slice['du3'] = sf3

            # Averages over all $s$ positions
            print(f'Averaging over all $s$ positions (batch {ibatch+1})')
            with ProgressBar():
                data_avers = data_slice.mean(
                    dim=('x', 'y'), skipna=True).compute()

        # Average over orientation
        print(f'Averaging over orientation (batch {ibatch+1})')
        batch_out = strucFunct2_ai.process_SF_samples(
            data_avers, rbins, mid_rbins)
        batch_results.append(batch_out)

    # Concatenate batches along time
    dudlt_aver_angl = xarray.concat(batch_results, dim='time')
    dudlt_aver_angl = dudlt_aver_angl.sortby('time')

    # Save
    dudlt_aver_angl.to_netcdf(outfile)
    print(f'Saved: {outfile}')

if __name__ == '__main__':

    # Full
    #test_full()

    # Regions for 100 days
    if True:
        for x0 in [300., 400, 500.]:
            for y0 in [300., 400, 500.]:
                run_one_region((x0, x0+100.), (y0, y0+100.), 
                            f'Output/SF_region_x{int(x0)}_y{int(y0)}_60days.nc', 
                            ndays=100, maxcorr=30)

    # 100km regions for 5 years
    if True:
        for x0 in [300., 400, 500.]:
            for y0 in [300., 400, 500.]:
                run_one_region((x0, x0+100.), (y0, y0+100.), 
                            f'Output/SF_region_x{int(x0)}_y{int(y0)}_5years.nc',
                            timelast=int(365*5.1),
                            ndays=365*5, maxcorr=30)

    # 200km regions for 5 years
    if True:
        dr = 5000 # meters
        rbins = np.arange(0, 1.6e5, dr) # 160 km
        for x0 in [200., 400, 600.]:
            for y0 in [200., 400, 600.]:
                run_one_region((x0, x0+200.), (y0, y0+200.),
                            f'Output/SF_region_x{int(x0)}_y{int(y0)}_200km_5years.nc',
                            rbins=rbins,
                            timelast=int(365*5.1),
                            ndays=365*5, maxcorr=50, reduce_xy=True)

    # 300km regions for 5 years
    if False:
        for x0 in [200., 500]:
            for y0 in [200., 500]:
                run_one_region((x0, x0+300.), (y0, y0+300.),
                            f'Output/SF_region_x{int(x0)}_y{int(y0)}_300km_5years.nc',
                            timelast=int(365*5.1),
                            ndays=365*5, maxcorr=90)

    # Drifter region for 100 days
    if False:
        x0, y0 = 450., 450.
        run_one_region((x0, x0+100.), (y0, y0+100.), 
            f'Output/SF_region_x{int(x0)}_y{int(y0)}_100days.nc',
            timelast=int(2199)-2, # Starts at 5001, like the drifter analysis
            ndays=100, maxcorr=30, clobber=True)

    # Testing
    if False:
        x0, y0 = 450., 450.
        run_one_region((x0, x0+12.), (y0, y0+12.), 
            f'Output/test_me.nc',
            timelast=int(2199)-2, # Starts at 5001, like the drifter analysis
            ndays=100, maxcorr=4, clobber=True)
