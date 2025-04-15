""" Codes by Manuel

https://github.com/manuelogtzv/SF3_RLS/blob/master/calcSF_QGmodel.ipynb

"""
import os
import numpy as np
import xarray

from scipy.stats import norm
from xhistogram.xarray import histogram
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from strucFunct2_ai import SF2_3
from strucFunct2_ai import process_SF_samples
from strucFunct2_ai import timescale
from strucFunct2_ai import calculateSF_2
#from strucFunct2_ai import mSF_15

QG_path = '/home/xavier/Projects/Oceanography/data/QG'

def calc_structure(eddyrun_lev, clobber:bool=False):
    # Calculates structure functions
    shiftdim = 'x','y'
    maxcorr = 60
    grid = 'm'

    # rbins = np.arange(1e3, 2e5, 4e3)

    # Gets last five years of data
    nyears = 5
    yr = 365

    i1 = -2-yr*nyears
    time3y = np.arange(i1, i1 + yr*nyears)

    # Chunks data

    chx = len(eddyrun_lev.x)
    chy = len(eddyrun_lev.y)
    cht = len(eddyrun_lev.time)

    chunks = {'x': chx, 'y': chy, 'time': cht}

    Udsn = eddyrun_lev.isel(lev=0, time=time3y).chunk(chunks)

    # Runs code for all datasets
    time_indices = np.arange(len(Udsn.time))  # All time indices

    # Define the chunk size
    chunk_size = 15

    # Loop over the time indices in chunks of 15
    for start in tqdm(range(0, len(time_indices), chunk_size), desc="Processing Chunks: "):
        filessv = os.path.join(QG_path, 'rawduLT', str(start)+'.nc')
        # Clobber?
        if os.path.exists(filessv) and not clobber:
            print('File {} exists'.format(filessv))
            continue
        # End
        end = start + chunk_size
        
        # Ensure that the 'end' index doesn't exceed the total number of time indices
        if end > len(time_indices):
            end = len(time_indices)
        
        # Slice the time indices for the current chunk
        indx_time = time_indices[start:end]
        data = Udsn.isel(time=indx_time).chunk({'x': chx, 'y': chy, 'time': chunk_size})
        
        # Runs code
        SFQG = calculateSF_2(data, maxcorr, shiftdim, grid)
        print('Save {}.nc file'.format(start))
        SFQG.to_netcdf(filessv)

def gen_spatavg():

    fileraw = '/data/SO3/manuelogv/MethodsKEFlux/rawduLT/'

    # Open the NetCDF files using xarray's open_mfdataset (multi-file dataset)
    nc_files = fileraw +'*.nc'  #
    dult = xarray.open_mfdataset(nc_files, engine='netcdf4', combine='by_coords', 
                            chunks={'time': 100, 'x': 256, 'y': 256, 'dcorr': 10}, 
                            parallel=True)

    dult = dult.sortby('time')

    tchunk_size = 15 # time slice length
    Ntot = len(dult.time)
    chunk_slic = {'time': tchunk_size, 'x': len(dult.x), 'y': len(dult.y), 'dcorr': 3599}
    dult = dult.chunk(chunk_slic)

    output_list = []
    ii = 0

    for start_time in tqdm(range(0, Ntot, tchunk_size), desc="Time slice", position=2):
        # Ensure the end time does not exceed the total length of the 'time' dimension
        end_time = min(start_time + tchunk_size, len(dult['time']))
        
        # Slice the data to include the current chunk
        data_slice = dult.isel(time=slice(start_time, end_time))
        
        # Calculates du1, du2 and du3
        sf2, sf3 = SF2_3(data_slice.dul, data_slice.dut)
        data_slice['du2'] = sf2
        data_slice['du3'] = sf3
        
        # Averages over all $s$ positions
        with ProgressBar():
            data_avers = data_slice.mean(dim=('x','y'), skipna=True).compute()
        
        fileSp = '/data/SO3/manuelogv/MethodsKEFlux/spatialaverduLT/SF_spatialaver_' + str(ii) + '.nc'
        print('Save SF_spatialaver_{}.nc file'.format(ii))
        data_avers.to_netcdf(fileSp)
        
        ii = ii + 1

def gen_mSF():

    fileaver = 'spatialaverduLT/'

    # Open the NetCDF files using xarray's open_mfdataset (multi-file dataset)
    nc_files3 = fileaver +'*.nc'  #
    dult_aver = xarray.open_mfdataset(nc_files3, engine='netcdf4', combine='by_coords')
    dult_aver = dult_aver.sortby('time').chunk({'time': 1825, 'dcorr': 2}).load()

    # Defines distance bins 
    dr = 5000 # meters
    rbins = np.arange(0, 3e5, dr)
    mid_rbins = 0.5*(rbins[:-1] + rbins[1:])

    # Process
    dudlt_aver_angl = process_SF_samples(dult_aver, rbins, mid_rbins)
    dudlt_aver_angl.to_netcdf('SFQG_aver_pos_orien_5yearb.nc')

def load_mSF():
    chunksSF15 = {'time': 100, 'mid_rbins': 53}
    mSF_15 = xarray.open_dataset('SFQG_aver_pos_orien_5yearb.nc', 
                                 chunks=chunksSF15)
    mSF_15['time'] = mSF_15.time/86400
    mSF_15['du1'] = mSF_15.ulls + mSF_15.utts

    return mSF_15

def first_order(mSF_15):

    # Calculates degrees of freedom
    nyears = 5
    yr2day = 365

    # Gets the first 40 index in r
    indx = 1
    indf = 40

    Tmax = yr2day*nyears*86400

    qg_tscale3 = timescale(mSF_15.du2.mean(dim='time').values[indx:indf], 
                        mSF_15.dr.mean(dim='time').values[indx:indf])

    qg_dof = Tmax/qg_tscale3

    nu3 = np.sqrt(qg_dof)

    # First order structure function
    sf1_mn = mSF_15.du1.mean(dim='time')[indx:indf]
    sf1_std = mSF_15.du1.std(dim='time')[indx:indf]
    rr1 = mSF_15.dr.mean(dim='time')[indx:indf].values
    du1 = mSF_15.du1.isel(mid_rbins=np.arange(indx, indf)).chunk({'mid_rbins':len(mSF_15.mid_rbins), 'time': 100})


    in1 = 2
    in2 = 5
    in3 = 10
    in4 = 18

    du1r = 0.4

    dull_mn = mSF_15.ulls.mean(dim='time')[indx:indf]
    dutt_mn = mSF_15.utts.mean(dim='time')[indx:indf]
    dull_std = mSF_15.ulls.std(dim='time')[indx:indf]
    dutt_std = mSF_15.utts.std(dim='time')[indx:indf]


    # Bins
    d1_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 6e-5)/sf1_std[in1].values
    d2_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 2e-4)/sf1_std[in2].values
    d3_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 6e-4)/sf1_std[in3].values
    d4_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 1e-3)/sf1_std[in4].values


    # Histograms
    sf1h0 =  histogram(du1.isel(mid_rbins=in1)/sf1_std[in1].values, bins=d1_bins, dim=['time'], density=True)
    sf1h2 = histogram(du1.isel(mid_rbins=in2)/sf1_std[in2].values, bins=d2_bins, dim=['time'], density=True)
    sf1h10 = histogram(du1.isel(mid_rbins=in3)/sf1_std[in3].values, bins=d3_bins, dim=['time'], density=True)
    sf1h15 = histogram(du1.isel(mid_rbins=in4)/sf1_std[in4].values, bins=d4_bins, dim=['time'], density=True)

    # Constructs Gaussian
    sf1p0 = norm.pdf(d1_bins, sf1_mn[in1]/sf1_std[in1].values, 1)
    sf1p2 = norm.pdf(d2_bins, sf1_mn[in2]/sf1_std[in2].values, 1)
    sf1p10 = norm.pdf(d3_bins, sf1_mn[in3]/sf1_std[in3].values, 1)
    sf1p15 = norm.pdf(d4_bins, sf1_mn[in4]/sf1_std[in4].values, 1)


    # Calculates kurtosis
    from scipy.stats import skew, kurtosis
    sf1_skew = np.zeros((len(rr1),))
    sf1_kurt = sf1_skew*0.

    for ii in range(len(rr1)):
        
        sf1_skew[ii] = skew(du1.isel(mid_rbins=ii).values, axis=0, bias=True)
        sf1_kurt[ii] = kurtosis(du1.isel(mid_rbins=ii).values, axis=0, fisher=True, bias=True)

    # Return
    return rr1, du1, sf1_mn, dull_mn, dutt_mn, sf1_std, dull_std, dutt_std

def main(flg:int):

    # Generate raw
    if flg == 0:
        fileQG = '/home/xavier/Projects/Oceanography/data/QG/QGModelOutput20years.nc'
        eddyrun_lev = xarray.open_dataset(fileQG)
        calc_structure(eddyrun_lev)

    if flg == 5:
        #gen_spatavg()
        #gen_mSF()
        mSF_15 = load_mSF()
        rr1, du1, sf1_mn, dull_mn, dutt_mn, sf1_std, dull_std, dutt_std = first_order(mSF_15)

# Command line
if __name__ == '__main__':
    import sys

    # Get the command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <flag>")
        sys.exit(1)
    flag = int(sys.argv[1])
    # Call the main function with the provided flag
    main(flag)



    