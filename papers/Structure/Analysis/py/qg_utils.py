""" Analysis utilities for QG outputs """


import os

import xarray

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import norm

# import fsspec
import matplotlib
import matplotlib.pyplot as plt
import gsw_xarray as gsw
from xhistogram.xarray import histogram

from dask.diagnostics import ProgressBar

from strucFunct2_ai import timescale

from IPython import embed

def load_qg():
    qg_file = os.path.join(os.getenv('OS_DATA'), 'QG', 'QGModelOutput20years.nc')
    qg = xarray.open_dataset(qg_file)

    #### Loads 15 years time series
    chunksSF15 = {'time': 100, 'mid_rbins': 53}

    mSF_15 = xarray.open_dataset(os.path.join(os.getenv('OS_DATA'), 'QG', 'SFQG_aver_pos_orien_5yearb.nc'), 
                                chunks=chunksSF15)
    mSF_15['time'] = mSF_15.time/86400
    mSF_15['du1'] = mSF_15.ulls + mSF_15.utts

    return qg, mSF_15



def load_last_time(ndays=6*30):
    # Load
    qg, _ = load_qg()

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


def calc_dus(qg, mSF_15, indx:int=1, indf:int=40, 
             subsets:bool=False):

    out_dict = {}

    # First order structure function
    #sf1_mn = mSF_15.du1.mean(dim='time')[indx:indf]
    #sf1_std = mSF_15.du1.std(dim='time')[indx:indf]
    rr1 = mSF_15.dr.mean(dim='time')[indx:indf].values
    # Orig
    du1 = mSF_15.du1.isel(mid_rbins=np.arange(indx, indf)).chunk({'mid_rbins':len(mSF_15.mid_rbins), 'time': 100})
    #du2 = mSF_15.du2.isel(mid_rbins=np.arange(indx, indf)).chunk({'mid_rbins':len(mSF_15.mid_rbins), 'time': 100})

    # LL only
    du1LL = mSF_15.ulls.isel(mid_rbins=np.arange(indx, indf)).chunk({'mid_rbins':len(mSF_15.mid_rbins), 'time': 100})


    in1 = 2
    in2 = 5
    in3 = 10
    in4 = 18

    du1r = 0.4

    dull_mn = mSF_15.ulls.mean(dim='time')[indx:indf]
    dutt_mn = mSF_15.utts.mean(dim='time')[indx:indf]
    dull_std = mSF_15.ulls.std(dim='time')[indx:indf]
    dutt_std = mSF_15.utts.std(dim='time')[indx:indf]

    # du2
    du1_mn = mSF_15.du1.mean(dim='time')[indx:indf]

    # du2
    du2_mn = mSF_15.du2.mean(dim='time')[indx:indf]

    # du3
    du3_mn = mSF_15.du3.mean(dim='time')[indx:indf]

    # Corrected
    du3_corr = du3_mn - 3*dull_mn*du2_mn**2 + 2*dull_mn**3

    out_dict['rr1'] = rr1
    out_dict['du1'] = du1
    out_dict['du1LL'] = du1LL
    out_dict['dull_mn'] = dull_mn
    out_dict['dull_std'] = dull_std
    out_dict['dutt_mn'] = dutt_mn
    out_dict['dutt_std'] = dutt_std
    out_dict['du1_mn'] = du1_mn
    out_dict['du2_mn'] = du2_mn
    out_dict['du3_mn'] = du3_mn

    # Bins
    #d1_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 6e-5)/sf1_std[in1].values
    #d2_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 2e-4)/sf1_std[in2].values
    #d3_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 6e-4)/sf1_std[in3].values
    #d4_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 1e-3)/sf1_std[in4].values

    # Every 25 and 50 days
    if subsets:
        du1_25 = []
        du1_50 = []
        dull_25 = []
        dull_50 = []
        ss = 0
        do_50 = True

        while ss < du1.time.size:
            # Grab em
            tmax25 = min(du1.time.size, ss+25)
            du1s25 = np.mean(du1.values[ss:tmax25,:], axis=0)
            du1_25.append(du1s25)
            #
            dulls25 = np.mean(du1LL.values[ss:tmax25,:], axis=0)
            dull_25.append(dulls25)
            if do_50:
                tmax50 = min(du1.time.size, ss+50)
                du1s50 = np.mean(du1.values[ss:tmax50,:], axis=0)
                du1_50.append(du1s50)
                dulls50 = np.mean(du1LL.values[ss:tmax50,:], axis=0)
                dull_50.append(dulls50)
            ss += 25
            if do_50 is False:
                do_50 = True
            else:
                do_50 = False
        # Array me
        du1_25 = np.array(du1_25)
        du1_50 = np.array(du1_50)
        dull_25 = np.array(dull_25)
        dull_50 = np.array(dull_50)
        # Add to out_dict
        out_dict['du1_25'] = du1_25
        out_dict['du1_50'] = du1_50
        out_dict['dull_25'] = dull_25
        out_dict['dull_50'] = dull_50


    # Return it all
    return out_dict

def calc_dus_limtime(mSF_15, ndays:int, t0:int=0, indx:int=1, indf:int=40):

    #embed(header='Calculating structure functions with limited time')
    rr1 = np.mean(mSF_15.dr.values[t0:t0+ndays,indx:indf], axis=0)

    # Grab em
    du1 = mSF_15.du1.isel(mid_rbins=np.arange(indx, indf)).chunk({'mid_rbins':len(mSF_15.mid_rbins), 'time': 100})
    du1LL = mSF_15.ulls.isel(mid_rbins=np.arange(indx, indf)).chunk({'mid_rbins':len(mSF_15.mid_rbins), 'time': 100})
    du2 = mSF_15.du2.isel(mid_rbins=np.arange(indx, indf)).chunk({'mid_rbins':len(mSF_15.mid_rbins), 'time': 100})
    du3 = mSF_15.du3.isel(mid_rbins=np.arange(indx, indf)).chunk({'mid_rbins':len(mSF_15.mid_rbins), 'time': 100})

    # Every 25 and 50 days
    du1s = np.mean(du1.values[t0:t0+ndays,:], axis=0)
    du1LLs = np.mean(du1LL.values[t0:t0+ndays,:], axis=0)
    du2s = np.mean(du2.values[t0:t0+ndays,:], axis=0)
    du3s = np.mean(du3.values[t0:t0+ndays,:], axis=0)

    #du3_corr = du3s - 3*du1LLs*du2s**2 + 2*du1LLs**3
    du3_corr = du3s - 3*du1s*du2s**2 + 2*du1s**3

    # Return
    return rr1, du1s, du1LLs, du2s, du3s, du3_corr

# Command line
if __name__ == "__main__":

    # Load data
    qg, mSF_15 = load_qg()

    calc_dus_limtime(mSF_15, 60, t0=100)
