""" Methods for high-resolution data products """

import numpy as np
from scipy.interpolate import interp1d

import xarray

from gsw import conversions, density
import gsw

from cugn import defs as cugn_defs

from IPython import embed

def calc_mld_N(depths, salinities, temperatures, oxygens,
               lat, lon, max_depth, return_extras:bool=False,
               Npeak_min:float=10., debug=False):

    extras = {}
    # z bins
    z_edges = np.arange(5, max_depth+15, 10)
    nbins = len(z_edges) - 1
    bin_means = np.zeros(nbins)

    # Pressure
    p = conversions.p_from_z(-1*depths, lat)

    # SA
    SA = conversions.SA_from_SP(salinities, p, lon, lat)

    # CT
    CT = conversions.CT_from_t(SA, temperatures, p)

    # Oxygen
    OC = gsw.O2sol(SA, CT, p, lon, lat)
    SO = oxygens / OC

    # sigma0 
    sigma0 = density.sigma0(SA, CT)

    # Sort
    srt_z = np.argsort(depths)
    sigma0 = sigma0[srt_z]
    z_sort = depths[srt_z]
    SO = SO[srt_z]

    # sigma0 at surface
    sigma0_0 = np.mean(sigma0[:3])

    # Calculate MLD
    f = interp1d(sigma0, z_sort, fill_value='extrapolate')
    MLD = f(sigma0_0 + cugn_defs.MLD_sigma0)


    # Buoyancy
    dsigmadz = np.gradient(sigma0, z_sort)
    dsigmadz[dsigmadz < 0.] = 0.
    buoyfreq = np.sqrt(9.8/1025*dsigmadz)/(2*np.pi)*3600

    # Smoothed
    sbuoyfreq = np.convolve(buoyfreq, np.ones(3)/3, mode='same')

    # Now grid the N values according to max depth
    bin_indices = np.digitize(z_sort, z_edges) - 1

    # Calculate means of N for each bin
    bin_means[:] = np.nan
    for i in range(nbins):
        mask = bin_indices == i
        if np.any(mask):
            bin_means[i] = np.nanmean(buoyfreq[mask])

    # N peak
    dN1 = buoyfreq - np.roll(buoyfreq, -1)
    dN2 = buoyfreq - np.roll(buoyfreq, -2)
    Npeaks = np.where((dN1 > 0) & (dN2 > 0) & (buoyfreq > Npeak_min))[0]
    if Npeaks.size > 0:
        z_Npeak = float(z_sort[Npeaks[0]])
    else:
        z_Npeak = np.nan

    # N depths
    N5 = sbuoyfreq > 5.
    if np.any(N5):
        zN5 = z_sort[np.where(N5)[0][0]]
    else:
        zN5 = np.nan
    N10 = sbuoyfreq > 10.
    if np.any(N10):
        zN10 = z_sort[np.where(N10)[0][0]]
    else:
        zN10 = np.nan

    # Fraction of SO > 1.1
    SO1 = SO > 1.1
    f5 = SO1 & (z_sort >= zN5)
    f10 = SO1 & (z_sort >= zN10)
    Nf5 = np.sum(f5)
    Nf10 = np.sum(f10)
    NSO = np.sum(SO1)

    # delta rho
    if return_extras:
        extras['delta_rho'] = sigma0-sigma0_0
        extras['S0'] = SO

    if debug and NSO > 0:
        print(f"z_Npeak={z_Npeak}, zN5={zN5}, zN10={zN10}")
        print(f"Nf5={Nf5}, Nf10={Nf10}, NSO={NSO}")

    # Extras?
    if return_extras:
        extras['N'] = buoyfreq
        extras['sigma0'] = sigma0
        extras['z_sort'] = z_sort
        return MLD, bin_means, z_Npeak, zN5, zN10, Nf5, Nf10, NSO, extras

    return MLD, bin_means, z_Npeak, zN5, zN10, Nf5, Nf10, NSO

def calc_mission(highres_file:str, mission_profiles:list, 
               min_depth:float=2.0,
               max_depth:float=100., debug:bool=False):
    """ Calculate MLD and N for a mission

    Args:
        highres_file (str): high resolution file
        mission_profiles (list): list of profiles
        min_depth (float, optional): Minimum depth. Defaults to 2.0.
        max_depth (float, optional): Maximum depth. Defaults to 100..

    Returns:
        np.array: MLDs
        np.array: Ns
        np.array: zNs
    """

    # High res
    ds_high = xarray.open_dataset(highres_file)

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


    MLDs = []
    Ns = []
    zNs = []
    zN5s = []
    zN10s = []
    Nf5s = []
    Nf10s = []
    NSOs = []

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
            Nf5, Nf10, NSO = calc_mld_N(
            ds_high.depth.values[my_obs],
            salinity[my_obs],
            temperature[my_obs],
            oxygen[my_obs],
            lat, lon, max_depth, debug=debug)

        # MLD, N
        MLDs.append(MLD)
        Ns.append(bin_means.copy())
        zNs.append(z_Npeak)
        zN5s.append(zN5)
        zN10s.append(zN10)
        Nf5s.append(Nf5)
        Nf10s.append(Nf10)
        NSOs.append(NSO)
        #embed(header='cugn/highres.py: 88')

    # Return
    return np.array(MLDs), np.array(Ns), np.array(zNs),\
        np.array(zN5s), np.array(zN10s), np.array(Nf5s),\
        np.array(Nf10s), np.array(NSOs)