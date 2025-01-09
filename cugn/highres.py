""" Methods for high-resolution data products """

import numpy as np
from scipy.interpolate import interp1d

import xarray

from gsw import conversions, density

from cugn import defs as cugn_defs

from IPython import embed

def calc_mld_N(depths, salinities, temperatures, 
               lat, lon, max_depth, return_extras:bool=False,
               Npeak_min:float=10.):

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

    # sigma0 
    sigma0 = density.sigma0(SA, CT)

    # Sort
    srt_z = np.argsort(depths)
    sigma0 = sigma0[srt_z]
    z_sort = depths[srt_z]

    # sigma0 at surface
    sigma0_0 = np.mean(sigma0[:5])

    # Calculate MLD
    f = interp1d(sigma0, z_sort)
    MLD = f(sigma0_0 + cugn_defs.MLD_sigma0)


    # Buoyancy
    dsigmadz = np.gradient(sigma0, z_sort)
    dsigmadz[dsigmadz < 0.] = 0.
    buoyfreq = np.sqrt(9.8/1025*dsigmadz)/(2*np.pi)*3600


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

    # Extras?
    if return_extras:
        extras['N'] = buoyfreq
        extras['sigma0'] = sigma0
        extras['z_sort'] = z_sort
        return MLD, bin_means, z_Npeak, extras

    return MLD, bin_means, z_Npeak

def calc_mission(highres_file:str, mission_profiles:list, 
               min_depth:float=2.0,
               max_depth:float=100.):
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

    # QC
    good_sal = ds_high.salinity_qc.values.astype(int) == 1
    good_sal |= ds_high.salinity_qc.values.astype(int) == 3
    good_temp = ds_high.temperature_qc.values.astype(int) == 1
    good_temp |= ds_high.temperature_qc.values.astype(int) == 3


    MLDs = []
    Ns = []
    zNs = []

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
        MLD, bin_means, z_Npeak = calc_mld_N(ds_high.depth.values[my_obs],
                                    salinity[my_obs],
                                    temperature[my_obs],
                                    lat, lon, max_depth)

        # MLD, N
        MLDs.append(MLD)
        Ns.append(bin_means.copy())
        zNs.append(z_Npeak)
        #embed(header='cugn/highres.py: 88')

    # Return
    return np.array(MLDs), np.array(Ns), np.array(zNs)