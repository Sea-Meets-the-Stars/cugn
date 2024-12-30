""" Methods for high-resolution data products """

import numpy as np
from scipy.interpolate import interp1d

import xarray

from gsw import conversions, density

from cugn import defs as cugn_defs

from IPython import embed


def calc_mld_N(highres_file:str, mission_profiles:list, min_depth:float=0.5,
               max_depth:float=100.):

    # High res
    ds_high = xarray.open_dataset(highres_file)

    lat = np.nanmedian(ds_high.latitude.data)
    lon = np.nanmedian(ds_high.longitude.data)

    salinity = ds_high.salinity.values
    temperature = ds_high.temperature.values

    # z bins
    z_edges = np.arange(5, max_depth+15, 10)
    nbins = len(z_edges) - 1
    bin_means = np.zeros(nbins)

    MLDs = []
    Ns = []

    for mission_profile in mission_profiles:
        #print(f'Working on {mission_name} {mission_profile}')
    

        # Find the obs
        my_obs = ds_high.profile_obs_index.values == mission_profile
        my_obs &= ds_high.depth.values > min_depth

        # Calculate density, etc.

        # Require finite
        my_obs &= np.isfinite(salinity) & np.isfinite(temperature)

        # Pressure
        p = conversions.p_from_z(-1*ds_high.depth.values[my_obs], lat)

        # SA
        SA = conversions.SA_from_SP(salinity[my_obs], p, lon, lat)

        # CT
        CT = conversions.CT_from_t(SA, temperature[my_obs], p)

        # sigma0 
        sigma0 = density.sigma0(SA, CT)

        # Sort
        srt_z = np.argsort(ds_high.depth[my_obs])
        sigma0 = sigma0[srt_z]
        z_sort = ds_high.depth[my_obs].values[srt_z]

        # sigma0 at surface
        sigma0_0 = np.mean(sigma0[:5])

        # Calculate MLD
        f = interp1d(sigma0, z_sort)
        MLD = f(sigma0_0 + cugn_defs.MLD_sigma0)
        MLDs.append(MLD)

        # Buoyancy
        dsigmadz = np.gradient(sigma0, z_sort)
        dsigmadz[dsigmadz < 0.] = 0.
        buoyfreq = np.sqrt(9.8/1025*dsigmadz)/(2*np.pi)*3600

        # Now grid the N values according to max depth
        bin_indices = np.digitize(z_sort, z_edges) - 1

        # Calculate means for each bin
        bin_means[:] = np.nan
        for i in range(nbins):
            mask = bin_indices == i
            if np.any(mask):
                bin_means[i] = np.nanmean(buoyfreq[mask])
        Ns.append(bin_means.copy())
        #embed(header='cugn/highres.py: 88')

    # Return
    return np.array(MLDs), np.array(Ns)