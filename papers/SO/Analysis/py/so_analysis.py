""" Analsis related to SO paper """

import numpy as np
import os
import glob

from scipy.interpolate import interp1d

from matplotlib import pyplot as plt

import pandas
import xarray

from gsw import conversions, density
import gsw

from ocpy.utils import plotting

from cugn import io as cugn_io
from cugn import defs
from cugn import utils as cugn_utils
from cugn import annualcycle


from IPython import embed

#lines =  ['56', '66', '80', '90']

def frac_within_x_days(line:str, dt_days:int=5, dd_km=5.):

    # Load
    items = cugn_io.load_up(line)#, skip_dist=True)
    grid_extrem = items[0]
    times = items[2]

    dists = grid_extrem.dist.values

    dt_pd = pandas.to_timedelta(dt_days, unit='d')

    # Loop through the profiles
    uni_prof = np.unique(grid_extrem.profile)
    n_prof = len(uni_prof)

    n_within = 0
    n_within_dd = 0
    max_ddist = 0.
    min_dists = []
    for prof in uni_prof:
        itime = times[grid_extrem.profile == prof][0]
        # Other
        other_times = times[grid_extrem.profile != prof]
        min_time = np.min(np.abs(itime - other_times))
        #
        if min_time < dt_pd:
            n_within += 1

        # Maximum distance
        if 'dist' not in grid_extrem.keys():
            continue


        imin_time = np.argmin(np.abs(itime - other_times))
        #embed(header='45 of so_analysis.py')
        ddist = np.abs(grid_extrem.dist[grid_extrem.profile != prof].values[imin_time] 
                       - grid_extrem.dist[grid_extrem.profile == prof].values[0])
        if ddist > max_ddist:
            max_ddist = ddist

        # Save min distance
        if ddist < dd_km:
            n_within_dd += 1
        min_dists.append(ddist)

    # Stats on min distance

    # Stats
    print("=====================================")
    print(f"Line {line}")
    print(f"Found {n_within} of {n_prof} profiles within {dt_days} days")
    print(f"Frac = {n_within/n_prof}")
    print(f"Max distance = {max_ddist}")
    print(f"Median min distance = {np.median(min_dists)}")
    print(f"Found {n_within_dd} of {n_prof} profiles within {dd_km} km")
    print(f"Frac dd = {n_within_dd/n_prof}")

def count_clusters(line:str):

    # Load
    items = cugn_io.load_up(line, kludge_MLDN=True)#, skip_dist=True)
    grid_extrem = items[0]
    times = items[2]

    Dt = (times.max() - times.min()).days

    # Number per year
    uni_cluster = np.unique(grid_extrem.cluster)
    nper_year = (uni_cluster.size - 1)/(Dt/365)

    print(f"Line: {line} -- clusters per year = {nper_year}")

    # Number in clusters
    in_cluster = np.sum(grid_extrem.cluster >= 0)
    print(f'Fraction of profiles in clusters = {in_cluster/len(grid_extrem)}')

    # Return
    return uni_cluster.size - 1
    
def count_profiles():
    """
    Counts the total number of profiles in the given dataset.

    """

    # Load
    nprofs = 0
    for line in defs.lines:
        items = cugn_io.load_up(line)#, skip_dist=True)
        grid_extrem = items[0]
        ds = items[1]
        times = items[2]
        # Count
        nprof = len(np.unique(ds.profile))
        nprofs += nprof
    #
    print(f"Total number of profiles = {nprofs}")

def load_annual(line:str, zmax:int=9, dmax:float=100.):

    # Load up
    items = cugn_io.load_up(line)#, skip_dist=True)
    grid_tbl = items[3]

    # Distance
    dist, _ = cugn_utils.calc_dist_offset(
        line, grid_tbl.lon.values, grid_tbl.lat.values)
    grid_tbl['dist'] = dist 
    times = pandas.to_datetime(grid_tbl.time.values)
    grid_tbl['doy'] = times.dayofyear

    # Cut down
    grid_tbl = grid_tbl[grid_tbl.depth <= zmax]
    grid_tbl = grid_tbl[grid_tbl.dist <= dmax]

    # Calculate <DO> at every location
    DO_annual = annualcycle.calc_for_grid(grid_tbl, line, 'oxumolkg')
    SO_annual = annualcycle.calc_for_grid(grid_tbl, line, 'ox')
    grid_tbl['ann_doxy'] = DO_annual
    grid_tbl['ann_SO'] = SO_annual

    return grid_tbl  

def check_mld_and_N(line:str, mission_name:str, mission_profile:int=None,
                    min_depth:float=0.5, debug:bool=False):

    # Load up
    items = cugn_io.load_up(line)#, gextrem='low')
    grid_extrem = items[0]
    ds = items[1]
    grid_tbl = items[3]

    # 
    lat = np.nanmedian(ds.lat.data)
    lon = np.nanmedian(ds.lon.data)

    # High res
    high_path = os.path.join(os.getenv('OS_SPRAY'), 'CUGN', 'HighRes')
    gfiles = glob.glob(os.path.join(high_path, f'SPRAY-FRSQ-{mission_name}-*.nc'))
    hfile = gfiles[0]                   
    ds_high = xarray.open_dataset(hfile)
    salinity = ds_high.salinity.values
    temperature = ds_high.temperature.values

    gm_idx = grid_extrem.mission.values == mission_name
    if mission_profile is None:
        mprofiles = np.unique(grid_extrem.mission_profile[gm_idx])
    else:
        mprofiles = [mission_profile]

    gMLDs, MLDs = [], []
    SOs, dMLDs = [], []
    for mission_profile in mprofiles:
        print(f'Working on {mission_name} {mission_profile}')
    
        # Pull the extrema
        gidx = (grid_extrem.mission.values == mission_name) & (
            grid_extrem.mission_profile.values == mission_profile)
        gMLD = np.median(grid_extrem.MLD.values[gidx])
        gMLDs.append(gMLD)

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

        # OC
        OC = gsw.O2sol(SA, CT, p, lon, lat)
        SO = ds_high.doxy.values[my_obs]/OC
        SOs += list(SO)

        # sigma0 
        sigma0 = density.sigma0(SA, CT)
        srt_z = np.argsort(ds_high.depth[my_obs])

        # sigma0 at surface
        sigma0_0 = np.mean(sigma0[srt_z[:5]])

        # Calculate MLD
        f = interp1d(sigma0[srt_z], ds_high.depth[my_obs].values[srt_z])
        MLD = f(sigma0_0 + defs.MLD_sigma0)
        MLDs.append(MLD)

        dMLDs += list(ds_high.depth[my_obs].values - MLD)

        if debug:
            gidx = (grid_tbl.mission.values == mission_name) & (
                grid_tbl.mission_profile.values == mission_profile)
            profile = grid_tbl[gidx].profile.values[0]
            zs = ds_high.depth[my_obs].values[srt_z]
            sigma0s = sigma0[srt_z]
            embed(header='166 of so_analysis.py')

    # Arrays
    MLDs = np.array(MLDs)
    gMLDs = np.array(gMLDs)
    dMLDs = np.array(dMLDs)
    SOs = np.array(SOs)

    # SO vs. dMLD
    fig = plt.figure(figsize=(18, 16))
    ax = plt.gca()

    ax.plot(dMLDs, SOs, 'o')
    ax.set_xlim(-50., 50.)
    ax.set_ylim(0.8, None)

    ax.axhline(1.1, color='k', ls=':')
    ax.axvline(0., color='k', ls='--')

    ax.set_xlabel('dMLD (m)')
    ax.set_ylabel('SO')
    plotting.set_fontsize(ax, 27.)

    plt.savefig('SO_vs_dMLD.png')
    plt.show()

    # Stats
    hyper = SOs > 1.1
    below = dMLDs[hyper] > 0
    print(f'Fraction of hyperoxia profiles with dMLD > 0 = {np.sum(below)/np.sum(hyper)}')
    embed(header='247 of so_analysis.py')
        
    # Plot
    fig = plt.figure(figsize=(18, 16))
    ax = plt.gca()

    ax.plot(gMLDs+5, MLDs, 'o')
    # 1-1 line
    ax.plot([0, np.max([gMLDs.max(),MLDs.max()])], 
             [0, np.max([gMLDs.max(),MLDs.max()])], 'k--')

    ax.set_xlabel('MLD from Grid')
    ax.set_ylabel('MLD from HighRes')
    plotting.set_fontsize(ax, 27.)

    plt.savefig('MLD_comparison.png')
    plt.show()

    # Check a few
    bad = (MLDs > 12.5) & (gMLDs < 8.)
    bprofiles = mprofiles[bad]
    ib = 0

    gidx = (grid_tbl.mission.values == mission_name) & (
        grid_tbl.mission_profile.values == bprofiles[ib])
    profile = grid_tbl[gidx].profile.values[0]
    mprofile = bprofiles[ib]
    my_obs = ds_high.profile_obs_index.values == mprofile


    embed(header='219 of so_analysis.py')
    bad = (MLDs < 5) & (gMLDs < 8.)
    bprofiles = mprofiles[bad]
    print(bprofiles[0])

# Command line execution
if __name__ == '__main__':

    '''
    # Clustering
    for line in defs.lines:
        frac_within_x_days(line, dt_days=1, dd_km=7.)
    '''

    # Count em
    nclusters = 0
    for line in defs.lines:
        nc = count_clusters(line)
        nclusters += nc
    print(f"Total number of clusters = {nclusters}")
    #

    #count_profiles()

    # Anamolies
    #anamolies('90.0')

    # ##
    # Assess MLD, N

    #mname = '20503001'
    #check_mld_and_N('90.0', mname)#, mission_profile=mprofile, debug=True)
    #mprofile = 14

    #mname = '22305801' # Line 80, 2022-07
    #check_mld_and_N('80.0', mname)#, mission_profile=mprofile, debug=True)