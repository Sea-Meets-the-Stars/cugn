""" Analsis related to SO paper """

import numpy as np

import pandas

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
    items = cugn_io.load_up(line)#, skip_dist=True)
    grid_extrem = items[0]
    times = items[2]

    Dt = (times.max() - times.min()).days

    # Number per year
    uni_cluster = np.unique(grid_extrem.cluster)
    nper_year = (uni_cluster.size - 1)/(Dt/365)

    print(f"Line: {line} -- clusters per year = {nper_year}")

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

# Command line execution
if __name__ == '__main__':

    '''
    # Clustering
    for line in defs.lines:
        frac_within_x_days(line, dt_days=1, dd_km=7.)

    # Count em
    nclusters = 0
    for line in defs.lines:
        nc = count_clusters(line)
        nclusters += nc
    print(f"Total number of clusters = {nclusters}")
    #
    '''

    #count_profiles()

    # Anamolies
    anamolies('90.0')