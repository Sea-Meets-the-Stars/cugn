""" Methods for clustering the outliers 
and then analyzing these clusters """

import numpy as np
import pandas 

from sklearn.cluster import AgglomerativeClustering, DBSCAN

from IPython import embed

def generate_clusters(grid_outliers:pandas.DataFrame,
                      time_scl:float=5.,
                      doff_scl:float=20./3,
                      #doff_scl:float=10.,
                      z_scl:float=5.,
                      min_samples:int=10):
    """ Generate clusters of outliers for a given line
    and percentage

    Table is modified in place

    Args:
        grid_outliers (pandas.DataFrame): table of outliers
        time_scl (float, optional): _description_. Defaults to 3..
        z_scl (float, optional): _description_. Defaults to 5..
        min_samples (int, optional): Minimum number of
            objects for a cluster in DBSCAN. Defaults to 10.

    """
    # ###########
    # Time
    ptimes = pandas.to_datetime(grid_outliers.time.values)
    mjd = ptimes.to_julian_date()

    # Offset
    t = mjd - mjd.min()
    # Scale
    tscl = t / time_scl

    # Longitdue
    #dl = (grid_outliers.lon.max() - grid_outliers.lon) 
    #lscl = dl.values * 100 / dl.max()

    # Distance from shore
    doff = (grid_outliers.dist.max() - grid_outliers.dist) 
    lscl = doff / doff_scl
    #lscl = doff.values * 100 / doff.max()

    # Depth
    zscl = grid_outliers.z.values / z_scl

    # Package for sklearn
    X = np.zeros((len(grid_outliers), 3))
    X[:,0] = tscl
    X[:,1] = lscl
    X[:,2] = zscl

    # Fit
    dbscan = DBSCAN(eps=3, min_samples=min_samples)
    dbscan.fit(X)
    print(f"Found {len(np.unique(dbscan.labels_))} unique clusters")

    grid_outliers['cluster'] = dbscan.labels_

def cluster_stats(grid_outliers:pandas.DataFrame):
    """
    Calculate statistics for each cluster in the given DataFrame.

    Method calculates the mean, max, and min values for each cluster

    Parameters:
        grid_outliers (pandas.DataFrame): 
            DataFrame containing the data for clustering.

    Returns:
        stats_tbl (pandas.DataFrame): 
            DataFrame containing the calculated statistics for each cluster.
    """
    # Stats
    cluster_IDs = np.unique(grid_outliers.cluster.values[
        grid_outliers.cluster.values >= 0])

    # Loop on clusters
    stats = {}
    mean_keys = ['z', 'lon','doxy', 'time', 'SA', 'CT', 
                 'sigma0', 'SO', 'chla', 'dist']
    for key in mean_keys:
        stats[key] = []
    max_keys = ['doxy', 'SO', 'chla', 'dist']
    for key in max_keys:
        stats['max_'+key] = []
        stats['min_'+key] = []
    # A few others
    for key in ['N', 'Dtime', 'Ddist', 'ID', 'Cdist']:
        stats[key] = []

    # Loop on clusters
    for cluster_ID in cluster_IDs:
        # Grab em
        in_cluster = grid_outliers.cluster.values == cluster_ID
        stats['ID'].append(cluster_ID)
        stats['N'].append(in_cluster.sum())

        # Means
        for key in mean_keys:
            stats[key].append(grid_outliers[in_cluster][key].mean())

        # Max/min
        for key in max_keys:
            stats['max_'+key].append(grid_outliers[in_cluster][key].max())
            stats['min_'+key].append(grid_outliers[in_cluster][key].min())

        # Duration
        stats['Dtime'].append(grid_outliers[in_cluster].time.max() - grid_outliers[in_cluster].time.min())

        # Distance
        stats['Ddist'].append(stats['max_dist'][-1] - stats['min_dist'][-1])
        stats['Cdist'].append((stats['max_dist'][-1] + stats['min_dist'][-1])/2.)

    # Package
    stats_tbl = pandas.DataFrame(stats)
    stats_tbl['cluster'] = cluster_IDs
    #embed(header='118 of clusters')

    return stats_tbl
