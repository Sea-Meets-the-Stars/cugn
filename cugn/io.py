""" I/O for CUGN data and analysis """
import os
import numpy as np
import xarray
import pandas

from cugn import grid_utils
from cugn import clusters
from cugn import utils as cugn_utils

from siosandbox import cat_utils

from IPython import embed

data_path = os.getenv('CUGN')

def line_files(line:str):
    """
    Returns a dictionary containing file paths for various data files related to the given line.

    Parameters:
    line (str): The line identifier.

    Returns:
    dict: A dictionary containing the following file paths:
        - datafile: The path to the data file for the given line.
        - gridtbl_file_full: The path to the full grid table file for the given line.
        - gridtbl_file_control: The path to the control grid table file for the given line.
        - edges_file: The path to the edges file for the given line.
    """
    datafile = os.path.join(data_path, f'CUGN_potential_line_{line[0:2]}.nc')
    gridtbl_file_full = os.path.join(data_path, f'full_grid_line{line[0:2]}.parquet')
    gridtbl_file_control = os.path.join(data_path, f'doxy_grid_line{line[0:2]}.parquet')
    edges_file = os.path.join(data_path, f'doxy_edges_line{line[0:2]}.npz')

    # dict em
    lfiles = dict(datafile=datafile, 
                  gridtbl_file_full=gridtbl_file_full, 
                  gridtbl_file_control=gridtbl_file_control, 
                  edges_file=edges_file)
    # Return
    return lfiles
    
def load_line(line:str, use_full:bool=False):
    """
    Load data for a given line.

    Parameters:
        line (str): The line identifier.
        use_full (bool, optional): Whether to use the full grid table file or the control grid table file. Defaults to False.

    Returns:
        dict: A dictionary containing the loaded data, including the dataset, grid table, and edges.
    """
    # Files
    lfiles = line_files(line)

    if use_full:
        grid_tbl = pandas.read_parquet(lfiles['gridtbl_file_full'])
    else:   
        grid_tbl = pandas.read_parquet(lfiles['gridtbl_file_control'])
    ds = xarray.load_dataset(lfiles['datafile'])
    edges = np.load(lfiles['edges_file'])

    # dict em
    items = dict(ds=ds, grid_tbl=grid_tbl, edges=edges)

    return items



def load_up(line:str, gextrem:str='high', use_full:bool=False):
    # Load
    items = load_line(line, use_full=use_full)
    grid_tbl = items['grid_tbl']
    ds = items['ds']

    # Fill
    grid_utils.fill_in_grid(grid_tbl, ds)

    # Extrema
    if gextrem == 'high':
        perc = 80.  # Low enough to grab them all
    elif gextrem == 'low':
        perc = 49.  # High enough to grab them all (Line 56.0)
    elif gextrem == 'low_noperc':
        perc = 50.
    elif gextrem == 'hi_noperc':
        perc = 50.
    else:
        raise IOError("Bad gextrem input")
    grid_outliers, tmp, _ = grid_utils.gen_outliers(line, perc)

    if gextrem == 'high':
        extrem = grid_outliers.SO > 1.1
    elif gextrem == 'low':
        extrem = (grid_outliers.SO < 0.9) & (
            grid_outliers.depth <= 1)
    elif gextrem == 'low_noperc':
        grid_outliers = grid_tbl.copy()
        extrem = (grid_outliers.SO < 0.9) & (
            grid_outliers.depth <= 1)
    elif gextrem == 'hi_noperc':
        grid_outliers = grid_tbl.copy()
        extrem = grid_outliers.SO > 1.1 
    else:
        raise IOError("Bad gextrem input")
    grid_extrem = grid_outliers[extrem].copy()
    times = pandas.to_datetime(grid_extrem.time.values)

    # DEBUG
    #tmin = pandas.Timestamp('2020-08-22')
    #tmax = pandas.Timestamp('2020-09-11')
    #in_event = (times >= tmin) & (times <= tmax)
    #embed(header='cugn/io.py: 89')
    #ttimes = pandas.to_datetime(grid_tbl.time.values)
    #in_t = (ttimes >= tmin) & (ttimes <= tmax) & (grid_tbl.depth <= 1)

    # Fill in N_p, chla_p
    grid_utils.find_perc(grid_tbl, 'N')
    grid_utils.find_perc(grid_tbl, 'chla')

    dp_gt = grid_tbl.depth*100000 + grid_tbl.profile
    dp_ge = grid_extrem.depth*100000 + grid_extrem.profile
    ids = cat_utils.match_ids(dp_ge, dp_gt, require_in_match=True)
    assert len(np.unique(ids)) == len(ids)

    grid_extrem['N_p'] = grid_tbl.N_p.values[ids]
    grid_extrem['chla_p'] = grid_tbl.chla_p.values[ids]

    # Add to df
    grid_extrem['year'] = times.year
    grid_extrem['doy'] = times.dayofyear

    # Add distance from shore
    dist, _ = cugn_utils.calc_dist_offset(
        line, grid_extrem.lon.values, grid_extrem.lat.values)
    grid_extrem['dist'] = dist

    # Cluster me
    clusters.generate_clusters(grid_extrem)
    cluster_stats = clusters.cluster_stats(grid_extrem)

    return grid_extrem, ds, times, grid_tbl

def gpair_filename(dataset:str, iz:int):

    # Generate a filename
    outfile = f'gpair_{dataset}_z{iz:02d}'
    outfile += '.json'

    return outfile