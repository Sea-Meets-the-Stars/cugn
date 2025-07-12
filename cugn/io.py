""" I/O for CUGN data and analysis """
import os
import numpy as np
import xarray
import pandas

from cugn import grid_utils
from cugn import clusters
from cugn import utils as cugn_utils
from cugn import defs as cugn_defs

from cugn import utils

from IPython import embed

data_path = cugn_defs.data_path

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
    fullres_file = os.path.join(data_path, f'fullres_{line[0:2]}.parquet')

    # dict em
    lfiles = dict(datafile=datafile, 
                  gridtbl_file_full=gridtbl_file_full,
                  gridtbl_file_control=gridtbl_file_control,
                  edges_file=edges_file,
                  fullres_file=fullres_file)
    # Return
    return lfiles
    
def load_line(line:str, use_full:bool=False, add_fullres:bool=False):
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
        grid_file = lfiles['gridtbl_file_full']
    else:   
        grid_file = lfiles['gridtbl_file_control']
    print(f"Loading: {os.path.basename(grid_file)}")
    grid_tbl = pandas.read_parquet(grid_file)
    ds = xarray.load_dataset(lfiles['datafile'])
    edges = np.load(lfiles['edges_file'])

    # dict em
    items = dict(ds=ds, grid_tbl=grid_tbl, edges=edges)

    if add_fullres:
        full_res = pandas.read_parquet(lfiles['fullres_file'])
        items['full_res'] = full_res

    return items


def load_up(line:str, gextrem:str='high', use_full:bool=False,
            kludge_MLDN:bool=False):
    """
    Load data and perform various operations on it.

    Args:
        line (str): The line to load data for.
        gextrem (str, optional): The type of extremum to consider. Defaults to 'high'.
        use_full (bool, optional): Whether to use the full data or not. Defaults to False.
        kludge_MLDN (bool, optional): Whether to kludge the MLDN data. Defaults to False.

    Returns:
        tuple: A tuple containing the following:
            - grid_extrem (pandas.DataFrame): DataFrame containing the extreme values.
            - ds (xarray.Dataset): The loaded dataset.
            - times (pandas.DatetimeIndex): Datetime index of the extreme values.
            - grid_tbl (pandas.DataFrame): DataFrame containing the loaded data.
    """
    # Load
    items = load_line(line, use_full=use_full)
    grid_tbl = items['grid_tbl']
    ds = items['ds']

    # Fill
    grid_utils.fill_in_grid(grid_tbl, ds, kludge_MLDN=kludge_MLDN)

    # Extrema
    if gextrem in ['high', 'highAOU']:
        perc = 80.  # Low enough to grab them all
    elif gextrem == 'low':
        perc = 49.  # High enough to grab them all (Line 56.0)
    elif gextrem == 'low_noperc':
        perc = 50.
    elif gextrem == 'hi_noperc':
        perc = 50.
    else:
        raise IOError("Bad gextrem input")
    grid_outliers, tmp, _ = grid_utils.gen_outliers(line, perc, grid_tbl=grid_tbl)

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
    elif gextrem == 'highAOU':
        extrem = grid_outliers.AOU > cugn_defs.AOU_hyper
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
    ids = utils.match_ids(dp_ge, dp_gt, require_in_match=True)
    assert len(np.unique(ids)) == len(ids)

    grid_extrem['N_p'] = grid_tbl.N_p.values[ids]
    grid_extrem['chla_p'] = grid_tbl.chla_p.values[ids]

    # Add to df
    grid_extrem['year'] = times.year
    grid_extrem['doy'] = times.dayofyear

    # Add distance from shore and offset from line
    dist, offset = cugn_utils.calc_dist_offset(
        line, grid_extrem.lon.values, grid_extrem.lat.values)
    grid_extrem['dist'] = dist
    grid_extrem['offset'] = offset

    # Cluster me
    clusters.generate_clusters(grid_extrem)
    cluster_stats = clusters.cluster_stats(grid_extrem)

    return grid_extrem, ds, times, grid_tbl

def load_upwelling():
    # CUTI
    cuti_file = os.path.join(os.getenv('OS_CCS'), 'Upwelling',
                             'CUTI_daily.nc')
    cuti = xarray.open_dataset(cuti_file)
    # 
    beuti_file = os.path.join(os.getenv('OS_CCS'), 'Upwelling',
                             'BEUTI_daily.nc')
    beuti = xarray.open_dataset(beuti_file)
    # Return
    return cuti, beuti

def gpair_filename(dataset:str, iz:int, same_glider:bool):
    """
    Generate a filename for a glider pair dataset.

    Parameters:
        dataset (str): The name of the dataset.
        iz (int): The depth level of the dataset.
        same_glider (bool): Indicates whether the glider pair is the same glider or not.

    Returns:
        str: The generated filename for the glider pair dataset.
    """



    same_lbl = 'self' if same_glider else 'other'

    # Generate a filename
    outfile = f'gpair_{dataset}_z{iz:02d}_{same_lbl}'
    outfile += '.json'

    return outfile