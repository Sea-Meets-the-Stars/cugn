import os
import sys
import numpy as np

from profiler import gliderdata
from profiler import profilerpairs

sys.path.append(os.path.abspath("../Analysis/py"))
import glider_io
import struct_defs

from IPython import embed

def rbinning(btype:str, dataset:str=None, nbins=None):
    # nbins
    if nbins is None:
        nbins = struct_defs.dataset_binning[dataset][btype]

    # Do it
    if btype == 'log':
        rbins = 10**np.linspace(0., np.log10(400), nbins) # km
    elif btype == 'lin':
        rbins = np.linspace(0,400,nbins);
    else:
        raise ValueError(f'Bad binning style: {btype}')
    return rbins

def load_SF(dataset:str, 
            variables = 'duLduLduL', 
            iz:int=5, 
            minN:int=struct_defs.minN, 
            btype:str=struct_defs.btype,
            max_time:float=struct_defs.max_time,
            avoid_same_glider:bool=True):
    """Load glider data and compute structure functions for a given dataset.

    Loads profiler data, generates profile pairs, computes velocity differences
    and structure functions (Sn) binned by separation distance.

    Args:
        dataset: Name of the dataset (e.g. 'Calypso2019', 'Calypso2022', 'ARCTERX-2023').
        variables: Structure function variable specification string.
        iz: Depth index for computing velocity differences. Negative values
            trigger isopycnal coordinate analysis.
        minN: Minimum number of pairs required per radial bin.
        btype: Type of binning ('log' or 'lin').
        max_time: Maximum time separation (days) for valid profile pairs.
        avoid_same_glider: If True, exclude pairs from the same glider.

    Returns:
        dict: Keys 'gPairs' (ProfilerPairs object), 'Sn_dict' (structure function
            results with bootstrapped errors), 'goodN' (boolean mask for bins
            with sufficient pairs), 'Skeys' (list of structure function key names).
    """
   # Load dataset
    profilers = glider_io.load_dataset(dataset)

    # Binning
    rbins = rbinning(btype, dataset)

    # Generate pairs
    #gData = gliderdata.load_dataset(dataset)
    gPairs = profilerpairs.ProfilerPairs(
        profilers, max_time=max_time,
        avoid_same_glider=avoid_same_glider,
        remove_nans=True,
        debug=False, 
        randomize=False)
    # Isopycnals?
    if iz < 0:
        gPairs.prep_isopycnals('t')
    #gData = gData.cut_on_good_velocity()
    #gData = gData.cut_on_reltime(tcut)

    gPairs.calc_delta(iz, variables, skip_velocity=False)
    gPairs.calc_Sn(variables)

    Sn_dict = gPairs.calc_Sn_vs_r(rbins, nboot=100)
    gPairs.calc_corr_Sn(Sn_dict)
    gPairs.add_meta(Sn_dict)

    goodN = np.array(Sn_dict['config']['N']) > minN
    Skeys = ['S1_duL', 'S2_duL**2', 'S3_'+variables]

    # Return
    rdict = dict(
        gPairs=gPairs, Sn_dict=Sn_dict, 
        goodN=goodN, Skeys=Skeys, rbins=rbins)
    return rdict