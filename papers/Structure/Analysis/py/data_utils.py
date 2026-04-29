import os
import sys
import numpy as np

from profiler import gliderdata
from profiler import profilerpairs

sys.path.append(os.path.abspath("../Analysis/py"))
import glider_io

from IPython import embed

def load_SF(dataset:str, variables = 'duLduLduL', iz:int=5,
    minN:int=10):
    """Load glider data and compute structure functions for a given dataset.

    Loads profiler data, generates profile pairs, computes velocity differences
    and structure functions (Sn) binned by separation distance.

    Args:
        dataset: Name of the dataset (e.g. 'Calypso2019', 'Calypso2022', 'ARCTERX-2023').
        variables: Structure function variable specification string.
        iz: Depth index for computing velocity differences. Negative values
            trigger isopycnal coordinate analysis.
        minN: Minimum number of pairs required per radial bin.

    Returns:
        dict: Keys 'gPairs' (ProfilerPairs object), 'Sn_dict' (structure function
            results with bootstrapped errors), 'goodN' (boolean mask for bins
            with sufficient pairs), 'Skeys' (list of structure function key names).
    """
   # Load dataset
    profilers = glider_io.load_dataset(dataset)

    # Cut on valid velocity data 
    nbins = 20
    rbins = 10**np.linspace(0., np.log10(400), nbins) # km

    # Generate pairs
    #gData = gliderdata.load_dataset(dataset)
    gPairs = profilerpairs.ProfilerPairs(
        profilers, max_time=10.,
        avoid_same_glider=True,
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
    rdict = dict(gPairs=gPairs, Sn_dict=Sn_dict,
                 goodN=goodN, Skeys=Skeys)
    return rdict