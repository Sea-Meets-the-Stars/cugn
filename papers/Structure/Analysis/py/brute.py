""" Perform a brute-force analysis of the structure function in 
    the dataasets. """

import os
import numpy as np

from cugn import gliderdata
from cugn import gliderpairs
from cugn import io as cugn_io
from cugn import utils as cugn_utils

from IPython import embed


def test_case(dataset='Calypso2022', iz=5):
    run(dataset, iz, ['duL','duL','duL'])


def run(dataset:str, iz:int, variables:list,
        max_time=10., avoid_same_glider=True, nbins=20,
        clobber:bool=True):

    rbins = 10**np.linspace(0., np.log10(400), nbins) # km

    # Load dataset
    gData = gliderdata.load_dataset(dataset)
    
    # Cut on valid velocity data 
    gData = gData.cut_on_good_velocity()

    # Generate pairs
    gPairs = gliderpairs.GliderPairs(
        gData, max_time=max_time, 
        avoid_same_glider=avoid_same_glider)

    # Velocity
    gPairs.calc_velocity(iz)

    if variables != ['duL','duL','duL']:
        raise ValueError('Only duL, duL, duL is supported')
    Sn_dict = gPairs.calc_Sn_vs_r(rbins, nboot=10000)
    gPairs.calc_corr_Sn(Sn_dict) 

    gPairs.add_meta(Sn_dict)

    # Output
    outfile = os.path.join('Outputs', cugn_io.gpair_filename(
        dataset, iz, Sn_dict['variables']))
    jdict = cugn_utils.jsonify(Sn_dict)
    cugn_utils.savejson(outfile, jdict, easy_to_read=True, overwrite=clobber)

if __name__ == '__main__':

    # Test
    test_case()

    # Calypso 2022
    #for iz in range(50):
        #run('Calypso2022', iz, ['duL','duL','duL'])