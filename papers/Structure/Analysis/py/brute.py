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
    run(dataset, iz, 'duLduLduL')


def run(dataset:str, iz:int, 
        max_time=10., avoid_same_glider=True, nbins=20,
        clobber:bool=False):

    outfile = os.path.join('Outputs', cugn_io.gpair_filename(
        dataset, iz, not avoid_same_glider))
    if os.path.exists(outfile) and not clobber:
        print(f'Exists: {outfile}')
        return

    rbins = 10**np.linspace(0., np.log10(400), nbins) # km

    # Load dataset
    gData = gliderdata.load_dataset(dataset)
    
    # Cut on valid velocity data 
    gData = gData.cut_on_good_velocity()

    # Generate pairs
    gPairs = gliderpairs.GliderPairs(
        gData, max_time=max_time, 
        avoid_same_glider=avoid_same_glider)


    all_dicts = []
    for variables in ['duLduLduL', 'duLdSdS', 'duLdTdT', 'duLduTduT',
                      'duTduTduT']:
        # Velocity
        gPairs.calc_delta(iz, variables)
        gPairs.calc_Sn(variables)

        Sn_dict = gPairs.calc_Sn_vs_r(rbins, nboot=10000)
        gPairs.calc_corr_Sn(Sn_dict) 

        gPairs.add_meta(Sn_dict)
        all_dicts.append(Sn_dict)


    # Merge the dicts
    final_dict = cugn_utils.merge_dicts(all_dicts)

    # Output
    jdict = cugn_utils.jsonify(final_dict)
    cugn_utils.savejson(outfile, jdict, easy_to_read=True, overwrite=clobber)
    print(f'Wrote: {outfile}')

if __name__ == '__main__':

    # Test
    #test_case()

    # Calypso 2022
    for dataset in ['Calypso2019', 'Calypso2022', 'ARCTERX']:
        for iz in range(50):
            if iz != 5:
                continue
            #run('Calypso2022', iz, 'duLduLduL])
            run(dataset, iz, clobber=True)
            run(dataset, iz, avoid_same_glider=False)