import numpy as np

from cugn import gliderdata
from cugn import gliderpairs
from cugn import io as cugn_io
from cugn import utils as cugn_utils

from IPython import embed


def test_case(dataset='Calypso2022', max_time=10., 
              avoid_same_glider=True, iz=5, nbins=20):

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

    rbins = 10**np.linspace(0., np.log10(400), nbins) # km
    Sn_dict = gPairs.calc_Sn_vs_r(rbins, nboot=10000)
    gPairs.calc_corr_Sn(Sn_dict) 

    gPairs.add_meta(Sn_dict)

    # Output
    outfile = cugn_io.gpair_filename(
        dataset, iz, Sn_dict['variables'])
    jdict = cugn_utils.jsonify(Sn_dict)
    cugn_utils.savejson(outfile, jdict, easy_to_read=True)

if __name__ == '__main__':
    test_case()