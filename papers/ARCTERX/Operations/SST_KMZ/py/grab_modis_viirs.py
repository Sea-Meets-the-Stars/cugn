""" Generate a Merged SST product for ARCTERX2 """

import xarray
import argparse

from matplotlib import pyplot as plt

from remote_sensing.download import podaac
from remote_sensing.healpix import rs_healpix
from remote_sensing.healpix import utils as hp_utils
from remote_sensing.healpix import combine as hp_combine
from remote_sensing import io as rs_io
from remote_sensing import units
from remote_sensing import kml as rs_kml

from IPython import embed

# Globals
lon_lim = (127.,134)
lat_lim = (18.,23)

def main(args):

    # Grab the latest data
    viirs_files, _ = podaac.grab_file_list(
            'VIIRS_NPP-STAR-L2P-v2.80',
            t_end=None,
            dt_past=dict(days=args.ndays),
            bbox='127,18,134,23')

    print(f"Downloading VIIRS files")
    local_viirs = podaac.download_files(viirs_files, verbose=True)

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("Merged SST KMZ script")
    parser.add_argument("--namsr2", type=int, 
                        default=1, help="Number of exposures of AMSR2 to combine")
    parser.add_argument("--ndays", type=int, 
                        default=30, help="Number of days into the past to consdier for images")
    parser.add_argument("--t_end", type=str, 
                        help="End time, ISO format e.g. 2025-02-07T04:00:00Z")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    parser.add_argument('-s', '--show', default=False, action='store_true',
                        help='show extra plots?')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Print more to the screen?')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Clobber existing files')
    parser.add_argument('--use_json', type=str, 
                        help='Load files from the JSON file')

    args = parser.parse_args()
    
    return args

        
if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()
    main(args)
    