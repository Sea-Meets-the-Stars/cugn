""" Methods related to Arcteryx """
import os

from scipy.io import loadmat

path = '/home/xavier/Projects/Oceanography/data/Arcteryx'

def load_ctd():
    """ Load the CTD data for Arcteryx """
    dfile = os.path.join(path, 'arcteryx_ctd.mat')
    mat_d = loadmat(dfile)

    # Generate a useful dict

    adict = {}

    # Scalars
    for key in ['x0', 'x1', 'y0', 'y1']:
        adict[key] = mat_d['ctd'][key][0][0][0][0]

    # Depth arrays
    for key in ['depth']:
        adict[key] = mat_d['ctd'][key][0][0].flatten()

    # Profile arrays
    for key in ['lat', 'lon', 'time', 'dist', 'offset']:
        adict[key] = mat_d['ctd'][key][0][0].flatten()

    # Profile + depth
    for key in ['udop', 'vdop']:
        adict[key] = mat_d['ctd'][key][0][0]

    # Return
    return adict