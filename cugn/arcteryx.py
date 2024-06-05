""" Methods related to Arcteryx """
import os
import numpy as np

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

def cut_on_time(tmax:float, adict:dict=None):
    """ Cut the CTD data on time """
    if adict is None:
        adict = load_ctd()

    # Cut
    t = adict['time']
    dt = np.zeros((t.size, t.size))
    for kk in range(t.size):
        dt[kk] = (t[kk] - t)/3600.  # hours

    # Restrcit to the positive values to avoid double counting
    pos = dt > 0.
    tcut = dt < tmax

    idx = np.where(tcut & pos)
    final_dt = dt[idx]

    # Return
    return final_dt, idx

def calc_r(adict, idx):
    d0 = adict['dist'][idx[0]]
    d1 = adict['dist'][idx[1]]
    #
    o0 = adict['offset'][idx[0]]
    o1 = adict['offset'][idx[1]]

    r = np.sqrt((d0-d1)**2 + (o0-o1)**2)

    # Generate the r dict
    rdict = {}
    rdict['r'] = r
    rdict['d'] = d1-d0
    rdict['o'] = o1-o0
    rdict['dN'] = (d1-d0)/r
    rdict['oN'] = (o1-o0)/r

    # Return
    return rdict

def calc_du(adict:dict, rdict:dict, idx:np.ndarray, iz:int=0):

    # Unpack
    u0 = adict['udop'][iz][idx[0]]
    u1 = adict['udop'][iz][idx[1]]
    v0 = adict['vdop'][iz][idx[0]]
    v1 = adict['vdop'][iz][idx[1]]

    # du
    du = u1-u0
    dv = v1-v0

    # duL
    duL = rdict['dN']*du + rdict['oN']*dv

    # Save
    udict = {}
    udict['du'] = du
    udict['dv'] = dv
    udict['duL'] = duL

    # Return
    return udict