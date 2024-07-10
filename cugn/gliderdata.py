""" Simple Class to hold glider data """
import os
import numpy as np

from abc import ABCMeta

from scipy.io import loadmat

from IPython import embed

def load_dataset(dataset:str):
    if dataset == 'ARCTERX':
        dfile = os.path.join(
            os.getenv('OS_SPRAY'), 'ARCTERX', 'arcterx_ctd.mat')
    elif dataset == 'Calypso2019':
        dfile = os.path.join(
            os.getenv('OS_SPRAY'), 'Calypso', 'calypso2019_ctd.mat')
    elif dataset == 'Calypso2022':
        dfile = os.path.join(
            os.getenv('OS_SPRAY'), 'Calypso', 'calypso2022_ctd.mat')
    else: 
        raise ValueError(f"Dataset {dataset} not supported")
    # Load
    cData = CTDData(dfile, dataset)
    return cData

class GliderData:
    """
    Abstract base class for non-water absoprtion

    Attributes:

    """
    __metaclass__ = ABCMeta

    dataset = None
    """
    The name of the glider data
    """

    dtype:str = None
    """
    The type of data
    """

    lat = None
    lon = None
    time = None
    dist = None
    offset = None
    missid = None
    udop = None
    vdop = None

    profile_arrays = None
    depth_arrays = None
    profile_depth_arrays = None


    def __init__(self, datafile:str, dataset:str):
        self.datafile = datafile
        self.dataset = None

        self.load_data()

    def load_data(self):
        pass

    @property
    def uniq_missions(self):
        """ Return the unique missions """
        return np.unique(self.missid)

    def profile_subset(self, profiles:np.ndarray):
        # Init
        gData = self.__class__(self.datafile, self.dataset)

        # Cut on profiles
        for key in self.profile_arrays:
            setattr(gData, key, getattr(self, key)[profiles])
        for key in self.profile_depth_arrays:
            setattr(gData, key, getattr(self, key)[:,profiles])

        # Return
        return gData

class CTDData(GliderData):
    """
    Class to hold CTD data 
    """
    dtype = 'CTD'

    def __init__(self, datafile:str, dataset:str):

        
        GliderData.__init__(self, datafile, dataset)

    def load_data(self):
        """ Load the CTD data for Arcteryx """
        mat_d = loadmat(self.datafile)

        # Scalars
        self.scalar_keys = ['x0', 'x1', 'y0', 'y1']
        for key in self.scalar_keys:
            setattr(self, key, mat_d['ctd'][key][0][0][0][0])

        # Depth arrays
        self.depth_arrays = ['depth']
        for key in self.depth_arrays:
            setattr(self, key, mat_d['ctd'][key][0][0].flatten())

        # Profile arrays
        self.profile_arrays = ['lat', 'lon', 'time', 'dist', 'offset', 'missid']
        for key in self.profile_arrays:
            setattr(self, key, mat_d['ctd'][key][0][0].flatten())

        # Profile + depth
        self.profile_depth_arrays = ['udop', 'vdop', 'udopacross', 'udopalong']
        for key in self.profile_depth_arrays:
            setattr(self, key, mat_d['ctd'][key][0][0])

    def cut_on_good_velocity(self):

        # Cut on velocity
        good = np.isfinite(self.udop) & np.isfinite(self.vdop)
        idx = np.where(good)
        gd_profiles = np.unique(idx[1])

        # Cut
        gData = self.profile_subset(gd_profiles)

        # Return
        return gData