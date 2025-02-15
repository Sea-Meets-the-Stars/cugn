""" Simple Class to hold glider data """
import os
import glob

import numpy as np
import warnings

from abc import ABCMeta

from scipy.io import loadmat

from cugn import utils as cugn_utils

from IPython import embed

def load_dataset(dataset:str):
    """
    Load a dataset based on the provided dataset name.

    Parameters:
        dataset (str): The name of the dataset to load.

    Returns:
        cData (CTDData): The loaded CTDData object.

    Raises:
        ValueError: If the provided dataset is not supported.
    """
    if dataset == 'ARCTERX':
        dfile = os.path.join(
            os.getenv('OS_SPRAY'), 'ARCTERX', 'arcterx_ctd.mat')
    elif dataset == 'ARCTERX-Leg2':
        dfiles = glob.glob(os.path.join(
            os.getenv('OS_SPRAY'), 'ARCTERX', 'Leg2', '*.mat'))
        embed(header='33 of figs')
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
    Abstract base class for Gliders

    Attributes:

    dataset (str): The name of the glider data


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

    profile_arrays:list = None
    depth_arrays:list = None
    profile_depth_arrays:list = None

    def __init__(self, datafile:str, dataset:str):
        self.datafile = datafile
        self.dataset = dataset

        self.load_data()

    @classmethod
    def from_list(cls, datafiles:list, dataset:str):
        """
        Create a GliderData object from a list of data files.

        Args:
            datafiles (list): A list of data files to load.
            dataset (str): The name of the dataset.

        Returns:
            GliderData: A GliderData object containing the data from the provided files.
        """
        # Load the first file
        gData = cls(datafiles[0], dataset)

        # Load the rest
        for datafile in datafiles[1:]:
            gData2 = cls(datafile, dataset)
            gData = gData + gData2

        # Return
        return gData

    def __add__(self, other):
        """
        Add two GliderData objects together.

        Args:
            other (GliderData): The other GliderData object to add.

        Returns:
            GliderData: A new GliderData object containing the combined data.
        """
        # Check the datasets
        if self.dataset != other.dataset:
            raise ValueError("Datasets do not match")

        # Check depths
        assert np.allclose(self.depth, other.depth)

        # Concatenate the data
        gData = self.__class__(self.datafile, self.dataset)
        for key in self.profile_arrays:
            setattr(gData, key, np.concatenate([getattr(self, key), getattr(other, key)]))
        for key in self.profile_depth_arrays:
            setattr(gData, key, np.concatenate([getattr(self, key), getattr(other, key)], axis=1))

        # Return
        return gData
    
    def load_data(self):
        pass

    @property
    def uniq_missions(self):
        """ Return the unique missions """
        return np.unique(self.missid)

    def profile_subset(self, profiles: np.ndarray, 
                       init:bool=True):
        """
        Create a subset of the GliderData object based on the given profiles.

        Args:
            profiles (np.ndarray): An array of profile indices to include in the subset.

        Returns:
            GliderData: A new GliderData object containing the subset of profiles.
        """
        # Init
        if init:
            gData = self.__class__(self.datafile, self.dataset)
        else:
            gData = self

        # Cut on profiles
        for key in self.profile_arrays:
            setattr(gData, key, getattr(self, key)[profiles])
        for key in self.profile_depth_arrays:
            setattr(gData, key, getattr(self, key)[:, profiles])

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
            #embed(header='144 of gliderdata')
            self.profile_depth_arrays = ['udop', 'vdop', 
                                         'udopacross', 'udopalong',
                                         's', 't']
            for key in self.profile_depth_arrays:
                setattr(self, key, mat_d['ctd'][key][0][0])

            # Survey specific cuts
            if self.dataset == 'Calypso2022':
                warnings.warn("Trimming last 3 weeks of Calypso2022 data")
                # Trim the last 3 weeks
                maxt = np.max(self.time)
                mint = np.min(self.time)
                goodt = self.time < (maxt - 12*24*3600)
                goodt &= (self.time > (mint + 3*24*3600))
                self = self.profile_subset(np.where(goodt)[0], init=False)


        def cut_on_good_velocity(self):
            """
            Cuts the glider data based on good velocity values.

            Returns:
                gData (GliderData): A subset of the original GliderData object containing only the profiles with good velocity values.
            """
            # Cut on velocity
            good = np.isfinite(self.udop) & np.isfinite(self.vdop)
            idx = np.where(good)
            gd_profiles = np.unique(idx[1])

            # Cut
            gData = self.profile_subset(gd_profiles)

            # Return
            return gData

        def cut_on_reltime(self, timecut:tuple):
            """
            Cuts the glider data based on good velocity values.

            Variables:
                timecut (tuple): range of times to include 0 to 1

            Returns:
                gData (GliderData): A subset of the original GliderData object containing only the profiles with good velocity values.
            """

            # Relative time
            min_time = self.time.min()
            max_time = self.time.max()

            reltime = (self.time - min_time) / (max_time - min_time)

            # Cut on time
            keep = (reltime > timecut[0]) & (reltime <= timecut[1])

            # Cut
            gData = self.profile_subset(np.where(keep)[0])

            # Return
            return gData

        def __repr__(self):
            """ Return the representation of the CTDData object """
            rstr = f"CTDData object for {self.dataset}\n"
            rstr += f"  Number of profiles: {len(self.time)}\n"
            rstr += f"  Time range: {self.time.min()} to {self.time.max()}\n"
            # Variables
            rstr += "  Variables:\n"
            for key in self.depth_arrays:
                rstr += f"    {key}: {getattr(self, key).shape}\n"
            for key in self.profile_arrays:
                rstr += f"    {key}: {getattr(self, key).shape}\n"
            for key in self.profile_depth_arrays:
                rstr += f"    {key}: {getattr(self, key).shape}\n"
            return rstr


class FieldData(CTDData):
    """
    Class to hold Spray data in the field
    """
    dtype = 'Field'

    def __init__(self, datafile:str, dataset:str):

        CTDData.__init__(self, datafile, dataset)

    def load_data(self):
        """ Load the Field data for ARCTERX """
        mat_d = loadmat(self.datafile)

        # Scalars
        #self.scalar_keys = ['x0', 'x1', 'y0', 'y1']
        #for key in self.scalar_keys:
        #    setattr(self, key, mat_d['ctd'][key][0][0][0][0])

        # Arrays
        variables = ['time', 'lat', 'lon', 
            'n', 'n', 'n', 'n', 'n',
            'depth', 't', 's', 'fl', 'theta']

        self.depth_arrays = ['depth']
        self.profile_arrays = ['lat', 'lon', 'time']
        self.profile_depth_arrays = ['s', 't', 'fl', 'theta']

        for ss, key in enumerate(variables):
            if key == 'n':
                continue
            idx = variables.index(key)
            # Ignore NaN
            if ss == 0:
                gdi = np.isfinite(mat_d['bindata'][0][0][idx].flatten())
            # one-d
            if mat_d['bindata'][0][0][key].shape[1] == 1:
                if key == 'depth':
                    setattr(self, key, mat_d['bindata'][0][0][idx].flatten())
                else:
                    setattr(self, key, mat_d['bindata'][0][0][idx].flatten()[gdi])
            else:
                setattr(self, key, mat_d['bindata'][0][0][idx][:,gdi])

        
        # Mission ID
        key = 'missid'
        self.profile_arrays += [key]
        missid = int(os.path.basename(self.datafile).split('.')[0])
        setattr(self, key, missid*np.ones_like(self.lat, dtype=int))

        # Generate dist and offset
        #  dist is distance to the North from the median lon (km)
        #  offset is distance to the East from the median lon
        self.med_lon = np.median(self.lon)
        self.med_lat = np.median(self.lat)
        latendpts = (self.med_lat-1., self.med_lat+1.)
        lonendpts = (self.med_lon, self.med_lon)

        # dist
        key = 'dist'
        self.profile_arrays += [key]
        dist, offset = cugn_utils.calc_dist_offset('None',
            self.lon, self.lat, endpoints=(lonendpts, latendpts))
        # Fill in
        self.dist = dist
        key = 'offset'
        self.profile_arrays += [key]
        self.offset = offset


    def __repr__(self):

        # Use super
        rstr = super().__repr__()

        # Replace 
        rstr = rstr.replace('CTDData', 'FieldData')

        # Return
        return rstr