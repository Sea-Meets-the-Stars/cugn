""" Classes for holding Float data """

import pymatreader

from cugn import utils as cugn_utils
from cugn import profiledata

from IPython import embed

class SoloData(profiledata.ProfileData):

    def __init__(self, datafile:str, dataset:str):

        profiledata.ProfileData.__init__(self, datafile, dataset)
