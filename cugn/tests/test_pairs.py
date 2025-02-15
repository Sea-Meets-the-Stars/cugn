""" Tests for Spray Gliders """

import glob

import pytest

from cugn import gliderdata
from cugn import profilepairs

from IPython import embed

dataset = 'ARCTERX-Leg2'

datafiles = glob.glob('/home/xavier/Projects/Oceanography/data/Spray/ARCTERX/Leg2/*.mat')
datafiles.sort()
arcterx2 = gliderdata.SprayData.from_list(datafiles, dataset, adcp_on=False, in_field=True)

max_time = 10.
gPairs = profilepairs.ProfilerPairs(arcterx2, max_time=max_time)

embed(header='21 of test_pairs')
