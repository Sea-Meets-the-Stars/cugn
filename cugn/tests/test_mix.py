""" Tests for Spray Gliders """

import glob

import pytest

from cugn import floatdata
from cugn import gliderdata

from IPython import embed

dataset = 'ARCTERX-Leg2'
fData = floatdata.load_dataset(dataset)
gData = gliderdata.load_dataset(dataset)

assets = gData.sum(fData, merge_on_depth=True)
embed(header='End of test_floats.py')