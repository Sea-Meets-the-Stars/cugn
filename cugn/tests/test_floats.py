""" Tests for Spray Gliders """

import glob

import pytest

from cugn import floatdata

from IPython import embed

dataset = 'ARCTERX-Leg2'
s8996 = floatdata.load_dataset(dataset)

embed(header='End of test_floats.py')