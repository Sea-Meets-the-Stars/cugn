""" Common definitions for CUGN project """
import os

data_path = os.path.join(os.getenv('OS_SPRAY'), 'CUGN')

lines =  ['56.0', '66.7', '80.0', '90.0']
line_colors = ['r','b','g','orange']
line_cmaps = ['Reds','Blues','Greens','Oranges']

# Extrema
SO_hyper = 1.1

AOU_hyper = 25.  # mu mol / kg

# MLD
MLD_sigma0 = 0.125  # kg/m^3