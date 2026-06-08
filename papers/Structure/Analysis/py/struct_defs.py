""" Definitions for the structure function paper. """

# Plot items
dataset_clrs = {'Calypso2019': 'blue', 
                'Calypso2022': 'red', 
                'ARCTERX-2023': 'green'}

# Binning
dataset_binning = {
    'Calypso2019': {'log': 40, 'lin': 80},
    'Calypso2022': {'log': 40, 'lin': 80},
    'ARCTERX-2023': {'log': 40, 'lin': 80},
    }

# Analysis
iz = 5
minN = 50
max_time = 10.
btype = 'log'
avoid_same_glider = True