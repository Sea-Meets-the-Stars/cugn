# Figures for the paper

# Code

Follow these guidelines:
- Use the same style and code flow as the other methods in figs_structure.py
- Use the same data as the other methods in figs_structure.py
- Use the same plotting style as the other methods in figs_structure.py
- Use inline comments to explain the code.
- Code in Python
- Place imports at the top of modules

# QG 5year in 100km region vs. full box

Add a new method named fig_qg_100km_vs_full() to the fig_structure.py module that plots the structure function of the QG 5year in a 100km region vs. the structure function of the QG 5year in the full box.

Examine methods in figs_structure.py and mimic their style and code flow.

## Data

For the full box, follow the code in fig_full_qg_SF().

For the 100km region, use the data in the file ../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_5years.nc with x0,y0 = 400, 400 as a default

## Modifications

1. Make the following changes

- Show the 100km region as red dots, not a dashed line
- Put all Legends in the upper left
- Use black for the full region
- Scale duL^3 by 1e-3 

2. Make the following changes

- Use the calc_dus() method in Analysis/py/qg_utils.py to perform the calculations for the 100km region
- Show the corrected values for the 100km region as open red circles.

3. Make the following changes:

- Use the qg_uL_SF.parse_SF() method in Analysis/py/qg_utils.py to perform the calculations for the 100km region

4. Make the following changes:

- Do not show the 100km region values for r>100km

# Prompts

1. Read this file.  Generate the QG 5year in 100km region vs. full box figure.
2. Re-read this file. Make the first set of modifications on the 100km region vs. full figure.
3. Re-read this file. Make the 2nd set of modifications on the 100km region vs. full figure.
4. Re-read this file. Make the 3rd set of modifications on the 100km region vs. full figure.
5. Re-read this file. Make the 4th set of modifications on the 100km region vs. full figure.