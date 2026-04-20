# Debugging analysis

# calc_SF_5years

## First step

When running `calc_SF_5years()` in `qg_uL_SF.py`, the following error occurs:

ValueError: conflicting sizes for dimension 'mid_rbins': length 59 on 'mid_rbins' and length 55 on {'time': 'dr', 'mid_rbins': 'dr'}

Can you help me debug this?

The files to be analyzed are located in /home/xavier/Oceanography/data/QG/SF_spatialav_duL/

If you need to run python, use the conda environment `os_313`.

# Running regions

When running `run_one_region()` in `qg_100km.py`,  I run out of memory when running on a 200km region for 5 years using maxcorr=50.  Do I have any other option than to reduce the maxcorr?



# Prompts

1. Read this file.  Work on the First step for debugging calc_SF_5years() as described above.

2. Read this file.  Work on the issue in Running regions as described above.