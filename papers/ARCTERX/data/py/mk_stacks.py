""" Module for generating stakcs of SST images for ARCTERX """

import os
import glob

import numpy as np

import xarray


def stack_me(files:list, outfile:str):
    """ Stack the files in the list and save to outfile

    Args:
        files (list): List of files to stack
        outfile (str): Output file
    """

    # Load the images
    imgs = []
    for ifile in files:
        ds = xarray.open_dataset(ifile)
        imgs.append(ds.sea_surface_temperature.data)
    imgs = np.concatenate(imgs)

    # Stack
    sst_stack = np.nanmedian(imgs, axis=0)

    # Save
    ds_stack = ds.sea_surface_temperature.copy()
    ds_stack.data = sst_stack.reshape((1,297,446))

    ds_stack.to_netcdf(outfile)
    print(f'Wrote: {outfile}')


def main(flg:int):

    # Daily's
    if flg == 1:
        for day in [26, 27, 28, 29]:
            # Grab the files
            files = glob.glob(os.path.join(
                os.getenv('ARCTERX'), 'SST', f'stcc_iop25_202501{day:02}*.nc'))
            # Stack me
            outfile = f'SST_stack_2025jan{day:02}.nc'
            stack_me(files, outfile)

# Command line execution
if __name__ == '__main__':
    import sys

    flg = int(sys.argv[1])
    main(flg)