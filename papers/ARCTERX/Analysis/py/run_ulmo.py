""" Run Ulmo on images """

import os

import numpy as np

import xarray

from ulmo.analysis import evaluate
from ulmo.models import io as model_io
from ulmo import io as ulmo_io

from cugn import utils as cugn_utils

pdict = dict(inpaint=True, median=False, downscale=False)

def run_ulmo(img_file:str, lat_slice:slice, lon_slice:slice): 


    # Load VIIRS
    pae = model_io.load_ulmo_model('viirs-98', local=True)

    # Load image
    ds = xarray.load_dataarray(img_file)

    # Slice
    sst = ds.sel(lat=lat_slice, lon=lon_slice).data

    # Mask
    mask = np.zeros_like(sst, dtype=bool)
    if np.sum(np.isnan(sst)) > 0:
        mask[np.isnan(sst)] = True

    # Run
    latents, LL, meta = evaluate.eval_raw_sst(
        pae, sst, mask, pdict=pdict)

    # Return
    return LL, meta

def main(flg:int):

    # Daily's at 
    if flg == 1:
        out_dict = {}
        lat_slice=slice(21.28, 20.) 
        lon_slice=slice(130., 131.28)
        for day in [26, 27, 28, 29]:
            # Data file
            img_file = f'../data/SST_stack_2025jan{day:02}.nc'
            LL, meta = run_ulmo(img_file, lat_slice, lon_slice)
            # Save
            day = f'2025jan{day:02}'
            out_dict[day] = {}
            out_dict[day]['LL'] = LL
            out_dict[day]['meta'] = meta
        # Save
        outfile = 'ulmo_daily_20.0_130.0.json'
        jdict = cugn_utils.jsonify(out_dict)
        cugn_utils.savejson(outfile, jdict, easy_to_read=True, overwrite=True)

        # Return
        print(f"Wrote: {outfile}")

# Command line execution
if __name__ == '__main__':
    import sys

    flg = int(sys.argv[1])
    main(flg)