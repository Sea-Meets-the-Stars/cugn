""" Generate a Merged SST product for ARCTERX2 """

import xarray
import argparse

from remote_sensing.download import podaac
from remote_sensing.healpix import rs_healpix
from remote_sensing.healpix import plotting as hp_plotting
from remote_sensing.healpix import utils as hp_utils
from remote_sensing.healpix import combine as hp_combine
from remote_sensing import io as rs_io
from remote_sensing import units
from remote_sensing import kml as rs_kml

from IPython import embed

# Globals
lon_lim = (127.,134)
lat_lim = (18.,23)

def main(args):

    # Grab the latest data
    amsr2_files, _ = podaac.grab_file_list(
        'AMSR2-REMSS-L2P_RT-v8.2', dt_past=dict(days=2),
        bbox='127,18,134,23')

    h09_files, _ = podaac.grab_file_list(
        'H09-AHI-L3C-ACSPO-v2.90', dt_past=dict(days=1),
        bbox='127,18,134,23')

    # Download
    print(f"Downloading AMSR2 files")
    local_amsr2 = podaac.download_files(amsr2_files, verbose=False)
    print(f"Downloading Himawari files")
    local_h09 = podaac.download_files(h09_files, verbose=False)
    print(f"All done")

    # Use the latest H09 file for the timestamp
    ds = xarray.open_dataset(local_h09[0])
    time_root = str(ds.time.data[0]).replace(':','')[0:13]

    # Save files to a JSON file
    json_file = f'Merged_SST_{time_root}.json'
    sdict = {}
    sdict['local_amsr2'] = local_amsr2
    sdict['local_h09'] = local_h09
    # 
    jdict = rs_io.jsonify(sdict)
    rs_io.savejson(json_file, jdict, overwrite=True)
    print(f"Wrote: {json_file}")

    # Healpix

    # #############################
    amsr2_hpxs = []
    for data_file in sdict['local_amsr2'][0:args.namsr2]:
        # Objectify
        rs_hpx = rs_healpix.RS_Healpix.from_dataset_file(
            data_file, 'sea_surface_temperature',
            time_isel=0, resol_km=11., 
            lat_slice=(18,23.),  lon_slice=(127., 134.))
        # 
        print(f"Generated RS_Healpix from {data_file}")
        # Add
        amsr2_hpxs.append(rs_hpx)

    # Combine?
    if args.namsr2 > 1:
        amsr2_stack = rs_healpix.RS_Healpix.from_list(amsr2_hpxs)
    else:
        amsr2_stack = amsr2_hpxs[0]

    if args.show:
        amsr2_stack.plot(figsize=(10.,6), cmap='jet', 
                           xlim=lon_lim, ylim=lat_lim, 
                           projection='platecarree', ssize=40., 
                           vmin=23.7, vmax=27., show=True)


    # #############################
    h09_hpxs = []
    for data_file in sdict['local_h09'][0:args.nh09]:
        # Objectify
        rs_hpx = rs_healpix.RS_Healpix.from_dataset_file(
            data_file, 'sea_surface_temperature',
            lat_slice=slice(23,18),  lon_slice=slice(127., 134.), time_isel=0)
        # 
        print(f"Generated RS_Healpix from {data_file}")
        # Add
        h09_hpxs.append(rs_hpx)
        del(rs_hpx)
    # Stack
    h09_stack = rs_healpix.RS_Healpix.from_list(h09_hpxs)
    if args.show:
        h09_stack.plot(figsize=(10.,6), cmap='jet', 
                       xlim=lon_lim, ylim=lat_lim, 
                       projection='platecarree',
                       show=True)

    # Fill in
    h09_stack.fill_in(amsr2_hpxs[0], (lon_lim[0], lon_lim[1], 
                                      lat_lim[0], lat_lim[1]))

    # #############################33
    # KMZ
    ax, img = h09_stack.plot(figsize=(10.,6), cmap='jet', 
                             xlim=lon_lim, ylim=lat_lim, 
                             add_colorbar=False, 
                             projection='platecarree', vmin=20., 
                             savefig='kml_test.png', dpi=1200, 
                             marker=',')
    rs_kml.colorbar(img, 'SST (C)', 'colorbar.png')

    # Write
    outfile = f'Merged_SST_{time_root}.kmz'
    rs_kml.make_kml(llcrnrlon=lon_lim[0], llcrnrlat=lat_lim[0],
        urcrnrlon=lon_lim[1], urcrnrlat=lat_lim[1],
        figs=['kml_test.png'], colorbar='colorbar.png',
        kmzfile=outfile, name='Merged SST')


def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("Merged SST KMZ script")
    parser.add_argument("--namsr2", type=int, 
                        default=1, help="Number of exposures of AMSR2 to combine")
    parser.add_argument("--nh09", type=int, 
                        default=10, help="Number of hours of Himawari images to combine")
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug?')
    parser.add_argument('--show', default=False, action='store_true',
                        help='show extra plots?')
    parser.add_argument('--clobber', default=False, action='store_true',
                        help='Clobber existing files')
    parser.add_argument('--redo', default=False, action='store_true',
                        help='Redo?')
    parser.add_argument("--outfile", type=str, 
                        help="Path to output file")
    parser.add_argument("--umap_file", type=str, 
                        help="Path to UMAP pickle file for analysis")
    parser.add_argument("--table_file", type=str, 
                        help="Path to Table file")
    parser.add_argument("--ncpu", type=int, help="Number of CPUs")
    parser.add_argument("--years", type=str, help="Years to analyze")

    args = parser.parse_args()
    
    return args

        
if __name__ == "__main__":
    # get the argument of training.
    args = parse_option()
    main(args)
    