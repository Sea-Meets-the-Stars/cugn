""" Routines to process the Spray CUGN data """

# imports
import os
import xarray
from glob import glob

import numpy as np
from scipy.interpolate import interp1d

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import pandas

from gsw import conversions, density
import gsw

from cugn import utils as cugn_utils
from cugn import grid_utils
from cugn import io as cugn_io
from cugn import defs as cugn_defs
from cugn import highres as cugn_highres

high_path = os.path.join(os.getenv('OS_SPRAY'), 'CUGN', 'HighRes')

from IPython import embed

def add_gsw():
    """ Add physical quantities to the Spray CUGN data
    using the TEOS-10 GSW package
    """

    # Spray files
    spray_files = glob(os.path.join(
        cugn_defs.data_path, 'CUGN_line_*.nc'))

    for spray_file in spray_files:

        print(f"Working on: {spray_file}")
        # Load the data
        ds = xarray.load_dataset(spray_file)

        # Bad times?
        bad = np.isnan(ds.time)
        if np.any(bad):
            tmp = pandas.Timestamp('1999-01-01')
            ds.time[bad] = tmp
            ds.time_uv[bad] = tmp

        lat = np.nanmedian(ds.lat.data)
        lon = np.nanmedian(ds.lon.data)

        # Prep for new variables
        CT = np.ones_like(ds.temperature.data) * np.nan
        SA = np.ones_like(ds.temperature.data) * np.nan
        OC = np.ones_like(ds.temperature.data) * np.nan
        SO = np.ones_like(ds.temperature.data) * np.nan
        AOU = np.ones_like(ds.temperature.data) * np.nan

        # Loop on depths
        for zz, z in enumerate(ds.depth.data):
            # Pressure
            p = conversions.p_from_z(-z, lat)

            # SA
            iSA = conversions.SA_from_SP(ds.salinity.data[zz,:], 
                                        p, lon, lat)
            SA[zz,:] = iSA

            # CT
            iCT = conversions.CT_from_t(iSA, ds.temperature.data[zz,:], 
                                        p)
            CT[zz,:] = iCT

            # Oxygen
            OC[zz,:] = gsw.O2sol(iSA, iCT, p, lon, lat)
            SO[zz,:] = ds.doxy[zz,:] / OC[zz,:] 
            AOU[zz,:] = ds.doxy[zz,:] - OC[zz,:] 

        # sigma0 
        sigma0 = density.sigma0(SA, CT)

        # Add to ds
        ds['CT'] = (('depth', 'profile'), CT)
        ds.CT.attrs = dict(units='Celsius', long_name='Conservative Temperature')
        ds['sigma0'] = (('depth', 'profile'), sigma0)
        ds.sigma0.attrs = dict(units='kg/m^3', long_name='potential density anomaly')
        ds['SA'] = (('depth', 'profile'), SA)
        ds.SA.attrs = dict(units='g/kg', long_name='Absolute Salinity')
        ds['SO'] = (('depth', 'profile'), SO)
        ds.SO.attrs = dict(long_name='Oxygen Saturation')
        ds['AOU'] = (('depth', 'profile'), AOU)
        ds.AOU.attrs = dict(long_name='Apparent Oxygen Utilization')

        # Buoyancy
        dsigmadz, _ = np.gradient(ds.sigma0.data, 
                                  float(ds.depth[1]-ds.depth[0]))
        dsigmadz[dsigmadz < 0.] = 0.
        buoyfreq = np.sqrt(9.8/1025*dsigmadz)/(2*np.pi)*3600
        ds['N'] = (('depth', 'profile'), buoyfreq)
        ds.N.attrs = dict(long_name='Buoyancy Frequency', units='cycles/hour')

        # MLD
        MLDs = []
        for iprofile in ds.profile.data:
            f = interp1d(ds.sigma0.data[:,iprofile], ds.depth.data-5.)
            MLD = f(ds.sigma0.data[0,iprofile]+cugn_defs.MLD_sigma0)
            MLDs.append(MLD)
        ds['MLD'] = (('profile'), MLDs)

        # Wipe out the 2020 trip to San Diego
        dskeys =  ['temperature', 'salinity', 'chlorophyll_a', 'u', 'v', 
                   'acoustic_backscatter', 'doxy', 'CT', 'sigma0', 'SA', 'SO', 'N', 'AOU']
        if 'line_80' in spray_file:
            dist, _ = cugn_utils.calc_dist_offset(
                '80.0', ds.lon.values, ds.lat.values)
            bad = dist < -50.
            # Zero em!!
            for key in dskeys:
                ds[key].data[:,bad] = np.nan

        # Mexican trip on Line 90
        if 'line_90' in spray_file:
            bad = ds.mission == 63
            for key in dskeys:
                ds[key].data[:,bad] = np.nan

        # Write
        new_spray_file = spray_file.replace('CUGN_', 'CUGN_potential_')
        ds.to_netcdf(new_spray_file)
        print(f"Wrote: {new_spray_file}")

def build_ds_grid(items,
                #line:str, line_file:str, gridtbl_outfile:str, edges_outfile:str, 
                  min_counts:int=50, 
                  debug:bool=False,
                  max_offset:float=90.,
                  warn_highres:bool=False):
    """ Grid up density and salinity for a line
    to generate a table of grid indices and values

    Args:
        items (tuple): Tuple of line, line_file, gridtbl_outfile, edges_outfile
            line (str): Line name ['90.0', '67.0']
            line_file (str): line file
            gridtbl_outfile (str): name of the output table
            edges_outfile (str): name of the output grid edges
        min_counts (int, optional): Minimum counts on the
            grid to be included in the analysis. Defaults to 50.
        debug (bool, optional): Debug. Defaults to False.
        max_offset (float, optional): Maximum offset from the line. Defaults to 90 km
        warn_highres (bool, optional): Warn if high res data is missing. Defaults to False.
            Otherwise crash out
    """
    line, line_file, gridtbl_outfile, edges_outfile = items
    # Dataset
    ds = xarray.load_dataset(line_file)

    # Cut on offset
    dist, offset = cugn_utils.calc_dist_offset(
            line, ds.lon.values, ds.lat.values)
    ok_off = (np.abs(offset) < max_offset) & np.isfinite(offset)
    ds = ds.isel(profile=ok_off)

    # Generate the grid
    mean_oxyT, SA_edges, sigma_edges, countsT, \
        grid_indices, gd_oxy, da_gd = grid_utils.gen_grid(ds, stat='mean')

    # Table me
    gd_depth, gd_profile = np.where(da_gd)
    grid_tbl = pandas.DataFrame()

    grid_tbl['depth'] = gd_depth
    grid_tbl['profile'] = ds.profile[gd_profile].values
    grid_tbl['row'] = grid_indices[0,:] - 1
    grid_tbl['col'] = grid_indices[1,:] - 1
    grid_tbl['doxy'] = gd_oxy

    # Trajectory, mission etc.
    tidx = ds.trajectory_index[gd_profile].values
    grid_tbl['mission'] = ds.mission_name[tidx].values.astype(str)
    grid_tbl['mission_profile'] = ds.mission_profile[gd_profile].values

    # Cut on counts
    gd_rows, gd_cols = np.where(countsT > min_counts)

    # Keep only the good ones
    keep = np.zeros(len(grid_tbl), dtype=bool)
    for gd_row, gd_col in zip(gd_rows, gd_cols):
        keep |= (grid_tbl.row == gd_row) & (grid_tbl.col == gd_col)

    grid_tbl = grid_tbl[keep].copy()
    grid_tbl.reset_index(inplace=True, drop=True)

    # Prep
    grid_tbl['MLD'] = np.nan
    grid_tbl['N'] = np.nan
    grid_tbl['zNpeak'] = np.nan
    grid_tbl['sigma0_0'] = np.nan

    # MLD and N at high resolution
    uni_missions = np.unique(grid_tbl.mission)
    for mission in uni_missions:
        gfiles = glob(os.path.join(high_path, f'SPRAY-FRSQ-{mission}-*.nc'))
        if len(gfiles) != 1:
            if warn_highres:
                print(f"Missing high res data for {mission}")
                continue
            else:
                raise ValueError(f"Missing high res data for {mission}")
        # Run
        gm_idx = grid_tbl.mission.values == mission
        mprofiles = np.unique(grid_tbl.mission_profile[gm_idx])
        iz = grid_tbl[gm_idx].depth.max()

        # Do it
        MLDs, Ns, zNs, zN5s, zN10s, Nf5s, Nf10s, NSOs, sigma0_0s = \
            cugn_highres.calc_mission(
            gfiles[0], mprofiles, max_depth=(iz+1)*10, debug=debug)
        # Fill in (this is slow)
        for mprofile, MLD, N, zN, zN5, zN10, Nf5, Nf10, NSO, sigma0_0 in zip(
            mprofiles, MLDs, Ns, zNs, zN5s, zN10s, Nf5s, Nf10s, NSOs, sigma0_0s):
            in_mission = (grid_tbl.mission == mission) & (
                grid_tbl.mission_profile == mprofile)
            # MLD
            grid_tbl.loc[in_mission, 'MLD'] = MLD
            grid_tbl.loc[in_mission, 'sigma0_0'] = sigma0_0
            grid_tbl.loc[in_mission, 'zNpeak'] = zN
            grid_tbl.loc[in_mission, 'zN5'] = zN5
            grid_tbl.loc[in_mission, 'zN10'] = zN10
            # Counting SO extrema
            grid_tbl.loc[in_mission, 'Nf5'] = Nf5
            grid_tbl.loc[in_mission, 'Nf10'] = Nf10
            grid_tbl.loc[in_mission, 'NSO'] = NSO
            # N
            these_N = np.array([N[ii] for ii in grid_tbl.depth.values[in_mission]])
            grid_tbl.loc[in_mission, 'N'] = these_N
            #embed(header='build_ds_grid: 230')


    if debug:
        grid_utils.fill_in_grid(grid_tbl, ds)
        tmin = pandas.Timestamp('2020-08-22')
        tmax = pandas.Timestamp('2020-09-11')
        ttimes = pandas.to_datetime(grid_tbl.time.values)
        in_t = (ttimes >= tmin) & (ttimes <= tmax) & (grid_tbl.depth <= 1)
        embed(header='build_ds_grid: 150')

    # Generate DO percentile
    grid_utils.find_perc(grid_tbl)

    # Test
    #in_cell = (grid_tbl.row == 41) & (grid_tbl.col == 45)
    #vals = grid_tbl.doxy.values[in_cell]
    #pct = np.nanpercentile(vals, 5)

    #embed(header='234 of process')
    # Save
    if not debug:
        grid_tbl.to_parquet(gridtbl_outfile)
        if edges_outfile is not None:
            np.savez(edges_outfile, SA_edges=SA_edges, 
                sigma_edges=sigma_edges,
                counts=countsT)
        print(f"Wrote: \n {gridtbl_outfile} \n {edges_outfile}")


def main(flg, debug=False):

    # Add potential density and salinity to the CUGN files
    if flg == 1:
        add_gsw()
        return

    # Grids
    if flg in [2,3]:
        warn_highres = True
        items = []
        for line in cugn_defs.lines:
            #if line != '80.0':
            #    continue
            #if line != '90.0':
            #    continue
            line_files = cugn_io.line_files(line)

            if flg == 2: # Full
                items.append((line, line_files['datafile'],
                    line_files['gridtbl_file_full'], 
                    None))
                min_counts=0
            if flg == 3: # Control
                items.append((line, line_files['datafile'],
                    line_files['gridtbl_file_control'], 
                    line_files['edges_file']))
                min_counts=50

        #build_ds_grid(line, line_files['datafile'],
        #        line_files['gridtbl_file_full'], 
        #        None,
        #        max_offset=90., warn_highres=warn_highres)

        if not debug:
            n_cores = 4
            map_fn = partial(build_ds_grid, min_counts=min_counts, warn_highres=warn_highres)
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
                answers = list(tqdm(executor.map(map_fn, items,
                                                chunksize=chunksize), total=len(items)))
        else:                                        
            build_ds_grid(items[0], min_counts=min_counts, warn_highres=warn_highres,
                          debug=True)

    # Check all full res files
    if flg == 4:
        for line in cugn_defs.lines:
            print(f"Working on line: {line}")
            line_files = cugn_io.line_files(line)
            ds = xarray.load_dataset(line_files['datafile'])

            # Missing highres
            mission_names  = np.unique(ds.mission_name.values.astype(str))
            for mission_name in mission_names:
                #print(f"Checking for mission: {mission_name}")
                gfiles = glob(os.path.join(high_path, f'SPRAY-FRSQ-{mission_name}-*.nc'))
                if len(gfiles) != 1:
                    print(f"Missing high res data for {mission_name} of line {line}")

if __name__ == '__main__':
    import sys
    flg = int(sys.argv[1])

    main(flg, debug=False)