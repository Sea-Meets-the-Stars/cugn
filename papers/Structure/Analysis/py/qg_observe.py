# Observe the QG model

import os
import sys

import qg_100km

sys.path.append(os.path.abspath("../Analysis/QG/py"))
import glider_analysis
import small_box_drifters
import calc_sf


from IPython import embed

def qg_root(qg_xt:dict):
    """Build a filename root string from QG experiment parameters.

    Args:
        qg_xt: Dict with keys 'ts' (start time), 'nd' (number of days),
            'x', 'y' (box origin), 'dx' (box size).

    Returns:
        str: Formatted root string, e.g. 'ts5001_nd100_x450_y450_dx100'.
    """
    root = f'ts{qg_xt["ts"]}_nd{qg_xt["nd"]}_x{qg_xt["x"]}_y{qg_xt["y"]}_dx{qg_xt["dx"]}'
    return root

def file_names(qg_xt:dict, obs_type:str):
    """Generate output file paths for trajectory and structure function files.

    Args:
        qg_xt: Dict of QG experiment parameters (passed to qg_root).
        obs_type: Observation type string, e.g. 'glider' or 'drifter'.

    Returns:
        tuple: (traj_file, sf_file) paths under the Output/ directory.
    """
    root = qg_root(qg_xt)

    traj_file = f'Output/qg_{obs_type}_traj_{root}.csv'
    sf_file = f'Output/qg_{obs_type}_SF_{root}.json'

    return traj_file, sf_file

def run_one(qg_xt:dict, clobber:bool=False):
    """Run glider, drifter, and Eulerian analyses for one QG sub-domain.

    Computes trajectories and structure functions for the specified
    space-time region of the QG model.

    Args:
        qg_xt: Dict with keys 'ts' (start time step), 'nd' (number of days),
            'x', 'y' (box origin in km), 'dx' (box width in km).
        clobber: If True, overwrite existing files.
    """
    dx_grid = 1000./256

    ## Gliders
    if True:
        glider_csv = 'QG/data/100km100day10gliders3h.csv'
        glider_traj, glider_sf = file_names(qg_xt, 'glider')
        glider_df, meta = glider_analysis.run_single(glider_csv, t_start=qg_xt['ts'], lev=1,
            offset_x=qg_xt['x']/dx_grid, 
            offset_y=qg_xt['y']/dx_grid,
            output_path=glider_traj, cache=not clobber)
        Sn_LL, Sn_TT = glider_analysis.compute_glider_sf(glider_df, meta)
        glider_analysis.save_sf(Sn_LL, glider_sf)

    ## Drifters
    if True:
        drifter_traj, drifter_sf = file_names(qg_xt, 'drifter')
        traj, meta = small_box_drifters.run_small_box(
            t_start=qg_xt['ts'], 
            n_days=qg_xt['nd'],
            box_center_x=(qg_xt['x']+0.5*qg_xt['dx'])/dx_grid,
            box_center_y=(qg_xt['x']+0.5*qg_xt['dx'])/dx_grid,
            box_size_km=qg_xt['dx'],
            drifter_spacing_km=10.,
            cache=not clobber,
            output_path=drifter_traj)
        Sn_LL, Sn_TT = calc_sf.calc_drifter_sf(drifter_traj, drifter_sf)
    
    ## Eulerian
    root = qg_root(qg_xt)
    eulerian_file = f'Output/qg_eulerian_SF_{root}.nc'
    qg_100km.run_one_region((qg_xt['x'], qg_xt['x']+qg_xt['dx']), 
        (qg_xt['y'], qg_xt['y']+qg_xt['dx']),
            eulerian_file,
            timelast=7200-qg_xt['ts']-2,
            ndays=qg_xt['nd'], maxcorr=30, clobber=clobber)

# Command line
if __name__ == '__main__':
    # ts = 5001
    qg_xt_5001 = {
        'ts': 5001,
        'nd': 100,
        'x': 450,
        'y': 450,
        'dx': 100,
    }
    # ts = 6001
    qg_xt_6001 = {
        'ts': 6001,
        'nd': 100,
        'x': 450,
        'y': 450,
        'dx': 100,
    }

    # Go
    run_one(qg_xt_5001, clobber=True)