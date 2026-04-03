# Observe the QG model

import os
import sys

sys.path.append(os.path.abspath("../Analysis/QG/py"))
import glider_analysis
import small_box_drifters
import calc_sf

from IPython import embed

def qg_root(qg_xt:dict):
    root = f'ts{qg_xt["ts"]}_nd{qg_xt["nd"]}_x{qg_xt["x"]}_y{qg_xt["y"]}_dx{qg_xt["dx"]}'
    return root

def file_names(qg_xt:dict, obs_type:str):
    root = qg_root(qg_xt)

    traj_file = f'Output/qg_{obs_type}_traj_{root}.csv'
    sf_file = f'Output/qg_{obs_type}_SF_{root}.json'

    return traj_file, sf_file

def run_one(qg_xt:dict):

    dx_grid = 1000./256

    if False:
    ## Gliders
        glider_csv = 'QG/data/100km100day10gliders3h.csv'
        glider_traj, glider_sf = file_names(qg_xt, 'glider')
        glider_df, meta = glider_analysis.run_single(glider_csv, t_start=qg_xt['ts'], lev=1,
            offset_x=qg_xt['x']/dx_grid, 
            offset_y=qg_xt['y']/dx_grid,
            output_path=glider_traj)
        Sn_LL, Sn_TT = glider_analysis.compute_glider_sf(glider_df, meta)
        glider_analysis.save_sf(Sn_LL, glider_sf)

    ## Drifters
    drifter_traj, drifter_sf = file_names(qg_xt, 'drifter')
    traj, meta = small_box_drifters.run_small_box(
        t_start=qg_xt['ts'], 
        n_days=qg_xt['nd'],
        box_center_x=(qg_xt['x']+0.5*qg_xt['dx'])/dx_grid,
        box_center_y=(qg_xt['x']+0.5*qg_xt['dx'])/dx_grid,
        box_size_km=qg_xt['dx'],
        drifter_spacing_km=10.,
        output_path=drifter_traj)
    Sn_LL, Sn_TT = calc_sf.calc_drifter_sf(drifter_traj, drifter_sf)

# Command line
if __name__ == '__main__':
    qg_xt = {
        'ts': 5001,
        'nd': 100,
        'x': 450,
        'y': 450,
        'dx': 100,
    }
    run_one(qg_xt)