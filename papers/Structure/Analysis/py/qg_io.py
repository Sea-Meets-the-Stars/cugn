import json
from pathlib import Path

import pandas as pd


def load_trajectories(csv_path):
    """Load drifter trajectories and metadata from CSV + JSON sidecar.

    Parameters
    ----------
    csv_path : str or Path
        Path to the trajectory CSV file.

    Returns
    -------
    traj : pd.DataFrame
        Trajectory data with columns: ID, x, y, t, x_m, y_m
    meta : dict
        Metadata dictionary (dx, nx, n_drifters, t_start, etc.).
    """
    csv_path = Path(csv_path)
    meta_path = Path(str(csv_path) + ".meta.json")

    traj = pd.read_csv(csv_path)

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return traj, meta

def qg_output_file(x0, y0, dx:int=None, dtime='5years'):    
    
    # Build it
    if dx == 200:
        output_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_200km_{dtime}.nc' 
    elif dx == 300:
        output_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_300km_{dtime}.nc' 
    else:
        output_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_{dtime}.nc' 
    
    # Return output_file
    return output_file