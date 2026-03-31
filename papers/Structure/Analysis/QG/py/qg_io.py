"""
I/O utilities for QG drifter trajectory data.
"""

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
