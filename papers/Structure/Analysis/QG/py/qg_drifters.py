"""
Python wrapper for the Julia QG drifter simulation.

Calls the Julia CLI script via subprocess and returns results as pandas DataFrames.

Usage:
    from py.qg_drifters import run_drifters, load_trajectories

    traj = run_drifters(t_start=5001, n_days=30, n_per_side=16)
    print(traj.head())
"""

import subprocess
import shutil
import json
import os
from pathlib import Path

import pandas as pd

# Paths
_THIS_DIR = Path(__file__).parent
_JL_DIR = _THIS_DIR.parent / "jl"
_CLI_SCRIPT = _JL_DIR / "run_drifters_cli.jl"
_DEFAULT_OUTPUT_DIR = Path("/tmp/qg_drifters")


def _find_julia():
    """Find the Julia executable."""
    # Check PATH
    julia = shutil.which("julia")
    if julia:
        return julia
    # Check juliaup default location
    juliaup_julia = Path.home() / ".juliaup" / "bin" / "julia"
    if juliaup_julia.exists():
        return str(juliaup_julia)
    raise FileNotFoundError(
        "Julia not found. Install via: curl -fsSL https://install.julialang.org | sh"
    )


def run_drifters(
    t_start=5001,
    n_days=30,
    n_per_side=16,
    lev=1,
    record_interval=1,
    box_center_x=None,
    box_center_y=None,
    box_size_km=None,
    drifter_spacing_km=None,
    output_path=None,
    cache=True,
    verbose=True,
):
    """Run the Julia drifter simulation and return trajectories as a DataFrame.

    Parameters
    ----------
    t_start : int
        Starting time index in the NetCDF file (1-based). Default 5001 (day ~5000).
    n_days : int
        Number of days to advect.
    n_per_side : int
        Drifters per side of the deployment grid (total = n_per_side^2).
        Ignored when box_size_km is specified.
    lev : int
        Vertical level (1=upper, 2=lower).
    record_interval : int
        Record positions every N days.
    box_center_x : float, optional
        Center x of deployment box in grid units. Default: domain center.
    box_center_y : float, optional
        Center y of deployment box in grid units. Default: domain center.
    box_size_km : float, optional
        Side length of deployment box in km. If None, uses full-domain deployment.
    drifter_spacing_km : float, optional
        Spacing between drifters in km. Default: 10 km (only used with box_size_km).
    output_path : str or Path, optional
        Path for the output CSV. Defaults to /tmp/qg_drifters/<params>.csv.
    cache : bool
        If True, skip re-running if output CSV already exists.
    verbose : bool
        If True, stream Julia stdout in real-time.

    Returns
    -------
    traj : pd.DataFrame
        Trajectory data with columns: ID, x, y, t, x_m, y_m
    meta : dict
        Metadata: dx, nx, n_drifters, t_start, n_days, n_per_side, lev
    """
    # Determine output path
    if output_path is None:
        _DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if box_size_km is not None:
            bcx = box_center_x if box_center_x is not None else "c"
            bcy = box_center_y if box_center_y is not None else "c"
            dsp = drifter_spacing_km if drifter_spacing_km is not None else 10
            fname = f"traj_t{t_start}_d{n_days}_box{box_size_km}km_sp{dsp}km_cx{bcx}_cy{bcy}_l{lev}_r{record_interval}.csv"
        else:
            fname = f"traj_t{t_start}_d{n_days}_n{n_per_side}_l{lev}_r{record_interval}.csv"
        output_path = _DEFAULT_OUTPUT_DIR / fname
    output_path = Path(output_path)
    meta_path = Path(str(output_path) + ".meta.json")

    # Check cache
    if cache and output_path.exists() and meta_path.exists():
        if verbose:
            print(f"Using cached results: {output_path}")
        return load_trajectories(output_path)

    # Find Julia
    julia = _find_julia()

    # Build command
    cmd = [
        julia,
        str(_CLI_SCRIPT),
        "--t_start", str(t_start),
        "--n_days", str(n_days),
        "--n_per_side", str(n_per_side),
        "--lev", str(lev),
        "--record_interval", str(record_interval),
        "--output", str(output_path),
    ]
    if box_size_km is not None:
        cmd.extend(["--box_size_km", str(box_size_km)])
    if box_center_x is not None:
        cmd.extend(["--box_center_x", str(box_center_x)])
    if box_center_y is not None:
        cmd.extend(["--box_center_y", str(box_center_y)])
    if drifter_spacing_km is not None:
        cmd.extend(["--drifter_spacing_km", str(drifter_spacing_km)])

    if verbose:
        print(f"Running: {' '.join(cmd[:3])} ...")
        print(f"Output: {output_path}")

    # Run Julia with real-time output streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    for line in process.stdout:
        output_lines.append(line)
        if verbose:
            print(f"  [julia] {line}", end="")

    returncode = process.wait()
    if returncode != 0:
        full_output = "".join(output_lines)
        raise RuntimeError(
            f"Julia drifter simulation failed (exit code {returncode}):\n{full_output}"
        )

    return load_trajectories(output_path)


def load_trajectories(csv_path):
    """Load trajectories and metadata from a previously saved run.

    Parameters
    ----------
    csv_path : str or Path
        Path to the trajectory CSV file.

    Returns
    -------
    traj : pd.DataFrame
        Trajectory data.
    meta : dict
        Metadata dictionary.
    """
    csv_path = Path(csv_path)
    meta_path = Path(str(csv_path) + ".meta.json")

    traj = pd.read_csv(csv_path)

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return traj, meta
