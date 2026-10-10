"""
Python wrapper for sampling QG velocity at glider positions.

Calls the Julia CLI script (run_gliders_cli.jl) via subprocess
and returns results as pandas DataFrames.

Usage:
    from py.qg_gliders import run_gliders, load_glider_velocities

    df, meta = run_gliders("data/100km100day10gliders3h.csv", t_start=5001)
    print(df.head())
"""

import subprocess
import shutil
import json
from pathlib import Path

import pandas as pd

# Paths
_THIS_DIR = Path(__file__).parent
_JL_DIR = _THIS_DIR.parent / "jl"
_CLI_SCRIPT = _JL_DIR / "run_gliders_cli.jl"
_DEFAULT_OUTPUT_DIR = Path("/tmp/qg_gliders")


def _find_julia():
    """Find the Julia executable."""
    julia = shutil.which("julia")
    if julia:
        return julia
    juliaup_julia = Path.home() / ".juliaup" / "bin" / "julia"
    if juliaup_julia.exists():
        return str(juliaup_julia)
    raise FileNotFoundError(
        "Julia not found. Install via: curl -fsSL https://install.julialang.org | sh"
    )


def run_gliders(
    glider_csv,
    t_start=5001,
    lev=1,
    offset_x=0.0,
    offset_y=0.0,
    coords_km=True,
    output_path=None,
    cache=True,
    verbose=True,
):
    """Sample QG velocity at glider positions by calling the Julia CLI.

    Parameters
    ----------
    glider_csv : str or Path
        Path to the input glider trajectory CSV (columns: x, y, time, missid).
    t_start : int
        QG time index at which glider time=0 begins (1-based). Default 5001.
    lev : int
        Vertical level (1=upper, 2=lower).
    offset_x : float
        Translation of glider positions in grid units (x-direction).
    offset_y : float
        Translation of glider positions in grid units (y-direction).
    coords_km : bool
        If True (default), input CSV x,y are in km and will be converted
        to grid-index units. If False, x,y are already in grid units.
    output_path : str or Path, optional
        Path for the output CSV. Defaults to /tmp/qg_gliders/<params>.csv.
    cache : bool
        If True, skip re-running if output CSV already exists.
    verbose : bool
        If True, stream Julia stdout in real-time.

    Returns
    -------
    df : pd.DataFrame
        Velocity data with columns: x, y, time, missid, x_m, y_m, u_qg, v_qg
    meta : dict
        Metadata: dx, nx, t_start, lev, n_gliders, offset_x, offset_y, glider_csv
    """
    glider_csv = Path(glider_csv).resolve()

    # Auto-generate output path from parameters
    if output_path is None:
        _DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        km_tag = "km" if coords_km else "gu"
        fname = (
            f"glider_vel_t{t_start}_l{lev}"
            f"_ox{offset_x}_oy{offset_y}"
            f"_{km_tag}_{glider_csv.stem}.csv"
        )
        output_path = _DEFAULT_OUTPUT_DIR / fname
    output_path = Path(output_path)
    meta_path = Path(str(output_path) + ".meta.json")

    # Check cache
    if cache and output_path.exists() and meta_path.exists():
        if verbose:
            print(f"Using cached results: {output_path}")
        return load_glider_velocities(output_path)

    # Find Julia
    julia = _find_julia()

    # Build command
    cmd = [
        julia,
        str(_CLI_SCRIPT),
        "--glider_csv", str(glider_csv),
        "--t_start", str(t_start),
        "--lev", str(lev),
        "--offset_x", str(offset_x),
        "--offset_y", str(offset_y),
        "--coords_km", str(coords_km).lower(),
        "--output", str(output_path),
    ]

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
            f"Julia glider velocity sampling failed (exit code {returncode}):\n{full_output}"
        )

    return load_glider_velocities(output_path)


def load_glider_velocities(csv_path):
    """Load glider velocity data and metadata from a previously saved run.

    Parameters
    ----------
    csv_path : str or Path
        Path to the output CSV file.

    Returns
    -------
    df : pd.DataFrame
        Velocity data with columns: x, y, time, missid, x_m, y_m, u_qg, v_qg
    meta : dict
        Metadata dictionary.
    """
    csv_path = Path(csv_path)
    meta_path = Path(str(csv_path) + ".meta.json")

    df = pd.read_csv(csv_path)

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return df, meta
