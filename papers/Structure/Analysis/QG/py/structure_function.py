"""
Structure function analysis for QG drifter trajectories.

Two independent approaches:
  1. analysis.py  — standalone Lagrangian SF computation (with periodic BCs).
  2. profiler/DrifterData + ProfilerPairs — reuses the profiler framework.

This module provides wrapper functions for each approach and a comparison
function that runs both.
"""

import numpy as np

from profiler.drifterdata import DrifterData
from profiler.profilerpairs import ProfilerPairs

# Approach 1 import (local module in the same QG/py/ directory)
from analysis import compute_structure_functions


# Default separation bins — 20 log-spaced edges from 5 km to 500 km (meters)
_DEFAULT_RBINS_M = np.logspace(np.log10(5e3), np.log10(500e3), 21)

# Same bins in km for ProfilerPairs (which works in km)
_DEFAULT_RBINS_KM = _DEFAULT_RBINS_M / 1e3


def run_analysis_approach(traj, meta, r_bins=None):
    """Approach 1: compute structure functions using analysis.py.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory data (ID, x, y, t, x_m, y_m).
    meta : dict
        Metadata dict (dx, nx, …).
    r_bins : array-like, optional
        Bin edges in meters.  Default: 20 log-spaced bins, 5–500 km.

    Returns
    -------
    sf : pd.DataFrame
        Columns: r_mid (m), D_LL, D_TT, D_2, n_pairs.
    """
    if r_bins is None:
        r_bins = _DEFAULT_RBINS_M
    # Delegate to the existing analysis module
    sf = compute_structure_functions(traj, meta, r_bins=r_bins)
    return sf


def run_profiler_approach(traj, meta, r_bins_km=None, max_time=1.0):
    """Approach 2: compute structure functions using DrifterData + ProfilerPairs.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory data.
    meta : dict
        Metadata dict.
    r_bins_km : array-like, optional
        Bin edges in km.  Default: 20 log-spaced bins, 5–500 km.
    max_time : float, optional
        Maximum time difference (hours) for pairing.  Default 1.0.

    Returns
    -------
    Sn_LL : dict
        Structure function dict for longitudinal component (duL).
    Sn_TT : dict
        Structure function dict for transverse component (duT).
    """
    if r_bins_km is None:
        r_bins_km = _DEFAULT_RBINS_KM

    # Build DrifterData objects — one per drifter
    drifters = DrifterData.all_from_QG_trajectory(traj, meta)

    # Construct ProfilerPairs — pairs drifters at the same time step
    pairs = ProfilerPairs(
        drifters,
        max_time=max_time,
        avoid_same_glider=True,
        randomize=False,  # deterministic pairing for reproducibility
    )

    # --- Longitudinal structure function (D_LL = S2 of duL) ---
    pairs.calc_delta(iz=0, variables='duLduLduL')
    pairs.calc_Sn(variables='duLduLduL')
    Sn_LL = pairs.calc_Sn_vs_r(r_bins_km)
    pairs.add_meta(Sn_LL)

    # --- Transverse structure function (D_TT = S2 of duT) ---
    pairs.calc_Sn(variables='duTduTduT')
    Sn_TT = pairs.calc_Sn_vs_r(r_bins_km)
    pairs.add_meta(Sn_TT)

    return Sn_LL, Sn_TT


def compare_approaches(traj, meta):
    """Run both approaches and return their results.

    Returns
    -------
    sf_analysis : pd.DataFrame
        Approach 1 results (D_LL, D_TT, D_2 vs r_mid in meters).
    Sn_LL : dict
        Approach 2 longitudinal structure function dict (r in km).
    Sn_TT : dict
        Approach 2 transverse structure function dict (r in km).
    """
    sf_analysis = run_analysis_approach(traj, meta)
    Sn_LL, Sn_TT = run_profiler_approach(traj, meta)
    return sf_analysis, Sn_LL, Sn_TT
