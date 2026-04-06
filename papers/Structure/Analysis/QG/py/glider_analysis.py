"""
Glider analysis for the QG model.

Samples QG velocity at glider positions and computes structure functions
using the profiler package (SprayData + ProfilerPairs).

Usage:
    from glider_analysis import run_and_analyze

    glider_df, meta, Sn_LL, Sn_TT = run_and_analyze(
        "data/100km100day10gliders3h.csv", t_start=5001)
"""

import numpy as np

from qg_gliders import run_gliders, load_glider_velocities
from profiler.gliderdata import SprayData
from profiler.profilerpairs import ProfilerPairs
from profiler import io as p_io

from IPython import embed

# Default radial bins matching real Spray data (figs_structure.py)
_NBINS = 20
_DEFAULT_RBINS_KM = 10**np.linspace(0., np.log10(400), _NBINS)


def run_single(glider_csv, t_start=5001, lev=1,
               offset_x=0.0, offset_y=0.0, 
               output_path=None, cache=True, verbose=True):
    """Run QG velocity sampling for one (start time, offset) combination.

    Parameters
    ----------
    glider_csv : str or Path
        Path to the input glider trajectory CSV.
    t_start : int
        QG time index at which glider time=0 begins.
    lev : int
        Vertical level (1=upper, 2=lower).
    offset_x, offset_y : float
        Translation of glider positions in grid units.
    output_path : str or Path, optional
        Output CSV path.
    cache : bool
        Skip re-running if output exists.
    verbose : bool
        Stream Julia output.

    Returns
    -------
    df : pd.DataFrame
        Velocity data (x, y, time, missid, x_m, y_m, u_qg, v_qg).
    meta : dict
        Metadata dict.
    """
    return run_gliders(
        glider_csv,
        t_start=t_start,
        lev=lev,
        offset_x=offset_x,
        offset_y=offset_y,
        output_path=output_path,
        cache=cache,
        verbose=verbose,
    )


def compute_glider_sf(glider_df, meta, r_bins_km=None, max_time:float=8.):
    """Compute structure functions from glider-sampled QG velocities.

    Builds SprayData objects from the glider output and uses
    ProfilerPairs to compute up to 3rd order structure functions.

    Parameters
    ----------
    glider_df : pd.DataFrame
        Glider velocity output (x, y, time, missid, x_m, y_m, u_qg, v_qg).
    meta : dict
        Metadata dict.
    r_bins_km : array-like, optional
        Bin edges in km. Default: 20 log-spaced bins from 1 to 400 km.
    max_time : float, optional
        Maximum time difference (hours) for pairing. Default: 8.0.

    Returns
    -------
    Sn_LL : dict
        Longitudinal structure function dict (S1, S2, S3 vs r).
    Sn_TT : dict
        Transverse structure function dict (S1, S2, S3 vs r).
    """
    if r_bins_km is None:
        r_bins_km = _DEFAULT_RBINS_KM

    # Build SprayData objects — one per glider
    gliders = SprayData.all_from_QG_glider(glider_df, meta)

    # Set distances
    for ss, glider in enumerate(gliders):
        profiles = glider.missid == ss
        glider.distE = glider_df.x_m.values[profiles]/1e3
        glider.distN = glider_df.y_m.values[profiles]/1e3

    # Construct ProfilerPairs
    pairs = ProfilerPairs(
        gliders,
        max_time=max_time,
        avoid_same_glider=True,
        randomize=False,
    )

    # Longitudinal: S1, S2, S3 of duL
    pairs.calc_delta(iz=0, variables='duLduLduL')
    pairs.calc_Sn(variables='duLduLduL')
    Sn_LL = pairs.calc_Sn_vs_r(r_bins_km)
    pairs.add_meta(Sn_LL)

    # Transverse: S1, S2, S3 of duT
    pairs.calc_Sn(variables='duTduTduT')
    Sn_TT = pairs.calc_Sn_vs_r(r_bins_km)
    pairs.add_meta(Sn_TT)

    return Sn_LL, Sn_TT


def run_and_analyze(glider_csv, t_start=5001, lev=1,
                    offset_x=0.0, offset_y=0.0,
                    r_bins_km=None, output_path=None,
                    cache=True, verbose=True):
    """Run velocity sampling and compute structure functions.

    Convenience function: calls run_single() then compute_glider_sf().

    Parameters
    ----------
    glider_csv : str or Path
        Path to the input glider trajectory CSV.
    t_start : int
        QG time index at which glider time=0 begins.
    lev : int
        Vertical level.
    offset_x, offset_y : float
        Translation in grid units.
    r_bins_km : array-like, optional
        Bin edges in km.
    output_path : str or Path, optional
        Output CSV path.
    cache : bool
        Skip re-running if output exists.
    verbose : bool
        Stream Julia output.

    Returns
    -------
    glider_df : pd.DataFrame
        Velocity data.
    meta : dict
        Metadata.
    Sn_LL : dict
        Longitudinal structure function dict.
    Sn_TT : dict
        Transverse structure function dict.
    """
    glider_df, meta = run_single(
        glider_csv, t_start=t_start, lev=lev,
        offset_x=offset_x, offset_y=offset_y,
        output_path=output_path, cache=cache, verbose=verbose,
    )

    Sn_LL, Sn_TT = compute_glider_sf(glider_df, meta, r_bins_km=r_bins_km)

    return glider_df, meta, Sn_LL, Sn_TT


def save_sf(Sn_LL, outfile):
    """Save structure function results to JSON.

    Parameters
    ----------
    Sn_LL : dict
        Structure function dict (from compute_glider_sf).
    outfile : str
        Output JSON path.
    """
    jdict = p_io.jsonify(Sn_LL)
    p_io.savejson(outfile, jdict, easy_to_read=True, overwrite=True)
    print(f'Saved: {outfile}')


if __name__ == "__main__":

    # Single-realization run
    #glider_df, meta, Sn_LL, Sn_TT = run_and_analyze(
    #    'data/100km100day10gliders3h.csv', t_start=5001,
    #    output_path='data/glider_sf_LL_ts5001.csv')
    glider_csv = 'data/100km100day10gliders3h.csv'
    glider_df, meta = run_single(
        glider_csv, t_start=5001, lev=1,
        offset_x=115.2, offset_y=115.2,
        output_path='data/gliders_ts5001_x450_y450.csv')

    # Structure functions
    Sn_LL, Sn_TT = compute_glider_sf(glider_df, meta)#, r_bins_km=r_bins_km)
    save_sf(Sn_LL, 'data/glider_sf_LL_ts5001.json')
    #save_sf(Sn_TT, 'data/glider_sf_TT_ts5001.json')
