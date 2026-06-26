""" Module for Tables in the velocity structure-function (energy cascade) paper.

Modeled after bing/papers/phytoplankton/Tables/py/tables_bing.py.

Builds a LaTeX table describing the three glider experiments analyzed in
the paper (Calypso 2019, Calypso 2022, ARCTERX 2023) and their data.

Column set confirmed in the Q&A (claude_prompts/obs_methods.md, T1-T5):
  Experiment | Location | Dates | Duration (days) | N_gliders | N_profiles | N_pairs
Location and Dates are derived from the data (median lon/lat and the
`time` array); counts are post-cut (the data actually analyzed); N_pairs
uses the same ProfilerPairs settings as the structure-function analysis.
"""
# Imports
import os
import sys
import datetime

import numpy as np

# Local: reuse the analysis pipeline so counts match the structure-function results
sys.path.append(os.path.abspath("../Analysis/py"))
sys.path.append(os.path.abspath("../../Analysis/py"))
import glider_io
import struct_defs

from profiler import profilerpairs

from IPython import embed


# Datasets, in paper order, with display names.
DATASETS = [
    dict(key='Calypso2019',  name='Calypso 2019'),
    dict(key='Calypso2022',  name='Calypso 2022'),
    dict(key='ARCTERX-2023', name='ARCTERX 2023'),
]


def _time_to_datetime(t_sec: float) -> datetime.datetime:
    """Convert one `time` value to a datetime.

    The glider `time` array is in seconds (the analysis forms (t-t)/3600
    -> hours and applies second-based survey-edge cuts). The calendar
    epoch is auto-detected and SHOULD BE SPOT-CHECKED against the data the
    first time this is run (see the print in mktab_experiments):
      - ~1e9 .. 4e9  -> seconds since the Unix epoch (1970)
      - otherwise    -> MATLAB datenum expressed in seconds (days*86400)
    """
    v = float(t_sec)
    if v > 1e12:            # milliseconds since Unix epoch
        v /= 1000.
    if 3e8 < v < 4e9:       # plausible Unix seconds (1979-2096)
        return datetime.datetime.utcfromtimestamp(v)
    # MATLAB datenum (days since 0000) expressed in seconds
    dn = v / 86400.
    return (datetime.datetime.fromordinal(int(dn))
            + datetime.timedelta(days=dn % 1)
            - datetime.timedelta(days=366))


def _fmt_lonlat(lon: float, lat: float) -> str:
    """Format a (lon, lat) center as e.g. '$3.1\\degr$E, $40.7\\degr$N'."""
    ew = 'E' if lon >= 0 else 'W'
    ns = 'N' if lat >= 0 else 'S'
    return f'${abs(lon):.1f}^\\circ${ew}, ${abs(lat):.1f}^\\circ${ns}'


def experiment_stats(dataset_key: str) -> dict:
    """Load one experiment (post-cut) and compute the table quantities."""
    profilers = glider_io.load_dataset(dataset_key)

    n_gliders = len(profilers)
    n_profiles = int(np.sum([p.time.size for p in profilers]))

    all_t = np.concatenate([p.time for p in profilers])
    lon = np.concatenate([p.lon for p in profilers])
    lat = np.concatenate([p.lat for p in profilers])

    t0, t1 = np.nanmin(all_t), np.nanmax(all_t)
    duration_days = float((t1 - t0) / 86400.)
    d0, d1 = _time_to_datetime(t0), _time_to_datetime(t1)

    # Pair count, matching data_utils.load_SF
    gPairs = profilerpairs.ProfilerPairs(
        profilers, max_time=struct_defs.max_time,
        avoid_same_glider=struct_defs.avoid_same_glider,
        remove_nans=True, randomize=False)
    n_pairs = int(gPairs.npairs)

    return dict(
        n_gliders=n_gliders, n_profiles=n_profiles,
        duration_days=duration_days, n_pairs=n_pairs,
        location=_fmt_lonlat(np.nanmedian(lon), np.nanmedian(lat)),
        date_start=d0, date_end=d1,
        raw_t0=float(t0), raw_t1=float(t1),
    )


def mktab_experiments(outfile: str = 'tab_experiments.tex'):
    """Write the LaTeX table describing the three experiments."""
    # Open
    tbfil = open(outfile, 'w')

    # Header
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\caption{The three glider experiments analyzed in this '
                'work and their data. \\label{tab:experiments}}\n')
    tbfil.write('\\begin{tabular}{llccccc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Experiment & Location & Dates & Duration & '
                '$N_{\\rm gliders}$ & $N_{\\rm profiles}$ & $N_{\\rm pairs}$ \\\\ \n')
    tbfil.write(' & & & (days) & & & \\\\ \n')
    tbfil.write('\\hline \n')

    # Rows
    for dset in DATASETS:
        s = experiment_stats(dset['key'])
        # Sanity print so the epoch can be verified on first run
        print('{:s}: raw time {:.6g}..{:.6g} -> {:s} to {:s}'.format(
            dset['key'], s['raw_t0'], s['raw_t1'],
            s['date_start'].strftime('%Y-%m-%d'),
            s['date_end'].strftime('%Y-%m-%d')))
        dates = '{:s}--{:s}'.format(
            s['date_start'].strftime('%d %b %Y'),
            s['date_end'].strftime('%d %b %Y'))
        tbfil.write('{:s} & {:s} & {:s} & {:0.0f} & {:d} & {:d} & {:d} \\\\ \n'.format(
            dset['name'], s['location'], dates, s['duration_days'],
            s['n_gliders'], s['n_profiles'], s['n_pairs']))

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    tbfil.write('\\\\ \n')
    tbfil.write('Notes: Location is the median glider position; counts are '
                'for the data actually analyzed (after survey-edge and '
                'good-velocity cuts). $N_{\\rm pairs}$ counts profile pairs '
                'from distinct gliders separated by no more than $\\Delta t = '
                f'{struct_defs.max_time:0.0f}$~hr. \\\\ \n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()
    print('Wrote {:s}'.format(outfile))


# Command line execution
if __name__ == '__main__':
    mktab_experiments()
