"""
Small-box drifter analysis for the QG model.

Deploys drifters in a localized box (default 100 km x 100 km).
Analysis functions are in analysis.py; I/O in io.py.

Usage:
    from small_box_drifters import run_small_box
    from analysis import compute_pair_separations, compute_structure_functions
    from qg_io import load_trajectories

    traj, meta = run_small_box(t_start=5001, n_days=100)
    pairs = compute_pair_separations(traj, meta)
    sf = compute_structure_functions(traj, meta)
"""

from qg_drifters import run_drifters
import qg_io


def run_small_box(
    t_start=5001,
    n_days=100,
    box_size_km=100.0,
    drifter_spacing_km=10.0,
    box_center_x=128.0,
    box_center_y=128.0,
    lev=1,
    record_interval=1,
    output_path=None,
    cache=True,
    verbose=True,
):
    """Run a small-box drifter simulation.

    Thin wrapper around qg_drifters.run_drifters() with box-specific defaults.

    Parameters
    ----------
    t_start : int
        Starting time index (1-based). Default 5001 (day ~5000).
    n_days : int
        Number of days to advect. Default 100.
    box_size_km : float
        Side length of the deployment box in km. Default 100.
    drifter_spacing_km : float
        Spacing between drifters in km. Default 10.
    box_center_x, box_center_y : float
        Center of the box in grid-index units. Default (128, 128).
    lev : int
        Vertical level. Default 1 (upper).
    record_interval : int
        Record positions every N days.
    output_path : str or Path, optional
        Output CSV path.
    cache : bool
        Skip re-running if output exists.
    verbose : bool
        Stream Julia output.

    Returns
    -------
    traj : pd.DataFrame
        Trajectory data with columns: ID, x, y, t, x_m, y_m
    meta : dict
        Metadata including box parameters.
    """
    return run_drifters(
        t_start=t_start,
        n_days=n_days,
        n_per_side=1,  # ignored when box_size_km is set
        lev=lev,
        record_interval=record_interval,
        box_center_x=box_center_x,
        box_center_y=box_center_y,
        box_size_km=box_size_km,
        drifter_spacing_km=drifter_spacing_km,
        output_path=output_path,
        cache=cache,
        verbose=verbose,
    )


def plot_test_results(traj, meta, outfile=None):
    """Plot trajectory results from test_small_box_drifters().

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory data.
    meta : dict
        Metadata.
    outfile : str, optional
        Path to save the figure. If None, calls plt.show().
    """
    from plotting import plot_trajectories_over_speed
    import matplotlib.pyplot as plt

    fig, ax = plot_trajectories_over_speed(traj, meta)
    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Saved: {outfile}")
    else:
        plt.show()


def test_small_box_drifters():
    """Test the small box drifters module."""
    traj, meta = run_small_box(t_start=5001, n_days=10,
        output_path='data/test_small_box_drifters.csv')
    plot_test_results(traj, meta, outfile='data/test_small_box_drifters.png')

def full_run(output_path:str='data/small_box_drifters.csv',
        t_start=5001, n_days=100):
    """Full run of the drifters in the small box."""
    traj, meta = run_small_box(
        output_path=output_path,
        t_start=t_start,
        n_days=n_days)

if __name__ == "__main__":
    # 10 day test run
    #test_small_box_drifters()

    ### 100 day full run

    #full_run(output_path='data/small_box_drifters_ts5001_nd100.csv',
    #    t_start=5001, n_days=100)
    traj, meta = qg_io.load_trajectories('data/small_box_drifters_ts5001_nd100.csv')
    plot_test_results(traj, meta, outfile='data/traj_small_box_drifters_ts5001_nd100.png')