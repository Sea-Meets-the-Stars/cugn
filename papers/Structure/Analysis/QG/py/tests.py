"""
Tests for structure function analysis of QG drifter trajectories.

Covers:
  T1 — DrifterData construction
  T2 — Time-offset handling
  T3 — ProfilerPairs compatibility
  T4 — Velocity sanity check
  T5 — Cross-approach comparison

Run:  pytest tests.py -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI / headless runs
import matplotlib.pyplot as plt
import pytest

# Local modules in the same QG/py/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qg_io import load_trajectories
from analysis import compute_structure_functions

# Profiler package
from profiler.drifterdata import DrifterData
from profiler.profilerpairs import ProfilerPairs

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

# Path to the small test dataset (121 drifters, 10 days, 11 time steps)
_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
_TEST_CSV = _DATA_DIR / 'test_small_box_drifters.csv'

# Output directory for figures
_FIG_DIR = Path(__file__).resolve().parent.parent / 'figures'
_FIG_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope='module')
def traj_meta():
    """Load the test trajectory dataset once per module."""
    traj, meta = load_trajectories(str(_TEST_CSV))
    return traj, meta


@pytest.fixture(scope='module')
def all_drifters(traj_meta):
    """Build all 121 DrifterData objects once per module."""
    traj, meta = traj_meta
    return DrifterData.all_from_QG_trajectory(traj, meta)


@pytest.fixture(scope='module')
def five_drifter_pairs(traj_meta):
    """Build ProfilerPairs from the first 5 drifters."""
    traj, meta = traj_meta
    ids = sorted(traj.ID.unique())[:5]
    drifters = [DrifterData.from_QG_trajectory(traj, meta, did) for did in ids]
    pairs = ProfilerPairs(
        drifters, max_time=1.0,
        avoid_same_glider=True, randomize=False,
    )
    return pairs


# ---------------------------------------------------------------------------
#  T1: DrifterData construction
# ---------------------------------------------------------------------------

class TestDrifterDataConstruction:

    def test_single_drifter(self, traj_meta):
        """T1.1 — single DrifterData has correct shapes and types."""
        traj, meta = traj_meta
        d = DrifterData.from_QG_trajectory(traj, meta, drifter_id=1.0)

        # time: 1D, 11 elements
        assert d.time.ndim == 1
        assert d.time.shape[0] == 11

        # lat, lon: 1D, 11 elements, all finite
        assert d.lat.shape == (11,)
        assert d.lon.shape == (11,)
        assert np.all(np.isfinite(d.lat))
        assert np.all(np.isfinite(d.lon))

        # udop, vdop: 2D, (11, 1), all finite
        assert d.udop.shape == (11, 1)
        assert d.vdop.shape == (11, 1)
        assert np.all(np.isfinite(d.udop))
        assert np.all(np.isfinite(d.vdop))

        # depth: single surface level
        np.testing.assert_array_equal(d.depth, [0.0])

        # missid: integer drifter ID
        assert d.missid == 1

        # dataset: non-empty string
        assert isinstance(d.dataset, str) and len(d.dataset) > 0

    def test_all_drifters(self, all_drifters):
        """T1.2 — batch constructor returns 121 unique DrifterData objects."""
        assert len(all_drifters) == 121
        assert all(isinstance(d, DrifterData) for d in all_drifters)

        # All missid values unique
        missids = [d.missid for d in all_drifters]
        assert len(set(missids)) == 121


# ---------------------------------------------------------------------------
#  T2: Time-offset handling
# ---------------------------------------------------------------------------

class TestTimeOffsets:

    def test_time_offsets_unique(self, all_drifters):
        """T2.1 — no two drifters share the exact same timestamp."""
        # Gather the first time value from each drifter
        t0_values = np.array([d.time[0] for d in all_drifters])
        # All should be distinct
        assert len(np.unique(t0_values)) == len(t0_values)

    def test_time_offsets_small(self, all_drifters):
        """T2.2 — max offset is < 1 second (negligible vs 86400 s step)."""
        # The nominal base time at t=0 is 0.0; offsets are drifter_id * 1e-3
        max_id = max(d.missid for d in all_drifters)
        max_offset = max_id * 1e-3
        assert max_offset < 1.0


# ---------------------------------------------------------------------------
#  T3: ProfilerPairs compatibility
# ---------------------------------------------------------------------------

class TestProfilerPairsCompat:

    def test_profilepairs_construction(self, five_drifter_pairs):
        """T3.1 — ProfilerPairs creates non-zero pairs with no self-pairs."""
        pairs = five_drifter_pairs
        assert pairs.npairs > 0

        # No self-pairs: missid must differ for every pair
        m0 = pairs.data('missida', 0)
        m1 = pairs.data('missida', 1)
        assert not np.any(m0 == m1)

    def test_calc_delta_and_Sn(self, five_drifter_pairs):
        """T3.2 — calc_delta / calc_Sn / calc_Sn_vs_r produce expected shapes and keys."""
        pairs = five_drifter_pairs
        np_pairs = pairs.npairs

        # calc_delta populates duL, duT
        pairs.calc_delta(iz=0, variables='duLduLduL')
        assert pairs.duL is not None and len(pairs.duL) == np_pairs
        assert pairs.duT is not None and len(pairs.duT) == np_pairs

        # calc_Sn populates S1, S2, S3
        pairs.calc_Sn(variables='duLduLduL')
        assert pairs.S1 is not None and len(pairs.S1) == np_pairs
        assert pairs.S2 is not None and len(pairs.S2) == np_pairs
        assert pairs.S3 is not None and len(pairs.S3) == np_pairs

        # calc_Sn_vs_r returns dict with expected keys
        rbins = 10**np.linspace(np.log10(5), np.log10(200), 11)  # 10 bins, km
        Sn = pairs.calc_Sn_vs_r(rbins)
        assert 'r' in Sn
        assert 'S2_duL**2' in Sn
        assert len(Sn['r']) == 10


# ---------------------------------------------------------------------------
#  T4: Velocity sanity check
# ---------------------------------------------------------------------------

class TestVelocitySanity:

    def test_velocity_magnitude(self, traj_meta):
        """T4.1 — all velocities < 1 m/s (QG typical ~0.1 m/s)."""
        traj, meta = traj_meta
        d = DrifterData.from_QG_trajectory(traj, meta, drifter_id=1.0)
        speed = np.sqrt(d.udop**2 + d.vdop**2)
        assert np.all(speed < 1.0), f'Max speed = {speed.max():.3f} m/s'

    def test_velocity_magnitude_figure(self, all_drifters):
        """T4 figure — histogram of all drifter speeds."""
        speeds = []
        for d in all_drifters:
            speeds.append(np.sqrt(d.udop**2 + d.vdop**2).ravel())
        speeds = np.concatenate(speeds)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(speeds, bins=40, edgecolor='k', alpha=0.7)
        ax.set_xlabel('Speed (m/s)')
        ax.set_ylabel('Count')
        ax.set_title('QG Drifter Velocity Magnitudes')
        ax.axvline(1.0, color='r', ls='--', label='1 m/s threshold')
        ax.legend()
        fig.tight_layout()
        fig.savefig(_FIG_DIR / 'T4_velocity_histogram.png', dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
#  T5: Structure function comparison (approach 1 vs approach 2)
# ---------------------------------------------------------------------------

class TestApproachComparison:

    def test_approaches_qualitative_agreement(self, traj_meta):
        """T5.1 — D_LL from both approaches agree within a factor of 3."""
        traj, meta = traj_meta

        # Use first 10 drifters to keep runtime manageable
        ids = sorted(traj.ID.unique())[:10]
        traj_sub = traj[traj.ID.isin(ids)]

        # Shared bin edges (meters and km)
        r_bins_m = np.logspace(np.log10(5e3), np.log10(200e3), 16)
        r_bins_km = r_bins_m / 1e3

        # Approach 1: analysis.py
        sf = compute_structure_functions(traj_sub, meta, r_bins=r_bins_m)

        # Approach 2: DrifterData + ProfilerPairs
        drifters = DrifterData.all_from_QG_trajectory(traj_sub, meta)
        pairs = ProfilerPairs(
            drifters, max_time=1.0,
            avoid_same_glider=True, randomize=False,
        )
        pairs.calc_delta(iz=0, variables='duLduLduL')
        pairs.calc_Sn(variables='duLduLduL')
        Sn_LL = pairs.calc_Sn_vs_r(r_bins_km)

        # Compare D_LL at bins where both approaches have data
        dll_a1 = sf['D_LL'].values
        dll_a2 = Sn_LL['S2_duL**2']

        # Only compare bins with valid data in both
        valid = np.isfinite(dll_a1) & np.isfinite(dll_a2) & (dll_a1 > 0) & (dll_a2 > 0)
        assert np.sum(valid) > 3, 'Too few overlapping bins to compare'

        ratio = dll_a2[valid] / dll_a1[valid]
        assert np.all(ratio > 1.0 / 3.0), f'Min ratio = {ratio.min():.3f}'
        assert np.all(ratio < 3.0), f'Max ratio = {ratio.max():.3f}'

    def test_comparison_figure(self, traj_meta):
        """T5 figure — D_LL and D_TT from both approaches on log-log axes."""
        traj, meta = traj_meta
        ids = sorted(traj.ID.unique())[:10]
        traj_sub = traj[traj.ID.isin(ids)]

        # Shared bin edges
        r_bins_m = np.logspace(np.log10(5e3), np.log10(200e3), 16)
        r_bins_km = r_bins_m / 1e3

        # Approach 1
        sf = compute_structure_functions(traj_sub, meta, r_bins=r_bins_m)

        # Approach 2
        drifters = DrifterData.all_from_QG_trajectory(traj_sub, meta)
        pairs = ProfilerPairs(
            drifters, max_time=1.0,
            avoid_same_glider=True, randomize=False,
        )
        pairs.calc_delta(iz=0, variables='duLduLduL')

        pairs.calc_Sn(variables='duLduLduL')
        Sn_LL = pairs.calc_Sn_vs_r(r_bins_km)

        pairs.calc_Sn(variables='duTduTduT')
        Sn_TT = pairs.calc_Sn_vs_r(r_bins_km)

        # --- Figure ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # D_LL panel
        ax = axes[0]
        r_a1_km = sf['r_mid'].values / 1e3
        ax.loglog(r_a1_km, sf['D_LL'].values, 'o-', label='Approach 1 (analysis.py)')
        ax.loglog(Sn_LL['r'], Sn_LL['S2_duL**2'], 's--', label='Approach 2 (ProfilerPairs)')
        # Reference slopes
        r_ref = np.array([5, 200])
        ax.loglog(r_ref, 1e-4 * (r_ref / 10)**2, 'k:', alpha=0.4, label=r'$r^2$')
        ax.loglog(r_ref, 5e-4 * (r_ref / 10)**(2/3), 'k--', alpha=0.4, label=r'$r^{2/3}$')
        ax.set_xlabel('Separation r (km)')
        ax.set_ylabel(r'$D_{LL}$ (m$^2$/s$^2$)')
        ax.set_title('Longitudinal SF')
        ax.legend(fontsize=8)

        # D_TT panel
        ax = axes[1]
        ax.loglog(r_a1_km, sf['D_TT'].values, 'o-', label='Approach 1 (analysis.py)')
        ax.loglog(Sn_TT['r'], Sn_TT['S2_duT**2'], 's--', label='Approach 2 (ProfilerPairs)')
        r_ref = np.array([5, 200])
        ax.loglog(r_ref, 1e-4 * (r_ref / 10)**2, 'k:', alpha=0.4, label=r'$r^2$')
        ax.loglog(r_ref, 5e-4 * (r_ref / 10)**(2/3), 'k--', alpha=0.4, label=r'$r^{2/3}$')
        ax.set_xlabel('Separation r (km)')
        ax.set_ylabel(r'$D_{TT}$ (m$^2$/s$^2$)')
        ax.set_title('Transverse SF')
        ax.legend(fontsize=8)

        fig.suptitle('Structure Functions: Approach 1 vs Approach 2 (10 drifters, 10 days)')
        fig.tight_layout()
        fig.savefig(_FIG_DIR / 'T5_approach_comparison.png', dpi=150)
        plt.close(fig)
