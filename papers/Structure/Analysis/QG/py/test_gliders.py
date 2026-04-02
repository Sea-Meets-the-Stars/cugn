"""
Tests for the QG glider velocity sampling pipeline.

Covers:
  T1 — Trajectory output correctness (columns, types, sizes)
  T2 — Velocity field interpolation sanity (magnitude, spatial coherence)
  T3 — Output file format (CSV columns, JSON metadata)

Each test generates a diagnostic figure in the figures/ directory.

Run:  pytest test_gliders.py -v
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import pytest

# Local modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qg_gliders import run_gliders, load_glider_velocities

# Paths
_QG_DIR = Path(__file__).resolve().parent.parent
_DATA_DIR = _QG_DIR / 'data'
_GLIDER_CSV = _DATA_DIR / '100km100day10gliders3h.csv'
_FIG_DIR = _QG_DIR / 'figures'
_FIG_DIR.mkdir(exist_ok=True)

# Test output location (avoids polluting real cache)
_TEST_OUTPUT = Path('/tmp/qg_gliders_test/test_output.csv')


# ---------------------------------------------------------------------------
#  Fixture: run the pipeline once for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def glider_result():
    """Run the glider pipeline once and return (df, meta)."""
    _TEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df, meta = run_gliders(
        str(_GLIDER_CSV),
        t_start=5001,
        lev=1,
        offset_x=0.0,
        offset_y=0.0,
        output_path=str(_TEST_OUTPUT),
        cache=False,
        verbose=False,
    )
    return df, meta


# ---------------------------------------------------------------------------
#  T1: Trajectory output correctness
# ---------------------------------------------------------------------------

class TestTrajectoryOutput:
    """Test that output DataFrame has correct structure and values."""

    def test_columns(self, glider_result):
        """Output must have exactly the 8 required columns."""
        df, _ = glider_result
        expected = ['x', 'y', 'time', 'missid', 'x_m', 'y_m', 'u_qg', 'v_qg']
        assert list(df.columns) == expected

    def test_n_records(self, glider_result):
        """10 gliders × 801 time steps = 8010 records."""
        df, _ = glider_result
        assert len(df) == 8010

    def test_n_gliders(self, glider_result):
        """Must have exactly 10 unique glider IDs."""
        df, meta = glider_result
        assert df.missid.nunique() == 10
        assert meta['n_gliders'] == 10

    def test_missid_dtype(self, glider_result):
        """missid must be integer, not float."""
        df, _ = glider_result
        assert df.missid.dtype in (np.int64, np.int32, int)

    def test_time_range(self, glider_result):
        """Time should span 0 to 8640000 seconds (100 days)."""
        df, _ = glider_result
        assert df.time.min() == 0.0
        assert df.time.max() == pytest.approx(8640000.0)

    def test_physical_coords(self, glider_result):
        """x_m and y_m must equal x * dx and y * dx."""
        df, meta = glider_result
        dx = meta['dx']
        np.testing.assert_allclose(df.x_m, df.x * dx, rtol=1e-10)
        np.testing.assert_allclose(df.y_m, df.y * dx, rtol=1e-10)

    def test_no_nans(self, glider_result):
        """No NaN values in any column."""
        df, _ = glider_result
        assert not df.isna().any().any()

    def test_figure_trajectories(self, glider_result):
        """Generate figure: glider trajectories over QG speed field."""
        df, meta = glider_result
        dx = meta['dx']
        nx = meta['nx']
        t_start = meta['t_start']

        nc_path = os.path.join(os.environ['OS_DATA'], 'QG', 'QGModelOutput20years.nc')
        qg = xr.open_dataset(nc_path)
        u0 = qg.u.isel(lev=0, time=t_start - 1).values
        v0 = qg.v.isel(lev=0, time=t_start - 1).values
        speed = np.sqrt(u0**2 + v0**2)
        x_km = qg.x.values / 1000
        y_km = qg.y.values / 1000
        qg.close()

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        ax.pcolormesh(x_km, y_km, speed.T, cmap='viridis', alpha=0.3, shading='auto')

        # Plot each glider trajectory
        for mid in sorted(df.missid.unique()):
            sub = df[df.missid == mid]
            ax.plot(sub.x_m / 1000, sub.y_m / 1000, '-', lw=0.8, alpha=0.7,
                    label=f'Glider {mid}')
            ax.plot(sub.x_m.iloc[0] / 1000, sub.y_m.iloc[0] / 1000, 'r.', ms=5)

        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_aspect('equal')
        ax.set_title(f'Glider trajectories over QG speed field (t_start={t_start})')
        ax.legend(fontsize=6, ncol=2, loc='upper right')
        plt.tight_layout()
        fig.savefig(_FIG_DIR / 'test_glider_trajectories.png', dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
#  T2: Velocity interpolation sanity
# ---------------------------------------------------------------------------

class TestVelocityInterpolation:
    """Test that interpolated velocities are physically reasonable."""

    def test_velocity_magnitude(self, glider_result):
        """All velocities must be < 1 m/s (QG model typical max ~0.3 m/s)."""
        df, _ = glider_result
        speed = np.sqrt(df.u_qg**2 + df.v_qg**2)
        assert speed.max() < 1.0, f"Max speed {speed.max():.3f} m/s exceeds 1 m/s"

    def test_velocity_mean_near_zero(self, glider_result):
        """Mean u and v should be near zero (no large-scale bias)."""
        df, _ = glider_result
        assert abs(df.u_qg.mean()) < 0.05, f"|mean(u)| = {abs(df.u_qg.mean()):.4f}"
        assert abs(df.v_qg.mean()) < 0.05, f"|mean(v)| = {abs(df.v_qg.mean()):.4f}"

    def test_velocity_std_reasonable(self, glider_result):
        """Velocity std should be in range 0.01–0.2 m/s for QG model."""
        df, _ = glider_result
        u_std = df.u_qg.std()
        v_std = df.v_qg.std()
        assert 0.01 < u_std < 0.2, f"u_std = {u_std:.4f}"
        assert 0.01 < v_std < 0.2, f"v_std = {v_std:.4f}"

    def test_temporal_continuity(self, glider_result):
        """Velocity should not jump wildly between consecutive 3h samples."""
        df, _ = glider_result
        # Check glider 1 for temporal smoothness
        g1 = df[df.missid == 1].sort_values('time')
        du = np.diff(g1.u_qg.values)
        dv = np.diff(g1.v_qg.values)
        # Max change in 3 hours should be << 0.5 m/s
        assert np.max(np.abs(du)) < 0.5, f"Max |Δu| = {np.max(np.abs(du)):.4f}"
        assert np.max(np.abs(dv)) < 0.5, f"Max |Δv| = {np.max(np.abs(dv)):.4f}"

    def test_different_gliders_different_velocities(self, glider_result):
        """At later times, gliders at different positions must have different velocities."""
        df, _ = glider_result
        # At the last time step, all 10 gliders are at different positions
        last_t = df.time.max()
        last = df[df.time == last_t]
        assert last.u_qg.nunique() == 10, "All 10 gliders should have distinct u_qg at final time"

    def test_figure_velocity_timeseries(self, glider_result):
        """Generate figure: velocity time series for selected gliders."""
        df, meta = glider_result

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        for mid in [1, 5, 10]:
            sub = df[df.missid == mid].sort_values('time')
            t_days = sub.time / 86400.0
            axes[0].plot(t_days, sub.u_qg, lw=0.8, alpha=0.8, label=f'Glider {mid}')
            axes[1].plot(t_days, sub.v_qg, lw=0.8, alpha=0.8, label=f'Glider {mid}')

        axes[0].set_ylabel('u_qg (m/s)')
        axes[0].legend(fontsize=8)
        axes[0].set_title('QG velocity sampled at glider positions')
        axes[0].axhline(0, color='k', lw=0.5, ls='--')
        axes[1].set_ylabel('v_qg (m/s)')
        axes[1].set_xlabel('Time (days)')
        axes[1].axhline(0, color='k', lw=0.5, ls='--')
        plt.tight_layout()
        fig.savefig(_FIG_DIR / 'test_glider_velocity_timeseries.png', dpi=150)
        plt.close(fig)

    def test_figure_velocity_scatter(self, glider_result):
        """Generate figure: velocity field snapshot with glider-sampled values."""
        df, meta = glider_result
        dx = meta['dx']
        t_start = meta['t_start']

        nc_path = os.path.join(os.environ['OS_DATA'], 'QG', 'QGModelOutput20years.nc')
        qg = xr.open_dataset(nc_path)

        # Show velocity at day 50 (halfway through simulation)
        day_50_s = 50 * 86400.0
        # QG snapshot index closest to this time
        t_idx = t_start + 50 - 1  # 0-indexed in xr
        u_field = qg.u.isel(lev=0, time=t_idx).values
        v_field = qg.v.isel(lev=0, time=t_idx).values
        x_km = qg.x.values / 1000
        y_km = qg.y.values / 1000
        qg.close()

        # Get glider positions nearest to day 50
        snap = df[(df.time >= day_50_s - 5400) & (df.time <= day_50_s + 5400)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: u-field with glider-sampled u overlaid
        im0 = axes[0].pcolormesh(x_km, y_km, u_field.T, cmap='RdBu_r',
                                  vmin=-0.2, vmax=0.2, shading='auto')
        sc0 = axes[0].scatter(snap.x_m / 1000, snap.y_m / 1000, c=snap.u_qg,
                               cmap='RdBu_r', vmin=-0.2, vmax=0.2,
                               edgecolors='k', linewidths=0.5, s=30, zorder=5)
        axes[0].set_title('u (m/s) — field + glider samples at day 50')
        axes[0].set_xlabel('x (km)')
        axes[0].set_ylabel('y (km)')
        axes[0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0], shrink=0.7)

        # Right: v-field with glider-sampled v overlaid
        im1 = axes[1].pcolormesh(x_km, y_km, v_field.T, cmap='RdBu_r',
                                  vmin=-0.2, vmax=0.2, shading='auto')
        sc1 = axes[1].scatter(snap.x_m / 1000, snap.y_m / 1000, c=snap.v_qg,
                               cmap='RdBu_r', vmin=-0.2, vmax=0.2,
                               edgecolors='k', linewidths=0.5, s=30, zorder=5)
        axes[1].set_title('v (m/s) — field + glider samples at day 50')
        axes[1].set_xlabel('x (km)')
        axes[1].set_ylabel('y (km)')
        axes[1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1], shrink=0.7)

        plt.tight_layout()
        fig.savefig(_FIG_DIR / 'test_glider_velocity_field.png', dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
#  T3: Output file format
# ---------------------------------------------------------------------------

class TestOutputFile:
    """Test that the output CSV and JSON files are correctly formatted."""

    def test_csv_exists(self, glider_result):
        """Output CSV must exist."""
        assert _TEST_OUTPUT.exists()

    def test_json_exists(self, glider_result):
        """Sidecar JSON must exist."""
        meta_path = Path(str(_TEST_OUTPUT) + '.meta.json')
        assert meta_path.exists()

    def test_csv_columns(self, glider_result):
        """CSV header must match the R5 spec exactly."""
        with open(_TEST_OUTPUT) as f:
            header = f.readline().strip()
        assert header == 'x,y,time,missid,x_m,y_m,u_qg,v_qg'

    def test_json_fields(self, glider_result):
        """JSON must contain all required metadata fields."""
        meta_path = Path(str(_TEST_OUTPUT) + '.meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        required = ['dx', 'nx', 't_start', 'lev', 'n_gliders',
                     'offset_x', 'offset_y', 'glider_csv']
        for key in required:
            assert key in meta, f"Missing metadata key: {key}"

    def test_json_values(self, glider_result):
        """JSON values must match what we passed in."""
        _, meta = glider_result
        assert meta['t_start'] == 5001
        assert meta['lev'] == 1
        assert meta['offset_x'] == 0.0
        assert meta['offset_y'] == 0.0
        assert meta['n_gliders'] == 10
        assert meta['dx'] == pytest.approx(3906.25)
        assert meta['nx'] == 256

    def test_roundtrip_load(self, glider_result):
        """load_glider_velocities must recover the same data."""
        df_orig, meta_orig = glider_result
        df_loaded, meta_loaded = load_glider_velocities(str(_TEST_OUTPUT))
        pd.testing.assert_frame_equal(df_orig, df_loaded)
        assert meta_orig == meta_loaded
