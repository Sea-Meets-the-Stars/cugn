# Direction 7 — Depth dependence of the structure functions

**Date:** 2026-07-10
**Author:** JXP & Claude (Fable 5)
**Code:** `papers/Structure/Analysis/py/depth_dep.py`, `figs_depth.py`
**Figure:** `Figures_new/fig_depth_sweep.png`
**Data:** depth levels 10–160 m (`iz` 0–15), $\Delta t\le10$ hr, linear bins,
`minN=100`, 500 bootstrap. Pairs built once per experiment (they depend only on
position/time), with `calc_delta` re-run at each level.

## Question

The paper fixes increments at 60 m. How do the diagnostics vary with depth, and
is 60 m a good choice?

## Results

**1. Energy is surface-intensified.** The median $D_{LL}$ decays monotonically
with depth in all three surveys (e.g. Calypso 2022: 0.023 m²/s² at 10 m → 0.005
at 160 m; ARCTERX: 0.075 → 0.019). Turbulent kinetic energy is concentrated in
the upper ocean, as expected.

**2. The $S_1$ bias is also surface-intensified.** max$|S_1|$ decreases with
depth for Calypso 2019 (0.107 → 0.034 m/s) and ARCTERX (0.147 → 0.057 m/s); for
Calypso 2022 it is small (~0.02 m/s) and nearly depth-independent. So the
sampling bias is worst near the surface, where the energetic mesoscale/
submesoscale flow is strongest.

**3. The velocity cascade is null at every depth.** The number of centered-$S_3$
bins beyond 2σ stays at the noise level (0–4 of 14–25) throughout the water
column for all three experiments. The paper's null is **not** a depth artifact —
it holds from 10 to 160 m.

**4. The scalar (temperature) cascade is present at nearly all depths.** The
mixed-moment slope $a$ ($\propto-\chi_T$) is **negative** (forward cascade) at
essentially every level, strongest at **mid-depth** (~30–100 m): ARCTERX peaks
at $a\approx-8\times10^{-7}$ near 50 m, Calypso 2019 at $\approx-1\times10^{-7}$
near 50–70 m. Calypso 2022 is weakly negative (its scalar signal is the
faintest, consistent with Direction 5). The forward scalar cascade is thus a
robust, depth-persistent feature, intensified in the subsurface where
thermocline fronts are sharpest — a strong corroboration of Direction 5.

## Is 60 m a good choice?

Yes. It sits below the noisy, most strongly biased surface layer, in the depth
range where the scalar cascade is robust and the energy is still appreciable.
Nothing qualitative changes across the resolved column, so the paper's fixed-
depth result is representative.

## Isopycnal check — not possible from these files

The stored SprayData for these experiments do **not** carry a density field
(`sigma`), so the fixed-depth-vs-isopycnal comparison the paper mentions could
not be reproduced here. `depth_dep.isopycnal_check()` is written and will run
once a density field is added (or computed from `t`, `s`, pressure via GSW —
a small follow-up). This is a data-availability limitation, not a result.

## Suggested paper use

- One figure/paragraph: "The velocity third-moment null holds at all depths
  (10–160 m); the energy and the $S_1$ bias are surface-intensified; and the
  forward scalar-variance cascade (Direction 5) is present throughout, peaking
  in the subsurface (30–100 m)."
- Justify the 60 m choice explicitly with the depth sweep.

## Follow-up

- Compute potential density from `t`, `s`, depth (GSW) to enable the isopycnal
  comparison and confirm the results are not an artifact of vertical heave.
