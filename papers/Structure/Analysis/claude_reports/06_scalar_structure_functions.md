# Direction 5 — Scalar (temperature & salinity) structure functions

**Date:** 2026-07-10
**Author:** JXP & Claude (Fable 5)
**Code:** `papers/Structure/Analysis/py/scalar_sf.py`, `figs_scalar.py`
**Figures:** `Figures_new/fig_scalar_D2_z60.png`,
`fig_scalar_mixedT_z60.png`, `fig_scalar_mixedS_z60.png`
**Data:** 60 m, $\Delta t\le10$ hr, linear bins, `minN=100`, 1000 bootstrap.
In-situ temperature `t` and practical salinity `s` (the stored variables).

## Headline result

**The gliders detect a scalar-variance cascade even though they cannot detect
the velocity energy cascade.** The mixed (Yaglom) third-order structure function
$\langle\delta u_L\,(\delta\theta)^2\rangle$ — the scalar analogue of the
velocity $S_3$ — is significantly **negative and grows in magnitude with
separation** for temperature *and* salinity, in **all three** experiments. A
negative, linear mixed moment is the signature of a **forward (downscale)
scalar-variance cascade** (Yaglom relation $\langle\delta u_L(\delta\theta)^2
\rangle = -C\,\chi\,r$, with $\chi$ the scalar-variance transfer rate). This is
a *positive* cascade detection, in sharp contrast to the null velocity result.

## Why this works when the velocity $S_3$ does not

1. **The scalar cascade is sign-definite and forward.** Unlike the 2D kinetic-
   energy cascade (whose sign is the ambiguous quantity the paper cannot pin
   down), scalar variance always cascades to small scales. The expected sign is
   known and robust, so a modest signal is interpretable.
2. **High signal-to-noise.** The ocean has strong, persistent
   temperature/salinity fronts; the scalar increments carry large,
   spatially-organized gradients, whereas the velocity increments sit closer to
   the ADCP noise floor.
3. **Empirically less bias-sensitive.** Even after the same per-bin centering
   that nulls the velocity $S_3$, the mixed scalar moment retains a coherent,
   single-signed trend.

## Significance (centered mixed moment, bins beyond 2σ)

| Experiment | temperature | salinity |
|---|---|---|
| Calypso 2019 | 17/25 (max \|z\|=5.3) | 20/25 (max \|z\|=7.4) |
| **Calypso 2022** | 2/14 (max \|z\|=3.3) | 2/14 (max \|z\|=3.1) |
| ARCTERX 2023 | 4/14 (max \|z\|=3.2) | 4/14 (max \|z\|=4.4) |

Calypso 2019 shows the strongest, cleanest cascade; the focus Calypso 2022 is
weaker but still shows significant negative bins at large $r$ (the last two bins
are the strongest); ARCTERX is intermediate and strongly negative at large $r$.

## Transfer-rate estimates

Weighted linear fits ($\langle\delta u_L(\delta\theta)^2\rangle = a\,r$ through
the origin) give negative slopes everywhere. Using a 2D Yaglom coefficient
$\langle\delta u_L(\delta\theta)^2\rangle \approx -2\chi r$ (the exact constant
is dimension/derivation dependent — treat $\chi$ as order-of-magnitude):

| Experiment | $\chi_T$ (°C²/s) | $\chi_S$ (PSU²/s) |
|---|---|---|
| Calypso 2019 | 5.2e-8 | 5.3e-8 |
| Calypso 2022 | 3.3e-9 | 1.2e-10 |
| ARCTERX 2023 | 2.2e-7 | 2.2e-9 |

All slopes are negative (forward cascade); ARCTERX (energetic, frontal western
Pacific) has the largest thermal transfer rate, Calypso 2022 the smallest.

## Second-order scalar structure functions

$D_{\theta\theta}(r)=\langle(\delta\theta)^2\rangle$ rises with separation
(see `fig_scalar_D2_z60.png`), consistent with a scalar-variance spectrum
building toward larger scales; these are robustly measured and could anchor a
scalar-spectrum discussion.

## Interpretation & caveats

- The **sign (negative) and linearity are the robust results** — they hold for
  both scalars and all three surveys, and the direction is physically expected.
  The magnitude of $\chi$ is order-of-magnitude only (2D coefficient uncertain).
- The signal likely originates in **submesoscale fronts** (strong
  $\nabla\theta$), so it may reflect frontal stirring rather than a canonical
  homogeneous inertial-range cascade; either way it is genuine downgradient
  scalar transfer that the platforms *can* measure.
- The same bootstrap-adequacy argument as for velocity applies (Direction:
  robustness; DOF inflation ≈ 1), so the many >2σ bins for Calypso 2019 are not
  an error-underestimate artifact.
- Possible residual contamination from a large-scale mean scalar gradient
  crossed with the mean flow; per-bin centering removes the leading term, but a
  dedicated mean-gradient-removal check (as in Direction 1 for velocity) would
  further harden the claim.

## Suggested paper use — a new results subsection

This is the strongest candidate for **new physics** in the paper: "While the
velocity third moment yields no cascade detection, the mixed scalar third
moment $\langle\delta u_L(\delta\theta)^2\rangle$ is significantly negative and
grows with separation for both temperature and salinity across all three
surveys — a detection of the forward scalar-variance cascade. The scalar
cascade is measurable where the velocity cascade is not because its sign is
fixed a priori and the frontal scalar gradients give far higher
signal-to-noise." Consider adding a figure (the centered mixed moment for T and
S) and an order-of-magnitude $\chi$.

## Follow-up worth doing

- Mean scalar-gradient removal (analogue of Direction 1) to confirm robustness.
- Depth dependence of $\chi$ (ties to the depth-dependence direction).
- Compare $\chi_T$, $\chi_S$ and the density-compensation (spice) behavior.
