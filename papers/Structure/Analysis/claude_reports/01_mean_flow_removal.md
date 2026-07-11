# Direction 1 — Mean-flow removal as a competing estimator

**Date:** 2026-07-10
**Author:** JXP & Claude (Fable 5)
**Code:** `papers/Structure/Analysis/py/mean_flow.py`, `figs_mean_flow.py`
**Figures:** `papers/Structure/Analysis/Figures_new/fig_meanflow_*_z60.png`,
`fig_meanflow_quiver_z60.png`
**Data:** Calypso 2019, Calypso 2022 (focus), ARCTERX 2023; ADCP velocities at
60 m (`iz=5`); pairs from distinct gliders with $\Delta t \le 10$ hr; linear
binning, `minN=100`, 1000 bootstrap realizations.

## Question

The paper removes the sampling-induced $S_1\neq0$ bias in the third moment
*algebraically*, via the centered moment
$S_{3,c} = S_3 - 3\,S_1 S_2 + 2\,S_1^3$. A natural physical alternative is to
**fit and subtract a smooth survey-scale mean flow** $\mathbf{u}(\mathbf{x})$
(constant + linear shear, optionally quadratic) from the profile velocities
*before* forming increments, then recompute the structure functions. Does that
attack the bias at its root, and does it agree with the algebraic centering?

The motivating identity: a spatially *constant* mean flow cancels exactly in the
velocity difference $\delta u = u_1 - u_0$, so only the spatially varying part
(the shear) can change any structure function. A pure affine field
$\mathbf{u} = \mathbf{A} + \mathsf{G}\mathbf{x}$ gives $\delta u = \mathsf{G}\,
\mathbf{r}$ — an increment that grows linearly with separation, exactly the
scale-dependent $S_1$ the surveys show.

## Method

`mean_flow.py` fits $u(E,N)$ and $v(E,N)$ independently by least squares in the
survey's E/N (km) frame (`distE`, `distN`), at order 0 (constant), 1 (affine),
or 2 (quadratic), subtracts the fit from the depth-60 m velocities, and
recomputes $S_1$, $S_3$ (raw and centered) on the **same** deterministic pairs
(`randomize=False`), so cases are directly comparable.

**Validation.** Order-0 removal (subtract a constant) is a no-op on the
structure functions to machine precision: `max|ΔS1| = 1.7e-18`,
`max|ΔS3| = 2.2e-19` over the good bins. The pipeline is correct.

## Results

| Experiment | affine $R^2(u),R^2(v)$ | affine KE frac | $S_1$ max before → after affine | vorticity / strain (s⁻¹) |
|---|---|---|---|---|
| Calypso 2019 | 0.05, 0.06 | 0.14 | 0.061 → 0.061 | +1.1e-6 / 1.4e-6 |
| **Calypso 2022** | 0.005, 0.016 | **0.02** | 0.0185 → 0.0228 | −4.3e-8 / 6.6e-7 |
| ARCTERX 2023 | 0.09, 0.06 | 0.19 | 0.099 → 0.086 | −9.4e-7 / 1.3e-6 |

Three robust conclusions:

1. **The survey-scale mean flow is weak and incoherent.** A global affine fit
   explains only 0.5–9% of the point velocity variance and holds 2–19% of the
   kinetic energy; the quadratic fit adds little (≤20%). For the focus
   experiment, Calypso 2022, the affine flow is essentially negligible (2% of
   KE). The quiver figure shows the observed velocities (gray) dominated by
   mesoscale eddies with only a faint coherent large-scale gradient (red).

2. **Removing the fitted flow does not reduce the $S_1$ bias.** After affine
   removal, $\max|S_1|$ is unchanged (Calypso 2019), slightly *larger* (Calypso
   2022), or only marginally smaller (ARCTERX 2023). The quadratic removal is no
   better. This is because the fitted linear shear ($\mathsf{G}\sim10^{-6}$ s⁻¹)
   contributes to $\delta u_L$ at the ~0.04 m/s level over $r\sim60$ km —
   comparable to $S_1$ itself — but its *global* least-squares direction is not
   what the pair ensemble samples, so subtracting it moves $S_1$ in an
   uncontrolled direction rather than toward zero.

3. **Flow removal does not reproduce the centered moment.** The affine-removed
   raw $S_3$ differs from the paper's centered $S_{3,c}$ by a median of
   $2.9\times10^{-4}$ (Calypso 2022) against a typical centered
   $|S_{3,c}|\sim5\times10^{-5}$ — i.e. ~6× larger. The two estimators are *not*
   interchangeable.

## Interpretation

The nonzero $S_1$ that biases the third moment is **not** a coherent, smoothly
removable background current. It is the imprint of an inhomogeneously and
sparsely **sampled mesoscale eddy field**: the velocity is eddy-dominated
(low fit $R^2$), so the mean increment in a separation bin reflects *which*
eddies the pairs happened to sample, not a subtractable mean shear.

Two consequences for the paper:

- The **algebraic centering is the right tool.** There is no superior "physical
  mean-flow subtraction" available from a smooth fit; a fitted flow neither
  reduces $S_1$ nor recovers $S_{3,c}$.
- This gives a **real-data corroboration of the paper's QG conclusion.** The QG
  testbed argues the obstacle is sparse sampling, not the absence of a cascade;
  here, from the observations alone (no QG), we find the $S_1$ bias cannot be
  attributed to a removable mean flow — it is intrinsic to sampling turbulence
  with a finite survey. The survey-scale relative vorticity is small,
  $\zeta/f\sim0.01$–0.02 (using $f\approx9.5\times10^{-5}$ s⁻¹ at Calypso,
  $4.9\times10^{-5}$ s⁻¹ at ARCTERX), confirming the large-scale gradients are
  weak.

## Suggested paper use

A short Discussion paragraph (or an appendix subsection): "We tested whether the
$S_1$ bias could be removed by subtracting a fitted survey-scale mean flow. A
global affine/quadratic fit captures <15% of the velocity variance and neither
reduces $S_1$ nor reproduces the centered moment, indicating the bias is not a
coherent removable shear but the signature of a sparsely sampled eddy field —
consistent with the sampling-limited picture developed with the QG testbed."

## Caveats / next steps

- The fit is a *global* smooth field; a scale-local (e.g. per-day or
  spatially-windowed) mean removal could be explored, but the low variance
  fraction makes a large improvement unlikely.
- Depth-dependence of the mean-flow strength is examined in the depth-dependence
  analysis (later direction).
