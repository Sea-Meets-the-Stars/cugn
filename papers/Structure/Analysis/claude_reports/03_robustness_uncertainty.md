# Direction (robustness) — Uncertainties, Δt sensitivity, last-bin broadening

**Date:** 2026-07-10
**Author:** JXP & Claude (Fable 5)
**Code:** `papers/Structure/Analysis/py/robustness.py`, `figs_robustness.py`
**Figures:** `Figures_new/fig_robust_dof_z60.png`,
`fig_robust_dtscan_z60.png`
**Data:** 60 m, linear bins, `minN=100`, 1000 bootstrap.

Three checks bearing directly on the paper's claim that the centered third
moment is "consistent with zero (no bin beyond ~2σ)."

## 1. Bootstrap vs. effective degrees of freedom

**Concern (Q&A #4):** the bootstrap resamples *pairs*, treating them as
independent. They are not — each profile enters many pairs. If the bin-mean
truly has fewer independent samples than $N_{\rm pairs}$, the bootstrap
under-estimates the error and inflates significance.

**Test:** per bin, count distinct profiles $N_{\rm prof}$ contributing (a
generous upper bound on the effective sample size, since a bin-mean cannot carry
more independent information than the number of profiles). Bootstrap SE scales
as $\sigma/\sqrt{N_{\rm pairs}}$, DOF SE as $\sigma/\sqrt{N_{\rm prof}}$, so the
under-estimate factor is $\sqrt{N_{\rm pairs}/N_{\rm prof}}$.

**Result — the bootstrap is adequate:**

| Experiment | median inflation | bins >2σ (bootstrap) | bins >2σ (DOF) | max \|z\| boot → DOF |
|---|---|---|---|---|
| Calypso 2019 | 1.01 | 1/25 | 1/25 | 2.23 → 2.26 |
| **Calypso 2022** | 1.19 | 0/14 | 0/14 | 1.83 → 1.81 |
| ARCTERX 2023 | 0.89 | 0/14 | 0/14 | 1.22 → 1.36 |

The inflation is only ~1.0–1.2×, and the significance verdict is unchanged. The
reason: within a *narrow* separation bin (~5 km) **and** a short time window
(≤10 hr), a given profile has only a handful of eligible partners, so pairs in a
bin are far less redundant than the total pair count suggests. **The paper's
bootstrap error bars are trustworthy; the "no bin beyond 2σ" statement survives
a degrees-of-freedom correction.**

(This also tempers the caveat in the Direction-2 report: the Calypso 2019 full-
$D_3$ excursions are not an error-underestimate artifact — they are a real
feature of that noisier, more anisotropic survey, and Calypso 2022 remains null.)

## 2. Δt sensitivity scan — validates the 10 hr window

Scanning the maximum pair time separation $\Delta t \in \{3,6,10,15,24\}$ hr:

- **Bias (max $|S_1|$):** for the focus Calypso 2022 the survey-mean $S_1$ is
  essentially **flat** in $\Delta t$ (~0.02–0.03 m/s); it is not the time window
  that sets the bias. Calypso 2019 rises modestly (0.060→0.084); ARCTERX stays
  high (~0.08–0.11).
- **Centered-$S_3$ significance:** here the window matters. At $\Delta t\le10$ hr
  **all** experiments have ≤1 centered-$S_3$ bin beyond 2σ. Increasing the
  window introduces *spurious* excursions — Calypso 2022 goes 0 (≤10 hr) → 1
  (15 hr) → 4 (24 hr); Calypso 2019 peaks at 3 (15 hr). Beyond ~10 hr the pairs
  are no longer synoptic: the flow evolves across the pair, so the increment
  conflates temporal change with spatial separation and biases even the centered
  moment.

**Recommendation:** $\Delta t = 10$ hr is near-optimal — long enough for
statistics (Calypso 2022: 31k pairs), short enough to keep the flow synoptic.
This answers the `TODO.txt` question ("Decide on Δt… more than 6"): 10 hr is a
good choice; do **not** push much beyond it, and 6–10 hr all give the same null.

## 3. Last-bin broadening

Merging the sparsely-sampled bins above 50 km into one wide bin:

| Experiment | merged last bin | standard-binning max \|z\| |
|---|---|---|
| Calypso 2019 | r~75 km, N=10491, z=−1.9 | 2.25 (r~94 km) |
| **Calypso 2022** | r~52 km, N=4402, z=1.8 | 1.84 (r~63 km) |
| ARCTERX 2023 | r~133 km, N=4259, z=2.6 | 1.28 (r~69 km) |

For Calypso 2022 the lone marginal bin the paper flags near r≈63 km sits at
z=1.84 here (below 2σ with 1000 bootstrap); **merging it away confirms no
significant large-r signal.** For ARCTERX the merged very-large-r bin
(r~133 km, spanning that ~300 km survey) reaches z=2.6 — worth a footnote but
based on a single, huge, sparsely populated bin.

## Bottom line for the paper

All three checks **strengthen the null result**: the bootstrap errors are
adequate (DOF correction negligible), Δt=10 hr is the right window (larger
windows manufacture false signals), and broadening the marginal large-r bin
removes the only near-2σ excursion for the focus experiment. These can be folded
into the Methods/Appendix as short robustness statements.
