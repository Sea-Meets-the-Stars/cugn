# Direction 2 — Full & transverse structure functions; isotropy check

**Date:** 2026-07-10
**Author:** JXP & Claude (Fable 5)
**Code:** `papers/Structure/Analysis/py/full_d3.py`, `figs_full_d3.py`
**Figures:** `Figures_new/fig_d2_LT_z60.png`, `Figures_new/fig_d3_full_z60.png`
**Data:** as in Direction 1 (60 m, $\Delta t\le10$ hr, linear bins, `minN=100`,
1000 bootstrap).

## Questions

The paper reports only the longitudinal third moment $D_{LLL}=\langle\delta
u_L^3\rangle$. The 2D-cascade literature (Lindborg 1999; Xie & Bühler 2019;
Balwada 2022; Gutiérrez 2025) uses the **full** third-order SF
$D_3 = \langle\delta u_L(\delta u_L^2+\delta u_T^2)\rangle = D_{LLL}+D_{LTT}$,
which carries the spectral kinetic-energy flux and can show the enstrophy branch
($\propto r^3$) that a longitudinal-only moment cannot. Two questions:

1. Is the 60 m flow consistent with **2D isotropy + incompressibility**?
2. Does the **full/centered $D_3$** reveal a cascade signal that the
   longitudinal-only moment misses?

## Method

Building the glider pairs once, `calc_delta` gives both $\delta u_L$ and
$\delta u_T$; I bin every second- and third-order combination. Crucially, the
third moments are **centered per bin**: I subtract the bin-mean increments
$\langle\delta u_L\rangle,\langle\delta u_T\rangle$ (forming fluctuations
$u',w'$) *before* cubing, which removes the homogeneity ($S_1$) bias for the
transverse and cross terms exactly as the algebraic correction does for the
longitudinal one.

**Validation.** The numerically-centered $D_{LLL,c}=\langle u'^3\rangle$ matches
the paper's algebraic $S_{3,c}=S_3-3S_1S_2+2S_1^3$ to machine precision
(max diff ≲ $4\times10^{-18}$) for all three experiments — both my centering and
the paper's correction are correct.

## Result 1 — the 60 m flow is isotropic (Calypso 2022)

For a homogeneous, isotropic, incompressible 2D field, the transverse and
longitudinal second-order SFs obey $D_{TT}(r)=D_{LL}(r)+r\,dD_{LL}/dr$. Testing
this (with $D_{LL}$ smoothed before differentiating):

| Experiment | $D_{TT}/D_{LL}$ median | $D_{TT}$ / isotropy prediction |
|---|---|---|
| Calypso 2019 | 1.17 | 0.79 |
| **Calypso 2022** | 1.20 | **0.96** |
| ARCTERX 2023 | 1.54 | 1.39 |

The focus experiment **Calypso 2022 satisfies the 2D incompressible-isotropy
relation to ~4%** — an independent validation of an assumption the paper's
theory relies on. Calypso 2019 and (especially) ARCTERX 2023 depart from it:
ARCTERX has $D_{TT}\gg D_{LL}$ (transverse-dominated), consistent with its strong
mean flow and more anisotropic sampling.

## Result 2 — the full centered $D_3$ is also null (Calypso 2022)

The **raw** full $D_3$ is large and positive-growing (10/14 bins > 2σ for
Calypso 2022, max $|z|=8$) — but this is entirely the $S_1$ bias, exactly as for
the raw longitudinal $S_3$. Once properly centered:

| Experiment | centered full $D_3$: bins > 2σ | max $|z|$ | median $|D_3^c|$ vs $|D_{LLL}^c|$ vs $|D_{LTT}^c|$ (m³/s³) |
|---|---|---|---|
| Calypso 2019 | 10 / 25 | 4.2 | 6.1e-4 / 3.0e-4 / 3.5e-4 |
| **Calypso 2022** | **1 / 14** | **2.7** | 4.2e-5 / 4.9e-5 / 4.3e-5 |
| ARCTERX 2023 | 3 / 14 | 4.2 | 2.0e-3 / 1.7e-3 / 1.1e-3 |

For **Calypso 2022 the centered full $D_3$ is consistent with zero** (1/14 bins
> 2σ): adding the transverse contribution $D_{LTT}$ does **not** rescue a
detection. The paper's longitudinal-only choice therefore loses nothing — the
null result is robust to using the full third-order structure function. No
enstrophy branch ($\propto r^3$) is visible either.

## A caveat that motivates the uncertainty work (Direction: robustness)

Calypso 2019 shows 10/25 centered-$D_3$ bins beyond 2σ — more than the ~1 in 20
expected by chance. This is suspicious: the bootstrap resamples *pairs*, but
pairs in a bin are **not independent** (repeated profiles along a track,
overlapping separations), so the bootstrap likely **under-estimates** the true
error and inflates the apparent significance. This is exactly the bootstrap-vs-
degrees-of-freedom question flagged in the Q&A, and it should be resolved before
reading any of these >2σ excursions as physical. (Calypso 2022, the best-sampled
and cleanest, is null regardless.)

## Suggested paper use

- Add the **isotropy check** as a short methods/appendix result: "At 60 m the
  Calypso 2022 second-order structure functions satisfy the 2D incompressible
  isotropy relation $D_{TT}=D_{LL}+r\,dD_{LL}/dr$ to within ~4%, supporting the
  isotropic-turbulence framework."
- Strengthen the null: "The full third-order structure function
  $D_3=D_{LLL}+D_{LTT}$, centered for homogeneity, is likewise consistent with
  zero for Calypso 2022; the transverse contribution does not alter the null."

## Files / reproducibility

`figs_full_d3.main()` regenerates both figures and prints the table numbers.
