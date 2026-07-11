# Direction 4 — Anisotropy and orientation dependence

**Date:** 2026-07-10
**Author:** JXP & Claude (Fable 5)
**Code:** `papers/Structure/Analysis/py/anisotropy.py`, `figs_anisotropy.py`
**Figures:** `Figures_new/fig_aniso_DLL_z60.png`, `fig_aniso_S1_z60.png`,
`fig_aniso_coherent_z60.png`
**Data:** 60 m, $\Delta t\le10$ hr, linear bins, `minN=100`, 4 orientation
sectors over 180°, orientation measured relative to the survey-mean-flow
direction.

## Question

The structure-function relations assume isotropy (orientation-averaged
statistics). A finite survey samples separations anisotropically, and a mean
flow selects a direction. How anisotropic is each survey, and how much of the
$S_1$ bias is a coherent directional flow versus incoherent turbulence?

## Method

Resolve pairs by the orientation of the separation vector (in 4 sectors,
relative to the bulk mean-flow direction), and measure per sector the
second-order SF $D_{LL}(r)$ (energy anisotropy) and the mean increment
$S_1(r)$. Separately, form the **coherent mean velocity-difference vector**
$\langle\delta\mathbf{u}\rangle(r)$ in the fixed E/N frame; its magnitude
relative to the RMS increment, $|\langle\delta\mathbf{u}\rangle|/u_{\rm rms}$,
is the fraction of the increment that is a coherent directional flow (removable)
versus turbulence (not).

## Results

| Experiment | bulk mean flow | $D_{LL}$ anisotropy (max/min over sectors) | coherent fraction (median / max) |
|---|---|---|---|
| Calypso 2019 | 0.053 m/s | 1.32 | 0.05 / 0.16 |
| **Calypso 2022** | **0.009 m/s** | **1.37** | **0.03 / 0.10** |
| ARCTERX 2023 | 0.111 m/s | **2.92** | 0.12 / 0.22 |

Two contrasting regimes emerge:

- **Calypso 2022 (focus) is nearly isotropic.** The bulk mean flow is tiny
  (0.009 m/s), the second-order SF varies by only ~37% across orientation, and
  just ~3% of the increment is coherent — 97% is incoherent turbulence. This is
  fully consistent with the 2D-isotropy relation it satisfied in Direction 2
  and with the failure of mean-flow removal in Direction 1: there is no strong
  directional flow to blame or remove.

- **ARCTERX 2023 is strongly anisotropic.** The energy is ~3× larger in one
  orientation than the perpendicular one, the mean flow is an order of
  magnitude stronger (0.11 m/s), and up to 22% of the increment is coherent.
  This directional coherent flow is the origin of ARCTERX's large $S_1$ bias and
  its $D_{TT}\gg D_{LL}$ (Direction 2), and it is why mean-flow removal moved its
  $S_1$ at all (Direction 1) while doing nothing for Calypso 2022.

The coherent fraction **grows with separation** for Calypso 2022 (~0.02 at a few
km to ~0.10 by 60 km): the directional/large-scale part of the increment becomes
relatively more important at larger $r$, which is exactly why $S_1$ (and hence
the raw-$S_3$ bias) rises with $r$ — but even at 60 km it remains a small
minority of the signal.

## Interpretation

Anisotropy cleanly separates the three surveys and explains their differing
$S_1$ biases:

- Where the flow is isotropic and the mean flow negligible (Calypso 2022), the
  $S_1$ bias is an intrinsic eddy-sampling effect (small coherent fraction),
  not a directional current — reinforcing Directions 1 and 2.
- Where a strong mean flow exists (ARCTERX), the survey is markedly anisotropic
  and a larger (but still <25%) fraction of the increment is coherent; this is
  the regime where a directional correction has the most leverage, yet even
  there the turbulent part dominates.

## Suggested paper use

- Report the isotropy of the focus experiment quantitatively: "Calypso 2022 is
  nearly isotropic at 60 m — the second-order SF varies by <40% with
  orientation and only ~3% of the velocity increment is a coherent directional
  flow — justifying the orientation-averaged, isotropic framework."
- Use the coherent fraction to characterize the contrast with ARCTERX, whose
  strong, anisotropic mean flow explains its larger $S_1$ bias.

## Caveats

- Four sectors is a coarse resolution; the anisotropy ratio is a summary, not a
  full angular spectrum.
- ARCTERX's large-$r$ sectors are sparsely populated (its survey spans ~300 km),
  so its high-$r$ anisotropy/coherent values are noisier.
