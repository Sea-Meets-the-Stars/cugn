# Energy Cascade deep dive — summary of new analysis directions

**Date:** 2026-07-10 / 2026-07-11
**Author:** JXP & Claude (Fable 5)
**Scope:** New directions on the real glider data only (Calypso 2019, Calypso
2022 [focus], ARCTERX 2023); no QG; no notebooks; env `ocean14`.
**Code:** `papers/Structure/Analysis/py/` (modules below).
**Figures:** `papers/Structure/Analysis/Figures_new/`.
**Reports:** `papers/Structure/Analysis/claude_reports/01`–`08`.

Baseline reproduced first: the paper's Calypso 2022 $S_1/S_3$ (raw positive,
centered null, marginal r≈63 km bin) match exactly.

## The one-paragraph story

The paper's null (the centered velocity third moment $S_{3,c}$ is consistent
with zero) is **robust in every way I tested** — to the error model, the time
window, bin width, depth, sub-period, to using the full $D_3$ instead of the
longitudinal moment, and to trying to remove a mean flow. The reason it is a
*sampling* limitation and not featureless data is positively demonstrated: the
increments are weakly intermittent (fat tails), the focus survey is
quantitatively isotropic, and — the standout new result — **the gliders DO
detect a forward scalar-variance (temperature & salinity) cascade** even where
the velocity energy cascade is undetectable.

## Results by direction

| # | Direction | Verdict for the paper |
|---|---|---|
| 1 | **Mean-flow removal** (`mean_flow.py`) | A global affine/quadratic flow fit captures <15% of the velocity variance and does **not** reduce $S_1$ or reproduce $S_{3,c}$. The bias is not a removable coherent shear → the algebraic centering is the right tool. *Strengthens null.* |
| 2 | **Full/transverse $D_3$ + isotropy** (`full_d3.py`) | Calypso 2022 satisfies the 2D incompressible isotropy relation $D_{TT}=D_{LL}+r\,dD_{LL}/dr$ to ~4%. The centered **full** $D_3=D_{LLL}+D_{LTT}$ is also null (1/14 bins >2σ) — the transverse part rescues no detection. *Strengthens null + validates isotropy.* |
| R | **Robustness** (`robustness.py`) | Bootstrap error ≈ DOF error (inflation ~1.0–1.2×): the error bars are adequate. Δt=10 hr is near-optimal — larger windows manufacture spurious >2σ bins (synoptic assumption breaks). Broadening the marginal large-r bin removes the only near-2σ excursion. *Strengthens null; settles TODO choices.* |
| 3 | **Intermittency** (`intermittency.py`) | Flatness modestly >3 (fat tails), so the data sense real turbulence — not noise. ESS exponents near K41 $p/3$ (weak anomalous scaling). Signed skewness null (matches $S_3$). *Supports "sampling-limited, not noise."* |
| 4 | **Anisotropy** (`anisotropy.py`) | Calypso 2022 nearly isotropic (mean flow 0.009 m/s, only ~3% of the increment coherent); ARCTERX strongly anisotropic (mean flow 0.11 m/s, $D_{LL}$ anisotropy ~3, up to 22% coherent). Explains the differing $S_1$ biases. |
| 5 | **Scalar (T/S) SFs** (`scalar_sf.py`) | **NEW PHYSICS:** the mixed Yaglom moment $\langle\delta u_L(\delta\theta)^2\rangle$ is significantly **negative and grows with $r$** for T and S in all three surveys — a **forward scalar-variance cascade** detected where the velocity cascade is null. Order-of-magnitude $\chi_T\sim10^{-9}$–$10^{-7}$ °C²/s. |
| 6 | **Temporal evolution** (`temporal.py`) | The $S_1$ bias **grows through each campaign** (sampling worsens as gliders spread); the velocity cascade is null in every sub-window; the scalar cascade persists/strengthens. *Reinforces sampling thesis; design implication: shorter/denser or Lagrangian sampling.* |
| 7 | **Depth dependence** (`depth_dep.py`) | Energy and $S_1$ are surface-intensified; velocity cascade null at all depths (10–160 m); forward scalar cascade present throughout, peaking at mid-depth (30–100 m). 60 m is a sound choice. (Isopycnal check needs a density field not stored in these files.) |

## What I'd put in the paper

1. **The scalar cascade (Direction 5) is the most promising new result** — a
   genuine positive detection to set against the velocity null. Candidate new
   results subsection + figure (centered mixed moment for T and S).
2. **Isotropy validation (Direction 2)** and the **bootstrap/DOF + Δt
   robustness (R)** are ready-made Methods/Appendix hardening of the null.
3. **Mean-flow-removal (Direction 1)** and **temporal growth of $S_1$
   (Direction 6)** are Discussion points that reframe the null as an intrinsic,
   worsening sampling limitation — corroborating the QG argument from the data
   side.

## Highest-value follow-ups

- Mean scalar-gradient removal (analogue of Direction 1) to harden the scalar-
  cascade claim.
- Compute potential density (GSW from `t`,`s`,depth) to enable the isopycnal
  check.
- Fix the exact 2D Yaglom coefficient to turn the $\chi$ estimates quantitative.

## Reproducibility

Each `figs_*.py` module has a `main()` that regenerates its figures and prints
its numbers, e.g. from `papers/Structure/Analysis/py/`:
`conda run -n ocean14 python figs_scalar.py`. Analysis lives in the sibling
non-`figs_` modules.
