# Direction 3 — Intermittency and higher-order statistics

**Date:** 2026-07-10
**Author:** JXP & Claude (Fable 5)
**Code:** `papers/Structure/Analysis/py/intermittency.py`, `figs_intermittency.py`
**Figures:** `Figures_new/fig_interm_skewflat_z60.png`,
`fig_interm_pdfs_Calypso2022_z60.png`, `fig_interm_ess_z60.png`
**Data:** 60 m, $\Delta t\le10$ hr, linear bins, `minN=100`, 1000 bootstrap.

## Question

The paper's null is about the *signed* third moment (the energy flux). But does
the glider data sense **real turbulent structure** at all? A turbulent velocity
field is intermittent: the increment PDF develops fat, non-Gaussian tails at
small separations (flatness > 3), and high-order structure functions scale with
anomalous exponents. If the increments were merely instrumental noise, they
would be Gaussian and non-intermittent. This is a positive counterpoint to the
null.

## Method

Per separation bin I center the increment ($u'=\delta u_L-\langle\delta
u_L\rangle$) and compute the skewness $\langle u'^3\rangle/\langle
u'^2\rangle^{3/2}$, the flatness (kurtosis) $\langle u'^4\rangle/\langle
u'^2\rangle^2$, the increment PDFs, and the absolute structure functions
$S_p=\langle|u'|^p\rangle$ for $p=1..6$. The extended-self-similarity (ESS)
exponents $\zeta_p/\zeta_3$ come from fitting $\log S_p$ vs $\log S_3$.

## Results

**Skewness — consistent with the null.** Away from one bin, the skewness
scatters around zero (Calypso 2022 range excluding the outlier: roughly
$-0.05$ to $+0.1$), consistent with the centered-$S_3$ null. The lone exception
is the recurring sparse bin at r≈63 km (skewness ≈ 0.8 ± 0.3) — the same
low-$N$ bin flagged in the robustness report; not significant.

**Flatness — mildly super-Gaussian (fat tails).** The flatness sits modestly
above the Gaussian value of 3 across most bins:

| Experiment | flatness range (good bins) |
|---|---|
| Calypso 2019 | 2.7 – 3.9 |
| **Calypso 2022** | 2.9 – 4.7 (4.7 is the r≈63 km outlier; ~3.2–3.7 elsewhere) |
| ARCTERX 2023 | 2.8 – 3.9 |

The increment PDFs (Calypso 2022) lie slightly above the Gaussian in the
±2–3σ tails at all shown separations. So the data **do** capture weakly
non-Gaussian, intermittent turbulence — not instrumental noise — but the
intermittency is modest over the resolved 4–70 km range.

**ESS exponents — close to self-similar (K41).** For the two well-sampled
Calypso surveys, $\zeta_p/\zeta_3$ tracks the non-intermittent K41 line $p/3$
almost exactly:

| p | 1 | 2 | 3 | 4 | 5 | 6 | K41 (p/3) |
|---|---|---|---|---|---|---|---|
| Calypso 2022 | 0.31 | 0.65 | 1.00 | 1.36 | 1.71 | 2.05 | 0.33/0.67/1/1.33/1.67/2 |
| Calypso 2019 | 0.32 | 0.66 | 1.00 | 1.34 | 1.68 | 2.02 | — |
| ARCTERX 2023 | 0.24 | 0.58 | 1.00 | 1.49 | 2.03 | 2.61 | — |

The Calypso surveys show only slight anomalous scaling (deviations ≤0.05 up to
6th order). ARCTERX deviates more strongly at high order ($\zeta_6/\zeta_3=2.6$
vs 2.0), i.e. more intermittent — but it is also the sparsest survey, so the
high-order moments are the least reliable.

## Interpretation

- The glider increments are **weakly intermittent**: fat-tailed PDFs and
  flatness above 3 confirm the measurement senses genuine turbulence, which
  rules out the trivial explanation that the null $S_3$ reflects featureless
  noise.
- The intermittency is **mild and the scaling near self-similar** over the
  narrow resolved range (4–70 km), so there is no strong anomalous-scaling
  signal to exploit.
- Consistent with the rest of the analysis: the *symmetric* structure (variance,
  flatness) is well measured, while the *signed* structure (skewness / $S_3$,
  the cascade direction) is null — the platforms resolve turbulent intensity but
  not the flux.

## Suggested paper use

A short paragraph strengthening the interpretation of the null: "The increment
PDFs are weakly non-Gaussian (flatness ≈ 3.2–3.7, fat tails at ±2–3σ) and the
ESS exponents are close to $p/3$, confirming the surveys resolve genuine,
mildly intermittent turbulence. The null therefore concerns the *signed* flux,
not the presence of turbulent structure."

## Caveats

- The r≈63 km Calypso 2022 bin recurs as an outlier (skewness/flatness spike);
  it is sparsely sampled and not significant.
- High-order ($p\ge5$) moments are sensitive to the largest increments and are
  least reliable for the sparse ARCTERX survey.
