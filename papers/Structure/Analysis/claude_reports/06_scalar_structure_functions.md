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

---

## BGC extension — chlorophyll fluorescence (2026-08-31)

**Date:** 2026-08-31
**Author:** JXP & Claude (Fable 5)
**Code:** `papers/Structure/Analysis/py/bgc_sf.py`, `figs_bgc.py` (plus a small
additive extension of `glider_io.load_dataset` for extra profile-depth fields,
and a `dFl` branch added to the shared `profiler.profilerpairs.calc_delta`,
mirroring the existing `dT`/`dS` handling).
**Figures:** `Figures_new/fig_bgc_D2_z60.png`, `Figures_new/fig_bgc_mixedFl_z60.png`
**Data:** 60 m, $\Delta t\le10$ hr, linear bins, `minN=100`, 1000 bootstrap
(identical setup to the T/S analysis above).

### Data availability (checked before running anything)

This was requested as "dissolved oxygen and Chl," but the underlying `.mat`
CTD files were checked directly and neither is fully available:

- **Dissolved oxygen is absent from all three experiments.** Calypso 2019/2022
  and ARCTERX 2023 CTD files contain only CTD + ADCP-velocity fields
  (`s`, `t`, `udop`, `vdop`, ...) plus, for two of the three, `fl`/`abs`.
  `doxy` exists only in the unrelated CUGN-line NetCDF pipeline used by the
  main `cugn` package — a different set of glider missions on fixed CalCOFI
  lines, with a different survey/sampling geometry, not something that
  plugs into this pair-based SF machinery. **Per JXP: oxygen is skipped
  entirely for this direction.**
- **Chlorophyll coverage is partial.** A fluorescence field `fl` (a standard
  chlorophyll-a proxy) exists in Calypso 2019 and ARCTERX 2023, but **not**
  in Calypso 2022 — the paper's own focus survey has no BGC sensor recorded
  in its CTD file. **Per JXP: proceed with the two available experiments**
  (Calypso 2022 excluded, and called out as such).
- An additional optical field `abs` is present alongside `fl` in the same two
  files (value range ~51–82, likely a beam-transmission/attenuation
  percentage); its exact calibration/units were not confirmed, so it was left
  out of this pass rather than guessed at.

### Method

Identical to the T/S mixed (Yaglom) moment above: second-order SF
$D_{FlFl}(r)=\langle(\delta fl)^2\rangle$, and the raw and per-bin-*centered*
mixed third moment $\langle\delta u_L\,(\delta fl)^2\rangle$, with bootstrap
errors (1000 realizations).

### Headline result — unlike T/S, chlorophyll does *not* show a clean, single-signed cascade

| Experiment | centered mixed moment, bins beyond 2σ | pattern |
|---|---|---|
| Calypso 2019 | 14/25 (max \|z\|=5.2) | **negative** at small $r$ (≲45 km, weakly significant), **sign flips to strongly positive** at large $r$ (≳63 km, the most significant bins) |
| ARCTERX 2023 | 2/14 (max \|z\|=3.8) | one negative bin at the smallest $r$, one isolated positive spike at mid $r$ — no coherent trend |

This is a genuinely different result from temperature/salinity, not a weaker
version of the same thing. T/S gave a **monotonic, single-signed, growing**
mixed moment in all three surveys — the textbook signature of a
passively-advected scalar cascading forward. Chlorophyll fluorescence instead
shows a **scale-dependent sign reversal** in the one survey where it is well
resolved (Calypso 2019), and is statistically indistinguishable from a null
result in ARCTERX 2023.

![Mixed (Yaglom) third moment for chlorophyll fluorescence, raw vs. centered,
Calypso 2019 (left) and ARCTERX 2023 (right). Calypso 2019 shows a clean sign
reversal — weakly negative below ~45 km, then a smooth, highly significant
rise to positive values above ~63 km. ARCTERX 2023 scatters around zero with
no coherent trend.](../Figures_new/fig_bgc_mixedFl_z60.png)

![Second-order chlorophyll-fluorescence structure function
$D_{FlFl}(r)=\langle(\delta fl)^2\rangle$ for the two available
experiments.](../Figures_new/fig_bgc_D2_z60.png)

### Interpretation

Chlorophyll is not a passively conserved tracer the way temperature and
salinity are: photosynthesis, grazing, sinking, and patch formation act as
scale- and time-dependent sources and sinks superimposed on any stirring by
the flow. A single-signed Yaglom relation assumes a materially conserved
scalar with a steady, scale-independent variance production; chlorophyll has
no reason to obey that assumption, and the data bear this out — the sign
reversal at ~50–60 km in Calypso 2019 is consistent with biology (not
turbulent stirring alone) setting the fluorescence field's small-vs-large-scale
structure. Given this, **no transfer-rate ($\chi_{Chl}$) estimate is reported**
here — fitting a single "$-2\chi r$" line through a moment that changes sign
would misrepresent the result; unlike T/S, this is not a case where a
noise-limited but single-signed trend can be described by one slope.

The comparison is itself informative for the paper: **the platforms can
resolve a genuine, high-significance third-moment signal in a biological
tracer, but that signal has different structure than a physically conserved
one** — supporting evidence that the clean T/S cascade detections above are a
real conservative-tracer effect and not a generic artifact of the pair/binning
method (a generic artifact would presumably show up the same way in
chlorophyll too).

### Caveats

- ARCTERX 2023 has far fewer pairs per bin here (~140–330 vs. ~500–2000+ for
  Calypso 2019 T/S), so its near-null result carries less statistical power
  than Calypso 2019's — it should be read as "not resolved," not as
  "confirmed absent."
- Calypso 2022, the paper's own focus survey, has no BGC data at all in this
  source file; if oxygen/chlorophyll is wanted for the focus survey
  specifically, that requires locating/obtaining a BGC-equipped CTD file for
  that mission (not currently in the repo) — flagged as a follow-up, not
  attempted here.
- `abs` (present alongside `fl`) was left unanalyzed pending confirmation of
  what it actually measures.

### Suggested paper use

Not a new-physics headline the way T/S is, but a useful **contrast/robustness
point**: cite it (briefly, e.g. in a footnote or appendix) as evidence that
the T/S mixed-moment detection is tracer-specific (conservative vs.
biologically active), not a generic property of the estimator.

### Follow-up worth doing

- Locate a BGC-equipped CTD source for Calypso 2022 so the focus survey can be
  included.
- Confirm what `abs` measures and analyze it alongside `fl`.
- If oxygen is ever wanted, this would need a CUGN-line-based (not
  Calypso/ARCTERX) version of the pair pipeline — a materially different
  dataset/geometry, out of scope here.
