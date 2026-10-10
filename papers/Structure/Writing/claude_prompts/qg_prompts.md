# Write the QG sections of the paper

## Context

### High-level

You are helping me and Dan Rudnick to write a paper about the structure function of the velocity field to be submitted to the Journal of Physical Oceanography.  

### Writing style

See the PDFs in cugn/papers/Structure/Writing/previous/ for several of my previous publications so that you learn my writing style.

### Gutierrez-Villanueva manuscript

Read the Gutierrez-Villanueva manuscript `Gutierrez-Villanueva2025.pdf` in Writing/.  They describe in detali the QG model that we are using.

### Code

See all of the files in the cugn/papers/Structure/ and the calls to the main Repo and other dependencies. 

### Overleaf

The git repo for this paper may be found here:
/home/xavier/Projects/Overleaf/Energy_Cascade

### Outline

Our outline is in the cugn/papers/Structure/Writing/ folder.  It is named outline.md.

## Writing

1. Read all of the files mentioned in Context in preparation of writing the sections on QG.  We will be starting with the theory section.  Ask me questions in the Q&A/Theory section. Log your work.

2. I have answered your questions in the Q&A/Theory section.  Please read them and let me know if you have any more.  If so add them to the Q&A/Theory section below.  Log your work.

3. I have answered your questions in the Q&A/Theory section.  Read them and then proceed to write the theory section for QG.  Log your work.

4. That section is excellent.  Let us now prepare the Results section for QG.  Ask me questions in the Q&A/Results section. Log your work.

5. I have answered your questions in the Q&A/Results section.  Read them and ask more questions if needed.  Log your work.

6. I have answered your questions in the Q&A/Results section.  Proceed to making figures and writing the text.  When you are dowe writing, push your changes to Overleaf.  Log your work.

7.  Going back a bit, please:
    - Write a figure caption for Figure 2
    - Move the Nmin=100 text on Figure 3 to the left (off the histogram)
    - Push Figure 4 into the Appendix
    - In the current Figure 8, the ARCTERX 2023 figure extends to too large of r-values.  Limit to 100km

## Q&A

### Theory

Questions before writing the QG theory subsection. Each has my recommended
default in **bold** — feel free to just confirm or correct inline (as in
`karman.md`).

**Q1 — Placement of the QG material.** The outline asks "QG analysis – here
[Theory] or Discussion?" and notes "Methods/outputs here, but limited, if any
results." The current draft has §2 "Theoretical underpinnings" (2D limit,
\kh\ equation, homogeneity) but no QG subsection. My plan: add a concise
**QG-model description as a new subsection at the end of §2 Theory** (e.g.
"A quasi-geostrophic reference field"), right after the 2D/K-H material it
builds on, and defer *all* QG results (full-box SF, subregion-vs-full, time
evolution, glider/drifter sampling) to §4 Results. The alternative is to put
the model description in §3 "Observations and Methods," next to the
observational methods. Default: **new subsection at the end of §2 Theory.**
Agree?
> A: Agree

**Q2 — Role / framing of the QG model.** I read the QG model as a controlled
2D turbulent **"ground truth"**: a statistically-steady field with a *known*
inverse energy cascade whose structure functions can be computed exactly over
the full 1000-km domain, against which we test whether realistic sampling (a
finite box, limited duration, few platforms) recovers the true sign and slope
of \Sn{3}. That makes it the bridge to the paper's thesis that the cascade
sign is hard to measure. Is that the intended role, or do you want it framed
differently (e.g. as a quantitative model of the California Current rather
than an idealized testbed)? Default: **idealized testbed / ground truth.**
> A: Yes, that is the intended role.

**Q3 — How much model setup, and where.** I propose a compact main-text
paragraph: a **two-layer PyQG simulation** on a 1000 km doubly-periodic
domain ($256^2$, $\Delta x \approx 3.9$ km), deformation radius $R_d=15$ km
(wavelength $L_d \approx 92$ km), $\beta=1.5\times10^{-11}$ m$^{-1}$s$^{-1}$,
bottom drag $r_{ek}=5.79\times10^{-7}$ s$^{-1}$, shear-forced
($U_1=0.025$ m/s, $U_2=0$), upper layer only, daily output over ~15 yr after
spin-up — citing pyqg, Ross et al. (2023) for the configuration, and
Gutierrez-Villanueva et al. (2025) as the immediate source. (I'll add these +
Xie–Bühler 2019 and Balwada et al. 2022 to `energy.bib` as needed.) Is this
the right level of detail for the theory section, or should the full setup go
to an appendix with a one-line summary in the main text? Default: **compact
paragraph in main text.**
> A: Yes, that is the right level of detail for the theory section.  But make sure we enunciate the motivations analyzing the model.

**Q4 — Is our QG output the GV run?** Our pipeline parameters match the
Gutierrez-Villanueva et al. (2025) configuration *exactly* (same domain,
$R_d$, $\beta$, $r_{ek}$, 20-yr run). Should I describe the field as **the GV
model output itself** (we analyze their simulation) or as an
**independently-run simulation with the same configuration**? This decides
whether GV is cited as the data source or only as the configuration
reference. Default: **state it as the same configuration and cite GV; correct
me if it is literally their output file.**
> A: Yes, we are analyzing the GV run.

**Q5 — Longitudinal-only vs. the full Xie–Bühler $D_3$.** The glider analysis
(and §2b) uses the purely longitudinal \Sn{3}. The QG code (and
GV/Balwada) also computes the full $D_3 = \langle\duL(\duL^2+\duT^2)\rangle =
D_{LLL}+D_{LTT}$ and maps it to the 2D cross-scale KE flux $F(k)$ via the
Xie & Bühler (2019) Hankel transform. For internal consistency I lean toward
presenting the QG results in the **same longitudinal \Sn{3} used throughout**,
mentioning the full-$D_3$ / spectral-flux framework only briefly to connect to
GV. Do you want (a) **longitudinal-only, with a brief mention of $D_3$/$F(k)$**,
(b) longitudinal-only and no $F(k)$ at all, or (c) develop the full Xie–Bühler
$D_3$/$F(k)$ framework as part of the QG theory? Default: **(a).**
> A:  We will go with (a)

**Q6 — State the model's "true" cascade, and how quantitatively.** Since the
value of the testbed is that the answer is known, I plan to state the expected
cascade explicitly: an upscale (inverse) energy cascade above the
forcing/deformation scale (positive, ~linear \Sn{3}) and a forward enstrophy
cascade below. Should I quote a specific energy-injection rate \eflux\ and
forcing scale here (do you have a preferred value/source?), or keep the theory
section qualitative (sign + slope only) and reserve any number for Results?
Default: **qualitative in Theory; quantitative \eflux\ reserved for Results.**
> A:  Actually, we find in our analysis of the longitudinal-only SF that there is no forward enstrophy cascade.  This should be discussed in the theory section.  It is possible this is because GV had a bug in their analysis code.

**Q7 — Sampling experiments (Eulerian / drifters / gliders).** The code
supports three samplings of the QG field: the exact **Eulerian** full-grid SF,
virtual **drifters** (Lagrangian), and virtual **gliders** (fixed samplers);
the outline lists "Analyze as gliders/drifters." Should the theory subsection
*introduce* these three sampling modes as the design of the QG test (with the
quantitative comparison in Results), or describe only the model + exact field
in Theory and leave all sampling description to Methods/Results? Default:
**theory subsection describes the model + exact field only; sampling modes
introduced in Methods/Results.**
> A: Use the default.

**Q8 — Notation.** I'll reuse the existing macros (`\duL`, `\Sn{n}`,
`\Snc{n}`) for the QG quantities so the QG and observational structure
functions are directly comparable, mapping the code's `du1`/`du2`/`du3` and
`du3_corr` onto \Sn{1}, \Sn{2}, \Sn{3}, \Snc{3} (the QG figures already label
axes $\langle\delta u_L\rangle$, etc.). OK? Default: **yes, reuse existing
macros.**
> A: Yes, reuse existing macros.

---

Follow-up after your answers (2026-06-28). Two questions, both prompted by your
Q6 answer (no forward enstrophy cascade; possible GV bug):

**Q9 — The enstrophy null result: how to frame it.** From the full-box SF
(`fig_full_qg_SF.png`) I read: \Sn{1} $\approx 0$ (homogeneity holds;
$\sim10^{-6}$ \mps), \Sn{2} rises to a plateau $\sim5.5\times10^{-3}$ \mpss\
near $\sim100$--$150$ km, and \Sn{3} is **positive at all resolved separations
($\approx7$--$200$ km)**, rising to a peak $\sim2.6\times10^{-5}$ \mpsss\ at the
deformation wavelength $L_d\approx92$ km and declining beyond --- i.e. a clean
**inverse energy cascade** (positive, $\sim$linear over $\sim15$--$92$ km) with
**no distinct $+r^3$ enstrophy range**. Is that the finding you want the theory
section to state? Before attributing the missing enstrophy range to a GV code
bug, I want to flag two more conservative explanations, because (per the
project's "be critical" guidance) a reviewer will raise them:
- (i) **Different diagnostic.** GV (and Lindborg/Bernard) state the enstrophy
  relation for the *full* $D_3=\langle\duL(\duL^2+\duT^2)\rangle$; we use
  longitudinal-only \Sn{3}. The two differ, so a clean enstrophy signature need
  not appear in \Sn{3} even if present in $D_3$.
- (ii) **Limited scale separation.** The enstrophy range lies below the
  injection scale ($\sim R_d=15$ km); our smallest reliable separations are
  $\sim7$ km, leaving essentially no resolved enstrophy subrange regardless of
  any bug.

  My recommendation: **state our inverse-energy finding and the absence of a
  resolved enstrophy range, attribute it primarily to (i)+(ii), and treat the
  GV discrepancy as an open question** rather than asserting a bug. Which do you
  want emphasized --- (i), (ii), a genuine GV bug, or a combination?
> A: Yes, I agree. Please do so.

**Q10 — The GV "bug" claim: tone and placement.** Asserting that
Gutierrez-Villanueva et al. (2025) have an analysis bug is a strong public
claim about a colleague's work. Have you (or Manuel) confirmed the discrepancy
by comparing our *full* $D_3$ against theirs (our pipeline computes both the
longitudinal-only `*_duL.nc` and the full `*_new.nc`), or is the bug so far
only a hypothesis from the longitudinal-only mismatch? My recommendation is to
**keep the theory section to our finding plus the (i)/(ii) caveats, and defer
any reconciliation with GV --- including a cautiously-worded bug possibility
--- to the Discussion**, rather than asserting it as fact in §2. Agree, or do
you want the bug possibility named explicitly in the theory section?
> A: Yes, I agree. Let us not raise the bug at this point.

### Results

Questions before drafting the QG Results subsection. As before, each has my
recommended default in **bold** — confirm or correct inline.

**R1 — Placement & title.** I'll add the QG results as a new subsection of §4
Results, after the three observational experiments (§4a \focusexp; §4b Calypso
2019 / ARCTERX 2023), e.g. §4c "A quasi-geostrophic testbed" or "Sampling the
quasi-geostrophic field." Confirm placement and any preferred title. Default:
**new §4c after the observational results.**
> A: Correct

**R2 — The central message / story arc.** I read the QG results as a
three-step argument: (1) the full-domain field has a clear positive
(inverse-energy) \Sn{3} [established in §2d]; (2) restricting to a survey-sized
($\sim100$ km) region introduces a nonzero mean \Sn{1} that biases the raw
\Sn{3}, while the *centered* \Snc{3} recovers the true positive signal **but
only with multi-year averaging** (Fig.\ `fig_qg_100km_vs_full`, which uses 5
yr); (3) realistic glider/drifter sampling ($\sim100$ days, few platforms)
cannot beat down the mean-flow bias in that time and so fails to recover the
sign --- which *explains the observational null* of §4a/b. The QG testbed thus
shows that it is the **sampling (duration $\times$ platform count), not the
region size per se**, that defeats the measurement. Is that the arc you want?
This also resolves the apparent tension (QG centering recovers the signal; the
real data are null) via averaging time. Default: **yes, that arc.**
> A: Yes, that is the arc we want.

**R3 — Region size & position: single or ensemble.** The pipeline has many
boxes (100/200/300/500/1000 km) at several positions, and some positions yield
a strongly *negative* mean \Sn{1}. For the headline subregion-vs-full figure do
you want a single representative 100 km box (the current figure uses
$x{=}300,y{=}400$ km), or to show the *spread* across positions/sizes to make
the bias point robustly? Default: **lead with one representative 100 km box
(survey scale); note the spread across positions/sizes in the text.**
> A: Use your default.

**R4 — Time-evolution / convergence figure.** Include the averaging-time
convergence figure (\Sn{1} and/or \Sn{3} averaged over 1, 60, 180, 365, … days;
`fig_qg_duL_vs_time`)? It directly quantifies *how much observing time* is
needed to recover the cascade and feeds the Discussion. Default: **yes, include
a time-evolution panel for \Sn{3} (and \Sn{1}), and use it to estimate the
averaging time required.**
> A: Yes, include the time-evolution panel for \Sn{3} (and \Sn{1}), and use it to estimate the averaging time required.

**R5 — The glider/drifter sampling experiment (the payoff).** Outputs already
exist (Eulerian, drifter, glider SF) for a 100 km box over 100 days at two QG
start times (ts=5001, 6001). Do you want a figure contrasting the *true*
(full-box / Eulerian) \Sn{3} with the drifter- and glider-sampled estimates, to
show explicitly that realistic sampling does not recover the sign? And do you
want both drifters and gliders, or gliders only (the paper is glider-focused)?
Default: **include the sampling comparison (it is the payoff for "humans will
not measure the sign"); show both drifters and gliders.**
> A: Use your default.

**R6 — Should I generate the needed figures?** Only two QG figures are
currently in the paper (full grid §2d; 100 km vs full). The time-evolution (R4)
and glider/drifter-sampling (R5) figures are not yet final paper PNGs. Shall I
generate them from the existing analysis outputs (the sampling JSONs are in
`Analysis/Output/`; the region/QG SF files need `$OS_DATA`), styled like the
existing paper figures, and add them to the Overleaf `Figures/` folder? Default:
**yes, I generate them.**
> A: Yes, generate them.

**R7 — Quantitative \eflux.** We deferred any number to Results (Theory Q6).
Do you want the Results section to quote a quantitative upscale energy-injection
rate \eflux\ from the inverse-energy relation $\Sn{3}=+\tfrac32\eflux r$ (slope
fit to the full-box positive branch), as a concrete target that the sampled
estimates miss? Default: **yes, quote \eflux\ from the full-box slope and
compare against the sampled estimates.**
> A: Yes, quote \eflux\ from the full-box slope and compare against the sampled estimates.

**R8 — $N_{\rm eddies}$ vs.\ time?** The outline lists "QG: $N_{\rm eddies}$
vs.\ time?" with a question mark. Include it (e.g.\ a measure of the number of
independent eddies / decorrelation time, to characterize the statistical
independence of the sampling), or drop it for now? Default: **drop unless you
have a specific use; it is not essential to the cascade-sign argument.**
> A: Hold off for now.

---

Follow-up after your R-answers (2026-06-29). Before drafting §4c I checked the
sampling outputs against the in-box truth, and the numbers refine the R2 arc:

I computed the centered \Snc{3} for the three samplings of the 100 km box over
100 days, at both available QG start times, and counted how many separation
bins ($5$–$70$ km) come out positive (the true sign):

| start time | Eulerian (dense, all grid pairs) | drifter (121) | glider (10) |
|---|---|---|---|
| ts=5001 | 13/13 bins $>0$ ✓ | 3/8 ✗ | 4/9 ✗ |
| ts=6001 | 11/13 ✓ | 8/8 ✓ | 8/9 ✓ |

**R9 — The sampling result is realization-dependent; this refines R2.** Two
points from the table contradict the arc as stated:
- (a) **Duration is not the barrier for dense sampling.** The Eulerian (every
  grid point, every day) in-box estimate recovers a positive \Snc{3} in *100
  days*, not multi-year. So "the centered moment recovers only with multi-year
  averaging" is true for *sparse track-based accumulation*, not for sampling
  density in general.
- (b) **Realistic sampling is realization-dependent.** The glider (10) and
  drifter (121) recover the correct positive sign at ts=6001 but *fail* (sign
  scattered, $\sim$half the bins negative) at ts=5001. And platform count
  (10 vs 121) matters far less than *which* 100-day window is sampled.

So the cleaner, more defensible message is: the underlying field has an
unambiguous inverse cascade, but with a realistic $\sim$100-day glider survey
the *recovered sign is unreliable --- it depends on the particular eddy
realization sampled* --- whereas complete spatial sampling (Eulerian) is robust.
This still supports "humans will not measure the sign," but via sampling
sparsity + realization variance rather than averaging time alone. Do you want
§4c framed this way? Default: **yes, reframe around realization-dependence /
sampling sparsity, keeping the Eulerian-vs-sparse contrast central.**
> A: Use the default.

**R10 — Build an ensemble of realizations?** The realization-dependence in R9
currently rests on only *two* start times. To claim it robustly (and to make a
much stronger figure than a single 3-curve comparison), I'd run the
glider/drifter/Eulerian sampling pipeline for an ensemble of start-time windows
(say 10–20, optionally also a few box positions) and show the *distribution* of
the recovered \Snc{3} sign/slope --- e.g.\ a histogram of recovered slopes with
the true value marked, or the fraction of realizations recovering the correct
sign. This needs the Julia+Python sampling pipeline (`qg_gliders`,
`glider_analysis`) run against the full QG output (present in `$OS_DATA`). Do
you want me to generate this ensemble, or keep the single-window
(ts5001/ts6001) comparison for now? Default: **run a modest ensemble
($\sim$12 windows) for the headline sampling figure, if the pipeline runs
cleanly here; fall back to the two existing windows otherwise.**
> A: Let's defer this for now.  Please remind me to consider it again later

## Logging

Append an entry to the **Logs** section of this file using the format:

```
### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>
```

## Logs

### 2026-06-28 (Read all QG Context files; posted 8 questions for the QG theory section)

Completed item 1 under **Writing**: read every file referenced in Context in
preparation for the QG sections, then posted questions in **Q&A/Theory**. Did
not write any LaTeX yet (the prompt asks only to prepare and ask).

**What I read and learned:**

- **Prior logs (`prep.md`, `karman.md`).** The §2 "Theoretical underpinnings"
  / Kármán–Howarth subsections are already written and pushed to Overleaf
  (commits up through `d03b43a`), along with a 23-equation appendix derivation.
  Established conventions I will reuse: macros `\duL`, `\Sn{n}`, `\Snc{n}`,
  `\eflux` (\eflux), `\zflux` (\eflux→enstrophy \zflux); the three signed
  third-order relations (2D inverse energy $+\tfrac32\eflux r$, 2D enstrophy
  $+\tfrac18\zflux r^3$, 3D four-fifths $-\tfrac45\eflux r$); and the framing
  of the **sign + slope of \Sn{3}** as the central measurement.

- **Current `energy_cascade.tex`.** The paper has advanced well past the K-H
  logs: §3 Methods (Data, Structure functions) and §4 Results (Calypso 2022;
  Calypso 2019 + ARCTERX 2023) are drafted. Key result so far: the **centered**
  third moment \Snc{3} is consistent with zero in all three experiments — the
  apparent positive raw $S_3$ is an artifact of survey-scale inhomogeneity
  (nonzero \Sn{1}). There is **no QG subsection yet** in either §2 or §4.

- **Gutierrez-Villanueva et al. (2025)** (`Gutierrez-Villanueva2025.pdf`, JTECH
  submission). Describes the QG model we use: a **two-layer PyQG** simulation,
  1000-km doubly-periodic domain, $256^2$ ($\Delta x\approx3.9$ km), $R_d=15$
  km ($L_d\approx92$ km), $\beta=1.5\times10^{-11}$, $r_{ek}=5.79\times10^{-7}$,
  shear-forced ($U_1=0.025$, $U_2=0$; $H_1=500$ m, $H_2=2000$ m), spun up 5 yr
  + run 15 yr, daily-averaged, upper layer only — **identical to our pipeline
  parameters** (see `drift_in_qg.md`). GV's third-order framework is the full
  $D_3=D_{LLL}+D_{LTT}=\langle\duL(\duL^2+\duT^2)\rangle$ (Balwada et al. 2022;
  Xie & Bühler 2019), related to the azimuthal 2D KE flux $F(k)$ by a Hankel
  transform; $F(k)<0$ ⇒ inverse (upscale) transfer. They invert $D_3$ for KE
  injection rates / spectral flux via regularized least-squares.

- **QG analysis code** (`Analysis/py/qg_utils.py`, `qg_uL_SF.py`,
  `qg_100km.py`; `Analysis/QG/*.md`). Pipeline: raw $\duL/\duT$ per daily field
  → `du2`,`du3` → spatial average over all base points → orientation average
  into 5-km $r$ bins → 5-yr time series of SF. Computes **both** a
  longitudinal-only SF (`*_duL.nc`, via `SF2_3_ul`) and the full
  longitudinal+transverse SF (`*_new.nc`, via `SF2_3`). The third-order
  cumulant correction matches the paper's: `du3_corr = du3 - 3*du1*du2 +
  2*du1**3`. `qg_100km.py` extracts subregions (100/200/300/500/1000 km) to
  contrast a small "survey-sized" box against the full domain. Three samplings
  of the field exist: exact **Eulerian** grid, virtual **drifters**
  (Lagrangian, `DrifterData`), and virtual **gliders** (fixed samplers,
  `SprayData.from_QG_glider`).

- **Figures already produced** (`Figures/py/figs_paper_structure.py`):
  `fig_full_qg_SF.png` (full-box \Sn{1}, \Sn{2}, \Sn{3} vs $r$, with $L_d=92$ km
  marked) and `fig_qg_100km_vs_full.png` (subregion vs full box, incl. raw and
  corrected $S_3$). These map directly onto the outline's QG figure list.

- **Outline + writing style.** Outline flags the QG model for the Theory
  section ("Methods/outputs here, but limited, if any results") and QG results
  for §4. Writing style (from prior logs on `BING.pdf`, `Nenya.pdf`):
  first-person plural, equation-forward, physically motivated, clear question
  framing.

**Questions posted (Q1–Q8 in Q&A/Theory):** placement of the QG subsection
(default: end of §2 Theory); framing as an idealized ground-truth testbed;
level of model-setup detail + citations; whether our output *is* the GV run or
a same-config rerun; longitudinal-only \Sn{3} vs. the full Xie–Bühler
$D_3$/$F(k)$ framework (default: longitudinal-only with brief mention);
qualitative vs. quantitative statement of the model's true cascade; whether to
introduce the three sampling modes in Theory or defer to Methods/Results; and
notation reuse. Awaiting answers before drafting the subsection.

### 2026-06-28 (Read Q1–Q8 answers; posted two follow-ups Q9–Q10)

Completed item 2 under **Writing**: read the answers to Q1–Q8 and posted two
follow-up questions (Q9, Q10) before drafting.

**Answers absorbed (Q1–Q8).** All defaults confirmed except Q6, plus one
emphasis on Q3:
- Q1: QG description = new subsection at the **end of §2 Theory**.
- Q2: framing = **idealized testbed / ground truth**.
- Q3: compact main-text paragraph — **but explicitly enunciate the motivations
  for analyzing the model** (added requirement).
- Q4: we are analyzing the **GV run itself** → cite GV as the data source.
- Q5: **(a)** longitudinal-only \Sn{3}, brief mention of the full $D_3$/$F(k)$.
- Q6: **new science** — our longitudinal-only SF shows **no forward enstrophy
  cascade**; to be discussed in the theory section; user speculates a possible
  GV analysis-code bug.
- Q7: default — model + exact field in Theory; sampling modes in
  Methods/Results.
- Q8: reuse existing macros.

**Why I had follow-ups (Q9–Q10).** Q6 is the consequential change. I inspected
`fig_full_qg_SF.png` to ground it: full-box \Sn{1} $\approx 0$ (homogeneity
holds), \Sn{2} plateaus near $100$--$150$ km, and \Sn{3} is **positive across
all resolved separations ($\sim7$--$200$ km)**, peaking at $L_d\approx92$ km
and declining beyond — a clean inverse energy cascade with no distinct $+r^3$
enstrophy range. Following the project's "be critical" guidance, I flagged two
more conservative explanations for the missing enstrophy range than a GV bug:
(i) we use longitudinal-only \Sn{3} whereas GV/Lindborg state the enstrophy
relation for the full $D_3=\langle\duL(\duL^2+\duT^2)\rangle$ — different
diagnostics; and (ii) the enstrophy range lies below the injection scale
($\sim R_d=15$ km) while our smallest reliable separations are $\sim7$ km, so
there is essentially no resolved enstrophy subrange. Q9 asks which framing to
emphasize (recommending the finding + (i)/(ii), GV discrepancy as an open
question); Q10 asks about the tone/placement of the "GV bug" claim
(recommending we keep §2 to our finding + caveats and defer any GV
reconciliation, including a cautiously-worded bug possibility, to Discussion),
and whether the full-$D_3$ vs. GV comparison has actually been run. Awaiting
answers before writing the subsection.

### 2026-06-29 (Wrote the QG theory subsection §2d in energy_cascade.tex)

Completed item 3 under **Writing**: read the answers to Q9–Q10 (both confirmed
my recommendations: emphasize the finding + caveats (i)/(ii), treat the GV
discrepancy as an open question, and **do not raise the bug** at this point)
and wrote the QG theory subsection into
`/home/xavier/Projects/Overleaf/Energy_Cascade/energy_cascade.tex`. The
document compiles cleanly (pdflatex + bibtex + 2 passes, 23 pages); the only
undefined reference (`sec:log_appendix`) and bibtex warning (`Rudnick2016`) are
pre-existing and unrelated.

**New subsection `\subsection{A quasi-geostrophic reference field}`**
(`\label{sec:qg}`), placed at the end of §2 "Theoretical underpinnings," after
"Homogeneity and the third moment" and before §3 Methods (per Q1). Five
paragraphs + one figure:
1. *Motivation* (per Q3) — the inertial-range relations of §2b are exact only
   under homogeneity/isotropy/steadiness/a developed inertial range; a finite,
   sparse glider survey violates all of these, so we need a setting where the
   cascade is known *a priori* and the SFs carry no sampling error.
2. *Why a 2D simulation* — known field everywhere/always; exact full-domain
   SFs; cascade direction fixed by the forcing; we can then subsample as
   gliders/drifters and ask whether the sign/slope of \Sn{3} survive
   (forward-ref to Results; sampling itself deferred per Q7).
3. *Model description* (compact, per Q3) — two-layer pyqg sim of GV, config of
   Ross et al. (2023): $1000$ km doubly-periodic, $256^2$, $\Delta x\approx3.9$
   km, $R_d=15$ km ($L_d\approx92$ km), $\beta$, $r_{ek}$, shear-forced
   ($U_1=0.025$, $U_2=0$), 5-yr spin-up + 15 yr, daily upper-layer velocities.
   States the field *is* the GV run (Q4) and stresses the two testbed virtues:
   2D by construction and exactly homogeneous/isotropic on the periodic domain.
4. *Exact full-domain ground truth* (Fig.~\ref{fig:qg_full}) — \Sn{1}$\approx0$
   to $\sim10^{-6}$ \mps (Eq.~S1zero holds essentially exactly, unlike the
   surveys); \Sn{2} plateau $\sim5.5\times10^{-3}$ \mpss near $100$–$150$ km;
   \Sn{3} positive at all separations, peaking near $L_d$ → inverse energy
   cascade (Eq.~s3\_2d\_energy). Framed as the "right answer" for Results.
5. *Enstrophy null result + caveats* (per Q6/Q9, no bug per Q10) — no resolved
   $+r^3$ enstrophy range in the longitudinal SF; attributed to (i) the
   enstrophy relation being conventionally stated for the full
   $D_3=\langle\duL(\duL^2+\duT^2)\rangle=D_{LLL}+D_{LTT}$ (Xie & Bühler 2019;
   Balwada 2022; GV) which the longitudinal \Sn{3} omits, and (ii) negligible
   scale separation below the injection scale ($R_d=15$ km vs.\ $\sim7$ km
   floor). This also serves as the brief mention of the full-$D_3$/$F(k)$
   framework (Q5). No GV bug mentioned.

Included `fig_full_qg_SF.png` (already present in the Overleaf `Figures/`
folder) as Figure~\ref{fig:qg_full}.

**References added to `energy.bib`:** `xie2019` (Xie & Bühler 2019, JFM 877 R3),
`balwada2022` (Balwada et al. 2022, Sci. Adv.), `ross2023` (Ross et al. 2023,
JAMES), `pyqg` (Abernathey et al. 2022, Zenodo), `gutierrez2025` (GV, submitted
to JTECH / EarthArXiv). All resolved in the `.bbl`. Reused existing macros
throughout (Q8: `\duL`, `\duT`, `\Sn{n}`, `\eflux`, `\zflux`, `\mps`, etc.).

**Not done (no instruction to):** did not commit/push to Overleaf — the prompt
asked only to write + log. Changes are local and compile-verified; ready to
push on request.

### 2026-06-29 (Pushed §2d to Overleaf)

On request, committed `energy_cascade.tex` + `energy.bib` and pushed to the
Overleaf remote. Had to `git pull --rebase` first (the remote had a direct
Overleaf edit, `d56ce3f "Update on Overleaf."`); rebase was clean, my subsection
and bib entries survived, and the rebased tree recompiled cleanly (23 pages).
Pushed `d56ce3f..d9a8d32 main`. Build artifacts excluded.

### 2026-06-29 (Surveyed QG results material; posted R1–R8 in Q&A/Results)

Completed item 4 under **Writing**: read the QG results code/figures/outputs in
preparation for the QG Results subsection, then posted eight questions in
**Q&A/Results**. No LaTeX written yet (prompt asks only to prepare + ask).

**What I surveyed (QG results material):**
- **Subregion vs.\ full box** (`fig_qg_subregion_vs_full` → `fig_qg_100km_vs_full.png`,
  already in Overleaf). Inspected it: in a 100 km box, \Sn{1} goes *negative*
  (to $\sim-3\times10^{-3}$ \mps) — a mean-flow bias absent in the full box; the
  **raw** \Sn{3} (open circles) sits near zero/negative, while the **centered**
  \Snc{3} (filled) climbs back toward the positive full-box curve. So a
  survey-sized box biases the raw third moment, and centering recovers the
  inverse-energy signal — *given 5 yr of averaging* (figure default Ndays=1825).
- **Time evolution** (`fig_qg_duL_vs_time`): \Sn{1} or \Sn{3} averaged over
  1/60/180/365/730/1825 days — convergence vs.\ averaging time. Also
  `fig_qg_duL_by_year` (year-to-year spread) and `fig_frac_duL3_vs_time`
  (how often raw du3 exceeds its correction term). Code exists; no final PNGs.
- **Sampling experiments** (the payoff for "analyze as gliders/drifters"):
  `Analysis/Output/` holds Eulerian, drifter, and glider SF (+ trajectories) for
  a 100 km box, 100 days, at two QG start times (ts=5001, 6001), x450 y450.
  These let us contrast the *true* field against drifter- and glider-sampled
  estimates. No figure generated yet.
- **Region inventory**: many boxes (100/200/300/500/1000 km) at several
  positions in `Analysis/Output/SF_region_*`; some positions give a strongly
  negative mean \Sn{1}. `fig_compare_duL_qg_data` overlays QG-region \Sn{1} on
  observed data.
- Only **two** QG figures are currently in the paper (full grid §2d; 100 km vs
  full). Time-evolution and sampling figures would need to be generated.

**Key scientific read (drives R2).** There is an apparent tension: in the QG
100 km box the *centered* \Snc{3} recovers the positive cascade, yet the real
observations (§4a/b) gave a null \Snc{3}. The resolution is **averaging time +
platform count**: the QG recovery uses 5 yr of dense (Eulerian) sampling, while
gliders get $\sim100$ days with a handful of platforms. The QG testbed thus
isolates *sampling* (duration $\times$ platforms), not region size, as what
defeats the measurement — the bridge to the paper's "humans will not measure
the sign" thesis.

**Questions posted (R1–R8 in Q&A/Results):** placement/title of the QG results
subsection (default §4c); the three-step story arc + the averaging-time
resolution of the QG-vs-data tension (R2, the central question); single box vs.
ensemble of positions/sizes (R3); whether to include the time-evolution
convergence figure (R4); whether to include the glider/drifter sampling
comparison and whether to show both platforms (R5); whether I should generate
the missing figures (R6); whether to quote a quantitative \eflux\ from the
full-box inverse-energy slope (R7); and whether to include $N_{\rm eddies}$
vs.\ time (R8, default drop). Awaiting answers before drafting §4c.

### 2026-06-29 (Read R1–R8 answers; verified sampling configs/data; posted follow-ups R9–R10)

Completed item 5 under **Writing**: read the R1–R8 answers (all defaults
confirmed: §4c placement; the three-step arc; single representative 100 km box
with spread noted in text; include the time-evolution panel; include the
glider+drifter sampling comparison; I generate the figures; quote \eflux\ from
the full-box slope; hold $N_{\rm eddies}$). Then, rather than ask reflexively, I
verified the sampling configurations and the data, and the numbers turned up a
real issue that warranted two follow-ups.

**Sampling configs (from `Analysis/Output/*.meta.json`).** Gliders: **10**
platforms, 100 days, 3-hourly, realistic survey track
(`100km100day10gliders3h.csv`) — the realistic case. Drifters: **121**
Lagrangian particles on a 10-km grid in the 100 km box — *dense*, not sparse.
Eulerian: all grid pairs in the box — the in-box truth. The "few platforms"
framing fits the gliders; the drifters are denser than any real deployment.

**Data.** Initially `$OS_DATA/QG/` looked empty (a double-slash `ls` glitch);
the user then fixed/confirmed it. Present: `QGModelOutput20years.nc` and the
three `SFQG_aver_pos_orien_5yearb*.nc`. Region SF (`SF_region_x300_y400_5years.nc`),
the Eulerian `.nc`, and the drifter/glider SF JSONs are local in
`Analysis/Output/`. So all figures (full-box, region-vs-full, time-evolution,
sampling) are generatable here.

**Key empirical finding (drove R9/R10).** I computed centered \Snc{3} for the
three samplings of the 100 km box over 100 days at both start times and counted
positive bins ($5$–$70$ km):

- ts=5001 — Eulerian 13/13 ✓; drifter 3/8 ✗; glider 4/9 ✗
- ts=6001 — Eulerian 11/13 ✓; drifter 8/8 ✓; glider 8/9 ✓

This **refines the confirmed R2 arc**: (a) dense Eulerian sampling recovers the
positive \Snc{3} in *100 days*, not multi-year — so duration is not the barrier
for dense sampling; (b) the realistic glider/drifter recovery is
**realization-dependent** (fails at ts5001, succeeds at ts6001), and platform
count (10 vs 121) matters far less than which 100-day window is sampled. The
cleaner message: the field has an unambiguous inverse cascade, but a realistic
$\sim$100-day survey recovers the sign *unreliably* (depends on the eddy
realization), while complete spatial sampling is robust.

**Follow-ups posted (R9–R10 in Q&A/Results).** R9: reframe §4c around
realization-dependence / sampling sparsity (vs.\ the averaging-time framing of
R2), keeping the Eulerian-vs-sparse contrast central. R10: whether to build an
ensemble of $\sim$12 start-time windows (and/or box positions) to show the
*distribution* of recovered sign/slope — making the unreliability claim robust
beyond the current $n=2$ — by running the `qg_gliders`/`glider_analysis`
pipeline against the full QG output. Awaiting answers before drafting §4c and
finalizing which figures to generate.

### 2026-06-29 (Made QG results figures, wrote §4c, pushed to Overleaf)

Completed item 6 under **Writing**: read the R9–R10 answers (R9: reframe around
realization-dependence / sampling sparsity — use default; R10: defer the
ensemble, use the two existing windows, **remind me later**), generated the
figures, wrote §4c, and pushed to Overleaf.

**🔔 REMINDER for JXP (per R10):** revisit whether to run an *ensemble* of
$\sim$12 start-time windows (± box positions) to show the distribution of
recovered \Snc{3} sign/slope and put the realization-dependence claim on firmer
footing than the current $n=2$ (ts5001/ts6001). Deferred for now.

**Figures generated** (new script `Figures/py/figs_qg_results.py`, env
`ocean14`):
- `fig_qg_time_evolution.png` — subregion ($x{=}300,y{=}400$, 100 km) \Sn{1}
  (left) and centered \Snc{3} (right) averaged over 30 d → 5 yr vs.\ the exact
  full-domain curves. Shows the mean \Sn{1} converges slowly while \Snc{3}
  recovers the positive sign almost immediately for dense sampling.
- `fig_qg_sampling.png` — centered \Snc{3} from 10 gliders (realistic track) and
  121 drifters sampling the 100 km box over 100 days, vs.\ the exact in-box
  (Eulerian) truth, for two start-time windows (panels scaled independently
  because the in-box truth itself differs between windows). Realization A:
  sparse platforms miss a strong positive signal (drifters $\approx0$); B: they
  track a weaker one. Realization-dependent; not fixed by platform count.
- Reused the existing `fig_qg_100km_vs_full.png` (subregion vs.\ full box) — it
  was in the `Figures/` folder but had not been included in the tex; §4c now
  includes it.

**\eflux from the full-box slope (R7).** Through-origin fit of
$\Sn{3}=\tfrac32\eflux r$ over the resolved rising branch (18–68 km) gives
$\eflux \approx 1.7\times10^{-10}$; quoted in §4c as an order-of-magnitude
$\eflux \approx 2\times10^{-10}\,{\rm m^2\,s^{-3}}$ (the branch is not a clean
unit-slope power law, so I framed it as order-of-magnitude).

**§4c "A quasi-geostrophic testbed"** (`\label{sec:results_qg}`), placed after
§4b and before Discussion. Four paragraphs + three figures, following the R9
arc: (1) the field has a known positive cascade, quote \eflux; (2) a
survey-sized subregion carries a nonzero \Sn{1} that biases raw $S_3$ while the
centered \Snc{3} recovers the truth — validating the centering correction
against ground truth (Fig.\ qg\_subregion); (3) with dense sampling the
centered moment recovers quickly — duration is not the obstacle (Fig.\
qg\_timeevo); (4) **the payoff** — realistic sparse sampling (10 gliders / 121
drifters, 100 d) recovers the sign only realization-dependently, explaining the
observational null of §4a/b (Fig.\ qg\_sampling). Reused existing macros.

**Compile + push.** Compiles cleanly (pdflatex + bibtex + 2 passes, **25
pages**); all three new figure refs resolve; only the pre-existing
`sec:log_appendix` / `Rudnick2016` warnings remain. Committed `energy_cascade.tex`
+ the two new PNGs, pulled --rebase (clean), and pushed to Overleaf
(`bf1a49a..bc45213 main`). Build artifacts excluded.

### 2026-06-30 (Figure touch-ups: caption, label, appendix move, ARCTERX r-limit)

Completed item 7 under **Writing** — four touch-ups to earlier figures.
Confirmed the figure→number mapping from the document order (the user's hints
matched): Fig 2 = `fig_experiments` (glider paths), Fig 3 =
`fig_separations_Calypso2022`, Fig 4 = `fig_loglin`, Fig 8 = `fig_S1S3_other`.

1. **Caption for Fig 2** (`fig:cugn`): replaced the stub "Glider paths" with a
   real caption (three experiments, regions, color = individual glider, pointer
   to Table~\ref{tab:experiments}).
2. **Fig 3 \Nmin label**: in `figs_paper_structure.py::fig_separations`, moved
   the $N_{\rm min}=100$ annotation from the right (x=0.97, `ha='right'`) to the
   left (x=0.03, `ha='left'`), off the histogram bars.
3. **Fig 4 → Appendix**: moved the `fig:loglin` (log-vs-linear binning) figure
   out of §3 into a new appendix subsection "Sensitivity to separation binning"
   (`sec:bin_appendix`). The in-text §3 reference still resolves.
4. **Fig 8 ARCTERX r-limit**: in `fig_S1S3_other`, pass `use_xlim=(0,100)` for
   the ARCTERX-2023 row so the panels span 0–100 km instead of 0–400.

**⚠️ Bug caught & fixed (process note).** The figure functions save to the
*current working directory* (`Figures/py/`), not `Figures/`. My first copy
step `cd Figures/ && cp ...` therefore copied the **stale** `Figures/*.png`
(unchanged) to Overleaf — a no-op — while the regenerated figures sat in
`Figures/py/`. Caught it via `git show --stat` (the figure commit changed only
the tex) and md5 comparison: `py/` versions (sep `749701…`, S1S3 `a627e4…`)
differed from the stale `Figures/` versions (`c0df…`, `7a78…`). Re-copied the
correct `py/` versions to both the canonical `Figures/` and Overleaf, and
verified by extracting+viewing the committed blob. **Lesson:** these figure
scripts write to CWD — always source figures from `Figures/py/` (or run from
`Figures/`), and verify a figure commit's `--stat` actually lists the PNGs.

**Compile + push.** Compiles cleanly (pdflatex + bibtex + 2 passes, **26
pages**; the appendix figure adds a page). Two commits — `c7cf05c` (tex: caption
+ appendix move + code-aligned edits) and `a4beedd` (corrected PNGs) — pulled
--rebase (clean, over a remote "Update on Overleaf"), pushed
(`3f4bef8..a4beedd main`). Verified my edits survived the rebase. The
`figs_paper_structure.py` / `figs_qg_results.py` code edits live in the cugn
repo and are **not committed there** (no request to); they remain in the working
tree. Pre-existing `sec:log_appendix` (isopycnal-coords appendix, separate from
the new `sec:bin_appendix`) is still an undefined ref — pre-existing, untouched.