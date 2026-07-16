# Add the new analysis to the paper

## Context

### High-level

You are helping me and Dan Rudnick to write a paper about the structure function of the velocity field to be submitted to the Journal of Physical Oceanography.  

### Writing style

See the PDFs in cugn/papers/Structure/Writing/previous/ for several of my previous publications so that you learn my writing style.

### New analysis

Review all of the new analysis that you have performed in the Analysis/ section.  In particular, review the files in the Analysis/claude_reports/ folder.

### Code

See all of the files in the cugn/papers/Structure/ and the calls to the main Repo and other dependencies. 

### Overleaf

The git repo for this paper may be found here:
/home/xavier/Projects/Overleaf/Energy_Cascade

### Outline

Our outline is in the cugn/papers/Structure/Writing/ folder.  It is named outline.md.

## Writing

1. Read all of the new analysis that you have performed in the Analysis/claude_reports/ folder.  Then proceed to have a Discussion with me in the section below on what to add to the paper and where.  Do not perform any writing yet.  Log your work.

2. Read my answers to the questions in the Discussion section below.  Run the agreed upon re-analysis.  And then add the robustness material to the paper, including figures.  If you have any new questions, ask me and I will answer them.  Use Fable if you can. Log your work.

## Discussion

I read all nine reports (`00_summary` + Directions 1–7, incl. the robustness
report). Below is what I think we should add and where, then the decisions I
need from you. **No paper writing yet** — this is the plan.

### Where the paper stands

- §1 Intro (empty); §2 Theory (2D limit, \kh, homogeneity, QG reference field);
  §3 Methods (Data, Structure functions); §4 Results (4a Calypso 2022, 4b
  Calypso 2019 / ARCTERX 2023, 4c QG testbed); **§5 Discussion and §6
  Conclusions are still empty**; Appendices (CUGN characteristics, binning, \kh
  derivation).
- Current thesis: the *sign* of the velocity energy cascade is not recoverable
  from a finite glider survey — the raw positive \Sn{3} is a sampling-bias
  artifact; the centered \Snc{3} is null; the QG testbed shows this is a
  sampling limitation, not the absence of a cascade.

### What the new analysis gives us

The eight directions split cleanly into **one new positive result** and a
**suite that hardens the existing null and fills the empty Discussion**.

| # | Direction | What it delivers | Proposed home | Priority |
|---|---|---|---|---|
| 5 | **Scalar (T/S) cascade** | Mixed Yaglom moment $\langle\duL(\delta\theta)^2\rangle$ **negative, growing with $r$** for T & S in all three surveys → forward scalar-variance cascade, *detected where the velocity cascade is null* | **New Results subsection §4d + figure**, short Yaglom bit in §2, reframed abstract/intro | **Headline — decision needed** |
| 2 | Full/transverse $D_3$ + isotropy | Calypso 2022 obeys 2D isotropy relation to ~4%; centered full $D_3=D_{LLL}+D_{LTT}$ also null → longitudinal-only loses nothing | §3 Methods or Appendix; one line in §4a | High (pre-empts a referee) |
| R | Robustness (bootstrap=DOF, Δt scan, last-bin) | Error bars adequate; $\Delta t=10$ hr near-optimal; merging the marginal $r\approx63$ km bin kills the only near-2σ excursion | §3 Methods + Appendix | High (defends the null) |
| 1 | Mean-flow removal | A fitted affine/quadratic flow captures <15% of variance and neither reduces \Sn{1} nor reproduces \Snc{3} → centering is the right tool; bias is intrinsic sparse-eddy sampling | §5 Discussion ¶ (+ optional appendix) | Medium–High |
| 6 | Temporal evolution | \Sn{1} bias **grows through each campaign**; velocity null in every sub-window; scalar cascade persists → design implication (shorter/denser or Lagrangian) | §5 Discussion ¶ | Medium |
| 4 | Anisotropy | Calypso 2022 ~isotropic (3% coherent); ARCTERX strongly anisotropic (22%, mean flow 0.11) → explains the differing \Sn{1} biases | Fold into §4b + §5 Discussion | Medium |
| 3 | Intermittency | Weakly non-Gaussian (flatness >3, fat tails), ESS near K41; signed skewness null → data sense *real* turbulence, null is about the *flux* not noise | §4a or §5 Discussion ¶ | Medium |
| 7 | Depth dependence | Velocity null at all depths 10–160 m; energy & \Sn{1} surface-intensified; scalar cascade throughout, peaks 30–100 m → justifies 60 m | §3 Methods sentence + Appendix; supports §4d | Medium |

### My recommended plan

1. **Promote the scalar cascade to a headline result (§4d).** It converts the
   paper from a pure null into a *contrast*: gliders **can** measure a
   sign-definite forward scalar-variance cascade but **cannot** measure the
   sign-ambiguous velocity energy cascade — because the scalar cascade has a
   fixed a-priori sign and high SNR (fronts), while the velocity flux is a small
   signed residual swamped by sampling bias. That is a stronger, more
   publishable story than the null alone, and it uses data already in hand.
2. **Harden the null with Directions 2 + R** in Methods/Appendix (isotropy
   relation to 4%; full-$D_3$ also null; bootstrap≈DOF; Δt=10 hr; last-bin
   merge). These are exactly the points a JPO referee will press on.
3. **Write §5 Discussion around the sampling thesis**, drawing on Directions 1
   (no removable mean flow), 6 (bias grows through the campaign), 4
   (isotropy/anisotropy contrast), 3 (real-but-intermittent turbulence). This
   corroborates the QG sampling argument *from the data side* and motivates the
   "ideal experiment" discussion the outline calls for.
4. **Justify 60 m and depth-robustness (Direction 7)** with a Methods sentence +
   an appendix panel.

### Decisions I need from you

**D1 — Scalar cascade: scope (the big one).** Options:
- (a) **Headline** — new §4d + figure, a short scalar-Yaglom paragraph in §2,
  and reframe the abstract/intro (and possibly the title) around the
  velocity-null / scalar-detection contrast. *My recommendation.*
- (b) **Minor** — a short subsection/paragraph noting the scalar detection
  without reframing the paper.
- (c) **Defer** — leave it out of this paper (a companion/follow-up), keep this
  paper tightly on the velocity cascade.
> A:  This may become the headline, but I'm not ready to add to the paper yet.  So, hold off for now.

**D2 — Prerequisite check before we headline the scalar result.** The reports
themselves flag that a **mean scalar-gradient removal** (the scalar analogue of
Direction 1) has *not* been done, and that a large-scale mean $\nabla\theta$
crossed with the mean flow could masquerade as a $\langle\delta u_L(\delta\theta)^2
\rangle$ trend. If we go with D1(a), I recommend running that check (and, ideally,
recomputing potential density via GSW for an isopycnal cross-check) *before*
building the section, so we don't headline a result a referee can undo. Do that
first? Default: **yes, run the mean-gradient check before writing §4d.**
> A:  Yes, run the check now.

**D3 — How much robustness material in the main text vs. appendix?** The null is
the paper's backbone, so I lean toward a compact **Methods** paragraph
(isotropy to 4%; full-$D_3$ null; bootstrap≈DOF; Δt=10 hr) with the scans and
tables in an **Appendix**. Alternative: put more of it in the Results. Default:
**concise in Methods, details in Appendix.**
> A: Use your default.

**D4 — Reframing the front matter.** If D1=(a), the abstract ("The energy
cascade is difficult to measure") and the (currently placeholder) title should
acknowledge the scalar result. Do you want me to propose a revised
title/abstract as part of this, or keep the front matter untouched for now?
Default: **propose a revised abstract; leave the title decision to you.**
> A: Hold off for now.

**D5 — Are these `Fable 5` numbers ours to trust as-is?** The reports were
produced by a different model and I have not re-run the pipelines. Before I
write any numbers into the paper I plan to re-execute the relevant `figs_*.py`
/ analysis modules (env `ocean14`) to confirm the key figures/numbers reproduce
— especially the scalar result and the isotropy/robustness tables. Default:
**yes, I re-verify each number I cite before it goes in the paper.**
> A: Sure, re-run

### Things I am *not* proposing to add

- The $\chi_T,\chi_S$ transfer rates as precise values — the 2D Yaglom
  coefficient is uncertain, so I'd quote them order-of-magnitude only (as the
  report does).
- ESS/high-order intermittency exponents as a quantitative result — they are
  near-K41 and the sparse surveys make $p\ge5$ unreliable; I'd use intermittency
  only qualitatively ("data sense real turbulence").
- Anything requiring new QG runs (the deferred ensemble from `qg_prompts.md` R10
  is still outstanding and tracked there).

## Logging

Append an entry to the **Logs** section of this file using the format:

```
### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>
```

## Logs

### 2026-07-13 (Read the 8 new-analysis reports; posted a Discussion proposal — no paper writing)

Completed item 1 under **Writing** (this file, `new_analysis.md`): read all nine
files in `Analysis/claude_reports/` (`00_summary` + Directions 1–7 and the
robustness report) and wrote a placement proposal + five decision points into
the **Discussion** section. No LaTeX written (prompt: discuss only).

**What the reports contain (all on the real glider data; no QG):**
- **Dir 5 — scalar (T/S) cascade [headline].** Mixed Yaglom moment
  $\langle\duL(\delta\theta)^2\rangle$ is significantly **negative and grows
  with $r$** for temperature and salinity in all three surveys → a forward
  scalar-variance cascade, *detected where the velocity cascade is null*.
  Strongest in Calypso 2019, weakest in Calypso 2022 (focus); $\chi$ is
  order-of-magnitude only (2D Yaglom coefficient uncertain). Likely frontal.
- **Dir 2 — full/transverse $D_3$ + isotropy.** Calypso 2022 obeys the 2D
  incompressible isotropy relation $D_{TT}=D_{LL}+r\,dD_{LL}/dr$ to ~4%; the
  centered full $D_3=D_{LLL}+D_{LTT}$ is also null → longitudinal-only loses
  nothing. Numerically-centered $D_{LLL,c}$ matches the paper's algebraic
  \Snc{3} to machine precision.
- **Dir R — robustness.** Bootstrap SE ≈ DOF SE (inflation ~1.0–1.2×), so the
  error bars are adequate; $\Delta t=10$ hr is near-optimal (larger windows
  manufacture spurious >2σ bins); merging the marginal $r\approx63$ km bin
  removes the only near-2σ excursion. Settles the `TODO.txt` Δt choice.
- **Dir 1 — mean-flow removal.** A fitted affine/quadratic survey-scale flow
  captures <15% of the velocity variance and neither reduces \Sn{1} nor
  reproduces \Snc{3} → the bias is not a removable coherent shear; algebraic
  centering is correct. Real-data corroboration of the QG sampling argument.
- **Dir 6 — temporal.** The \Sn{1} bias **grows through each campaign** (gliders
  spread, patch drifts); velocity cascade null in every sub-window; scalar
  cascade persists/strengthens. Design implication: shorter/denser or Lagrangian.
  (Also fixed a `profile_subset` in-place-mutation bug in the profiler package.)
- **Dir 4 — anisotropy.** Calypso 2022 ~isotropic (mean flow 0.009 m/s, ~3%
  coherent); ARCTERX strongly anisotropic (0.11 m/s, ~22% coherent) → explains
  the differing \Sn{1} biases across surveys.
- **Dir 3 — intermittency.** Flatness modestly >3 (fat tails), ESS near K41,
  skewness null → the data resolve genuine (weakly intermittent) turbulence, so
  the null is about the *signed flux*, not featureless noise.
- **Dir 7 — depth.** Velocity null at all depths 10–160 m; energy & \Sn{1}
  surface-intensified; forward scalar cascade throughout, peaking 30–100 m →
  justifies the 60 m choice. (Isopycnal check needs a density field not stored
  in these files — GSW follow-up.)

**Proposal posted (Discussion section):** promote Dir 5 to a headline Results
subsection §4d (velocity-null vs. scalar-detection *contrast*); harden the null
with Dir 2 + R in Methods/Appendix; build the empty §5 Discussion from Dir 1/6/4/3
(the sampling thesis, corroborating QG from the data side); justify 60 m with
Dir 7. Decision points raised: **D1** scalar-cascade scope (headline / minor /
defer — I recommend headline); **D2** run the mean-scalar-gradient check *before*
headlining Dir 5 (recommend yes); **D3** robustness in Methods vs. Appendix
(recommend concise Methods + Appendix detail); **D4** reframe abstract/title
(recommend propose new abstract, leave title to JXP); **D5** re-verify the
`Fable 5` numbers by re-running `figs_*` before citing them (recommend yes).
Flagged what I would *not* add: precise $\chi$ values, quantitative ESS
exponents, anything needing new QG runs. Awaiting answers before any writing.

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