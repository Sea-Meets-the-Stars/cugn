# Write the Karman-Howarth section of the paper

## Context

### High-level

You are helping me and Dan Rudnick to write a paper about the structure function of the velocity field to be submitted to the Journal of Physical Oceanography.  

### Writing style

See the PDFs in cugn/papers/Structure/Writing/previous/ for several of my previous publications so that you learn my writing style.

### Code

See all of the files in the cugn/papers/Structure/ and the calls to the main Repo and other dependencies. 

### Overleaf

The git repo for this paper may be found here:
/home/xavier/Projects/Overleaf/Energy_Cascade

### Karman-Howarth

I have put a copy of the PDF file kindly provided by Rick Salmon in the cugn/papers/Structure/Writing/ folder.  It is named karman.howarth.pdf

### Outline

Our outline is in the cugn/papers/Structure/Writing/ folder.  It is named outline.md.

## Writing

1. Read all of the files mentioned in Context.  Log your work.

2. Read the Outline.  Then 

    - Prepare to Write Section 2a (ignore the QG model for now) of the paper in the Overleaf Project in the energy_cascade.tex file.  
    - First, ask me any questions you have about what to do.  Put them in the Q&A section below.
    
3. I have answered your questions in the Q&A section below.  Please read them and let me know if you have any more.  If so add them to the Q&A section below.  Log your work
    
4. I have answered Q8. Read it and write the Karman-Howarth section of the paper in the Overleaf Project in the energy_cascade.tex file.  

    - Use my scientific writing style.  
    - Use macros and \ensuremath{} to render LaTeX expressions whenever you can.  
    - push your changes via git.   You will need the token.  It is provided as OVER_TOKEN in the .bashrc file.
    - Log your work

### Q&A

Questions before writing Section 2a (Karman-Howarth; QG deferred). Each has my recommended default in **bold** — feel free to just confirm or correct inline.

**Q1 — Section identity & numbering.** I read "Section 2a" as the *Karman-Howarth calculations* subsection under a new top-level **§2 "Theoretical underpinnings"** (matching the outline), placed between the Introduction (§1) and "Observations and Methods" (which becomes §3). The QG analysis would be the eventual §2b but is skipped for now. Is that the right placement, or do you want the K-H material as a subsection inside the existing Methods section instead?
> A: That is correct.

**Q2 — 2D vs. 3D in the main text.** The outline stresses "2D is critical." My plan: present the **2D Karman-Howarth equation and the 2D third-order relation in the main text**, mention the 3D 4/5 law only as the familiar reference result, and relegate the full 3D derivation (and the wavenumber-space flux derivations) to the Appendix/Supplemental. Do you agree, or do you want both 2D and 3D given equal weight in the main text?
> A: I agree.

**Q3 — How much derivation vs. results.** I propose the main text **states** the key results (definitions of $f$, $K$, the structure functions; the K-H equation; the exact third-order/flux relation) with brief physical justification, and the **step-by-step algebra from Salmon's notes goes to an Appendix** ("Other in Appendix/Supplemental"). Roughly how many displayed equations do you want in the main-text §2a — a tight ~5–8, or a fuller treatment?
> A:  Let's start with a tight ~5–8.

**Q4 — The exact 2D relation we test.** Salmon's notes give the 2D K-H equation (eq 83) and the 2D energy/enstrophy fluxes (eqs 159/166) in terms of $\langle[\Delta v]^3\rangle$, but they do not write a clean "2D analog of the 4/5 law." Do you want me to (a) **derive/state the inertial-range third-order relation directly from Salmon's 2D K-H equation** and cite the standard 2D literature (e.g., Lindborg 1999; Bernard 1999; Cerbus & Chakraborty) for the inverse-cascade / enstrophy-cascade results, or (b) stay strictly within Salmon's notes and cite only Batchelor/Davidson/Frisch? My default is **(a)**.
> A: I agree with (a).

**Q5 — The $\langle du \rangle = 0$ thread.** I'd like §2a to **state the homogeneity result $\langle du \rangle = 0$ and show how a nonzero measured mean biases $\langle duL^3 \rangle$**, then point forward to the Methods section where the cumulant ("raw moment") correction is applied. Should that bias bookkeeping live here in the theory section, or be deferred entirely to Methods? Default: **brief statement here, full treatment in Methods.**
> A: I like your idea.

**Q6 — Notation & macros.** The code uses `duL`; Salmon uses $\langle[\Delta v]^n\rangle$; the outline uses $\langle duL\rangle$, $\langle du^2\rangle$, $\langle du^3\rangle$. I propose standardizing on **$\delta u_L$ for the longitudinal velocity increment and $\langle \delta u_L^n \rangle$ for the structure functions**, with `\newcommand` macros (e.g. `\duL`, `\Sn`) and `\ensuremath{}`. OK, or do you have a preferred symbol set?
> A:  I like your proposal.

**Q7 — References.** The current `energy.bib` is inherited from the oxygen paper and contains **no turbulence references** (no Batchelor, Davidson, Frisch, Kolmogorov 1941, Lindborg, etc.). I'll need to add them. Shall I **add the needed `@book`/`@article` entries to `energy.bib`** as I write, using standard citation keys? Default: **yes.**
> A: Yes, please add them as you go.

---

Follow-up after your answers (2026-06-21):

**Q8 — Which 2D inertial-range relations to display, and the sign framing.** Your answers (Q2, Q4) put the 2D third-order relation in the main text. 2D turbulence has *two* inertial ranges with opposite-signed third-order relations, and the sign is exactly the diagnostic our paper is about:
> - inverse **energy** cascade (larger scales): $\langle \delta u_L^3 \rangle = +\tfrac{3}{2}\,\epsilon\, r$ (Lindborg 1999), positive;
> - forward **enstrophy** cascade (smaller scales): $\langle \delta u_L^3 \rangle = +\tfrac{1}{8}\,\eta\, r^3$ (Bernard 1999), positive but steeper;
> - contrasted with the 3D forward **energy** cascade: $\langle \delta u_L^3 \rangle = -\tfrac{4}{5}\,\epsilon\, r$ (negative).

My plan for §2a is to display **all three** (they fit the ~5–8 budget) and frame the **sign and slope of $\langle \delta u_L^3 \rangle$ as the central testable prediction** — i.e. a positive, ~linear $\langle \delta u_L^3 \rangle$ signals an inverse energy cascade. Is that the framing you want, or would you rather I show only the 2D inverse-energy relation in the main text and defer the enstrophy-range relation?
> A: Yes, display all three.

## Logging

Append an entry to the **Logs** section of this file using the format:

```
### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>
```

## Logs

### 2026-06-21 (Read all Context files in preparation for writing the Karman-Howarth section)

Read every file referenced in the Context section and surveyed the supporting code. Summary of what I read and learned:

**Karman-Howarth derivation (`Writing/karman.howarth.pdf`, all 17 pages, by Rick Salmon).**
A self-contained, step-by-step derivation (leaning on Batchelor, Davidson, Frisch) intended to spare the reader pencil work. Key chain and the author's highlighted points:
- *Definitions* (eqs 1–8): correlation tensor $Q_{ij}(\mathbf{r})$, triple-correlation $S_{ijk}$, the longitudinal/transverse correlation functions $f(r)$, $g(r)$, and $K(r)$; the second- and third-order structure functions $\langle[\Delta v]^2\rangle = 2u^2(1-f)$ and $\langle[\Delta v]^3\rangle = 6u^3K$.
- *Isotropy + incompressibility* give $Q_{ij}$ and $S_{ijk}$ in terms of $f$ and $K$ alone, worked separately for $d=2$ and $d=3$ (eqs 24/25, 51/53). **2D vs 3D is treated explicitly throughout** — directly relevant since the outline flags "2D is critical for our experiment."
- *Dynamics* (Navier-Stokes, eqs 55–61): pressure terms vanish by incompressibility; yields the **Karman-Howarth equation** in 3D (eq 73, `KH3`) and 2D (eq 83, `KH2`).
- *Kolmogorov 4/5 law* (eqs 84–95, `4-5law`): $\langle[\Delta v]^3\rangle = -\tfrac{4}{5}r\epsilon$, an **exact** consequence of KH (Frisch's point), NOT dependent on K41. Salmon's own caveat (highlighted red): the 4/5 law was *derived assuming $\epsilon$ is constant*; it should be tested against data, not used to interpret it, and it is easy to over-estimate its usefulness.
- *Wavenumber-space evolution* for both 3D (eqs 98–130, energy flux $\Pi_E(k)$, `pie3`) and 2D (eqs 131–166, energy flux `PiE2` and enstrophy flux `PiZ2`). The 2D fluxes are written in terms of $\langle[\Delta v]^3\rangle$ — the observable we measure.
- Salmon's tagged equations (`defAB`, `grfr`, `Sijkr`, `Sijk`, `Qii3`, `Gamma3`, `KH3`, `Qii2`, `Gamma2`, `KH2`, `kh90`, `4-5law`, `inequal`, `forward/inverse`, `defspek`, `EkRr`, `RrEk`, `dRdt`, `pie3`, `EkRr2`, `RrEk2`, `Ekfr2`, `PiE2`, `PiZ2`) are likely the equations he intends for the main text vs. appendix. The outline note "Really important equations in main text / Other in Appendix/Supplemental" maps onto these.
- Recurring theme matching our paper's thesis: $\langle du \rangle = 0$ is required by homogeneity; measuring the sign of the cascade is hard.

**Current paper draft (`/home/xavier/Projects/Overleaf/Energy_Cascade/`).**
- `energy_cascade.tex`: AMS template (JPO), still a skeleton — title "Energy cascade", authors TBD (JXP + D. Rudnick), one-line abstract, empty Intro/Methods/Results/Discussion/Conclusions, one figure included (`fig_experiments.png`, glider paths). Many `\newcommand` macros inherited from a prior CUGN/oxygen paper (mostly not yet relevant here).
- `TODO.txt`: analysis decisions — focus dataset = **Calypso2022**; $\Delta t$ currently 10 hr (want >6); minN=100; depth=60 m; **linear** binning (maybe broaden last bin); use "centered".
- `energy.bib` is the bibliography; `Figures/` mirrors most of `Structure/Figures/`.

**Writing style (previous publications in `Writing/previous/`).**
- `BING.pdf` (Prochaska & Frouin 2025, *Biogeosciences*): IOP retrieval / Bayesian inference. Style notes — numbered motivations, equations defined inline, "Here, we…" framing of contributions, methods split into formalism / forward model / dataset subsections.
- `Nenya.pdf` (Prochaska et al. 2023, *IEEE TGRS*): self-supervised learning on SST. Style — Nomenclature/Acronyms table, dense literature-grounded intro motivating the questions, footnotes for asides. Both papers: first-person plural, physically-motivated, equation-forward, clear question framing in the intro.

**Analysis code (`Structure/Analysis/` + `../profiler/` dependency).**
- Velocity differences between profile pairs computed in `profiler/profilerpairs.py:calc_delta()`: longitudinal $duL = \hat{\mathbf{r}}\cdot\Delta\mathbf{u}$, transverse $duT$, plus $du$, $dS$, $dT$.
- Structure functions in `calc_Sn_vs_r()`: per separation bin, raw moments $S_1=\langle duL\rangle$, $S_2=\langle duL^2\rangle$, $S_3=\langle duL^3\rangle$, with bootstrap/SEM errors.
- **Corrected (cumulant) moments** in `calc_corr_Sn()`: $S_2^{corr}=S_2-\langle duL\rangle^2$ and $S_3^{corr}=S_3-3\langle duL\rangle S_2+2\langle duL\rangle^3$ — i.e. the central/cumulant third moment, removing bias from a nonzero mean $\langle duL\rangle$. This is the "$\langle du3\rangle$ corrections (aka raw moments)" in the outline and ties directly to the homogeneity requirement $\langle du\rangle=0$.
- Key params (`struct_defs.py`): max_time ($\Delta t$), minN, depth index iz (~60 m), log vs linear binning, avoid_same_glider=True.
- QG analysis (`qg_uL_SF.py`, `qg_100km.py`, `qg_utils.py`): 2-layer pyqg model, structure functions over the full ~1000 km domain vs. extracted 100 km² (and 200 km²) subregions, with time evolution (1/30/90 day) preserved; analogous cumulant correction $du3_{corr}=du3-3\,du1\,du2^2+2\,du1^3$. Lets us compare a "perfect" model field against the glider-sampled estimate.
- Datasets: **Calypso2022** (focus), Calypso2019, ARCTERX-2023 (+ ARCTERX 2025 in progress), and the QG model.

**Takeaway for writing the K-H section:** present the K-H equation (2D emphasized, since gliders sample a quasi-2D field), define $f$, $K$, and the structure functions, state the 4/5 law (and its 2D analog / enstrophy considerations) as the exact result we test, and foreground the $\langle du\rangle=0$ homogeneity requirement and the cumulant correction to $\langle duL^3\rangle$ as the practical bridge between Salmon's theory and our measurements. Main-text vs. appendix equation split should follow Salmon's tagged equations.

### 2026-06-21 (Reviewed Q&A answers; one follow-up question added)

Read the answers to Q1–Q7. All confirmed my recommended defaults: (1) §2a is the Karman-Howarth subsection of a new top-level §2 "Theoretical underpinnings", with Observations & Methods becoming §3; (2) 2D in the main text, 3D + wavenumber-space fluxes in an appendix; (3) tight ~5–8 displayed equations in §2a; (4) derive the inertial-range third-order relation from Salmon's 2D K-H equation and cite the standard 2D literature (Lindborg 1999; Bernard 1999; Cerbus & Chakraborty); (5) brief $\langle\delta u\rangle=0$/bias statement in §2a, full cumulant correction in Methods; (6) standardize on $\delta u_L$ with `\newcommand` macros + `\ensuremath{}`; (7) add the missing turbulence references to `energy.bib` as I write.

I raised **one** follow-up (Q8) because it materially shapes the equation budget and the paper's central claim: whether to display all three opposite-signed third-order relations (3D forward energy $-\tfrac45\epsilon r$; 2D inverse energy $+\tfrac32\epsilon r$; 2D forward enstrophy $+\tfrac18\eta r^3$) and frame the **sign + slope of $\langle\delta u_L^3\rangle$** as the testable prediction — or show only the 2D inverse-energy relation in the main text. My recommendation is to show all three. Awaiting the answer to Q8 before writing §2a (item 4).

Planned §2a equation set (pending Q8): (i) definition of $\delta u_L$ and $\langle\delta u_L^n\rangle$; (ii) $\langle\delta u_L^2\rangle = 2u^2(1-f)$; (iii) the 2D Karman-Howarth equation; (iv) 3D 4/5 law as the reference result; (v) 2D inverse-energy third-order relation; (vi) 2D enstrophy-range third-order relation; (vii) homogeneity $\langle\delta u\rangle=0$ and its bias on $\langle\delta u_L^3\rangle$. ~7 displayed equations.
