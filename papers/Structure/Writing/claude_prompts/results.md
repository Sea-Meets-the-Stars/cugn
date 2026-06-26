# Write the Results section of the paper

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

### Outline

Our outline is in the cugn/papers/Structure/Writing/ folder.  It is named outline.md.

## Writing

1. Read all of the files mentioned in Context in preparation of writing the Results section.  Log your work.

2. Add to the Q&A section any questions that you have about the Results section.  Do no writing yet. Log your work.

3.  I have answered your questions in the Q&A section below.  Please read them and let me know if you have any more.  If so add them to the Q&A section below.  Log your work

4.  I have answered your questions in the Q&A section below.  Please read them and let me know if you have any more.  If so add them to the Q&A section below.  Log your work
    
5. Read my answers.  Then make the figure changes that you suggested.  Log your work.

6. Please modify all of the key results figures to use linear binning instead of log. Log your work.
    
7. Proceed to write Results section 4a on Calypso 2002 of the paper in the Overleaf Project in the energy_cascade.tex file.  

    - Use my scientific writing style.  
    - Use macros and \ensuremath{} to render LaTeX expressions whenever you can.  
    - push your changes via git.   You will need the token.  It is provided as OVER_TOKEN in the .bashrc file.
    - Refer to the figures
    - Embed the figures in the text with \includegraphics
    - Write figure captions for the figures.
    - Log your work

## Q&A

Questions before writing the **Results** section. Each has my recommended default in **bold** — confirm or correct inline.

**Q1 — Section/subsection structure.** Add `\section{Results}` (`sec:results`, already forward-referenced from §2) organized as: **§4a covering the Calypso 2022 structure functions ($\langle\delta u_L\rangle$, $\langle\delta u_L^2\rangle$, raw $\langle\delta u_L^3\rangle$), then §4b on the centered third moment and the cascade-sign result**, with the other two experiments' 1st/3rd moments placed in the Appendix (per the outline). Agree, or would you prefer a different grouping (e.g., one subsection per moment)?
> A: 4a will examine Calypso 2022 only.  4b will include the other two experiments.

**Q2 — QG results: now or later?** The earlier sections deferred the QG model ("ignore the QG model for now"), but the outline has a QG-results block ($\langle\delta u_L\rangle$ full box vs sub-regions, time evolution, $\langle\delta u_L^3\rangle$, "analyze as gliders/drifters"). Default: **write only the glider (Calypso/ARCTERX) Results now and defer the QG block to a later pass.** Or should I include QG results in this pass?
> A:  We will defer the QG results to a later pass.

**Q3 — Power-law fit for $\langle\delta u_L^2\rangle$, and recomputing exact numbers.** The second-order panel rises roughly as a power law over $\sim$5–30 km before a plateau. Plan: **fit a slope over the rising range and compare it to the 2D expectations** (an energy-range $r^{2/3}$ vs an enstrophy-range $\sim r^2$), noting the plateau/roll-off. To quote exact numbers (slope, the magnitude of the $\langle\delta u_L\rangle$ bias, RMS increment), I'll **re-run `load_SF` for Calypso 2022 from the data ($OS\_SPRAY$ available)**. OK to (a) fit/quote a slope and (b) recompute values for the text?
> A:  Yes, we will fit a slope over the rising range and compare it to the 2D expectations.  

**Q4 — Figures: layout and main-text vs Appendix.** Plan for the main text: a Calypso 2022 figure showing $\langle\delta u_L\rangle$ and $\langle\delta u_L^2\rangle$ (possibly combined as a multi-panel), and the raw-vs-centered $\langle\delta u_L^3\rangle$ figure (`fig_corrected_s3`) as the key result; the other two experiments (`fig_S1S3_other`) go in the **Appendix**. Agree? And do you want the moments shown as **separate single panels (as currently generated) or combined multi-panel** figures? Default: **combine the Calypso 2022 moments into one multi-panel figure + keep `fig_corrected_s3` standalone.**
> A:  We will show Figures of all the other experiments in 4b.  Use Figure fig_S1S3_other.png for the other two experiments.

**Q5 — How strongly to state the null result in Results.** The centered $\langle\delta u_L^3\rangle\approx0$ (cascade sign undetectable) is the paper's punchline. Default: **state the empirical result plainly in Results** — raw $\langle\delta u_L^3\rangle$ is positive, but it is dominated by the $3\langle\delta u_L\rangle\langle\delta u_L^2\rangle$ bias and the centered moment is consistent with zero — and **reserve the broad "humans cannot measure the sign of the cascade" framing for the Discussion.** Agree?
> A: Agreed

**Q6 — Two small calls.** (a) The single largest-$r$ ($\sim$70 km) centered-$S_3$ bin is positive with a large error and sits where pairs are sparse — **keep it but flag it as a low-significance single bin** (vs. trimming it)? (b) Methods promised an isopycnal-vs-fixed-depth check "agree (Results)" — handle with **a brief sentence asserting agreement** here (figure to Appendix only if you want one)? Defaults in bold.
> A:  Keep it but comment that it as a low-significance single bin.  We will handle the isopycnal check with a brief sentence asserting agreement here.

---

Follow-up after your answers (2026-06-25):

Structure now locked: **§4a — Calypso 2022 only** ($\langle\delta u_L\rangle$, $\langle\delta u_L^2\rangle$ with a fitted slope, raw vs centered $\langle\delta u_L^3\rangle$); **§4b — the other two experiments** (Calypso 2019, ARCTERX-2023) using `fig_S1S3_other.png`; QG deferred.

**Q7 — Regenerating the result figures (they are stale) + two layout calls.**
- (a) **The result figures now in Overleaf (`fig_duL`, `fig_duL2`, `fig_duL3`, `fig_corrected_s3`, `fig_S1S3_other`) were all generated on 2026-06-21 at `minN=50`** — before we set `minN=100`. So the bins shown (the `goodN` mask) are no longer what the paper describes. I'll **regenerate all five at `minN=100`** (item 5) so the figures match the text. Confirm.
- (b) **§4a layout:** combine $\langle\delta u_L\rangle$ and $\langle\delta u_L^2\rangle$ into a **single two-panel figure**, and keep `fig_corrected_s3` (raw vs centered $\langle\delta u_L^3\rangle$) standalone as the key result — so §4a has two figures? (Alternative: keep three separate single panels.) Default: **two-panel + standalone corrected-$S_3$.**
- (c) **Power-law fit:** overlay the fitted slope on the $\langle\delta u_L^2\rangle$ panel (regenerate with a fit line and the slope annotated), or just quote the slope in the text and leave the panel as data-only? Default: **overlay the fit on the panel.**
> A: (a) yes, please do;  (b) $\langle\delta u_L\rangle$ is one of the key results and should stand alone; (c) yes, overlay the fit.

---

Follow-up after your answers (2026-06-25):

Layout locked: **§4a Calypso 2022** uses three standalone panels — $\langle\delta u_L\rangle$ (`fig_duL`), $\langle\delta u_L^2\rangle$ with the fit overlaid (`fig_duL2`), and raw-vs-centered $\langle\delta u_L^3\rangle$ (`fig_corrected_s3`); **§4b** uses `fig_S1S3_other`. All regenerated at $N_{\min}=100$.

**Q8 — Axis scaling and fit range for the $\langle\delta u_L^2\rangle$ panel.** To show the power-law fit as a straight line (and read off the slope), the natural choice is a **log–log** panel for $\langle\delta u_L^2\rangle$ (it is strictly positive; by contrast $\langle\delta u_L\rangle$ and $\langle\delta u_L^3\rangle$ change sign and must stay linear-$y$). Default: **render `fig_duL2` log–log, fit a power law over the monotonically-rising bins (≈5–30 km, exact range set from the data and stated in the caption/text), and annotate the slope; the plateau/roll-off bins are shown but excluded from the fit.** Confirm the log–log choice and that an auto-selected ~5–30 km rising range is acceptable (or give me a fixed range).
> A: No, keep it linear.

## Logging

Append an entry to the **Logs** section of this file using the format:

```
### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>
```

## Logs

### 2026-06-25 (Read Context in preparation for the Results section)

Re-read the Context with a focus on the **Results**. Writing style, outline, and the analysis pipeline were already digested (see entries above and the `karman.md`/`obs_methods.md` logs); this pass concentrated on the **results-figure code** and on **inspecting the actual figures** so I know what the data show before writing.

**Outline — Results scope.** Calypso 2022 (focus): $\langle\delta u_L\rangle$; $\langle\delta u_L^2\rangle$ (fit power law(s)); $\langle\delta u_L^3\rangle$ (uncertainty at large $r$, constraints at small $r$); the centered ("corrected") third moment; (struck-through: $\langle\delta u_L T^2\rangle$, $\langle\delta u_L S^2\rangle$). Then the other two experiments (1st & 3rd moments; Appendix), and a QG-results block. Figures listed: $\langle\delta u_L\rangle$, $\langle\delta u_L^2\rangle$; $\langle\delta u_L^3\rangle$ raw vs the $3\langle\delta u_L\rangle\langle\delta u_L^2\rangle$ correction; plus QG panels.

**Results-figure code (`Figures/py/figs_paper_structure.py`).** `plot_single_order(order, ax, rdict, corrected=)` draws $S_n(r)$ for `goodN` bins with bootstrap error bars; on a log-$x$ axis; `corrected=True` overlays raw (×) and centered (•) and adds a legend; a green dashed line marks zero. Driver functions: `fig_single_order` (flg 3/4/5 → $\langle\delta u_L\rangle$/$\langle\delta u_L^2\rangle$/$\langle\delta u_L^3\rangle$ for Calypso 2022), `fig_corrected_s3` (flg 6 → raw-vs-centered $S_3$), `fig_S1S3_other` (flg 7 → 1st & 3rd moments for the other two experiments), and QG figs (flg 8/9). All read parameters from `struct_defs` (now `minN=100`, `iz=5`≈60 m, `max_time=10` hr, `btype='log'` — note the result panels are plotted on a log-$x$ axis even though linear binning is the adopted scheme).

**What the figures show (Calypso 2022; values approximate, from the panels):**
- `fig_duL_Calypso2022` — $\langle\delta u_L\rangle$ scatters about zero at small $r$ but is **systematically positive** ($\sim$+0.005 to +0.02 m/s) for $r\gtrsim15$ km. Homogeneity ($\langle\delta u_L\rangle=0$) is only approximately satisfied; a real, $r$-growing mean is present.
- `fig_duL2_Calypso2022` — $\langle\delta u_L^2\rangle$ rises from $\sim$0.006 (m/s)$^2$ at 3 km to a plateau $\sim$0.010 (m/s)$^2$ at 30–50 km (RMS increment $\sim$0.08–0.1 m/s), rolling off at the largest $r$. Roughly power-law over $\sim$5–30 km — the panel to fit a slope.
- `fig_duL3_Calypso2022` — **raw** $\langle\delta u_L^3\rangle$ is mostly **positive and grows with $r$** ($\sim$+0.0004–0.0007 (m/s)$^3$ at 20–70 km). Taken at face value, the positive sign would indicate an **inverse energy cascade** (2D prediction $+\tfrac32\eflux r$).
- `fig_corrected_s3_Calypso2022` — **the key result.** Overlaying raw (×) and **centered** (•): the centered third moment is **consistent with zero** across essentially the whole range (error bars straddle 0), except possibly the single largest-$r$ bin ($\sim$70 km). The apparent positive raw signal is therefore largely an **artifact of the nonzero mean** $\langle\delta u_L\rangle$ entering the $3\,S_1 S_2$ term — exactly where $\langle\delta u_L\rangle$ was largest. **The glider data do not robustly detect the sign of the cascade** — directly supporting the paper's thesis ("humans will not measure the sign of the energy cascade").
- `fig_S1S3_other` — Calypso 2019 and ARCTERX-2023 tell the same story: nonzero (sometimes large, e.g. ARCTERX $\langle\delta u_L\rangle\sim-0.1$ m/s) means, raw $S_3$ excursions, but **centered $S_3$ consistent with zero**.

**Central narrative for the Results section:** (1) $\langle\delta u_L^2\rangle$ shows a credible turbulent rise/power law; (2) the raw $\langle\delta u_L^3\rangle$ is positive and would suggest an inverse cascade; (3) but $\langle\delta u_L\rangle\ne0$ reveals residual inhomogeneity, and once centered, $\langle\delta u_L^3\rangle\approx0$ — the cascade sign is not measurable; (4) the same holds for the other two experiments.

**Open questions for item 2 (to post in Q&A):** (a) include the QG-results block now or defer (the earlier sections deferred QG)?; (b) power-law fit details for $\langle\delta u_L^2\rangle$ — fit range and whether to quote a slope/compare to $r^{2/3}$ or $r^2$; (c) which figures go in the main text vs. Appendix (outline puts the other-experiment moments in an Appendix); (d) how strongly to state the null result on the cascade sign (it is the paper's punchline); (e) the largest-$r$ centered-$S_3$ bin — discuss or trim. No section text written yet (item 1 is read + log).

### 2026-06-25 (Posted Results Q&A; no writing)

Per item 2, posted **six questions (Q1–Q6)** to the Q&A section, each with a recommended default, and wrote no section text. They cover: (Q1) the §4 Results subsection structure (Calypso 2022 moments in §4a, the centered third moment / cascade-sign result in §4b, other experiments → Appendix); (Q2) whether to defer the QG-results block (default: defer, as earlier sections did); (Q3) a power-law fit + slope for $\langle\delta u_L^2\rangle$ and permission to re-run `load_SF` to quote exact numbers; (Q4) figure layout (combine the Calypso 2022 moments into a multi-panel; keep `fig_corrected_s3` standalone; other experiments to Appendix); (Q5) how strongly to state the null cascade-sign result in Results vs. Discussion (default: empirical result in Results, broad framing in Discussion); (Q6) two small calls — keep-but-flag the sparse largest-$r$ centered bin, and handle the isopycnal check with a brief sentence. Awaiting answers before drafting.

### 2026-06-25 (Reviewed Results Q&A answers; posted one follow-up)

Read the answers to Q1–Q6. Locked-in decisions: **§4a = Calypso 2022 only** (including the centered third moment / cascade-sign result — note: this lives in 4a, *not* a separate subsection as I had first proposed); **§4b = the other two experiments** (Calypso 2019, ARCTERX-2023), shown in the **main text** using `fig_S1S3_other.png` (not relegated to an Appendix); **QG deferred**; fit a power-law slope to $\langle\delta u_L^2\rangle$ over the rising range and compare to 2D expectations; state the null cascade-sign result empirically in Results, reserve the broad framing for Discussion; keep the sparse largest-$r$ centered bin but flag it; handle the isopycnal check with one sentence.

Raised **one follow-up (Q7)** because verification turned up a real issue: **all five result figures in Overleaf are dated 2026-06-21 and were generated at `minN=50`** (before we set `minN=100`), so their `goodN` bin masks no longer match the text. Q7 asks to (a) regenerate all five at `minN=100` (item-5 work), (b) confirm the §4a layout (a two-panel $\langle\delta u_L\rangle$+$\langle\delta u_L^2\rangle$ figure plus the standalone `fig_corrected_s3`), and (c) whether to overlay the fitted power law on the $\langle\delta u_L^2\rangle$ panel. No section text written. Awaiting the Q7 answer.

### 2026-06-25 (Reviewed Q7 answers; posted one more follow-up)

Read the Q7 answers: (a) regenerate all five result figures at $N_{\min}=100$ — yes; (b) **$\langle\delta u_L\rangle$ stands alone** (so §4a uses three separate single panels — `fig_duL`, `fig_duL2`, `fig_corrected_s3` — rather than a combined two-panel); (c) overlay the fit on the $\langle\delta u_L^2\rangle$ panel — yes. Layout for §4 fully determined; QG deferred; §4b uses `fig_S1S3_other`.

Posted **one follow-up (Q8)**: to display the power-law fit as a straight line and read off a slope, default to rendering the $\langle\delta u_L^2\rangle$ panel **log–log** (it is strictly positive, unlike the sign-changing $\langle\delta u_L\rangle$/$\langle\delta u_L^3\rangle$ panels), fitting over the monotonically-rising bins (~5–30 km, exact range from the data, stated in the text), and excluding the plateau from the fit. Awaiting confirmation before item 5 (figure regeneration). No section text written.

### 2026-06-25 (Item 5: made the fit-overlay code change; figure regeneration BLOCKED on missing data)

Q8 answer: **keep the $\langle\delta u_L^2\rangle$ panel linear** (do not switch to log–log), but still overlay the power-law fit and annotate the slope (per Q7c).

**Code change made** (`Figures/py/figs_paper_structure.py`, `fig_single_order`): added a `fit_range=(5,30)` km parameter; for `order==2` it fits $\log_{10} S_2$ vs $\log_{10} r$ over the good bins in that range, overlays the curve $A\,r^{p}$ on the existing linear-$y$/log-$x$ axes, annotates the slope $p$ in the legend (`$r^{p}$ (5–30 km)`), and prints the fitted $p,A$. Guarded to skip if <2 bins fall in range. `python -m py_compile` passes.

**⚠️ BLOCKER — data not available.** Regenerating the five result figures requires loading `$OS_SPRAY/Calypso/calypso2022_ctd.mat`, but **`/home/xavier/Projects/Oceanography/data/Spray/` no longer exists** in the current environment (it was present on 2026-06-24 when I made the threshold/loglin figures, but is gone now). `OS_SPRAY` is still set (to `.../data/Spray/`), but the directory is absent. So I could **not**:
- regenerate `fig_duL`, `fig_duL2`, `fig_duL3`, `fig_corrected_s3`, `fig_S1S3_other` at `minN=100`;
- verify the fit overlay or extract the exact numbers (slope $p$, $\langle\delta u_L\rangle$ bias magnitude, RMS increment, centered-$S_3$ zero-crossing) needed for item 6.

**Still pending (needs data restored):** run the regeneration + fit overlay, visually verify, copy the five PNGs to the Overleaf `Figures/` folder, and record the fitted slope/numbers. **Asked JXP to re-make `OS_SPRAY` data available.** Nothing pushed; the only change so far is the uncommitted `cugn`-repo code edit above.

### 2026-06-25 (Data restored; regenerated all result figures at minN=100; recorded numbers)

JXP restored `$OS_SPRAY` (confirmed `calypso2022_ctd.mat` present). Regenerated all five result figures at `minN=100`, visually verified the two key ones, and **copied all five into the Overleaf `Figures/` folder**: `fig_duL`, `fig_duL2`, `fig_duL3`, `fig_corrected_s3`, `fig_S1S3_other` (all `_Calypso2022` except the last).

**Fit overlay (`fig_duL2`).** The power-law fit renders correctly on the linear-$y$/log-$x$ panel; legend fixed to show **only** the fit line (the underlying `errorbar` carries a hardcoded `label='Corrected'` that was mislabeling the $S_2$ points — now suppressed via `handles=[fit_line]`). Fitted slope **$p\approx0.42$** over 5–30 km ($S_2\approx2.6\times10^{-3}\,r^{0.42}$). NB the binned central values carry small run-to-run jitter because `calc_Sn_vs_r` reports a *bootstrap-resampled* mean (not the plain sample mean), so $p$ wobbles ~0.42±0.01 between runs — quote as $\approx$0.4. This slope is **shallower than the 2D energy-range $r^{2/3}$** and far from the enstrophy-range $r^2$.

**Key Calypso 2022 numbers (minN=100, 60 m, $\Delta t\le10$ hr):**
- 19 good bins, $r=4.3$–66.9 km, $N=109$–4375 pairs/bin.
- $\langle\delta u_L^2\rangle$: $5.0\times10^{-3}$ → $1.06\times10^{-2}$ (m/s)$^2$ (RMS increment 0.071–0.103 m/s), rising over ~5–30 km then a plateau ~30–50 km and a roll-off at the largest $r$.
- $\langle\delta u_L\rangle$: ranges $-0.006$ to $+0.018$ m/s; $|\langle\delta u_L\rangle|$ up to $\sim$0.018 m/s at large $r$ (the inhomogeneity that biases $S_3$).
- raw $\langle\delta u_L^3\rangle$ up to $\sim+7\times10^{-4}$ (m/s)$^3$ at large $r$; **centered** $\langle\delta u_L^3\rangle$ consistent with zero — only **1 of 19** bins exceeds $2\sigma$.
- **Smoking gun:** at $r\approx50$ km, raw $S_3\approx4.4\times10^{-4}$ is almost entirely the bias $3\langle\delta u_L\rangle\langle\delta u_L^2\rangle\approx4.0\times10^{-4}$, leaving centered $\approx4\times10^{-5}$.
- Largest-$r$ bin ($r=66.9$ km): centered $S_3=6.5\times10^{-4}$, $\pm4.1\times10^{-4}$ (≈1.6$\sigma$, $N=232$) — the low-significance single bin to keep-but-flag.

**Code change** (`cugn` repo, uncommitted): `fig_single_order` gained a `fit_range=(5,30)` km parameter and the order-2 power-law overlay + legend fix.

**Not pushed:** item 5 carries no git instruction (push is in item 6). The five regenerated figures sit staged in the Overleaf working tree, to go up with the §4 write/push. Ready for item 6.

### 2026-06-26 (Item 6: wrote Results §4a Calypso 2022; resolved merge; pushed)

Wrote **§4 Results → §4a Calypso 2022** in `energy_cascade.tex` (my style, macros + `\ensuremath`), embedding the three figures with captions, and pushed to Overleaf (commit `3f88032`, 18 pages, compiles clean).

**Content (five paragraphs):** (1) intro — focus experiment, $\Nmin=100$ gives $r\approx4$–70 km, fixed depth 60 m with the isopycnal check noted; (2) $\Sn{1}$ — ~zero at small $r$, systematically positive ($\approx0.02$ m/s) for $r\gtrsim15$ km → residual inhomogeneity; (3) $\Sn{2}$ — rises $\approx5\times10^{-3}\to1.1\times10^{-2}$ (m/s)$^2$ (RMS 0.07–0.10 m/s), power law $p\approx0.4$ over 5–30 km (shallower than $r^{2/3}$, well below $r^2$), plateau ~30–50 km; (4) third moment — raw $S_3$ positive, $\to\approx7\times10^{-4}$ (m/s)$^3$; centered $\Snc{3}$ consistent with zero (1 of 19 bins $>2\sigma$); $r\approx50$ km bias accounts for nearly all the raw value; 67 km bin marginal (~1.6$\sigma$); (5) empirical null result, broad framing deferred to Discussion (per Q5). Figures `fig:duL`/`fig:duL2`(fit)/`fig:s3`(raw vs centered) as Figs 4–6, captioned. Macros added: `\mps`, `\mpss`, `\mpsss`, `\nbinscalypso`=19.

**Merge/push:** push rejected → `git pull --rebase` conflicted in `energy_cascade.tex` (JXP had added a bare `\subsection{Calypso 2022}` heading via two "Update on Overleaf" commits). Resolved keeping the heading + my content (their side empty), recompiled clean, `rebase --continue`, pushed `3f88032`. The earlier `\dtysm` fix and §3b reproducibility edits rode along.

**Open flags (for JXP):** (a) binning still `btype='log'` vs text "linear"; (b) `\valnboot=1000` (JXP) but figures used `nboot=100`; (c) `sec:log_appendix` undefined forward ref. None block compilation except (c) renders "??".