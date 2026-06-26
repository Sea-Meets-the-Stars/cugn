# Write the Observations and Methods section of the paper

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

1. Read all of the files mentioned in Context.  Log your work.

2.  Great catch on the sign error!  I have fixed it (please confirm).  Read the Outline.  Then 

    - Prepare to write the Structure function portion of the Observations and Methods (ignore the Data section for now) of the paper in the Overleaf Project in the energy_cascade.tex file.  
    - Make sure you understand from the code how the structure function is calculated for the gliders
    - First, ask me any questions you have about what to do.  Put them in the Q&A section below.
    - Do not write any text yet.
    - Log your work
    
3.  I have answered your questions in the Q&A section below.  Please read them and let me know if you have any more.  If so add them to the Q&A section below.  Log your work

4.  I have answered your questions in the Q&A section below.  Please read them and let me know if you have any more.  If so add them to the Q&A section below.  Log your work
    
5. Make the figure changes that you suggested.  Log your work.
    
6. Proceed to write the Structure function portion of the Observations and Methods (ignore the Data section for now) of the paper in the Overleaf Project in the energy_cascade.tex file.  

    - Use my scientific writing style.  
    - Use macros and \ensuremath{} to render LaTeX expressions whenever you can.  
    - push your changes via git.   You will need the token.  It is provided as OVER_TOKEN in the .bashrc file.
    - Log your work

7. Please prepare to generate a Table describing the three experiments and their data.  Do this by:
    - Generating a Python script named tabs_structure.py in cugn/papers/Structure/Table/py. 
    - Model it after tabs_bing.py in bing/papers/phytoplankton/Tables/py
    - Ask questions before writing the table, add them to the Q&A section below.
    - Do not create the table yet.
    - Log your work.

8. Read my answers to the Q&A section below.  Then generate the table.  Log your work.

9. Please make another pass on the text and be sure that you have included sufficient detail that anyone could reproduce the results.  Log your work.

## Q&A

Sign fix confirmed: `calc_corr_Sn` now computes `S3corr = S3 − 3 S1 S2 + 2 S1³` (profilerpairs.py:613–614) — the correct third central moment, matching §2a Eq. `S3corr`. Thanks.

Questions before writing the **Structure functions** portion of Observations & Methods (Data section deferred). Each has my recommended default in **bold** — confirm or correct inline.

**Q1 — Section/subsection structure.** Rename the current `\section{Methods}` (now §3) to **"Observations and Methods"**, and add a `\subsection{Structure functions}` for this material, leaving an empty `\subsection{Data}` stub above it (to be filled later, per "ignore the Data section for now")? Or keep "Methods" as the title?
> A: Yes

**Q2 — Parameter values: macros + which numbers.** I'd like to define `\newcommand` macros for the key choices (so they update in one place) and quote the current values in the text: focus dataset **Calypso 2022**, time separation $\Delta t \le$ **10 hr**, minimum pairs per bin $N_{\min} =$ **100**, depth $z =$ **60 m**. Two mismatches to resolve: the code has `minN=50` (you want 100?), and `max_time=10` is **hours** (the outline/`TODO` "more than 6" — keep 10 hr?). Confirm the four values.
> A: Yes, that looks good

**Q3 — Binning (linear vs log).** `TODO` says linear. Plan: describe **linear bins, 0–400 km, ~80 bins, with the last bin possibly widened**, as the adopted scheme; state that we verified consistency with logarithmic binning and show the comparison once (a Results figure, e.g. `fig_loglin_sep`). Agree, or should log be primary / shown differently?
> A: Yes, that looks good

**Q4 — Fixed depth vs isopycnal.** Plan: present **fixed depth ($z=60$ m) as the primary analysis**, and describe the isopycnal-coordinate alternative (the `iz<0` path) as a comparison reported in Results/Appendix; defer the "multiple/averaged depths" exploration to later. Agree, or give isopycnal equal footing in the main Methods text?
> A:  Agreed

**Q5 — Meaning of "centered" (`TODO` item 5).** I read "use centered" as **reporting each binned structure function at the mean pair separation within the bin** (the code's `avg_r`), rather than at the nominal bin center/edge. Is that what you mean, or does "centered" refer to something else (e.g., a centered velocity/finite-difference, or removing a survey-mean velocity)?
> A:  Use "centered" to describe the corrected moments. It does not refer to centering in a radial bin.

**Q6 — Raw vs corrected moments + estimator notation.** Plan for the subsection: (i) define the binned **raw** moments $S_1,S_2,S_3$ as sample averages over the pairs in each separation bin; (ii) give the **corrected (central) moments** $S_2^{\rm c}=S_2-S_1^2$ and $S_3^{\rm c}=S_3-3S_1S_2+2S_1^3$, cross-referencing §2a Eq. `S3corr` and the homogeneity argument; (iii) state that uncertainties are **bootstrap** estimates (resampling pairs within each bin, `nboot=100`). For notation, reuse §2a's $\delta u_L$ / $\langle\delta u_L^n\rangle$ and add a hat (e.g. $\widehat{\langle\delta u_L^n\rangle}(r)$) to mark the sample estimator vs. the true ensemble moment. OK, or would you rather not distinguish estimator from ensemble?
> A:  Yes, that looks good

---

Follow-up after your answers (2026-06-24):

**Q7 — "centered" terminology + a separations figure.** Two small items:
- (a) Per Q5, I'll **name the corrected (central) moments "centered"** in the text — e.g. "the centered second and third moments, $S_2^{\rm c}$ and $S_3^{\rm c}$" — and use "centered" rather than "corrected" throughout (and I'll align §2a's wording to match if you'd like). Confirm that's the usage you want.
- (b) Should the **Structure functions** subsection include a figure illustrating the **pairs of separation** (the outline's "Separation pairs" / `fig_separations_Calypso2022.png`, or a histogram of pair separations `fig_histogram_dr.png`), or keep all figures in the Data section / Results and leave this subsection text-only? Default: **include one separations figure here** (it motivates the binning), referenced but with the exact panel TBD.
> A: (a) Yes, that is right; (b) yes, include one separations figure here.  The Figures are in the Overleaf Figures/ folder.  You should inspect them.  And if you want any modifications, let me know.

---

Follow-up after inspecting the figures (2026-06-24):

I looked at the two relevant panels in `Figures/`. **`fig_separations_Calypso2022.png`** (number of pairs vs. separation, log $x$-axis, with a dashed line near $\sim$100 that I read as the $N_{\min}$ cut) is well suited to the Structure-functions subsection — I'll use it as the "pairs of separation" figure. **`fig_loglin_sep_Calypso2019.png`** (left: log-spaced; right: linear-spaced; annotated "$t<10.0$ hours") is the natural "show both once" binning-comparison figure.

**Q8 — two small figure items.**
- (a) The log-vs-linear comparison figure is for **Calypso 2019**, but our focus dataset is **Calypso 2022**. Use the 2019 panel as-is (it only illustrates the binning choice), or would you like it **regenerated for Calypso 2022** for consistency? If the latter, I can run the analysis to remake it provided `OS_SPRAY`/data are available and you point me at the plotting script — otherwise you may prefer to remake it.
- (b) On `fig_separations_Calypso2022.png`, shall I have the **dashed threshold line labeled** "$N_{\min}=100$" (it's currently unlabeled)? Default: **yes, label it** — but this is a regeneration, so same caveat as (a).

If you'd rather not regenerate anything now, I'll write the subsection referencing the existing figures as-is and we can polish the panels later.
> A:  Please make the figure changes that you suggest.  I have made the `OS_SPRAY`/data available to you.  You will need to copy the new figures to the Overleaf Figures/ folder.  You have my permission to do so.

---

Item 7 — questions before generating the experiments Table (2026-06-26).

I drafted the script `cugn/papers/Structure/Tables/py/tabs_structure.py` (modeled on `tables_bing.py`), but have **not** created the table. NB the existing repo directory is `Structure/Tables/py` (plural); the prompt said `Structure/Table/py` — I used the existing `Tables/py`. Questions, each with my recommended default in **bold**:

**T1 — Columns.** Proposed default column set, one row per experiment: **Experiment | Region | Duration (days) | $N_{\rm gliders}$ | $N_{\rm profiles}$ | $N_{\rm pairs}$** ($N_{\rm pairs}$ = distinct-glider pairs within $\Delta t\le10$ hr, matching the analysis). Add/drop any? Candidates I left out: number of usable separation bins, RMS increment / velocity scale, depth range, sampling cadence, glider model (all-Spray?), CTD vs ADCP note.
> A: This looks fine.

**T2 — Region & dates.** I can compute $N_{\rm gliders}$, $N_{\rm profiles}$, duration, and $N_{\rm pairs}$ directly from the data. But **Region** and **absolute calendar dates** are not reliably in the loaded arrays (the analysis only uses time *differences*, and the epoch of `time` is unconfirmed). Shall I (a) **quote the region names and start/end dates that you provide** (please give them for each of Calypso 2019, Calypso 2022, ARCTERX 2023), or (b) drop those columns and report only computed quantities + a duration in days? Default: **(a) if you supply region + dates; otherwise (b).**
> A: You can get lat,lon, and time from the data. Do so.

**T3 — Which experiments / order.** Include all **three** (Calypso 2019, Calypso 2022, ARCTERX 2023) in the order Calypso 2019, Calypso 2022, ARCTERX 2023? (ARCTERX 2025 is still in progress and excluded.) Confirm.
> A: Yes, include all three.

**T4 — Counts: pre- or post-cut?** Report $N_{\rm profiles}$ as the count **after** the survey-edge and good-velocity cuts applied in `load_dataset` (e.g. Calypso 2022 drops the first 3 / last 12 days), i.e. the profiles actually used? Default: **yes, post-cut (the data actually analyzed).**
> A: Yes, post-cut.

**T5 — Table file & placement.** Emit `tab_experiments.tex` (label `tab:experiments`) as a `table*` (full-width), to be `\input{}` into the Data subsection (§3a) of `energy_cascade.tex`? Default: **yes.**
> A: Yes

## Logging

Append an entry to the **Logs** section of this file using the format:

```
### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>
```

## Logs

### 2026-06-24 (Read Context files for the Observations & Methods section)

Read the Context files with a focus on what the **Observations and Methods** section needs. The writing-style PDFs (`BING.pdf`, `Nenya.pdf`), the outline, and the Overleaf draft were already digested while writing the theory section (see the `karman.md` prompt logs); this pass concentrated on the **data pipeline and the structure-function methodology in the code**, per the refined item 2. Files read this pass: `Analysis/py/glider_io.py`, `struct_defs.py`, `data_utils.py`; `profiler/profilerpairs.py` (`__init__`, `calc_dist`, `generate_pairs`, `calc_delta`, `calc_Sn`, `calc_Sn_vs_r`, `calc_corr_Sn`); `profiler/gliderdata.py` (`SprayData`); and the current `energy_cascade.tex` Methods stub.

**How the glider structure function is computed (the pipeline `data_utils.load_SF`).**
1. `glider_io.load_dataset(dataset)` loads a list of `SprayData` objects, one per mission ID (`missid` = individual glider), from binned `.mat` CTD files under `$OS_SPRAY` (`from_binned_file(..., 'idg', ...)`). Horizontal velocity is the ADCP Doppler profile `udop`, `vdop` vs depth (`SprayData` subclasses `ADCPData`).
2. `ProfilerPairs(profilers, max_time, avoid_same_glider, remove_nans, randomize=False)` forms all profile pairs. `calc_dist` puts separations in an E/N frame relative to a meridian through the survey's median (lon, lat); `rxN, ryN` are the unit separation components.
3. `generate_pairs`: keep pairs with time separation `dt` below `max_time` and **only positive `dt`** (avoid double counting); `avoid_same_glider=True` drops same-mission pairs.
4. `calc_delta(iz, 'duLduLduL')`: at depth index `iz`, `du=u1−u0`, `dv=v1−v0`; **longitudinal** `duL = rxN·du + ryN·dv`, transverse `duT = ryN·du − rxN·dv`.
5. `calc_Sn`: per pair `S1=duL`, `S2=duL²`, `S3=duL³`.
6. `calc_Sn_vs_r(rbins, nboot=100)`: bin by separation `r`; per bin report N, the **mean** `r` of the pairs (`avg_r`), and the means of S1/S2/S3 with **bootstrap** errors.
7. `calc_corr_Sn`: corrected moments (see flag below).

**Key parameters (`struct_defs.py` + `TODO.txt`).**
- `iz=5` ≈ **60 m** fixed depth; `iz<0` switches to isopycnal coordinates (`prep_isopycnals`). Outline: try both and compare.
- `max_time=10`. **Unit check: `dt` is computed in hours (`(t_k−t)/3600`), so this is 10 hours**, matching `TODO.txt` ("Dt currently 10 hrs"). NB the `load_SF` docstring says "days" — a doc error; the code is hours.
- `minN`: bins kept if N > minN; code default **50**, but `TODO.txt`/outline imply **100**.
- Binning (`rbinning`): **log** = 40 bins $10^0$→400 km; **lin** = 80 bins 0→400 km. Code default `log`; `TODO.txt` says **linear** for the paper ("maybe broaden the last bin"). Reported `r` is the **mean** separation in the bin — likely what `TODO.txt` "use centered" means (confirm with JXP).
- `variables='duLduLduL'` is the main case; alternatives (`duTduTduT`, `duLdSdS`, `duLdTdT`, `duLduTduT`, `dTdTdT`) back the other/struck-through structure-function bullets.

**⚠️ Likely bug to flag before the Results / corrected-$\langle duL^3\rangle$ work.** `calc_corr_Sn` (profilerpairs.py:613–614) computes
`S3corr = S3 − (3·S1·S2 + 2·S1³) = S3 − 3 S1 S2 − 2 S1³`.
The third central moment — and the relation written in §2a (Eq. `S3corr`: `S3 − 3 S1 S2 + 2 S1³`) — has `+2 S1³`. **The cubic term's sign is flipped in the code.** `S2corr = S2 − S1²` is correct. Since `S1=⟨duL⟩` is small where homogeneity nearly holds, the `2 S1³` term is third-order-small and the numerical impact is probably minor, but it is a genuine code/theory mismatch worth fixing before "corrected" third moments go in the paper.

**Open Methods decisions to confirm with JXP** (will become Q&A items for item 2): focus dataset = Calypso2022; `max_time` (≥6 hr, currently 10 hr); `minN` (50 vs 100); depth 60 m fixed and/or isopycnal; linear binning with possibly-widened last bin; meaning of "centered"; whether to show log and linear once (per outline).

**Current draft state:** `energy_cascade.tex` Methods (now §3) is still a stub — only the glider-paths figure (`fig_experiments.png`, `fig:cugn`) plus empty Results/Discussion. The §2 Theory section already forward-references this Methods correction.

### 2026-06-24 (Confirmed sign fix; prepared Structure-function methods; posted Q&A)

Confirmed the user's fix to `calc_corr_Sn` (profilerpairs.py:613–614): now `S3corr += -3·S1·S2 + 2·S1³`, i.e. `S3corr = S3 − 3 S1 S2 + 2 S1³` — the correct third central moment, matching §2a Eq. `S3corr`. `S2corr = S2 − S1²` unchanged/correct.

Re-read the outline's **Structure functions** bullets (dtime + avoid-same-glider; pairs of separation; binning linear-vs-log "show both once"; fixed depth vs isopycnals "try both, compare" + multiple/averaged depths; $\langle du^3\rangle$ corrections aka raw moments) and reconfirmed the computation chain from the code (`data_utils.load_SF` → `ProfilerPairs` → `calc_delta`/`calc_Sn`/`calc_Sn_vs_r`/`calc_corr_Sn`), already documented in the 06-24 reading entry above.

Per item 2, wrote **no section text**. Posted **six questions (Q1–Q6)** to the Q&A section, each with a recommended default, covering: (Q1) renaming §3 to "Observations and Methods" + a `\subsection{Structure functions}` with a Data stub; (Q2) macros and the four key parameter values, flagging the `minN` 50-vs-100 and the `max_time` 10-hr/units points; (Q3) linear binning as primary with log shown once; (Q4) fixed depth primary vs isopycnal as comparison; (Q5) the meaning of "centered" (my read: report at the mean pair separation `avg_r`); (Q6) raw vs corrected (central) moments, bootstrap errors, and estimator-vs-ensemble notation. Awaiting answers before drafting (item 3/4).

### 2026-06-24 (Reviewed Q&A answers; one follow-up posted)

Read the answers to Q1–Q6. All confirmed my recommended defaults: (Q1) rename §3 → "Observations and Methods" with a `\subsection{Structure functions}` and a `\subsection{Data}` stub; (Q2) Calypso 2022, $\Delta t \le 10$ hr, $N_{\min}=100$, $z=60$ m (so `struct_defs.minN` will be quoted as 100, not the code's current 50); (Q3) linear bins (0–400 km, ~80, last bin possibly widened) primary, log shown once; (Q4) fixed depth primary, isopycnal as comparison; (Q6) raw + corrected/central moments, bootstrap errors, estimator-hat notation.

**Important reinterpretation from Q5:** "centered" does **not** mean radial-bin centering (my earlier guess was wrong). JXP: *"Use 'centered' to describe the corrected moments."* So the corrected (central) moments ARE the "centered" moments — the existing `calc_corr_Sn` approach already satisfies the `TODO` "use centered" item, and the paper should use the word **"centered"** for $S_2^{\rm c}$, $S_3^{\rm c}$. No separate radial-centering step is implied. (I will still state that each bin is reported at the mean pair separation `avg_r`, but as a plain methodological fact, not under the "centered" label.)

Posted **one follow-up (Q7)**: (a) confirm I should name the corrected moments "centered" throughout (and optionally realign §2a wording), and (b) whether to include a pairs-of-separation figure in this subsection (default: yes, panel TBD). Will draft the subsection once Q7 is answered.

### 2026-06-24 (Reviewed Q7 answer; inspected figures; posted Q8)

Read the Q7 answer: (a) yes — name the corrected moments **"centered"** throughout, and align §2a wording to match; (b) yes — include one separations figure here; figures live in the Overleaf `Figures/` folder, inspect them and flag any wanted modifications.

Inspected the two relevant panels:
- `fig_separations_Calypso2022.png` — number of pairs vs. separation (log $x$-axis, 1–100 km), peaking ~20–40 km, with a dashed horizontal line near ~100 that I read as the $N_{\min}$ cut. Good fit for the "pairs of separation" figure in the Structure-functions subsection.
- `fig_loglin_sep_Calypso2019.png` — two-panel log- vs. linear-spaced separation histograms, annotated "$t<10.0$ hours". The natural "show both once" binning-comparison figure — **but for Calypso 2019, not the focus dataset (2022)**.

Posted **one follow-up (Q8)** on two small figure items: (a) whether to regenerate the log-vs-linear comparison for Calypso 2022 (vs. using the 2019 panel as-is), and (b) whether to label the dashed threshold line on the separations histogram as "$N_{\min}=100$" — both would require regenerating the figures (need `OS_SPRAY`/data + the plotting script), so I offered to proceed referencing the existing figures as-is and polish later. No section text written (that is item 5). Awaiting the Q8 answer.

### 2026-06-24 (Made the Q8 figure changes; regenerated and copied to Overleaf)

JXP approved Q8 (make both changes; `OS_SPRAY`/data provided; permission to copy into the Overleaf `Figures/` folder). Done:

**Code edits** (in the `cugn` repo, not yet committed):
- `Analysis/py/struct_defs.py`: `minN` **50 → 100** — the value agreed in Q2, and the source the separations figure reads for its threshold line. This also tightens the `goodN` bin mask used elsewhere; worth noting for the Results stage.
- `Figures/py/figs_paper_structure.py` (`fig_separations`): added a text label `$N_{\rm min}=100$` on the dashed threshold line (rendered dynamically from `struct_defs.minN` via `ax.get_yaxis_transform()`), so it stays in sync if `minN` changes.

**Regenerated figures** (ran from `papers/Structure/Figures/` with `py/` + `../Analysis/py` on the path; data confirmed at `$OS_SPRAY=/home/xavier/Projects/Oceanography/data/Spray/`):
- `fig_separations_Calypso2022.png` — now shows the labeled `$N_{\rm min}=100$` threshold line. Visually verified.
- `fig_loglin_sep_Calypso2022.png` — **new**, the log-vs-linear separation comparison for the focus dataset (red, "$t<10.0$ hours"); left log-spaced, right linear 0–400 km. Visually verified (right panel makes clear pairs concentrate below ~80 km, motivating a possibly-widened last linear bin).

**Copied** both PNGs into `/home/xavier/Projects/Overleaf/Energy_Cascade/Figures/`. The superseded `fig_loglin_sep_Calypso2019.png` was left in place (unreferenced; can be removed later). 

**Not pushed:** item 5 carries no git instruction (the push lives in item 6, the writing step), so the new figures sit staged in the Overleaf working tree and the two `cugn` code edits are uncommitted — both await item 6 / explicit instruction.

### 2026-06-24 (Wrote the Structure functions subsection; pushed to Overleaf)

Wrote the **Structure functions** portion of Observations & Methods in `energy_cascade.tex` and pushed to Overleaf (commit `7e7f342` on `main`). Document compiles cleanly (pdflatex + bibtex + 2 passes, 14 pages, no undefined refs/citations); both new figures resolve.

**Macros added** (preamble): `\Shat{n}` (binned sample estimator $\widehat{\langle\delta u_L^n\rangle}$), `\Snc{n}` (centered moment $\langle\delta u_L^n\rangle_{\rm c}$), `\focusexp` (Calypso 2022), `\dtsym`/`\Nmin` (symbols), and value macros `\valdt` (10 hr), `\valNmin` (100), `\valdepth` (60 m), `\valrmax` (400 km), `\valnbins` (80), `\valnboot` (100).

**§2a alignment** (per Q7a): the homogeneity subsection now calls $\Snc{3}$ the **"centered"** third moment (was "central"/"corrected"), uses the `\Snc` macro in Eq. `S3corr`, and forward-references the centering to `sec:structfunc` (and the homogeneity conditions to Results).

**New material** — `\section{Methods}` renamed to **"Observations and Methods"** (`sec:methods`); added a `\subsection{Data}` (`sec:data`) stub holding the existing glider-paths figure (`fig:cugn`), and a `\subsection{Structure functions}` (`sec:structfunc`) covering: the longitudinal increment + estimator (Eq. `Snhat`); pair selection (distinct gliders; $\dtsym\le$ 10 hr, with the synoptic-window rationale); linear binning 0–400 km (~80 bins, widened last bin), reporting at mean separation, $N_{\rm min}=100$ cut; fixed-depth 60 m primary with isopycnal as a check; raw vs **centered** moments (Eq. `centered`, cross-ref §2a); and bootstrap errors (100 realizations). Added Fig. `separations` (`fig_separations_Calypso2022.png`) and Fig. `loglin` (`fig_loglin_sep_Calypso2022.png`) with captions.

**Push note:** the first push was rejected — the remote had an "Update on Overleaf" commit (`e54e575`; JXP had reordered §2 so "The two-dimensional limit" leads the section). I `git pull --rebase`d (clean, no conflicts), recompiled the merged result to confirm it still builds, and pushed. The committed files: `energy_cascade.tex` plus the two figures from item 5 (`fig_separations_Calypso2022.png`, `fig_loglin_sep_Calypso2022.png`). The two `cugn`-repo code edits (`struct_defs.py` minN→100, `figs_paper_structure.py` label) remain uncommitted in the `cugn` repo (not part of the Overleaf push; awaiting explicit instruction).

### 2026-06-26 (Item 7: drafted the experiments-Table script; posted Q&A; table not created)

Located the model `tables_bing.py` (at `bing/papers/phytoplankton/Tables/py/`, not `tabs_bing.py`) — a simple module: load data, open a `.tex` file, write a `table*` with `\caption`, a `tabular`, header row, a per-item loop writing rows, a notes line, and `\end{table*}`; a `__main__` calls it.

**Created** `cugn/papers/Structure/Tables/py/tabs_structure.py` modeled on it (note: repo dir is `Tables/py` plural; the prompt said `Table/py`). It defines `experiment_stats(key)` — loads each experiment via `glider_io.load_dataset` and computes $N_{\rm gliders}$, $N_{\rm profiles}$, duration (days), and $N_{\rm pairs}$ (built with `ProfilerPairs` using the paper's `max_time`/`avoid_same_glider`, so the count matches `data_utils.load_SF`) — and `mktab_experiments()` which writes `tab_experiments.tex` (label `tab:experiments`) with the proposed default columns. `python -m py_compile` passes. **Did NOT run it** (table not created yet, per the instruction); the `__main__` is annotated accordingly.

**Posted five questions (T1–T5)** to the Q&A section: T1 the column set; T2 region/dates (not in the data — need JXP to supply, else drop those columns); T3 the three experiments and order; T4 whether $N_{\rm profiles}$ is post-cut (default yes); T5 emit `tab_experiments.tex` as a full-width `table*` to be `\input` into the Data subsection. Awaiting answers before item 8 (generate the table). No Overleaf push (item 7 has no git step).

### 2026-06-26 (Item 8: finalized the table script per T1–T5; generation BLOCKED — data absent again)

Read the T1–T5 answers: T1 column set fine; **T2 — derive Location and Dates from the data's lat/lon/time** (don't ask JXP to supply); T3 all three, order Calypso 2019 / 2022 / ARCTERX 2023; T4 post-cut counts; T5 emit `tab_experiments.tex` as a full-width `table*` to `\input`.

**Updated `tabs_structure.py` to the confirmed design:** columns **Experiment | Location | Dates | Duration (days) | $N_{\rm gliders}$ | $N_{\rm profiles}$ | $N_{\rm pairs}$**. `experiment_stats` now also returns median (lon, lat) → a formatted Location, and start/end datetimes from the `time` array; added `_time_to_datetime` (time is in seconds — the epoch is auto-detected, Unix-seconds vs MATLAB-datenum, and the row loop prints the raw range + decoded dates so the epoch can be spot-checked on first run) and `_fmt_lonlat`. `python -m py_compile` passes.

**⚠️ BLOCKER — data not available again.** `/home/xavier/Projects/Oceanography/data/Spray/` is absent once more (it was present earlier today for the figure regeneration). Generating the table requires loading all three `*_ctd.mat` files, so I **could not run `mktab_experiments()`** to produce `tab_experiments.tex`, verify the decoded dates/epoch, or wire the `\input` into §3a. The script is ready; one run will produce the table once the data is restored. **Asked JXP to remount `OS_SPRAY`.** Nothing pushed.

### 2026-06-26 (Item 8 cont.: data restored; table generated, wired, verified)

Data restored (all three `*_ctd.mat` present). Ran `tabs_structure.py` → **`tab_experiments.tex`**. The time epoch auto-detected as **Unix seconds** (e.g. Calypso 2019 raw 1.553e9 → 2019-03-20); the decoded dates and median positions are sensible:
- Calypso 2019 — $1.8^\circ$W, $36.4^\circ$N (W. Mediterranean / Alboran), 20 Mar–20 May 2019, 61 d, 7 gliders, 2045 profiles, 23{,}781 pairs.
- Calypso 2022 — $3.1^\circ$E, $40.7^\circ$N (Gulf of Lion), 28 Mar–16 Jun 2022, 80 d, 8 gliders, 2681 profiles, 31{,}291 pairs.
- ARCTERX 2023 — $141.2^\circ$E, $19.6^\circ$N (W. Pacific), 31 May–31 Jul 2023, 62 d, 11 gliders, 1084 profiles, 5906 pairs.

**Placement/wiring (per T5):** copied `tab_experiments.tex` into the Overleaf repo and added `\input{tab_experiments}` to the Data subsection (`sec:data`). Full document compiles (pdflatex+bibtex+2; 14 pages); the table renders as **Table 1** (page 6), 7 columns, no overfull boxes, label `tab:experiments` resolved. Visually verified.

**Compile-breaking typo fixed (not mine):** the build first failed on `\dtysm` at line 388 — a typo of my macro `\dtsym` introduced in an Overleaf edit of the §3b synoptic-window sentence ("…yet this $\Delta t$ is wide enough…"). Changed only that token `\dtysm`→`\dtsym`, preserving the edited wording; the doc then compiled.

**Flag (left as-is):** an undefined reference `sec:log_appendix` (line 408, the isopycnal-check sentence — JXP changed my `sec:results` ref to point at a not-yet-existing appendix). It only yields a "??" and is JXP's intentional forward pointer, so I left it.

**Not pushed.** Item 8 has no git instruction, so I did not push. NB: the Overleaf remote still contains the `\dtysm` typo, so **the project will not compile on Overleaf until the fix is pushed.** Offered to push (would also carry the regenerated result figures from the results.md item-5 work and the table). Awaiting the go-ahead.