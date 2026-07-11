# Direction 6 — Temporal evolution within a survey

**Date:** 2026-07-11
**Author:** JXP & Claude (Fable 5)
**Code:** `papers/Structure/Analysis/py/temporal.py`, `figs_temporal.py`
**Figures:** `Figures_new/fig_temporal_evolution.png`,
`fig_temporal_pairtimes.png`
**Data:** 60 m, $\Delta t\le10$ hr, linear bins, 3 equal-count (time-quantile)
windows, 500 bootstrap.

## Question

Does the survey-mean flow (the $S_1$ bias) drift over the multi-week campaign,
and does any cleaner sub-window recover the velocity cascade better than the
full survey?

## A methodological catch (documented so it isn't repeated)

Usable pairs span essentially the whole campaign — the distribution of pair
mid-times covers 53–69 of the ~60–80 nominal days (`fig_temporal_pairtimes.png`)
— so the surveys *can* be subdivided in time. An initial attempt appeared to
collapse to a single window; that was an artifact of `ProfilerData.profile_subset`
**mutating the profiler in place**, so subsetting window 0 corrupted the shared
profilers before windows 1–2 were built. Deep-copying each profiler before
subsetting fixes it. (Worth a note in the profiler package: `profile_subset`
mutates in place and returns self.)

## Results (3 time-quantile windows)

| Experiment | max $\|S_1\|$ by window (m/s) | velocity $S_3$ bins >2σ | scalar-$T$ slope ($\propto-\chi_T$) |
|---|---|---|---|
| Calypso 2019 | 0.083 → 0.138 → 0.157 | 0, 1, 1 | +4e-8, −1.5e-7, −2.3e-7 |
| **Calypso 2022** | 0.027 → 0.022 → 0.053 | 0, 1, 0 | −7e-9, −1.4e-8, −1.4e-9 |
| ARCTERX 2023 | 0.047 → 0.074 → 0.131 | 0, 0, 0 | nan, −6.0e-7, −4.2e-7 |

Three findings:

1. **The $S_1$ bias grows through the campaign.** For Calypso 2019 and ARCTERX
   the survey-mean increment roughly doubles-to-triples from the first to the
   last third; Calypso 2022 is flatter but also rises at the end. As a survey
   proceeds, the gliders spread and the sampled patch drifts, so the
   large-scale inhomogeneity — and hence the raw-$S_3$ bias — *worsens* with
   time. The cleanest sampling is early.

2. **No sub-window recovers the velocity cascade.** The centered $S_3$ stays at
   the noise level (≤1 bin >2σ) in every window of every survey. There is no
   privileged early/late period hiding a detectable energy cascade — the null
   is robust in time as well as depth.

3. **The scalar forward cascade persists and tends to strengthen later.** The
   mixed-moment slope is negative (forward $T$-variance cascade) in most
   windows and grows in magnitude toward the end for Calypso 2019 (−2.3e-7) and
   ARCTERX (−4 to −6e-7), consistent with frontal sharpening as the surveys
   progress. Calypso 2022's scalar signal is weak in every window (matching
   Directions 5 and 7).

## Interpretation

Time-resolving the surveys reinforces the whole picture: the sampling problem
*grows* during a campaign (rising $S_1$), the velocity cascade is undetectable
in any sub-window, and the scalar cascade is the one robust, persistent signal.
For future experiments this argues for either **shorter, denser sampling** (the
early window is the least biased) or a design that keeps the sampled patch
Lagrangian to limit the growth of $S_1$.

## Suggested paper use

A short Discussion point: "Subdividing each campaign in time shows the survey-
mean bias $S_1$ grows as the survey proceeds and the sampled region drifts,
while the centered velocity $S_3$ remains null in every sub-window — the cascade
is not hidden in any cleaner sub-period. The forward scalar cascade persists
throughout."

## Caveats

- Three windows is the practical limit before per-bin counts fall too low;
  finer time-resolution is not supported by the data.
- The growth of $S_1$ with time is clearest for the two surveys with strong mean
  flows (Calypso 2019, ARCTERX); Calypso 2022's weak, near-isotropic flow shows
  only a modest late rise.
