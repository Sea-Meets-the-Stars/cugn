# Figures in the paper and the code that makes them

Paper: `Projects/Overleaf/Energy_Cascade/energy_cascade.tex`. The figure numbers
come from a compile on 2026-11-17, and the PNGs live in the Overleaf `Figures/`
folder.

How to read the paths:
- **Figures/py** = `papers/Structure/Figures/py/`. Run these scripts from that
  directory. Each writes its PNG into the current directory.
- **Analysis/py** = `papers/Structure/Analysis/py/`. Run these scripts from that
  directory. Each writes its PNG to `../Figures_new/`, which is
  `Analysis/Figures_new/`.

After running a script, copy the PNG to the Overleaf `Figures/` folder.

## Main text

| Fig. | Label | File (Overleaf `Figures/`) | Code module | Function | How to run |
|---|---|---|---|---|---|
| 1 | `fig:qg_full` | `fig_full_qg_SF.png` | Figures/py/`figs_paper_structure.py` | `fig_full_qg_SF()` | `python figs_paper_structure.py 8` |
| 2 | `fig:cugn` | `fig_experiments.png` | Figures/py/`figs_paper_structure.py` | `fig_experiments()` | `python figs_paper_structure.py 1` |
| 3 | `fig:separations` | `fig_separations_Calypso2022.png` | Figures/py/`figs_paper_structure.py` | `fig_separations('Calypso2022')` | `python figs_paper_structure.py 2` |
| 4 | `fig:duL` | `fig_duL_Calypso2022.png` | Figures/py/`figs_paper_structure.py` | `fig_single_order('Calypso2022', order=1)` | `python figs_paper_structure.py 3` |
| 5 | `fig:duL2` | `fig_duL2_Calypso2022.png` | Figures/py/`figs_paper_structure.py` | `fig_single_order('Calypso2022', order=2)` | `python figs_paper_structure.py 4` |
| 6 | `fig:s3` | `fig_corrected_s3_Calypso2022.png` | Figures/py/`figs_paper_structure.py` | `fig_corrected_s3('Calypso2022', ...)` | `python figs_paper_structure.py 6` |
| 7 | `fig:s1s3other` | `fig_S1S3_other.png` | Figures/py/`figs_paper_structure.py` | `fig_S1S3_other(['Calypso2019', 'ARCTERX-2023'])` | `python figs_paper_structure.py 7` |
| 8 | `fig:qg_subregion` | `fig_qg_100km_vs_full.png` | Figures/py/`figs_paper_structure.py` | `fig_qg_subregion_vs_full(x0=300, y0=400)` | `python figs_paper_structure.py 9` |
| 9 | `fig:qg_timeevo` | `fig_qg_time_evolution.png` | Figures/py/`figs_qg_results.py` | `fig_qg_time_evolution()` (x0=300, y0=400) | `python figs_qg_results.py 1` |
| 10 | `fig:qg_ensemble` | `fig_qg_ensemble.png` | Figures/py/`figs_qg_results.py` (uses Analysis/py/`qg_ensemble.py`) | `fig_qg_ensemble(ndays=100)` | `python figs_qg_results.py 4` |
| 11 | `fig:qg_tracks` | `fig_qg_tracks.png` | Figures/py/`figs_qg_results.py` | `fig_qg_tracks()` | `python figs_qg_results.py 3` |
| 12 | `fig:qg_sampling` | `fig_qg_sampling.png` | Figures/py/`figs_qg_results.py` | `fig_qg_sampling()` | `python figs_qg_results.py 2` |
| 13 | `fig:scalar_mixed` | `fig_scalar_mixedT_z60.png` | Analysis/py/`figs_scalar.py` (calculations in `scalar_sf.py`) | `fig_mixed(all_res, ..., 'T')` | `python figs_scalar.py` |
| 14 | `fig:scalar_grad` | `fig_scalar_gradremoval_T_z60.png` | Analysis/py/`figs_followups.py` (calculations in `scalar_gradient.py`) | `fig_gradient_removal(grad, ..., 'T')` | `python figs_followups.py` |

## Appendix (Supplemental Materials)

| Fig. | Label | File (Overleaf `Figures/`) | Code module | Function | How to run |
|---|---|---|---|---|---|
| A1 | `fig:loglin` | `fig_loglin_sep_Calypso2022.png` | Figures/py/`figs_structure.py` | `fig_loglin_separations('Calypso2022', xmax=100.)` | flag 50 of `figs_structure.py`, **but the Calypso2022 call is commented out** (flag 50 now makes Calypso2019) |
| A2 | `fig:robust_iso` | `fig_d2_LT_z60.png` | Analysis/py/`figs_full_d3.py` (calculations in `full_d3.py`) | `fig_second_order(all_res, ...)` | `python figs_full_d3.py` |
| A3 | `fig:robust_d3` | `fig_d3_full_z60.png` | Analysis/py/`figs_full_d3.py` (calculations in `full_d3.py`) | `fig_third_order(all_res, ...)` | `python figs_full_d3.py` |
| A4 | `fig:robust_dof` | `fig_robust_dof_z60.png` | Analysis/py/`figs_robustness.py` (calculations in `robustness.py`) | `run_dof()` | `python figs_robustness.py` |
| A5 | `fig:robust_dt` | `fig_robust_dtscan_z60.png` | Analysis/py/`figs_robustness.py` (calculations in `robustness.py`) | `run_dt_scan()` | `python figs_robustness.py` |
| A6 | `fig:isopycnal` | `fig_isopycnal_S3_z60.png` | Analysis/py/`figs_followups.py` (calculations in `isopycnal.py`) | `fig_isopycnal(iso, ...)` | `python figs_followups.py` |
| A7 | `fig:depth` | `fig_depth_sweep.png` | Analysis/py/`figs_depth.py` (calculations in `depth_dep.py`) | `fig_depth(all_res, ...)` | `python figs_depth.py` |

## Notes

- **Flag comments in `figs_paper_structure.py`.** The comments in `main()` ("Figure 1",
  "Figure 4", ...) no longer match the paper's numbering. For example, flag 8
  makes paper Figure 1 and flag 9 makes Figure 8. Use the table above.
- **Two copies of `fig_qg_subregion_vs_full`.** Figure 8 comes from the copy in
  `figs_paper_structure.py` (flag 9, x0=300, y0=400). A near-duplicate with the
  same output name exists in `figs_structure.py`, where its default is x0=400,
  y0=400.
- **Figure A1** is generated by `figs_structure.py`, not `figs_paper_structure.py`,
  and its current `main()` call is set to Calypso2019. There is no local copy of
  `fig_loglin_sep_Calypso2022.png` in the repo; it exists only in Overleaf.
- **Stale copies.** Most local copies of these PNGs (in `Figures/`,
  `Analysis/Figures_new/` and `papers/Structure/Figures_new/`) are not
  byte-identical to the Overleaf versions. Treat Overleaf `Figures/` as the
  authoritative set. For example, `Figures/fig_qg_ensemble.png` is an older render;
  the current one is `Figures/py/fig_qg_ensemble.png`, which matches Overleaf.
- **QG data.** All QG figures read the regional SF files in `Analysis/Output/` and
  the full-domain files in `$OS_DATA/QG/`.
- **Glider data.** All glider figures load data through `Analysis/py/glider_io.py`.
