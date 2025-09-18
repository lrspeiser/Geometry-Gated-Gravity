# LogTail analysis notes

This file tracks code changes and how to reproduce the latest LogTail results, including SPARC joins, BTFR, RAR, outer slopes, and lensing checks.

Last updated: 2025-09-18

What changed in this update
- rigor/scripts/add_models_and_tests.py
  - Attach SPARC baryonic masses from data/sparc_master_clean.parquet when available.
- Export BTFR tables with valid M_bary_Msun and fit slope/scatter to btfr_logtail_fit.json. Added unit heuristic (1e10 Msun -> Msun) and positivity checks in fit.
  - Compute curved RAR stats (approx. orthogonal scatter) for observed and LogTail: rar_obs_curved_stats.json, rar_logtail_curved_stats.json.
  - Export outer-slope distribution for observed vs LogTail: outer_slopes_logtail.csv.
- Lensing comparison helper: lensing_logtail_comparison.json (requires a provided stack CSV if amplitude comparison is desired). Now robustly sorts/cleans arrays; slope is ~ -1 for the 1/R tail. Shape CSV remains lensing_logtail_shapes.csv.
  - Optional mass-coupled LogTail v0(Mb) = A*(Mb/1e10)^(1/4) with grid search; exports predictions_with_LogTail_mass_coupled.csv and corresponding BTFR.

How to run
1) Generate or provide per-radius predictions CSV (columns must include galaxy/gal_id, R_kpc, Vbar_kms, Vobs_kms). You can export LogTail/MuPhi-ready predictions via your existing pipeline or use rigor/scripts/export_predictions_by_radius.py for xi-based baselines.

2) Run the analysis and summaries:
   python rigor/scripts/add_models_and_tests.py \
     --pred_csv <path/to/predictions.csv> \
     --out_dir out/analysis/type_breakdown \
     --sparc_master_parquet data/sparc_master_clean.parquet \
     --mass_coupled_v0 \
     --lensing_stack_csv <optional: path/to/reference_lensing_stack.csv>

Outputs written to out/analysis/type_breakdown/
- predictions_with_LogTail_MuPhi.csv
- summary_logtail_muphi.json
- btfr_logtail.csv, btfr_muphi.csv, btfr_logtail_fit.json (now includes two-form reporting with bootstrap CIs: Mb vs v and v vs Mb, plus alpha_from_beta)
- btfr_observed.csv, btfr_observed_fit.json (observational BTFR sanity check), btfr_qc.txt (correlation check), btfr_mass_join_audit.csv (join audit)
- rar_logtail.csv, rar_muphi.csv, rar_obs_curved_stats.json, rar_logtail_curved_stats.json
- outer_slopes_logtail.csv
- lensing_logtail_shapes.csv, lensing_logtail_comparison.json
- (if --mass_coupled_v0) predictions_with_LogTail_mass_coupled.csv, summary_logtail_mass_coupled.json, btfr_logtail_mass_coupled.csv, btfr_logtail_mass_coupled_fit.json

Notes
- BTFR previously showed NaN masses; this run joins masses from data/sparc_master_clean.parquet. If missing, BTFR fit will report n=0.
- Lensing amplitude comparison requires a reference stack CSV with columns [R_kpc, DeltaSigma_Msun_per_kpc2]. If not provided, we still report predicted slope and amplitudes at 50/100 kpc.
- RAR curved stats use a median relation and approximate orthogonal scatter in log space; consider a more formal orthogonal regression as a future improvement.
- For cross-validated fairness vs ΛCDM (NFW c–M), no change yet—planned next.
