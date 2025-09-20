# Root‑M (√M) experimental law

This folder contains code and outputs for the "Root‑M" (√M) tail proposal, which scales the extra field with the enclosed baryonic mass:

v_tail^2(r) = A^2 * sqrt( M_b(<r) / M_ref ),   g_tail = v_tail^2 / r.

Goals
- Protect the existing pipeline and results (no edits to prior scripts)
- Provide parallel scripts to evaluate Root‑M on SPARC, the Milky Way, and clusters
- Store all Root‑M outputs under this folder

Defaults
- A_kms = 140.0 (km/s)
- Mref_Msun = 6.0e10 (∼MW baryonic scale)

Structure
- root-m/README.md (this file)
- root-m/common.py — shared helpers (v_tail2_rootm_soft)
- root-m/sparc_rootm.py — compute Root‑M(+soft) predictions for SPARC tables; optional K‑fold CV over (A, rc)
- root-m/mw_rootm.py — same logic for Milky Way tables (generic input path)
- root-m/cluster_hse_rootm.py — integral-HSE (no d ln n/dr) temperature and M(<r) for clusters using Root‑M(+soft) tail
- root-m/out/... — all generated outputs live here

Usage
1) SPARC (reads data/sparc_predictions_by_radius.csv):
   # Cross-validate (A, rc) once, then reuse everywhere
   py -u root-m/sparc_rootm.py --in data/sparc_predictions_by_radius.csv \
      --out_dir root-m/out/sparc_soft --cv 5 --A_grid 120,140,160,180 --rc_grid 10,15,20
   # Or run direct with a chosen pair
   py -u root-m/sparc_rootm.py --in data/sparc_predictions_by_radius.csv \
      --out_dir root-m/out/sparc_soft --A_kms 140 --rc_kpc 15 --cv 0 --Mref 6.0e10

2) Milky Way (point to your MW-by-radius CSV):
   py -u root-m/mw_rootm.py --in <path-to-mw-predictions-by-radius.csv> --out_dir root-m/out/mw --A_kms 140 --Mref 6.0e10

3) Clusters (expects data/clusters/<CLUSTER> with gas_profile.csv and temp_profile.csv):
   # Reuse the (A, rc) that won SPARC CV
   py -u root-m/cluster_hse_rootm.py --cluster ABELL_0426 --A_kms 140 --rc_kpc 15
   py -u root-m/cluster_hse_rootm.py --cluster ABELL_1689 --A_kms 140 --rc_kpc 15

Outputs
- SPARC: predictions_by_radius_rootm.csv, predictions_by_galaxy_rootm.csv, summary_by_type_rootm.csv, summary_rootm.json
- MW: predictions_by_radius_rootm.csv, summary_rootm.json (same schema idea)
- Clusters: root-m/out/clusters/<CLUSTER>/cluster_rootm_results.png and cluster_rootm_metrics.json

Notes
- Existing LogTail scripts and outputs remain untouched.
- Root‑M scripts compute their own metrics and write only to root-m/out.
