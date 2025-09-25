# Universal G³ Concept

This folder groups the unified G³ model (single global parameter set), its production runs, and validation reports and figures.

Contents
- g3_unified_global.py: Unified G³ global law implementation (derives inner/outer behavior from baryon geometry). Supports GPU via CuPy where available.
- g3_universal_fix.py: Universal model with fixes (provable Newtonian limit, C² continuity, screening/gating fixes).
- run_universal_production.py: End‑to‑end production run over MW + SPARC with analysis.
- run_universal_search.py, run_full_unified_tests.py: Search/validation utilities.
- verify_and_plot_results.py: Comparison plots vs Gaia, GR baseline; distribution diagnostics.
- Reports: ACTUAL_PERFORMANCE_REPORT.md, FINAL_REPORT.md, FINAL_UNIFIED_G3_RESULTS.md, UNIVERSAL_G3_SUMMARY.md
- Figures/results: universal_g3_complete.png, universal_g3_mw_details.png, universal_g3_production_results.png, universal_optimization_results.json
- Tests: test_universal_formula.py, test_unified_g3_accuracy.py, test_universal_g3_complete.py

Data sources
- MW (Gaia): data/gaia_mw_real.csv or data/mw_gaia_144k.npz (see concepts/milky_way_tools)
- SPARC: data/sparc_rotmod_ltg.parquet, data/Rotmod_LTG/MasterSheet_SPARC.csv

Typical commands
- Production (mixed MW + SPARC):
  python concepts/universal_g3/run_universal_production.py

- Validation and plots:
  python concepts/universal_g3/verify_and_plot_results.py

Expected outputs
- Figures under this folder and/or out/mw_orchestrated
- JSON parameter snapshots and summary metrics
