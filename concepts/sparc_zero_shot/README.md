# SPARC Zero‑Shot Concept

This folder groups the zero‑shot SPARC workflow: sweeping formulas, validating without per‑galaxy tuning, and searching for unified parameters.

Contents
- sparc_formula_sweep.py: Explore formula variants on all SPARC galaxies simultaneously.
- sparc_zero_shot_validation.py: LOTO, stratified CV, bootstrap stability; robust metrics and filtering.
- optimize_sparc_unified.py: Unified parameter compromise for SPARC (and MW optional).
- sparc_zero_shot_runner.py, zero_shot_test.py: Thin entry points/quick tests.

Data sources
- SPARC parquet: data/sparc_rotmod_ltg.parquet
- SPARC master/meta: data/sparc_master_clean.parquet or data/Rotmod_LTG/MasterSheet_SPARC.csv

Typical commands
- Formula sweep (GPU if available via g3_gpu_solver_suite):
  python concepts/sparc_zero_shot/sparc_formula_sweep.py --max_variants 30

- Zero‑shot validation suite:
  python concepts/sparc_zero_shot/sparc_zero_shot_validation.py --all --max_iter 100

Expected outputs
- out/zero_shot_validation/* and JSON/CSV summaries next to scripts