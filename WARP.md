# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Repository: GravityCalculator (Windows, PowerShell)

- Languages and tools: Python 3, PyQt6/PyQt5, NumPy, Matplotlib, Pandas, PyArrow; optional JAX/NumPyro under rigor/
- Data: Large SPARC datasets checked into data/ via Git LFS (parquet, png, mrt, etc.)
- Preferred CLI on Windows: use fd and rg for search; avoid PowerShell-specific wrappers

Quick start

- Install Python deps (create/activate your venv first if desired):
  - python -m pip install --upgrade pip
  - python -m pip install -r requirements.txt
- Make sure Git LFS is set up to fetch large files:
  - git lfs install
  - git lfs pull

Common commands (run from repo root)

- Run the Qt desktop app (recommended UI)
  - python src/gravity_qt_app.py
  - Notes: Uses Matplotlib QtAgg backend and PyQt6 if available (falls back to PyQt5). If Qt cannot initialize, install PyQt6: python -m pip install PyQt6

- Run the PLUS interactive script (Matplotlib widget UI)
  - python src/toy_galaxy_rotation_calculator_plus.py

- SPARC predictor (power-law G(M) or shell model)
  - Power-law (see README for formula and defaults):
    - python src/shell_model/sparc_predict.py --parquet data/sparc_rotmod_ltg.parquet --mass-csv <path/to/masses.csv> --output-dir data
  - Shell model (smooth radial enhancement with mass scaling):
    - python src/shell_model/sparc_predict.py --parquet data/sparc_rotmod_ltg.parquet --sparc-master-csv data/Rotmod_LTG/MasterSheet_SPARC.csv --model shell --output-dir data
  - Boundaries and MRT options:
    - ... --boundaries-csv data/boundaries.csv
    - ... --boundary-frac 0.5    # if boundaries CSV not provided
    - ... --sparc-mrt data/SPARC_Lelli2016c.mrt    # derive masses/Rd from MRT when mass CSV not supplied
  - Outputs are written into data/ by default: sparc_predictions_by_radius.csv, sparc_predictions_by_galaxy.csv, human-friendly CSVs

- Build SPARC parquet inputs from MRT (legacy builder)
  - python src/legacy/build_sparc_parquet.py
  - Produces: data/sparc_all_tables.parquet and data/sparc_master_clean.parquet

- Rigor (hierarchical Bayesian, vendored package under rigor/)
  - One-time editable install (Option B):
    - python -m pip install -e rigor
  - Compare baselines (GR/MOND/Burkert):
    - python -m rigor --parquet data/sparc_rotmod_ltg.parquet --master data/Rotmod_LTG/MasterSheet_SPARC.csv --out_json out/baselines_summary.json --outer
  - SPARC fit (outer-only example; adjust platform and sampling as needed):
    - python -m rigor.fit_sparc --parquet data/sparc_rotmod_ltg.parquet --master data/Rotmod_LTG/MasterSheet_SPARC.csv --xi shell_logistic_radius --outer sigma --sigma_th 10 --use_outer_only --outdir out/xi_shell_logistic_radius --platform cpu --warmup 1500 --samples 1500 --chains 4
  - Posterior overlays (68% bands):
    - python -m rigor.plotting --post out/xi_shell_logistic_radius/posterior_samples.npz --figdir out/xi_shell_logistic_radius/figs --parquet data/sparc_rotmod_ltg.parquet --master data/Rotmod_LTG/MasterSheet_SPARC.csv --outer sigma --sigma_th 10

- Tests
  - These test scripts are run directly with Python (no pytest configuration in this repo):
    - Single test: python tests/check_matching.py
    - Single test: python tests/check_build_parquet.py
  - Purpose:
    - check_matching.py verifies robust name matching and MRT parsing against the rotmod parquet
    - check_build_parquet.py sanity-checks outputs from the legacy builder

- Search the repo (Windows, prefer Rust tools)
  - Find files: fd -H -t f "<pattern>" .
  - Search content: rg -n --hidden "<needle>" .

Makefile targets (if make is available)

Windows may not have make installed; you can either use a make-compatible shell or invoke the underlying Python commands shown above. If you do have make, useful targets include:

- make run           # power-law predictor with mass CSV -> data/
- make run-master    # predictor using MasterSheet_SPARC.csv -> data/
- make run-mrt       # predictor deriving from SPARC_Lelli2016c.mrt -> data/
- make run-shell     # shell model example with tuned params -> data/
- make optimize-shell
- make plot-overlays IN_DIR="..." OUT_DIR="..."
- make run-density-models     # legacy density model sweep
- make build-parquet          # build sparc_* parquet files
- make install-rigor          # pip install -e rigor
- make fit-bayes              # SPARC hierarchical fit (outer-only)
- make fit-bayes-joint        # joint SPARC + Milky Way fit
- make compare-baselines
- make bayes-overlays

High-level architecture and data flow

- UI and toy galaxy engine (src/gravity_qt_app.py)
  - Purpose: Desktop Qt UI for exploring a density-dependent gravity toy model and comparing “Final” vs “GR-like” speeds
  - Core types:
    - DensityKnobs: user-adjustable parameters (g_scale, dens_k, dens_R, dens_alpha, dens_thresh_frac, well_k, well_R, boundary_R)
    - GravityModel: builds a synthetic galaxy mass distribution (disk + bulge), caches per-source modifiers, and computes per-body acceleration vectors
      - compute_G_eff(p, knobs): local-density-based enhancement at an evaluation point p using global/reference density (rho_ref) and thresholding
      - well_scales(knobs): per-source scaling based on each star’s local density “well”; cached for performance and gated by boundary_R
      - per_body_vectors(p, knobs): per-body contributions and resulting acceleration at p; speeds computed via circular_speed_from_accel
  - UI structure:
    - MainWindow wires Qt controls to knobs, runs golden-section solvers to match a target observed velocity, overlays low-density rings, and exports rotation curves/solutions
    - CompareAcrossPresetsDialog/CompareWorker batch-evaluate presets, optionally re-solving parameters per preset and exporting CSV
  - Data flow: build synthetic positions/masses -> compute per-point density and effective G -> integrate forces -> derive v(R) curves and percent closeness to target

- SPARC predictor (src/shell_model/sparc_predict.py)
  - Purpose: Batch analysis over real SPARC rotation curves; supports two families:
    - Power-law scaling: Vpred = sqrt(G_pred) * Vbar with G_pred = A * (M_bary / M0)^beta
    - Shell model: smooth inner/middle/outer radial enhancement with mass-capped growth
  - Key components:
    - norm_name: canonicalize galaxy identifiers across sources
    - load_masses / parse_sparc_mrt: ingest masses and optional Rd from user CSV, MasterSheet_SPARC (CSV/MRT), or MRT table(s) directly
    - load_boundaries: supply per-galaxy outer-region boundaries (CSV or fraction of R_last)
    - compute_G_enhanced: shell-model radial law with mass-dependent caps and smooth transitions
  - Inputs/outputs:
    - Inputs: data/sparc_rotmod_ltg.parquet (or built from MRT), masses via CSV or derived, optional boundaries
    - Outputs: per-radius and per-galaxy CSVs in data/ with outer-region scoring and human-friendly variants
  - Note: README examples refer to src/scripts/sparc_predict.py; in this repository the implemented path is src/shell_model/sparc_predict.py

- Legacy parquet builder (src/legacy/build_sparc_parquet.py)
  - Purpose: Derive clean parquet tables from SPARC MRT files by auto-parsing “Byte-by-byte” headers, with alignment heuristics for fixed-width tables
  - Outputs: sparc_all_tables.parquet (union of parsed rows) and sparc_master_clean.parquet (per-galaxy essentials with M_bary and galaxy_key)

- Rigor pipeline (rigor/)
  - Purpose: Higher-rigor, hierarchical Bayesian fitting and posterior analysis implemented as a vendored package
  - Entrypoints: python -m rigor.fit_sparc, python -m rigor.plotting, python -m rigor (baselines), python -m rigor.fit_joint_mw
  - Dependencies: JAX/NumPyro (CPU/GPU depending on your platform); install editable with python -m pip install -e rigor
  - See docs/rigor.md for run recipes and GPU notes

Data and Git LFS

- Many artifacts are tracked under Git LFS (.gitattributes): *.parquet, *.png, *.mrt, *.npz, *.nc, etc.
- Before running analyses that touch data/, ensure:
  - git lfs install
  - git lfs pull

Notes and gotchas

- Windows specifics:
  - Use python instead of python3 in commands. If make is not available, call the underlying Python scripts directly.
  - For repo/file search in Windows shells, prefer Rust CLIs:
    - fd -H -t f "<pattern>" .
    - rg -n --hidden "<needle>" .
- README vs script defaults:
  - The README documents a specific power-law (A, beta). The current src/shell_model/sparc_predict.py ships its own defaults and also supports a shell model. Override A/beta via CLI if you need the README-stated values.
- Paths:
  - Examples in README reference src/scripts/sparc_predict.py; the active implementation lives in src/shell_model/sparc_predict.py.

Git conventions for this repo

- Large files are managed with Git LFS; avoid committing binaries without LFS filters
- The working preference is to commit directly to main (no long-lived branches). When making code changes, include clear commit messages.

Further references

- README.md: project overview, SPARC workflow, and UI notes
- docs/rigor.md and rigor/README.md: hierarchical pipeline usage
- Makefile: curated recipes for common analyses

