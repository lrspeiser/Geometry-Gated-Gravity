# GravityCalculator — analysis and UI

This repository contains:
- A Qt desktop app (recommended UI) to explore galaxy rotation with density-based gravity adjustments.
- A PLUS interactive script using Matplotlib widgets.
- A SPARC analysis script that applies an inverse mass–gravity power-law to predict outer rotation speeds across galaxies.

Repository layout
- src/
  - gravity_qt_app.py — Qt desktop UI
  - toy_galaxy_rotation_calculator_plus.py — Matplotlib interactive UI
  - toy_galaxy_rotation_calculator.py — Minimal toy variant
  - scripts/
    - sparc_predict.py — SPARC analysis (mass–gravity power-law)
- data/ — input and output data files (parquet/csv/json)
- assets/ — images
- requirements.txt

Install
- macOS example using a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Qt desktop app
- Run the Qt app (PyQt6 preferred; falls back to PyQt5):

```
python src/gravity_qt_app.py
```

- The Qt app embeds Matplotlib for plotting and uses native controls.

PLUS interactive script

```
python src/toy_galaxy_rotation_calculator_plus.py
```

- Options include: --backend, --save, --config, --seed, --gscale, --vobs, etc. See the script header for details.

SPARC workflow (data, formula, outputs)
- Overview: We use SPARC rotation-curve products and per-galaxy baryonic masses to evaluate an inverse correlation between gravity scaling and baryonic mass.
- Formula applied:
  - G_pred = A × (M_bary / M0)^beta, with beta < 0 (inverse correlation)
  - Default parameters from our latest fit (excluding ultra-diffuse):
    - A = 5.589712133866334
    - beta = -0.6913962885091143
    - M0 = 1e10 Msun
  - Predicted speed: Vpred = sqrt(G_pred) × Vbar
    - Vbar is the baryonic curve, computed as sqrt(Vgas^2 + Vdisk^2 + Vbul^2)

- Getting SPARC data
  - Option 1 (provided here): Use the included data/sparc_rotmod_ltg.parquet for quick runs.
  - Option 2 (derive from SPARC MRT): Download SPARC_Lelli2016c.mrt from the SPARC Database (CWRU: https://astroweb.cwru.edu/SPARC/). The script supports --sparc-mrt and will derive M_bary and Rd heuristically using common columns (see script for details). You can also provide a mass CSV directly via --mass-csv. If your column names differ, rename them to the accepted names.

- Boundaries (outer-region scoring)
  - If no boundaries CSV is provided, a hybrid rule is used per galaxy:
    - boundary = min(max(3.2×Rd, 0.4×R_last), 0.8×R_last); if Rd missing, fallback = 0.5×R_last
  - We report GR performance (pure baryons, no scaling) alongside the mass–gravity prediction.

- Running the analysis

```
# From repo root; defaults point to data/
python src/scripts/sparc_predict.py \
  --parquet data/sparc_rotmod_ltg.parquet \
  --mass-csv /path/to/sparc_mass_table.csv \
  --boundary-frac 0.5 \
  --output-dir data
```

- If you have only the SPARC MRT file:
```
python src/scripts/sparc_predict.py \
  --parquet data/sparc_rotmod_ltg.parquet \
  --sparc-mrt /path/to/SPARC_Lelli2016c.mrt \
  --output-dir data
```

- Outputs (written to data/ unless overridden)
  - sparc_predictions_by_radius.csv
    - galaxy, type, R_kpc, boundary_kpc, is_outer, Vobs_kms, Vbar_kms, Vgr_kms, gr_percent_close, gr_failing, G_pred, Vpred_kms, percent_close
  - sparc_predictions_by_galaxy.csv
    - galaxy, type, M_bary_Msun, boundary_kpc, outer_points, median_percent_close, mean_percent_close, gr_failing_points, median_percent_close_on_gr_failing, mean_percent_close_on_gr_failing
  - boundaries.csv (for transparency when derived automatically)

Expected results
- For each galaxy, in the outer region (R ≥ boundary_kpc), Vpred should track Vobs more closely than pure GR baryons in many cases when beta < 0. See the CSVs for percent_close metrics.

Notes on implementation
- The SPARC script logs the formula and parameters used on each run.
- Project-relative defaults avoid absolute paths and write outputs to data/.
- Large binaries (.parquet, .png) are tracked in Git LFS.

Troubleshooting
- Parquet read requires pyarrow (installed via requirements.txt). If you see a parquet engine error, ensure pyarrow is installed.
- astropy is only needed when using --sparc-mrt.
- If Qt cannot initialize, try: pip install PyQt6 (or PyQt5), then rerun the Qt app.

Licensing and data sources
- SPARC data courtesy of Lelli et al. (2016) — see the SPARC site for terms. This repo does not redistribute SPARC proprietary content beyond derived products for local analysis.
