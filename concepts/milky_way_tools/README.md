# Milky Way Tools Concept

This folder groups data preparation and evaluation utilities specific to the MW Gaia workflows.

Contents
- generate_mw_test_data.py: Create synthetic Gaia-like stars and a surface density table.
- convert_gaia_to_mw.py: Convert raw Gaia DR3 to Galactocentric coordinates and circular-speed observables.
- prepare_gaia_npz.py: Build NPZ inputs with interpolated Sigma_loc and Newtonian acceleration for GPU runs.
- test_mw_ad_correction.py: Tests for asymmetric drift policy and corrections.

Data sources
- Gaia CSVs: data/gaia_mw_real.csv (or your actual extracts)
- Surface density table: data/mw_sigma_disk.csv

Typical commands
- Generate synthetic test data:
  python concepts/milky_way_tools/generate_mw_test_data.py

- Convert raw Gaia to MW format:
  python concepts/milky_way_tools/convert_gaia_to_mw.py --input data/gaia_raw.csv --output data/gaia_mw_real.csv --quality_cuts

- Prepare NPZ inputs for GPU orchestrator:
  python concepts/milky_way_tools/prepare_gaia_npz.py
