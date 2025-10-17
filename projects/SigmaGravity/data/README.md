# Data expectations for SigmaGravity replication

SPARC (galaxy branch)
- Root: external_data/Rotmod_LTG
- Required: MasterSheet_SPARC.mrt and many *_rotmod.dat files (standard SPARC distribution).
- The many_path_model/validation_suite.py expects this layout; do not move files.

Clusters (projected Σ-kernel branch)
- File: data/clusters/master_catalog.csv
- Columns include (minimum):
  - cluster_name, tier, z_lens, z_source, R_500_kpc, M_500_Msun,
  - theta_E_obs_arcsec, theta_E_err_arcsec,
  - fgas_R500, TX_central_keV
- See scripts/validate_holdout_mass_scaled.py and scripts/run_mass_scaled_emcee.py for usage.

Verification
- Run projects/SigmaGravity/data/verify_data.py (writes provenance/data_md5.json)
- The holdout validator also checks that the catalog MD5 matches the value embedded in the calibration posterior (manifest).
