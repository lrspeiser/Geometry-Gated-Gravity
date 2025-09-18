# Data catalog for /data

This document classifies the contents of the data directory and lists the columns (schema) for each dataset. For families of files that repeat the same structure across many runs (e.g., opt_shell and shell_run outputs), the schema is documented once and referenced for all matching files.

Last updated: 2025-09-18

---

## Contents overview

Top-level families:
- SPARC-derived summaries and predictions (CSV, TXT)
- SPARC source tables and rotation-curve products (MRT, DAT, SFB, DENS, CSV, Parquet)
- GAIA sky slices (CSV, Parquet)
- BOSS DR12 consensus results (TXT) and covariances (TXT)
- Pantheon+SH0ES (DAT, COV)
- Optimizer outputs (opt_shell/*, shell_run/*) — replicate schemas of SPARC-derived CSVs
- Model results (CSV, JSON)
- Miscellaneous (images, logs, archives)

---

## SPARC-derived summaries and predictions (CSV/TXT)

The files below define the schemas used by many repeated outputs under opt_shell/* and shell_run/*.

- data/sparc_summary_by_type.csv
  Columns: Galaxy_Type, Galaxy_Class, Mass_Category, Galaxies, Avg_GR_Percent_Off, Median_GR_Percent_Off, Avg_Model_Percent_Off, Median_Model_Percent_Off, Median_G_Ratio_Outer, Mean_G_Ratio_Outer

- data/sparc_rotmod_summary.csv
  Columns: galaxy, R_kpc, Vobs_kms

- data/sparc_predictions_by_radius.csv
  Columns: galaxy, type, R_kpc, boundary_kpc, is_outer, Vobs_kms, Vbar_kms, Vgr_kms, gr_percent_close, gr_failing, G_pred, G_required, G_ratio, Vpred_kms, percent_close

- data/sparc_predictions_by_galaxy.csv
  Columns: galaxy, type, M_bary_Msun, boundary_kpc, outer_points, median_percent_close, mean_percent_close, avg_gr_percent_off, median_gr_percent_off, avg_model_percent_off, median_model_percent_off, gr_failing_points, median_percent_close_on_gr_failing, mean_percent_close_on_gr_failing, median_G_Required_Outer, mean_G_Required_Outer, median_G_Ratio_Outer, mean_G_Ratio_Outer

- data/sparc_missing_masses.csv
  Columns: galaxy, galaxy_key

- data/sparc_human_by_radius.csv
  Columns: Galaxy, Galaxy_Type, Galaxy_Class, Radius_kpc, Boundary_kpc, In_Outer_Region, Observed_Speed_km_s, Baryonic_Speed_km_s, GR_Speed_km_s, GR_Percent_Off, G_Predicted, Predicted_Speed_km_s, Model_Percent_Off, Baryonic_Mass_Msun

- data/sparc_human_by_galaxy.csv
  Columns: Galaxy, Galaxy_Type, Galaxy_Class, Baryonic_Mass_Msun, Mass_Category, Boundary_kpc, Outer_Points_Count, Avg_GR_Percent_Off, Median_GR_Percent_Off, Avg_Model_Percent_Off, Median_Model_Percent_Off, Median_G_Required_Outer, Mean_G_Required_Outer, Median_G_Ratio_Outer, Mean_G_Ratio_Outer

- data/sparc_greq_vs_mass.csv
  Columns: Galaxy, Galaxy_Type, Galaxy_Class, Baryonic_Mass_Msun, Median_G_Required_Outer, Mean_G_Required_Outer, Median_G_Ratio_Outer, Mean_G_Ratio_Outer

- data/boundaries.csv
  Columns: galaxy, boundary_kpc

- data/sparc_greq_mass_fit.txt
  Key-value pairs (one per line): A_fit, beta_fit

Notes on replicated outputs:
- opt_shell/*/* and shell_run/* contain repeated outputs with identical schemas to the top-level files listed above (e.g., sparc_predictions_by_radius.csv, sparc_predictions_by_galaxy.csv, sparc_summary_by_type.csv, sparc_human_by_radius.csv, sparc_human_by_galaxy.csv, sparc_missing_masses.csv, sparc_greq_vs_mass.csv, sparc_greq_mass_fit.txt, boundaries.csv). Only parameter presets differ by directory; the column structures are the same.

---

## SPARC source tables and rotation-curve products

Parquet tables:
- data/sparc_master_clean.parquet
  Columns: galaxy_key, galaxy, T, D, L36, MHI, Rdisk, M_bary, source_file

- data/sparc_rotmod_ltg.parquet
  Columns: galaxy, R_kpc, Vobs_kms, eVobs_kms, Vgas_kms, Vdisk_kms, Vbul_kms

- data/sparc_all_tables.parquet
  Columns: Galaxy, T, D, e_D, f_D, Inc, e_Inc, L36, e_L36, Reff, SBeff, Rdisk, SBdisk, MHI, RHI, Vflat, e_Vflat, Q, Ref, galaxy, galaxy_key, M_bary, source_file, R, Vobs, e_Vobs, Vgas, Vdisk, Vbul, SBbul

Machine-readable tables (MRT) from Lelli et al. (2016):
- data/SPARC_Lelli2016c.mrt — Byte-by-byte descriptions present. Key labels include: Galaxy, T, D, e_D, f_D, Inc, e_Inc, L[3.6], e_L[3.6], Reff, SBeff (plus additional fields as documented in the file).
- data/Rotmod_LTG/MasterSheet_SPARC.mrt — Same documentation style as above.
- data/Rotmod_LTG/MassModels_Lelli2016c.mrt — Byte-by-byte example labels: ID, D, R, Vobs, e_Vobs, Vgas, Vdisk, Vbul, SBdisk, SBbul.

Rotation-curve products per galaxy:
- Rotmod_LTG/*.rotmod.dat
  Header units then tab-separated columns:
  Columns: Rad (kpc), Vobs (km/s), errV (km/s), Vgas (km/s), Vdisk (km/s), Vbul (km/s), SBdisk (L/pc^2), SBbul (L/pc^2)

- Rotmod_LTG/*.sfb
  Columns: radius, mu, kill, error

- Rotmod_LTG/*.dens
  Columns: Rad[kpc], SBdisk[Lsun/pc^2], SBbulge[Lsun/pc^2]

Bulge/disk decompositions:
- BulgeDiskDec_LTG/*.dens
  Columns: Rad[kpc], SBdisk[Lsun/pc^2], SBbulge[Lsun/pc^2]

Auxiliary CSV:
- Rotmod_LTG/download_summary.csv
  Columns: galaxy, HIrad_local, SB_local, ROTMOD_local

- Rotmod_LTG/MasterSheet_SPARC.csv — note: this is a textual export with title metadata rather than a structured data table.

---

## GAIA sky slices

Raw sky-slice CSVs and all-sky CSV:
- gaia_sky_slices/all_sky_gaia.csv
- gaia_sky_slices/raw_L###-###.csv (e.g., raw_L000-030.csv, raw_L030-060.csv, ...)
  Columns: source_id, ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, radial_velocity, radial_velocity_error, ruwe, phot_g_mean_mag, b, l

Processed parquet per longitude bin:
- gaia_sky_slices/processed_L###-###.parquet (e.g., processed_L000-030.parquet, ..., processed_L330-360.parquet)
  Columns: source_id, X_gc_kpc, Y_gc_kpc, Z_gc_kpc, R_kpc, phi_rad, z_kpc, v_R_kms, v_phi_kms, v_z_kms, v_obs, sigma_v, quality_flag

---

## BOSS DR12 consensus

Results (TXT; whitespace-separated rows with three fields):
- data/BOSS/DR12_consensus/COMBINEDDR12_BAO_consensus_dM_Hz/BAO_consensus_results_dM_Hz.txt
- data/BOSS/DR12_consensus/COMBINEDDR12_FS_consensus_dM_Hz/FS_consensus_results_dM_Hz_fsig.txt
- data/BOSS/DR12_consensus/COMBINEDDR12_final_consensus_dM_Hz/final_consensus_results_dM_Hz_fsig.txt
  Row format: z_bin, observable_label, value
  Examples of observable_label: dM(rsfid/rs), Hz(rs/rsfid), fsig8

Covariance/auxiliary matrices (TXT; whitespace-separated, typically symmetric matrices or tabulations):
- data/BOSS/DR12_consensus/**/cov*txt (e.g., FS_GRIEB_cov_z1.txt, FS_BEUTLER_cov_z3.txt, FS_consensus_covtot_dM_Hz_fsig.txt, final_consensus_covtot_dM_Hz_fsig.txt, BAO_consensus_covtot_dM_Hz.txt)
  Structure: numeric matrix; no header row.

Archive:
- data/BOSS/ALAM_ET_AL_2016_consensus_and_individual_Gaussian_constraints.tar.gz — upstream archive of the BOSS consensus products.

---

## Pantheon+SH0ES

- data/pantheon/Pantheon+SH0ES.dat
  Columns (as documented in header): CID, IDSURVEY, zHD, zHDERR, zCMB, zCMBERR, zHEL, zHELERR, m_b_corr, m_b_corr_err_DIAG, MU_SH0ES, MU_SH0ES_ERR_DIAG, CEPH_DIST, IS_CALIBRATOR, USED_IN_SH0ES_HF, c, cERR, x1, x1ERR, mB, mBERR, x0, x0ERR, COV_x1_c, COV_x1_x0, COV_c_x0, RA, DEC, HOST_RA, HOST_DEC, HOST_ANGSEP, VPEC, VPECERR, MWEBV, HOST_LOGMASS, HOST_LOGMASS_ERR, PKMJD, PKMJDERR, NDOF, FITCHI2, FITPROB, m_b_corr_err_RAW, m_b_corr_err_VPEC, biasCor_m_b, biasCorErr_m_b, biasCor_m_b_COVSCALE, biasCor_m_b_COVADD

- data/pantheon/Pantheon+SH0ES_STAT+SYS.cov
  Structure: covariance matrix, no header row.

---

## Optimizer outputs (opt_shell/* and shell_run/*)

These directories contain many runs with parameterized presets. Each directory includes CSV/TXT files that mirror the schemas of the top-level SPARC-derived outputs:
- sparc_summary_by_type.csv — same columns as data/sparc_summary_by_type.csv
- sparc_predictions_by_radius.csv — same columns as data/sparc_predictions_by_radius.csv
- sparc_predictions_by_galaxy.csv — same columns as data/sparc_predictions_by_galaxy.csv
- sparc_missing_masses.csv — same columns as data/sparc_missing_masses.csv
- sparc_human_by_radius.csv — same columns as data/sparc_human_by_radius.csv
- sparc_human_by_galaxy.csv — same columns as data/sparc_human_by_galaxy.csv
- sparc_greq_vs_mass.csv — same columns as data/sparc_greq_vs_mass.csv
- sparc_greq_mass_fit.txt — same keys as data/sparc_greq_mass_fit.txt
- boundaries.csv — same columns as data/boundaries.csv

Summary CSV across runs:
- data/opt_shell/opt_summary.csv
  Columns: mass_exp, middle, max, out_dir, point_mean_close, point_median_close, point_frac_ge90, gal_mean_of_medians, gal_median_of_medians, gal_frac_med_ge90, point_avg_gr_off, point_avg_model_off, point_improve_off, point_frac_better_than_gr

Note: data/shell_run/* contains a single snapshot of outputs from a prior run; schemas are identical to the top-level and opt_shell variants.

---

## Model results

- data/model_results/model_comparison.csv
  Columns: Model, Train Score, Test Score, Within 10%, Within 20%, Median Error %, Parameters

- data/model_results/best_model.json
  Top-level keys: model_name, parameters, param_bounds, train_score, test_score
  parameters fields: G_shell1, G_shell2, r1_frac, r2_frac, width1, width2

---

## Boundaries and solutions (JSON)

- data/boundaries.json
  Structure: mapping of galaxy class/category -> default boundary_kpc
  Example keys: "MW-like disk", "Central-dominated", "Dwarf disk", "LMC-like dwarf", "Ultra-diffuse disk", "Compact nuclear disk"

- data/solutions.json
  Structure: mapping of Preset -> parameter dict
  Example (Preset = "MW-like disk") keys: Preset, Vobs, G, Dk, Dr, Wk, Wr

---

## Miscellaneous data and media

- data/g_vs_mass.csv
  Columns: preset, boundary_R_kpc, G_solved, M_bary_Msun, G_over_M, vF_kms, percent_close

- data/rotation_curve.csv
  Columns: R_kpc, v_newton_kms, v_gr_no_boost_kms, v_final_kms, G_scale, GR_k_toy, extra_atten, boost_fraction, softening_kpc

- data/compare_presets.csv
  Columns: preset, N, err_fixed, err_solved, g, dens_k, dens_R, well_k, well_R, time_s

- data/sept15presettest.csv
  Columns: preset, N, err_fixed, err_solved, close_fixed_pct, close_solved_pct, g, dens_k, dens_R, well_k, well_R, g_req_fixed, g_req_solved, time_s

- data/sparc_run_master.log — log output from a SPARC processing run

- Images (figures/plots):
  data/camb_sparc.png, data/camb_anim_smoke.png, data/ddo161_anim_smoke.png, data/eso079_sparc.png, data/ddo161_sparc.png, data/greq_vs_mass.png, data/model_results/model_examples.png

- System/auxiliary:
  data/.DS_Store, data/opt_shell/.DS_Store — system metadata files (can be ignored)

---

## Notes

- CSV schemas are derived from header rows present in the files.
- Parquet schemas are read from file metadata (pyarrow). MRT, DAT, SFB, and DENS formats include header rows or byte-by-byte descriptions that define columns and units.
- opt_shell and shell_run directories contain many parameterized runs; for brevity we document the schema once and reference it for all matching filenames.
- If you add a new dataset, please add it to this catalog with its columns and a short description.
