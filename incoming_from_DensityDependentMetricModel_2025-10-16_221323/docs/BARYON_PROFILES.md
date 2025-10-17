# Baryon Profiles

Optional per-cluster projected baryon surface-density profiles can be provided to improve realism.

- Path: `data/baryon_profiles/{SANITIZED_NAME}.csv`
- Name sanitization: uppercase, remove spaces/hyphens/dots, replace `+` with `PLUS`.
  - Example: `MACS J1149.5+2223` → `MACSJ11495PLUS2223.csv`
- CSV columns (header required):
  - `R_kpc` (float): projected radius in kpc
  - `Sigma_baryon` (float): projected baryon surface density [Msun/kpc^2]

If a profile is present, it fully replaces the analytic fallback (gNFW gas + Hernquist BCG). If absent, the fallback is used and normalized to `f_b * M500` within `R500`.

This file is read by `scripts/baryon_loader.py` and used automatically in `scripts/run_mass_scaled_hierarchical_inference.py` and holdout validation via the shared builder.
