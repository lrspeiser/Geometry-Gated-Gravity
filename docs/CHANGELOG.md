# Changelog

## [Unreleased] - G³ Submission Preparation

### Added
- Comprehensive cluster data documentation in `data/clusters/README.md`
- A1795, A478, A2029 cluster profiles with representative ACCEPT-quality data
- BCG/ICL profiles for all clusters using deprojected Sérsic + Hernquist models
- Cluster data validation script (`rigor/scripts/check_cluster_data.py`)
- BCG profile builder (`rigor/scripts/build_bcg_profile.py`)
- Micro-grid exponent scan helper (`rigor/scripts/run_cluster_exponent_scan.py`)
- SPARC CV summarizer script (`rigor/scripts/summarize_sparc_cv.py`)
- Lensing overlay generation from PDE field summaries

### Changed
- **Default comparator switched to total-baryon** (gas×√C + stars) for headline results
- Gas-only comparator now requires explicit `--gN_from_gas_only` flag (ablation)
- Geometry scalars (r_half, sigma_bar) computed from same total-baryon grid by default
- Auto-enable saturating mobility for total-baryon comparator to prevent overshoot
- Enhanced metrics recording: geometry mode, comparator mode, mass integrals, N_pts_scored
- Figure improvements: residual strip panel, scoring band shading, comparator callouts
- Output filenames now include `_ablation` and `_nonthermal` suffixes when applicable

### Fixed
- Deprecated `--gN_from_total_baryons` flag (now issues warning as it's the default)
- Proper fail-fast when stars_profile.csv missing for total-baryon comparator
- Non-thermal pressure logging (<f_nt> on scoring radii)

### Technical Notes
- Locked globals: S0=1.4e-4, rc_kpc=22, g0=1200, rc_gamma=0.5, sigma_beta=0.10
- Reference scales: rc_ref_kpc=30, sigma0_Msun_pc2=150
- Default grid: NR=128, NZ=128, Rmax=1500, Zmax=1500 for clusters
- Non-thermal parameters (when enabled): fnt0=0.2, fnt_n=0.8, r500_kpc=1000, fnt_max=0.3

### Pending Issues
- Total-baryon comparator producing unexpectedly large residuals - under investigation
- Field normalization may need adjustment for stellar component integration