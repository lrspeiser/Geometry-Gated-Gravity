# SigmaGravity Replication Hub

Purpose
- Turnkey replication folder for Σ‑Gravity results (galaxies: SPARC path‑spectrum; clusters: projected Σ‑kernel with triaxiality and BCG).
- Self‑contained docs, env files, wrappers, provenance tools, and regression matrix.
- Non‑intrusive: does not move/modify core code; wrappers call existing drivers in the repository.

Repository layout (this folder)
- README.md (this file)
- requirements.txt, environment.yml
- .gitattributes (LFS for outputs/images/results here)
- scripts/ … PowerShell and bash wrappers for all runs
- config/ … paper knobs and the exact cluster calibration setup
- data/ … data expectations + verifier
- provenance/ … manifest schema + aggregator
- tests/ … regression matrix + harness
- results/, images/ … outputs written by the harness/tools

A. What you need (software, data, repo state)
A1. Repository & versioning
- Repo: this working GravityCalculator repository with SigmaGravity under projects/.
- Commit/tag: cite the paper commit once finalized (placeholder: clusters‑N=6‑Pzs‑BCG). The provenance aggregator records the code hash from manifests produced by the drivers.
- Submodules/historical code: not required here; this hub calls the canonical drivers already present.

A2. Python environment
- Create a clean environment (conda or venv). See requirements.txt and environment.yml in this folder.
  Packages: python>=3.10; numpy, scipy, pandas, numexpr; matplotlib, corner; astropy; emcee; pyyaml. Optional: h5py, scikit‑learn.

A3. Data (with provenance)
- SPARC
  - external_data/Rotmod_LTG/MasterSheet_SPARC.mrt and the *_rotmod.dat files in the same tree (see data/README.md).
- Clusters (CLASH‑like)
  - data/clusters/master_catalog.csv (see schema in data/README.md).
  - Mass/radius per catalog description; θ_E and z values per references.
- Cosmology: flat ΛCDM (H0=70, Ωm=0.3) as used by the existing drivers.
- Guardrails: The drivers write run manifests (including catalog MD5). The provenance tools here will validate.

B. What to run (end‑to‑end)
Use the wrappers in projects/SigmaGravity/scripts (Windows .ps1; bash .sh included).

B1. Galaxy (SPARC) pipeline — unchanged physics branch
- Sanity/physics: scripts/run_sparc_validation.ps1  (calls many_path_model/validation_suite.py --all)

B2. Cluster pipeline (Σ‑kernel, triaxial, BCG)
- Diagnostics (MACS0416): scripts/macs0416_diagnostics.ps1 and scripts/simple_einstein_check.ps1
- Population calibration (PyMC, paper variant selection):
  scripts/run_cluster_calibration_pymc_paper.ps1
- Convert trace for validator (auto-run by calibration): projects/SigmaGravity/output/pymc_mass_scaled/flat_samples_from_pymc.npz
- Blind hold‑out validation (paper selection): scripts/validate_holdout_paper.ps1
- Emcee alternative (legacy): scripts/run_cluster_calibration_paper.ps1 → scripts/validate_holdout.ps1

Notes
- The current drivers in this repo include triaxial projection and BCG. Source‑z distribution P(z_s) is supported via --pzs; the harness reads projects/SigmaGravity/config/paper_settings.yaml.
- Known issue under investigation: using lognormal P(z_s) here produced unphysically large θ_E due to an apparent Σ_crit_eff bug; median z_s passes and is set in the config for replication while we patch the distribution path. See projects/SigmaGravity/provenance/manifests_index.json and output/holdout_validation_* for evidence.

C. Script map (where things live)
- Projected Σ‑kernel (clusters): core/kernel2d_sigma.py (local‑coherence normalization baked in this repo’s implementation)
- Triaxial projection: core/triaxial_lensing.py
- Gas/BCG profiles: core/gnfw_gas_profiles.py, core/bcg_profiles.py
- NFW conversions: core/nfw_mass_conversion.py
- Cluster calibration drivers: scripts/run_mass_scaled_emcee.py, scripts/validate_holdout_mass_scaled.py, scripts/plot_macs0416_diagnostics.py, scripts/simple_einstein_check.py
- Galaxy branch: many_path_model/validation_suite.py (and its modules)
- Provenance: calibration writes flat_samples.npz (with manifest); this hub validates MD5 and aggregates manifests.

D. Configuration knobs to match (paper)
- Kernel mode: projected Σ‑kernel for clusters; path‑spectrum for galaxies.
- Normalization: local‑coherence (cluster kernel).
- Amplitude prior: μ_A in ~[2,8].
- Source‑z distribution: enable P(z_s) when exposed upstream; for now drivers use catalog z_s.
- Geometry grid: 13×13 (q_plane, q_LOS) in the paper; current driver uses a fixed 13×13 grid internally.
- BCG: enabled (adds inner mass).

E. Expected headline numbers (sanity)
- Galaxy branch: Newtonian‑limit and symmetry tests pass; RAR/BTFR consistent with manuscript.
- Cluster branch: posterior medians near μ_A≈4–6, σ_A≈1–2, ℓ0⋆≈200 kpc, weak γ; A1689 holds; MACS1149 low.

F. Regression‑test matrix
- See tests/REGRESSION_MATRIX.md and scripts/run_regressions.ps1. The harness will:
  1) verify data, 2) SPARC validation, 3) MACS0416 diagnostics, 4) simple Einstein check, 5) calibration, 6) hold‑outs, 7) triaxial validation, 8) manifest aggregation, and 9) print a PASS/FAIL summary.

G. Reproducing figures/tables
- MACS0416 diagnostics → scripts/macs0416_diagnostics.ps1 (boost_profile.png, convergence profiles/maps, cumulative mass)
- Triaxial sensitivity panels → scripts/validate_triaxial_lensing.ps1
- Calibration posteriors (PyMC) → scripts/run_cluster_calibration_pymc_paper.ps1 (trace.nc + posterior_summary.csv under projects/SigmaGravity/output/pymc_mass_scaled/)
- Holdout predicted vs observed → scripts/generate_paper_figures.py (saves figures under projects/SigmaGravity/figures/)

H. Pitfalls and guardrails
- Wrong posterior for hold‑outs → validator checks run manifest and catalog MD5
- Accidental global normalization → this repo’s kernel uses local coherence; wrappers echo settings
- Geometry signal loss → we use projected 2D kernel with triaxial Σ input
- Amplitude prior drift → configs document intended ranges

I. Paper status and updates
- This hub tracks the live drivers; small script/config changes may happen as results finalize. We do not remove or move any content; we only add wrappers/docs under projects/SigmaGravity/ so a separate repo export is straightforward later.

One‑screen quick start (Windows PowerShell)
1) Environment
  python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1; pip install -r projects\\SigmaGravity\\requirements.txt
2) Verify data
  python projects\\SigmaGravity\\data\\verify_data.py
3) Run full replication harness
  projects\\SigmaGravity\\scripts\\run_replication.ps1
4) Or run individual steps
  - SPARC:   projects\\SigmaGravity\\scripts\\run_sparc_validation.ps1
  - Calib:   projects\\SigmaGravity\\scripts\\run_cluster_calibration_paper.ps1
  - Holdout: projects\\SigmaGravity\\scripts\\validate_holdout.ps1

Provenance & manifests
- Calibration writes a manifest alongside outputs; PyMC calibration also writes trace.nc and posterior_summary.csv. Use provenance/collect_manifests.py to build projects/SigmaGravity/provenance/manifests_index.{json,csv}. The tool checks catalog MD5 mismatches.

Data verifier
- Run projects\SigmaGravity\data\verify_data.py to confirm SPARC layout and catalog presence; produces MD5s for audit.
