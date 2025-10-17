# SigmaGravity replication plan (mapped to manuscript)

Scope
- This plan maps the latest manuscript (docs/paper.md) to runnable commands under projects/SigmaGravity/scripts. Use the PyMC path to match the paper.

Step 0: Environment
- python -m venv .venv && .\.venv\Scripts\Activate.ps1 && pip install -r projects\SigmaGravity\requirements.txt

Step 1: Data verifier
- python projects\SigmaGravity\data\verify_data.py
- Requires: data/clusters/master_catalog.csv; optional SPARC data.

Step 2: Physics validations
- SPARC suite: projects\SigmaGravity\scripts\run_sparc_validation.ps1
- Triaxial suite: projects\SigmaGravity\scripts\validate_triaxial_lensing.ps1
- MACS0416 diagnostics: projects\SigmaGravity\scripts\macs0416_diagnostics.ps1

Step 3: Paper calibration (PyMC)
- Configure selection: projects\SigmaGravity\config\cluster_selection_paper.yaml (variant: current_catalog|paper)
- Run: projects\SigmaGravity\scripts\run_cluster_calibration_pymc_paper.ps1
- Outputs: projects/SigmaGravity/output/pymc_mass_scaled/{trace.nc, posterior_summary.csv, manifest.json, flat_samples_from_pymc.npz}

Step 4: Blind holdouts (paper variant)
- projects\SigmaGravity\scripts\validate_holdout_paper.ps1
- Output: projects/SigmaGravity/output/holdout_paper/holdout_results.json (+ plot)

Step 5: Figures
- python projects\SigmaGravity\scripts\generate_paper_figures.py
- Output: projects/SigmaGravity/figures/holdouts_pred_vs_obs.png

Optional: emcee path
- Calibration: projects\SigmaGravity\scripts\run_cluster_calibration_paper.ps1
- Holdouts: projects\SigmaGravity\scripts\validate_holdout.ps1

Notes
- P(z_s) is controlled via projects/SigmaGravity/config/paper_settings.yaml (mode: median by default pending Σ_crit_eff distribution fix).