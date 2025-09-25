# G³-Σ Unified Concept

This folder groups the final and production implementations of the G³-Σ model, along with their optimization utilities and published outputs.

Contents
- g3_sigma_final.py: Final G³-Σ model with automatic solar-system recovery (Sigma screen + saturating mobility). Includes verification for planetary orbits.
- g3_sigma_production.py: GPU-capable production 3D PDE solver with geometry-aware scaling and sigma screening.
- optimize_g3_sigma_unified.py: Optimizer to fit a single global parameter tuple across regimes (galaxies, MW, clusters) using simplified fast formulas.
- g3_sigma_complete_results.json, g3_sigma_final_results.json, g3_sigma_optimal_params.json: Saved results/parameters from optimization and production runs.
- g3_sigma_sparc_results.csv: CSV summary for SPARC comparisons.
- g3_sigma_publication_figures.png: Figure(s) used for reports/publication.

Data sources
- SPARC: data/sparc_rotmod_ltg.parquet, data/Rotmod_LTG/MasterSheet_SPARC.csv
- Milky Way: data/gaia_mw_real.csv or data/mw_gaia_144k.npz (from concepts/milky_way_tools)
- Clusters: see concepts/phi3d_solver for voxelized profiles and PDE evaluation

Typical commands
- Final model verification (solar system):
  python concepts/g3_sigma_unified/g3_sigma_final.py

- Production PDE (GPU-enabled where available):
  python concepts/g3_sigma_unified/g3_sigma_production.py

- Unified parameter optimization across regimes:
  python concepts/g3_sigma_unified/optimize_g3_sigma_unified.py

Expected outputs
- JSON parameter snapshots and summary metrics next to scripts
- CSV summaries for SPARC comparisons
- Publication figures under this folder or under reports/figures
