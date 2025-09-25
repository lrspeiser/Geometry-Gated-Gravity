# Cluster Lensing Concept

Lensing analysis and diagnostics using the universal G³ model.

Contents
- cluster_lensing_analysis.py: Compute Σ(R), κ(R), γ_t(R), Einstein radius; compare baryonic vs G³-effective lensing. Saves plots and JSON.
- cluster_lensing_diagnostic.py: Compare G³ predictions with observed Einstein radii; NFW comparisons; diagnostics and required scaling.

Data sources
- Uses universal G³ parameters from concepts/universal_g3 (best.json, etc.) when available
- Observational lensing references are embedded in the scripts; see comments

Typical commands
- Main analysis:
  python concepts/cluster_lensing/cluster_lensing_analysis.py

- Diagnostic comparisons:
  python concepts/cluster_lensing/cluster_lensing_diagnostic.py

Expected outputs
- out/cluster_lensing/<cluster_name>/*.png and CSVs
- out/cluster_lensing/summaries.json