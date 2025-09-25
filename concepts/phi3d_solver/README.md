# 3D PDE Solver Concept (phi3d)

This folder groups the 3D nonlinear PDE solvers and comprehensive analyses for the G³ field, including CPU and GPU versions and cluster workflows.

Contents
- solve_phi3d.py: Reference CPU implementation (multigrid, sigma screen and mobility hooks), with detailed documentation.
- solve_phi3d_gpu.py: GPU‑accelerated solver (CuPy + custom CUDA kernels), optimized for RTX 5090.
- g3_pde_full_analysis.py, comprehensive_3d_analysis.py: End‑to‑end analyses across galaxies/clusters with parameter sweeps.
- g3_solver_wrapper.py: Wrapper/abstraction for PDE solver integration.
- run_3d_pde_on_clusters.py: Cluster pipeline (voxelization, solve, HSE temperature, comparisons).

Data sources
- SPARC voxelization inputs: data/Rotmod_LTG/* (via analyses)
- Cluster profiles: data/*_baryon_profile.csv or data/*_profiles.dat.txt

Typical commands
- Cluster PDE pipeline:
  python concepts/phi3d_solver/run_3d_pde_on_clusters.py

- GPU solver smoke test:
  python concepts/phi3d_solver/solve_phi3d_gpu.py

Expected outputs
- Figures under out/cluster_lensing/* and analysis CSV/JSON summaries next to scripts