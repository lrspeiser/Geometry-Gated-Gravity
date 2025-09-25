# GPU Solver Suite Concept

This folder groups the GPU-based solvers and orchestration tools used for Milky Way and SPARC optimization runs.

Contents
- g3_gpu_solver_suite.py: Core GPU/CPU-agnostic implementation of the unified tail (CuPy fallback to NumPy) plus optimizers (PSO/DE) and evaluation utilities.
- mw_gpu_orchestrator.py, mw_gpu_search_runner.py, mw_g3_gpu_opt.py: MW-focused orchestrators and runners for parameter searches.
- g3_mw_gpu_eval.py: GPU-accelerated MW evaluator with asymmetric drift correction.
- Tests: test_cuda_kernels.py, test_gpu_solver_simple.py, test_kernel_execution.py

Data sources
- MW: data/gaia_mw_real.csv, data/mw_gaia_144k.npz
- SPARC: used indirectly via sparc_zero_shot workflows

Typical commands
- MW orchestrated search:
  python concepts/gpu_solver_suite/mw_gpu_orchestrator.py

- GPU suite tests:
  python concepts/gpu_solver_suite/test_gpu_solver_simple.py