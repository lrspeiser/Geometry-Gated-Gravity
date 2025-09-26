# gravity_learn (experimental)

Purpose
- A self-contained sandbox to explore data-driven equation discovery for the gravity learning concept (O2 galaxies, O3 clusters/lensing) without modifying any existing code or paper.md in this repository.
- Reuses data already in ../data and (where sensible) imports existing utilities from rigor/ to avoid duplication.

What this adds (and nothing else)
- New code and configs live strictly under gravity_learn/.
- No edits to existing files; no paper.md changes.
- Experiments artifacts are kept under gravity_learn/experiments and tracked via Git LFS rules local to this folder.

Data
- Galaxies (SPARC): ../data/sparc_rotmod_ltg.parquet (and optional MasterSheet CSV if present).
- Milky Way, clusters and lensing: use your existing ../data/* files. We will wire O3 later.

GPU notes (RTX 5090)
- Optional CuPy acceleration: install cupy-cuda12x for CUDA 12.
- JAX (for PINN): install jax[cuda12] from the official wheels matching your CUDA, or run on CPU.
- All compute paths have CPU fallbacks for smoke tests; GPU strongly recommended for heavier runs.

Symbolic Regression (PySR)
- PySR requires Julia. Quick start:
  - pip install pysr
  - The first run may bootstrap a local Julia; see PySR docs if you want system Julia.
- This folder includes a wrapper that can call the existing rigor.rigor.formula_search utilities, and/or run minimal local SR.
- In code we point back to this README for setup details.

Windows usage tips (PowerShell)
- Prefer python -m style invocations to avoid PATH quirks.
- Use fd/rg (Rust tools) for search if installed, per your preference.
- Avoid smart quotes  and trailing backslashes in the shell (see WARP.md tips).

Quickstart
1) (Optional) create an environment
   - conda create -n gravlearn python=3.11 -y && conda activate gravlearn
2) Install extras for this sandbox only:
   - pip install -r gravity_learn/requirements.txt
   - For GPU: pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   - For CuPy: pip install cupy-cuda12x
3) Run a tiny O2 training smoke test (PINN or SR):
   - python -m gravity_learn.train.train_o2 --cfg gravity_learn/configs/o2_default.yaml
   - To force SR path: python -m gravity_learn.train.train_o2 --cfg gravity_learn/configs/o2_default.yaml --mode sr
4) Run unit tests:
   - pytest -q gravity_learn/tests

Scope of the initial scaffold
- features/geometry.py: dimensionless features (x=R/Rd or R_half; fSigma, grad ln 7fSigma)
- physics/poisson_abel.py: spherical mass integrals and an Abel projector (NumPy first, optional CuPy)
- models/pinn_o2.py: minimal JAX gating model for g_tail (few params, quick to iterate)
- models/symbolic_srx.py: SR wrapper (can reuse rigor.rigor.formula_search or run a local PySR pass)
- train/train_o2.py: CLI entrypoint to run O2 (choose PINN or SR) and save artifacts into experiments/
- eval/zero_shot.py: placeholder to structure across-type evaluation
- tests/: light unit tests for the Abel projector and feature shapers

LFS and artifacts
- Only gravity_learn/experiments/**/* is LFS-tracked via gravity_learn/.gitattributes.
- Keep large runs, checkpoints, and tables under gravity_learn/experiments/.

Notes on duplication
- To avoid multiple active versions of similar code, we prefer importing from rigor/ where suitable (e.g., SPARC loader, SR helpers).
- If we must copy a small snippet, it will live under gravity_learn/src/gravity_learn/copied_examples with a header "copied from <path> on <date>, not used elsewhere".

Contacts & provenance
- No API keys are required for this sandbox.
- If we later add any web services, we will add code comments pointing back to this README for setup (per your rule).
