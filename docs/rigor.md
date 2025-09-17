# rigor integration (hierarchical Bayesian pipeline)

This repository vendors a high-rigor, GPU-ready Bayesian pipeline under `rigor/`.
It provides hierarchical fits with smooth xi(·) families, objective outer-region
selection, posterior-predictive overlays, baselines (GR/MOND/Burkert), and optional
symbolic regression to discover simple closed-form laws.

Install (vendored; Option B)
- One-time install into your active environment:

```bash
make install-rigor
```

Quick smoke test

```bash
# import check
python -c "import rigor, rigor.xi, rigor.data; print('rigor ok')"

# quick baseline comparison (writes out/baselines_summary.json)
make compare-baselines

# short fit (reduce samples/chains for speed)
python -m rigor.fit_sparc \
  --parquet data/sparc_rotmod_ltg.parquet \
  --master  data/Rotmod_LTG/MasterSheet_SPARC.csv \
  --outer sigma --sigma_th 10 --use_outer_only \
  --outdir out/smoke --platform cpu \
  --warmup 200 --samples 200 --chains 2

# overlays with 68% bands
python -m rigor.plotting \
  --post out/smoke/posterior_samples.npz \
  --figdir out/smoke/figs \
  --parquet data/sparc_rotmod_ltg.parquet \
  --master  data/Rotmod_LTG/MasterSheet_SPARC.csv \
  --outer sigma --sigma_th 10
```

Makefile targets
- Install: `make install-rigor`
- SPARC fit (outer-only): `make fit-bayes`
- Joint SPARC + MW: `make fit-bayes-joint`
- Baselines: `make compare-baselines`
- Posterior overlays: `make bayes-overlays`

GPU notes (MacOS)
- CPU works by default (`--platform cpu`).
- Apple Silicon GPU: `pip install jax-metal numpyro[jax]` then use `--platform mps`.
- NVIDIA GPU: install JAX-CUDA per JAX docs and use `--platform gpu`.

Large artifacts (Git LFS)
- Posterior files `*.npz` and `*.nc` are tracked via Git LFS.
