# Rigor pipeline (GPU-ready, hierarchical)

This directory contains a minimal, academically rigorous pipeline that implements the upgrades we discussed:
- **Hierarchical Bayesian** fit (NumPyro/JAX) with global `xi` parameters and per-galaxy nuisances.
- **Smooth shell** and **density-tied** `xi` families (replace hard caps with logistic transitions).
- **Data-driven outer boundary** (Σ-threshold, k×R_d, or flat-slope).
- **Posterior predictive overlays** and robust summary metrics.
- **Optional symbolic regression** (PySR) to discover compact `xi` formulas.

## Install

```bash
# Suggested environment
conda create -n rcfit python=3.11 -y
conda activate rcfit

# JAX + NumPyro (CUDA or ROCm as appropriate)
pip install numpyro[jax] arviz pandas pyarrow matplotlib

# Optional: PySR for symbolic regression (requires Julia)
pip install pysr
# The first run will bootstrap Julia packages.
```

## Run (SPARC only to start)

```bash
python -m rigor.fit_sparc   --parquet data/sparc_rotmod_ltg.parquet   --master data/Rotmod_LTG/MasterSheet_SPARC.csv   --xi shell_logistic_radius   --outer sigma --sigma_th 10   --outdir out/xi_shell_logistic_radius   --use_outer_only   --platform gpu   --warmup 1500 --samples 1500 --chains 4
```

Outputs:
- `out/.../posterior_samples.npz` — NumPy archive of posterior draws
- `out/.../figs/*.png` — Posterior predictive overlays with 68% bands

## Switch `xi` family

Inside `rigor/xi.py`, available `xi` choices:
- `shell_logistic_radius` (smooth shell-onset in ln R with mass-scaling)
- `logistic_density` (density-threshold driven)
- `logistic_radius_power` (cap via 1/(1+(R/R0)^m))

You can also **combine** radius & density effects (as in the model file), or replace them with more expressive forms for experiments.

## Symbolic regression (discovering a simple formula)

```python
from rigor.data import load_sparc
from rigor.formula_search import prepare_training_table, run_pysr_search

ds = load_sparc("data/sparc_rotmod_ltg.parquet", "data/Rotmod_LTG/MasterSheet_SPARC.csv",
                outer_method="sigma", sigma_th=10.0)
train = prepare_training_table(ds, use_outer_only=True)
model, table = run_pysr_search(train, out_csv="out/pysr_equations.csv", niterations=500)
print(table.head())
```

Then take the winning expression for `xi(R, Σ, Mbar)` and drop it into `rigor/xi.py` as a new entry.

## Notes

- The likelihood uses a **Student-t** to robustify against outliers and a per-galaxy intrinsic scatter.
- A **log-normal M/L offset** per galaxy is included (applies multiplicatively to `Vbar`).
- You can run **SPARC-only**, **MW-only**, and **joint** analyses by adding a similar loader for your binned Milky Way curve and concatenating datasets.
