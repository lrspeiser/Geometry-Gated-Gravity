
"""Symbolic regression over xi(R, Sigma, Mbar, ...) using PySR.

This produces a candidate closed-form xi, to be later validated in the full
hierarchical Bayesian pipeline. Requires:
    pip install pysr
and a working Julia installation (PySR will prompt to install dependencies).

We construct training targets as xi_empirical = (Vobs / (Vbar * sqrt(exp(0.5*dlogML_g))))^2,
optionally smoothed and restricted to outer regions where the signal is cleanest.
"""
from __future__ import annotations
import os, numpy as np, pandas as pd

def prepare_training_table(dataset, use_outer_only=True, min_vbar=20.0):
    rows = []
    for g in dataset.galaxies:
        mask = g.outer_mask if use_outer_only else np.isfinite(g.R_kpc)
        mask &= np.isfinite(g.Vbar_kms) & (g.Vbar_kms>=min_vbar)
        if not np.any(mask): continue
        xi_emp = (g.Vobs_kms[mask] / np.maximum(g.Vbar_kms[mask], 1e-6))**2
        rows.append(pd.DataFrame({
            "Galaxy": g.name,
            "R_kpc": g.R_kpc[mask],
            "Sigma_bar": g.Sigma_bar[mask] if g.Sigma_bar is not None else np.nan,
            "Mbar_Msun": (g.Mbar_Msun or np.nan),
            "xi_emp": xi_emp
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Galaxy","R_kpc","Sigma_bar","Mbar_Msun","xi_emp"])

def run_pysr_search(train_df, out_csv="pysr_results.csv", niterations=200, unary_operators=None, binary_operators=None):
    from pysr import PySRRegressor
    X_cols = [c for c in ["R_kpc","Sigma_bar","Mbar_Msun"] if c in train_df.columns]
    X = train_df[X_cols].to_numpy()
    y = train_df["xi_emp"].to_numpy()
    model = PySRRegressor(
        niterations=niterations,
        unary_operators=unary_operators or ["log", "sqrt", "exp"],
        binary_operators=binary_operators or ["+", "-", "*", "/"],
        loss="loss(x, y) = (x - y)^2",
        model_selection="best",
        procs=0,  # let PySR pick
        populations=30,
        progress=True
    )
    model.fit(X, y)
    # Save equations and scores
    results = model.equations_
    results.to_csv(out_csv, index=False)
    return model, results
