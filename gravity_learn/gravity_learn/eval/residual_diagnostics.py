#!/usr/bin/env python
from __future__ import annotations
import os
import argparse
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rigor.rigor.data import load_sparc
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma


def fX_from_family(family: str, params, x, Sh, dlnS, gbar=None):
    if family == "ratio":
        a, b = params
        denom = (a - b * Sh)
        denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
        return np.maximum(0.0, (x * x) / denom)
    elif family == "exp":
        alpha, c = params
        alpha = max(alpha, 0.0)
        return np.maximum(0.0, alpha * (x * x) * (np.exp(Sh) + c))
    elif family == "ratio_curv":
        a, b, d = params
        denom = (a - b * Sh - d * np.abs(dlnS))
        denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
        return np.maximum(0.0, (x * x) / denom)
    elif family == "exp_curv":
        alpha, c, d = params
        alpha = max(alpha, 0.0)
        return np.maximum(0.0, alpha * (x * x) * (np.exp(Sh) + c + d * np.abs(dlnS)))
    elif family == "ratio_curv_gbar":
        a, b, d, e = params
        gb = np.sqrt(np.maximum(gbar, 0.0)) if gbar is not None else 0.0
        denom = (a - b * Sh - d * np.abs(dlnS) + e * gb)
        denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
        return np.maximum(0.0, (x * x) / denom)
    else:
        # fallback
        k = params[0] if params else 2.56
        return np.maximum(0.0, k * x)


def diagnostics(best_json: str, outdir: str, montage_limit: int = 16):
    with open(best_json, "r", encoding="utf-8") as f:
        bj = json.load(f)
    fam = bj.get("best_family", bj.get("family", "ratio"))
    params = bj.get("params", [])

    ds = load_sparc()

    rows = []
    pergal = []
    for g in ds.galaxies:
        R = g.R_kpc; Vobs = g.Vobs_kms; Vbar = g.Vbar_kms
        if R is None or Vobs is None or Vbar is None:
            continue
        mask = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
        R = np.asarray(R)[mask]
        Vobs = np.asarray(Vobs)[mask]
        Vbar = np.asarray(Vbar)[mask]
        if R.size < 6:
            continue
        Sigma = g.Sigma_bar[mask] if (g.Sigma_bar is not None) else np.maximum(1e-3, np.exp(-R / np.maximum(1.0, np.nanmedian(R))))
        x = np.asarray(dimensionless_radius(R, Rd=(g.Rd_kpc or None)))
        Sh = np.asarray(sigma_hat(Sigma))
        dlnS = np.asarray(grad_log_sigma(R, Sigma))
        gbar = (Vbar * Vbar) / np.maximum(R, 1e-9)
        fX = fX_from_family(fam, params, x, Sh, dlnS, gbar=gbar)
        Vmod = Vbar * np.sqrt(np.maximum(0.0, 1.0 + fX))
        resid = Vmod - Vobs
        ape = np.abs(resid) / np.maximum(np.abs(Vobs), 1e-9)
        for i in range(len(R)):
            rows.append({
                "Galaxy": g.name,
                "R_kpc": float(R[i]),
                "Vobs": float(Vobs[i]),
                "Vbar": float(Vbar[i]),
                "Vmod": float(Vmod[i]),
                "resid_kms": float(resid[i]),
                "ape": float(ape[i]),
                "x": float(x[i]),
                "Sh": float(Sh[i]),
                "abs_dlnS": float(abs(dlnS[i]))
            })
        pergal.append({
            "Galaxy": g.name,
            "rmse": float(np.sqrt(np.mean((Vmod - Vobs) ** 2))),
            "median_ape": float(np.median(ape)),
            "n_points": int(len(R))
        })

    df = pd.DataFrame(rows)
    dfp = pd.DataFrame(pergal)

    diag_dir = os.path.join(outdir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df.to_csv(os.path.join(diag_dir, f"residuals_points_{fam}_{ts}.csv"), index=False)
    dfp.to_csv(os.path.join(diag_dir, f"residuals_per_galaxy_{fam}_{ts}.csv"), index=False)

    def scatter(xcol, ycol, fname, ylabel):
        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(df[xcol], df[ycol], s=10, alpha=0.4)
        plt.xlabel(xcol)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, fname), dpi=150)
        plt.close()

    # Residual vs features
    scatter("R_kpc", "resid_kms", f"resid_vs_R_{fam}_{ts}.png", "Residual (km/s)")
    scatter("x", "resid_kms", f"resid_vs_x_{fam}_{ts}.png", "Residual (km/s)")
    scatter("Sh", "resid_kms", f"resid_vs_Sh_{fam}_{ts}.png", "Residual (km/s)")
    scatter("abs_dlnS", "resid_kms", f"resid_vs_absdlnS_{fam}_{ts}.png", "Residual (km/s)")

    # APE vs features
    scatter("R_kpc", "ape", f"ape_vs_R_{fam}_{ts}.png", "Absolute Percentage Error")
    scatter("x", "ape", f"ape_vs_x_{fam}_{ts}.png", "Absolute Percentage Error")
    scatter("Sh", "ape", f"ape_vs_Sh_{fam}_{ts}.png", "Absolute Percentage Error")
    scatter("abs_dlnS", "ape", f"ape_vs_absdlnS_{fam}_{ts}.png", "Absolute Percentage Error")

    # 2D heatmaps (median APE)
    def heat2d(xcol, ycol, fname):
        xb = np.linspace(df[xcol].quantile(0.01), df[xcol].quantile(0.99), 24)
        yb = np.linspace(df[ycol].quantile(0.01), df[ycol].quantile(0.99), 24)
        xi = np.digitize(df[xcol], xb) - 1
        yi = np.digitize(df[ycol], yb) - 1
        grid = np.full((len(xb)+1, len(yb)+1), np.nan)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                m = (xi == i) & (yi == j)
                if np.any(m):
                    grid[i, j] = float(np.median(df.loc[m, "ape"]))
        plt.figure(figsize=(6.5, 5))
        im = plt.imshow(grid.T, origin="lower", aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Median APE")
        plt.xlabel(xcol)
        plt.ylabel(ycol)
        plt.title(f"Median APE: {xcol} vs {ycol}")
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, fname), dpi=150)
        plt.close()

    heat2d("x", "Sh", f"heat_mape_x_Sh_{fam}_{ts}.png")
    heat2d("x", "abs_dlnS", f"heat_mape_x_absdlnS_{fam}_{ts}.png")

    # Identify tail galaxies by median APE
    dfp_sorted = dfp.sort_values("median_ape", ascending=False)
    dfp_sorted.to_csv(os.path.join(diag_dir, f"tail_galaxies_by_median_ape_{fam}_{ts}.csv"), index=False)

    print(f"[diagnostics] wrote outputs to {diag_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_json", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    diagnostics(args.best_json, args.outdir)


if __name__ == "__main__":
    main()
