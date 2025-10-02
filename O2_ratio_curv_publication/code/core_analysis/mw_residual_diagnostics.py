#!/usr/bin/env python
from __future__ import annotations
import os
import argparse
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        k = params[0] if params else 2.56
        return np.maximum(0.0, k * x)


def load_mw_rc_preferred():
    # Prefer curated MW radial RC derived from Gaia
    p1 = os.path.join("data", "milkyway", "gaia_predictions_by_radius.csv")
    p2 = os.path.join("out", "mw", "mw_predictions_by_radius.csv")
    if os.path.exists(p1):
        df = pd.read_csv(p1)
        # Drop rows without Vobs (0 with huge error)
        df = df[(df["Vobs_kms"] > 0)]
        R = df["R_kpc"].to_numpy()
        Vobs = df["Vobs_kms"].to_numpy()
        Vbar = df["Vbar_kms"].to_numpy()
        Verr = df["Verr_kms"].to_numpy() if "Verr_kms" in df.columns else None
        return R, Vobs, Vbar, Verr, p1
    elif os.path.exists(p2):
        df = pd.read_csv(p2)
        df = df[df["gal_id"] == "MilkyWay"][df["v_obs_kms"] > 0]
        R = df["r_kpc"].to_numpy()
        Vobs = df["v_obs_kms"].to_numpy()
        Vbar = df["vbar_kms"].to_numpy()
        Verr = None
        return R, Vobs, Vbar, Verr, p2
    else:
        raise FileNotFoundError("No MW RC source found: data/milkyway/gaia_predictions_by_radius.csv or out/mw/mw_predictions_by_radius.csv")


def load_sigma_and_interp(R_target: np.ndarray):
    ps = os.path.join("data", "mw_sigma_disk.csv")
    df = pd.read_csv(ps)
    R = df["R_kpc"].to_numpy()
    Sigma = df["Sigma_Msun_pc2"].to_numpy()
    # Monotonic increasing R expected
    # Interpolate to target radii
    Sig_interp = np.interp(R_target, R, Sigma)
    return R, Sigma, Sig_interp, ps


def estimate_Rd_from_sigma(R: np.ndarray, Sigma: np.ndarray, r_min=3.0, r_max=15.0):
    # Fit ln(Sigma) ~ ln(Sigma0) - R/Rd => slope = -1/Rd
    msk = (R >= r_min) & (R <= r_max) & np.isfinite(Sigma) & (Sigma > 0)
    if not np.any(msk):
        return 2.5  # fallback typical MW disk scale [kpc]
    x = R[msk]
    y = np.log(Sigma[msk])
    if x.size < 5:
        return 2.5
    p = np.polyfit(x, y, 1)
    slope = p[0]
    if slope < 0:
        Rd = -1.0 / slope
        # Constrain to a sensible band
        return float(np.clip(Rd, 1.5, 5.0))
    return 2.5


def run_mw(best_json: str, outdir: str):
    with open(best_json, "r", encoding="utf-8") as f:
        bj = json.load(f)
    fam = bj.get("best_family", bj.get("family", "ratio"))
    params = bj.get("params", [])

    R, Vobs, Vbar, Verr, rc_src = load_mw_rc_preferred()
    R_sig, Sigma_all, Sigma_R, sigma_src = load_sigma_and_interp(R)
    Rd = estimate_Rd_from_sigma(R_sig, Sigma_all)

    x = R / max(Rd, 1e-6)
    Sh = sigma_hat(Sigma_R)
    dlnS = grad_log_sigma(R, Sigma_R)
    gbar = (Vbar * Vbar) / np.maximum(R, 1e-9)

    fX = fX_from_family(fam, params, x, Sh, dlnS, gbar=gbar)
    Vmod = Vbar * np.sqrt(np.maximum(0.0, 1.0 + fX))

    resid = Vmod - Vobs
    ape = np.abs(resid) / np.maximum(np.abs(Vobs), 1e-9)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_mw = os.path.join(outdir, "mw")
    os.makedirs(out_mw, exist_ok=True)

    # Save CSV
    df = pd.DataFrame({
        "R_kpc": R,
        "Vobs": Vobs,
        "Vbar": Vbar,
        "Vmod": Vmod,
        "resid_kms": resid,
        "ape": ape,
        "x": x,
        "Sh": Sh,
        "abs_dlnS": np.abs(dlnS),
        "Rd_kpc": Rd,
        "rc_source": rc_src,
        "sigma_source": sigma_src,
        "family": fam,
    })
    csv_path = os.path.join(out_mw, f"mw_residuals_points_{fam}_{ts}.csv")
    df.to_csv(csv_path, index=False)

    # Overlay plot
    plt.figure(figsize=(7, 5))
    if Verr is not None:
        plt.errorbar(R, Vobs, yerr=Verr, fmt='k.', alpha=0.8, label='Observed ± err')
    else:
        plt.plot(R, Vobs, 'k.', label='Observed')
    plt.plot(R, Vbar, color='#1f77b4', alpha=0.8, label='Baryons')
    plt.plot(R, Vmod, color='#d62728', alpha=0.9, label=f'Model [{fam}]')
    plt.xlabel('R [kpc]'); plt.ylabel('V [km/s]'); plt.grid(True, alpha=0.3)
    plt.title(f"Milky Way overlay [{fam}] (Rd≈{Rd:.2f} kpc)")
    plt.legend()
    overlay_png = os.path.join(out_mw, f"mw_overlay_{fam}_{ts}.png")
    plt.tight_layout(); plt.savefig(overlay_png, dpi=150); plt.close()

    # Scatter helpers
    def scatter(xcol, ycol, fname, xlabel, ylabel):
        plt.figure(figsize=(6.2, 4.6))
        plt.scatter(xcol, ycol, s=30, alpha=0.7)
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(out_mw, fname), dpi=150); plt.close()

    scatter(R, resid, f"mw_resid_vs_R_{fam}_{ts}.png", "R [kpc]", "Residual (km/s)")
    scatter(x, resid, f"mw_resid_vs_x_{fam}_{ts}.png", "x = R/Rd", "Residual (km/s)")
    scatter(Sh, resid, f"mw_resid_vs_Sh_{fam}_{ts}.png", "Sigma-hat", "Residual (km/s)")
    scatter(np.abs(dlnS), resid, f"mw_resid_vs_absdlnS_{fam}_{ts}.png", "|d ln Sigma|", "Residual (km/s)")

    scatter(R, ape, f"mw_ape_vs_R_{fam}_{ts}.png", "R [kpc]", "Absolute Percentage Error")
    scatter(x, ape, f"mw_ape_vs_x_{fam}_{ts}.png", "x = R/Rd", "Absolute Percentage Error")
    scatter(Sh, ape, f"mw_ape_vs_Sh_{fam}_{ts}.png", "Sigma-hat", "Absolute Percentage Error")
    scatter(np.abs(dlnS), ape, f"mw_ape_vs_absdlnS_{fam}_{ts}.png", "|d ln Sigma|", "Absolute Percentage Error")

    # Heatmaps (with few points, still useful)
    def heat2d(xvals, yvals, zvals, xlab, ylab, fname):
        if len(xvals) < 8:
            return
        xi = np.linspace(np.quantile(xvals, 0.02), np.quantile(xvals, 0.98), 24)
        yi = np.linspace(np.quantile(yvals, 0.02), np.quantile(yvals, 0.98), 24)
        xb = np.digitize(xvals, xi) - 1
        yb = np.digitize(yvals, yi) - 1
        grid = np.full((len(xi)+1, len(yi)+1), np.nan)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                m = (xb == i) & (yb == j)
                if np.any(m):
                    grid[i, j] = float(np.median(zvals[m]))
        plt.figure(figsize=(6, 4.8))
        im = plt.imshow(grid.T, origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Median APE')
        plt.xlabel(xlab); plt.ylabel(ylab)
        plt.tight_layout(); plt.savefig(os.path.join(out_mw, fname), dpi=150); plt.close()

    heat2d(x, Sh, ape, "x = R/Rd", "Sigma-hat", f"mw_heat_mape_x_Sh_{fam}_{ts}.png")
    heat2d(x, np.abs(dlnS), ape, "x = R/Rd", "|d ln Sigma|", f"mw_heat_mape_x_absdlnS_{fam}_{ts}.png")

    print(f"[mw-diagnostics] Rd={Rd:.3f} kpc | wrote to {out_mw}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_json", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    run_mw(args.best_json, args.outdir)


if __name__ == "__main__":
    main()
