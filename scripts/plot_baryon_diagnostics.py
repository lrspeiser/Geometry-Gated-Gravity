#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    angular_diameter_distance_kpc,
    sigma_crit_Msun_per_kpc2,
    ne_to_rho_gas_Msun_kpc3,
    abel_project_sigma as abel_project_sigma_ref,
)

DATA_DIR = ROOT / "data" / "clusters"
OUT_DIR = ROOT / "out" / "cluster_lensing_comparison" / "diagnostics"


def load_baryon_profiles(cluster: str):
    import pandas as pd
    cpath = DATA_DIR / cluster
    gas_df = pd.read_csv(cpath / "gas_profile.csv")
    stars_df = pd.read_csv(cpath / "stars_profile.csv")
    r_g = gas_df["r_kpc"].to_numpy(float)
    if "n_e_cm3" in gas_df.columns:
        ne = gas_df["n_e_cm3"].to_numpy(float)
        rho_g = ne_to_rho_gas_Msun_kpc3(ne)
    else:
        rho_g = gas_df["rho_gas_Msun_per_kpc3"].to_numpy(float)
    r_s = stars_df["r_kpc"].to_numpy(float)
    rho_s = stars_df["rho_star_Msun_per_kpc3"].to_numpy(float)
    r_all = np.unique(np.concatenate([r_g[r_g>0], r_s[r_s>0]])).astype(float)
    rho = np.interp(r_all, r_g, rho_g, left=0.0, right=0.0) + np.interp(r_all, r_s, rho_s, left=0.0, right=0.0)
    return r_all, rho


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot baryon diagnostics for a cluster")
    parser.add_argument("cluster", help="Cluster directory name under data/clusters (e.g., MACSJ0416)")
    parser.add_argument("--zl", type=float, required=True, help="Lens redshift")
    parser.add_argument("--zs", type=float, default=2.0, help="Source redshift")
    args = parser.parse_args()

    r3d, rho3d = load_baryon_profiles(args.cluster)
    R = np.logspace(np.log10(max(0.1, float(r3d.min()))), np.log10(float(r3d.max())), 800)
    Sigma = abel_project_sigma_ref(r3d, rho3d, R)
    M_enc = cumulative_trapezoid(Sigma * 2*np.pi*R, R, initial=0.0)

    Dd = float(angular_diameter_distance_kpc(args.zl))
    Ds = float(angular_diameter_distance_kpc(args.zs))
    Sigma_crit = float(sigma_crit_Msun_per_kpc2(args.zl, args.zs))
    theta_R = (R / max(Dd, 1e-12)) * 206265.0

    Sbar = M_enc / (np.pi * np.maximum(R, 1e-9)**2)
    kbar = Sbar / max(Sigma_crit, 1e-30)
    alpha_R = kbar * theta_R

    # Plots
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    ax = axs[0,0]
    ax.loglog(R, Sigma, lw=2)
    ax.set_xlabel("R [kpc]"); ax.set_ylabel("Sigma(R) [Msun/kpc^2]")
    ax.set_title(f"{args.cluster}: Surface density")
    ax.grid(True, which='both', alpha=0.3)

    ax = axs[0,1]
    ax.loglog(R, Sbar/1e6, lw=2)  # convert to Msun/pc^2
    ax.axhline(100.0, color='k', ls=':')
    ax.set_xlabel("R [kpc]"); ax.set_ylabel("mean Sigma(<R) [Msun/pc^2]")
    ax.set_title("Mean surface density")
    ax.grid(True, which='both', alpha=0.3)

    ax = axs[1,0]
    ax.loglog(R, M_enc, lw=2)
    ax.set_xlabel("R [kpc]"); ax.set_ylabel("M(<R) [Msun]")
    ax.set_title("Enclosed mass")
    ax.grid(True, which='both', alpha=0.3)

    ax = axs[1,1]
    ax.plot(theta_R, alpha_R, lw=2)
    ax.plot(theta_R, theta_R, ls='--', color='k', alpha=0.5)
    ax.set_xlabel("theta [arcsec]"); ax.set_ylabel("alpha_GR(theta) [arcsec]")
    ax.set_title("GR deflection vs θ")
    ax.grid(True, alpha=0.3)

    out_png = OUT_DIR / f"{args.cluster}_diagnostics.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()