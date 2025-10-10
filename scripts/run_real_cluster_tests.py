#!/usr/bin/env python3
from __future__ import annotations
import sys
import json
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Units/cosmology helpers from existing pipeline
from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    sigma_crit_Msun_per_kpc2,
    angular_diameter_distance_kpc,
    ne_to_rho_gas_Msun_kpc3,
)

# Observations helper (theta_E from gold standard)
from scripts.lensing_utils import get_thetaE_observed

DATA_DIR = ROOT / "data" / "clusters"
OUT_DIR = ROOT / "out" / "cluster_lensing_comparison" / "real_tests"

@dataclass
class Features:
    R_edge: float
    edge_sharp: float
    M_core: float

# ------------------ IO ------------------

def load_baryon_profiles(cluster: str) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    cpath = DATA_DIR / cluster
    gas_df = pd.read_csv(cpath / "gas_profile.csv")
    stars_df = pd.read_csv(cpath / "stars_profile.csv")

    r_g = gas_df["r_kpc"].to_numpy(float)
    # Convert n_e -> rho_gas
    if "n_e_cm3" in gas_df.columns:
        ne = gas_df["n_e_cm3"].to_numpy(float)
        rho_g = ne_to_rho_gas_Msun_kpc3(ne)
    else:
        rho_g = gas_df["rho_gas_Msun_per_kpc3"].to_numpy(float)

    r_s = stars_df["r_kpc"].to_numpy(float)
    rho_s = stars_df["rho_star_Msun_per_kpc3"].to_numpy(float)

    r_all = np.unique(np.concatenate([r_g[r_g > 0], r_s[r_s > 0]])).astype(float)
    rho_gi = np.interp(r_all, r_g, rho_g, left=0.0, right=0.0)
    rho_si = np.interp(r_all, r_s, rho_s, left=0.0, right=0.0)
    rho_b = rho_gi + rho_si
    return r_all, rho_b

# ------------------ Physics ------------------

def abel_project_sigma(r_3d: np.ndarray, rho_3d: np.ndarray, R_2d: np.ndarray) -> np.ndarray:
    Sigma = np.zeros_like(R_2d)
    for i, R in enumerate(R_2d):
        mask = r_3d >= R
        if not np.any(mask):
            continue
        rr = r_3d[mask]
        rh = rho_3d[mask]
        denom = np.sqrt(np.maximum(rr**2 - R**2, 1e-30))
        Sigma[i] = 2.0 * np.trapz(rh * rr / denom, rr)
    return Sigma  # Msun/kpc^2


def mean_sigma_inside_R(R: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    M_enc = cumulative_trapezoid(Sigma * 2.0 * math.pi * R, R, initial=0.0)
    return M_enc / (math.pi * R**2 + 1e-30)


def alpha_gr_from_baryons(cluster: str, z_l: float, z_s: float, theta_arcsec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r3d, rho3d = load_baryon_profiles(cluster)
    R = np.logspace(np.log10(max(0.1, float(r3d.min()))), np.log10(float(r3d.max())), 700)
    Sigma = abel_project_sigma(r3d, rho3d, R)  # Msun/kpc^2

    M_enc = cumulative_trapezoid(Sigma * 2.0 * math.pi * R, R, initial=0.0)  # Msun

    Sigma_crit = float(sigma_crit_Msun_per_kpc2(z_l, z_s))
    D_d_kpc = float(angular_diameter_distance_kpc(z_l))
    theta_R = (R / max(D_d_kpc, 1e-12)) * 206265.0

    kbar_R = M_enc / (math.pi * R**2 * max(Sigma_crit, 1e-30))
    alpha_R = kbar_R * theta_R

    alpha_gr = np.interp(theta_arcsec, theta_R, alpha_R, left=0.0, right=float(alpha_R[-1]))
    return R, Sigma, M_enc, alpha_R, alpha_gr

# ------------------ Features & Slip ------------------

def extract_features(R: np.ndarray, Sigma: np.ndarray, Sigma0_pc2: float = 100.0) -> Features:
    lnR = np.log(R + 1e-6)
    lnS_pc2 = np.log(np.maximum(Sigma / 1e6, 1e-12))
    lnS_s = gaussian_filter1d(lnS_pc2, sigma=2.0)

    # R_edge: Σ̄(<R) crosses Sigma0_pc2
    Sigma_bar_pc2 = mean_sigma_inside_R(R, Sigma) / 1e6
    idx_edge = int(np.argmin(np.abs(np.log10(np.maximum(Sigma_bar_pc2, 1e-20)) - math.log10(Sigma0_pc2))))
    R_edge = float(R[idx_edge])

    # edge_sharp ε = max |d ln Σ / d ln R| around the edge
    band = (R > 0.5 * R_edge) & (R < 1.5 * R_edge)
    grad = np.gradient(lnS_s, lnR)
    edge_sharp = float(np.max(np.abs(grad[band]))) if np.any(band) else float(np.max(np.abs(grad)))

    # M_core (<100 kpc)
    core_mask = R <= 100.0
    M_core = float(cumulative_trapezoid(Sigma[core_mask] * 2.0 * math.pi * R[core_mask], R[core_mask], initial=0.0)[-1]) if np.any(core_mask) else 0.0

    return Features(R_edge=R_edge, edge_sharp=edge_sharp, M_core=M_core)


def predict_slip_params(features: Features) -> Tuple[float, float]:
    # Universal scalings (population-level defaults)
    # S_inf = 1 + 10 * epsilon^0.6 * (M_core/1e13)^0.25 ; Rs = 0.9 * R_edge
    S_inf = 1.0 + 10.0 * (features.edge_sharp ** 0.6) * ((max(features.M_core, 1e-30) / 1e13) ** 0.25)
    Rs = 0.9 * max(features.R_edge, 1e-3)
    return float(S_inf), float(Rs)


def logistic(x: np.ndarray, x0: float, w: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - x0) / max(w, 1e-6)))


def slip_profile(R: np.ndarray, Sigma: np.ndarray, S_inf: float, Rs: float,
                 p: float = 1.0, eta: float = 1.0, x0: float = 0.3, w: float = 0.3,
                 Sigma0_pc2: float = 100.0, S_cap: float = 200.0) -> np.ndarray:
    Sigma_bar_pc2 = mean_sigma_inside_R(R, Sigma) / 1e6
    Shat = np.log10(np.maximum(Sigma_bar_pc2 / max(Sigma0_pc2, 1e-9), 1e-12))
    gate = 1.0 - logistic(Shat, x0=x0, w=w)
    ramp = 1.0 - np.exp(- (R / max(Rs, 1e-6)) ** p)
    S = 1.0 + S_inf * (ramp ** eta) * gate
    return np.clip(S, 1.0, S_cap)


def predict_alpha_model(theta_arcsec: np.ndarray, R: np.ndarray, alpha_R: np.ndarray, S_R: np.ndarray, D_d_kpc: float) -> np.ndarray:
    theta_R = (R / max(D_d_kpc, 1e-12)) * 206265.0
    alpha_R_model = alpha_R * S_R
    # Interpolate S-weighted α to requested θ
    return np.interp(theta_arcsec, theta_R, alpha_R_model, left=float(alpha_R_model[0]), right=float(alpha_R_model[-1]))

# ------------------ Evaluation ------------------

def find_einstein_radius(theta: np.ndarray, alpha_theta: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(alpha_theta - theta)))
    return float(theta[idx])


def run_cluster(cluster: str, z_l: float, z_s: float) -> Dict:
    theta = np.linspace(5.0, 120.0, 220)  # arcsec

    # GR baseline
    R, Sigma, M_enc, alpha_R, alpha_gr = alpha_gr_from_baryons(cluster, z_l, z_s, theta)

    # Features -> slip params -> S(R)
    feats = extract_features(R, Sigma)
    S_inf, Rs = predict_slip_params(feats)
    Dd = float(angular_diameter_distance_kpc(z_l))
    S_R = slip_profile(R, Sigma, S_inf, Rs, p=1.2, eta=1.0, x0=0.3, w=0.3)

    alpha_model = predict_alpha_model(theta, R, alpha_R, S_R, Dd)

    # Observed Einstein radius if available
    thetaE_obs = get_thetaE_observed(cluster)

    # Metrics
    thetaE_model = find_einstein_radius(theta, alpha_model)
    metrics = {
        "cluster": cluster,
        "z_l": z_l,
        "z_s": z_s,
        "R_edge_kpc": feats.R_edge,
        "edge_sharp": feats.edge_sharp,
        "M_core_Msun": feats.M_core,
        "S_inf": S_inf,
        "Rs_kpc": Rs,
        "thetaE_model_arcsec": thetaE_model,
        "thetaE_obs_arcsec": thetaE_obs,
        "thetaE_abs_err_arcsec": None if thetaE_obs is None else abs(thetaE_model - thetaE_obs),
    }

    # Persist per-cluster outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{cluster}_real_test.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

# ------------------ CLI ------------------

def main():
    parser = argparse.ArgumentParser(description="Run real-units slip tests against observed lensing")
    parser.add_argument("--clusters", nargs="*", default=["MACSJ0416", "MACSJ0717", "MACSJ1149"], help="Cluster IDs (directory names under data/clusters)")
    parser.add_argument("--zl", nargs="*", type=float, default=[0.396, 0.546, 0.544], help="Lens redshifts for clusters")
    parser.add_argument("--zs", nargs="*", type=float, default=[2.0, 2.0, 2.0], help="Source redshifts")
    args = parser.parse_args()

    rows = []
    for i, cluster in enumerate(args.clusters):
        z_l = args.zl[i] if i < len(args.zl) else args.zl[-1]
        z_s = args.zs[i] if i < len(args.zs) else args.zs[-1]
        m = run_cluster(cluster, z_l, z_s)
        rows.append(m)
        print(m)

    # Write summary table
    out_csv = OUT_DIR / "summary_real_tests.json"
    with out_csv.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()