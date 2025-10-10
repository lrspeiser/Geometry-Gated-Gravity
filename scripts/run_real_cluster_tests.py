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
    abel_project_sigma as abel_project_sigma_ref,
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

def load_baryon_profiles(cluster: str, debug: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    cpath = DATA_DIR / cluster
    gas_df = pd.read_csv(cpath / "gas_profile.csv")
    stars_df = pd.read_csv(cpath / "stars_profile.csv")

    r_g = gas_df["r_kpc"].to_numpy(float)
    # Convert n_e -> rho_gas
    if "n_e_cm3" in gas_df.columns:
        ne = gas_df["n_e_cm3"].to_numpy(float)
        # Clean NaN values
        ne = np.nan_to_num(ne, nan=0.0, posinf=0.0, neginf=0.0)
        if debug:
            print(f"  n_e range: [{ne.min():.6f}, {ne.max():.6f}] cm^-3, nonzero={np.count_nonzero(ne)}")
        rho_g = ne_to_rho_gas_Msun_kpc3(ne)
        rho_g = np.nan_to_num(rho_g, nan=0.0, posinf=0.0, neginf=0.0)
        if debug:
            print(f"  rho_g range: [{rho_g.min():.3e}, {rho_g.max():.3e}] Msun/kpc^3, nonzero={np.count_nonzero(rho_g)}")
    else:
        rho_g = gas_df["rho_gas_Msun_per_kpc3"].to_numpy(float)
        rho_g = np.nan_to_num(rho_g, nan=0.0, posinf=0.0, neginf=0.0)

    r_s = stars_df["r_kpc"].to_numpy(float)
    rho_s = stars_df["rho_star_Msun_per_kpc3"].to_numpy(float)
    rho_s = np.nan_to_num(rho_s, nan=0.0, posinf=0.0, neginf=0.0)
    if debug:
        print(f"  r_g: [{r_g.min():.2f}, {r_g.max():.2f}] kpc, n={len(r_g)}")
        print(f"  r_s: [{r_s.min():.2f}, {r_s.max():.2f}] kpc, n={len(r_s)}")
        print(f"  rho_s range: [{rho_s.min():.3e}, {rho_s.max():.3e}] Msun/kpc^3, nonzero={np.count_nonzero(rho_s)}")

    # Simpler approach: since gas and stars have same r-grid, just use it directly
    # Filter out zero or negative r values
    valid_mask = (r_g > 0) & (r_s > 0) & np.isfinite(r_g) & np.isfinite(r_s)
    r_all = r_g[valid_mask]
    rho_gi = rho_g[valid_mask]
    rho_si = rho_s[valid_mask]
    
    if debug:
        print(f"  r_all: [{r_all.min():.2f}, {r_all.max():.2f}] kpc, n={len(r_all)}")
        print(f"  rho_gi direct: range=[{rho_gi.min():.3e}, {rho_gi.max():.3e}], nonzero={np.count_nonzero(rho_gi)}")
        print(f"  rho_si direct: range=[{rho_si.min():.3e}, {rho_si.max():.3e}], nonzero={np.count_nonzero(rho_si)}")
    if debug:
        print(f"  rho_gi interp: range=[{rho_gi.min():.3e}, {rho_gi.max():.3e}], nonzero={np.count_nonzero(rho_gi)}")
        print(f"  rho_si interp: range=[{rho_si.min():.3e}, {rho_si.max():.3e}], nonzero={np.count_nonzero(rho_si)}")
    rho_b = rho_gi + rho_si
    if debug:
        print(f"  rho_b final: range=[{rho_b.min():.3e}, {rho_b.max():.3e}], nonzero={np.count_nonzero(rho_b)}")
    
    # Ensure r is sorted ascending (required for Abel integral)
    sort_idx = np.argsort(r_all)
    r_all = r_all[sort_idx]
    rho_b = rho_b[sort_idx]
    
    if debug:
        is_sorted = np.all(np.diff(r_all) > 0)
        print(f"  r_all sorted: {is_sorted}, dr range: [{np.diff(r_all).min():.3e}, {np.diff(r_all).max():.3e}]")
    
    return r_all, rho_b

# ------------------ Physics ------------------



def mean_sigma_inside_R(R: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    M_enc = cumulative_trapezoid(Sigma * 2.0 * math.pi * R, R, initial=0.0)
    return M_enc / (math.pi * R**2 + 1e-30)


def alpha_gr_from_baryons(cluster: str, z_l: float, z_s: float, theta_arcsec: np.ndarray, debug: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r3d, rho3d = load_baryon_profiles(cluster)
    if debug:
        print(f"\n=== DEBUG: {cluster} ===")
        print(f"r3d range: [{r3d.min():.2f}, {r3d.max():.2f}] kpc, n_points={len(r3d)}")
        print(f"rho3d range: [{rho3d.min():.3e}, {rho3d.max():.3e}] Msun/kpc^3")
        print(f"rho3d nonzero: {np.count_nonzero(rho3d)} / {len(rho3d)}")
    
    R = np.logspace(np.log10(max(0.1, float(r3d.min()))), np.log10(float(r3d.max())), 700)
    if debug:
        print(f"R range: [{R.min():.2f}, {R.max():.2f}] kpc, n_points={len(R)}")
    
    Sigma = abel_project_sigma_ref(r3d, rho3d, R)  # Msun/kpc^2
    if debug:
        print(f"Sigma range: [{Sigma.min():.3e}, {Sigma.max():.3e}] Msun/kpc^2")
        print(f"Sigma nonzero: {np.count_nonzero(Sigma)} / {len(Sigma)}")

    M_enc = cumulative_trapezoid(Sigma * 2.0 * math.pi * R, R, initial=0.0)  # Msun
    if debug:
        print(f"M_enc range: [{M_enc.min():.3e}, {M_enc.max():.3e}] Msun")

    Sigma_crit = float(sigma_crit_Msun_per_kpc2(z_l, z_s))
    D_d_kpc = float(angular_diameter_distance_kpc(z_l))
    theta_R = (R / max(D_d_kpc, 1e-12)) * 206265.0

    kbar_R = M_enc / (math.pi * R**2 * max(Sigma_crit, 1e-30))
    alpha_R = kbar_R * theta_R
    
    if debug:
        print(f"Sigma_crit: {Sigma_crit:.3e} Msun/kpc^2")
        print(f"D_d_kpc: {D_d_kpc:.2f} kpc")
        print(f"theta_R range: [{theta_R.min():.2f}, {theta_R.max():.2f}] arcsec")
        print(f"kbar_R range: [{kbar_R.min():.3e}, {kbar_R.max():.3e}]")
        print(f"alpha_R range: [{alpha_R.min():.3e}, {alpha_R.max():.3e}] arcsec")
        # Find where alpha ~ theta
        cross_idx = np.where(np.diff(np.sign(alpha_R - theta_R)))[0]
        if len(cross_idx) > 0:
            print(f"alpha crosses theta near R={R[cross_idx[0]]:.1f} kpc, theta={theta_R[cross_idx[0]]:.1f} arcsec")

    alpha_gr = np.interp(theta_arcsec, theta_R, alpha_R, left=0.0, right=float(alpha_R[-1]))
    return R, Sigma, M_enc, alpha_R, alpha_gr

# ------------------ Features & Slip ------------------

def extract_features(R: np.ndarray, Sigma: np.ndarray, Sigma0_pc2: float = 100.0) -> Features:
    lnR = np.log(np.maximum(R, 1e-6))
    lnS_pc2 = np.log(np.maximum(Sigma / 1e6, 1e-30))
    lnS_s = gaussian_filter1d(lnS_pc2, sigma=2.0)

    # R_edge: find crossing of mean Σ with Sigma0_pc2 within a physical window
    Sigma_bar_pc2 = mean_sigma_inside_R(R, Sigma) / 1e6
    window = (R >= 30.0) & (R <= 1000.0)
    if np.any(window):
        idxw = np.where(window)[0]
        irel = int(np.argmin(np.abs(Sigma_bar_pc2[window] - Sigma0_pc2)))
        idx_edge = idxw[irel]
    else:
        idx_edge = int(np.argmin(np.abs(Sigma_bar_pc2 - Sigma0_pc2)))
    R_edge = float(R[idx_edge])

    # edge_sharp ε = max |d ln Σ / d ln R| near the edge (±~0.3 dex)
    grad = np.gradient(lnS_s, lnR)
    j0 = max(0, idx_edge - 5)
    j1 = min(len(R), idx_edge + 6)
    edge_sharp = float(np.max(np.abs(grad[j0:j1])))

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

from scipy.optimize import brentq

def find_einstein_radius(theta: np.ndarray, alpha_theta: np.ndarray) -> float:
    f = alpha_theta - theta
    s = np.sign(f)
    cross = np.where(s[:-1] * s[1:] < 0)[0]
    if cross.size == 0:
        # fallback: nearest approach
        idx = int(np.argmin(np.abs(f)))
        return float(theta[idx])
    i = int(cross[0])
    a, b = float(theta[i]), float(theta[i+1])
    return float(brentq(lambda t: np.interp(t, theta, alpha_theta) - t, a, b))


def run_cluster(cluster: str, z_l: float, z_s: float, debug: bool = True) -> Dict:
    theta = np.linspace(5.0, 120.0, 220)  # arcsec

    # GR baseline
    R, Sigma, M_enc, alpha_R, alpha_gr = alpha_gr_from_baryons(cluster, z_l, z_s, theta)

    # Features -> slip params -> S(R)
    feats = extract_features(R, Sigma)
    S_inf, Rs = predict_slip_params(feats)
    Dd = float(angular_diameter_distance_kpc(z_l))
    S_R = slip_profile(R, Sigma, S_inf, Rs, p=1.2, eta=1.0, x0=0.3, w=0.3)
    
    if debug:
        print(f"\n=== SLIP MODEL ===")
        print(f"Features: R_edge={feats.R_edge:.2f} kpc, edge_sharp={feats.edge_sharp:.3f}, M_core={feats.M_core:.3e} Msun")
        print(f"Predicted: S_inf={S_inf:.3f}, Rs={Rs:.2f} kpc")
        print(f"S(R) range: [{S_R.min():.3f}, {S_R.max():.3f}]")
        print(f"S(R=100 kpc): {np.interp(100.0, R, S_R):.3f}")

    alpha_model = predict_alpha_model(theta, R, alpha_R, S_R, Dd)
    
    if debug:
        print(f"alpha_model(theta) range: [{alpha_model.min():.3f}, {alpha_model.max():.3f}] arcsec")
        print(f"alpha_model(35\") = {np.interp(35.0, theta, alpha_model):.3f} arcsec")

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