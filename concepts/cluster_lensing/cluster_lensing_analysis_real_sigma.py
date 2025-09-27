#!/usr/bin/env python3
"""
Cluster Lensing Analysis with Real Σ(R) Gating (Universal G³)
=============================================================

Uses real cluster gas+stars profiles (data/clusters/<name>/) to build ρ_b(r),
project to Σ_bar(R) via Abel, derive r_half from M_bar(r), and then compute a
G³ tail with gates/screens driven by Σ_bar (in Msun/pc^2) instead of a disk
surrogate. Lensing quantities (κ, κ̄, γ_t, α, θ_E) are then computed.

Outputs:
- out/cluster_lensing_real/<cluster>/* (plots, profiles.csv, summary.json)
- out/cluster_lensing_real/summaries.json

Note: Units for Σ in the universal params were tuned in Msun/pc^2, so Σ_bar
computed in Msun/kpc^2 is converted by dividing by 1e6 when used in screens.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Universal G³ parameters (copied from concepts/cluster_lensing/cluster_lensing_analysis.py)
G = 4.300917270e-6  # kpc km^2 s^-2 Msun^-1
c_km_s = 299792.458  # km/s

UNIVERSAL_PARAMS = {
    "v0_kms": 237.27,
    "rc0_kpc": 28.98,
    "gamma": 0.77,
    "beta": 0.22,
    "sigma_star": 84.49,  # Msun/pc^2
    "alpha": 1.04,
    "kappa": 1.75,
    "eta": 0.96,
    "delta_kpc": 1.50,
    "p_in": 1.71,
    "p_out": 0.88,
    "g_sat": 2819.45
}
BEST_VARIANT = {
    "gating_type": "rational",
    "screen_type": "sigmoid",
    "exponent_type": "logistic_r"
}

# Cosmology
H0 = 70.0  # km/s/Mpc
Omega_m = 0.3
Omega_L = 1.0 - Omega_m
Mpc_to_kpc = 1_000.0

MU_E = 1.17
M_P_G = 1.67262192369e-24
KPC_CM = 3.0856775814913673e21
MSUN_G = 1.988409870698051e33


def Ez(z):
    return np.sqrt(Omega_m * (1 + z) ** 3 + Omega_L)


def comoving_distance_Mpc(z: float, n: int = 2048) -> float:
    if z <= 0:
        return 0.0
    zs = np.linspace(0.0, z, n)
    integrand = 1.0 / Ez(zs)
    Dc = (c_km_s / H0) * np.trapezoid(integrand, zs)
    return Dc


def angular_diameter_distance_kpc(z: float) -> float:
    Dc = comoving_distance_Mpc(z)
    Da = Dc / (1 + z)
    return Da * Mpc_to_kpc


def sigma_crit_Msun_per_kpc2(z_d: float, z_s: float) -> float:
    if z_s <= z_d:
        return np.inf
    Dd = angular_diameter_distance_kpc(z_d)
    Ds = angular_diameter_distance_kpc(z_s)
    Dc_s = comoving_distance_Mpc(z_s) * Mpc_to_kpc
    Dc_d = comoving_distance_Mpc(z_d) * Mpc_to_kpc
    Dds_ang = (Dc_s - Dc_d) / (1 + z_s)
    return (c_km_s ** 2) / (4 * math.pi * G) * (Ds / (Dd * max(Dds_ang, 1e-12)))


def enclosed_mass_to_density(r: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
    dM_dr = np.gradient(M_enc, r, edge_order=2)
    rho = dM_dr / (4 * math.pi * np.clip(r, 1e-9, None) ** 2)
    return np.maximum(rho, 0.0)


def abel_project_sigma(r: np.ndarray, rho: np.ndarray, R: np.ndarray) -> np.ndarray:
    r = np.asarray(r)
    rho = np.asarray(rho)
    R = np.asarray(R)
    Sigma = np.zeros_like(R)
    for i, Rp in enumerate(R):
        mask = r >= max(Rp, r[0])
        rr = r[mask]
        rh = rho[mask]
        if rr.size < 2:
            Sigma[i] = 0.0
            continue
        integrand = rh * rr / np.sqrt(np.clip(rr ** 2 - Rp ** 2, 1e-20, None))
        Sigma[i] = 2.0 * np.trapezoid(integrand, rr)
    return Sigma


def sigma_to_Mproj(R: np.ndarray, Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Mproj = np.array([2 * math.pi * np.trapezoid(Sigma[:i+1] * R[:i+1], R[:i+1]) for i in range(len(R))])
    area = math.pi * R ** 2
    Sbar = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area > 0)
    return Mproj, Sbar


@dataclass
class ClusterSpec:
    name: str
    z_lens: float
    z_source: float


def load_real_cluster_profiles(name: str) -> Tuple[np.ndarray, np.ndarray]:
    base = Path('data') / 'clusters' / name
    gas_path = base / 'gas_profile.csv'
    star_path = base / 'stars_profile.csv'
    r_g = rho_g = r_s = rho_s = None
    if gas_path.exists():
        g = pd.read_csv(gas_path)
        if 'rho_gas_Msun_per_kpc3' in g.columns:
            r_g = g['r_kpc'].to_numpy(float)
            rho_g = g['rho_gas_Msun_per_kpc3'].to_numpy(float)
        elif 'n_e_cm3' in g.columns:
            r_g = g['r_kpc'].to_numpy(float)
            ne = g['n_e_cm3'].to_numpy(float)
            rho_g_cm3 = MU_E * M_P_G * ne
            rho_g = rho_g_cm3 * (KPC_CM**3) / MSUN_G
    if star_path.exists():
        s = pd.read_csv(star_path)
        if 'rho_star_Msun_per_kpc3' in s.columns:
            r_s = s['r_kpc'].to_numpy(float)
            rho_s = s['rho_star_Msun_per_kpc3'].to_numpy(float)
    if r_g is None and r_s is None:
        raise FileNotFoundError(f"No cluster profiles found for {name}")
    if r_g is None:
        r = r_s; rho = rho_s
    elif r_s is None:
        r = r_g; rho = rho_g
    else:
        r = np.union1d(r_g, r_s)
        rho = np.interp(r, r_g, rho_g) + np.interp(r, r_s, rho_s)
    m = np.isfinite(r) & np.isfinite(rho) & (r > 0)
    r = r[m]; rho = np.maximum(0.0, rho[m])
    i = np.argsort(r)
    return r[i], rho[i]


def g3_tail_with_real_sigma(r: np.ndarray, r_half: float, Sigma_bar_kpc2: np.ndarray, params: Dict) -> np.ndarray:
    p = params
    # Convert Σ from Msun/kpc^2 to Msun/pc^2 for screening
    Sigma_pc2 = Sigma_bar_kpc2 / 1e6
    # Effective core radius scaling
    rc_eff = p["rc0_kpc"] * (r_half / 8.0)**p["gamma"]
    # Use mean Σ within r_half as scale for rc_eff's Sigma dependence
    mean_S = float(np.mean(Sigma_pc2[r <= r_half])) if np.any(r <= r_half) else float(np.mean(Sigma_pc2))
    rc_eff = rc_eff * (max(mean_S, 1e-8) / 100.0) ** (-p["beta"])  # same exponent use
    # Variable exponent (logistic_r)
    transition_r = p["eta"] * r_half
    x = (r - transition_r) / (p["delta_kpc"] + 1e-12)
    gate_exp = 1.0 / (1.0 + np.exp(-x))
    p_r = p["p_in"] * (1 - gate_exp) + p["p_out"] * gate_exp
    # Gating (rational)
    with np.errstate(divide='ignore', invalid='ignore'):
        gate = (r**p_r) / (r**p_r + rc_eff**p_r + 1e-20)
    # Screening (sigmoid) on real Σ
    screen = 1.0 / (1.0 + (np.maximum(Sigma_pc2, 1e-12) / (p["sigma_star"] + 1e-12))**p["alpha"])**p["kappa"]
    g_tail = (p["v0_kms"]**2 / np.maximum(r, 1e-12)) * gate * screen
    g_tail = p["g_sat"] * np.tanh(g_tail / (p["g_sat"] + 1e-12))
    return g_tail


def compute_cluster(name: str, z_lens: float, z_source: float, outdir: Path) -> Dict:
    params = UNIVERSAL_PARAMS
    r, rho = load_real_cluster_profiles(name)
    # Enclosed mass and g_bar
    M_enc = np.zeros_like(r)
    if r.size > 1:
        integrand = rho * r * r
        M_enc[1:] = 4.0 * math.pi * np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))
    g_bar = G * M_enc / np.maximum(r*r, 1e-12)
    V_bar = np.sqrt(np.maximum(g_bar, 0.0) * r)
    # Project Σ_bar
    R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
    Sigma_bar = abel_project_sigma(r, rho, R)
    # Compute half-mass radius
    Mb_tot = float(M_enc[-1]) if M_enc.size > 0 else 1.0
    half = 0.5 * Mb_tot
    idx = np.searchsorted(M_enc, half)
    if 0 < idx < len(r):
        r_half = float(r[idx])
    else:
        r_half = float(np.median(r))
    # Tail using real Σ
    g_tail_R = g3_tail_with_real_sigma(R, r_half, Sigma_bar, params)
    # gbar on R
    Vbar_R = np.interp(R, r, V_bar)
    gbar_R = Vbar_R**2 / np.maximum(R, 1e-12)
    g_total_R = gbar_R + g_tail_R
    # Effective mass and lensing
    M_eff_R = g_total_R * R**2 / G
    rho_eff = enclosed_mass_to_density(R, M_eff_R)
    Sigma_eff = abel_project_sigma(R, rho_eff, R)
    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)
    kappa_bar = Sigma_bar / Sigma_crit
    kappa_eff = Sigma_eff / Sigma_crit
    Mproj_bar, Sbar_bar = sigma_to_Mproj(R, Sigma_bar)
    Mproj_eff, Sbar_eff = sigma_to_Mproj(R, Sigma_eff)
    kappa_bar_mean = Sbar_bar / Sigma_crit
    kappa_eff_mean = Sbar_eff / Sigma_crit

    def find_theta_E(R_kpc, kappa_mean):
        idx = np.where(kappa_mean >= 1.0)[0]
        if idx.size == 0:
            return None
        i = idx[0]
        if i == 0:
            R_E = R_kpc[0]
        else:
            x0, y0 = R_kpc[i-1], kappa_mean[i-1]
            x1, y1 = R_kpc[i], kappa_mean[i]
            if y1 == y0:
                R_E = x1
            else:
                R_E = x0 + (1 - y0) * (x1 - x0) / (y1 - y0)
        Dd = angular_diameter_distance_kpc(z_lens)
        theta_rad = R_E / max(Dd, 1e-12)
        theta_arcsec = theta_rad * (180/math.pi) * 3600
        return R_E, theta_arcsec

    RE_bar = find_theta_E(R, kappa_bar_mean)
    RE_eff = find_theta_E(R, kappa_eff_mean)

    # Plots
    outdir.mkdir(parents=True, exist_ok=True)
    fig1, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].loglog(R, kappa_bar, label='κ_bar', lw=2)
    ax[0].loglog(R, kappa_eff, label='κ_eff (G³ real-Σ)', lw=2)
    ax[0].loglog(R, kappa_bar_mean, '--', label='\u0304κ_bar(<R)')
    ax[0].loglog(R, kappa_eff_mean, '--', label='\u0304κ_eff(<R)')
    ax[0].axhline(1.0, color='k', ls=':')
    ax[0].set_xlabel('R (kpc)'); ax[0].set_ylabel('Convergence κ')
    ax[0].set_title(f'{name}: Convergence (real-Σ)')
    ax[0].grid(True, which='both', alpha=0.3); ax[0].legend(fontsize=8)

    ax[1].loglog(R, kappa_bar_mean - kappa_bar, label='γ_t (baryons)', lw=2)
    ax[1].loglog(R, kappa_eff_mean - kappa_eff, label='γ_t (G³ real-Σ)', lw=2)
    ax[1].set_xlabel('R (kpc)'); ax[1].set_ylabel('Tangential shear γ_t')
    ax[1].set_title(f'{name}: Shear (real-Σ)')
    ax[1].grid(True, which='both', alpha=0.3); ax[1].legend(fontsize=8)
    fig1.tight_layout(); fig1.savefig(outdir / 'kappa_gamma_realSigma.png', dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6,5))
    ax2.loglog(R, Sigma_bar, label='Σ_bar', lw=2)
    ax2.loglog(R, Sigma_eff, label='Σ_eff (G³ real-Σ)', lw=2)
    ax2.axhline(Sigma_crit, color='k', ls=':', label='Σ_crit')
    ax2.set_xlabel('R (kpc)'); ax2.set_ylabel('Σ (Msun/kpc^2)')
    ax2.set_title(f'{name}: Surface density (real-Σ)')
    ax2.grid(True, which='both', alpha=0.3); ax2.legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig(outdir / 'surface_density_realSigma.png', dpi=150)

    # Save
    import csv
    with open(outdir / 'profiles_realSigma.csv', 'w', newline='') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['R_kpc','Sigma_bar','Sigma_eff','kappa_bar','kappa_eff','kappa_bar_mean','kappa_eff_mean'])
        for i in range(len(R)):
            w.writerow([float(R[i]), float(Sigma_bar[i]), float(Sigma_eff[i]), float(kappa_bar[i]), float(kappa_eff[i]), float(kappa_bar_mean[i]), float(kappa_eff_mean[i])])

    summary = {
        'cluster': name,
        'z_lens': z_lens,
        'z_source': z_source,
        'Einstein_radius_arcsec_baryons': None if RE_bar is None else float(RE_bar[1]),
        'Einstein_radius_kpc_baryons': None if RE_bar is None else float(RE_bar[0]),
        'Einstein_radius_arcsec_realSigma': None if RE_eff is None else float(RE_eff[1]),
        'Einstein_radius_kpc_realSigma': None if RE_eff is None else float(RE_eff[0]),
        'max_kappa_eff_mean': float(np.nanmax(kappa_eff_mean)),
        'note': 'G³ tail gated by real Σ(R) with universal params.'
    }
    with open(outdir / 'summary_realSigma.json', 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    clusters = [
        ('Abell_1689', 0.183, 1.0),
        ('Coma', 0.0231, 0.8),
        ('Bullet', 0.296, 1.2),
        ('A2029', 0.0767, 1.0),
        ('A478', 0.0881, 1.0),
        ('A1795', 0.0622, 1.0),
        ('ABELL_0426', 0.0179, 0.8),
    ]
    out_root = Path('out') / 'cluster_lensing_real'
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict] = []
    for name, z_l, z_s in clusters:
        try:
            s = compute_cluster(name, z_l, z_s, out_root / name)
            summaries.append(s)
            print(f"[real-Σ] {name}: θ_E={s['Einstein_radius_arcsec_realSigma']} arcsec")
        except Exception as e:
            print(f"[real-Σ] Skipped {name}: {e}")
    with open(out_root / 'summaries.json', 'w') as f:
        json.dump(summaries, f, indent=2)


if __name__ == '__main__':
    main()
