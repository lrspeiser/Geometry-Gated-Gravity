#!/usr/bin/env python
from __future__ import annotations
import os
import json
import math
import argparse
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma
from gravity_learn.eval.global_fit_o2 import FAMILIES

# Physical constants
G = 4.300917270e-6  # kpc km^2 s^-2 Msun^-1
c_km_s = 299792.458

# Flat LCDM distances (simple numeric integrals)
H0 = 70.0  # km/s/Mpc
Omega_m = 0.3
Omega_L = 1.0 - Omega_m
Mpc_to_kpc = 1_000.0


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
    # Angular diameter distances in kpc
    Dd = angular_diameter_distance_kpc(z_d)
    Ds = angular_diameter_distance_kpc(z_s)
    Dc_s = comoving_distance_Mpc(z_s) * Mpc_to_kpc
    Dc_d = comoving_distance_Mpc(z_d) * Mpc_to_kpc
    Dds_ang = (Dc_s - Dc_d) / (1 + z_s)
    return (c_km_s ** 2) / (4 * math.pi * G) * (Ds / (Dd * max(Dds_ang, 1e-12)))


def enclosed_mass_to_density(r: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
    dM_dr = np.gradient(M_enc, r, edge_order=2)
    rho = dM_dr / (4 * math.pi * np.clip(r, 1e-9, None) ** 2)
    rho = np.maximum(rho, 0.0)
    return rho


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
    r200_kpc: float
    M200_Msun: float
    rhalf_kpc: float  # geometric scale proxy


def baryon_mass_profile(r: np.ndarray, r200: float, M200: float) -> np.ndarray:
    f_b = 0.15
    Mb_tot = f_b * M200
    c = 3.0
    rs = r200 / c
    def m_nfw(x):
        return np.log(1 + x) - x / (1 + x)
    norm = m_nfw(c)
    M_enc = Mb_tot * m_nfw(r / rs) / max(norm, 1e-12)
    return np.clip(M_enc, 0.0, Mb_tot)


def load_best(best_json: str) -> Tuple[str, List[float]]:
    with open(best_json, 'r', encoding='utf-8') as f:
        d = json.load(f)
    fam = d.get('best_family', d.get('family', 'ratio'))
    params = d.get('params', [])
    return fam, params


def compute_lensing_o2_for_cluster(spec: ClusterSpec, fam_key: str, params: List[float], outdir: Path):
    f_entry = FAMILIES[fam_key]
    f = f_entry['func']

    # Grids
    r = np.logspace(np.log10(5.0), np.log10(spec.r200_kpc), 800)  # kpc
    R = np.logspace(np.log10(1.0), np.log10(spec.r200_kpc), 600)

    # Baryon mass and Vbar
    M_bar = baryon_mass_profile(r, spec.r200_kpc, spec.M200_Msun)
    Vbar = np.sqrt(G * np.maximum(M_bar, 0.0) / np.maximum(r, 1e-12))

    # Baryon rho and projected Sigma (for O2 features)
    rho_bar = enclosed_mass_to_density(r, M_bar)
    Sigma_bar_kpc2 = abel_project_sigma(r, rho_bar, R)  # Msun/kpc^2
    # Use same radii for features/evaluation on R
    x = dimensionless_radius(R, r_half=spec.rhalf_kpc)
    Sh = sigma_hat(Sigma_bar_kpc2)  # scale-free
    dlnS = grad_log_sigma(R, Sigma_bar_kpc2)

    # Map Vbar from r to R (same radial grid so we interpolate)
    Vbar_R = np.interp(R, r, Vbar)
    gbar_R = Vbar_R ** 2 / np.maximum(R, 1e-12)

    # fX and total g
    try:
        fX = f(params, x, Sh, dlnS, gbar_R)
    except TypeError:
        # For older family signatures without gbar param
        fX = f(params, x, Sh, dlnS)
    fX = np.maximum(fX, 0.0)
    g_total_R = gbar_R * (1.0 + fX)

    # Build effective mass profile from g_total (work in R then map back to r grid for projection)
    M_eff_R = g_total_R * R ** 2 / G
    # Interpolate M_eff onto 3D radius grid r for differentiation
    M_eff = np.interp(r, R, M_eff_R)
    rho_eff = enclosed_mass_to_density(r, M_eff)

    # Project to Σ
    Sigma_eff = abel_project_sigma(r, rho_eff, R)

    # Lensing quantities
    Sigma_crit = sigma_crit_Msun_per_kpc2(spec.z_lens, spec.z_source)
    kappa_bar = Sigma_bar_kpc2 / Sigma_crit
    kappa_eff = Sigma_eff / Sigma_crit

    Mproj_bar, Sbar_bar = sigma_to_Mproj(R, Sigma_bar_kpc2)
    Mproj_eff, Sbar_eff = sigma_to_Mproj(R, Sigma_eff)
    kappa_bar_mean = Sbar_bar / Sigma_crit
    kappa_eff_mean = Sbar_eff / Sigma_crit

    # Einstein radius where mean kappa crosses 1
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
        Dd = angular_diameter_distance_kpc(spec.z_lens)
        theta_rad = R_E / max(Dd, 1e-12)
        theta_arcsec = theta_rad * (180 / math.pi) * 3600
        return R_E, theta_arcsec

    RE_bar = find_theta_E(R, kappa_bar_mean)
    RE_eff = find_theta_E(R, kappa_eff_mean)

    # Tangential shear
    gamma_bar = kappa_bar_mean - kappa_bar
    gamma_eff = kappa_eff_mean - kappa_eff

    # Deflection (arcsec)
    Dc_s = comoving_distance_Mpc(spec.z_source) * Mpc_to_kpc
    Dc_d = comoving_distance_Mpc(spec.z_lens) * Mpc_to_kpc
    Dds_ang = (Dc_s - Dc_d) / (1 + spec.z_source)
    Ds_ang = angular_diameter_distance_kpc(spec.z_source)
    geom = max(Dds_ang / max(Ds_ang, 1e-12), 0.0)
    alpha_bar_rad = 4 * G * Mproj_bar / (c_km_s ** 2 * np.maximum(R, 1e-12)) * geom
    alpha_eff_rad = 4 * G * Mproj_eff / (c_km_s ** 2 * np.maximum(R, 1e-12)) * geom
    alpha_bar_arcsec = alpha_bar_rad * (180 / math.pi) * 3600
    alpha_eff_arcsec = alpha_eff_rad * (180 / math.pi) * 3600

    outdir.mkdir(parents=True, exist_ok=True)

    # Save profiles
    import csv
    with open(outdir / 'profiles_o2.csv', 'w', newline='') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['R_kpc','Sigma_bar','Sigma_eff','kappa_bar','kappa_eff','kappa_bar_mean','kappa_eff_mean','gamma_bar','gamma_eff','alpha_bar_arcsec','alpha_eff_arcsec','fX','gbar'])
        for i in range(len(R)):
            w.writerow([float(R[i]), float(Sigma_bar_kpc2[i]), float(Sigma_eff[i]), float(kappa_bar[i]), float(kappa_eff[i]), float(kappa_bar_mean[i]), float(kappa_eff_mean[i]), float(gamma_bar[i]), float(gamma_eff[i]), float(alpha_bar_arcsec[i]), float(alpha_eff_arcsec[i]), float(fX[i]), float(gbar_R[i])])

    # Plots
    fig1, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].loglog(R, kappa_bar, label='κ_bar (baryons)', lw=2)
    ax[0].loglog(R, kappa_eff, label='κ_eff (O2)', lw=2)
    ax[0].loglog(R, kappa_bar_mean, '--', label='\u0304κ_bar(<R)')
    ax[0].loglog(R, kappa_eff_mean, '--', label='\u0304κ_eff(<R)')
    ax[0].axhline(1.0, color='k', ls=':', lw=1)
    ax[0].set_xlabel('R (kpc)'); ax[0].set_ylabel('Convergence κ')
    ax[0].set_title(f'{spec.name}: Convergence (O2)')
    ax[0].grid(True, which='both', alpha=0.3); ax[0].legend(fontsize=8)

    ax[1].loglog(R, gamma_bar, label='γ_t (baryons)', lw=2)
    ax[1].loglog(R, gamma_eff, label='γ_t (O2)', lw=2)
    ax[1].set_xlabel('R (kpc)'); ax[1].set_ylabel('Tangential shear γ_t')
    ax[1].set_title(f'{spec.name}: Shear (O2)')
    ax[1].grid(True, which='both', alpha=0.3); ax[1].legend(fontsize=8)
    fig1.tight_layout(); fig1.savefig(outdir / 'kappa_gamma_o2.png', dpi=150, bbox_inches='tight')

    fig2, ax2 = plt.subplots(figsize=(6,5))
    ax2.loglog(R, alpha_bar_arcsec, label='α (baryons)', lw=2)
    ax2.loglog(R, alpha_eff_arcsec, label='α (O2)', lw=2)
    ax2.set_xlabel('R (kpc)'); ax2.set_ylabel('Deflection α (arcsec)')
    ax2.set_title(f'{spec.name}: Deflection (O2)')
    ax2.grid(True, which='both', alpha=0.3); ax2.legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig(outdir / 'deflection_o2.png', dpi=150, bbox_inches='tight')

    fig3, ax3 = plt.subplots(figsize=(6,5))
    ax3.loglog(R, Sigma_bar_kpc2, label='Σ_bar', lw=2)
    ax3.loglog(R, Sigma_eff, label='Σ_eff (O2)', lw=2)
    Sigma_crit = sigma_crit_Msun_per_kpc2(spec.z_lens, spec.z_source)
    ax3.axhline(Sigma_crit, color='k', ls=':', label='Σ_crit')
    ax3.set_xlabel('R (kpc)'); ax3.set_ylabel('Σ (Msun/kpc^2)')
    ax3.set_title(f'{spec.name}: Surface density (O2)')
    ax3.grid(True, which='both', alpha=0.3); ax3.legend(fontsize=8)
    fig3.tight_layout(); fig3.savefig(outdir / 'surface_density_o2.png', dpi=150, bbox_inches='tight')

    summary = {
        'cluster': spec.name,
        'family': fam_key,
        'params': [float(p) for p in params],
        'z_lens': spec.z_lens,
        'z_source': spec.z_source,
        'r200_kpc': spec.r200_kpc,
        'M200_Msun': spec.M200_Msun,
        'Einstein_radius_arcsec_baryons': None if RE_bar is None else float(RE_bar[1]),
        'Einstein_radius_kpc_baryons': None if RE_bar is None else float(RE_bar[0]),
        'Einstein_radius_arcsec_O2': None if RE_eff is None else float(RE_eff[1]),
        'Einstein_radius_kpc_O2': None if RE_eff is None else float(RE_eff[0]),
        'max_kappa_eff_mean': float(np.nanmax(kappa_eff_mean)),
        'max_kappa_bar_mean': float(np.nanmax(kappa_bar_mean)),
        'note': 'O2-effective lensing computed via g_total = gbar*(1+fX(x,Sh,|dlnS|,gbar)).',
    }
    with open(outdir / 'summary_o2.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary


def run(best_json: str, outdir: str):
    fam, params = load_best(best_json)
    out_root = Path(outdir) / 'lensing_o2'
    clusters = [
        ClusterSpec(name='Abell_1689', z_lens=0.183, z_source=1.0, r200_kpc=2200.0, M200_Msun=2.0e15, rhalf_kpc=700.0),
        ClusterSpec(name='Coma', z_lens=0.0231, z_source=0.8, r200_kpc=2000.0, M200_Msun=1.5e15, rhalf_kpc=650.0),
        ClusterSpec(name='Bullet', z_lens=0.296, z_source=1.2, r200_kpc=1800.0, M200_Msun=1.0e15, rhalf_kpc=600.0),
    ]
    summaries: List[Dict] = []
    for spec in clusters:
        out_c = out_root / spec.name
        summary = compute_lensing_o2_for_cluster(spec, fam, params, out_c)
        summaries.append(summary)
        print(f"[o2-lensing] {spec.name}: θ_E(O2)={summary['Einstein_radius_arcsec_O2']} arcsec; max k̄={summary['max_kappa_eff_mean']:.3f}")
    with open(out_root / 'summaries_o2.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--best_json', type=str, required=True, help='Path to best_family.json from O2 global fit')
    ap.add_argument('--outdir', type=str, required=True, help='Parent output dir (e.g., run outdir)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    run(args.best_json, args.outdir)


if __name__ == '__main__':
    main()
