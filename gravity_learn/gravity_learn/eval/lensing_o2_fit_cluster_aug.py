#!/usr/bin/env python
from __future__ import annotations
"""
Fit small geometry-based augmentation weights for O2 lensing on real cluster profiles.

We keep the O2 best_family params fixed (from SPARC fits), and add a multiplicative
augmentation to the lensing field:
  g_total_aug(R) = g_total(R) * (1 + w_env * SigL_hat(R) + w_curv * C_hat(R))

where:
- SigL_hat is a Gaussian-smoothed Σ_bar(R) normalized to [0,1] by its finite-median.
- C_hat is a curvature-like feature from Σ: the magnitude of second derivative of ln Σ vs ln R,
  normalized by its finite-median.

Objective: fit (w_env, w_curv) globally across clusters to match observed θ_E, while
penalizing spurious strong lensing where none is observed.

Outputs per run:
- gravity_learn/experiments/eval/global_fit/<RUN>/lensing_o2_aug/<cluster>/* (plots, profiles)
- gravity_learn/experiments/eval/global_fit/<RUN>/lensing_o2_aug/summaries_aug.json (fit and per-cluster summary)
"""
import os
import math
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from gravity_learn.eval.lensing_o2_diagnostics import (
    load_best,
    abel_project_sigma,
    enclosed_mass_to_density,
    sigma_crit_Msun_per_kpc2,
    comoving_distance_Mpc, Mpc_to_kpc, angular_diameter_distance_kpc,
    ClusterSpec as O2ClusterSpec,
    FAMILIES,
)
# Reuse real profile loader logic
from gravity_learn.eval.lensing_o2_diagnostics import _load_real_cluster_profiles as load_profiles

G = 4.300917270e-6
c_km_s = 299792.458

OBS_THETA = {
    'Abell_1689': 47.0,
    'A2029': 28.0,
    'A478': 31.0,
    'A1795': None,
    'ABELL_0426': None,
    'Bullet': 16.0,
    'Coma': None,
}


def gaussian_weights(R: np.ndarray, ell_kpc: float) -> np.ndarray:
    R = np.asarray(R, float)
    N = R.size
    ell = max(float(ell_kpc), 1e-6)
    d = R.reshape(-1, 1) - R.reshape(1, -1)
    W = np.exp(-0.5 * (d / ell) ** 2)
    W /= (W.sum(axis=1, keepdims=True) + 1e-30)
    return W


def curvature_sigma_ln(R: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    R = np.asarray(R, float)
    S = np.maximum(np.asarray(Sigma, float), 1e-30)
    lnR = np.log(np.maximum(R, 1e-30))
    lnS = np.log(S)
    d1 = np.gradient(lnS, lnR, edge_order=2)
    d2 = np.gradient(d1, lnR, edge_order=2)
    return np.abs(d2)


def features_from_sigma(R: np.ndarray, Sigma: np.ndarray, ell_kpc: float = 150.0) -> Tuple[np.ndarray, np.ndarray]:
    W = gaussian_weights(R, ell_kpc)
    SigL = W @ np.asarray(Sigma, float)
    med = np.median(SigL[np.isfinite(SigL)]) if np.any(np.isfinite(SigL)) else 1.0
    SigL_hat = SigL / max(med, 1e-30)
    Curv = curvature_sigma_ln(R, Sigma)
    medC = np.median(Curv[np.isfinite(Curv)]) if np.any(np.isfinite(Curv)) else 1.0
    Curv_hat = Curv / max(medC, 1e-30)
    return SigL_hat, Curv_hat


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
    return R_E


def compute_o2_baseline(spec: O2ClusterSpec, fam_key: str, params: List[float]) -> Dict:
    # Real profiles
    r, rho, _ = load_profiles(spec.name)
    if r is None or rho is None:
        raise FileNotFoundError(f"No real profiles for {spec.name}")
    R = np.logspace(np.log10(max(1.0, r[0])), np.log10(spec.r200_kpc), 600)
    # Σ_bar
    Sigma_bar = abel_project_sigma(r, rho, R)
    # Half-mass radius and gbar
    M_enc = np.zeros_like(r)
    if r.size > 1:
        integrand = rho * r * r
        M_enc[1:] = 4.0 * math.pi * np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))
    Vbar = np.sqrt(G * np.maximum(M_enc, 0.0) / np.maximum(r, 1e-12))
    Vbar_R = np.interp(R, r, Vbar)
    gbar_R = Vbar_R ** 2 / np.maximum(R, 1e-12)
    # O2 features
    from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma
    # r_half from half of M_enc
    Mb_tot = float(M_enc[-1]) if M_enc.size > 0 else 1.0
    idx = np.searchsorted(M_enc, 0.5 * Mb_tot)
    r_half = float(r[idx]) if 0 < idx < len(r) else float(np.median(r))
    x = dimensionless_radius(R, r_half=r_half)
    Sh = sigma_hat(Sigma_bar)
    dlnS = grad_log_sigma(R, Sigma_bar)
    # O2 fX
    f_entry = FAMILIES[fam_key]
    f = f_entry['func']
    try:
        fX = f(params, x, Sh, dlnS, gbar_R)
    except TypeError:
        fX = f(params, x, Sh, dlnS)
    fX = np.maximum(fX, 0.0)
    g_total = gbar_R * (1.0 + fX)
    return {
        'R': R, 'r': r, 'rho': rho, 'Sigma_bar': Sigma_bar,
        'gbar_R': gbar_R, 'g_total': g_total,
        'z_lens': spec.z_lens, 'z_source': spec.z_source,
    }


def lensing_from_g(R: np.ndarray, g_R: np.ndarray, z_lens: float, z_source: float) -> Dict:
    M_eff_R = g_R * R ** 2 / G
    # Map to monotonic 3D grid for projection (reuse R as 3D)
    rho_eff = enclosed_mass_to_density(R, M_eff_R)
    Sigma_eff = abel_project_sigma(R, rho_eff, R)
    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)
    kappa = Sigma_eff / Sigma_crit
    Mproj_eff, Sbar_eff = (np.array([2 * math.pi * np.trapezoid(Sigma_eff[:i+1] * R[:i+1], R[:i+1]) for i in range(len(R))]),
                           np.zeros_like(R))
    area = math.pi * R ** 2
    Sbar_eff = np.divide(Mproj_eff, area, out=np.zeros_like(Mproj_eff), where=area > 0)
    kappa_mean = Sbar_eff / Sigma_crit
    return {'Sigma_eff': Sigma_eff, 'kappa': kappa, 'kappa_mean': kappa_mean}


def fit_aug(best_json: str, outdir: str, clusters: List[Tuple[str, float, float]], ell_kpc: float = 150.0):
    fam, params = load_best(best_json)
    out_root = Path(outdir) / 'lensing_o2_aug'
    out_root.mkdir(parents=True, exist_ok=True)
    specs = [O2ClusterSpec(name=c[0], z_lens=c[1], z_source=c[2], r200_kpc=2200.0, M200_Msun=2e15, rhalf_kpc=700.0) for c in clusters]
    # Compute baselines and features
    data = {}
    for sp, (name, z_l, z_s) in zip(specs, clusters):
        try:
            base = compute_o2_baseline(sp, fam, params)
        except Exception:
            continue
        R = base['R']; Sigma_bar = base['Sigma_bar']
        SigL_hat, Curv_hat = features_from_sigma(R, Sigma_bar, ell_kpc=ell_kpc)
        data[name] = {**base, 'SigL_hat': SigL_hat, 'Curv_hat': Curv_hat}
    # Grid-search weights
    w_grid = np.linspace(0.0, 2.0, 21)
    best = None
    for w_env in w_grid:
        for w_curv in w_grid:
            total = 0.0; valid = True
            for (name, z_l, z_s) in clusters:
                if name not in data:
                    continue
                d = data[name]
                R = d['R']; g = d['g_total']; SigL = d['SigL_hat']; Curv = d['Curv_hat']
                aug = np.maximum(0.0, 1.0 + w_env * SigL + w_curv * Curv)
                g_aug = g * aug
                lens = lensing_from_g(R, g_aug, z_l, z_s)
                kbar = lens['kappa_mean']
                RE_kpc = find_theta_E(R, kbar)
                theta_obs = OBS_THETA.get(name)
                if theta_obs is not None:
                    if RE_kpc is None or not np.isfinite(RE_kpc):
                        total += 10.0  # big penalty if we fail to lens where expected
                    else:
                        Dd = angular_diameter_distance_kpc(z_l)
                        theta_pred = (RE_kpc / max(Dd, 1e-12)) * (180.0 / math.pi) * 3600.0
                        rel = abs(theta_pred - theta_obs) / max(theta_obs, 1e-6)
                        total += rel
                else:
                    # No strong lensing expected: penalize if we produce one
                    if RE_kpc is not None and np.isfinite(RE_kpc):
                        total += 1.0
            if best is None or total < best['score']:
                best = {'w_env': float(w_env), 'w_curv': float(w_curv), 'score': float(total)}
    # Write best and per-cluster outputs
    summaries = {'best': best, 'clusters': []}
    for (name, z_l, z_s) in clusters:
        if name not in data:
            continue
        d = data[name]
        R = d['R']; g = d['g_total']; SigL = d['SigL_hat']; Curv = d['Curv_hat']
        aug = np.maximum(0.0, 1.0 + best['w_env'] * SigL + best['w_curv'] * Curv)
        g_aug = g * aug
        lens_aug = lensing_from_g(R, g_aug, z_l, z_s)
        lens_base = lensing_from_g(R, g, z_l, z_s)
        RE_aug = find_theta_E(R, lens_aug['kappa_mean'])
        RE_base = find_theta_E(R, lens_base['kappa_mean'])
        Dd = angular_diameter_distance_kpc(z_l)
        theta_aug = None if RE_aug is None else (RE_aug / max(Dd, 1e-12)) * (180.0 / math.pi) * 3600.0
        theta_base = None if RE_base is None else (RE_base / max(Dd, 1e-12)) * (180.0 / math.pi) * 3600.0
        out_c = out_root / name
        out_c.mkdir(parents=True, exist_ok=True)
        # Plots
        plt.figure(figsize=(6.5, 5))
        plt.loglog(R, lens_base['kappa_mean'], label='k̄ base')
        plt.loglog(R, lens_aug['kappa_mean'], label='k̄ aug')
        plt.axhline(1.0, color='k', ls=':')
        plt.xlabel('R (kpc)'); plt.ylabel('k̄(<R)'); plt.grid(True, which='both', alpha=0.3)
        plt.legend(); plt.tight_layout(); plt.savefig(out_c / 'kappa_mean_aug.png', dpi=150); plt.close()
        # Save summary
        summaries['clusters'].append({
            'cluster': name,
            'theta_obs_arcsec': OBS_THETA.get(name),
            'theta_base_arcsec': None if theta_base is None else float(theta_base),
            'theta_aug_arcsec': None if theta_aug is None else float(theta_aug),
        })
    with open(out_root / 'summaries_aug.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)
    print('[o2-aug] best:', best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--best_json', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    clusters = [
        ('Abell_1689', 0.183, 1.0),
        ('A2029', 0.0767, 1.0),
        ('A478', 0.0881, 1.0),
        ('A1795', 0.0622, 1.0),
        ('ABELL_0426', 0.0179, 0.8),
        ('Bullet', 0.296, 1.2),
        ('Coma', 0.0231, 0.8),
    ]
    fit_aug(args.best_json, args.outdir, clusters, ell_kpc=150.0)


if __name__ == '__main__':
    main()
