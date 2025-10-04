#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from concepts.squared_gravity.geometric_exponent import GeometricExponentGravity
from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    abel_project_sigma, enclosed_mass_to_density, sigma_crit_Msun_per_kpc2,
    load_real_cluster_profiles, angular_diameter_distance_kpc
)

# CLASH mapping and lens redshifts (same as other scripts)
CLASH: List[Tuple[str, str, float]] = [
    ('a1423', 'ABELL_1423', 0.213), ('a209', 'ABELL_0209', 0.206), ('a2261', 'ABELL_2261', 0.224),
    ('a383', 'ABELL_0383', 0.187), ('a611', 'ABELL_0611', 0.288), ('clj1226', 'CLJ1226', 0.890),
    ('macs0329', 'MACSJ0329', 0.450), ('macs0416', 'MACSJ0416', 0.396), ('macs0429', 'MACSJ0429', 0.399),
    ('macs0647', 'MACSJ0647', 0.584), ('macs0717', 'MACSJ0717', 0.548), ('macs0744', 'MACSJ0744', 0.686),
    ('macs1115', 'MACSJ1115', 0.352), ('macs1149', 'MACSJ1149', 0.544), ('macs1206', 'MACSJ1206', 0.440),
    ('macs1311', 'MACSJ1311', 0.494), ('macs1423', 'MACSJ1423', 0.545), ('macs1720', 'MACSJ1720', 0.391),
    ('macs1931', 'MACSJ1931', 0.352), ('macs2129', 'MACSJ2129', 0.570), ('ms2137', 'MS2137', 0.313),
    ('rxj1347', 'RXJ1347', 0.451), ('rxj1532', 'RXJ1532', 0.345), ('rxj2129', 'RXJ2129', 0.234), ('rxj2248', 'RXJ2248', 0.348)
]


def find_theta_E(R_kpc: np.ndarray, kappa_mean: np.ndarray, z_lens: float) -> float | None:
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
    theta_arcsec = theta_rad * (180/np.pi) * 3600
    return float(theta_arcsec)


def build_sigma_bar(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load 3D baryon density for a cluster and project to Σ_bar(R)."""
    r, rho = load_real_cluster_profiles(name)
    # Build a smooth R grid between min/max r
    R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
    # For numerical stability we rebuild a smooth ρ via M_enc then back to ρ (optional)
    # Use given rho directly for Abel projection
    Sigma_bar = abel_project_sigma(r, rho, R)
    return R, Sigma_bar


def evaluate_cluster_GE(local_name: str, z_lens: float, z_source: float,
                         model: GeometricExponentGravity,
                         Rd_kpc: float, R_scale_kpc: float) -> Dict:
    R, Sigma_bar = build_sigma_bar(local_name)
    Sigma_eff, fX, beta = model.Sigma_effective(R, Sigma_bar, Rd_kpc)

    # Lensing
    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)
    kappa_bar = Sigma_bar / Sigma_crit
    kappa_eff = Sigma_eff / Sigma_crit

    # Compute mean kappas
    def sigma_to_Mproj(R, Sigma):
        Mproj = np.array([2*np.pi*np.trapz(Sigma[:i+1]*R[:i+1], R[:i+1]) for i in range(len(R))])
        area = np.pi * R**2
        Sbar = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area>0)
        return Mproj, Sbar

    _, Sbar_bar = sigma_to_Mproj(R, Sigma_bar)
    _, Sbar_eff = sigma_to_Mproj(R, Sigma_eff)
    kappa_bar_mean = Sbar_bar / Sigma_crit
    kappa_eff_mean = Sbar_eff / Sigma_crit

    theta_E_bar = find_theta_E(R, kappa_bar_mean, z_lens)
    theta_E_eff = find_theta_E(R, kappa_eff_mean, z_lens)

    return {
        'R_kpc': R.tolist(),
        'Sigma_bar': Sigma_bar.tolist(),
        'Sigma_eff': Sigma_eff.tolist(),
        'beta': beta.tolist(),
        'fX': fX.tolist(),
        'theta_E_bar_arcsec': theta_E_bar,
        'theta_E_eff_arcsec': theta_E_eff,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--zs', type=float, default=2.0)
    ap.add_argument('--rscale', type=float, default=100.0, help='R_scale for beta (kpc)')
    ap.add_argument('--rd', type=float, default=1000.0, help='Rd scale (kpc) for x=R/Rd')
    ap.add_argument('--a', type=float, default=0.8)
    ap.add_argument('--b', type=float, default=0.2)
    ap.add_argument('--d', type=float, default=0.1)
    ap.add_argument('--gamma1', type=float, default=0.5)
    ap.add_argument('--gamma2', type=float, default=0.3)
    args = ap.parse_args()

    outdir = ROOT / 'data' / 'clash' / 'processed' / 'squared_gravity'
    outdir.mkdir(parents=True, exist_ok=True)

    model = GeometricExponentGravity(a=args.a, b=args.b, d=args.d,
                                     gamma1=args.gamma1, gamma2=args.gamma2,
                                     R_scale_kpc=args.rscale)

    rows = []
    details = {}
    for cid, local, zl in CLASH:
        try:
            res = evaluate_cluster_GE(local, zl, args.zs, model, Rd_kpc=args.rd, R_scale_kpc=args.rscale)
            rows.append({'cluster_id': cid,
                         'theta_E_bar_arcsec': res['theta_E_bar_arcsec'],
                         'theta_E_eff_arcsec': res['theta_E_eff_arcsec']})
            details[cid] = res
        except Exception as e:
            rows.append({'cluster_id': cid,
                         'theta_E_bar_arcsec': None,
                         'theta_E_eff_arcsec': None,
                         'error': str(e)})
    pd.DataFrame(rows).to_csv(outdir / 'einstein_radii_sg.csv', index=False)
    with open(outdir / 'profiles_sg.json', 'w', encoding='utf-8') as f:
        json.dump(details, f, indent=2)
    print(f"[SG] wrote {outdir/'einstein_radii_sg.csv'} and profiles_sg.json")


if __name__ == '__main__':
    main()
