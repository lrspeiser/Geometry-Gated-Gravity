#!/usr/bin/env python3
from __future__ import annotations
"""
Run the telegraph operator on CLASH clusters and compute θE predictions.

Inputs:
- data/clusters/<NAME>/{gas_profile.csv,stars_profile.csv,temp_profile.csv}
- Observed lens redshifts and θE in data/clash/einstein_radii_observed.csv

Outputs:
- data/clash/processed/telegraph/einstein_radii_telegraph.csv
- data/clash/processed/telegraph/profiles/<cluster>.csv (R, Σ_bar, Σ_eff, kappas)
- data/clash/processed/eval/er_telegraph_metrics.json (written by separate eval script)

This is NOT an NFW fit; it’s a nonlocal mapping of Σ_bar to Σ_eff governed by a
few global hyperparameters. It aims to raise central convergence in a way that
responds to baryon geometry.
"""
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import existing baryon/lensing utilities
from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    load_real_cluster_profiles, abel_project_sigma, sigma_crit_Msun_per_kpc2,
    angular_diameter_distance_kpc,
)
from concepts.cluster_lensing.telegraph_operator import TelegraphParams, apply_telegraph

# CLASH list (id, local_name, z_lens)
CLASH: List[Tuple[str, str, float]] = [
    ('a1423','ABELL_1423',0.213), ('a209','ABELL_0209',0.206), ('a2261','ABELL_2261',0.224),
    ('a383','ABELL_0383',0.187), ('a611','ABELL_0611',0.288), ('clj1226','CLJ1226',0.890),
    ('macs0329','MACSJ0329',0.450), ('macs0416','MACSJ0416',0.396), ('macs0429','MACSJ0429',0.399),
    ('macs0647','MACSJ0647',0.584), ('macs0717','MACSJ0717',0.548), ('macs0744','MACSJ0744',0.686),
    ('macs1115','MACSJ1115',0.352), ('macs1149','MACSJ1149',0.544), ('macs1206','MACSJ1206',0.440),
    ('macs1311','MACSJ1311',0.494), ('macs1423','MACSJ1423',0.545), ('macs1720','MACSJ1720',0.391),
    ('macs1931','MACSJ1931',0.352), ('macs2129','MACSJ2129',0.570), ('ms2137','MS2137',0.313),
    ('rxj1347','RXJ1347',0.451), ('rxj1532','RXJ1532',0.345), ('rxj2129','RXJ2129',0.234), ('rxj2248','RXJ2248',0.348),
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
    theta_arcsec = (R_E / max(Dd, 1e-12)) * (180.0/np.pi) * 3600.0
    return float(theta_arcsec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zs', type=float, default=2.0, help='Source redshift for Sigma_crit(zl,zs)')
    ap.add_argument('--lambda_amp', type=float, default=0.35)
    ap.add_argument('--mu_logR', type=float, default=-0.6)
    ap.add_argument('--sigma_logR', type=float, default=0.45)
    ap.add_argument('--clip_up', type=float, default=1.0)
    ap.add_argument('--clip_down', type=float, default=0.8)
    ap.add_argument('--no_renorm', action='store_true', help='Disable total-mass renormalization')
    ap.add_argument('--include', type=str, default=None, help='Comma-separated cluster_ids to include (else all)')
    ap.add_argument('--exclude', type=str, default=None, help='Comma-separated cluster_ids to exclude')
    args = ap.parse_args()

    params = TelegraphParams(
        lambda_amp=args.lambda_amp,
        mu_logR=args.mu_logR,
        sigma_logR=args.sigma_logR,
        clip_up=args.clip_up,
        clip_down=args.clip_down,
        renorm_total_mass=(not args.no_renorm),
    )

    include_set = set([s.strip().lower() for s in args.include.split(',')]) if args.include else None
    exclude_set = set([s.strip().lower() for s in args.exclude.split(',')]) if args.exclude else set()

    out_root = ROOT / 'data' / 'clash' / 'processed' / 'telegraph'
    out_root.mkdir(parents=True, exist_ok=True)
    prof_dir = out_root / 'profiles'
    prof_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for cid, local, zl in CLASH:
        if include_set is not None and cid not in include_set:
            continue
        if cid in exclude_set:
            continue
        try:
            r, rho = load_real_cluster_profiles(local)
            # Log-spaced R grid
            R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
            Sigma_bar = abel_project_sigma(r, rho, R)

            # Apply telegraph operator
            Sigma_eff, diag = apply_telegraph(R, Sigma_bar, params)

            # Lensing quantities
            Sigma_crit = sigma_crit_Msun_per_kpc2(zl, args.zs)
            # κ and κ̄
            def sigma_to_Mproj(R, S):
                Mproj = np.array([2*np.pi*np.trapezoid(S[:i+1]*R[:i+1], R[:i+1]) for i in range(len(R))])
                area = np.pi * R**2
                Sbar = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area>0)
                return Mproj, Sbar
            _, Sbar_bar = sigma_to_Mproj(R, Sigma_bar)
            _, Sbar_eff = sigma_to_Mproj(R, Sigma_eff)
            kbar_bar = Sbar_bar / Sigma_crit
            kbar_eff = Sbar_eff / Sigma_crit

            theta_bar = find_theta_E(R, kbar_bar, zl)
            theta_eff = find_theta_E(R, kbar_eff, zl)

            rows.append(dict(
                cluster_id=cid,
                cluster_label=local,
                z_lens=zl,
                z_source=args.zs,
                theta_E_bar_arcsec=None if theta_bar is None else float(theta_bar),
                theta_E_eff_arcsec=None if theta_eff is None else float(theta_eff),
                lambda_amp=float(params.lambda_amp),
                mu_logR=float(params.mu_logR),
                sigma_logR=float(params.sigma_logR),
                clip_up=float(params.clip_up),
                clip_down=float(params.clip_down),
                renorm_total_mass=bool(params.renorm_total_mass),
            ))

            # Save profile for diagnostics
            dfp = pd.DataFrame(dict(
                R_kpc=R,
                Sigma_bar_kpc2=Sigma_bar,
                Sigma_eff_kpc2=Sigma_eff,
                kappa_bar_mean=kbar_bar,
                kappa_eff_mean=kbar_eff,
            ))
            dfp.to_csv(prof_dir / f'{cid}.csv', index=False)
            print(f'[telegraph] {cid}: θE_eff={theta_eff}')
        except Exception as e:
            rows.append(dict(
                cluster_id=cid,
                cluster_label=local,
                z_lens=zl,
                z_source=args.zs,
                theta_E_bar_arcsec=None,
                theta_E_eff_arcsec=None,
                error=str(e),
            ))
            print(f'[telegraph] {cid} FAILED: {e}')

    # Write predictions CSV
    out_csv = out_root / 'einstein_radii_telegraph.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()
