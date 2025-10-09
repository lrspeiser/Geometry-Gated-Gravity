#!/usr/bin/env python3
from __future__ import annotations
"""
Report θE comparisons for gold clusters using HFF accepted maps vs GR and GE.

Outputs: data/frontier/gold_standard/report_thetaE.csv

- For each cluster and accepted team/version present under data/frontier/hlsp,
  compute θE_crit from κ/γ maps and θE from:
    * GR (baryons)
    * GE (our formula; params via CLI)

Notes:
- This uses the maps as distributed (no redshift rescaling unless the map encodes
  exact z normalization implicitly). For per-z magnification files, the critical
  curve is internally consistent with that z; we use κ/γ here with native scale.
"""
import argparse
from pathlib import Path
import sys
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lensing_utils import (
    list_frontier_models, compute_thetaEcrit_from_maps,
    alpha_fun_GR_baryons, alpha_fun_GE, solve_theta_E_from_alpha,
    CLASH,
)

GOLD = ['macs0416','macs0717','macs1149']
PREFS = [('cats','v4.1'), ('williams','v4'), ('caminha','v4')]

def choose_models(cluster_id: str):
    found = []
    models = list_frontier_models(cluster_id)
    for team, ver in PREFS:
        if team in models:
            versions = [v.lower() for v in models[team]]
            if ver in versions:
                found.append((team, ver))
            elif versions:
                found.append((team, models[team][0]))
    # de-dup
    seen = set(); uniq = []
    for t,v in found:
        key = (t,v)
        if key not in seen:
            uniq.append(key); seen.add(key)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clusters', type=str, default=','.join(GOLD))
    ap.add_argument('--zs', type=float, default=2.0)
    # GE params
    ap.add_argument('--ge_gamma1', type=float, default=0.2)
    ap.add_argument('--ge_gamma2', type=float, default=0.1)
    ap.add_argument('--ge_a', type=float, default=3.0)
    ap.add_argument('--ge_b', type=float, default=0.2)
    ap.add_argument('--ge_d', type=float, default=0.1)
    ap.add_argument('--ge_Rd_kpc', type=float, default=1000.0)
    ap.add_argument('--ge_Rscale_kpc', type=float, default=100.0)
    ap.add_argument('--out', type=str, default=str(ROOT / 'data' / 'frontier' / 'gold_standard' / 'report_thetaE.csv'))
    args = ap.parse_args()

    clusters = [c.strip().lower() for c in args.clusters.split(',') if c.strip()]

    rows = []
    for cid in clusters:
        if cid not in CLASH:
            print(f'skip unknown cluster: {cid}')
            continue
        local_name, z_lens = CLASH[cid]
        # GR/GE alpha functions
        alpha_gr = alpha_fun_GR_baryons(local_name, z_lens, args.zs)
        alpha_ge = alpha_fun_GE(local_name, z_lens, args.zs,
                                a=args.ge_a, b=args.ge_b, d=args.ge_d,
                                gamma1=args.ge_gamma1, gamma2=args.ge_gamma2,
                                Rd_kpc=args.ge_Rd_kpc, R_scale_kpc=args.ge_Rscale_kpc,
                                beta_clip=(1.0,5.0))
        # initial bracket using 5..80 arcsec
        def solve_any(alpha_fun):
            if alpha_fun is None:
                return np.nan
            val = solve_theta_E_from_alpha(alpha_fun, 20.0, 5.0, 80.0)
            return float(val) if val is not None else np.nan
        thetaE_GR = solve_any(alpha_gr)
        thetaE_GE = solve_any(alpha_ge)

        for team, ver in choose_models(cid):
            thetaE_crit = compute_thetaEcrit_from_maps(cid, team, ver)
            rows.append({
                'cluster_id': cid,
                'team': team,
                'version': ver,
                'zs': args.zs,
                'thetaE_crit_arcsec': (float(thetaE_crit) if thetaE_crit is not None else np.nan),
                'thetaE_GR_arcsec': thetaE_GR,
                'thetaE_GE_arcsec': thetaE_GE,
            })

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['cluster_id','team','version','zs','thetaE_crit_arcsec','thetaE_GR_arcsec','thetaE_GE_arcsec'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'Wrote {outp}')

if __name__ == '__main__':
    main()
