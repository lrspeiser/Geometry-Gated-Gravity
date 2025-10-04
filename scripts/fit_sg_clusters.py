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
    load_real_cluster_profiles, abel_project_sigma, sigma_crit_Msun_per_kpc2
)

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


def sigma_to_Mproj(R: np.ndarray, Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Mproj = np.array([2*np.pi*np.trapz(Sigma[:i+1]*R[:i+1], R[:i+1]) for i in range(len(R))])
    area = np.pi * R**2
    Sbar = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area>0)
    return Mproj, Sbar


def grid_from_spec(spec: str) -> np.ndarray:
    """Parse start:stop:count -> linspace; supports floats."""
    s, e, n = spec.split(':')
    return np.linspace(float(s), float(e), int(n))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--zs', type=float, default=2.0)
    ap.add_argument('--rscale', type=float, default=100.0)
    ap.add_argument('--rd', type=float, default=1000.0)
    ap.add_argument('--gamma1', type=str, default='0.0:1.0:6')
    ap.add_argument('--gamma2', type=str, default='0.0:0.6:4')
    ap.add_argument('--a', type=str, default='0.3:1.5:5')
    ap.add_argument('--b', type=str, default='0.0:0.6:5')
    ap.add_argument('--d', type=str, default='0.0:0.6:5')
    args = ap.parse_args()

    outdir = ROOT / 'data' / 'clash' / 'processed' / 'squared_gravity'
    outdir.mkdir(parents=True, exist_ok=True)

    # Observations
    obs_path = ROOT / 'data' / 'clash' / 'einstein_radii_observed.csv'
    split_path = ROOT / 'data' / 'clash' / 'train_test_split.csv'
    obs = pd.read_csv(obs_path)
    split = pd.read_csv(split_path)
    obs['cluster_id'] = obs['cluster_id'].str.lower()
    split['cluster_id'] = split['cluster_id'].str.lower()
    zmap = {cid: zl for cid, _, zl in CLASH}
    namemap = {cid: local for cid, local, _ in CLASH}
    train_ids = split[split['set']=='train']['cluster_id'].tolist()

    # Grids
    g1_grid = grid_from_spec(args.gamma1)
    g2_grid = grid_from_spec(args.gamma2)
    a_grid = grid_from_spec(args.a)
    b_grid = grid_from_spec(args.b)
    d_grid = grid_from_spec(args.d)

    def predict_thetaE(cid: str, model: GeometricExponentGravity) -> float | None:
        local = namemap[cid]
        zl = zmap[cid]
        try:
            r, rho = load_real_cluster_profiles(local)
            R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
            Sigma_bar = abel_project_sigma(r, rho, R)
            Sigma_eff, _, _ = model.Sigma_effective(R, Sigma_bar, Rd_kpc=args.rd)
            Sigma_crit = sigma_crit_Msun_per_kpc2(zl, args.zs)
            _, Sbar_eff = sigma_to_Mproj(R, Sigma_eff)
            kappa_eff_mean = Sbar_eff / Sigma_crit
            # Find θ_E
            idx = np.where(kappa_eff_mean >= 1.0)[0]
            if idx.size == 0:
                return None
            i = idx[0]
            if i == 0:
                R_E = R[0]
            else:
                x0, y0 = R[i-1], kappa_eff_mean[i-1]
                x1, y1 = R[i], kappa_eff_mean[i]
                R_E = x0 if y1==y0 else x0 + (1 - y0)*(x1 - x0)/(y1 - y0)
            # Convert to arcsec
            from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import angular_diameter_distance_kpc
            Dd = angular_diameter_distance_kpc(zl)
            theta = (R_E / max(Dd, 1e-12)) * (180/np.pi) * 3600
            return float(theta)
        except Exception:
            return None

    # Grid search
    best = None
    for g1 in g1_grid:
        for g2 in g2_grid:
            for aa in a_grid:
                for bb in b_grid:
                    for dd in d_grid:
                        model = GeometricExponentGravity(a=aa, b=bb, d=dd,
                                                         gamma1=g1, gamma2=g2,
                                                         R_scale_kpc=args.rscale)
                        preds = []
                        targs = []
                        for cid in train_ids:
                            p = predict_thetaE(cid, model)
                            if p is None or not np.isfinite(p):
                                continue
                            preds.append(p)
                            t_obs = float(obs[obs['cluster_id']==cid]['theta_E_observed_arcsec'].iloc[0])
                            targs.append(t_obs)
                        if len(preds) == 0:
                            continue
                        mae = float(np.mean(np.abs(np.asarray(preds) - np.asarray(targs))))
                        if (best is None) or (mae < best['mae_arcsec']):
                            best = {'gamma1': float(g1), 'gamma2': float(g2), 'a': float(aa), 'b': float(bb), 'd': float(dd),
                                    'mae_arcsec': mae, 'count': len(preds)}
                            print(f"[SG] new best {best}")

    if best:
        with open(outdir / 'eval_sg.json', 'w', encoding='utf-8') as f:
            json.dump({'best': best}, f, indent=2)
        print(f"[SG] wrote {outdir/'eval_sg.json'}")
    else:
        print("[SG] no valid predictions in grid")


if __name__ == '__main__':
    main()
