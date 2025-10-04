#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import compute_cluster, UNIVERSAL_PARAMS

# CLASH mapping and lens redshifts
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

DEFAULT_ZS = 2.0
OBS = ROOT / 'data' / 'clash' / 'einstein_radii_observed.csv'
SPLIT = ROOT / 'data' / 'clash' / 'train_test_split.csv'
OUTDIR = ROOT / 'data' / 'clash' / 'processed'


def mae(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))


def mape(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.mean(np.abs((b - a) / np.maximum(a, 1e-9))) * 100.0)


def sample_params(rng: np.random.Generator) -> tuple[Dict, Dict]:
    # Dispersion params (continuous sampling)
    sigma0 = rng.uniform(80.0, 200.0)
    alpha = rng.uniform(0.8, 3.0)
    e = rng.uniform(0.5, 5.0)
    b = rng.uniform(0.0, 2.0)
    d = rng.uniform(0.0, 2.0)
    a = rng.uniform(-2.0, 2.0)
    # log-uniform for scale
    log_sc = rng.uniform(np.log10(0.1), np.log10(10.0))
    scale = float(10**log_sc)
    dp = {
        'sigma0_kms': float(sigma0),
        'alpha': float(alpha),
        'e': float(e),
        'b': float(b),
        'd': float(d),
        'a': float(a),
        'scale': float(scale)
    }
    # Gate exponent/rc scaling
    p_out = rng.uniform(0.6, 1.4)
    rc_scale = rng.uniform(0.7, 1.4)
    par = {
        'p_out': float(p_out),
        'rc0_kpc': float(UNIVERSAL_PARAMS['rc0_kpc'] * rc_scale)
    }
    return dp, par


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    obs = pd.read_csv(OBS)
    split = pd.read_csv(SPLIT)
    obs['cluster_id'] = obs['cluster_id'].str.lower()
    split['cluster_id'] = split['cluster_id'].str.lower()
    zmap = {cid: zl for cid, _, zl in CLASH}
    namemap = {cid: local for cid, local, _ in CLASH}

    train_ids = split[split['set'] == 'train']['cluster_id'].tolist()
    test_ids = split[split['set'] == 'test']['cluster_id'].tolist()
    train_obs = {row['cluster_id']: float(row['theta_E_observed_arcsec']) for _, row in obs.iterrows()}

    rng = np.random.default_rng(args.seed)

    best = None

    def predict_for(cid: str, disp_params: Dict, params_override: Dict) -> float | None:
        local = namemap[cid]
        zl = zmap[cid]
        key = f"po{params_override['p_out']:.3f}_rc{params_override['rc0_kpc']:.2f}_a{disp_params['a']:.2f}_b{disp_params['b']:.2f}_d{disp_params['d']:.2f}_e{disp_params['e']:.2f}_al{disp_params['alpha']:.2f}_s0{int(disp_params['sigma0_kms'])}_sc{disp_params['scale']:.2f}"
        outdir = ROOT / 'out' / 'cluster_lensing_sigma_gate_random' / key / cid
        try:
            s = compute_cluster(local, zl, DEFAULT_ZS, outdir,
                                generate_plots=False, phi_iterations=2, phi_relax=0.5,
                                amp_mode='exp', gate_mode='sigma', disp_params=disp_params,
                                params_override=params_override)
            return s.get('Einstein_radius_arcsec_realSigma')
        except Exception:
            return None

    # Random sweep
    tried = 0
    for _ in range(int(args.samples)):
        dp, par = sample_params(rng)
        preds, targs = [], []
        for cid in train_ids:
            p = predict_for(cid, dp, par)
            if p is None or not np.isfinite(p):
                continue
            preds.append(float(p))
            targs.append(train_obs[cid])
        if len(preds) == 0:
            continue
        tried += 1
        score = mae(targs, preds)
        if (best is None) or (score < best['mae_arcsec']):
            best = {'disp_params': dp, 'params_override': par, 'mae_arcsec': float(score), 'count': len(preds)}
            print(f"[sigma-gate-rand] new best {best} (tried={tried})")

    if best is None:
        print('No valid dispersion gate configuration found.')
        return

    # Predict for all with best
    rows = []
    for cid, _, _ in CLASH:
        p = predict_for(cid, best['disp_params'], best['params_override'])
        rows.append({'cluster_id': cid, 'theta_E_pred_arcsec': None if p is None else float(p)})
    dfp = pd.DataFrame(rows)
    dfp.to_csv(OUTDIR / 'einstein_radii_sigma_gate_random.csv', index=False)

    # Evaluate
    m = dfp.merge(obs[['cluster_id','theta_E_observed_arcsec']], on='cluster_id', how='inner') \
            .merge(split[['cluster_id','set']], on='cluster_id', how='left')
    m = m.dropna(subset=['theta_E_pred_arcsec','theta_E_observed_arcsec'])
    metrics = {}
    for name, d in {'overall': m, 'train': m[m['set']=='train'], 'test': m[m['set']=='test']}.items():
        if len(d) == 0:
            metrics[name] = {'count': 0, 'mae_arcsec': None, 'mape_percent': None}
        else:
            metrics[name] = {
                'count': int(len(d)),
                'mae_arcsec': mae(d['theta_E_observed_arcsec'], d['theta_E_pred_arcsec']),
                'mape_percent': mape(d['theta_E_observed_arcsec'], d['theta_E_pred_arcsec'])
            }

    with open(OUTDIR / 'eval_sigma_gate_random.json', 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'metrics': metrics, 'samples_tried': tried}, f, indent=2)
    print('[sigma-gate-rand] best:', best)
    print('[sigma-gate-rand] metrics:', json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
