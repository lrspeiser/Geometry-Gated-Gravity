#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import sys

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import compute_cluster

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
OUTCSV = ROOT / 'data' / 'clash' / 'processed' / 'einstein_radii_baryon_beta.csv'
METRICS = ROOT / 'data' / 'clash' / 'processed' / 'eval' / 'er_baryon_beta_metrics.json'
BEST = ROOT / 'data' / 'clash' / 'processed' / 'eval' / 'best_beta.json'


def mae(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))

def mape(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.mean(np.abs((b - a) / np.maximum(a, 1e-9))) * 100.0)


def main():
    obs = pd.read_csv(OBS)
    split = pd.read_csv(SPLIT)
    obs['cluster_id'] = obs['cluster_id'].str.lower()
    split['cluster_id'] = split['cluster_id'].str.lower()
    zmap = {cid: zl for cid, _, zl in CLASH}
    namemap = {cid: local for cid, local, _ in CLASH}

    # Build train/test id lists
    train_ids = split[split['set'] == 'train']['cluster_id'].tolist()
    test_ids = split[split['set'] == 'test']['cluster_id'].tolist()

    # Grid-search beta
    betas = np.linspace(0.0, 0.08, 41)  # step 0.002
    phi0 = 1.0e4
    best = None

    def predict_for(cid: str, beta: float) -> float | None:
        local = namemap[cid]
        zl = zmap[cid]
        outdir = ROOT / 'out' / 'cluster_lensing_real_beta' / cid
        try:
            s = compute_cluster(local, zl, DEFAULT_ZS, outdir, beta=beta, phi0_km2s2=phi0, generate_plots=False)
            return s.get('Einstein_radius_arcsec_realSigma')
        except Exception:
            return None

    # Fit on train
    train_obs = {row['cluster_id']: float(row['theta_E_observed_arcsec']) for _, row in obs.iterrows()}
    for b in betas:
        preds = []
        targs = []
        for cid in train_ids:
            p = predict_for(cid, b)
            if p is None or not np.isfinite(p):
                continue
            preds.append(float(p))
            targs.append(train_obs[cid])
        if len(preds) == 0:
            continue
        score = mae(targs, preds)
        if (best is None) or (score < best['mae_arcsec']):
            best = {'beta': float(b), 'mae_arcsec': float(score), 'count': len(preds)}

    if best is None:
        print('No valid beta fit (no predictions).')
        return

    # Predict for all with best beta
    rows = []
    for cid, _, _ in CLASH:
        p = predict_for(cid, best['beta'])
        rows.append({'cluster_id': cid, 'theta_E_pred_arcsec': None if p is None else float(p)})
    dfp = pd.DataFrame(rows)
    dfp.to_csv(OUTCSV, index=False)

    # Evaluate
    m = dfp.merge(obs[['cluster_id','theta_E_observed_arcsec']], on='cluster_id', how='inner') \
            .merge(split[['cluster_id','set']], on='cluster_id', how='left')
    m = m.dropna(subset=['theta_E_pred_arcsec','theta_E_observed_arcsec'])
    results = {}
    for name, d in {'overall': m, 'train': m[m['set']=='train'], 'test': m[m['set']=='test']}.items():
        if len(d) == 0:
            results[name] = {'count': 0, 'mae_arcsec': None, 'mape_percent': None}
        else:
            results[name] = {
                'count': int(len(d)),
                'mae_arcsec': mae(d['theta_E_observed_arcsec'], d['theta_E_pred_arcsec']),
                'mape_percent': mape(d['theta_E_observed_arcsec'], d['theta_E_pred_arcsec'])
            }

    METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS, 'w', encoding='utf-8') as f:
        json.dump({'best': best, 'metrics': results}, f, indent=2)
    with open(BEST, 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2)
    print('[beta-fit] best:', best)
    print('[beta-fit] metrics:', json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
