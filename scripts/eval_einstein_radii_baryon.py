#!/usr/bin/env python3
"""
Evaluate baryon-based Einstein radius predictions (real-Σ pipeline) vs observed.

Inputs:
- data/clash/einstein_radii_observed.csv
- data/clash/processed/einstein_radii_baryon.csv (from run_real_sigma_for_clash.py)
- data/clash/train_test_split.csv

Outputs:
- data/clash/processed/eval/er_baryon_by_cluster.csv
- data/clash/processed/eval/er_baryon_metrics.json
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / 'data' / 'clash' / 'einstein_radii_observed.csv'
PRED = ROOT / 'data' / 'clash' / 'processed' / 'einstein_radii_baryon.csv'
SPLIT = ROOT / 'data' / 'clash' / 'train_test_split.csv'
OUTDIR = ROOT / 'data' / 'clash' / 'processed' / 'eval'

OUTDIR.mkdir(parents=True, exist_ok=True)

obs = pd.read_csv(OBS)
pred = pd.read_csv(PRED)
split = pd.read_csv(SPLIT)

# Normalize
for df in (obs, pred, split):
    df['cluster_id'] = df['cluster_id'].str.lower()

m = (
    pred[['cluster_id','theta_E_pred_arcsec']]
    .merge(obs[['cluster_id','theta_E_observed_arcsec']], on='cluster_id', how='inner')
    .merge(split[['cluster_id','set']], on='cluster_id', how='left')
)

m = m.dropna(subset=['theta_E_pred_arcsec','theta_E_observed_arcsec'])

mae = lambda a,b: float(np.mean(np.abs(np.asarray(a,float) - np.asarray(b,float))))
mape = lambda a,b: float(np.mean(np.abs((np.asarray(b,float) - np.asarray(a,float)) / np.maximum(np.asarray(a,float),1e-9))) * 100.0)

res = {}
for name, df in {'overall': m, 'train': m[m['set']=='train'], 'test': m[m['set']=='test']}.items():
    if len(df) == 0:
        res[name] = {'count': 0, 'mae_arcsec': None, 'mape_percent': None}
    else:
        res[name] = {
            'count': int(len(df)),
            'mae_arcsec': mae(df['theta_E_observed_arcsec'], df['theta_E_pred_arcsec']),
            'mape_percent': mape(df['theta_E_observed_arcsec'], df['theta_E_pred_arcsec'])
        }

m[['cluster_id','theta_E_observed_arcsec','theta_E_pred_arcsec','set']].to_csv(OUTDIR / 'er_baryon_by_cluster.csv', index=False)
with open(OUTDIR / 'er_baryon_metrics.json','w',encoding='utf-8') as f:
    json.dump(res, f, indent=2)

print('[eval-baryon] metrics:', json.dumps(res, indent=2))