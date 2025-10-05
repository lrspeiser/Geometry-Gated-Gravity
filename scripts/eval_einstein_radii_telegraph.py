#!/usr/bin/env python3
"""
Evaluate telegraph operator θE predictions vs observed.

Inputs:
- data/clash/einstein_radii_observed.csv
- data/clash/processed/telegraph/einstein_radii_telegraph.csv
- data/clash/train_test_split.csv

Outputs:
- data/clash/processed/eval/er_telegraph_by_cluster.csv
- data/clash/processed/eval/er_telegraph_metrics.json
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / 'data' / 'clash' / 'einstein_radii_observed.csv'
PRED = ROOT / 'data' / 'clash' / 'processed' / 'telegraph' / 'einstein_radii_telegraph.csv'
SPLIT = ROOT / 'data' / 'clash' / 'train_test_split.csv'
OUTDIR = ROOT / 'data' / 'clash' / 'processed' / 'eval'

OUTDIR.mkdir(parents=True, exist_ok=True)

obs = pd.read_csv(OBS)
pred = pd.read_csv(PRED)
split = pd.read_csv(SPLIT)

obs['cluster_id'] = obs['cluster_id'].str.lower()
pred['cluster_id'] = pred['cluster_id'].str.lower()
split['cluster_id'] = split['cluster_id'].str.lower()

m = (
    pred[['cluster_id', 'theta_E_eff_arcsec']]
    .rename(columns={'theta_E_eff_arcsec': 'theta_E_pred_arcsec'})
    .merge(obs[['cluster_id', 'theta_E_observed_arcsec']], on='cluster_id', how='inner')
    .merge(split, on='cluster_id', how='left')
)

m = m.dropna(subset=['theta_E_pred_arcsec', 'theta_E_observed_arcsec'])

def mae(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))

def mape(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    denom = np.maximum(a, 1e-9)
    return float(np.mean(np.abs((b - a) / denom)) * 100.0)

res = {}
for name, df in {
    'overall': m,
    'train': m[m['set'] == 'train'],
    'test': m[m['set'] == 'test']
}.items():
    if len(df) == 0:
        res[name] = {'count': 0, 'mae_arcsec': None, 'mape_percent': None}
    else:
        res[name] = {
            'count': int(len(df)),
            'mae_arcsec': mae(df['theta_E_observed_arcsec'], df['theta_E_pred_arcsec']),
            'mape_percent': mape(df['theta_E_observed_arcsec'], df['theta_E_pred_arcsec'])
        }

by_cluster = m[['cluster_id', 'theta_E_observed_arcsec', 'theta_E_pred_arcsec', 'set']]
by_cluster.to_csv(OUTDIR / 'er_telegraph_by_cluster.csv', index=False)

with open(OUTDIR / 'er_telegraph_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2)

print('[eval-telegraph] metrics:', json.dumps(res, indent=2))
