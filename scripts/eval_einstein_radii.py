#!/usr/bin/env python3
"""
Evaluate Einstein radius predictions against observed values.

Inputs:
- data/clash/einstein_radii_observed.csv
    columns: cluster_id, cluster_label, z_lens, theta_E_observed_arcsec, theta_E_error_arcsec, n_images, source
- data/clash/processed/einstein_radii_clash.csv
    columns: cluster_id, cluster_label, z_lens, method_detA_arcsec, method_kappaMean_arcsec, pixel_scale_arcsec, model_dir
- data/clash/train_test_split.csv
    columns: cluster_id, set (train|test)

Outputs:
- data/clash/processed/eval/er_by_cluster.csv
- data/clash/processed/eval/er_metrics.json (overall, train, test MAE/MAPE)

Note: method_detA_arcsec derives from CLASH lensing models (not independent). This
      script provides a baseline comparison pipeline; replace predictions with your
      baryon-based model outputs for a proper validation.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # .../scripts -> project root
OBS = ROOT / 'data' / 'clash' / 'einstein_radii_observed.csv'
PRED = ROOT / 'data' / 'clash' / 'processed' / 'einstein_radii_clash.csv'
SPLIT = ROOT / 'data' / 'clash' / 'train_test_split.csv'
OUTDIR = ROOT / 'data' / 'clash' / 'processed' / 'eval'

OUTDIR.mkdir(parents=True, exist_ok=True)

obs = pd.read_csv(OBS)
pred = pd.read_csv(PRED)
split = pd.read_csv(SPLIT)

# Normalize IDs
obs['cluster_id'] = obs['cluster_id'].str.lower()
pred['cluster_id'] = pred['cluster_id'].str.lower()
split['cluster_id'] = split['cluster_id'].str.lower()

# Merge
m = (
    pred[['cluster_id', 'method_detA_arcsec']]
    .rename(columns={'method_detA_arcsec': 'theta_E_pred_arcsec'})
    .merge(obs[['cluster_id', 'theta_E_observed_arcsec']], on='cluster_id', how='inner')
    .merge(split, on='cluster_id', how='left')
)

# Drop NaNs
m = m.dropna(subset=['theta_E_pred_arcsec', 'theta_E_observed_arcsec'])

# Metrics
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

# Save by-cluster table
by_cluster = m.copy()
by_cluster = by_cluster[['cluster_id', 'theta_E_observed_arcsec', 'theta_E_pred_arcsec', 'set']]
by_cluster.to_csv(OUTDIR / 'er_by_cluster.csv', index=False)

# Save metrics
with open(OUTDIR / 'er_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2)

print('[eval] metrics:', json.dumps(res, indent=2))