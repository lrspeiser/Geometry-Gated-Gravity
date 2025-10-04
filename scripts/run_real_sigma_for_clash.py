#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Ensure project root on sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse real-Σ pipeline
from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import compute_cluster

# CLASH 25 clusters: mapping to local data/clusters/<name> and lens redshifts
CLASH_CLUSTERS: List[Tuple[str, str, float]] = [
    ('a1423', 'ABELL_1423', 0.213),
    ('a209', 'ABELL_0209', 0.206),
    ('a2261', 'ABELL_2261', 0.224),
    ('a383', 'ABELL_0383', 0.187),
    ('a611', 'ABELL_0611', 0.288),
    ('clj1226', 'CLJ1226', 0.890),
    ('macs0329', 'MACSJ0329', 0.450),
    ('macs0416', 'MACSJ0416', 0.396),
    ('macs0429', 'MACSJ0429', 0.399),
    ('macs0647', 'MACSJ0647', 0.584),
    ('macs0717', 'MACSJ0717', 0.548),
    ('macs0744', 'MACSJ0744', 0.686),
    ('macs1115', 'MACSJ1115', 0.352),
    ('macs1149', 'MACSJ1149', 0.544),
    ('macs1206', 'MACSJ1206', 0.440),
    ('macs1311', 'MACSJ1311', 0.494),
    ('macs1423', 'MACSJ1423', 0.545),
    ('macs1720', 'MACSJ1720', 0.391),
    ('macs1931', 'MACSJ1931', 0.352),
    ('macs2129', 'MACSJ2129', 0.570),
    ('ms2137', 'MS2137', 0.313),
    ('rxj1347', 'RXJ1347', 0.451),
    ('rxj1532', 'RXJ1532', 0.345),
    ('rxj2129', 'RXJ2129', 0.234),
    ('rxj2248', 'RXJ2248', 0.348),
]

# Default source redshift; adjust as needed per cluster if you have a table
DEFAULT_ZS = 2.0

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / 'out' / 'cluster_lensing_real'
PRED_CSV = ROOT / 'data' / 'clash' / 'processed' / 'einstein_radii_baryon.csv'
PRED_JSON = ROOT / 'data' / 'clash' / 'processed' / 'summaries_realSigma.json'


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    preds: List[Dict] = []
    for cid, local_name, z_l in CLASH_CLUSTERS:
        outdir = OUT_ROOT / cid
        try:
            s = compute_cluster(local_name, z_l, DEFAULT_ZS, outdir, beta=0.0, phi0_km2s2=1.0e4, generate_plots=False)
            theta = s.get('Einstein_radius_arcsec_realSigma')
            preds.append({
                'cluster_id': cid,
                'cluster_label': local_name,
                'z_lens': z_l,
                'z_source': DEFAULT_ZS,
                'theta_E_pred_arcsec': None if theta is None else float(theta)
            })
            print(f"[real-Σ] {cid} -> θ_E={theta} arcsec")
        except Exception as e:
            preds.append({
                'cluster_id': cid,
                'cluster_label': local_name,
                'z_lens': z_l,
                'z_source': DEFAULT_ZS,
                'theta_E_pred_arcsec': None,
                'error': str(e)
            })
            print(f"[real-Σ] {cid} FAILED: {e}")
    # Write predictions CSV
    import pandas as pd
    PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds)[['cluster_id','cluster_label','z_lens','z_source','theta_E_pred_arcsec']].to_csv(PRED_CSV, index=False)
    with open(PRED_JSON, 'w', encoding='utf-8') as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote predictions: {PRED_CSV}")


if __name__ == '__main__':
    main()