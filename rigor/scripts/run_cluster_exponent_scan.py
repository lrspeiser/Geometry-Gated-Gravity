# -*- coding: utf-8 -*-
"""
rigor/scripts/run_cluster_exponent_scan.py

Micro-grid over (rc_gamma, sigma_beta) for clusters using the PDE pipeline.
Defaults to the total-baryon comparator for the headline; supports gas-only ablation.
Writes a CSV summary and prints best tuples.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'root-m' / 'pde' / 'run_cluster_pde.py'
OUT_BASE = ROOT / 'outputs' / 'cluster_scan'

DEFAULT_S0 = 1.4e-4
DEFAULT_RC = 22.0
DEFAULT_G0 = 1200.0


def parse_csv_list(s: str | None) -> List[str]:
    if not s:
        return []
    return [tok.strip() for tok in s.split(',') if tok.strip()]


def parse_float_list(s: str | None) -> List[float]:
    return [float(x) for x in parse_csv_list(s)]


def run_one(cluster: str, gamma: float, beta: float, comparator: str,
            NR: int, NZ: int, Rmax: float, Zmax: float,
            S0: float, rc: float, g0: float) -> Tuple[float, Path]:
    """Run a single configuration and return (median_abs_frac_T, run_dir)."""
    args = [
        sys.executable, '-u', str(RUN),
        '--cluster', cluster,
        '--S0', f'{S0}', '--rc_kpc', f'{rc}', '--g0_kms2_per_kpc', f'{g0}',
        '--rc_gamma', f'{gamma}', '--sigma_beta', f'{beta}',
        '--NR', str(NR), '--NZ', str(NZ), '--Rmax', f'{Rmax}', '--Zmax', f'{Zmax}'
    ]
    # comparator selection
    if comparator == 'gas-only':
        args.append('--gN_from_gas_only')
    # total-baryon is default; no flag needed

    print('[RUN]', ' '.join(str(a) for a in args))
    env = os.environ.copy()
    env.setdefault('PYTHONHASHSEED', '0')
    subprocess.run(args, check=True, env=env)

    od = ROOT / 'root-m' / 'out' / 'pde_clusters' / cluster
    mpath = od / 'metrics.json'
    if not mpath.exists():
        raise RuntimeError(f'metrics.json not found for {cluster} at {mpath}')
    with open(mpath, 'r') as f:
        mj = json.load(f)
    frac = float(mj.get('temp_median_frac_err', float('nan')))
    return frac, od


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clusters', type=str, default='ABELL_0426,ABELL_1689')
    ap.add_argument('--gammas', type=str, default='0.4,0.5,0.6')
    ap.add_argument('--betas', type=str, default='0.08,0.10,0.12')
    ap.add_argument('--comparator', type=str, default='total-baryon', choices=['total-baryon','gas-only'])
    ap.add_argument('--NR', type=int, default=128)
    ap.add_argument('--NZ', type=int, default=128)
    ap.add_argument('--Rmax', type=float, default=1200.0)
    ap.add_argument('--Zmax', type=float, default=1200.0)
    ap.add_argument('--S0', type=float, default=DEFAULT_S0)
    ap.add_argument('--rc_kpc', type=float, default=DEFAULT_RC)
    ap.add_argument('--g0_kms2_per_kpc', type=float, default=DEFAULT_G0)
    ap.add_argument('--out_csv', type=str, default=str(OUT_BASE / 'scan_summary.csv'))
    args = ap.parse_args()

    clusters = parse_csv_list(args.clusters)
    gammas = parse_float_list(args.gammas)
    betas = parse_float_list(args.betas)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)

    rows = []
    best_overall = None  # (frac, cluster, gamma, beta)

    for cl in clusters:
        best_for_cluster = None
        for g in gammas:
            for b in betas:
                try:
                    frac, run_dir = run_one(
                        cluster=cl, gamma=g, beta=b, comparator=args.comparator,
                        NR=args.NR, NZ=args.NZ, Rmax=args.Rmax, Zmax=args.Zmax,
                        S0=args.S0, rc=args.rc_kpc, g0=args.g0_kms2_per_kpc
                    )
                except Exception as e:
                    print(f'[ERR] {cl} γ={g} β={b}: {e}')
                    continue
                rows.append({
                    'cluster': cl,
                    'gamma': g,
                    'beta': b,
                    'comparator': args.comparator,
                    'median_abs_frac_T': frac,
                    'run_dir': str(run_dir)
                })
                if (best_for_cluster is None) or (frac < best_for_cluster[0]):
                    best_for_cluster = (frac, g, b)
                if (best_overall is None) or (frac < best_overall[0]):
                    best_overall = (frac, cl, g, b)
        if best_for_cluster:
            print(f"[BEST] {cl}: frac={best_for_cluster[0]:.3f} at γ={best_for_cluster[1]}, β={best_for_cluster[2]}")

    # Write CSV
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cluster','gamma','beta','comparator','median_abs_frac_T','run_dir'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if best_overall:
        print(f"[BEST-OVERALL] {best_overall[1]}: frac={best_overall[0]:.3f} at γ={best_overall[2]}, β={best_overall[3]}")


if __name__ == '__main__':
    main()
