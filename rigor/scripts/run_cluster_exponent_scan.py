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
# Cluster-specific pass thresholds (median |ΔT|/T)
PASS_THRESH = {
    'ABELL_0426': 0.30,
    'ABELL_1689': 0.60,
}

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
            S0: float, rc: float, g0: float,
            fnt0: float | None = None, fnt_n: float | None = None,
            r500_kpc: float | None = None, fnt_max: float | None = None) -> Tuple[float, Path]:
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
    # non-thermal options
    if fnt0 is not None:
        args += ['--fnt0', f'{fnt0}']
    if fnt_n is not None:
        args += ['--fnt_n', f'{fnt_n}']
    if r500_kpc is not None:
        args += ['--r500_kpc', f'{r500_kpc}']
    if fnt_max is not None:
        args += ['--fnt_max', f'{fnt_max}']

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
    ap.add_argument('--Rmax', type=float, default=1500.0)
    ap.add_argument('--Zmax', type=float, default=1500.0)
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
        fracs_for_cl = []
        combos_for_cl = []
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
                pass_thr = PASS_THRESH.get(cl, 0.60)
                row = {
                    'cluster': cl,
                    'gamma': g,
                    'beta': b,
                    'median_residual': frac,
                    'pass_target': (frac <= pass_thr),
                    'run_dir': str(run_dir)
                }
                rows.append(row)
                fracs_for_cl.append(frac)
                combos_for_cl.append((g, b))
                if (best_for_cluster is None) or (frac < best_for_cluster[0]):
                    best_for_cluster = (frac, g, b)
                if (best_overall is None) or (frac < best_overall[0]):
                    best_overall = (frac, cl, g, b)
        if best_for_cluster:
            print(f"[BEST] {cl}: frac={best_for_cluster[0]:.3f} at γ={best_for_cluster[1]}, β={best_for_cluster[2]}")
        # Non-thermal fallback if all A1689 runs fail target
        if cl == 'ABELL_1689' and fracs_for_cl and all(fr > PASS_THRESH['ABELL_1689'] for fr in fracs_for_cl):
            g_best, b_best = best_for_cluster[1], best_for_cluster[2]
            try:
                frac_nt, run_dir_nt = run_one(
                    cluster=cl, gamma=g_best, beta=b_best, comparator=args.comparator,
                    NR=args.NR, NZ=args.NZ, Rmax=args.Rmax, Zmax=args.Zmax,
                    S0=args.S0, rc=args.rc_kpc, g0=args.g0_kms2_per_kpc,
                    fnt0=0.2, fnt_n=0.8, r500_kpc=1000.0, fnt_max=0.3
                )
                # Append using same columns; note nonthermal in run_dir suffix
                rows.append({
                    'cluster': cl,
                    'gamma': g_best,
                    'beta': b_best,
                    'median_residual': frac_nt,
                    'pass_target': (frac_nt <= PASS_THRESH['ABELL_1689']),
                    'run_dir': str(run_dir_nt) + ' (nonthermal)'
                })
            except Exception as e:
                print(f'[WARN] Non-thermal fallback failed for {cl}: {e}')

    # Write CSV
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cluster','gamma','beta','median_residual','pass_target','run_dir'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if best_overall:
        print(f"[BEST-OVERALL] {best_overall[1]}: frac={best_overall[0]:.3f} at γ={best_overall[2]}, β={best_overall[3]}")


if __name__ == '__main__':
    main()
