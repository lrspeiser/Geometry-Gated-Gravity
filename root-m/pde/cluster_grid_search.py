# -*- coding: utf-8 -*-
"""
root-m/pde/cluster_grid_search.py
Grid search over (S0, rc_kpc) for clusters to approximate acceptable HSE errors.
Writes a JSON/CSV summary with per-cluster metrics.
"""
import json, itertools
import numpy as np
from pathlib import Path
import subprocess, sys
import argparse

CLUSTERS = ["ABELL_0426", "ABELL_1689"]

def run_one(cluster, S0, rc, Rmax, Zmax, NR, NZ):
    cmd = [sys.executable, '-u', 'root-m/pde/run_cluster_pde.py',
           '--cluster', cluster,
           '--S0', str(S0), '--rc_kpc', str(rc),
           '--Rmax', str(Rmax), '--Zmax', str(Zmax), '--NR', str(NR), '--NZ', str(NZ)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    # load metrics
    mpath = Path('root-m/out/pde_clusters')/cluster/'metrics.json'
    data = json.loads(mpath.read_text())
    return { 'cluster': cluster, 'S0': S0, 'rc_kpc': rc, 'temp_median_frac_err': data['temp_median_frac_err'] }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--S0_grid', default='1e-6,3e-6,1e-5,3e-5,1e-4')
    ap.add_argument('--rc_grid', default='10,15,20,30')
    ap.add_argument('--Rmax', type=float, default=600.0)
    ap.add_argument('--Zmax', type=float, default=600.0)
    ap.add_argument('--NR', type=int, default=64)
    ap.add_argument('--NZ', type=int, default=64)
    ap.add_argument('--out', default='root-m/out/pde_clusters/grid_search.json')
    args = ap.parse_args()

    S0s = [float(x) for x in args.S0_grid.split(',') if x.strip()]
    rcs = [float(x) for x in args.rc_grid.split(',') if x.strip()]

    rows = []
    for S0, rc in itertools.product(S0s, rcs):
        for c in CLUSTERS:
            try:
                res = run_one(c, S0, rc, args.Rmax, args.Zmax, args.NR, args.NZ)
                rows.append(res)
                print(f"{c} S0={S0:g} rc={rc:g} -> med|dT|/T={res['temp_median_frac_err']:.3g}")
            except Exception as e:
                print(f"ERROR: {c} S0={S0} rc={rc}: {e}")

    # aggregate
    best = None
    for S0, rc in itertools.product(S0s, rcs):
        subset = [r for r in rows if r['S0']==S0 and r['rc_kpc']==rc]
        if len(subset) != len(CLUSTERS):
            continue
        score = np.median([r['temp_median_frac_err'] for r in subset])
        rec = {'S0': S0, 'rc_kpc': rc, 'median_clusters_temp_frac_err': float(score), 'by_cluster': subset}
        if best is None or score < best['median_clusters_temp_frac_err']:
            best = rec
    out = {'rows': rows, 'best': best}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(best, indent=2))

if __name__ == '__main__':
    main()
