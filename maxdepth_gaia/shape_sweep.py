#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
import pandas as pd
import numpy as np


def run_sparc(sparc_dir: str, names: list[str], m: float, eta: float, dR: float,
              mond_kind: str, mond_a0: float, out_dir: str) -> str:
    cmd = [sys.executable, "-m", "maxdepth_gaia.run_pipeline",
           "--use_source", "sparc",
           "--sparc_dir", sparc_dir,
           "--baryon_model", "mw_multi",
           "--fix_m", str(m),
           "--eta_rs", str(eta),
           "--gate_width_kpc", str(dR),
           "--mond_kind", mond_kind,
           "--mond_a0", str(mond_a0)]
    if names:
        cmd.extend(["--sparc_names", ",".join(names)])
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)
    # Copy the summary
    src_sum = os.path.join(out_dir, 'sparc_summary.csv')
    label = f"m{m:.2f}_eta{eta:.2f}_dR{dR:.2f}"
    dst = os.path.join(out_dir, 'sweep', f'sparc_summary_{label}.csv')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src_sum, dst)
    return dst


def analyze(summary_csv: str) -> dict:
    df = pd.read_csv(summary_csv)
    # Smaller BIC is better
    sw = df['sw_bic'].to_numpy()
    nf = df['nfw_bic'].to_numpy()
    mo = df['mond_bic'].to_numpy()
    bb = df['bary_bic'].to_numpy()
    wins_vs_nfw = int(np.sum(sw < nf))
    wins_vs_mond = int(np.sum(sw < mo))
    wins_vs_bary = int(np.sum(sw < bb))
    total = len(df)
    return dict(total=total,
                sum_sw=float(np.nansum(sw)), sum_nf=float(np.nansum(nf)), sum_mo=float(np.nansum(mo)), sum_bb=float(np.nansum(bb)),
                wins_vs_nfw=wins_vs_nfw, wins_vs_mond=wins_vs_mond, wins_vs_bary=wins_vs_bary)


def main():
    ap = argparse.ArgumentParser(description="Parameter sweep for saturated-well tail shape on SPARC subset.")
    ap.add_argument("--sparc_dir", default=os.path.join("data","Rotmod_LTG"))
    ap.add_argument("--names", default=None, help="Comma-separated list of SPARC galaxy names to run (defaults to first 10).")
    ap.add_argument("--m_grid", default="1.30,1.50,2.00")
    ap.add_argument("--eta_grid", default="0.16,0.20,0.30")
    ap.add_argument("--dR_grid", default="0.60,0.80,1.20")
    ap.add_argument("--mond_kind", default="simple")
    ap.add_argument("--mond_a0", type=float, default=1.2e-10)
    args = ap.parse_args()

    out_dir = os.path.join("maxdepth_gaia","outputs")
    names = [s.strip() for s in args.names.split(',')] if args.names else []
    m_grid = [float(x) for x in args.m_grid.split(',') if x]
    eta_grid = [float(x) for x in args.eta_grid.split(',') if x]
    dR_grid = [float(x) for x in args.dR_grid.split(',') if x]

    rows = []
    for m in m_grid:
        for eta in eta_grid:
            for dR in dR_grid:
                try:
                    summary_csv = run_sparc(args.sparc_dir, names, m, eta, dR, args.mond_kind, args.mond_a0, out_dir)
                    stats = analyze(summary_csv)
                    rows.append(dict(m=m, eta=eta, dR=dR, **stats, summary_csv=summary_csv))
                    print(f"[sweep] m={m:.2f}, eta={eta:.2f}, dR={dR:.2f} -> wins_vs_nfw={stats['wins_vs_nfw']}/{stats['total']}, wins_vs_mond={stats['wins_vs_mond']}/{stats['total']}")
                except subprocess.CalledProcessError as e:
                    rows.append(dict(m=m, eta=eta, dR=dR, error=str(e)))
    df = pd.DataFrame(rows)
    sweep_csv = os.path.join(out_dir, 'sweep', 'sparc_shape_sweep_results.csv')
    os.makedirs(os.path.dirname(sweep_csv), exist_ok=True)
    df.to_csv(sweep_csv, index=False)
    # Pick best by smallest sum_sw
    best = df.dropna(subset=['sum_sw']).sort_values('sum_sw').head(1)
    if not best.empty:
        b = best.iloc[0]
        print(f"Best shape by aggregate BIC: m={b['m']:.2f}, eta={b['eta']:.2f}, dR={b['dR']:.2f} (sum_sw={b['sum_sw']:.1f})")
    else:
        print("No successful runs recorded. Check logs.")


if __name__ == "__main__":
    main()