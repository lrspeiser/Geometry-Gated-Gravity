#!/usr/bin/env python3
"""
Optimize shell-model parameters over SPARC galaxies.

This script sweeps shell parameters for src/scripts/sparc_predict.py --model shell
and evaluates accuracy across all galaxies.

Metrics (computed on outer-region points only):
- point_mean_close:       mean(percent_close) across all outer points
- point_median_close:     median(percent_close) across all outer points
- point_frac_ge90:        fraction of outer points with percent_close >= 90
- gal_mean_of_medians:    mean of per-galaxy median(percent_close)
- gal_median_of_medians:  median of per-galaxy median(percent_close)
- gal_frac_med_ge90:      fraction of galaxies with median(percent_close) >= 90
- point_avg_gr_off:       average GR_Percent_Off across outer points
- point_avg_model_off:    average Model_Percent_Off across outer points
- point_improve_off:      avg (GR_Percent_Off - Model_Percent_Off) across outer points
- point_frac_better_than_gr: fraction of outer points where Model_Percent_Off < GR_Percent_Off

Usage (from repo root):
  python src/scripts/optimize_shell.py \
    --parquet data/sparc_rotmod_ltg.parquet \
    --master-csv data/Rotmod_LTG/MasterSheet_SPARC.csv \
    --out-base data/opt_shell

Notes:
- Writes summary CSV and JSON under out-base.
- Each parameter combo writes its own output-dir under out-base and calls sparc_predict.py
  with that output dir to avoid clobbering.
- Requires pandas, numpy, pyarrow.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_predict(parquet: Path, master_csv: Path | None, mrt: Path | None,
                out_dir: Path, mass_ref: float, mass_exp: float,
                middle: float, max_enh: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    script = repo_root() / 'src/scripts/sparc_predict.py'
    cmd = [sys.executable, str(script),
           '--parquet', str(parquet),
           '--model', 'shell',
           '--shell-M-ref', str(mass_ref),
           '--shell-mass-exp', str(mass_exp),
           '--shell-middle', str(middle),
           '--shell-max', str(max_enh),
           '--output-dir', str(out_dir)]
    # Prefer MasterSheet if available, else MRT
    if master_csv and Path(master_csv).exists():
        cmd.extend(['--sparc-master-csv', str(master_csv)])
    elif mrt and Path(mrt).exists():
        cmd.extend(['--sparc-mrt', str(mrt)])
    else:
        # Fall back to masses parquet inside predictor, if present
        pass
    # Run
    subprocess.run(cmd, check=True)


def evaluate_metrics(out_dir: Path) -> dict:
    by_radius = pd.read_csv(out_dir / 'sparc_predictions_by_radius.csv')
    human_radius = pd.read_csv(out_dir / 'sparc_human_by_radius.csv')

    # Outer mask (predictions_by_radius has boolean is_outer)
    mask_outer = by_radius['is_outer'].astype(bool)

    # percent_close
    pct = pd.to_numeric(by_radius['percent_close'], errors='coerce')
    pct_outer = pct[mask_outer]

    # Point-level stats
    point_mean_close = float(np.nanmean(pct_outer)) if len(pct_outer) else float('nan')
    point_median_close = float(np.nanmedian(pct_outer)) if len(pct_outer) else float('nan')
    point_frac_ge90 = float(np.mean((pct_outer >= 90.0))) if np.isfinite(pct_outer).any() else float('nan')

    # Per-galaxy medians
    df_outer = by_radius.loc[mask_outer, ['galaxy', 'percent_close']].copy()
    df_outer['percent_close'] = pd.to_numeric(df_outer['percent_close'], errors='coerce')
    g_meds = df_outer.groupby('galaxy')['percent_close'].median()
    gal_mean_of_medians = float(np.nanmean(g_meds)) if len(g_meds) else float('nan')
    gal_median_of_medians = float(np.nanmedian(g_meds)) if len(g_meds) else float('nan')
    gal_frac_med_ge90 = float(np.mean((g_meds >= 90.0))) if len(g_meds) else float('nan')

    # Off metrics from human_by_radius
    hr = human_radius.copy()
    hr['In_Outer_Region'] = hr['In_Outer_Region'].astype(bool)
    hr_outer = hr[hr['In_Outer_Region']]
    gr_off = pd.to_numeric(hr_outer['Avg_GR_Percent_Off'] if 'Avg_GR_Percent_Off' in hr_outer.columns else hr_outer['GR_Percent_Off'], errors='coerce')
    model_off = pd.to_numeric(hr_outer['Avg_Model_Percent_Off'] if 'Avg_Model_Percent_Off' in hr_outer.columns else hr_outer['Model_Percent_Off'], errors='coerce')
    # If the columns Avg_* are not present (they won't be in per-radius), switch to per-radius names
    if 'Avg_GR_Percent_Off' not in hr_outer.columns:
        gr_off = pd.to_numeric(hr_outer['GR_Percent_Off'], errors='coerce')
    if 'Avg_Model_Percent_Off' not in hr_outer.columns:
        model_off = pd.to_numeric(hr_outer['Model_Percent_Off'], errors='coerce')
    point_avg_gr_off = float(np.nanmean(gr_off)) if len(gr_off) else float('nan')
    point_avg_model_off = float(np.nanmean(model_off)) if len(model_off) else float('nan')
    point_improve_off = float(point_avg_gr_off - point_avg_model_off) if (math.isfinite(point_avg_gr_off) and math.isfinite(point_avg_model_off)) else float('nan')
    point_frac_better_than_gr = float(np.mean((model_off < gr_off))) if (len(model_off) and len(gr_off)) else float('nan')

    return dict(
        point_mean_close=point_mean_close,
        point_median_close=point_median_close,
        point_frac_ge90=point_frac_ge90,
        gal_mean_of_medians=gal_mean_of_medians,
        gal_median_of_medians=gal_median_of_medians,
        gal_frac_med_ge90=gal_frac_med_ge90,
        point_avg_gr_off=point_avg_gr_off,
        point_avg_model_off=point_avg_model_off,
        point_improve_off=point_improve_off,
        point_frac_better_than_gr=point_frac_better_than_gr,
    )


def parse_list(arg: str, cast=float) -> list[float]:
    vals = []
    for tok in arg.split(','):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(cast(tok))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', type=Path, default=(repo_root() / 'data/sparc_rotmod_ltg.parquet'))
    ap.add_argument('--master-csv', type=Path, default=(repo_root() / 'data/Rotmod_LTG/MasterSheet_SPARC.csv'))
    ap.add_argument('--mrt', type=Path, default=(repo_root() / 'data/SPARC_Lelli2016c.mrt'))
    ap.add_argument('--out-base', type=Path, default=(repo_root() / 'data/opt_shell'))
    ap.add_argument('--mass-ref', type=float, default=1e10)
    ap.add_argument('--mass-exp-list', type=str, default='-0.5,-0.35,-0.2')
    ap.add_argument('--middle-list', type=str, default='1.5,2.0,2.5')
    ap.add_argument('--max-list', type=str, default='3.5,4.5,5.5')
    ap.add_argument('--refine', action='store_true', help='Perform a local refine pass around the best combo')
    args = ap.parse_args()

    out_base: Path = args.out_base
    out_base.mkdir(parents=True, exist_ok=True)

    mass_exp_vals = parse_list(args.mass_exp_list, float)
    middle_vals = parse_list(args.middle_list, float)
    max_vals = parse_list(args.max_list, float)

    rows = []
    tried = set()
    def key(me, mi, mx):
        return (round(float(me), 6), round(float(mi), 6), round(float(mx), 6))

    # Grid search
    combos = list(itertools.product(mass_exp_vals, middle_vals, max_vals))
    print(f"Running grid: {len(combos)} combos")
    for i, (me, md, mx) in enumerate(combos, 1):
        k = key(me, md, mx)
        if k in tried:
            continue
        tried.add(k)
        run_dir = out_base / f"me_{me:+.3f}__mid_{md:.2f}__max_{mx:.2f}"
        print(f"[{i}/{len(combos)}] me={me}, middle={md}, max={mx} -> {run_dir}")
        try:
            run_predict(args.parquet, args.master_csv, args.mrt, run_dir, args.mass_ref, me, md, mx)
            metrics = evaluate_metrics(run_dir)
            rows.append({
                'mass_exp': me, 'middle': md, 'max': mx, 'out_dir': str(run_dir),
                **metrics,
            })
        except subprocess.CalledProcessError as e:
            print(f"[warn] predictor failed for {k}: {e}")
        except Exception as e:
            print(f"[warn] evaluation failed for {k}: {e}")

    if not rows:
        print("No successful runs; aborting.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    csv_path = out_base / 'opt_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Wrote summary: {csv_path}")

    # Choose best by galaxy-level median of medians, tie-breaker on point_mean_close
    df_sorted = df.sort_values(
        by=['gal_median_of_medians', 'point_mean_close', 'point_frac_ge90'], ascending=[False, False, False]
    )
    best = df_sorted.iloc[0].to_dict()
    best_json = out_base / 'best_params.json'
    with best_json.open('w') as f:
        json.dump(best, f, indent=2)
    print("Best (coarse):", json.dumps(best, indent=2))

    # Optional refine around best
    if args.refine:
        me0, md0, mx0 = float(best['mass_exp']), float(best['middle']), float(best['max'])
        me_set = sorted(set([me0 - 0.10, me0 - 0.05, me0, me0 + 0.05, me0 + 0.10]))
        md_set = sorted(set([md0 - 0.50, md0 - 0.25, md0, md0 + 0.25, md0 + 0.50]))
        mx_set = sorted(set([mx0 - 1.0, mx0 - 0.5, mx0, mx0 + 0.5, mx0 + 1.0]))
        # Clamp to reasonable ranges
        def clamp_list(vals, lo, hi):
            return [float(np.clip(v, lo, hi)) for v in vals]
        me_set = clamp_list(me_set, -1.5, 0.0)
        md_set = clamp_list(md_set, 1.0, 4.0)
        mx_set = clamp_list(mx_set, 2.0, 8.0)
        combos2 = list(itertools.product(me_set, md_set, mx_set))
        print(f"Refine grid: {len(combos2)} combos around best")
        for i, (me, md, mx) in enumerate(combos2, 1):
            k = key(me, md, mx)
            if k in tried:
                continue
            tried.add(k)
            run_dir = out_base / f"refine__me_{me:+.3f}__mid_{md:.2f}__max_{mx:.2f}"
            print(f"[ref {i}/{len(combos2)}] me={me}, middle={md}, max={mx} -> {run_dir}")
            try:
                run_predict(args.parquet, args.master_csv, args.mrt, run_dir, args.mass_ref, me, md, mx)
                metrics = evaluate_metrics(run_dir)
                rows.append({
                    'mass_exp': me, 'middle': md, 'max': mx, 'out_dir': str(run_dir),
                    **metrics,
                })
            except subprocess.CalledProcessError as e:
                print(f"[warn] predictor failed for {k}: {e}")
            except Exception as e:
                print(f"[warn] evaluation failed for {k}: {e}")
        df2 = pd.DataFrame(rows)
        df2.to_csv(csv_path, index=False)
        df_sorted2 = df2.sort_values(
            by=['gal_median_of_medians', 'point_mean_close', 'point_frac_ge90'], ascending=[False, False, False]
        )
        best2 = df_sorted2.iloc[0].to_dict()
        with best_json.open('w') as f:
            json.dump(best2, f, indent=2)
        print("Best (refined):", json.dumps(best2, indent=2))


if __name__ == '__main__':
    main()
