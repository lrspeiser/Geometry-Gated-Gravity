#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Parse SPARC MRT for quality flags Q (1=High, 2=Medium, 3=Low)
def parse_mrt_q(mrt_path: str) -> Dict[str, int]:
    qmap: Dict[str, int] = {}
    if not os.path.exists(mrt_path):
        return qmap
    try:
        with open(mrt_path, 'r', encoding='utf-8') as f:
            for ln in f:
                s = ln.strip()
                if not s or 'Byte-by-byte' in s or 'Note (' in s or '----' in s or s.startswith('Title'):
                    continue
                toks = [t for t in s.split(' ') if t]
                if len(toks) < 5:
                    continue
                # Find last numeric token before refs (which contain letters)
                last_num = None
                for t in reversed(toks):
                    if any(c.isalpha() for c in t):
                        continue
                    try:
                        last_num = int(float(t))
                        break
                    except Exception:
                        continue
                # First token is galaxy name (aligned in columns)
                gal = toks[0]
                if gal and (last_num is not None) and (1 <= last_num <= 3):
                    qmap[gal] = int(last_num)
    except Exception:
        pass
    return qmap


def main():
    ap = argparse.ArgumentParser(description='Compute percent closeness to actuals, with and without outliers.')
    ap.add_argument('--predictions', required=True, help='Path to predictions_by_radius.csv (columns include galaxy,R_kpc,is_outer,Vobs_kms,Vpred_kms)')
    ap.add_argument('--outliers-json', required=True, help='Path to by_type_summary.json (contains outliers.top_rmse/top_abs_bias lists)')
    ap.add_argument('--mrt', default='data/SPARC_Lelli2016c.mrt', help='SPARC MRT file to map galaxy->Q quality flag')
    ap.add_argument('--save-dir', required=True, help='Directory to write closeness_summary.{json,md,csv}')
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    if 'is_outer' not in df.columns:
        raise SystemExit('Predictions file must include is_outer column')
    df['is_outer'] = df['is_outer'].astype(bool)

    # Compute percent closeness per point
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_close = 100.0 * (1.0 - (np.abs(df['Vpred_kms'] - df['Vobs_kms']) / np.maximum(np.abs(df['Vobs_kms']), 1e-9)))
    df['pct_close'] = pct_close.where(np.isfinite(pct_close))

    # Load outliers list
    summ = json.load(open(args.outliers_json, 'r', encoding='utf-8'))
    outs = summ.get('outliers', {}) or {}
    out_names: List[str] = []
    for key in ('top_rmse','top_abs_bias'):
        for r in (outs.get(key) or []):
            name = r.get('galaxy')
            if name:
                out_names.append(name)
    excluded = sorted(set(out_names))

    # Compute aggregates
    outer = df[df['is_outer']]
    outer_no_out = outer[~outer['galaxy'].isin(excluded)]

    agg_all = {
        'outer_points': int(outer.shape[0]),
        'mean_pct_close': float(outer['pct_close'].mean(skipna=True)),
        'median_pct_close': float(outer['pct_close'].median(skipna=True)),
    }
    agg_no_out = {
        'outer_points': int(outer_no_out.shape[0]),
        'mean_pct_close': float(outer_no_out['pct_close'].mean(skipna=True)),
        'median_pct_close': float(outer_no_out['pct_close'].median(skipna=True)),
    }

    # Per-galaxy summary (outer only)
    gal_rows = (
        outer_no_out.groupby('galaxy', as_index=False)
        .agg(outer_points=('pct_close','size'),
             mean_pct_close=('pct_close','mean'),
             median_pct_close=('pct_close','median'))
    )

    # Confidence from MRT Q
    qmap = parse_mrt_q(args.mrt)
    conf_map = {1:'High', 2:'Medium', 3:'Low'}
    outlier_with_conf = []
    for g in excluded:
        q = qmap.get(g)
        outlier_with_conf.append({'galaxy': g, 'Q': int(q) if q is not None else None, 'confidence': conf_map.get(q, 'Unknown')})

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    base = os.path.join(args.save_dir, 'closeness_summary')
    out_json = {
        'excluded_outliers': outlier_with_conf,
        'agg_all_outer': agg_all,
        'agg_no_outliers_outer': agg_no_out,
    }
    json.dump(out_json, open(base + '.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    gal_rows.to_csv(base + '_per_galaxy.csv', index=False)

    # Markdown write-up
    md = []
    md.append('# Percent closeness to actuals (outer region)')
    md.append('')
    md.append(f"All outer points: mean={agg_all['mean_pct_close']:.1f}%  median={agg_all['median_pct_close']:.1f}%  (N={agg_all['outer_points']})")
    md.append(f"Excluding outliers: mean={agg_no_out['mean_pct_close']:.1f}%  median={agg_no_out['median_pct_close']:.1f}%  (N={agg_no_out['outer_points']})")
    md.append('')
    md.append('Excluded outliers and data confidence (SPARC Q):')
    if outlier_with_conf:
        for r in outlier_with_conf:
            md.append(f"- {r['galaxy']}: confidence={r['confidence']}{' (Q='+str(r['Q'])+')' if r['Q'] else ''}")
    else:
        md.append('- (none)')
    md.append('')
    open(base + '.md', 'w', encoding='utf-8').write('\n'.join(md))
    print(f"Saved {base}.json/.md and {base}_per_galaxy.csv")

if __name__ == '__main__':
    main()