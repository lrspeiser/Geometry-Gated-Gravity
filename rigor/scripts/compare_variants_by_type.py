#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse, math
from typing import Dict, Any, List, Tuple
import pandas as pd

from rigor.scripts.breakdown_by_type import compute_per_type, parse_master_types
from rigor.rigor.data import load_sparc

def load_leaderboard(outdir: str) -> List[Dict[str, Any]]:
    js = os.path.join(outdir, 'leaderboard.json')
    csv = os.path.join(outdir, 'leaderboard.csv')
    rows: List[Dict[str, Any]] = []
    if os.path.exists(js):
        rows = json.load(open(js, 'r', encoding='utf-8'))
    elif os.path.exists(csv):
        rows = pd.read_csv(csv).to_dict(orient='records')
    else:
        raise FileNotFoundError('leaderboard.json/csv not found in ' + outdir)
    # Normalize keys
    norm = []
    for r in rows:
        norm.append({
            'xi': str(r.get('xi')),
            'gating': str(r.get('gating')),
            'avg_mean_off': float(r.get('avg_mean_off', float('nan'))),
            'avg_rmse': float(r.get('avg_rmse', float('nan'))),
            'avg_pct_close_90': float(r.get('avg_pct_close_90', float('nan'))),
            'galaxies': int(r.get('galaxies', 0)),
            'outer_points': int(r.get('outer_points', 0)),
        })
    return norm

def best_params_path(outdir: str, xi: str, gating: str) -> str:
    tag = f"xi_{xi}__gate_{gating}"
    return os.path.join(outdir, tag, 'best_params.json')

def main():
    ap = argparse.ArgumentParser(description='Compare variants by Hubble type and pick best per type.')
    ap.add_argument('--outdir', default='out/cupy_nocap')
    ap.add_argument('--parquet', default='data/sparc_rotmod_ltg.parquet')
    ap.add_argument('--master', default='data/Rotmod_LTG/MasterSheet_SPARC.csv')
    ap.add_argument('--save-dir', default=None)
    ap.add_argument('--save-prefix', default='by_type_best_variants')
    ap.add_argument('--rank-metric', default='rmse', choices=['rmse','abs_mean_off'])
    args = ap.parse_args()

    lb = load_leaderboard(args.outdir)
    if not lb:
        raise SystemExit('Empty leaderboard')
    # baseline = best (sorted by avg_mean_off, like prior)
    baseline = sorted(lb, key=lambda r: r.get('avg_mean_off', float('inf')))[0]
    base_xi, base_gate = baseline['xi'], baseline['gating']

    # Load dataset and types once
    ds = load_sparc(args.parquet, args.master)
    type_map = parse_master_types(args.master)

    # For each variant, compute per-type metrics
    per_variant_per_type: Dict[Tuple[str,str], Dict[str, Any]] = {}
    for r in lb:
        xi = r['xi']; gate = r['gating']
        params_path = best_params_path(args.outdir, xi, gate)
        if not os.path.exists(params_path):
            # skip variants without params
            continue
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)['params']
        res = compute_per_type(ds, xi, gate, params, type_map)
        per_variant_per_type[(xi, gate)] = res['per_type']

    # For each type, pick best variant by chosen metric
    winners: Dict[str, Dict[str, Any]] = {}
    rank_key = 'rmse' if args.rank_metric == 'rmse' else 'mean_off'
    for (xi, gate), per_type in per_variant_per_type.items():
        for tcode_str, m in per_type.items():
            # define score
            if args.rank_metric == 'abs_mean_off':
                score = abs(float(m.get('mean_off', float('inf'))))
            else:
                score = float(m.get('rmse', float('inf')))
            cur = winners.get(tcode_str)
            if (cur is None) or (score < cur['score']):
                winners[tcode_str] = {
                    'label': m.get('label'),
                    'xi': xi, 'gating': gate,
                    'score': score,
                    'metrics': m,
                }

    # Also capture baseline metrics per type for delta
    baseline_per_type = per_variant_per_type.get((base_xi, base_gate), {})
    # Build outputs
    by_type_rows = []
    winner_counts: Dict[Tuple[str,str], int] = {}
    for tcode_str, w in sorted(winners.items(), key=lambda kv: int(kv[0])):
        xi, gate = w['xi'], w['gating']
        winner_counts[(xi,gate)] = winner_counts.get((xi,gate), 0) + 1
        m = w['metrics']
        base = baseline_per_type.get(tcode_str)
        deltas = {}
        if base:
            deltas = {
                'rmse_delta': float(m['rmse']) - float(base['rmse']),
                'abs_mean_off_delta': abs(float(m['mean_off'])) - abs(float(base['mean_off'])),
                'under_pct_delta': float(m['under_pct']) - float(base['under_pct']),
            }
        by_type_rows.append({
            'type_code': int(tcode_str),
            'label': w.get('label'),
            'best_xi': xi, 'best_gating': gate,
            'best_rmse': float(m['rmse']),
            'best_under_pct': float(m['under_pct']),
            'best_over_pct': float(m['over_pct']),
            'best_mean_off': float(m['mean_off']),
            'baseline_xi': base_xi, 'baseline_gating': base_gate,
            'baseline_rmse': float(base['rmse']) if base else float('nan'),
            'baseline_under_pct': float(base['under_pct']) if base else float('nan'),
            'baseline_mean_off': float(base['mean_off']) if base else float('nan'),
            **deltas,
        })

    # Patterns summary
    pattern_rows = [
        {'xi': xi, 'gating': gate, 'types_won': cnt}
        for (xi,gate), cnt in sorted(winner_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    out = {
        'baseline_variant': {'xi': base_xi, 'gating': base_gate},
        'rank_metric': args.rank_metric,
        'by_type': by_type_rows,
        'winner_counts': pattern_rows,
    }

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        base = os.path.join(args.save_dir, args.save_prefix)
        with open(base + '.json', 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        # CSV
        pd.DataFrame(by_type_rows).sort_values(['type_code']).to_csv(base + '.csv', index=False)
        pd.DataFrame(pattern_rows).to_csv(base + '_patterns.csv', index=False)
        # Markdown (includes a plain-language summary section outline)
        md = []
        md.append('# Best Variant per Hubble Type')
        md.append('')
        md.append(f"Baseline (global best): xi={base_xi}, gate={base_gate}; ranking metric: {args.rank_metric}")
        md.append('')
        md.append('Per type: best variant vs baseline (Δ = best − baseline):')
        md.append('')
        md.append('| Type | Label | Best xi | Best gate | RMSE | ΔRMSE | |mean_off| Δ | Under% | ΔUnder% |')
        md.append('|------|-------|---------|-----------|------|-------|-----------|--------|----------|')
        for r in sorted(by_type_rows, key=lambda r: r['type_code']):
            d_rmse = r.get('rmse_delta', float('nan'))
            d_abs = r.get('abs_mean_off_delta', float('nan'))
            d_u = r.get('under_pct_delta', float('nan'))
            md.append(f"| {r['type_code']} | {r['label']} | {r['best_xi']} | {r['best_gating']} | {r['best_rmse']:.1f} | {d_rmse:+.1f} | {abs(r['best_mean_off']):.1f} | {d_abs:+.1f} | {r['best_under_pct']:.1f}% | {d_u:+.1f}% |")
        md.append('')
        md.append('Winners by variant:')
        for p in pattern_rows:
            md.append(f"- {p['xi']} / {p['gating']}: {p['types_won']} types")
        md.append('')
        md.append('Plain-language summary (see also assistant write-up):')
        md.append('- Which model style tends to win across types and where it differs.')
        md.append('- Whether improvements are large (e.g., >5 km/s RMSE) or minor.')
        md.append('- Whether late types (Sd/Im) show different bias than earlier spirals (Sa/Sb).')
        with open(base + '.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))
        print(f"Saved: {base}.json/.csv/.md and {base}_patterns.csv")
    else:
        print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
