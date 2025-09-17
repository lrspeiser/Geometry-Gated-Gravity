#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse, math, statistics
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from rigor.rigor.data import load_sparc
from rigor.rigor.cupy_utils import (
    xi_shell_logistic_radius,
    xi_logistic_density,
    xi_combined_radius_density,
    xi_combined_radius_density_gamma,
)

MREF = 1e10

# Ensure we can safely convert CuPy arrays to NumPy without triggering implicit conversions
try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cp = None  # type: ignore

def asnumpy(x):
    try:
        if (cp is not None) and isinstance(x, cp.ndarray):  # type: ignore[attr-defined]
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)

LABELS = {
    0: "S0", 1: "Sa", 2: "Sab", 3: "Sb", 4: "Sbc", 5: "Sc",
    6: "Scd", 7: "Sd", 8: "Sdm", 9: "Sm", 10: "Im", 11: "BCD",
}

def compactness(g) -> float:
    try:
        R_last = float(np.nanmax(g.R_kpc))
        Rd = float(g.Rd_kpc) if (g.Rd_kpc is not None) else float('nan')
        if np.isfinite(Rd) and Rd > 0:
            return float(R_last / Rd)
    except Exception:
        pass
    return float('nan')

# gating helpers (same as in breakdown_under_over)
def gate_fixed(xi, R_kpc, gate_R_kpc: float, gate_width: float):
    R = np.asarray(R_kpc)
    lnR = np.log(np.maximum(R, 1e-6))
    lnR_gate = np.log(max(gate_R_kpc, 1e-6))
    w = max(gate_width, 1e-3)
    H = 1.0/(1.0 + np.exp(-(lnR - lnR_gate)/w))
    return 1.0 + H*(xi - 1.0)

def gate_learned(xi, R_kpc, Mbar: float, lnR_gate_base: float, width_gate: float, alpha_gate_M: float):
    R = np.asarray(R_kpc)
    lnR = np.log(np.maximum(R, 1e-6))
    mass_term = 0.0
    if (Mbar is not None) and np.isfinite(Mbar) and Mbar>0:
        mass_term = np.log(max(Mbar/MREF, 1e-12))
    lnR_gate = lnR_gate_base + alpha_gate_M * mass_term
    w = max(width_gate, 1e-3)
    H = 1.0/(1.0 + np.exp(-(lnR - lnR_gate)/w))
    return 1.0 + H*(xi - 1.0)

def gate_learned_compact(xi, R_kpc, Mbar: float, comp: float, comp_base: float,
                          lnR_gate_base: float, width_gate: float, alpha_gate_M: float, alpha_gate_C: float):
    R = np.asarray(R_kpc)
    lnR = np.log(np.maximum(R, 1e-6))
    mass_term = 0.0
    if (Mbar is not None) and np.isfinite(Mbar) and Mbar>0:
        mass_term = np.log(max(Mbar/MREF, 1e-12))
    ln_comp_ratio = 0.0
    if np.isfinite(comp) and comp>0 and np.isfinite(comp_base) and comp_base>0:
        ln_comp_ratio = np.log(comp/comp_base)
    lnR_gate = lnR_gate_base + alpha_gate_M * mass_term + alpha_gate_C * ln_comp_ratio
    w = max(width_gate, 1e-3)
    H = 1.0/(1.0 + np.exp(-(lnR - lnR_gate)/w))
    return 1.0 + H*(xi - 1.0)

def parse_master_types(path_master: str) -> Dict[str, int]:
    """Parse SPARC master CSV and return mapping Galaxy -> T code (0..11).

    The MasterSheet CSV in this repo is not a simple data table; it contains a
    header section and then an embedded fixed-width table. We exploit the fact
    that the last 12 rows list the code mapping (Note (1)) and that the main
    data rows follow the header separator (lines with many commas). As a
    robust fallback, we also try to parse the separate machine-readable .mrt
    file if present.
    """
    # Prefer the MRT if present; it is structured
    mrt = path_master.replace('.csv', '.mrt')
    if os.path.exists(mrt):
        # Attempt to parse with a simple fixed-width-like split: many spaces or commas
        names: Dict[str, int] = {}
        try:
            with open(mrt, 'r', encoding='utf-8') as f:
                lines = [ln.rstrip('\n') for ln in f]
            # Data lines start after the separator and include galaxy name then T code as integer in fixed columns.
            for ln in lines:
                # skip header/notes
                if 'Byte-by-byte' in ln or 'Note (' in ln or '----' in ln or 'Title:' in ln:
                    continue
                # Heuristic split: fields are aligned with spaces; galaxy name occupies columns 1-11
                # Extract name as the first contiguous token and the next integer token as T
                tokens = [tok for tok in ln.strip().split(' ') if tok]
                if len(tokens) >= 2 and tokens[1].lstrip('+-').isdigit():
                    name = tokens[0].strip()
                    try:
                        t = int(tokens[1])
                    except Exception:
                        continue
                    if 0 <= t <= 11 and name:
                        names[name] = t
            if names:
                return names
        except Exception:
            pass
    # Fallback: try to read CSV lines and extract rows that look like data: name in col0, integer T in col1
    if not os.path.exists(path_master):
        return {}
    try:
        df = pd.read_csv(path_master, header=None, dtype=str, engine='python', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(path_master, header=None, dtype=str, engine='python', on_bad_lines='skip', sep=None)
    # Heuristic: valid data rows typically have galaxy name (string) in col0 and integer T 0..11 in col1
    s = df.iloc[:,1].astype(str).str.strip()
    mask = s.str.fullmatch(r'\d+')
    df2 = df[mask].copy()
    # Filter to plausible galaxy name tokens (reject header byte/format rows)
    # Assume names without spaces and without '[' or ']' are galaxy names in SPARC.
    df2[0] = df2[0].astype(str).str.strip()
    df2 = df2[~df2[0].str.contains('Bytes|Format|Units|Label|Note', case=False, na=False)]
    # Cast and clamp
    df2[1] = pd.to_numeric(df2[1], errors='coerce').astype('Int64')
    df2 = df2[df2[1].between(0, 11, inclusive='both')]
    type_map = {str(r[0]).strip(): int(r[1]) for r in df2[[0,1]].itertuples(index=False, name=None)}
    return type_map

def compute_per_type(ds, xi_name: str, gating: str, params: Dict[str, Any], type_map: Dict[str, int]) -> Dict[str, Any]:
    comps = [compactness(g) for g in ds.galaxies]
    comp_base = float(np.nanmedian([c for c in comps if np.isfinite(c)])) if any(np.isfinite(c) for c in comps) else 1.0

    # accumulators
    per_type: Dict[int, Dict[str, Any]] = {}
    overall_dv: List[float] = []
    per_type_dv: Dict[int, List[float]] = {}
    per_type_outer_counts: Dict[int, int] = {}
    per_type_under: Dict[int, int] = {}
    per_type_over: Dict[int, int] = {}
    per_type_gal_major_under: Dict[int, int] = {}
    per_type_gal_major_over: Dict[int, int] = {}
    per_type_gal_ties: Dict[int, int] = {}
    per_type_ngal: Dict[int, int] = {}

    total_outer = 0
    total_under = 0
    total_over = 0
    gal_major_under = 0
    gal_major_over = 0
    gal_ties = 0

    def add_type_init(t: int):
        if t not in per_type_dv:
            per_type_dv[t] = []
            per_type_outer_counts[t] = 0
            per_type_under[t] = 0
            per_type_over[t] = 0
            per_type_gal_major_under[t] = 0
            per_type_gal_major_over[t] = 0
            per_type_gal_ties[t] = 0
            per_type_ngal[t] = 0

    for g in ds.galaxies:
        R = np.asarray(g.R_kpc)
        Vbar = np.asarray(g.Vbar_kms)
        Vobs = np.asarray(g.Vobs_kms)
        Sigma = np.asarray(g.Sigma_bar) if g.Sigma_bar is not None else None
        Mbar = g.Mbar_Msun or MREF
        # xi core
        if xi_name == 'shell_logistic_radius':
            xi = asnumpy(xi_shell_logistic_radius(R, Mbar,
                                                  xi_max=params.get('xi_max_r', params.get('xi_max', 3.0)),
                                                  lnR0_base=params.get('lnR0_base', np.log(3.0)),
                                                  width=params.get('width', 0.6),
                                                  alpha_M=params.get('alpha_M', -0.2),
                                                  Mref=MREF))
        elif xi_name == 'logistic_density':
            xi = asnumpy(xi_logistic_density(R, Sigma, Mbar,
                                             xi_max=params.get('xi_max_d', params.get('xi_max', 3.0)),
                                             lnSigma_c=params.get('lnSigma_c', np.log(10.0)),
                                             width_sigma=params.get('width_sigma', 0.6),
                                             n_sigma=params.get('n_sigma', 1.0)))
        elif xi_name == 'combined_radius_density':
            xi = asnumpy(xi_combined_radius_density(R, Sigma, Mbar,
                                                    xi_cap=params.get('xi_cap', None),
                                                    xi_max_r=params.get('xi_max_r', 3.0), lnR0_base=params.get('lnR0_base', np.log(3.0)), width=params.get('width', 0.6), alpha_M=params.get('alpha_M', -0.2),
                                                    xi_max_d=params.get('xi_max_d', 3.0), lnSigma_c=params.get('lnSigma_c', np.log(10.0)), width_sigma=params.get('width_sigma', 0.6), n_sigma=params.get('n_sigma', 1.0)))
        elif xi_name == 'combined_radius_density_gamma':
            xi = asnumpy(xi_combined_radius_density_gamma(R, Sigma, Mbar,
                                                          xi_cap=params.get('xi_cap', None),
                                                          xi_max_r=params.get('xi_max_r', 3.0), lnR0_base=params.get('lnR0_base', np.log(3.0)), width=params.get('width', 0.6), alpha_M=params.get('alpha_M', -0.2),
                                                          xi_max_d=params.get('xi_max_d', 3.0), lnSigma_c=params.get('lnSigma_c', np.log(10.0)), width_sigma=params.get('width_sigma', 0.6), n_sigma=params.get('n_sigma', 1.0),
                                                          gamma=params.get('gamma', 1.0)))
        else:
            raise SystemExit(f"Unknown xi {xi_name}")
        # gating
        if gating == 'fixed':
            xi = gate_fixed(xi, R, gate_R_kpc=params.get('gate_R_kpc', 3.0), gate_width=params.get('gate_width', 0.4))
        elif gating == 'learned':
            xi = gate_learned(xi, R, Mbar,
                              lnR_gate_base=params.get('lnR_gate_base', np.log(3.0)),
                              width_gate=params.get('width_gate', 0.4),
                              alpha_gate_M=params.get('alpha_gate_M', -0.2))
        elif gating == 'learned_compact':
            xi = gate_learned_compact(xi, R, Mbar, compactness(g), comp_base,
                                      lnR_gate_base=params.get('lnR_gate_base', np.log(3.0)),
                                      width_gate=params.get('width_gate', 0.4),
                                      alpha_gate_M=params.get('alpha_gate_M', -0.2),
                                      alpha_gate_C=params.get('alpha_gate_C', -0.2))
        # prediction and residuals on outer points
        Vpred = Vbar * np.sqrt(np.clip(xi, 1.0, 1e6))
        mask = np.asarray(g.outer_mask, dtype=bool)
        if not mask.any():
            continue
        dv = Vpred[mask] - Vobs[mask]
        overall_dv.extend([float(x) for x in dv])
        total_outer += int(mask.sum())
        total_under += int(np.sum(dv < 0))
        total_over  += int(np.sum(dv > 0))
        med = float(np.median(dv))
        if med < -1e-6:
            gal_major_under += 1
        elif med > 1e-6:
            gal_major_over += 1
        else:
            gal_ties += 1
        # type accumulators
        tcode: Optional[int] = type_map.get(g.name)
        if tcode is None:
            continue  # exclude from type-level stats but kept in overall
        add_type_init(tcode)
        per_type_dv[tcode].extend([float(x) for x in dv])
        per_type_outer_counts[tcode] += int(mask.sum())
        per_type_under[tcode] += int(np.sum(dv < 0))
        per_type_over[tcode]  += int(np.sum(dv > 0))
        if med < -1e-6:
            per_type_gal_major_under[tcode] += 1
        elif med > 1e-6:
            per_type_gal_major_over[tcode] += 1
        else:
            per_type_gal_ties[tcode] += 1
        per_type_ngal[tcode] += 1

    # overall metrics (all galaxies)
    overall_rmse = float(math.sqrt(np.mean(np.square(overall_dv)))) if overall_dv else float('nan')
    overall_mean = float(np.mean(overall_dv)) if overall_dv else float('nan')
    overall = {
        'total_outer_points': total_outer,
        'under_points': total_under,
        'over_points': total_over,
        'under_pct': (100.0*total_under/total_outer) if total_outer else float('nan'),
        'over_pct': (100.0*total_over/total_outer) if total_outer else float('nan'),
        'mean_off': overall_mean,
        'rmse': overall_rmse,
        'gal_major_under': gal_major_under,
        'gal_major_over': gal_major_over,
        'gal_ties': gal_ties,
        'compactness_base': comp_base,
    }

    # per-type metrics
    per_type_metrics: Dict[str, Any] = {}
    for tcode, dvs in per_type_dv.items():
        pts = per_type_outer_counts.get(tcode, 0)
        if pts <= 0:
            continue
        under = per_type_under.get(tcode, 0)
        over  = per_type_over.get(tcode, 0)
        mean_off = float(np.mean(dvs)) if dvs else float('nan')
        rmse = float(math.sqrt(np.mean(np.square(dvs)))) if dvs else float('nan')
        per_type_metrics[str(tcode)] = {
            'label': LABELS.get(tcode, f'T{tcode}'),
            'n_galaxies': per_type_ngal.get(tcode, 0),
            'total_outer_points': pts,
            'under_points': under,
            'over_points': over,
            'under_pct': (100.0*under/pts) if pts else float('nan'),
            'over_pct': (100.0*over/pts) if pts else float('nan'),
            'mean_off': mean_off,
            'rmse': rmse,
            'gal_major_under': per_type_gal_major_under.get(tcode, 0),
            'gal_major_over': per_type_gal_major_over.get(tcode, 0),
            'gal_ties': per_type_gal_ties.get(tcode, 0),
        }

    # flag problematic types
    flagged: List[Dict[str, Any]] = []
    rmse_thresh = overall_rmse * 1.25 if math.isfinite(overall_rmse) else float('inf')
    for tcode_str, m in per_type_metrics.items():
        reasons = []
        if m['under_pct'] >= 65.0 or m['over_pct'] >= 65.0:
            reasons.append('under/over imbalance ≥ 65%')
        if abs(m.get('mean_off', 0.0)) >= 10.0:
            reasons.append('|mean offset| ≥ 10 km/s')
        if math.isfinite(rmse_thresh) and (m.get('rmse', 0.0) > rmse_thresh):
            reasons.append(f'rmse > {rmse_thresh:.1f}')
        small = (m.get('n_galaxies', 0) < 3)
        if small:
            reasons.append('small-sample (n_gal < 3)')
        if reasons:
            flagged.append({'type_code': int(tcode_str), 'label': m.get('label'), 'reasons': reasons})

    # simulate excluding flagged types
    flagged_codes = {f['type_code'] for f in flagged}
    excl_dv: List[float] = []
    excl_under = 0
    excl_over = 0
    excl_total = 0
    for tcode, dvs in per_type_dv.items():
        if tcode in flagged_codes:
            continue
        pts = per_type_outer_counts.get(tcode, 0)
        excl_total += pts
        excl_under += per_type_under.get(tcode, 0)
        excl_over  += per_type_over.get(tcode, 0)
        excl_dv.extend(dvs)
    excl_metrics = {
        'total_outer_points': excl_total,
        'under_points': excl_under,
        'over_points': excl_over,
        'under_pct': (100.0*excl_under/excl_total) if excl_total else float('nan'),
        'over_pct': (100.0*excl_over/excl_total) if excl_total else float('nan'),
        'mean_off': float(np.mean(excl_dv)) if excl_dv else float('nan'),
        'rmse': float(math.sqrt(np.mean(np.square(excl_dv)))) if excl_dv else float('nan'),
    }

    return {
        'overall': overall,
        'per_type': per_type_metrics,
        'flagged_types': flagged,
        'exclude_flagged': excl_metrics,
    }

def main():
    ap = argparse.ArgumentParser(description='Breakdown under/over by Hubble type and simulate excluding problematic types.')
    ap.add_argument('--outdir', default='out/cupy_nocap', help='Directory with leaderboard.csv and best_params.json under top variant')
    ap.add_argument('--parquet', default='data/sparc_rotmod_ltg.parquet')
    ap.add_argument('--master', default='data/Rotmod_LTG/MasterSheet_SPARC.csv')
    ap.add_argument('--save-dir', default=None, help='Where to write by-type summaries (JSON/MD/CSV)')
    ap.add_argument('--save-prefix', default='by_type_summary')
    args = ap.parse_args()

    lb_path = os.path.join(args.outdir, 'leaderboard.csv')
    lb = pd.read_csv(lb_path)
    top = lb.sort_values('avg_mean_off').iloc[0]
    xi = str(top['xi'])
    gating = str(top['gating'])
    tag = f"xi_{xi}__gate_{gating}"
    params_path = os.path.join(args.outdir, tag, 'best_params.json')
    with open(params_path, 'r') as f:
        best = json.load(f)['params']

    ds = load_sparc(args.parquet, args.master)
    type_map = parse_master_types(args.master)
    res = compute_per_type(ds, xi, gating, best, type_map)

    print(f"Top variant: xi={xi}, gate={gating}")
    print(json.dumps(res['overall'], indent=2))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        base = os.path.join(args.save_dir, args.save_prefix)
        combined = {
            'top_variant': {'xi': xi, 'gating': gating},
            **res,
        }
        with open(base + '.json', 'w', encoding='utf-8') as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        # CSV per type
        rows = []
        for tcode_str, m in res['per_type'].items():
            rows.append({
                'type_code': int(tcode_str), 'label': m['label'],
                'n_galaxies': m['n_galaxies'], 'total_outer_points': m['total_outer_points'],
                'under_points': m['under_points'], 'over_points': m['over_points'],
                'under_pct': round(float(m['under_pct']), 6), 'over_pct': round(float(m['over_pct']), 6),
                'mean_off': round(float(m['mean_off']), 6), 'rmse': round(float(m['rmse']), 6),
                'gal_major_under': m['gal_major_under'], 'gal_major_over': m['gal_major_over'], 'gal_ties': m['gal_ties'],
            })
        df = pd.DataFrame(rows).sort_values(['type_code'])
        df.to_csv(base + '.csv', index=False)
        # Markdown report
        md_lines = []
        md_lines.append('# Under/Over by Hubble Type')
        md_lines.append('')
        md_lines.append(f"Top variant: xi={xi}, gate={gating}")
        md_lines.append('')
        ov = res['overall']
        md_lines.append('Overall (all types included):')
        md_lines.append(f"- Total outer points: {ov['total_outer_points']}")
        md_lines.append(f"- Under: {ov['under_points']} ({ov['under_pct']:.2f}%)  Over: {ov['over_points']} ({ov['over_pct']:.2f}%)")
        md_lines.append(f"- Mean offset: {ov['mean_off']:.2f} km/s  RMSE: {ov['rmse']:.2f} km/s")
        md_lines.append('')
        md_lines.append('Per type:')
        md_lines.append('')
        md_lines.append('| Type | Label | n_gal | outer_pts | under% | over% | mean_off | rmse | flags |')
        md_lines.append('|------|-------|-------|-----------|--------|-------|----------|------|-------|')
        flagged_by_code = {f['type_code']: ', '.join(f['reasons']) for f in res['flagged_types']}
        for tcode_str, m in sorted(res['per_type'].items(), key=lambda kv: int(kv[0])):
            tcode = int(tcode_str)
            flags = flagged_by_code.get(tcode, '')
            md_lines.append(
                f"| {tcode} | {m['label']} | {m['n_galaxies']} | {m['total_outer_points']} | "
                f"{m['under_pct']:.1f}% | {m['over_pct']:.1f}% | {m['mean_off']:.1f} | {m['rmse']:.1f} | {flags} |"
            )
        md_lines.append('')
        ex = res['exclude_flagged']
        md_lines.append('If we exclude flagged types:')
        md_lines.append(f"- Total outer points: {ex['total_outer_points']}")
        md_lines.append(f"- Under: {ex['under_points']} ({ex['under_pct']:.2f}%)  Over: {ex['over_points']} ({ex['over_pct']:.2f}%)")
        md_lines.append(f"- Mean offset: {ex['mean_off']:.2f} km/s  RMSE: {ex['rmse']:.2f} km/s")
        md_lines.append('')
        md_lines.append(f"Data sources: {os.path.join(args.outdir, 'leaderboard.csv')} and {os.path.join(args.outdir, tag, 'best_params.json')} ; types from {args.master}")
        with open(base + '.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        print(f"Saved: {base}.json/.csv/.md")

if __name__ == '__main__':
    main()
