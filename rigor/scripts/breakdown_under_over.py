#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
from typing import Dict, Any

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
    """Return a NumPy array from either NumPy or CuPy input without implicit conversion.

    CuPy forbids implicit __array__ conversions; use cp.asnumpy explicitly when needed.
    """
    try:
        if (cp is not None) and isinstance(x, cp.ndarray):  # type: ignore[attr-defined]
            return cp.asnumpy(x)
    except Exception:
        # Fall back to NumPy conversion if CuPy type checks fail unexpectedly
        pass
    return np.asarray(x)

def compactness(g) -> float:
    try:
        R_last = float(np.nanmax(g.R_kpc))
        Rd = float(g.Rd_kpc) if (g.Rd_kpc is not None) else float('nan')
        if np.isfinite(Rd) and Rd > 0:
            return float(R_last / Rd)
    except Exception:
        pass
    return float('nan')


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


def compute_predictions(ds, xi_name: str, gating: str, params: Dict[str, Any]) -> Dict[str, Any]:
    # compute dataset compactness baseline
    comps = [compactness(g) for g in ds.galaxies]
    comp_base = float(np.nanmedian([c for c in comps if np.isfinite(c)])) if any(np.isfinite(c) for c in comps) else 1.0

    total_under = 0
    total_over = 0
    total_outer = 0
    gal_major_under = 0
    gal_major_over = 0
    gal_ties = 0

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
        # prediction
        Vpred = Vbar * np.sqrt(np.clip(xi, 1.0, 1e6))
        mask = np.asarray(g.outer_mask, dtype=bool)
        if not mask.any():
            continue
        dv = Vpred[mask] - Vobs[mask]
        total_under += int(np.sum(dv < 0))
        total_over  += int(np.sum(dv > 0))
        total_outer += int(mask.sum())
        med = float(np.median(dv))
        if med < -1e-6:
            gal_major_under += 1
        elif med > 1e-6:
            gal_major_over += 1
        else:
            gal_ties += 1

    return {
        'total_outer_points': total_outer,
        'under_points': total_under,
        'over_points': total_over,
        'under_pct': (100.0*total_under/total_outer) if total_outer else float('nan'),
        'over_pct': (100.0*total_over/total_outer) if total_outer else float('nan'),
        'gal_major_under': gal_major_under,
        'gal_major_over': gal_major_over,
        'gal_ties': gal_ties,
        'compactness_base': comp_base,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='out/cupy_nocap')
    ap.add_argument('--parquet', default='data/sparc_rotmod_ltg.parquet')
    ap.add_argument('--master', default='data/Rotmod_LTG/MasterSheet_SPARC.csv')
    # Optional: save human-readable and machine-readable summaries
    ap.add_argument('--save-dir', default=None, help='Directory to write summaries (JSON/MD/CSV). If omitted, only prints to stdout.')
    ap.add_argument('--save-prefix', default='over_under_summary', help='Base filename for saved summaries (without extension).')
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
    res = compute_predictions(ds, xi, gating, best)

    print(f"Top variant: xi={xi}, gate={gating}")
    print(json.dumps(res, indent=2))

    # Optionally persist summaries
    if args.save_dir:
        import csv
        from datetime import datetime
        os.makedirs(args.save_dir, exist_ok=True)
        generated_at = datetime.now().isoformat(timespec='seconds')
        combined = {
            'top_variant': {'xi': xi, 'gating': gating},
            **res,
            'generated_at': generated_at,
        }
        base = os.path.join(args.save_dir, args.save_prefix)
        # JSON (combined)
        with open(base + '.json', 'w', encoding='utf-8') as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
        # JSON (raw result only)
        with open(base + '_raw.json', 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        # Markdown
        under_pct = float(res.get('under_pct', float('nan')))
        over_pct = float(res.get('over_pct', float('nan')))
        bar_width = 30
        def bar(pct: float) -> str:
            try:
                filled = int(round(bar_width * max(0.0, min(100.0, pct)) / 100.0))
            except Exception:
                filled = 0
            empty = max(0, bar_width - filled)
            return '#' * filled + '.' * empty
        md_lines = [
            '# Under vs Over Summary',
            '',
            f'- Top variant: xi={xi}, gate={gating}',
            f"- Generated: {generated_at}",
            f"- Total outer points: {res['total_outer_points']}",
            '',
            'Counts',
            f"- Under: {res['under_points']} ({under_pct:.2f}%)",
            f"- Over: {res['over_points']} ({over_pct:.2f}%)",
            '',
            'Per-galaxy majority',
            f"- Under: {res['gal_major_under']}",
            f"- Over: {res['gal_major_over']}",
            f"- Ties: {res['gal_ties']}",
            '',
            'Visualization',
            f"Under  [{bar(under_pct)}] {under_pct:.2f}%",
            f"Over   [{bar(over_pct)}] {over_pct:.2f}%",
            '',
            f"Data source: {lb_path} and best_params.json under {os.path.join(args.outdir, tag)}.",
            '',
        ]
        with open(base + '.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
        # CSV
        header = ['xi','gating','total_outer_points','under_points','over_points','under_pct','over_pct','gal_major_under','gal_major_over','gal_ties','compactness_base']
        row = [
            xi, gating, res['total_outer_points'], res['under_points'], res['over_points'],
            round(under_pct, 6), round(over_pct, 6), res['gal_major_under'], res['gal_major_over'], res['gal_ties'], res['compactness_base']
        ]
        with open(base + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)
        print(f"Saved summaries to: {args.save_dir}")

if __name__ == '__main__':
    main()
