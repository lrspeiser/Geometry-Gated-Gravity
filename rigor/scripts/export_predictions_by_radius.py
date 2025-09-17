#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse
from typing import Dict, Any
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

# Optional CuPy import to safely convert to NumPy
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None  # type: ignore

def asnumpy(x):
    try:
        if (cp is not None) and isinstance(x, cp.ndarray):  # type: ignore[attr-defined]
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)

# Gating helpers (mirrored from breakdown_under_over.py)
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

def compute_xi(R, Sigma, Mbar, xi_name: str, gating: str, params: Dict[str, Any], comp: float, comp_base: float):
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
        xi = gate_learned_compact(xi, R, Mbar, comp, comp_base,
                                  lnR_gate_base=params.get('lnR_gate_base', np.log(3.0)),
                                  width_gate=params.get('width_gate', 0.4),
                                  alpha_gate_M=params.get('alpha_gate_M', -0.2),
                                  alpha_gate_C=params.get('alpha_gate_C', -0.2))
    return np.asarray(xi)

def compactness(g) -> float:
    try:
        R_last = float(np.nanmax(g.R_kpc))
        Rd = float(g.Rd_kpc) if (g.Rd_kpc is not None) else float('nan')
        if np.isfinite(Rd) and Rd > 0:
            return float(R_last / Rd)
    except Exception:
        pass
    return float('nan')

def main():
    ap = argparse.ArgumentParser(description='Export per-radius predictions (with G_ratio) for the best variant.')
    ap.add_argument('--outdir', default='out/cupy_nocap', help='Directory with leaderboard.csv and per-variant best_params.json')
    ap.add_argument('--parquet', default='data/sparc_rotmod_ltg.parquet')
    ap.add_argument('--master', default='data/Rotmod_LTG/MasterSheet_SPARC.csv')
    ap.add_argument('--out', required=True, help='Path to write CSV (predictions_by_radius.csv)')
    args = ap.parse_args()

    # Resolve best variant from leaderboard
    lb_path = os.path.join(args.outdir, 'leaderboard.csv')
    lb = pd.read_csv(lb_path)
    top = lb.sort_values('avg_mean_off').iloc[0]
    xi = str(top['xi']); gating = str(top['gating'])
    tag = f"xi_{xi}__gate_{gating}"
    params_path = os.path.join(args.outdir, tag, 'best_params.json')
    with open(params_path, 'r') as f:
        params = json.load(f)['params']

    ds = load_sparc(args.parquet, args.master)
    # Dataset compactness baseline for learned_compact
    comps = [compactness(g) for g in ds.galaxies]
    comp_base = float(np.nanmedian([c for c in comps if np.isfinite(c)])) if any(np.isfinite(c) for c in comps) else 1.0

    rows = []
    for g in ds.galaxies:
        R = np.asarray(g.R_kpc)
        Vbar = np.asarray(g.Vbar_kms)
        Vobs = np.asarray(g.Vobs_kms)
        Sigma = np.asarray(g.Sigma_bar) if g.Sigma_bar is not None else None
        Mbar = g.Mbar_Msun or MREF
        xi_arr = compute_xi(R, Sigma, Mbar, xi, gating, params, compactness(g), comp_base)
        Vpred = Vbar * np.sqrt(np.clip(xi_arr, 1.0, 1e6))
        mask = np.asarray(g.outer_mask, dtype=bool)
        # Compute G_required = (Vobs/Vbar)^2 safely
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(Vbar>0, Vobs/np.maximum(Vbar, 1e-9), np.nan)
            G_required = ratio**2
        G_pred = xi_arr
        with np.errstate(divide='ignore', invalid='ignore'):
            G_ratio = G_required / np.maximum(G_pred, 1e-12)
        # boundary: use min R where outer mask is True as a proxy
        boundary = np.nan
        if mask.any():
            try:
                boundary = float(np.nanmin(R[mask]))
            except Exception:
                boundary = np.nan
        for i in range(len(R)):
            rows.append({
                'galaxy': g.name,
                'R_kpc': float(R[i]),
                'boundary_kpc': boundary,
                'is_outer': bool(mask[i]) if i < len(mask) else False,
                'Vobs_kms': float(Vobs[i]),
                'Vbar_kms': float(Vbar[i]),
                'Vpred_kms': float(Vpred[i]) if i < len(Vpred) else float('nan'),
                'G_required': float(G_required[i]) if i < len(G_required) else float('nan'),
                'G_pred': float(G_pred[i]) if i < len(G_pred) else float('nan'),
                'G_ratio': float(G_ratio[i]) if i < len(G_ratio) else float('nan'),
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote predictions: {args.out}  (rows={len(df)})")

if __name__ == '__main__':
    main()