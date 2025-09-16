#!/usr/bin/env python3
"""
Mass vs G (per-radius) scatter colored by normalized radius.

- X: baryonic mass per galaxy (Msun)
- Y: G_required or G_ratio at each radius
- Color: normalized radius r_norm = R_kpc / boundary_kpc

Inputs:
- data/sparc_predictions_by_radius.csv (must include: galaxy, R_kpc, boundary_kpc, is_outer, G_required, G_ratio)
- data/sparc_predictions_by_galaxy.csv (must include: galaxy, M_bary_Msun)

Outputs:
- Scatter PNG (e.g., outputs/aggregates/mass_vs_G_required_by_radius.png)
- Optional hexbin PNG (median of Y over (log10 mass, r_norm) bins)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_data(pred_radius_path: Path, by_gal_path: Path) -> pd.DataFrame:
    pr = pd.read_csv(pred_radius_path)
    bg = pd.read_csv(by_gal_path)
    if 'galaxy' not in pr.columns or 'galaxy' not in bg.columns:
        raise ValueError("Both inputs must contain a 'galaxy' column")
    if 'M_bary_Msun' not in bg.columns:
        raise ValueError("by-galaxy CSV must contain 'M_bary_Msun'")
    # Merge mass into per-radius
    df = pr.merge(bg[['galaxy','M_bary_Msun']], on='galaxy', how='left')
    # Compute normalized radius where possible
    if 'boundary_kpc' in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df['r_norm'] = df['R_kpc'] / df['boundary_kpc']
    else:
        df['r_norm'] = np.nan
    return df


def scatter_colored_by_radius(df: pd.DataFrame, out_path: Path, y_field: str, outer_only: bool, logx: bool, logy: bool, vmin: float|None=None, vmax: float|None=None) -> Path:
    d = df.copy()
    # Filter rows
    req_cols = ['M_bary_Msun','r_norm','R_kpc', y_field]
    for c in req_cols:
        if c not in d.columns:
            raise ValueError(f"Missing column '{c}' for plotting")
    if outer_only and 'is_outer' in d.columns:
        d = d[d['is_outer'] == True]
    # Require finite, positive boundary for r_norm
    if 'boundary_kpc' in d.columns:
        d = d[(d['boundary_kpc'] > 0) & np.isfinite(d['boundary_kpc'])]
    # Drop non-finite
    d = d[np.isfinite(d['M_bary_Msun']) & np.isfinite(d['r_norm']) & np.isfinite(d[y_field])]
    # If logy requested, must be positive
    if logy:
        d = d[d[y_field] > 0]
    # X
    x = d['M_bary_Msun'].to_numpy(dtype=float)
    y = d[y_field].to_numpy(dtype=float)
    c = d['r_norm'].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, c=c, cmap='viridis', s=14, alpha=0.6, edgecolor='none', vmin=vmin, vmax=vmax)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('Baryonic mass (Msun)')
    ylbl = 'G required (Vobs/Vbar)^2' if y_field == 'G_required' else ('G ratio (G_required / G_pred)' if y_field == 'G_ratio' else y_field)
    ax.set_ylabel(ylbl)
    ax.grid(True, ls=':', alpha=0.5)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Normalized radius (R / boundary)')
    fig.tight_layout()
    ensure_dir(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def hexbin_heatmap(df: pd.DataFrame, out_path: Path, y_field: str, outer_only: bool, gridsize: int=40) -> Path:
    d = df.copy()
    # Prepare columns
    for c in ['M_bary_Msun','r_norm', y_field]:
        if c not in d.columns:
            raise ValueError(f"Missing column '{c}' for hexbin")
    if outer_only and 'is_outer' in d.columns:
        d = d[d['is_outer'] == True]
    # Valid r_norm
    d = d[(d['boundary_kpc'] > 0) & np.isfinite(d['boundary_kpc'])]
    d = d[np.isfinite(d['M_bary_Msun']) & np.isfinite(d['r_norm']) & np.isfinite(d[y_field])]
    # Coordinates: log10 mass on x, r_norm on y
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.log10(d['M_bary_Msun'].to_numpy(dtype=float))
    y = d['r_norm'].to_numpy(dtype=float)
    c = d[y_field].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x = x[mask]; y = y[mask]; c = c[mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(x, y, C=c, reduce_C_function=np.median, gridsize=gridsize, cmap='viridis', mincnt=5)
    ax.set_xlabel('log10 Baryonic mass (Msun)')
    ax.set_ylabel('Normalized radius (R / boundary)')
    cb = fig.colorbar(hb, ax=ax)
    ylbl = 'Median ' + ('G required' if y_field == 'G_required' else ('G ratio' if y_field == 'G_ratio' else y_field))
    cb.set_label(ylbl)
    ax.grid(True, ls=':', alpha=0.3)
    fig.tight_layout()
    ensure_dir(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description='Mass vs G (per-radius) colored by normalized radius')
    ap.add_argument('--pred-radius-path', type=Path, default=Path('data/sparc_predictions_by_radius.csv'))
    ap.add_argument('--by-galaxy-path', type=Path, default=Path('data/sparc_predictions_by_galaxy.csv'))
    ap.add_argument('--y', type=str, choices=['G_required','G_ratio'], default='G_required', help='Field to plot on Y')
    ap.add_argument('--outer-only', action='store_true', help='Filter to outer region (is_outer==True)')
    ap.add_argument('--logx', action='store_true', help='Log scale on X (mass)')
    ap.add_argument('--logy', action='store_true', help='Log scale on Y (G)')
    ap.add_argument('--out', type=Path, default=Path('outputs/aggregates/mass_vs_G_required_by_radius.png'), help='Scatter output PNG path')
    ap.add_argument('--hexbin-out', type=Path, default=None, help='Optional hexbin output PNG path')
    args = ap.parse_args()

    df = load_data(args.pred_radius_path, args.by_galaxy_path)

    # Scatter
    out_scatter = scatter_colored_by_radius(df, args.out, y_field=args.y, outer_only=args.outer_only, logx=args.logx, logy=args.logy)
    print(f"Wrote scatter: {out_scatter}")

    # Hexbin (optional)
    if args.hexbin_out is not None:
        out_hex = hexbin_heatmap(df, args.hexbin_out, y_field=args.y, outer_only=args.outer_only)
        print(f"Wrote hexbin: {out_hex}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
