#!/usr/bin/env python3
"""
Create a scatter plot: baryonic mass vs median G ratio (outer region).

Inputs (default):
- outputs/aggregates/g_enhancement_vs_mass.csv (produced by summarize_g_enhancement_vs_mass.py)
  Required columns: galaxy, M_bary_Msun, median_G_Ratio_Outer

Fallback:
- data/sparc_predictions_by_galaxy.csv

Output:
- outputs/aggregates/g_ratio_vs_mass.png

Options allow log scaling of axes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def load_data(summary_csv: Path, by_gal_csv: Path) -> pd.DataFrame:
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        needed = {'galaxy','M_bary_Msun','median_G_Ratio_Outer'}
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{summary_csv} missing columns: {missing}")
        return df[['galaxy','M_bary_Msun','median_G_Ratio_Outer']].copy()
    if by_gal_csv.exists():
        df = pd.read_csv(by_gal_csv)
        needed = {'galaxy','M_bary_Msun','median_G_Ratio_Outer'}
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{by_gal_csv} missing columns: {missing}")
        return df[['galaxy','M_bary_Msun','median_G_Ratio_Outer']].copy()
    raise FileNotFoundError("Could not find input CSVs.")


def scatter_plot(df: pd.DataFrame, out_path: Path, logx: bool, logy: bool) -> Path:
    # Clean
    df = df.copy()
    df = df[df['M_bary_Msun'].notna() & df['median_G_Ratio_Outer'].notna()].copy()
    df = df[df['M_bary_Msun'] > 0]
    if logy:
        df = df[df['median_G_Ratio_Outer'] > 0]

    x = df['M_bary_Msun'].to_numpy(dtype=float)
    y = df['median_G_Ratio_Outer'].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(x, y, s=22, alpha=0.75, edgecolor='none', color='tab:blue')

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    ax.set_xlabel('Baryonic mass (Msun)')
    ax.set_ylabel('Median G ratio (outer region)')
    ax.grid(True, ls=':', alpha=0.5)

    # Optional correlation annotation on log scale of mass and linear y
    with np.errstate(divide='ignore', invalid='ignore'):
        lx = np.log10(x)
        mask = np.isfinite(lx) & np.isfinite(y)
        if mask.sum() >= 3:
            try:
                from scipy.stats import pearsonr
                r, p = pearsonr(lx[mask], y[mask])
                ax.set_title(f'G_ratio vs Mass (N={mask.sum()}) — Pearson r(log10 M, y) = {r:.3f}')
            except Exception:
                pass

    fig.tight_layout()
    ensure_dir(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Scatter: baryonic mass vs median G ratio (outer region)')
    ap.add_argument('--summary-csv', type=Path, default=Path('outputs/aggregates/g_enhancement_vs_mass.csv'))
    ap.add_argument('--by-galaxy-csv', type=Path, default=Path('data/sparc_predictions_by_galaxy.csv'))
    ap.add_argument('--out', type=Path, default=Path('outputs/aggregates/g_ratio_vs_mass.png'))
    ap.add_argument('--logx', action='store_true', help='Use log scale on X (mass)')
    ap.add_argument('--logy', action='store_true', help='Use log scale on Y (G ratio)')
    args = ap.parse_args()

    df = load_data(args.summary_csv, args.by_galaxy_csv)
    out = scatter_plot(df, args.out, logx=args.logx, logy=args.logy)
    print(f'Wrote scatter: {out}')


if __name__ == '__main__':
    main()
