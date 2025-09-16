#!/usr/bin/env python3
"""
Generate rotation-curve overlay plots for SPARC galaxies from a model run.

Inputs
- A model output directory produced by sparc_predict.py or optimize_shell.py, e.g.:
  data/opt_shell/refine__me_-0.400__mid_2.50__max_4.00
  which contains:
    - sparc_predictions_by_radius.csv
    - sparc_human_by_galaxy.csv (optional, for selection heuristics)

Outputs
- A set of PNG figures (one per galaxy) saved under --out-dir. PNGs are tracked via Git LFS
  due to .gitattributes entries.

Usage examples (from repo root):
  # Generate overlays for the best refined shell run into reports/figures/... (default selection)
  python src/scripts/plot_overlays.py \
      --in-dir data/opt_shell/refine__me_-0.400__mid_2.50__max_4.00 \
      --out-dir reports/figures/opt_shell_refine_-0.40_2.50_4.00

  # Limit to a few specific galaxies
  python src/scripts/plot_overlays.py \
      --in-dir data/opt_shell/refine__me_-0.400__mid_2.50__max_4.00 \
      --out-dir reports/figures/opt_shell_refine_-0.40_2.50_4.00 \
      --select "DDO154,IC2574,NGC2403,NGC3198,NGC6503,ESO079-G014"

Notes
- This script uses matplotlib in headless mode (Agg). Dependencies are declared in requirements.txt.
- If you add or modify web-service usage in the future, add a comment pointing to README with instructions.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend for CI/servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sanitize(name: str) -> str:
    bad = " /\\:?*\"'<>|"
    out = name
    for ch in bad:
        out = out.replace(ch, "_")
    return out


def choose_default_galaxies(df_pred: pd.DataFrame, max_n: int = 12) -> List[str]:
    """Select a diverse, representative set of galaxies.

    Strategy:
    - Prefer a mix of common types and classes present in the data
    - If sparc_human_by_galaxy.csv is not available, simply pick top-N by number of radii
    """
    # Default seed list that are commonly present in SPARC and good for communication
    seed = [
        "DDO154", "IC2574", "NGC2403", "NGC3198", "NGC6503", "ESO079-G014",
        "NGC6946", "UGC06983", "NGC0055", "NGC7793", "NGC2903", "UGC07524",
    ]
    present = [g for g in seed if g in set(df_pred['galaxy'].unique())]
    # If not enough, fill by most-sampled galaxies
    if len(present) < max_n:
        counts = df_pred.groupby('galaxy')['R_kpc'].count().sort_values(ascending=False)
        for g in counts.index:
            if g not in present:
                present.append(g)
            if len(present) >= max_n:
                break
    return present[:max_n]


def plot_one(ax, gdf: pd.DataFrame, show_gr: bool = True) -> None:
    # Ensure numeric types
    for col in [
        'R_kpc', 'Vobs_kms', 'Vbar_kms', 'Vgr_kms', 'Vpred_kms', 'G_pred', 'G_required', 'G_ratio'
    ]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

    R = gdf['R_kpc'].to_numpy()
    Vobs = gdf['Vobs_kms'].to_numpy()
    Vbar = gdf['Vbar_kms'].to_numpy() if 'Vbar_kms' in gdf.columns else None
    Vpred = gdf['Vpred_kms'].to_numpy() if 'Vpred_kms' in gdf.columns else None
    Vgr = gdf['Vgr_kms'].to_numpy() if 'Vgr_kms' in gdf.columns else None
    is_outer = gdf['is_outer'].astype(bool).to_numpy() if 'is_outer' in gdf.columns else np.zeros_like(R, dtype=bool)
    boundary = np.nan
    if 'boundary_kpc' in gdf.columns:
        try:
            boundary = float(gdf['boundary_kpc'].dropna().iloc[0])
        except Exception:
            boundary = np.nan

    # Plot observed points (outer in a different style)
    ax.scatter(R[~is_outer], Vobs[~is_outer], s=18, color='tab:blue', alpha=0.6, label='Observed (inner)')
    ax.scatter(R[is_outer], Vobs[is_outer], s=22, color='tab:blue', edgecolor='k', linewidths=0.4, label='Observed (outer)')

    # Overlay predicted curves/lines
    if Vpred is not None:
        ax.plot(R, Vpred, color='tab:red', linewidth=2.0, label='Model Vpred')
    if show_gr and (Vgr is not None):
        ax.plot(R, Vgr, color='tab:green', linestyle='--', linewidth=1.6, alpha=0.9, label='GR baseline')
    if Vbar is not None:
        ax.plot(R, Vbar, color='tab:gray', linestyle=':', linewidth=1.4, alpha=0.9, label='Baryonic Vbar')

    if np.isfinite(boundary):
        ax.axvline(boundary, color='k', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.text(boundary, ax.get_ylim()[1]*0.92, 'outer boundary', rotation=90, va='top', ha='right', fontsize=8, alpha=0.7)

    ax.set_xlabel('Radius R (kpc)')
    ax.set_ylabel('Speed (km/s)')
    ax.grid(True, alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', type=Path, required=True, help='Model output dir containing *by_radius.csv files')
    ap.add_argument('--out-dir', type=Path, required=True, help='Directory to write PNGs')
    ap.add_argument('--select', type=str, default='', help='Comma-separated galaxy names to plot')
    ap.add_argument('--max', type=int, default=12, help='Max galaxies if --select not provided')
    ap.add_argument('--dpi', type=int, default=140)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = args.in_dir / 'sparc_predictions_by_radius.csv'
    if not pred_path.exists():
        raise FileNotFoundError(f'Missing predictions file: {pred_path}')

    df_pred = pd.read_csv(pred_path)
    # Normalize column names in case of mixed caps
    df_pred.columns = [c.strip() for c in df_pred.columns]

    select_list: List[str]
    if args.select.strip():
        select_list = [x.strip() for x in args.select.split(',') if x.strip()]
    else:
        select_list = choose_default_galaxies(df_pred, max_n=args.max)

    plotted = []
    for gal in select_list:
        gmask = (df_pred['galaxy'] == gal)
        if not gmask.any():
            continue
        gdf = df_pred.loc[gmask].sort_values('R_kpc')
        gtype = str(gdf['type'].iloc[0]) if 'type' in gdf.columns else ''
        fname = f"{sanitize(gal)}.png"
        fpath = args.out_dir / fname

        fig, ax = plt.subplots(figsize=(7, 4.5), tight_layout=True)
        plot_one(ax, gdf)
        title = gal if not gtype else f"{gal}  ({gtype})"
        ax.set_title(title)
        # One legend, trimmed duplicates
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=9, framealpha=0.9)
        fig.savefig(fpath, dpi=args.dpi)
        plt.close(fig)
        plotted.append((gal, fpath))

    # Write an index markdown file to embed in paper or browse locally
    idx_path = args.out_dir / 'index.md'
    with idx_path.open('w') as f:
        f.write('# Rotation-curve overlays\n\n')
        f.write(f'Input dir: `{args.in_dir}`\n\n')
        for gal, fpath in plotted:
            f.write(f'## {gal}\n\n')
            rel = os.path.relpath(fpath, args.out_dir)
            f.write(f'![{gal}]({rel})\n\n')

    print(f"Wrote {len(plotted)} figures to {args.out_dir}")


if __name__ == '__main__':
    main()