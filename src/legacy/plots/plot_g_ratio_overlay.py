#!/usr/bin/env python3
"""
Overlay G_ratio curves from SPARC predictions into a single plot.

- Reads data/sparc_predictions_by_radius.csv (or a provided path)
- Produces a single PNG with two subplots by default:
  - Left: raw radius (R_kpc) vs G_ratio for all galaxies
  - Right: normalized radius (R_kpc / boundary_kpc) vs G_ratio where boundary_kpc is available
- Output: outputs/aggregates/g_ratio_overlay.png (default)

Notes:
- Rows with non-finite G_ratio or missing R_kpc are skipped.
- Normalized subplot only includes rows with finite, positive boundary_kpc.
- See README (SPARC workflow) for data definitions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def load_predictions(pred_path: Path) -> pd.DataFrame:
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions-by-radius file not found: {pred_path}")
    if pred_path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if pred_path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(pred_path, sep=sep)
    else:
        df = pd.read_parquet(pred_path)
    required = {"galaxy", "R_kpc"}
    if not required.issubset(df.columns):
        raise ValueError(f"File {pred_path} must contain columns: {sorted(required)}")
    # Prefer existing G_ratio; otherwise, try to compute if G_required and G_pred available
    if "G_ratio" not in df.columns:
        if {"G_required", "G_pred"}.issubset(df.columns):
            with np.errstate(divide='ignore', invalid='ignore'):
                df["G_ratio"] = df["G_required"] / df["G_pred"]
        else:
            raise ValueError("File must contain G_ratio or both G_required and G_pred to compute it.")
    return df


def overlay_plot(
    df: pd.DataFrame,
    out_path: Path,
    x_mode: str = "both",
    alpha: float = 0.2,
    lw: float = 0.7,
    max_series: Optional[int] = None,
) -> Path:
    # Clean data
    df = df.copy()
    # Basic filters
    df = df[df["galaxy"].notna() & df["R_kpc"].notna() & df["G_ratio"].notna()].copy()
    # Drop non-finite ratios
    df = df[np.isfinite(df["G_ratio"].to_numpy())].copy()

    # Group by galaxy
    groups = list(df.groupby("galaxy"))
    total_gal = len(groups)
    if max_series is not None:
        groups = groups[:max_series]

    # Decide layout
    want_raw = x_mode in ("raw", "both")
    want_norm = x_mode in ("norm", "both")

    if want_norm and "boundary_kpc" not in df.columns:
        want_norm = False

    ncols = (1 if want_raw and not want_norm else 2 if want_raw and want_norm else 1)

    fig, axes = plt.subplots(1, ncols, figsize=(8*ncols, 6), squeeze=False)
    ax_idx = 0

    if want_raw:
        ax = axes[0, ax_idx]
        for gal, g in groups:
            g = g.sort_values("R_kpc")
            x = g["R_kpc"].to_numpy()
            y = g["G_ratio"].to_numpy()
            ax.plot(x, y, '-', lw=lw, alpha=alpha, color='tab:blue')
        ax.set_title(f"G_ratio overlay (raw radius) — {len(groups)} of {total_gal} galaxies")
        ax.set_xlabel("Radius (kpc)")
        ax.set_ylabel("G ratio (G_required / G_pred)")
        ax.grid(True, ls=":", alpha=0.5)
        ax_idx += 1

    if want_norm:
        ax = axes[0, ax_idx]
        # Keep only finite positive boundaries
        dfb = df[(df["boundary_kpc"].notna()) & np.isfinite(df["boundary_kpc"].to_numpy()) & (df["boundary_kpc"] > 0)].copy()
        groups_b = list(dfb.groupby("galaxy"))
        if max_series is not None:
            groups_b = groups_b[:max_series]
        for gal, g in groups_b:
            g = g.sort_values("R_kpc")
            x = (g["R_kpc"] / g["boundary_kpc"]).to_numpy()
            y = g["G_ratio"].to_numpy()
            ax.plot(x, y, '-', lw=lw, alpha=alpha, color='tab:orange')
        ax.set_title(f"G_ratio overlay (normalized radius) — {len(groups_b)} of {total_gal} galaxies")
        ax.set_xlabel("Normalized radius (R / boundary)")
        ax.set_ylabel("G ratio (G_required / G_pred)")
        ax.grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    ensure_dir(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Overlay G_ratio lines for all galaxies in a single plot.")
    ap.add_argument("--pred-radius-path", default="data/sparc_predictions_by_radius.csv", help="Path to predictions-by-radius CSV or Parquet with G_ratio.")
    ap.add_argument("--out", default="outputs/aggregates/g_ratio_overlay.png", help="Output PNG path.")
    ap.add_argument("--x", choices=["raw", "norm", "both"], default="both", help="Choose X-axis representation: raw R_kpc, normalized R/boundary, or both (two subplots).")
    ap.add_argument("--alpha", type=float, default=0.2, help="Line alpha (transparency).")
    ap.add_argument("--lw", type=float, default=0.7, help="Line width.")
    ap.add_argument("--max-series", type=int, default=None, help="If set, limit the number of galaxy series plotted (for quick preview).")
    args = ap.parse_args()

    pred_path = Path(args.pred_radius_path)
    out_path = Path(args.out)

    df = load_predictions(pred_path)
    out = overlay_plot(
        df,
        out_path,
        x_mode=args.x,
        alpha=args.alpha,
        lw=args.lw,
        max_series=args.max_series,
    )
    print(f"Wrote overlay plot: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
