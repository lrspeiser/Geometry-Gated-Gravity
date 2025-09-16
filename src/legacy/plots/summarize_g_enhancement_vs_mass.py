#!/usr/bin/env python3
"""
Build a summary table linking G enhancement and baryonic mass per galaxy.

Inputs (prefer existing project outputs):
- data/sparc_predictions_by_galaxy.csv
  Expected columns: galaxy, type, M_bary_Msun,
                    median_G_Required_Outer, mean_G_Required_Outer,
                    median_G_Ratio_Outer, mean_G_Ratio_Outer

Fallback (if by-galaxy CSV missing):
- data/sparc_predictions_by_radius.csv
  We will group by galaxy on is_outer=True and compute medians; join
  baryonic mass from data/sparc_master_clean.parquet if available.

Outputs:
- outputs/aggregates/g_enhancement_vs_mass.csv
  Columns: galaxy, type, M_bary_Msun, median_G_Required_Outer, mean_G_Required_Outer,
           median_G_Ratio_Outer, mean_G_Ratio_Outer,
           g_req_per_1e10Msun, log10_mass, log10_median_G_required
- outputs/aggregates/g_enhancement_mass_correlation.txt
  Pearson and Spearman correlation between log10(M_bary) and log10(median G_required outer)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def load_by_gal(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    needed = {
        'galaxy','type','M_bary_Msun',
        'median_G_Required_Outer','mean_G_Required_Outer',
        'median_G_Ratio_Outer','mean_G_Ratio_Outer'
    }
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return None
    return df[list(needed)].copy()


def build_from_radius(pred_radius_path: Path, master_parquet: Path | None) -> pd.DataFrame:
    if not pred_radius_path.exists():
        raise FileNotFoundError(f"Missing {pred_radius_path}")
    df = pd.read_csv(pred_radius_path)
    # Filter to outer region for G_required stats
    if 'is_outer' in df.columns:
        df_outer = df[df['is_outer'] == True].copy()
    else:
        # If no outer mask, use all (less ideal)
        df_outer = df.copy()
    # Compute medians/means per galaxy
    agg = df_outer.groupby('galaxy').agg(
        median_G_Required_Outer=('G_required','median'),
        mean_G_Required_Outer=('G_required','mean'),
        median_G_Ratio_Outer=('G_ratio','median') if 'G_ratio' in df_outer.columns else ('G_required','median'),
        mean_G_Ratio_Outer=('G_ratio','mean') if 'G_ratio' in df_outer.columns else ('G_required','mean'),
        type=('type','first') if 'type' in df_outer.columns else ('galaxy','size'),
    ).reset_index()
    # Join mass
    mass = None
    if master_parquet is not None and master_parquet.exists():
        mp = pd.read_parquet(master_parquet)
        # Try to locate mass column
        if 'M_bary' in mp.columns:
            mcol = 'M_bary'
        elif 'M_bary_Msun' in mp.columns:
            mcol = 'M_bary_Msun'
        else:
            mcol = None
        name_col = 'galaxy' if 'galaxy' in mp.columns else ('Galaxy' if 'Galaxy' in mp.columns else None)
        if name_col and mcol:
            mass = mp[[name_col, mcol]].rename(columns={name_col:'galaxy', mcol:'M_bary_Msun'}).copy()
    if mass is None and (pred_radius_path.exists()):
        # As a last resort, attempt to merge distinct mass info from predictions-by-galaxy if present alongside
        by_gal_csv = pred_radius_path.parent / 'sparc_predictions_by_galaxy.csv'
        if by_gal_csv.exists():
            mg = pd.read_csv(by_gal_csv)
            if 'galaxy' in mg.columns and 'M_bary_Msun' in mg.columns:
                mass = mg[['galaxy','M_bary_Msun']].copy()
    if mass is not None:
        agg = agg.merge(mass, on='galaxy', how='left')
    else:
        agg['M_bary_Msun'] = np.nan
    return agg


def summarize(by_gal: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = by_gal.copy()
    # Derived columns
    out['g_req_per_1e10Msun'] = out['median_G_Required_Outer'] / (out['M_bary_Msun'] / 1e10)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['log10_mass'] = np.log10(out['M_bary_Msun'])
        out['log10_median_G_required'] = np.log10(out['median_G_Required_Outer'])

    # Correlations (drop non-finite)
    mask = np.isfinite(out['log10_mass']) & np.isfinite(out['log10_median_G_required'])
    pearson = np.nan
    spearman = np.nan
    n = int(mask.sum())
    summary_lines = []
    if n >= 2:
        from scipy.stats import pearsonr, spearmanr  # if missing, we will fallback
        try:
            r_p, p_p = pearsonr(out.loc[mask,'log10_mass'], out.loc[mask,'log10_median_G_required'])
            pearson = r_p
        except Exception as e:
            summary_lines.append(f"Pearson failed: {e}")
        try:
            r_s, p_s = spearmanr(out.loc[mask,'log10_mass'], out.loc[mask,'log10_median_G_required'])
            spearman = r_s
        except Exception as e:
            summary_lines.append(f"Spearman failed: {e}")
    summary = (
        f"Pairs used (finite): {n}\n"
        f"Pearson r(log10 M, log10 median G_required outer) = {pearson}\n"
        f"Spearman r(log10 M, log10 median G_required outer) = {spearman}\n"
    )
    if summary_lines:
        summary += "\n" + "\n".join(summary_lines) + "\n"
    return out, summary


def main():
    ap = argparse.ArgumentParser(description="Summarize G enhancement vs baryonic mass per galaxy")
    ap.add_argument('--by-galaxy-csv', type=Path, default=Path('data/sparc_predictions_by_galaxy.csv'))
    ap.add_argument('--by-radius-csv', type=Path, default=Path('data/sparc_predictions_by_radius.csv'))
    ap.add_argument('--master-parquet', type=Path, default=Path('data/sparc_master_clean.parquet'))
    ap.add_argument('--out-csv', type=Path, default=Path('outputs/aggregates/g_enhancement_vs_mass.csv'))
    ap.add_argument('--out-summary', type=Path, default=Path('outputs/aggregates/g_enhancement_mass_correlation.txt'))
    args = ap.parse_args()

    by_gal = load_by_gal(args.by_galaxy_csv)
    if by_gal is None:
        by_gal = build_from_radius(args.by_radius_csv, args.master_parquet)
    # Ensure expected base columns
    for col in ['median_G_Required_Outer','M_bary_Msun']:
        if col not in by_gal.columns:
            raise ValueError(f"Missing expected column '{col}' in summary input")

    out_df, summary_txt = summarize(by_gal)

    ensure_dir(args.out_csv)
    out_df.to_csv(args.out_csv, index=False)
    ensure_dir(args.out_summary)
    args.out_summary.write_text(summary_txt)

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_summary}")
    print(summary_txt)


if __name__ == '__main__':
    main()
