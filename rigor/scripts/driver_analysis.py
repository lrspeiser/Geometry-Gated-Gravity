#!/usr/bin/env python3
from __future__ import annotations
import os, json, argparse
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rigor.rigor.data import load_sparc
from rigor.scripts.breakdown_by_type import parse_master_types

MREF = 1e10

def compactness_of(R: np.ndarray, Rd: float | None) -> float:
    try:
        R_last = float(np.nanmax(R))
        if (Rd is not None) and np.isfinite(Rd) and Rd > 0:
            return float(R_last / Rd)
    except Exception:
        pass
    return float('nan')

def add_fit(ax, x: np.ndarray, y: np.ndarray, color='k') -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    coef = np.polyfit(x[mask], y[mask], 1)
    xx = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
    yy = coef[0]*xx + coef[1]
    ax.plot(xx, yy, color=color, alpha=0.7, lw=1.5, linestyle='--', label='lin fit')


def main():
    ap = argparse.ArgumentParser(description='Correlate per-galaxy performance with physical/kinematic features.')
    ap.add_argument('--parquet', default='data/sparc_rotmod_ltg.parquet')
    ap.add_argument('--master', default='data/Rotmod_LTG/MasterSheet_SPARC.csv')
    ap.add_argument('--predictions', required=True, help='predictions_by_radius.csv from export_predictions_by_radius')
    ap.add_argument('--closeness-per-galaxy', required=True, help='closeness_summary_per_galaxy.csv (outer, outliers excluded)')
    ap.add_argument('--save-dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    ds = load_sparc(args.parquet, args.master)
    tmap = parse_master_types(args.master)

    # Build per-galaxy features
    feats: List[Dict[str, Any]] = []
    for g in ds.galaxies:
        R = np.asarray(g.R_kpc)
        Vobs = np.asarray(g.Vobs_kms)
        mask = np.asarray(g.outer_mask, dtype=bool)
        outer_v = Vobs[mask] if mask.any() else np.asarray([], dtype=float)
        feats.append({
            'galaxy': g.name,
            'type_code': tmap.get(g.name, None),
            'R_last': float(np.nanmax(R)) if R.size else float('nan'),
            'Rd_kpc': float(g.Rd_kpc) if g.Rd_kpc is not None else float('nan'),
            'compactness': compactness_of(R, g.Rd_kpc),
            'Mbar_Msun': float(g.Mbar_Msun) if (g.Mbar_Msun is not None) else float('nan'),
            'outer_points': int(mask.sum()),
            'outer_avg_vobs': float(np.nanmean(outer_v)) if outer_v.size else float('nan'),
            'outer_max_vobs': float(np.nanmax(outer_v)) if outer_v.size else float('nan'),
            'overall_avg_vobs': float(np.nanmean(Vobs)) if Vobs.size else float('nan'),
        })
    df_feat = pd.DataFrame(feats)

    # Load predictions_by_radius to compute RMSE/bias per galaxy (outer)
    pr = pd.read_csv(args.predictions)
    pr['is_outer'] = pr['is_outer'].astype(bool)
    grp = pr[pr['is_outer']].groupby('galaxy', as_index=False)
    per_gal_pred = grp.apply(lambda d: pd.Series({
        'rmse_outer': float(np.sqrt(np.nanmean((d['Vpred_kms']-d['Vobs_kms'])**2))),
        'bias_outer': float(np.nanmean(d['Vpred_kms']-d['Vobs_kms'])),
        'under_pct_outer': float(100.0*np.mean((d['Vpred_kms']-d['Vobs_kms'])<0.0)),
    }))

    # Load closeness summary per galaxy (outer, outliers excluded)
    clos = pd.read_csv(args.closeness_per_galaxy)

    # Merge
    # Avoid overlapping column names by renaming before merge
    per_gal_pred = per_gal_pred.rename(columns={'outer_points':'outer_points_pred'})
    clos = clos.rename(columns={'outer_points':'outer_points_scored'})
    df = df_feat.merge(per_gal_pred, on='galaxy', how='left').merge(clos, on='galaxy', how='left')

    # Correlations (Pearson & Spearman) for targets vs features
    targets = ['mean_pct_close', 'median_pct_close', 'rmse_outer', 'bias_outer']
    features = ['outer_avg_vobs','outer_max_vobs','overall_avg_vobs','R_last','Rd_kpc','compactness','Mbar_Msun','type_code']
    corr = {t: {'pearson': {}, 'spearman': {}} for t in targets}
    for t in targets:
        for f in features:
            x = df[f]
            y = df[t]
            try:
                corr[t]['pearson'][f] = float(pd.concat([x,y], axis=1).corr(method='pearson').iloc[0,1])
            except Exception:
                corr[t]['pearson'][f] = float('nan')
            try:
                corr[t]['spearman'][f] = float(pd.concat([x,y], axis=1).corr(method='spearman').iloc[0,1])
            except Exception:
                corr[t]['spearman'][f] = float('nan')

    # Save tables
    base = os.path.join(args.save_dir, 'driver_analysis')
    df.to_csv(base + '_per_galaxy.csv', index=False)
    json.dump({'correlations': corr, 'n_galaxies': int(df.shape[0])}, open(base + '.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)

    # Markdown summary
    def top_corrs(metric: str, method: str='pearson', k: int=5) -> List[str]:
        items = [(f, corr[metric][method][f]) for f in features]
        items = [it for it in items if np.isfinite(it[1])]
        items.sort(key=lambda kv: -abs(kv[1]))
        return [f"{name}: {val:+.2f}" for name, val in items[:k]]

    md = []
    md.append('# Driver analysis (features vs performance)')
    md.append('')
    for t in targets:
        md.append(f'## {t}')
        md.append('- Top Pearson correlations: ' + ', '.join(top_corrs(t, 'pearson')))
        md.append('- Top Spearman correlations: ' + ', '.join(top_corrs(t, 'spearman')))
        md.append('')

    open(base + '.md','w',encoding='utf-8').write('\n'.join(md))

    # Plots
    plot_dir = os.path.join(args.save_dir, 'driver_plots')
    os.makedirs(plot_dir, exist_ok=True)

    def scatter_plot(xcol: str, ycol: str, fname: str, xlabel: str=None, ylabel: str=None):
        fig, ax = plt.subplots(figsize=(6.5,4.5), tight_layout=True)
        x = df[xcol].to_numpy()
        y = df[ycol].to_numpy()
        ax.scatter(x, y, s=20, color='tab:blue', alpha=0.6)
        add_fit(ax, x, y)
        ax.set_xlabel(xlabel or xcol)
        ax.set_ylabel(ylabel or ycol)
        ax.grid(True, alpha=0.3)
        outp = os.path.join(plot_dir, fname)
        fig.savefig(outp, dpi=150)
        plt.close(fig)
        return outp

    scatter_plot('outer_avg_vobs','mean_pct_close','mean_pct_close_vs_outer_avg_vobs.png','Outer avg Vobs (km/s)','Mean % closeness (outer)')
    scatter_plot('outer_avg_vobs','rmse_outer','rmse_vs_outer_avg_vobs.png','Outer avg Vobs (km/s)','RMSE (outer)')
    scatter_plot('compactness','mean_pct_close','mean_pct_close_vs_compactness.png','Compactness (R_last/Rd)','Mean % closeness (outer)')
    scatter_plot('R_last','rmse_outer','rmse_vs_R_last.png','R_last (kpc)','RMSE (outer)')
    print(f"Saved driver analysis to {args.save_dir}")

if __name__ == '__main__':
    main()