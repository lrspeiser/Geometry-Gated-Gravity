#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_binned(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {'r_kpc','vbar_kms','v_obs_kms'}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns {missing} in {path}")
    out = df[['r_kpc','vbar_kms','v_obs_kms']].copy()
    if 'v_err_kms' in df.columns:
        out['v_err_kms'] = pd.to_numeric(df['v_err_kms'], errors='coerce')
    out['r_kpc'] = pd.to_numeric(out['r_kpc'], errors='coerce')
    out['vbar_kms'] = pd.to_numeric(out['vbar_kms'], errors='coerce')
    out['v_obs_kms'] = pd.to_numeric(out['v_obs_kms'], errors='coerce')
    out = out.dropna(subset=['r_kpc','vbar_kms','v_obs_kms']).sort_values('r_kpc')
    return out


def plot_gr_only(df: pd.DataFrame, out_png: Path) -> None:
    R = df['r_kpc'].to_numpy()
    Vbar = df['vbar_kms'].to_numpy()
    Vobs = df['v_obs_kms'].to_numpy()
    Verr = df['v_err_kms'].to_numpy() if 'v_err_kms' in df.columns else None

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    if Verr is not None and np.isfinite(Verr).any():
        ax.errorbar(R, Vobs, yerr=Verr, fmt='o', ms=3.5, lw=0.8, elinewidth=0.8, capsize=2.0,
                    color='k', alpha=0.9, label='Observed (Gaia bins ±1σ)')
    else:
        ax.scatter(R, Vobs, s=26, color='k', alpha=0.8, label='Observed (Gaia bins)')
    ax.plot(R, Vbar, '-', color='C2', lw=2.0, label='GR (baryons)')

    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('v (km/s)')
    ax.set_title('Milky Way rotation curve (Gaia bins) — GR only')
    ax.set_xlim(0, float(np.nanmax(R)) if np.isfinite(R).any() else None)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)


def main():
    ap = argparse.ArgumentParser(description='Plot MW: Observed (Gaia bins) + GR (baryons) only.')
    ap.add_argument('--pred_csv', default=str(Path('out')/'mw'/'mw_predictions_by_radius_0p1_full.csv'))
    ap.add_argument('--out_png', default=str(Path('figs')/'mw_gr_only.png'))
    args = ap.parse_args()

    src = Path(args.pred_csv)
    df = load_binned(src)
    plot_gr_only(df, Path(args.out_png))

    # provenance
    prov = {
        'pred_csv_used': str(src.resolve()),
        'columns_used': {'observed':'v_obs_kms', 'observed_err':'v_err_kms (if present)', 'baryons_GR':'vbar_kms'},
        'note': 'Observed bins and vbar_kms come directly from the Gaia-binned CSV; no modeled curves are included.'
    }
    Path(args.out_png).with_suffix('.provenance.json').write_text(json.dumps(prov, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
