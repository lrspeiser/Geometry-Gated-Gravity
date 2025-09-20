#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys as _sys
# Make rigor importable
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigor.baselines import mond_simple, nfw_velocity_kms


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns)
    def pick(*names):
        for n in names:
            if n in cols: return n
        return None
    m = {
        'r': pick('r_kpc','R_kpc','R'),
        'vbar': pick('vbar_kms','Vbar_kms','Vbar'),
        'vobs': pick('v_obs_kms','Vobs_kms','V_obs_kms','Vobs'),
        'vlt': pick('v_LogTail_kms','v_LogTail_preview_kms'),
    }
    missing = [k for k in ('r','vbar','vobs') if m[k] is None]
    if missing:
        raise KeyError(f"Missing required columns {missing}; have {list(cols)}")
    out = pd.DataFrame({
        'r_kpc': pd.to_numeric(df[m['r']], errors='coerce'),
        'vbar_kms': pd.to_numeric(df[m['vbar']], errors='coerce'),
        'v_obs_kms': pd.to_numeric(df[m['vobs']], errors='coerce'),
    })
    if m['vlt'] is not None:
        out['v_LogTail_kms'] = pd.to_numeric(df[m['vlt']], errors='coerce')
    return out.dropna(subset=['r_kpc','vbar_kms','v_obs_kms'])


def fit_nfw_for_mw(R: np.ndarray, Vobs: np.ndarray, Vbar: np.ndarray) -> np.ndarray:
    # Coarse grid; adjust if needed
    rs_grid   = np.geomspace(3.0, 50.0, 18)
    rhos_grid = np.geomspace(1e6, 1e10, 18)
    best=None; best_sse=np.inf
    for rs in rs_grid:
        for rhos in rhos_grid:
            Vh = nfw_velocity_kms(R, rhos, rs)
            Vpred = np.sqrt(np.maximum(Vbar**2 + Vh**2, 0.0))
            sse = float(np.sum((Vobs - Vpred)**2))
            if sse < best_sse:
                best_sse = sse; best=(rhos, rs)
    if best is None:
        return np.full_like(Vobs, np.nan)
    rho_s, r_s = best
    Vh = nfw_velocity_kms(R, rho_s, r_s)
    return np.sqrt(np.maximum(Vbar**2 + Vh**2, 0.0))


def main():
    ap = argparse.ArgumentParser(description='Plot Milky Way rotation curve: Observed vs GR(baryons) vs MOND vs LogTail vs NFW (DM).')
    ap.add_argument('--pred_csv', default=str(Path('out')/'mw'/'results_logtail_only'/'predictions_with_LogTail.csv'))
    ap.add_argument('--out_png', default=str(Path('figs')/'mw_rc_compare.png'))
    ap.add_argument('--a0', type=float, default=1.2e-10, help='MOND a0 (m/s^2)')
    args = ap.parse_args()

    df = normalize(pd.read_csv(args.pred_csv))
    df = df.sort_values('r_kpc')
    R = df['r_kpc'].to_numpy()
    Vbar = df['vbar_kms'].to_numpy()
    Vobs = df['v_obs_kms'].to_numpy()

    # Baselines
    Vgr = Vbar.copy()
    Vmond = mond_simple(Vbar, R, a0=float(args.a0))
    Vnfw = fit_nfw_for_mw(R, Vobs, Vbar)
    Vlt = df['v_LogTail_kms'].to_numpy() if 'v_LogTail_kms' in df.columns else None

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    # Observed points
    ax.scatter(R, Vobs, s=26, color='k', alpha=0.8, label='Observed (Gaia bins)')
    # Curves
    ax.plot(R, Vgr,  '-', color='C2', lw=2.0, label='GR (baryons)')
    ax.plot(R, Vmond,'--', color='C3', lw=2.0, label='MOND (simple)')
    if Vlt is not None:
        ax.plot(R, Vlt, '-', color='C0', lw=2.2, label='LogTail (best)')
    if np.isfinite(Vnfw).any():
        ax.plot(R, Vnfw,'-.', color='C1', lw=2.0, label='NFW (best fit)')

    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('v (km/s)')
    ax.set_title('Milky Way rotation curve (Gaia bins)')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=180)
    print(f'Wrote {args.out_png}')


if __name__ == '__main__':
    main()
