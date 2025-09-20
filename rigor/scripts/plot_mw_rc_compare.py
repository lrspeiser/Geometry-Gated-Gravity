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


def _mond_simple_analytic(Vbar_kms: np.ndarray, R_kpc: np.ndarray, a0_si: float = 1.2e-10) -> np.ndarray:
    # Compute MOND (simple) with explicit unit handling: a0 in m/s^2, g_N in (km/s)^2/kpc
    # Convert a0 to (km/s)^2 per kpc
    AUNIT = 3.2407789e-14  # (km/s)^2 per kpc per m/s^2
    a0 = float(a0_si) / AUNIT
    R = np.maximum(np.asarray(R_kpc, dtype=float), 1e-9)
    vbar = np.maximum(np.asarray(Vbar_kms, dtype=float), 0.0)
    gN = (vbar**2) / R
    g  = 0.5*(gN + np.sqrt(np.maximum(gN*gN + 4.0*gN*a0, 0.0)))
    return np.sqrt(np.maximum(g*R, 0.0))

def _parse_fixed(s: str):
    kv = dict(tok.split('=') for tok in s.split(',') if '=' in tok)
    return dict(v0=float(kv['v0']), rc=float(kv['rc']), r0=float(kv['r0']), d=float(kv.get('delta', kv.get('d', 0.0))))

def _logtail_predict(Vbar_kms: np.ndarray, R_kpc: np.ndarray, v0: float, rc: float, r0: float, d: float) -> np.ndarray:
    vbar2 = np.maximum(np.asarray(Vbar_kms, dtype=float), 0.0)**2
    R = np.maximum(np.asarray(R_kpc, dtype=float), 1e-9)
    S = 0.5*(1.0 + np.tanh((R - float(r0))/max(float(d), 1e-6)))
    tail = (float(v0)**2) * (R/(R + max(float(rc), 1e-6))) * S
    V2 = vbar2 + np.maximum(tail, 0.0)
    return np.sqrt(np.maximum(V2, 0.0))

def main():
    ap = argparse.ArgumentParser(description='Plot Milky Way rotation curve: Observed vs GR(baryons) vs MOND vs LogTail vs NFW (DM).')
    ap.add_argument('--pred_csv', default=str(Path('out')/'mw'/'results_logtail_only'/'predictions_with_LogTail.csv'))
    ap.add_argument('--out_png', default=str(Path('figs')/'mw_rc_compare.png'))
    ap.add_argument('--a0', type=float, default=1.2e-10, help='MOND a0 (m/s^2)')
    ap.add_argument('--logtail_global', type=str, default='', help='Optional SPARC-global params v0=140,rc=15,r0=3,delta=4 to overlay')
    args = ap.parse_args()

    df = normalize(pd.read_csv(args.pred_csv))
    df = df.sort_values('r_kpc')
    R = df['r_kpc'].to_numpy()
    Vbar = df['vbar_kms'].to_numpy()
    Vobs = df['v_obs_kms'].to_numpy()

    # Baselines
    Vgr = Vbar.copy()
    # Use analytic MOND with explicit unit conversion for a0
    Vmond = _mond_simple_analytic(Vbar, R, a0_si=float(args.a0))
    Vnfw = fit_nfw_for_mw(R, Vobs, Vbar)
    Vlt_refit = df['v_LogTail_kms'].to_numpy() if 'v_LogTail_kms' in df.columns else None

    # Optional SPARC-global LogTail overlay
    Vlt_global = None
    if args.logtail_global:
        try:
            p = _parse_fixed(args.logtail_global)
            Vlt_global = _logtail_predict(Vbar, R, p['v0'], p['rc'], p['r0'], p['d'])
        except Exception:
            Vlt_global = None

    # Plot
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    # Observed points
    ax.scatter(R, Vobs, s=26, color='k', alpha=0.8, label='Observed (Gaia bins)')
    # Curves
    ax.plot(R, Vgr,  '-', color='C2', lw=2.0, label='GR (baryons)')
    ax.plot(R, Vmond,'--', color='C3', lw=2.0, label='MOND (simple)')
    if Vlt_refit is not None:
        ax.plot(R, Vlt_refit, '-', color='C0', lw=2.2, label='LogTail (MW refit)')
    if Vlt_global is not None:
        ax.plot(R, Vlt_global, ':', color='C0', lw=2.2, label='LogTail (SPARC global)')
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
