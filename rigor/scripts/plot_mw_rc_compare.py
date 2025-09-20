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

def _v_circ_mn_disk_kms(R_kpc: np.ndarray, M_d: float, a: float, b: float) -> np.ndarray:
    R = np.asarray(R_kpc, dtype=float)
    A = a + b
    denom = np.power(R*R + A*A, 1.5)
    G = 4.300917270e-6
    v2 = G * M_d * (R*R) / np.maximum(denom, 1e-12)
    return np.sqrt(np.maximum(v2, 0.0))

def _v_circ_hernquist_kms(R_kpc: np.ndarray, M_b: float, a: float) -> np.ndarray:
    R = np.asarray(R_kpc, dtype=float)
    G = 4.300917270e-6
    v2 = G * M_b * R / np.maximum((R + a)**2, 1e-12)
    return np.sqrt(np.maximum(v2, 0.0))

def _vbar_from_meta(R_kpc: np.ndarray, meta_path: Path|None) -> tuple[np.ndarray, dict|None]:
    if meta_path is None or (not meta_path.exists()):
        return np.array([]), None
    try:
        import json
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        p = meta.get('fit_params', {})
        Md = float(p.get('M_d_Msun'))
        ad = float(p.get('a_d_kpc'))
        bd = float(p.get('b_d_kpc'))
        Mb = float(p.get('M_b_Msun'))
        ab = float(p.get('a_b_kpc'))
        vd = _v_circ_mn_disk_kms(R_kpc, Md, ad, bd)
        vb = _v_circ_hernquist_kms(R_kpc, Mb, ab)
        return np.sqrt(np.maximum(vd*vd + vb*vb, 0.0)), p
    except Exception:
        return np.array([]), None

def _fit_nfw_params(R: np.ndarray, Vobs: np.ndarray, Vbar: np.ndarray) -> tuple[float,float]|None:
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
    return best

def main():
    ap = argparse.ArgumentParser(description='Plot Milky Way rotation curve: Observed vs GR(baryons) vs MOND vs LogTail vs NFW (DM).')
    ap.add_argument('--pred_csv', default=str(Path('out')/'mw'/'results_logtail_only'/'predictions_with_LogTail.csv'))
    ap.add_argument('--out_png', default=str(Path('figs')/'mw_rc_compare.png'))
    ap.add_argument('--a0', type=float, default=1.2e-10, help='MOND a0 (m/s^2)')
    ap.add_argument('--logtail_global', type=str, default='', help='SPARC-global params v0=...,rc=...,r0=...,delta=... (required)')
    ap.add_argument('--only_global', action='store_true', help='If set, suppress MW-refit LogTail curve')
    ap.add_argument('--meta_json', type=str, default=str(Path('out')/'mw'/'mw_predictions_by_radius.meta.json'), help='Meta JSON with baryon fit params (MN+Hern)')
    ap.add_argument('--extrap_frac', type=float, default=1.1, help='Extend curves to extrap_frac * R_max')
    ap.add_argument('--rmin_kpc', type=float, default=1.0, help='Plot/extrapolate starting radius (kpc), default 1.0')
    args = ap.parse_args()

    df = normalize(pd.read_csv(args.pred_csv))
    df = df.sort_values('r_kpc')
    R_bins = df['r_kpc'].to_numpy()
    Vbar_bins = df['vbar_kms'].to_numpy()
    Vobs_bins = df['v_obs_kms'].to_numpy()
    Verr_bins = df['v_err_kms'].to_numpy() if 'v_err_kms' in df.columns else None

    # Extended R grid
    Rmax = float(np.nanmax(R_bins)) if np.isfinite(R_bins).any() else 20.0
    Rmin = max(0.0, float(args.rmin_kpc))
    R_ext = np.arange(Rmin, max(Rmin+0.1, args.extrap_frac*Rmax) + 1e-9, 0.1)

    # Recompute Vbar on extended grid if meta is available
    meta_path = Path(args.meta_json) if args.meta_json else None
    Vbar_ext, fit_params = _vbar_from_meta(R_ext, meta_path)
    if Vbar_ext.size == 0:
        # fallback: interpolate Vbar_bins
        Vbar_ext = np.interp(R_ext, R_bins, Vbar_bins, left=Vbar_bins[0] if len(Vbar_bins)>0 else 0.0, right=Vbar_bins[-1] if len(Vbar_bins)>0 else 0.0)

    # Baselines on extended grid
    Vgr_ext = Vbar_ext.copy()
    Vmond_ext = _mond_simple_analytic(Vbar_ext, R_ext, a0_si=float(args.a0))

    # NFW: fit on bins, then evaluate on extended grid
    best_nfw = _fit_nfw_params(R_bins, Vobs_bins, Vbar_bins)
    Vnfw_ext = None
    if best_nfw is not None:
        rhos, rs = best_nfw
        Vh_ext = nfw_velocity_kms(R_ext, rhos, rs)
        Vnfw_ext = np.sqrt(np.maximum(Vbar_ext**2 + Vh_ext**2, 0.0))

    # LogTail SPARC-global on extended grid
    Vlt_refit = None if args.only_global else (df['v_LogTail_kms'].to_numpy() if 'v_LogTail_kms' in df.columns else None)
    Vlt_global = None
    if args.logtail_global:
        try:
            p = _parse_fixed(args.logtail_global)
            Vlt_global = _logtail_predict(Vbar_ext, R_ext, p['v0'], p['rc'], p['r0'], p['d'])
        except Exception:
            Vlt_global = None

    # Plot
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    # Observed bins with error bars
    if Verr_bins is not None and np.isfinite(Verr_bins).any():
        ax.errorbar(R_bins, Vobs_bins, yerr=Verr_bins, fmt='o', ms=3.5, lw=0.8, elinewidth=0.8, capsize=2.0,
                    color='k', alpha=0.9, label='Observed (Gaia bins ±1σ)')
    else:
        ax.scatter(R_bins, Vobs_bins, s=26, color='k', alpha=0.8, label='Observed (Gaia bins)')
    # Curves on extended grid
    ax.plot(R_ext, Vgr_ext,  '-', color='C2', lw=2.0, label='GR (baryons)')
    ax.plot(R_ext, Vmond_ext,'--', color='C3', lw=2.0, label='MOND (simple)')
    if Vlt_global is not None:
        ax.plot(R_ext, Vlt_global, '-', color='C0', lw=2.2, label='LogTail (SPARC global)')
    if (Vnfw_ext is not None) and np.isfinite(Vnfw_ext).any():
        ax.plot(R_ext, Vnfw_ext,'-.', color='C1', lw=2.0, label='NFW (best fit)')
    # Optionally show MW refit on bins only (suppressed by --only_global)
    if (not args.only_global) and (Vlt_refit is not None):
        ax.plot(R_bins, Vlt_refit, ':', color='C0', lw=1.8, label='LogTail (MW refit)')

    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('v (km/s)')
    ax.set_title('Milky Way rotation curve (Gaia bins) — SPARC-global LogTail')
    ax.set_xlim(R_ext[0], R_ext[-1])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=180)
    print(f'Wrote {args.out_png}')

    # Provenance
    prov = {
        'pred_csv_used': str(Path(args.pred_csv).resolve()),
        'meta_json': str(meta_path.resolve()) if (meta_path and meta_path.exists()) else None,
        'columns_used': {
            'observed': 'v_obs_kms',
            'observed_err': 'v_err_kms' if ('v_err_kms' in df.columns) else None,
            'baryons_GR': 'recomputed from MN+Hern params in meta, else interp(vbar_kms)',
            'logtail_global': args.logtail_global or None,
            'logtail_refit': None if args.only_global else ('v_LogTail_kms' if ('v_LogTail_kms' in df.columns) else None),
            'mond': 'computed from vbar(R) with a0(m/s^2) converted to (km/s)^2/kpc',
            'nfw': 'best-fit (rho_s, r_s) on bins, then evaluated on R_ext'
        },
        'extrapolation': {'R_max_bins': Rmax, 'R_max_plotted': R_ext[-1], 'step_kpc': 0.1},
        'a0_si_m_s2': float(args.a0),
    }
    prov_path = Path(args.out_png).with_suffix('.provenance.json')
    try:
        import json
        prov_path.write_text(json.dumps(prov, indent=2), encoding='utf-8')
    except Exception:
        pass


if __name__ == '__main__':
    main()
