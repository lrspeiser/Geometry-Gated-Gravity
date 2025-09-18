# -*- coding: utf-8 -*-
from __future__ import annotations
import json, math, numpy as np, pandas as pd
from pathlib import Path

G_kpc_km2_s2_Msun = 4.300917270e-6  # G in (kpc km^2 s^-2 Msun^-1)

def smooth_gate_r(r_kpc, r0_kpc, delta_kpc):
    x = (np.asarray(r_kpc) - r0_kpc) / max(delta_kpc, 1e-9)
    return 0.5*(1.0 + np.tanh(x))

def v_model_logtail(vbar_kms, r_kpc, v0_kms=120., rc_kpc=10., r0_kpc=3., delta_kpc=3.):
    vbar2 = np.asarray(vbar_kms)**2
    r = np.asarray(r_kpc)
    S = smooth_gate_r(r, r0_kpc, delta_kpc)
    add = (v0_kms**2) * (r/(r + rc_kpc)) * S
    v2 = vbar2 + add
    return np.sqrt(np.maximum(v2, 0.0))

def v_model_muphi(vbar_kms, r_kpc, eps=1.5, v_c_kms=160., p=3.0):
    vbar2 = np.asarray(vbar_kms)**2
    Phi_c = (v_c_kms**2)
    mu = 1.0 + eps / (1.0 + np.power(np.maximum(vbar2, 0.0)/Phi_c, p))
    v2 = vbar2 * mu
    return np.sqrt(np.maximum(v2, 0.0))

def _first_present(cols: list[str], *cands: str, default: str|None=None) -> str|None:
    for c in cands:
        if c in cols:
            return c
    return default

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a view with normalized column names expected by this script.
    Required logical fields: gal_id, r_kpc, vbar_kms, v_obs_kms, is_outer (optional), M_bary_Msun (optional).
    """
    cols = list(df.columns)
    mapping = {}
    mapping['gal_id']   = _first_present(cols, 'gal_id','galaxy','Galaxy')
    mapping['r_kpc']    = _first_present(cols, 'r_kpc','R_kpc','R')
    mapping['vbar_kms'] = _first_present(cols, 'vbar_kms','Vbar_kms','Vbar')
    mapping['v_obs_kms']= _first_present(cols, 'v_obs_kms','Vobs_kms','V_obs_kms','Vobs')
    mapping['is_outer'] = _first_present(cols, 'is_outer','outer_flag')
    mapping['M_bary_Msun'] = _first_present(cols, 'M_bary_Msun','Baryonic_Mass_Msun','M_bary')
    # Validate required
    required = ['gal_id','r_kpc','vbar_kms','v_obs_kms']
    missing = [k for k in required if mapping[k] is None]
    if missing:
        raise KeyError(f"Missing required columns (logical): {missing}. Available columns: {cols}")
    out = pd.DataFrame({
        'gal_id': df[mapping['gal_id']],
        'r_kpc': pd.to_numeric(df[mapping['r_kpc']], errors='coerce'),
        'vbar_kms': pd.to_numeric(df[mapping['vbar_kms']], errors='coerce'),
        'v_obs_kms': pd.to_numeric(df[mapping['v_obs_kms']], errors='coerce'),
    })
    if mapping['is_outer'] is not None:
        out['is_outer'] = df[mapping['is_outer']].astype(bool)
    if mapping['M_bary_Msun'] is not None:
        out['M_bary_Msun'] = pd.to_numeric(df[mapping['M_bary_Msun']], errors='coerce')
    return out

def infer_outer_mask(df: pd.DataFrame) -> np.ndarray:
    if 'is_outer' in df.columns:
        return df['is_outer'].astype(bool).values
    mask = np.zeros(len(df), dtype=bool)
    for gid, sub in df.groupby('gal_id'):
        idx = sub.sort_values('r_kpc').index
        k = max(1, int(0.3*len(idx)))
        mask[idx[-k:]] = True
    return mask

def median_closeness(v_obs, v_pred) -> float:
    v_obs = np.asarray(v_obs); v_pred = np.asarray(v_pred)
    c = 1.0 - np.abs(v_obs - v_pred)/np.maximum(v_obs, 1.0)
    return 100.0 * float(np.median(c))

def fit_global_grid(df: pd.DataFrame, model_name: str, grid: list[dict], outer_only=True):
    outer_mask = infer_outer_mask(df) if outer_only else np.ones(len(df), bool)
    x = df.loc[outer_mask, ['r_kpc','vbar_kms','v_obs_kms']].to_numpy()
    r, vbar, vobs = x[:,0], x[:,1], x[:,2]
    best, best_score = None, -1e9
    for params in grid:
        if model_name == 'LogTail':
            v = v_model_logtail(vbar, r, **params)
        elif model_name == 'MuPhi':
            v = v_model_muphi(vbar, r, **params)
        else:
            raise ValueError('Unknown model')
        score = median_closeness(vobs, v)
        if score > best_score:
            best, best_score = params.copy(), float(score)
    return best, best_score

def apply_model(df: pd.DataFrame, model_name: str, params: dict):
    r, vbar = df['r_kpc'].to_numpy(), df['vbar_kms'].to_numpy()
    if model_name == 'LogTail':
        v = v_model_logtail(vbar, r, **params)
    else:
        v = v_model_muphi(vbar, r, **params)
    col = f'v_{model_name}_kms'
    out = df.copy(); out[col] = v
    return out, col

def v_flat_per_gal(df_pred: pd.DataFrame, model_col: str, frac_outer=0.3) -> pd.DataFrame:
    rows = []
    for gid, sub in df_pred.groupby('gal_id'):
        sub = sub.sort_values('r_kpc')
        k = max(1, int(frac_outer*len(sub)))
        vflat = float(np.median(sub[model_col].tail(k).values))
        Mb = float(sub['M_bary_Msun'].iloc[0]) if 'M_bary_Msun' in sub.columns else float('nan')
        rows.append((gid, vflat, Mb))
    return pd.DataFrame(rows, columns=['gal_id','vflat_kms','M_bary_Msun'])

def rar_table(df_pred: pd.DataFrame, model_col: str) -> pd.DataFrame:
    r = df_pred['r_kpc'].to_numpy()
    vobs2 = df_pred['v_obs_kms'].to_numpy()**2
    vbar2 = df_pred['vbar_kms'].to_numpy()**2
    g_obs = vobs2 / np.maximum(r, 1e-9)
    g_bar = vbar2 / np.maximum(r, 1e-9)
    return pd.DataFrame({'gal_id': df_pred['gal_id'], 'r_kpc': df_pred['r_kpc'],
                         'g_obs': g_obs, 'g_bar': g_bar,
                         'g_mod': df_pred[model_col]**2 / np.maximum(r, 1e-9)})

def delta_sigma_logtail_SIS(R_kpc, v0_kms, R_trunc_kpc):
    R = np.asarray(R_kpc)
    dSig = v0_kms**2 / (4.0 * G_kpc_km2_s2_Msun * np.maximum(R, 1e-9))
    dSig[R > R_trunc_kpc] = 0.0
    return dSig

# --- New utilities for BTFR, outer slopes, curved RAR, and lensing comparison ---

def join_masses(df: pd.DataFrame, mass_parquet: Path | None) -> pd.DataFrame:
    """Attach SPARC baryonic mass (M_bary_Msun) by galaxy if available.
    mass_parquet should contain columns: 'galaxy' and 'M_bary'.
    df must contain logical 'gal_id'.
    """
    if mass_parquet is None or (not Path(mass_parquet).exists()):
        return df
    try:
        m = pd.read_parquet(mass_parquet)
    except Exception:
        return df
    # Normalize columns
    cols = list(m.columns)
    gcol = 'galaxy' if 'galaxy' in cols else ('Galaxy' if 'Galaxy' in cols else None)
    mbcol = 'M_bary' if 'M_bary' in cols else ('M_bary_Msun' if 'M_bary_Msun' in cols else None)
    if (gcol is None) or (mbcol is None):
        return df
    mm = m[[gcol, mbcol]].rename(columns={gcol: 'gal_id', mbcol: 'M_bary_Msun'})
    # Ensure comparable types
    left = df.copy()
    left['gal_id'] = left['gal_id'].astype(str)
    mm['gal_id'] = mm['gal_id'].astype(str)
    j = left.merge(mm, on='gal_id', how='left')
    return j

def btfr_fit(btfr_df: pd.DataFrame) -> dict:
    """Fit log10(M_b) vs log10(v_flat) with OLS; return slope alpha and scatter (dex)."""
    d = btfr_df.dropna(subset=['M_bary_Msun','vflat_kms']).copy()
    if d.empty:
        return {'n': 0}
    x = np.log10(np.clip(d['vflat_kms'].to_numpy(), 1e-6, None))
    y = np.log10(np.clip(d['M_bary_Msun'].to_numpy(), 1e-6, None))
    A = np.vstack([x, np.ones_like(x)]).T
    alpha, beta = np.linalg.lstsq(A, y, rcond=None)[0]  # y = alpha*x + beta
    yhat = alpha*x + beta
    resid = y - yhat
    scatter = float(np.sqrt(np.mean(resid**2)))  # dex
    return {'n': int(len(d)), 'alpha': float(alpha), 'beta': float(beta), 'scatter_dex': scatter}

def outer_slopes(df_pred: pd.DataFrame, vcol: str, frac_outer=0.3) -> pd.DataFrame:
    """Return per-galaxy slopes s = d ln v / d ln r over the outer fraction of points."""
    rows = []
    for gid, sub in df_pred.groupby('gal_id'):
        sub = sub.sort_values('r_kpc')
        k = max(2, int(frac_outer*len(sub)))
        tail = sub.tail(k)
        def slope(vname):
            r = np.asarray(tail['r_kpc']); v = np.asarray(tail[vname])
            mask = np.isfinite(r) & (r>0) & np.isfinite(v) & (v>0)
            if mask.sum() < 2:
                return float('nan')
            X = np.log(r[mask]); Y = np.log(v[mask])
            A = np.vstack([X, np.ones_like(X)]).T
            a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
            return float(a)
        rows.append((gid, slope('v_obs_kms'), slope(vcol)))
    return pd.DataFrame(rows, columns=['gal_id','s_obs','s_model'])

def rar_curved_stats(rar_df: pd.DataFrame, ycol: str='g_mod', n_bins: int=30) -> dict:
    """Compute a curved RAR median relation and approximate orthogonal scatter in log space.
    ycol in {'g_mod','g_obs'}. Returns dict with scatter_dex and R2-like metric.
    """
    d = rar_df.dropna(subset=['g_bar', ycol]).copy()
    if d.empty:
        return {'n': 0}
    x = np.log10(np.clip(d['g_bar'].to_numpy(), 1e-12, None))
    y = np.log10(np.clip(d[ycol].to_numpy(), 1e-12, None))
    # Bin in x and compute median y
    bins = np.linspace(np.nanmin(x), np.nanmax(x), n_bins+1)
    idx = np.digitize(x, bins) - 1
    med_x, med_y = [], []
    for i in range(n_bins):
        m = (idx == i)
        if np.any(m):
            med_x.append(np.median(x[m]))
            med_y.append(np.median(y[m]))
    if len(med_x) < 2:
        return {'n': int(len(x))}
    med_x = np.array(med_x); med_y = np.array(med_y)
    # Interpolate median curve at each x
    y_curve = np.interp(x, med_x, med_y, left=med_y[0], right=med_y[-1])
    # Approximate local slope via finite differences on the median curve
    slope_local = np.gradient(med_y, med_x)
    slope_at_x = np.interp(x, med_x, slope_local, left=slope_local[0], right=slope_local[-1])
    # Orthogonal residual approx: vertical residual divided by sqrt(1 + m^2)
    vert = y - y_curve
    orth = vert / np.sqrt(1.0 + slope_at_x**2)
    scatter = float(np.sqrt(np.mean(orth**2)))
    # R2-like metric w.r.t. constant model
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum(vert**2)
    r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else float('nan')
    return {'n': int(len(x)), 'scatter_dex_orth': scatter, 'r2_vs_const': r2}

def lensing_compare_basic(R_pred_kpc: np.ndarray, dSig_pred: np.ndarray, lensing_stack_csv: Path | None) -> dict:
    """Compare predicted lensing with a provided stack CSV (R_kpc, DeltaSigma_Msun_per_kpc2).
    Returns slope of log-log pred, and amplitude ratios at 50 and 100 kpc if stack provided.
    """
    # Pred slope in log-log
    X = np.log10(np.clip(R_pred_kpc, 1e-9, None)); Y = np.log10(np.clip(dSig_pred, 1e-30, None))
    A = np.vstack([X, np.ones_like(X)]).T
    m_pred, b_pred = np.linalg.lstsq(A, Y, rcond=None)[0]
    out = {'pred_slope_loglog': float(m_pred)}
    # Amplitudes
    for Rq in [50.0, 100.0]:
        d = float(np.interp(Rq, R_pred_kpc, dSig_pred))
        out[f'pred_DeltaSigma_{int(Rq)}kpc'] = d
    # If stack provided, compare amplitudes
    if lensing_stack_csv is not None and Path(lensing_stack_csv).exists():
        try:
            obs = pd.read_csv(lensing_stack_csv)
            # Expect columns like R_kpc, DeltaSigma_Msun_per_kpc2
            R = np.asarray(obs.iloc[:,0]); DS = np.asarray(obs.iloc[:,1])
            for Rq in [50.0, 100.0]:
                dobs = float(np.interp(Rq, R, DS))
                dpre = float(np.interp(Rq, R_pred_kpc, dSig_pred))
                out[f'obs_DeltaSigma_{int(Rq)}kpc'] = dobs
                out[f'amp_ratio_pred_over_obs_{int(Rq)}kpc'] = (dpre / dobs) if dobs > 0 else float('nan')
        except Exception:
            pass
    return out

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--do_cv', action='store_true')
    ap.add_argument('--sparc_master_parquet', default=str(Path('data')/'sparc_master_clean.parquet'))
    ap.add_argument('--mass_coupled_v0', action='store_true', help='Enable LogTail v0(Mb)=A*(Mb/1e10)^(1/4) grid search')
    ap.add_argument('--lensing_stack_csv', default=None, help='Optional CSV of stacked lensing to compare against')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.pred_csv)
    df = normalize_columns(raw)
    # Attach baryonic masses from SPARC master, if available
    df = join_masses(df, Path(args.sparc_master_parquet) if args.sparc_master_parquet else None)

    # Narrow grids based on your prior run
    grid_logtail = [
        dict(v0_kms=v0, rc_kpc=rc, r0_kpc=r0, delta_kpc=dl)
        for v0 in [100, 120, 140]
        for rc in [5, 10, 15]
        for r0 in [2, 3, 4]
        for dl in [2, 3, 4]
    ]
    grid_muphi = [
        dict(eps=eps, v_c_kms=vc, p=p)
        for eps in [1.0, 1.5, 2.0]
        for vc in [140, 160, 180]
        for p  in [2.0, 3.0, 4.0]
    ]

    best_logtail, score_lt = fit_global_grid(df, 'LogTail', grid_logtail, outer_only=True)
    best_muphi,   score_mp = fit_global_grid(df, 'MuPhi',   grid_muphi,   outer_only=True)

    df1, col_lt = apply_model(df, 'LogTail', best_logtail)
    df2, col_mp = apply_model(df1, 'MuPhi', best_muphi)
    # Merge back with original columns to preserve any extra fields
    merged = raw.copy()
    merged[col_lt] = df2[col_lt]
    merged[col_mp] = df2[col_mp]
    merged.to_csv(out_dir/'predictions_with_LogTail_MuPhi.csv', index=False)

    outer = infer_outer_mask(df2)
    sum_lt = dict(model='LogTail', median=median_closeness(df2.loc[outer,'v_obs_kms'], df2.loc[outer, col_lt]), params=best_logtail)
    sum_mp = dict(model='MuPhi',   median=median_closeness(df2.loc[outer,'v_obs_kms'], df2.loc[outer, col_mp]), params=best_muphi)
    with open(out_dir/'summary_logtail_muphi.json', 'w') as f:
        json.dump({'LogTail': sum_lt, 'MuPhi': sum_mp}, f, indent=2)

    # BTFR and RAR exports (with masses attached if available)
    btfr_lt = v_flat_per_gal(df2, col_lt)
    btfr_mp = v_flat_per_gal(df2, col_mp)
    btfr_lt.to_csv(out_dir/'btfr_logtail.csv', index=False)
    btfr_mp.to_csv(out_dir/'btfr_muphi.csv', index=False)
    # Fit BTFR slope/scatter for LogTail
    with open(out_dir/'btfr_logtail_fit.json', 'w') as f:
        json.dump(btfr_fit(btfr_lt), f, indent=2)

    rar_lt = rar_table(df2, col_lt)
    rar_mp = rar_table(df2, col_mp)
    rar_lt.to_csv(out_dir/'rar_logtail.csv', index=False)
    rar_mp.to_csv(out_dir/'rar_muphi.csv', index=False)

    # Curved RAR stats (data/obs vs g_bar, and LogTail vs g_bar)
    with open(out_dir/'rar_obs_curved_stats.json', 'w') as f:
        json.dump(rar_curved_stats(rar_lt.rename(columns={'g_mod':'g_obs'}), ycol='g_obs'), f, indent=2)
    with open(out_dir/'rar_logtail_curved_stats.json', 'w') as f:
        json.dump(rar_curved_stats(rar_lt, ycol='g_mod'), f, indent=2)

    # Outer-slope distribution
    slopes = outer_slopes(df2.rename(columns={col_lt:'v_model_kms'}).rename(columns={'v_model_kms':col_lt}), col_lt)
    slopes.to_csv(out_dir/'outer_slopes_logtail.csv', index=False)

    # Lensing shape (LogTail) + optional comparison
    R_kpc = np.geomspace(10, 300, 30)
    dSig = delta_sigma_logtail_SIS(R_kpc, best_logtail['v0_kms'], R_trunc_kpc=150.0)
    lens_df = pd.DataFrame({'R_kpc': R_kpc, 'DeltaSigma_Msun_per_kpc2': dSig})
    lens_df.to_csv(out_dir/'lensing_logtail_shapes.csv', index=False)
    lens_cmp = lensing_compare_basic(R_kpc, dSig, Path(args.lensing_stack_csv) if args.lensing_stack_csv else None)
    with open(out_dir/'lensing_logtail_comparison.json', 'w') as f:
        json.dump(lens_cmp, f, indent=2)

    # Optional: Mass-coupled LogTail v0(Mb) = A * (Mb/1e10)^(1/4)
    if args.mass_coupled_v0 and ('M_bary_Msun' in df2.columns):
        def apply_logtail_mass(df_in: pd.DataFrame, A_kms: float, rc: float, r0: float, dl: float):
            Mb = df_in['M_bary_Msun'].to_numpy()
            v0_vec = A_kms * np.power(np.clip(Mb/1e10, 1e-12, None), 0.25)
            r = df_in['r_kpc'].to_numpy(); vbar = df_in['vbar_kms'].to_numpy()
            S = smooth_gate_r(r, r0, dl)
            add = (v0_vec**2) * (r/(r + rc)) * S
            v2 = vbar**2 + add
            v = np.sqrt(np.clip(v2, 0.0, None))
            out = df_in.copy(); col = 'v_LogTail_mass_coupled_kms'
            out[col] = v
            return out, col
        grid_A = [100, 120, 140, 160]
        rc_list = [5, 10, 15]; r0_list = [2, 3, 4]; dl_list = [2, 3, 4]
        best, best_score = None, -1e9
        for A in grid_A:
            for rc in rc_list:
                for r0 in r0_list:
                    for dl in dl_list:
                        tmp, col = apply_logtail_mass(df, A, rc, r0, dl)
                        outer_m = infer_outer_mask(tmp)
                        score = median_closeness(tmp.loc[outer_m,'v_obs_kms'], tmp.loc[outer_m, col])
                        if score > best_score:
                            best = dict(A_kms=A, rc_kpc=rc, r0_kpc=r0, delta_kpc=dl)
                            best_score = score
        if best is not None:
            df_m, col_mass = apply_logtail_mass(df, **best)
            merged_mass = raw.copy(); merged_mass[col_mass] = df_m[col_mass]
            merged_mass.to_csv(out_dir/'predictions_with_LogTail_mass_coupled.csv', index=False)
            sum_mass = dict(model='LogTail_mass_coupled', median=median_closeness(df_m.loc[infer_outer_mask(df_m),'v_obs_kms'], df_m.loc[infer_outer_mask(df_m), col_mass]), params=best)
            with open(out_dir/'summary_logtail_mass_coupled.json', 'w') as f:
                json.dump(sum_mass, f, indent=2)
            btfr_mass = v_flat_per_gal(df_m, col_mass)
            btfr_mass.to_csv(out_dir/'btfr_logtail_mass_coupled.csv', index=False)
            with open(out_dir/'btfr_logtail_mass_coupled_fit.json', 'w') as f:
                json.dump(btfr_fit(btfr_mass), f, indent=2)

    print('Done.')
