# -*- coding: utf-8 -*-
from __future__ import annotations
import json, math, numpy as np, pandas as pd, re, difflib
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

def _norm_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'[\s\-_\.]+', '', s)
    s = re.sub(r'[^a-z0-9]', '', s)
    for pre in ('ngc','ugc','ic','eso','ddo'):
        if s.startswith(pre):
            m = re.match(r'([a-z]+)(\d+)(.*)$', s)
            if m:
                pre2, num, rest = m.groups()
                s = f"{pre2}{num.zfill(4)}{rest}"
            break
    return s

def join_masses(df: pd.DataFrame, primary_parquet: Path|None, fallback_parquet: Path|None=None,
                allow_fuzzy: bool=True, fuzzy_cutoff: float=0.88, audit_out: Path|None=None) -> pd.DataFrame:
    """
    Attach M_bary_Msun using a robust join with name normalization and optional fuzzy matching.
    """
    def _load(pth: Path|None):
        if pth is None or (not Path(pth).exists()):
            return None
        try:
            return pd.read_parquet(pth)
        except Exception:
            return None

    m = _load(primary_parquet)
    if m is None:
        m = _load(fallback_parquet)
    if m is None:
        return df

    # choose best mass column
    cand_cols = [c for c in ('M_bary','M_bary_Msun','Mbary','Mb') if c in m.columns]
    if not cand_cols:
        if {'M_star','M_HI'}.issubset(set(m.columns)):
            m = m.copy()
            m['M_bary'] = pd.to_numeric(m['M_star'], errors='coerce') + 1.33*pd.to_numeric(m['M_HI'], errors='coerce')
            cand_cols = ['M_bary']
        else:
            return df
    mcol = cand_cols[0]

    # pick a name column
    gcol = None
    for c in ('galaxy','Galaxy','name','Name','gal_name'):
        if c in m.columns:
            gcol = c; break
    if gcol is None:
        return df

    left = df.copy()
    left['gal_id'] = left['gal_id'].astype(str)
    left['_norm'] = left['gal_id'].map(_norm_name)

    mm = m[[gcol, mcol]].rename(columns={gcol:'gal_src', mcol:'M_bary_Msun'})
    mm['gal_src'] = mm['gal_src'].astype(str)
    mm['_norm'] = mm['gal_src'].map(_norm_name)

    # exact on raw names
    j1 = left.merge(mm[['gal_src','M_bary_Msun']], left_on='gal_id', right_on='gal_src', how='left', suffixes=('','_src'))
    # exact on normalized
    j_norm = left.merge(mm[['_norm','M_bary_Msun']], on='_norm', how='left', suffixes=('','_norm'))
    j1['M_bary_Msun'] = j1['M_bary_Msun'].fillna(j_norm['M_bary_Msun'])

    # optional fuzzy for remaining
    if allow_fuzzy:
        still = j1['M_bary_Msun'].isna()
        if still.any():
            src_names = mm['gal_src'].tolist()
            src_map = dict(zip(mm['gal_src'], mm['M_bary_Msun']))
            targets = j1.loc[still, 'gal_id'].tolist()
            mb_fuzzy = []
            for t in targets:
                match = difflib.get_close_matches(t, src_names, n=1, cutoff=fuzzy_cutoff)
                mb_fuzzy.append(src_map[match[0]] if match else np.nan)
            j1.loc[still, 'M_bary_Msun'] = mb_fuzzy

    # Heuristic unit fix (1e10 Msun -> Msun) if needed
    mb = pd.to_numeric(j1['M_bary_Msun'], errors='coerce')
    med = float(np.nanmedian(mb))
    if np.isfinite(med) and (0.01 <= med <= 100.0):
        mb = mb * 1e10
    j1['M_bary_Msun'] = mb

    # audit
    if audit_out is not None:
        audit = j1[['gal_id','M_bary_Msun','_norm']].copy()
        audit['has_mass'] = audit['M_bary_Msun'].notna()
        audit.to_csv(audit_out, index=False)

    return j1.drop(columns=['gal_src'], errors='ignore')

def btfr_fit(btfr_df: pd.DataFrame) -> dict:
    """Fit log10(M_b) vs log10(v_flat) with OLS; return slope alpha and scatter (dex)."""
    d = btfr_df.dropna(subset=['M_bary_Msun','vflat_kms']).copy()
    # Require positive values
    d = d[(d['M_bary_Msun'] > 0) & (d['vflat_kms'] > 0)]
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

# Two-form BTFR with bootstrap CIs

def btfr_fit_two_forms(btfr_df: pd.DataFrame, n_boot: int = 2000, seed: int = 1337) -> dict:
    """Return BTFR fits in both directions with bootstrap CIs.
    Forms:
      1) log10(Mb) = alpha * log10(vflat) + beta
      2) log10(vflat) = beta * log10(Mb) + gamma  (also report alpha_from_beta = 1/beta)
    Scatter is vertical RMS in the dependent variable's log space.
    """
    d = btfr_df.dropna(subset=['M_bary_Msun','vflat_kms']).copy()
    d = d[(d['M_bary_Msun']>0) & (d['vflat_kms']>0)]
    if d.empty:
        return {'n': 0}

    xv = np.log10(np.clip(d['vflat_kms'].to_numpy(), 1e-6, None))
    yM = np.log10(np.clip(d['M_bary_Msun'].to_numpy(), 1e-6, None))

    # OLS fits
    A_v = np.vstack([xv, np.ones_like(xv)]).T
    alpha, beta = np.linalg.lstsq(A_v, yM, rcond=None)[0]   # yM = alpha*xv + beta
    yhatM = alpha*xv + beta
    scatter_M = float(np.sqrt(np.mean((yM - yhatM)**2)))

    A_M = np.vstack([yM, np.ones_like(yM)]).T
    beta2, gamma = np.linalg.lstsq(A_M, xv, rcond=None)[0]  # xv = beta2*yM + gamma
    xhatv = beta2*yM + gamma
    scatter_v = float(np.sqrt(np.mean((xv - xhatv)**2)))

    # Bootstrap CIs
    rng = np.random.default_rng(seed)
    alpha_bs, beta2_bs = [], []
    n = len(d)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xv_b, yM_b = xv[idx], yM[idx]
        A_vb = np.vstack([xv_b, np.ones_like(xv_b)]).T
        a_b, _ = np.linalg.lstsq(A_vb, yM_b, rcond=None)[0]
        A_Mb = np.vstack([yM_b, np.ones_like(yM_b)]).T
        b2_b, _ = np.linalg.lstsq(A_Mb, xv_b, rcond=None)[0]
        alpha_bs.append(a_b); beta2_bs.append(b2_b)
    alpha_ci = (float(np.percentile(alpha_bs, 2.5)), float(np.percentile(alpha_bs, 97.5)))
    beta2_ci  = (float(np.percentile(beta2_bs, 2.5)), float(np.percentile(beta2_bs, 97.5)))

    return {
        'n': int(n),
        'form_Mb_vs_v': {
            'alpha': float(alpha), 'beta': float(beta), 'scatter_dex': scatter_M,
            'alpha_CI95': alpha_ci
        },
        'form_v_vs_Mb': {
            'beta': float(beta2), 'gamma': float(gamma), 'scatter_dex': scatter_v,
            'beta_CI95': beta2_ci,
            'alpha_from_beta': float(1.0/beta2) if beta2 != 0 else float('nan')
        }
    }

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
    # Clean and sort
    R = np.asarray(R_pred_kpc, dtype=float)
    DS = np.asarray(dSig_pred, dtype=float)
    m = np.isfinite(R) & np.isfinite(DS) & (R > 0) & (DS > 0)
    R, DS = R[m], DS[m]
    order = np.argsort(R)
    R, DS = R[order], DS[order]
    # Pred slope in log-log (should be ~ -1 for 1/R)
    X, Y = np.log10(R), np.log10(DS)
    m_pred, b_pred = np.polyfit(X, Y, 1)
    out = {'pred_slope_loglog': float(m_pred)}
    # Amplitudes
    for Rq in [50.0, 100.0]:
        out[f'pred_DeltaSigma_{int(Rq)}kpc'] = float(np.interp(Rq, R, DS))
    # If stack provided, compare amplitudes
    if lensing_stack_csv is not None and Path(lensing_stack_csv).exists():
        try:
            obs = pd.read_csv(lensing_stack_csv)
            R_o = np.asarray(obs.iloc[:,0], dtype=float)
            DS_o = np.asarray(obs.iloc[:,1], dtype=float)
            m2 = np.isfinite(R_o) & np.isfinite(DS_o) & (R_o > 0)
            R_o, DS_o = R_o[m2], DS_o[m2]
            order2 = np.argsort(R_o)
            R_o, DS_o = R_o[order2], DS_o[order2]
            for Rq in [50.0, 100.0]:
                dobs = float(np.interp(Rq, R_o, DS_o))
                dpre = float(np.interp(Rq, R, DS))
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
    # Attach baryonic masses using all_tables first, then fallback to master_clean; write audit CSV
    df = join_masses(
        df,
        primary_parquet=Path('data')/'sparc_all_tables.parquet',
        fallback_parquet=Path(args.sparc_master_parquet) if args.sparc_master_parquet else None,
        allow_fuzzy=False,
        fuzzy_cutoff=0.90,
        audit_out=out_dir/'btfr_mass_join_audit.csv'
    )

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
    # Fit BTFR slope/scatter for LogTail and MuPhi in both forms (with CIs)
    with open(out_dir/'btfr_logtail_fit.json', 'w') as f:
        json.dump(btfr_fit_two_forms(btfr_lt), f, indent=2)
    with open(out_dir/'btfr_muphi_fit.json', 'w') as f:
        json.dump(btfr_fit_two_forms(btfr_mp), f, indent=2)

    rar_lt = rar_table(df2, col_lt)
    rar_mp = rar_table(df2, col_mp)
    rar_lt.to_csv(out_dir/'rar_logtail.csv', index=False)
    rar_mp.to_csv(out_dir/'rar_muphi.csv', index=False)

    # Curved RAR stats (data/obs vs g_bar, and LogTail vs g_bar)
    with open(out_dir/'rar_obs_curved_stats.json', 'w') as f:
        json.dump(rar_curved_stats(rar_lt, ycol='g_obs'), f, indent=2)
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

    # Observed BTFR (sanity check on mass join)
    def vflat_obs_per_gal(df_in: pd.DataFrame, frac_outer=0.3) -> pd.DataFrame:
        rows = []
        for gid, sub in df_in.groupby('gal_id'):
            sub = sub.sort_values('r_kpc')
            k = max(1, int(frac_outer*len(sub)))
            vflat = float(np.median(sub['v_obs_kms'].tail(k).values))
            Mb = float(sub['M_bary_Msun'].iloc[0]) if 'M_bary_Msun' in sub.columns else float('nan')
            rows.append((gid, vflat, Mb))
        return pd.DataFrame(rows, columns=['gal_id','vflat_obs_kms','M_bary_Msun'])

    def btfr_fit_table(df_btfr: pd.DataFrame, vcol: str):
        dtmp = df_btfr.rename(columns={vcol: 'vflat_kms'})
        return btfr_fit_two_forms(dtmp)

    btfr_obs = vflat_obs_per_gal(df2)
    btfr_obs.to_csv(out_dir/'btfr_observed.csv', index=False)
    with open(out_dir/'btfr_observed_fit.json','w') as f:
        json.dump(btfr_fit_table(btfr_obs, 'vflat_obs_kms'), f, indent=2)

    # Quick QC: correlation should be positive if join is correct
    qc = btfr_obs.dropna()
    if len(qc) > 5:
        xv = np.log10(np.clip(qc['vflat_obs_kms'].to_numpy(), 1e-6, None))
        yM = np.log10(np.clip(qc['M_bary_Msun'].to_numpy(), 1e-6, None))
        corr = float(np.corrcoef(xv, yM)[0,1])
        (out_dir/'btfr_qc.txt').write_text(f"Pearson corr(log vflat_obs, log Mb) = {corr:.3f}\n")

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
            df_m, col_mass = apply_logtail_mass(df, best['A_kms'], best['rc_kpc'], best['r0_kpc'], best['delta_kpc'])
            merged_mass = raw.copy(); merged_mass[col_mass] = df_m[col_mass]
            merged_mass.to_csv(out_dir/'predictions_with_LogTail_mass_coupled.csv', index=False)
            sum_mass = dict(model='LogTail_mass_coupled', median=median_closeness(df_m.loc[infer_outer_mask(df_m),'v_obs_kms'], df_m.loc[infer_outer_mask(df_m), col_mass]), params=best)
            with open(out_dir/'summary_logtail_mass_coupled.json', 'w') as f:
                json.dump(sum_mass, f, indent=2)
            btfr_mass = v_flat_per_gal(df_m, col_mass)
            btfr_mass.to_csv(out_dir/'btfr_logtail_mass_coupled.csv', index=False)
            with open(out_dir/'btfr_logtail_mass_coupled_fit.json', 'w') as f:
                json.dump(btfr_fit_two_forms(btfr_mass), f, indent=2)

    print('Done.')
