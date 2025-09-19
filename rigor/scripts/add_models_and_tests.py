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
    mapping['r_kpc']    = _first_present(cols, 'r_kpc','R_kpc','R','Radius_kpc')
    mapping['vbar_kms'] = _first_present(cols, 'vbar_kms','Vbar_kms','Vbar','Baryonic_Speed_km_s')
    mapping['v_obs_kms']= _first_present(cols, 'v_obs_kms','Vobs_kms','V_obs_kms','Vobs','Observed_Speed_km_s')
    mapping['is_outer'] = _first_present(cols, 'is_outer','outer_flag','In_Outer_Region')
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
    # Build an index-aligned boolean mask to be robust to filtered subsets
    mask = pd.Series(False, index=df.index)
    for gid, sub in df.groupby('gal_id'):
        idx = sub.sort_values('r_kpc').index
        k = max(1, int(0.3*len(idx)))
        mask.loc[idx[-k:]] = True
    return mask.values

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
    # remove common suffix artifacts like _rotmod
    s = re.sub(r'(?:\b|_)?rotmod\b', '', s)
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

def load_name_overrides(path: Path|None) -> dict:
    """Optional CSV with columns: gal_id, gal_src. Returns dict gal_id -> gal_src."""
    if path is None or (not path.exists()):
        return {}
    try:
        df = pd.read_csv(path)
        if {'gal_id','gal_src'}.issubset(df.columns):
            return dict(zip(df['gal_id'].astype(str), df['gal_src'].astype(str)))
    except Exception:
        pass
    return {}

def join_masses_enhanced(df: pd.DataFrame,
                         primary_parquet: Path|None,
                         fallback_parquet: Path|None=None,
                         overrides_csv: Path|None=None,
                         allow_fuzzy: bool=False, fuzzy_cutoff: float=0.95,
                         audit_out: Path|None=None) -> pd.DataFrame:
    """
    Attach M_bary_Msun using manual overrides, exact raw and normalized matches, and optional fuzzy matching.
    Writes an audit CSV with join_method and matched name.
    """
    def _load(p: Path|None):
        if p is None or (not p.exists()):
            return None
        try:
            return pd.read_parquet(p)
        except Exception:
            return None

    m = _load(primary_parquet)
    if m is None:
        m = _load(fallback_parquet)
    if m is None:
        return df

    mass_cols = [c for c in ('M_bary','M_bary_Msun','Mbary','Mb') if c in m.columns]
    if not mass_cols:
        if {'M_star','M_HI'}.issubset(set(m.columns)):
            m = m.copy()
            m['M_bary'] = pd.to_numeric(m['M_star'], errors='coerce') + 1.33*pd.to_numeric(m['M_HI'], errors='coerce')
            mass_cols = ['M_bary']
        else:
            return df
    mcol = mass_cols[0]

    gcol = next((c for c in ('galaxy','Galaxy','name','Name','gal_name') if c in m.columns), None)
    if gcol is None:
        return df

    left = df.copy()
    left['gal_id'] = left['gal_id'].astype(str)
    left['_norm'] = left['gal_id'].map(_norm_name)

    mm = m[[gcol, mcol]].rename(columns={gcol:'gal_src', mcol:'M_bary_Msun'})
    mm['gal_src'] = mm['gal_src'].astype(str)
    mm['_norm'] = mm['gal_src'].map(_norm_name)

    audit = pd.DataFrame({'gal_id': left['gal_id'], '_norm': left['_norm']})
    audit['join_method'] = 'missing'
    audit['match_name'] = pd.NA
    joined_mass = pd.Series(index=left.index, dtype='float64')

    overrides = load_name_overrides(overrides_csv)
    if overrides:
        ov_df = pd.DataFrame({'gal_id': list(overrides.keys()), 'gal_src': list(overrides.values())})
        left = left.merge(ov_df, on='gal_id', how='left')
        mask_ov = left['gal_src'].notna()
        if mask_ov.any():
            tmp = left[mask_ov].merge(mm[['gal_src','M_bary_Msun']], on='gal_src', how='left')
            joined_mass.loc[mask_ov] = tmp['M_bary_Msun'].values
            audit.loc[mask_ov, 'join_method'] = 'override'
            audit.loc[mask_ov, 'match_name'] = left.loc[mask_ov, 'gal_src']

    # exact raw
    mask_need = joined_mass.isna()
    if mask_need.any():
        left_sub = left[mask_need].copy()
        left_sub = left_sub.reset_index().rename(columns={'index':'_idx'})
        tmp = left_sub.merge(mm[['gal_src','M_bary_Msun']], left_on='gal_id', right_on='gal_src', how='left', suffixes=('', '_src'))
        col_mass = 'M_bary_Msun_src' if 'M_bary_Msun_src' in tmp.columns else ('M_bary_Msun' if 'M_bary_Msun' in tmp.columns else None)
        ok = tmp[col_mass].notna() if col_mass else pd.Series(False, index=tmp.index)
        if ok.any():
            idxs = tmp.loc[ok, '_idx'].to_numpy()
            joined_mass.loc[idxs] = tmp.loc[ok, col_mass].to_numpy()
            audit.loc[idxs, 'join_method'] = 'exact_raw'
            audit.loc[idxs, 'match_name'] = tmp.loc[ok, 'gal_src'].to_numpy()

    # exact norm
    mask_need = joined_mass.isna()
    if mask_need.any():
        left_sub = left[mask_need].copy()
        left_sub = left_sub.reset_index().rename(columns={'index':'_idx'})
        tmp = left_sub.merge(mm[['_norm','M_bary_Msun','gal_src']], on='_norm', how='left', suffixes=('', '_src'))
        col_mass = 'M_bary_Msun_src' if 'M_bary_Msun_src' in tmp.columns else ('M_bary_Msun' if 'M_bary_Msun' in tmp.columns else None)
        ok = tmp[col_mass].notna() if col_mass else pd.Series(False, index=tmp.index)
        if ok.any():
            idxs = tmp.loc[ok, '_idx'].to_numpy()
            joined_mass.loc[idxs] = tmp.loc[ok, col_mass].to_numpy()
            audit.loc[idxs, 'join_method'] = 'exact_norm'
            audit.loc[idxs, 'match_name'] = tmp.loc[ok, 'gal_src'].to_numpy()

    # fuzzy on raw (strict)
    if allow_fuzzy:
        mask_need = joined_mass.isna()
        if mask_need.any():
            src_names = mm['gal_src'].tolist()
            src_map = dict(zip(mm['gal_src'], mm['M_bary_Msun']))
            targets = left.loc[mask_need, 'gal_id'].tolist()
            picks, mb = [], []
            for t in targets:
                match = difflib.get_close_matches(t, src_names, n=1, cutoff=fuzzy_cutoff)
                if match:
                    picks.append(match[0]); mb.append(src_map[match[0]])
                else:
                    picks.append(pd.NA); mb.append(np.nan)
            idxs = mask_need[mask_need].index
            joined_mass.loc[idxs] = mb
            audit.loc[idxs, 'join_method'] = np.where(pd.notna(picks), 'fuzzy', 'missing')
            audit.loc[idxs, 'match_name'] = picks

    left['M_bary_Msun'] = pd.to_numeric(joined_mass, errors='coerce')
    med = float(np.nanmedian(left['M_bary_Msun']))
    if np.isfinite(med) and (0.01 <= med <= 100.0):
        left['M_bary_Msun'] = left['M_bary_Msun'] * 1e10

    if audit_out is not None:
        audit['M_bary_Msun'] = left['M_bary_Msun']
        audit.to_csv(audit_out, index=False)

    out = df.copy()
    out['M_bary_Msun'] = left['M_bary_Msun'].values
    return out


def attach_observed_vflat(df: pd.DataFrame, all_tables_parquet: Path|None, audit_out: Path|None=None) -> pd.DataFrame:
    """Attach catalog Vflat (km/s) from sparc_all_tables.parquet to each row by galaxy."""
    if all_tables_parquet is None or (not all_tables_parquet.exists()):
        return df
    try:
        t = pd.read_parquet(all_tables_parquet)
    except Exception:
        return df
    name_col = next((c for c in ('galaxy','Galaxy','name','Name','gal_name') if c in t.columns), None)
    if name_col is None or 'Vflat' not in t.columns:
        return df
    tt = t[[name_col,'Vflat']].rename(columns={name_col:'gal_src', 'Vflat':'Vflat_obs_kms'})
    tt['gal_src'] = tt['gal_src'].astype(str)
    tt['_norm'] = tt['gal_src'].map(_norm_name)
    left = df.copy()
    left['gal_id'] = left['gal_id'].astype(str)
    left['_norm'] = left['gal_id'].map(_norm_name)
    # exact raw then normalized
    j = left.merge(tt[['gal_src','Vflat_obs_kms']], left_on='gal_id', right_on='gal_src', how='left')
    need = j['Vflat_obs_kms'].isna()
    if need.any():
        idx_missing = j.index[need]
        left_sub = left.loc[idx_missing].reset_index().rename(columns={'index':'_idx'})
        j2 = left_sub.merge(tt[['_norm','Vflat_obs_kms']], on='_norm', how='left')
        ok = j2['Vflat_obs_kms'].notna()
        if ok.any():
            idxs = j2.loc[ok, '_idx'].to_numpy()
            j.loc[idxs, 'Vflat_obs_kms'] = j2.loc[ok, 'Vflat_obs_kms'].to_numpy()
    out = df.copy()
    out['Vflat_obs_kms'] = pd.to_numeric(j['Vflat_obs_kms'], errors='coerce')
    if audit_out is not None:
        a = pd.DataFrame({'gal_id': out['gal_id'], 'Vflat_obs_kms': out['Vflat_obs_kms']})
        a.to_csv(audit_out, index=False)
    return out


def attach_morph_type(df: pd.DataFrame, all_tables_parquet: Path|None) -> pd.DataFrame:
    """Attach morphological type T from sparc_all_tables.parquet by galaxy name normalization."""
    if all_tables_parquet is None or (not all_tables_parquet.exists()):
        return df
    try:
        t = pd.read_parquet(all_tables_parquet)
    except Exception:
        return df
    name_col = next((c for c in ('galaxy','Galaxy','name','Name','gal_name') if c in t.columns), None)
    if name_col is None or ('T' not in t.columns):
        return df
    tt = t[[name_col,'T']].rename(columns={name_col:'gal_src', 'T':'T_type'})
    tt['gal_src'] = tt['gal_src'].astype(str)
    tt['_norm'] = tt['gal_src'].map(_norm_name)
    left = df.copy()
    left['gal_id'] = left['gal_id'].astype(str)
    left['_norm'] = left['gal_id'].map(_norm_name)
    j = left.merge(tt[['gal_src','T_type']], left_on='gal_id', right_on='gal_src', how='left')
    need = j['T_type'].isna()
    if need.any():
        # Align boolean mask with left's index via j.index
        idx_missing = j.index[need]
        left_sub = left.loc[idx_missing].reset_index().rename(columns={'index':'_idx'})
        j2 = left_sub.merge(tt[['_norm','T_type']], on='_norm', how='left')
        ok = j2['T_type'].notna()
        if ok.any():
            idxs = j2.loc[ok, '_idx'].to_numpy()
            j.loc[idxs, 'T_type'] = j2.loc[ok, 'T_type'].to_numpy()
    out = df.copy()
    out['T_type'] = pd.to_numeric(j['T_type'], errors='coerce')
    return out


def btfr_outlier_report(btfr_obs_df: pd.DataFrame, out_csv: Path, alpha_canon: float = 4.0) -> None:
    """Top-25 residual outliers vs Mb ~ A * v^alpha_canon to highlight likely join issues."""
    d = btfr_obs_df.dropna(subset=['M_bary_Msun','vflat_obs_kms']).copy()
    d = d[(d['M_bary_Msun']>0) & (d['vflat_obs_kms']>0)]
    if d.empty:
        pd.DataFrame().to_csv(out_csv, index=False); return
    v = np.asarray(d['vflat_obs_kms'])
    Mb = np.asarray(d['M_bary_Msun'])
    A = np.median(Mb / np.power(v, alpha_canon))
    Mb_pred = A * np.power(v, alpha_canon)
    resid = np.log10(Mb) - np.log10(Mb_pred)
    d = d.assign(log10Mb=np.log10(Mb), log10v=np.log10(v), log10Mb_pred=np.log10(Mb_pred), resid_dex=resid)
    d = d.reindex(columns=['gal_id','vflat_obs_kms','M_bary_Msun','log10v','log10Mb','log10Mb_pred','resid_dex'])
    d = d.iloc[np.argsort(-np.abs(d['resid_dex']))][:25]
    d.to_csv(out_csv, index=False)


def suggest_name_overrides(outliers_csv: Path,
                           all_tables_parquet: Path,
                           out_csv: Path,
                           cutoff: float = 0.95) -> None:
    """
    For galaxies in btfr_join_outliers.csv, suggest a likely catalog name (gal_src)
    from sparc_all_tables.parquet using strict fuzzy matching. Writes a CSV:
      gal_id, suggestion_gal_src, similarity
    Review and copy confirmed rows into data/galaxy_name_overrides.csv.
    """
    if (not outliers_csv.exists()) or (not all_tables_parquet.exists()):
        pd.DataFrame().to_csv(out_csv, index=False); return
    try:
        outl = pd.read_csv(outliers_csv)
        t = pd.read_parquet(all_tables_parquet)
    except Exception:
        pd.DataFrame().to_csv(out_csv, index=False); return
    name_col = next((c for c in ('galaxy','Galaxy','name','Name','gal_name') if c in t.columns), None)
    if name_col is None:
        pd.DataFrame().to_csv(out_csv, index=False); return
    src_names = t[name_col].astype(str).tolist()
    sugg_rows = []
    for gid in outl['gal_id'].astype(str).unique():
        match = difflib.get_close_matches(gid, src_names, n=1, cutoff=cutoff)
        if match:
            sim = difflib.SequenceMatcher(None, gid, match[0]).ratio()
            sugg_rows.append({'gal_id': gid, 'suggestion_gal_src': match[0], 'similarity': sim})
    pd.DataFrame(sugg_rows, columns=['gal_id','suggestion_gal_src','similarity']).to_csv(out_csv, index=False)

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
    ap.add_argument('--cv_k', type=int, default=5, help='Number of CV folds (by galaxy)')
    ap.add_argument('--cv_seed', type=int, default=1337, help='Random seed for CV fold split')
    ap.add_argument('--sparc_master_parquet', default=str(Path('data')/'sparc_master_clean.parquet'))
    ap.add_argument('--mass_coupled_v0', action='store_true', help='Enable LogTail v0(Mb)=A*(Mb/1e10)^(1/4) grid search')
    ap.add_argument('--lensing_stack_csv', default=None, help='Optional CSV of stacked lensing to compare against')
    ap.add_argument('--suggest_overrides', action='store_true', help='Write galaxy_name_override_suggestions.csv derived from btfr_join_outliers.csv.')
    ap.add_argument('--enable_strict_fuzzy', action='store_true', help='Allow fuzzy matching (cutoff=0.95) for remaining mass joins after overrides.')
    ap.add_argument('--btfr_quick_only', action='store_true', help='Run only the mass join + observed BTFR + audits/suggestions, then exit.')
    ap.add_argument('--btfr_min_corr', type=float, default=None, help='If set, require Pearson corr(log vflat_obs, log Mb) >= this value; exit(2) otherwise.')
    ap.add_argument('--btfr_min_slope', type=float, default=None, help='If set, require CI lower bounds for alpha (Mb vs v) and beta (v vs Mb) to be >= this value; exit(2) otherwise.')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.pred_csv)
    df = normalize_columns(raw)
    # Attach baryonic masses using all_tables first, then fallback to master_clean; write audit CSV
    df = join_masses_enhanced(
        df,
        primary_parquet=Path('data')/'sparc_all_tables.parquet',
        fallback_parquet=Path(args.sparc_master_parquet) if args.sparc_master_parquet else None,
        overrides_csv=Path('data')/'galaxy_name_overrides.csv',
        allow_fuzzy=args.enable_strict_fuzzy,
        fuzzy_cutoff=0.95,
        audit_out=out_dir/'btfr_mass_join_audit.csv'
    )
    # Attach catalog Vflat if available for observed BTFR
    df = attach_observed_vflat(df, Path('data')/'sparc_all_tables.parquet', audit_out=out_dir/'btfr_vflat_catalog_audit.csv')

    # Optional quick BTFR-only flow for fast iteration on name overrides
    if args.btfr_quick_only:
        def vflat_obs_per_gal_quick(df_in: pd.DataFrame, frac_outer=0.3) -> pd.DataFrame:
            rows = []
            for gid, sub in df_in.groupby('gal_id'):
                sub = sub.sort_values('r_kpc')
                if 'Vflat_obs_kms' in sub.columns and np.isfinite(sub['Vflat_obs_kms']).any():
                    vflat = float(sub['Vflat_obs_kms'].dropna().iloc[0])
                else:
                    k = max(1, int(frac_outer*len(sub)))
                    vflat = float(np.median(sub['v_obs_kms'].tail(k).values))
                Mb = float(sub['M_bary_Msun'].iloc[0]) if 'M_bary_Msun' in sub.columns else float('nan')
                rows.append((gid, vflat, Mb))
            return pd.DataFrame(rows, columns=['gal_id','vflat_obs_kms','M_bary_Msun'])

        # Prefer MRT-derived observed table if present (catalog-of-record for names and Vflat)
        mrt_obs_path = out_dir/'btfr_observed.csv'
        if mrt_obs_path.exists():
            try:
                btfr_obs_quick = pd.read_csv(mrt_obs_path)
                # Basic sanity: ensure expected columns
                if not {'gal_id','Vflat_obs_kms'}.issubset(btfr_obs_quick.columns):
                    btfr_obs_quick = vflat_obs_per_gal_quick(df)
            except Exception:
                btfr_obs_quick = vflat_obs_per_gal_quick(df)
        else:
            btfr_obs_quick = vflat_obs_per_gal_quick(df)
        btfr_obs_quick.to_csv(out_dir/'btfr_observed.csv', index=False)
        # Handle either case of vflat column name
        vcol = 'vflat_obs_kms' if 'vflat_obs_kms' in btfr_obs_quick.columns else (
            'Vflat_obs_kms' if 'Vflat_obs_kms' in btfr_obs_quick.columns else None)
        df_fit = btfr_obs_quick.rename(columns={vcol: 'vflat_kms'}) if vcol else btfr_obs_quick
        fits_quick = btfr_fit_two_forms(df_fit)
        with open(out_dir/'btfr_observed_fit.json','w') as f:
            json.dump(fits_quick, f, indent=2)

        qc = btfr_obs_quick.dropna()
        corr = float('nan')
        if len(qc) > 5:
            vcol = 'vflat_obs_kms' if 'vflat_obs_kms' in qc.columns else (
                'Vflat_obs_kms' if 'Vflat_obs_kms' in qc.columns else None)
            if vcol:
                xv = np.log10(np.clip(qc[vcol].to_numpy(), 1e-6, None))
                yM = np.log10(np.clip(qc['M_bary_Msun'].to_numpy(), 1e-6, None))
                corr = float(np.corrcoef(xv, yM)[0,1])
        (out_dir/'btfr_qc.txt').write_text(f"Pearson corr(log vflat_obs, log Mb) = {corr:.3f}\n")

        # Outliers + suggestions to accelerate manual overrides
        try:
            btfr_outlier_report(btfr_obs_quick, out_dir/'btfr_join_outliers.csv', alpha_canon=4.0)
        except Exception:
            pass
        if args.suggest_overrides:
            try:
                suggest_name_overrides(out_dir/'btfr_join_outliers.csv', Path('data')/'sparc_all_tables.parquet', out_dir/'galaxy_name_override_suggestions.csv', cutoff=0.95)
            except Exception:
                pass

        # Optional guard rails for CI or scripted validation
        if (args.btfr_min_corr is not None) and (not (math.isfinite(corr) and corr >= args.btfr_min_corr)):
            print('BTFR quick pass failed minimum correlation threshold.', flush=True)
            raise SystemExit(2)
        if args.btfr_min_slope is not None:
            try:
                a_lo = float(fits_quick['form_Mb_vs_v']['alpha_CI95'][0])
                b_lo = float(fits_quick['form_v_vs_Mb']['beta_CI95'][0])
                if not (a_lo >= args.btfr_min_slope and b_lo >= args.btfr_min_slope):
                    print(f"BTFR quick pass failed slope CI guard: alpha_lo={a_lo:.3f}, beta_lo={b_lo:.3f} < {args.btfr_min_slope}", flush=True)
                    raise SystemExit(2)
            except Exception:
                print('BTFR quick pass: unable to evaluate slope CI guard.', flush=True)
                raise SystemExit(2)
        print('BTFR quick pass complete.', flush=True)
        raise SystemExit(0)

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

    # Optional K-fold CV by galaxy: fit on 4/5, report test medians per fold
    if getattr(args, 'do_cv', False):
        def kfold_ids(ids: list[str], k: int, seed: int):
            rng = np.random.default_rng(seed)
            idx = np.arange(len(ids))
            rng.shuffle(idx)
            folds = np.array_split(idx, k)
            return [ [ids[i] for i in fold] for fold in folds ]
        gal_ids = sorted(df['gal_id'].astype(str).unique().tolist())
        folds = kfold_ids(gal_ids, int(args.cv_k), int(args.cv_seed))
        cv_root = out_dir / 'cv'
        cv_root.mkdir(parents=True, exist_ok=True)
        cv_summary = []
        cv_rar_rows = []
        cv_by_T_rows = []
        for i, test_ids in enumerate(folds, start=1):
            test_mask = df['gal_id'].astype(str).isin(test_ids)
            train_mask = ~test_mask
            dtr = df.loc[train_mask]
            dte = df.loc[test_mask]
            # Fit on training only
            best_lt_tr, score_lt_tr = fit_global_grid(dtr, 'LogTail', grid_logtail, outer_only=True)
            best_mp_tr, score_mp_tr = fit_global_grid(dtr, 'MuPhi',   grid_muphi,   outer_only=True)
            # Apply to test
            dte_lt, col_lt_te = apply_model(dte, 'LogTail', best_lt_tr)
            dte_mp, col_mp_te = apply_model(dte_lt, 'MuPhi', best_mp_tr)
            outer_te = infer_outer_mask(dte_mp)
            med_lt_te = median_closeness(dte_mp.loc[outer_te,'v_obs_kms'], dte_mp.loc[outer_te, col_lt_te])
            med_mp_te = median_closeness(dte_mp.loc[outer_te,'v_obs_kms'], dte_mp.loc[outer_te, col_mp_te])
            # RAR OOS stats (test fold)
            rar_te = rar_table(dte_mp, col_lt_te)
            rar_obs_stats = rar_curved_stats(rar_te, ycol='g_obs')
            rar_mod_stats = rar_curved_stats(rar_te, ycol='g_mod')
            # Per-type medians in test fold
            dte_T = attach_morph_type(dte_mp, Path('data')/'sparc_all_tables.parquet')
            by_T = []
            for Tval, sub in dte_T.groupby('T_type'):
                if len(sub) < 5:
                    continue
                o = infer_outer_mask(sub)
                by_T.append({'fold': i,
                             'T_type': float(Tval) if pd.notna(Tval) else None,
                             'LogTail_test_median': float(median_closeness(sub.loc[o,'v_obs_kms'], sub.loc[o, col_lt_te])),
                             'MuPhi_test_median': float(median_closeness(sub.loc[o,'v_obs_kms'], sub.loc[o, col_mp_te])),
                             'n_points': int(len(sub))})
            # Write fold outputs
            fold_dir = cv_root / f'fold_{i}'
            fold_dir.mkdir(exist_ok=True)
            with open(fold_dir/'summary_logtail_muphi.json','w') as f:
                json.dump({
                    'fold': i,
                    'train': {'LogTail': {'median': score_lt_tr, 'params': best_lt_tr},
                              'MuPhi':   {'median': score_mp_tr, 'params': best_mp_tr}},
                    'test':  {'LogTail': {'median': float(med_lt_te)}, 'MuPhi': {'median': float(med_mp_te)}},
                    'rar_test': {'observed': rar_obs_stats, 'model': rar_mod_stats},
                    'n_train_points': int(len(dtr)), 'n_test_points': int(len(dte))
                }, f, indent=2)
            # Save per-type CV medians for this fold
            if by_T:
                pd.DataFrame(by_T).to_csv(fold_dir/'cv_by_T.csv', index=False)
                cv_by_T_rows.extend(by_T)
            # Aggregate summaries
            cv_summary.append({'fold': i, 'LogTail_train_median': score_lt_tr, 'LogTail_test_median': float(med_lt_te),
                               'MuPhi_train_median': score_mp_tr, 'MuPhi_test_median': float(med_mp_te),
                               'n_train_points': int(len(dtr)), 'n_test_points': int(len(dte))})
            cv_rar_rows.append({'fold': i,
                                'rar_obs_scatter_dex_orth': rar_obs_stats.get('scatter_dex_orth', None),
                                'rar_obs_r2_vs_const': rar_obs_stats.get('r2_vs_const', None),
                                'rar_model_scatter_dex_orth': rar_mod_stats.get('scatter_dex_orth', None),
                                'rar_model_r2_vs_const': rar_mod_stats.get('r2_vs_const', None)})
        pd.DataFrame(cv_summary).to_csv(cv_root/'cv_summary.csv', index=False)
        pd.DataFrame(cv_rar_rows).to_csv(cv_root/'cv_rar_summary.csv', index=False)
        if cv_by_T_rows:
            pd.DataFrame(cv_by_T_rows).to_csv(cv_root/'cv_by_T_summary.csv', index=False)

    outer = infer_outer_mask(df2)
    sum_lt = dict(model='LogTail', median=median_closeness(df2.loc[outer,'v_obs_kms'], df2.loc[outer, col_lt]), params=best_logtail)
    sum_mp = dict(model='MuPhi',   median=median_closeness(df2.loc[outer,'v_obs_kms'], df2.loc[outer, col_mp]), params=best_muphi)
    with open(out_dir/'summary_logtail_muphi.json', 'w') as f:
        json.dump({'LogTail': sum_lt, 'MuPhi': sum_mp}, f, indent=2)

    # BTFR and RAR exports (with masses attached if available)
    btfr_lt = v_flat_per_gal(df2, col_lt)
    btfr_mp = v_flat_per_gal(df2, col_mp)
    # If an observed BTFR table exists (from SPARC MRT), prefer its masses for model BTFRs
    def _find_btfr_obs(pred_csv_path: Path, out_dir_path: Path) -> Path|None:
        cand1 = out_dir_path/'btfr_observed.csv'
        cand2 = pred_csv_path.parent/'btfr_observed.csv'
        if cand1.exists():
            return cand1
        if cand2.exists():
            return cand2
        return None
    # Resolve pred_csv path if available
    try:
        pred_csv_path = Path(getattr(args, 'pred_csv', ''))
    except Exception:
        pred_csv_path = Path('')
    obs_path = _find_btfr_obs(pred_csv_path, out_dir)
    if obs_path is not None:
        try:
            obs_btfr = pd.read_csv(obs_path)
            obs_btfr = obs_btfr[['gal_id','M_bary_Msun']].dropna()
            # Name-normalized merge to handle suffixes like _rotmod
            def _add_norm(df_in, col='gal_id'):
                d = df_in.copy(); d[col] = d[col].astype(str); d['_norm'] = d[col].map(_norm_name); return d
            btfr_lt_n = _add_norm(btfr_lt, 'gal_id')
            btfr_mp_n = _add_norm(btfr_mp, 'gal_id')
            obs_n = _add_norm(obs_btfr, 'gal_id').rename(columns={'M_bary_Msun':'M_bary_Msun_obs'})
            btfr_lt = btfr_lt_n.drop(columns=['M_bary_Msun'], errors='ignore').merge(obs_n[['_norm','M_bary_Msun_obs']], on='_norm', how='left')
            btfr_mp = btfr_mp_n.drop(columns=['M_bary_Msun'], errors='ignore').merge(obs_n[['_norm','M_bary_Msun_obs']], on='_norm', how='left')
            btfr_lt = btfr_lt.rename(columns={'M_bary_Msun_obs':'M_bary_Msun'}).drop(columns=['_norm'], errors='ignore')
            btfr_mp = btfr_mp.rename(columns={'M_bary_Msun_obs':'M_bary_Msun'}).drop(columns=['_norm'], errors='ignore')
        except Exception:
            pass
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

    # Per-type RC medians and RAR stats
    df2_T = attach_morph_type(df2, Path('data')/'sparc_all_tables.parquet')
    outer_mask_all = infer_outer_mask(df2_T)
    rc_by_T = {}
    for Tval, sub in df2_T.groupby('T_type'):
        if len(sub) < 5:
            continue
        outer_sub = infer_outer_mask(sub)
        rc_by_T[str(Tval)] = {
            'LogTail_median': float(median_closeness(sub.loc[outer_sub,'v_obs_kms'], sub.loc[outer_sub, col_lt])),
            'MuPhi_median':   float(median_closeness(sub.loc[outer_sub,'v_obs_kms'], sub.loc[outer_sub, col_mp])),
            'n_points': int(len(sub))
        }
    with open(out_dir/'rc_medians_by_T.json','w') as f:
        json.dump(rc_by_T, f, indent=2)

    # RAR by type (observed and model)
    # Map T by galaxy to rar table
    gid_to_T = df2_T.dropna(subset=['T_type']).drop_duplicates('gal_id').set_index('gal_id')['T_type'].to_dict()
    rar_lt_T = rar_lt.copy()
    rar_lt_T['T_type'] = rar_lt_T['gal_id'].map(gid_to_T)
    rar_by_T_obs, rar_by_T_mod = {}, {}
    for Tval, sub in rar_lt_T.dropna(subset=['T_type']).groupby('T_type'):
        stats_obs = rar_curved_stats(sub, ycol='g_obs')
        stats_mod = rar_curved_stats(sub, ycol='g_mod')
        rar_by_T_obs[str(Tval)] = stats_obs
        rar_by_T_mod[str(Tval)] = stats_mod
    with open(out_dir/'rar_obs_curved_by_T.json','w') as f:
        json.dump(rar_by_T_obs, f, indent=2)
    with open(out_dir/'rar_logtail_curved_by_T.json','w') as f:
        json.dump(rar_by_T_mod, f, indent=2)

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

    # Observed BTFR (prefer SPARC-MRT catalog-of-record if present)
    def vflat_obs_per_gal(df_in: pd.DataFrame, frac_outer=0.3) -> pd.DataFrame:
        rows = []
        for gid, sub in df_in.groupby('gal_id'):
            sub = sub.sort_values('r_kpc')
            if 'Vflat_obs_kms' in sub.columns and np.isfinite(sub['Vflat_obs_kms']).any():
                vflat = float(sub['Vflat_obs_kms'].dropna().iloc[0])
            else:
                k = max(1, int(frac_outer*len(sub)))
                vflat = float(np.median(sub['v_obs_kms'].tail(k).values))
            Mb = float(sub['M_bary_Msun'].iloc[0]) if 'M_bary_Msun' in sub.columns else float('nan')
            rows.append((gid, vflat, Mb))
        return pd.DataFrame(rows, columns=['gal_id','vflat_obs_kms','M_bary_Msun'])

    def btfr_fit_table(df_btfr: pd.DataFrame, vcol: str):
        cols = set(df_btfr.columns)
        vname = vcol
        if vname not in cols:
            # try alternate capitalization
            if vcol == 'vflat_obs_kms' and 'Vflat_obs_kms' in cols:
                vname = 'Vflat_obs_kms'
            elif vcol == 'vflat_kms' and 'Vflat_kms' in cols:
                vname = 'Vflat_kms'
        dtmp = df_btfr.rename(columns={vname: 'vflat_kms'}) if vname in df_btfr.columns else df_btfr.copy()
        if 'vflat_kms' not in dtmp.columns:
            # As a last resort, pass through and let fit function return n:0
            return {'n': 0}
        return btfr_fit_two_forms(dtmp)

    obs_mrt = out_dir/'btfr_observed_from_mrt.csv'
    if obs_mrt.exists():
        try:
            btfr_obs = pd.read_csv(obs_mrt)
            # Ensure expected columns exist
            need_cols = {'gal_id','Vflat_obs_kms','M_bary_Msun'}
            if need_cols.issubset(btfr_obs.columns):
                # keep a copy as the QC file
                btfr_obs.to_csv(out_dir/'btfr_observed.csv', index=False)
            else:
                btfr_obs = vflat_obs_per_gal(df2)
                btfr_obs.to_csv(out_dir/'btfr_observed.csv', index=False)
        except Exception:
            btfr_obs = vflat_obs_per_gal(df2)
            btfr_obs.to_csv(out_dir/'btfr_observed.csv', index=False)
    else:
        btfr_obs = vflat_obs_per_gal(df2)
        btfr_obs.to_csv(out_dir/'btfr_observed.csv', index=False)

    with open(out_dir/'btfr_observed_fit.json','w') as f:
        json.dump(btfr_fit_table(btfr_obs, 'vflat_obs_kms'), f, indent=2)

    # Quick QC: correlation should be positive if join is correct
    qc = btfr_obs.dropna()
    if len(qc) > 5:
        vcol_obs = 'vflat_obs_kms' if 'vflat_obs_kms' in qc.columns else ('Vflat_obs_kms' if 'Vflat_obs_kms' in qc.columns else None)
        if vcol_obs is not None:
            xv = np.log10(np.clip(qc[vcol_obs].to_numpy(), 1e-6, None))
            yM = np.log10(np.clip(qc['M_bary_Msun'].to_numpy(), 1e-6, None))
            corr = float(np.corrcoef(xv, yM)[0,1])
            (out_dir/'btfr_qc.txt').write_text(f"Pearson corr(log vflat_obs, log Mb) = {corr:.3f}\n")
    # Outlier list to quickly fix joins
    try:
        btfr_outlier_report(btfr_obs, out_dir/'btfr_join_outliers.csv', alpha_canon=4.0)
    except Exception:
        pass

    # Optional: write override suggestions for review
    if args.suggest_overrides:
        try:
            suggest_name_overrides(out_dir/'btfr_join_outliers.csv', Path('data')/'sparc_all_tables.parquet', out_dir/'galaxy_name_override_suggestions.csv', cutoff=0.95)
        except Exception:
            pass

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
