# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np, pandas as pd, re, glob
from pathlib import Path

def _try_astropy_read(mrt_path: Path):
    try:
        from astropy.io import ascii
        return ascii.read(str(mrt_path)).to_pandas()
    except Exception:
        return None

def _read_mrt_fallback(mrt_path: Path) -> pd.DataFrame:
    # Fallback for AAS MRTs: skip byte-by-byte header, then read the whitespace-delimited table.
    # The SPARC .mrt file contains a long header with a byte-by-byte description and separator lines.
    # We look for the last long separator (----- or =====) and start reading on the next line.
    lines = mrt_path.read_text(errors='ignore').splitlines()
    # Find candidate separator lines and choose the last one
    sep_idxs = [i for i, ln in enumerate(lines) if re.match(r'^[\-=]{5,}\s*$', ln.strip())]
    start = (max(sep_idxs) + 1) if sep_idxs else 0
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[start:])),
                     sep=r"\s+", engine='python', comment='#', header=None, on_bad_lines='skip')
    # Best-effort column naming based on SPARC Table 1 description
    expected = ['Galaxy','T','D','e_D','f_D','Inc','e_Inc','L[3.6]','e_L[3.6]',
                'Reff','SBeff','Rdisk','SBdisk','MHI','RHI','Vflat','e_Vflat','Q','Ref.']
    n = min(len(expected), df.shape[1])
    cols = expected[:n] + [f'col{j}' for j in range(n, df.shape[1])]
    df.columns = cols
    return df

def read_sparc_catalog(mrt_path: Path) -> pd.DataFrame:
    """Return a catalog DataFrame with at least: Galaxy, Vflat (if present), L36, MHI, M_bary if present.

    Adds debug output to help diagnose column detection issues and avoids KeyError by
    filling missing expected columns with NaN.
    """
    df = _try_astropy_read(mrt_path)
    if df is None:
        df = _read_mrt_fallback(mrt_path)
    # Normalize column names to simple strings
    df.columns = [str(c).strip() for c in df.columns]

    def pick(df_local: pd.DataFrame, *patterns):
        for pat in patterns:
            for c in df_local.columns:
                if re.fullmatch(pat, c, flags=re.I):
                    return c
        return None

    gcol  = pick(df, r'Galaxy|Name')
    if gcol is None:
        # Try the rotmod master MRT as a fallback source of names
        alt = (mrt_path.parent / 'Rotmod_LTG' / 'MasterSheet_SPARC.mrt')
        if alt.exists():
            df_alt = _try_astropy_read(alt) or _read_mrt_fallback(alt)
            df_alt.columns = [str(c).strip() for c in df_alt.columns]
            gcol_alt = pick(df_alt, r'Galaxy|Name')
            if gcol_alt is not None:
                # Replace df with alt for name extraction, keep masses/vels from original where possible later
                df = df_alt
                gcol = gcol_alt

    # Pick columns from the (possibly updated) df
    vcol  = pick(df, r'Vflat|V_flat|VFLAT')
    mhib  = pick(df, r'MHI|M_HI')
    l36   = pick(df, r'L\[?3\.6\]?|L36|L_3\.6|L3p6|L\(3\.6\)')
    mbcol = pick(df, r'M[_ ]?bary|M_bary_Msun|Mbary')

    # Debug: show what we found
    try:
        print(f"[read_sparc_catalog] columns (first 25): {list(df.columns)[:25]} ... (n={len(df.columns)})")
        print(f"[read_sparc_catalog] picked gcol={gcol}, vcol={vcol}, mhib={mhib}, l36={l36}, mbcol={mbcol}")
    except Exception:
        pass

    keep = {}
    if gcol:
        keep['gal_id'] = df[gcol].astype(str)
    if vcol:
        keep['Vflat_obs_kms'] = pd.to_numeric(df[vcol], errors='coerce')
    if mhib:
        keep['MHI'] = pd.to_numeric(df[mhib], errors='coerce')
    if l36:
        keep['L36'] = pd.to_numeric(df[l36], errors='coerce')
    if mbcol:
        keep['M_bary_Msun'] = pd.to_numeric(df[mbcol], errors='coerce')

    out = pd.DataFrame(keep)

    # If M_bary is missing, synthesize M* + 1.33*MHI from L36 with a conservative M/L
    if 'M_bary_Msun' not in out.columns:
        if 'L36' in out.columns:
            L = out['L36'].to_numpy()
            # If looks like log10(L), exponentiate
            if np.nanmax(L) < 20.0:
                L = np.power(10.0, L)
            Mstar = 0.5 * L  # baseline SPARC M/L_3.6 ~ 0.5
        else:
            Mstar = np.full(len(out), np.nan)
        MHI = out['MHI'].to_numpy() if 'MHI' in out.columns else np.full(len(out), np.nan)
        Mb = np.where(np.isfinite(Mstar), Mstar, 0.0) + 1.33*np.where(np.isfinite(MHI), MHI, 0.0)
        out['M_bary_Msun'] = np.where(Mb>0, Mb, np.nan)

    # Ensure expected columns exist to avoid KeyError downstream
    for c in ('gal_id','Vflat_obs_kms','M_bary_Msun'):
        if c not in out.columns:
            out[c] = np.nan
            try:
                print(f"[read_sparc_catalog] WARNING: missing {c}; filled with NaN")
            except Exception:
                pass

    return out[['gal_id','Vflat_obs_kms','M_bary_Msun']].drop_duplicates('gal_id')

def read_rotmod_folder(rotmod_glob: str) -> pd.DataFrame:
    """Read per-galaxy Rotmod_LTG/*.rotmod.dat and build r_kpc, Vobs, Vgas, Vdisk, Vbul, Vbar."""
    rows = []
    for path in glob.glob(rotmod_glob):
        p = Path(path)
        gal = p.stem.replace('.rotmod','')
        try:
            dat = pd.read_csv(p, comment='#', delim_whitespace=True, header=None)
        except Exception:
            dat = pd.read_csv(p, comment='#', delim_whitespace=True)
        cols = [c.lower() for c in (dat.columns if isinstance(dat.columns[0], str) else [])]
        def pick_col(key, fallback_idx):
            for i, c in enumerate(cols):
                if re.fullmatch(key, c):
                    return dat.iloc[:, i]
            return dat.iloc[:, fallback_idx]
        if isinstance(dat.columns[0], str):
            R     = pick_col(r'(rad|r(_kpc)?)', 0)
            Vobs  = pick_col(r'(vobs)', 1)
            Vgas  = pick_col(r'(vgas)', 3 if dat.shape[1]>3 else 2)
            Vdisk = pick_col(r'(vdisk)', 4 if dat.shape[1]>4 else 3)
            Vbul  = pick_col(r'(vbul)', 5 if dat.shape[1]>5 else min(4, dat.shape[1]-1))
        else:
            R, Vobs, Vgas, Vdisk = dat.iloc[:,0], dat.iloc[:,1], dat.iloc[:,3], dat.iloc[:,4]
            Vbul = dat.iloc[:,5] if dat.shape[1] > 5 else 0.0
        R = pd.to_numeric(R, errors='coerce')
        Vobs = pd.to_numeric(Vobs, errors='coerce')
        Vgas = pd.to_numeric(Vgas, errors='coerce')
        Vdisk= pd.to_numeric(Vdisk, errors='coerce')
        Vbul = pd.to_numeric(Vbul, errors='coerce') if np.ndim(Vbul)!=0 else np.zeros_like(R)
        Vbar = np.sqrt(np.clip(Vgas**2 + Vdisk**2 + Vbul**2, 0, None))
        df_g = pd.DataFrame({'gal_id': gal, 'r_kpc': R, 'v_obs_kms': Vobs, 'vbar_kms': Vbar})
        rows.append(df_g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['gal_id','r_kpc','v_obs_kms','vbar_kms'])

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mrt', required=True, help='Path to data/SPARC_Lelli2016c.mrt')
    ap.add_argument('--rotmod_glob', default='data/Rotmod_LTG/*.rotmod.dat', help='Glob for per-galaxy rotmod files')
    ap.add_argument('--out_dir', required=True, help='Where to write btfr_observed.csv and sparc_predictions_by_radius.csv')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build BTFR observed table straight from SPARC MRT
    cat = read_sparc_catalog(Path(args.mrt))
    cat = cat[(cat['M_bary_Msun']>0) & (cat['M_bary_Msun'].notna())]
    cat.to_csv(out_dir/'btfr_observed_from_mrt.csv', index=False)

    # 2) Build predictions_by_radius-style table from rotmod files (for RC/RAR + outer medians)
    rc = read_rotmod_folder(args.rotmod_glob)
    rc = rc.dropna(subset=['r_kpc','v_obs_kms','vbar_kms'])
    rc.to_csv(out_dir/'sparc_predictions_by_radius.csv', index=False)

    # 3) If Vflat missing in MRT, patch it from outer-median of RC
    have_vflat = 'Vflat_obs_kms' in cat.columns and cat['Vflat_obs_kms'].notna()
    if have_vflat.any():
        cat[['gal_id','Vflat_obs_kms','M_bary_Msun']].to_csv(out_dir/'btfr_observed.csv', index=False)
    else:
        vflat_from_rc = []
        for gid, sub in rc.groupby('gal_id'):
            sub = sub.sort_values('r_kpc')
            k = max(1, int(0.3*len(sub)))
            vflat_from_rc.append((gid, float(np.median(sub['v_obs_kms'].tail(k)))))
        vtab = pd.DataFrame(vflat_from_rc, columns=['gal_id','Vflat_obs_kms'])
        cat2 = cat.drop(columns=['Vflat_obs_kms'], errors='ignore').merge(vtab, on='gal_id', how='left')
        cat2[['gal_id','Vflat_obs_kms','M_bary_Msun']].to_csv(out_dir/'btfr_observed.csv', index=False)

    print("Wrote:", out_dir/'btfr_observed.csv', "and", out_dir/'sparc_predictions_by_radius.csv')

if __name__ == '__main__':
    main()
