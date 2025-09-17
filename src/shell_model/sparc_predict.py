#!/usr/bin/env python3
"""
SPARC predictor: compute per-radius predicted speeds using a power-law G(M_bary) and compare to observed.

Inputs
- Rotmod LTG Parquet (rows across many galaxies)
  Default: data/sparc_rotmod_ltg.parquet
  Columns expected (any missing filled with 0):
    galaxy, R_kpc, Vobs_kms, [eVobs_kms], [Vgas_kms], [Vdisk_kms], [Vbul_kms]
- Mass CSV (required to compute G_pred): a table mapping galaxy -> M_bary (Msun)
  - Accepts columns: galaxy (or Name), and either M_bary (Msun) OR components Mstar_disk, Mstar_bulge, Mgas (Msun).
  - If components provided, M_bary is computed as: Mstar_disk + Mstar_bulge + 1.33 * Mgas (helium correction).
  - Optional column: type (morphological, e.g., LTG/ETG/other). If missing, defaults to LTG.
- Boundaries CSV (optional): galaxy,boundary_kpc
  If absent, a rule “boundary_frac * R_last” is used.

Outputs (written to --output-dir)
- sparc_predictions_by_radius.csv:
  galaxy, type, R_kpc, boundary_kpc, is_outer, Vobs_kms, Vbar_kms, G_pred, Vpred_kms, percent_close
- sparc_predictions_by_galaxy.csv:
  galaxy, type, M_bary_Msun, boundary_kpc, outer_points, median_percent_close, mean_percent_close

Power-law G(M_bary)
  G_pred = A * (M_bary / M0)^beta
  Default A and beta derived from our latest fit (excluding ultra-diffuse):
    A = 5.589712133866334
    beta = -0.6913962885091143

Usage (from repo root)
  python src/scripts/sparc_predict.py \
    --parquet data/sparc_rotmod_ltg.parquet \
    --mass-csv /path/to/sparc_mass_table.csv \
    --boundaries-csv data/boundaries.csv \
    --boundary-frac 0.5 \
    --output-dir data

Notes
- See README (section: SPARC workflow) for step-by-step instructions and references.
- Where to get SPARC masses? You can either provide a mass CSV (columns described below) or supply the SPARC_Lelli2016c.mrt file via --sparc-mrt and we will derive masses heuristically.
- Boundary logic: we only score percent_close for R_kpc >= boundary_kpc (outer region where modification applies).
- Vbar_kms is computed as sqrt(Vgas^2 + Vdisk^2 + Vbul^2) with missing components treated as 0.
- Vpred_kms = sqrt(G_pred) * Vbar_kms (scaling Newtonian baryon curve by sqrt of G factor).
"""
import argparse
from pathlib import Path
import math
import os
import pandas as pd
import numpy as np

A_DEFAULT = 3.1125315951391666
BETA_DEFAULT = 0.07417088846132323
M0_DEFAULT = 1e10  # Msun

# Project-relative defaults (no absolute paths). We locate the repo root assuming this file lives at src/scripts/.
# See README (SPARC workflow) for details.
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR: Path = repo_root() / 'data'


import re

def norm_name(s: str) -> str:
    """Canonicalize galaxy names across sources.
    Rules:
    - Uppercase
    - Remove spaces, hyphens, underscores
    - For common prefixes (NGC/UGC/IC/ESO/PGC), strip leading zeros in the numeric part
    - Keep alphanumerics only in the main token (retain letters+digits)
    """
    if s is None:
        return ''
    t = str(s).strip().upper()
    # Remove separators
    t = t.replace(' ', '').replace('-', '').replace('_', '')
    # Canonical for known catalog prefixes
    for pref in ('NGC', 'UGC', 'IC', 'ESO', 'PGC'):
        if t.startswith(pref):
            rest = t[len(pref):]
            # split numeric prefix
            m = re.match(r'^(0*)(\d+)(.*)$', rest)
            if m:
                zeros, digits, tail = m.groups()
                t = f"{pref}{int(digits)}{tail}"
            break
    # For remaining names, keep alphanumerics only (drop punctuation)
    t = re.sub(r'[^A-Z0-9]', '', t)
    return t


def load_masses(mass_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(mass_csv)
    # Normalize name
    name_col = 'galaxy' if 'galaxy' in df.columns else ('Name' if 'Name' in df.columns else None)
    if name_col is None:
        raise ValueError("Mass CSV must have a 'galaxy' or 'Name' column")
    df['galaxy'] = df[name_col].astype(str)
    # Get type if present
    if 'type' not in df.columns:
        # Try common alternatives
        tcol = None
        for cand in ['Type','Morph','morph','morph_type']:
            if cand in df.columns:
                tcol = cand; break
        df['type'] = df[tcol] if tcol else 'LTG'
    # Compute M_bary
    if 'M_bary' in df.columns:
        df['M_bary'] = df['M_bary'].astype(float)
    else:
        # Accept component columns
        for c in ['Mstar_disk','Mstar_bulge','Mgas']:
            if c not in df.columns:
                df[c] = 0.0
        df['M_bary'] = (df['Mstar_disk'].astype(float)
                        + df['Mstar_bulge'].astype(float)
                        + 1.33 * df['Mgas'].astype(float))
    # Keep necessary columns
    out = df[['galaxy','type','M_bary']].copy()
    out['galaxy_key'] = out['galaxy'].apply(norm_name)
    return out


def load_boundaries(boundaries_csv: Path | None, rotmod_df: pd.DataFrame, boundary_frac: float) -> pd.DataFrame:
    # Compute R_last per galaxy
    rlast = rotmod_df.groupby('galaxy', as_index=False)['R_kpc'].max().rename(columns={'R_kpc':'R_last_kpc'})
    if boundaries_csv is None:
        rlast['boundary_kpc'] = boundary_frac * rlast['R_last_kpc']
        return rlast[['galaxy','boundary_kpc']]
    b = pd.read_csv(boundaries_csv)
    if 'galaxy' not in b.columns or 'boundary_kpc' not in b.columns:
        raise ValueError("Boundaries CSV must have columns: galaxy,boundary_kpc")
    return b[['galaxy','boundary_kpc']]


def compute_G_enhanced(R_kpc, M_bary, boundary_kpc, shell_params: dict) -> float:
    """Shell-based gravity enhancement factor G(R).

    Concept:
    - Inner region (R < 0.5*boundary): G = 1.0 (Newtonian)
    - Middle region (0.5*boundary <= R < boundary): smooth cosine transition towards an enhanced value
    - Outer region (R >= boundary): gentle logarithmic growth (NO exterior cap applied).

    Note: We deliberately avoid any exterior caps on G in the outer region to prevent
    systematic underprediction. Inner clamp (G=1) and smooth transition are preserved.

    Parameters (from shell_params with defaults):
    - M_ref (default 1e10 Msun)
    - mass_exp (default -0.35) — negative exponent boosts lower-mass galaxies relatively more
    - inner_enhance (default 1.0)
    - middle_enhance (default 2.0)

    Returns
    - float G_pred at this radius. If inputs are invalid, returns NaN to match pipeline semantics for missing masses.

    See README (SPARC workflow: Shell model) for usage.
    """
    try:
        R = float(R_kpc)
        Mb = float(M_bary)
        B = float(boundary_kpc)
    except Exception:
        return float('nan')

    if not (np.isfinite(Mb) and Mb > 0.0):
        # Keep NaN to signal missing mass, aligning with existing pipeline behavior
        return float('nan')
    if not (np.isfinite(B) and B > 0.0):
        return float('nan')
    if not np.isfinite(R):
        return float('nan')

    M_ref = float(shell_params.get('M_ref', 1e10))
    mass_exp = float(shell_params.get('mass_exp', -0.35))
    inner_enhance = float(shell_params.get('inner_enhance', 1.0))
    middle_enhance = float(shell_params.get('middle_enhance', 2.0))
    # max_enhance removed (no exterior cap)
    _ = shell_params.get('max_enhance', None)

    if not (np.isfinite(M_ref) and M_ref > 0.0):
        M_ref = 1e10

    # Negative exponent => lower-mass galaxies get relatively more enhancement
    try:
        mass_factor = (Mb / M_ref) ** mass_exp
    except Exception:
        mass_factor = 1.0

    inner_boundary = 0.5 * B
    outer_boundary = B

    if R < inner_boundary:
        G = inner_enhance
    elif R < outer_boundary:
        denom = max(outer_boundary - inner_boundary, 1e-9)
        t = (R - inner_boundary) / denom
        t = min(max(t, 0.0), 1.0)
        smooth_t = 0.5 * (1.0 - math.cos(math.pi * t))
        G = inner_enhance + (middle_enhance - inner_enhance) * smooth_t * mass_factor
    else:
        excess = max((R - outer_boundary) / B, 0.0)
        growth = 1.0 + 0.5 * math.log10(1.0 + excess)
        # No exterior cap; allow gentle log growth
        G = middle_enhance * growth * mass_factor

    return float(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', type=Path, default=(repo_root() / 'data/sparc_rotmod_ltg.parquet'))
    ap.add_argument('--mass-csv', type=Path, required=False)
    ap.add_argument('--sparc-master-csv', type=Path, required=False,
                    help='Path to SPARC MasterSheet_SPARC.csv. If provided, derive masses and types from this CSV.')
    ap.add_argument('--sparc-mrt', type=Path, required=False,
                    help='Path to SPARC_Lelli2016c.mrt (fixed-width). Used to derive masses and Rd when mass-csv not given. See README.')
    ap.add_argument('--boundaries-csv', type=Path)
    ap.add_argument('--boundary-frac', type=float, default=0.5)
    ap.add_argument('--A', type=float, default=A_DEFAULT)
    ap.add_argument('--beta', type=float, default=BETA_DEFAULT)
    ap.add_argument('--M0', type=float, default=M0_DEFAULT)
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

def run(args):
    # Model banner
    model = getattr(args, 'model', 'shell')
    print(f"Reading rotmod parquet from: {args.parquet}")
    if getattr(args, 'mass_csv', None):
        print(f"Mass CSV: {args.mass_csv}")
    if getattr(args, 'sparc_master_csv', None):
        print(f"SPARC MasterSheet: {args.sparc_master_csv}")
    if getattr(args, 'sparc_mrt', None):
        print(f"SPARC MRT: {args.sparc_mrt}")
    if model == 'shell':
        print("Using shell-based gravity enhancement model (inner G=1, smooth transition, capped outer growth)")
    elif model == 'gr':
        print("Using GR baseline (no scaling): G=1 across all radii")
    else:
        raise SystemExit(f"Unsupported model '{model}'. Use 'shell' or 'gr'.")
    # Load rotmod parquet
    rot = pd.read_parquet(args.parquet)
    # Ensure expected columns
    for c in ['Vgas_kms','Vdisk_kms','Vbul_kms']:
        if c not in rot.columns:
            rot[c] = 0.0
    # Normalize galaxy names
    rot['galaxy'] = rot['galaxy'].astype(str)
    rot['galaxy_key'] = rot['galaxy'].apply(norm_name)

    # Optionally parse SPARC MRT to produce masses and Rd
    def parse_sparc_mrt(path: Path) -> pd.DataFrame:
        import pandas as _pd
        from astropy.io import ascii
        from astropy.table import Table
        import re as _re

        # Helpers: sanitize label, extract colspecs, alignment heuristic
        def _sanitize_label(label: str) -> str:
            lbl = label.strip().replace('[3.6]', '36').replace('Ref.', 'Ref')
            lbl = _re.sub(r'[^A-Za-z0-9_]+', '', lbl)
            if lbl == 'ID':
                return 'Galaxy'
            return lbl

        def _extract_colspecs_from_header(text: str):
            lines = text.splitlines()
            hdr_idx = None
            for i, line in enumerate(lines):
                if _re.search(r'\bBytes\b\s+Format\b', line):
                    hdr_idx = i
                    break
            if hdr_idx is None:
                return None
            spec_start = None
            for j in range(hdr_idx+1, min(hdr_idx+6, len(lines))):
                s = lines[j].strip('-= ')
                if _re.match(r'^[-=]{5,}$', s) or _re.match(r'^[-=]{5,}$', lines[j].strip()):
                    spec_start = j+1
                    break
            if spec_start is None:
                spec_start = hdr_idx + 1
            colspecs = []
            for k in range(spec_start, len(lines)):
                line = lines[k]
                if _re.match(r'^[-=]{5,}$', line.strip('-= ')):
                    break
                m = _re.match(r'^\s*(\d+)\s*-\s*(\d+)\s+\S+\s+\S+\s+([^\s].*?)\s{2,}.*$', line)
                if not m:
                    continue
                a, b, label = m.groups()
                name = _sanitize_label(label)
                colspecs.append((int(a)-1, int(b), name))
            return colspecs or None

        def _read_fwf(path: Path, colspecs):
            names = [c[2] for c in colspecs]
            specs = [(c[0], c[1]) for c in colspecs]
            return _pd.read_fwf(path, colspecs=specs, names=names, dtype=str, header=None)

        def _try_align_table1(path: Path, base_specs):
            # keep Galaxy fixed, shift others
            try:
                galaxy_idx = [i for i, (_, _, n) in enumerate(base_specs) if n == 'Galaxy'][0]
            except IndexError:
                galaxy_idx = 0
            best_df = _pd.DataFrame(); best_shift = 0; best_score = -1
            for shift in (-1, 0, 1, 2):
                shifted = []
                for i, (s, e, n) in enumerate(base_specs):
                    if i == galaxy_idx:
                        shifted.append((s, e, n))
                    else:
                        shifted.append((max(0, s+shift), max(0, e+shift), n))
                try:
                    df_try = _read_fwf(path, shifted)
                except Exception:
                    continue
                for c in ('T','D'):
                    if c in df_try.columns:
                        df_try[c] = _pd.to_numeric(df_try[c], errors='coerce')
                score = int(df_try['D'].notna().sum()) if 'D' in df_try.columns else 0
                if score > best_score:
                    best_score = score; best_df = df_try; best_shift = shift
            return best_df, best_shift

        # Derive colspecs from header and align
        text = Path(path).read_text(encoding='utf-8', errors='ignore')
        specs = _extract_colspecs_from_header(text)
        df = None
        used_shift = 0
        if specs is not None:
            try:
                df = _read_fwf(path, specs)
                # If looks like Table1, align if needed
                if {'T','D'}.issubset(set(df.columns)):
                    d_non = _pd.to_numeric(df['D'], errors='coerce').notna().sum()
                    if d_non < 50:
                        df, used_shift = _try_align_table1(path, specs)
            except Exception:
                df = None
        if df is None:
            # Fallback readers
            try:
                df = _read_fwf(path, [(0,11,'Galaxy'),(11,13,'T'),(13,19,'D'),(19,24,'e_D'),(24,26,'f_D'),(26,30,'Inc'),(30,34,'e_Inc'),(34,41,'L36'),(41,48,'e_L36'),(48,53,'Reff'),(53,61,'SBeff'),(61,66,'Rdisk'),(66,74,'SBdisk'),(74,81,'MHI'),(81,86,'RHI'),(86,91,'Vflat'),(91,96,'e_Vflat'),(96,99,'Q'),(99,113,'Ref')])
            except Exception:
                try:
                    t = ascii.read(path, format='mrt', guess=False)
                except Exception:
                    try:
                        t = ascii.read(path, format='fixed_width', guess=False)
                    except Exception:
                        t = Table.read(path)
                df = t.to_pandas()
                df.columns = [str(c).strip() for c in df.columns]
                if 'Galaxy' not in df.columns:
                    df.rename(columns={df.columns[0]:'Galaxy'}, inplace=True)
        # Sanity: coerce numeric and filter
        for c in ['T','D','e_D','f_D','Inc','e_Inc','L36','e_L36','Reff','SBeff','Rdisk','SBdisk','MHI','RHI','Vflat','e_Vflat','Q']:
            if c in df.columns:
                df[c] = _pd.to_numeric(df[c], errors='coerce')
        if 'T' in df.columns:
            df = df[df['T'].between(0, 11, inclusive='both')]
        if 'D' in df.columns:
            df = df[df['D'].notna()]
        out = _pd.DataFrame()
        out['galaxy'] = df['Galaxy'].astype(str).str.strip()
        out['galaxy_key'] = out['galaxy'].apply(norm_name)
        # Type if present
        tcol = None
        for cand in ['type','morph','morph_type']:
            if cand in df.columns:
                tcol = cand; break
        out['type'] = df[tcol] if tcol else 'LTG'
        # Disk scale Rd
        rd = None
        for cand in ['Rd_kpc','Rd','rd','r_d','rd_kpc','r_d_kpc']:
            if cand in df.columns:
                rd = df[cand]; break
        out['Rd_kpc'] = pd.to_numeric(rd, errors='coerce') if rd is not None else np.nan
        # Mass estimate from L36 & MHI in 1e9 units
        L36_col = 'L36' if 'L36' in df.columns else None
        MHI_col = 'MHI' if 'MHI' in df.columns else None
        Mb = None
        if L36_col is not None or MHI_col is not None:
            L36_1e9 = pd.to_numeric(df.get(L36_col, 0.0), errors='coerce') if L36_col else 0.0
            MHI_1e9 = pd.to_numeric(df.get(MHI_col, 0.0), errors='coerce') if MHI_col else 0.0
            Mstar = float(getattr(args, 'ml_star36', 0.5)) * L36_1e9 * 1e9
            Mgas = 1.33 * MHI_1e9 * 1e9
            Mb = Mstar + Mgas
        if Mb is None:
            # Fallbacks same as before
            for cand in ['m_bary','mbary','mbar','m_baryon']:
                if cand in df.columns:
                    Mb = pd.to_numeric(df[cand], errors='coerce'); break
        if Mb is None:
            Mstar = None; Mgas = None; Mbul = None
            for cand in ['mstar','m_star','mstar_disk','mstar_tot','mstar_kpc']:
                if cand in df.columns:
                    Mstar = pd.to_numeric(df[cand], errors='coerce'); break
            for cand in ['mbul','m_bulge','mstar_bulge']:
                if cand in df.columns:
                    Mbul = pd.to_numeric(df[cand], errors='coerce'); break
            for cand in ['mgas','m_gas']:
                if cand in df.columns:
                    Mgas = pd.to_numeric(df[cand], errors='coerce'); break
            if Mstar is None:
                for cand in ['logmstar','log_mstar','log10mstar']:
                    if cand in df.columns:
                        Mstar = np.power(10.0, pd.to_numeric(df[cand], errors='coerce')); break
            if Mbul is None:
                for cand in ['logmbul','log_mbul','logmbulge','log_mbulge']:
                    if cand in df.columns:
                        Mbul = np.power(10.0, pd.to_numeric(df[cand], errors='coerce')); break
            if Mgas is None:
                for cand in ['logmgas','log_mgas']:
                    if cand in df.columns:
                        Mgas = np.power(10.0, pd.to_numeric(df[cand], errors='coerce')); break
            zero = pd.Series(0.0, index=df.index)
            Mstar = Mstar if Mstar is not None else zero
            Mbul = Mbul if Mbul is not None else zero
            Mgas = Mgas if Mgas is not None else zero
            Mb = Mstar + Mbul + 1.33*Mgas
        out['M_bary'] = pd.to_numeric(Mb, errors='coerce')
        return out[['galaxy','galaxy_key','type','M_bary','Rd_kpc']]

    # Build a types_df (for Galaxy_Type) and masses_df (for M_bary). We allow combining sources.
    types_df = None
    rd_df = None
    masses_df = None

    # Preferred: use prebuilt Parquet with clean masses and T (from build_sparc_parquet.py)
    masses_parquet_default = repo_root() / 'data' / 'sparc_master_clean.parquet'
    masses_parquet = getattr(args, 'masses_parquet', None) or masses_parquet_default
    if masses_parquet and Path(masses_parquet).exists():
        clean = pd.read_parquet(masses_parquet)
        # Masses
        masses_df = clean[['galaxy_key','M_bary']].copy()
        # Rd_kpc from Rdisk if present
        if 'Rdisk' in clean.columns:
            rd_df = clean[['galaxy_key','Rdisk']].rename(columns={'Rdisk':'Rd_kpc'})
        # Types from T mapping
        T_TO_MORPH = {0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc', 5: 'Sc', 6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm', 10: 'Im', 11: 'BCD'}
        tnum = pd.to_numeric(clean.get('T', pd.Series([np.nan]*len(clean))), errors='coerce')
        morph = tnum.map(T_TO_MORPH)
        def morph_to_class(s: str) -> str:
            s = (s or '').strip()
            if s == 'S0':
                return 'lenticular'
            if s in ('Sa','Sab','Sb','Sbc','Sc','Scd','Sd','Sdm'):
                return 'spiral'
            if s == 'BCD':
                return 'dwarf BCD'
            if s in ('Sm','Im'):
                return 'dwarf irregular'
            return ''
        types_df = pd.DataFrame({
            'galaxy_key': clean['galaxy_key'],
            'type': morph.fillna('LTG').astype(str),
            'class': morph.fillna('').astype(str).apply(morph_to_class),
        })
        print(f"Masses and types loaded from Parquet: {masses_parquet}")
    else:
        # Types and Rd from MasterSheet if available
        master_path = None
        if getattr(args, 'sparc_master_csv', None):
            master_path = args.sparc_master_csv
        else:
            # Prefer .mrt master if present, else CSV
            cand_mrt = repo_root() / 'data/Rotmod_LTG/MasterSheet_SPARC.mrt'
            cand_csv = repo_root() / 'data/Rotmod_LTG/MasterSheet_SPARC.csv'
            if cand_mrt.exists():
                master_path = cand_mrt
            elif cand_csv.exists():
                master_path = cand_csv
        if master_path and Path(master_path).exists():
            def parse_sparc_master_csv(path: Path) -> pd.DataFrame:
                df = None
                if str(path).lower().endswith('.mrt'):
                    # Use MRT reader
                    from astropy.io import ascii as _ascii
                    try:
                        t = _ascii.read(path, format='mrt', guess=False)
                    except Exception:
                        t = _ascii.read(path, format='fixed_width', guess=False)
                    df = t.to_pandas()
                else:
                    # Robust CSV reader with fallbacks for odd delimiters/quotes
                    try:
                        df = pd.read_csv(path, comment='#')
                    except Exception:
                        try:
                            df = pd.read_csv(path, comment='#', engine='python', sep=None, on_bad_lines='warn')
                        except Exception:
                            df = pd.read_table(path, comment='#', delim_whitespace=True, engine='python')
                # Normalize name
                name_col = None
                for cand in ['galaxy','Galaxy','Name','name','objname']:
                    if cand in df.columns:
                        name_col = cand; break
                if name_col is None:
                    name_col = df.columns[0]
                out = pd.DataFrame()
                out['galaxy'] = df[name_col].astype(str)
                out['galaxy_key'] = out['galaxy'].apply(norm_name)
                # Morphological type: prefer explicit text columns, else map numeric T
                tcol_txt = None
                for cand in ['type','Type','Morph','morph','morph_type']:
                    if cand in df.columns:
                        tcol_txt = cand; break
                morph_text = None
                if tcol_txt is not None:
                    morph_text = df[tcol_txt].astype(str)
                else:
                    T_TO_MORPH = {0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc', 5: 'Sc', 6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm', 10: 'Im', 11: 'BCD'}
                    if 'T' in df.columns:
                        try:
                            tnum = pd.to_numeric(df['T'], errors='coerce')
                        except Exception:
                            tnum = pd.Series([np.nan]*len(df))
                        morph_text = tnum.map(T_TO_MORPH)
                out['type'] = morph_text if morph_text is not None else 'LTG'
                # High-level class from morphology
                def morph_to_class(s: str) -> str:
                    s = (s or '').strip()
                    if s == 'S0':
                        return 'lenticular'
                    if s in ('Sa','Sab','Sb','Sbc','Sc','Scd','Sd','Sdm'):
                        return 'spiral'
                    if s == 'BCD':
                        return 'dwarf BCD'
                    if s in ('Sm','Im'):
                        return 'dwarf irregular'
                    return ''
                out['class'] = out['type'].astype(str).apply(morph_to_class)
                # Rd if present
                rd = None
                for cand in ['Rd_kpc','Rd','rd','r_d','r_d_kpc']:
                    if cand in df.columns:
                        rd = df[cand]; break
                out['Rd_kpc'] = pd.to_numeric(rd, errors='coerce') if rd is not None else np.nan
                return out[['galaxy','galaxy_key','type','class','Rd_kpc']]
            types_df = parse_sparc_master_csv(Path(master_path))
            rd_df = types_df[['galaxy_key','Rd_kpc']]
            print(f"Galaxy types loaded from MasterSheet: {master_path}")

        # Masses from preferred source
        if args.mass_csv is not None and Path(args.mass_csv).exists():
            masses_df = load_masses(args.mass_csv)[['galaxy_key','M_bary']]
            print(f"Masses loaded from mass CSV: {args.mass_csv}")
        elif args.sparc_mrt and Path(args.sparc_mrt).exists():
            mrt_df = parse_sparc_mrt(args.sparc_mrt)
            masses_df = mrt_df[['galaxy_key','M_bary']]
            if rd_df is None:
                rd_df = mrt_df[['galaxy_key','Rd_kpc']]
            print(f"Masses derived from SPARC MRT: {args.sparc_mrt}")
        elif master_path and Path(master_path).exists():
            ms_df = parse_sparc_master_csv(Path(master_path))
            if 'M_bary' in ms_df.columns:
                masses_df = ms_df[['galaxy_key','M_bary']]
                if rd_df is None and 'Rd_kpc' in ms_df.columns:
                    rd_df = ms_df[['galaxy_key','Rd_kpc']]
                print("Masses taken from MasterSheet")
        else:
            raise SystemExit("Provide one of --masses-parquet, --mass-csv or --sparc-mrt, or ensure MasterSheet is present with masses.")

    if masses_df is None:
        print("WARNING: No masses found; G_pred and model outputs will be NaN where M_bary is missing.")

    # Missing masses report (by normalized key)
    rot_gal = rot[['galaxy','galaxy_key']].drop_duplicates()
    base_df = masses_df if masses_df is not None else pd.DataFrame({'galaxy_key':[]})
    missing = rot_gal.merge(base_df, on='galaxy_key', how='left', indicator=True)
    missing = missing[missing['_merge'] == 'left_only'][['galaxy','galaxy_key']]
    report_path = (args.output_dir if hasattr(args, 'output_dir') else DEFAULT_DATA_DIR) / 'sparc_missing_masses.csv'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(report_path, index=False)
    print(f"Wrote missing masses report: {report_path} ({len(missing)} unmatched)")

    # Build boundaries using hybrid rule if not provided
    # boundary = min(max(3.2*Rd, 0.4*R_last), 0.8*R_last); if Rd missing -> 0.5*R_last
    rlast = rot.groupby('galaxy', as_index=False)['R_kpc'].max().rename(columns={'R_kpc':'R_last_kpc'})
    rot_key = rot[['galaxy','galaxy_key']].drop_duplicates()
    rlast = rlast.merge(rot_key, on='galaxy', how='left')
    if args.boundaries_csv is None:
        rd_map = rd_df if rd_df is not None else pd.DataFrame({'galaxy_key':[], 'Rd_kpc':[]})
        bdf = rlast.merge(rd_map, on='galaxy_key', how='left')
        cand = np.maximum(3.2 * bdf['Rd_kpc'].fillna(0.0), 0.4 * bdf['R_last_kpc'])
        hybrid = np.minimum(cand, 0.8 * bdf['R_last_kpc'])
        fallback = 0.5 * bdf['R_last_kpc']
        boundary_kpc = np.where(np.isfinite(bdf['Rd_kpc']), hybrid, fallback)
        boundaries = pd.DataFrame({'galaxy': bdf['galaxy'], 'boundary_kpc': boundary_kpc})
    else:
        boundaries = load_boundaries(args.boundaries_csv, rot, args.boundary_frac)

    # Join type & mass (combine sources)
    df = rot.merge(masses_df, on='galaxy_key', how='left')
    if types_df is not None:
        df = df.merge(types_df[['galaxy_key','type','class']], on='galaxy_key', how='left')
    else:
        df['type'] = 'LTG'
        df['class'] = ''
    # Join boundaries by original name
    df = df.merge(boundaries, on='galaxy', how='left')

    # Compute baryonic velocity and predicted gravitational scaling
    vbar = np.sqrt(np.maximum(0.0, df['Vgas_kms'].fillna(0.0))**2
                   + np.maximum(0.0, df['Vdisk_kms'].fillna(0.0))**2
                   + np.maximum(0.0, df['Vbul_kms'].fillna(0.0))**2)
    df['Vbar_kms'] = vbar

    # Compute G_pred from selected model
    model = getattr(args, 'model', 'shell')
    if model == 'shell':
        shell_params = {
            'M_ref': float(getattr(args, 'shell_M_ref', 1e10)),
            'mass_exp': float(getattr(args, 'shell_mass_exp', -0.35)),
            'inner_enhance': 1.0,
            'middle_enhance': float(getattr(args, 'shell_middle', 2.0)),
            'max_enhance': float(getattr(args, 'shell_max', 5.0)),
        }
        print(
            f"Shell params: M_ref={shell_params['M_ref']:.2e}, mass_exp={shell_params['mass_exp']:.3f}, "
            f"inner=1.0, middle={shell_params['middle_enhance']:.2f}, max={shell_params['max_enhance']:.2f}"
        )
        # Row-wise computation to consume per-radius R and per-galaxy boundary and mass
        def _row_G(row):
            return compute_G_enhanced(row['R_kpc'], row.get('M_bary', np.nan), row.get('boundary_kpc', np.nan), shell_params)
        df['G_pred'] = df.apply(_row_G, axis=1)
    elif model == 'gr':
        # GR baseline: no scaling
        df['G_pred'] = 1.0
    else:
        raise SystemExit(f"Unsupported model '{model}'. Use 'shell' or 'gr'.")

    # Predicted speed assuming velocity scales with sqrt(G)
    df['Vpred_kms'] = np.sqrt(np.maximum(0.0, df['G_pred'])) * df['Vbar_kms']

    # GR predicted speed (pure baryons, no scaling)
    df['Vgr_kms'] = df['Vbar_kms']
    gr_pct = 100.0 * (1.0 - np.abs(df['Vgr_kms'] - df['Vobs_kms']) / df['Vobs_kms'].replace(0, np.nan))
    df['gr_percent_close'] = gr_pct

    # Required G per-star to match Vobs from baryon-only curve
    with np.errstate(divide='ignore', invalid='ignore'):
        df['G_required'] = (df['Vobs_kms'] / df['Vbar_kms'].replace(0, np.nan)) ** 2
        df['G_ratio'] = df['G_required'] / df['G_pred']

    # Outer-region mask and percent_close
    df['is_outer'] = df['R_kpc'] >= df['boundary_kpc'].fillna(np.inf)  # if no boundary, nothing is outer
    # percent close only for outer and where Vobs>0 and G_pred>0
    valid = (df['is_outer']) & (df['Vobs_kms'] > 0.0) & (df['Vpred_kms'] >= 0.0)
    pct = 100.0 * (1.0 - np.abs(df['Vpred_kms'] - df['Vobs_kms']) / df['Vobs_kms'].replace(0, np.nan))
    df['percent_close'] = np.where(valid, pct, np.nan)

    # Flag GR-failing stars in the outer region (below threshold percent close)
    thresh = float(getattr(args, 'gr_fail_thresh', 0.9)) * 100.0
    df['gr_failing'] = (df['is_outer']) & (df['gr_percent_close'] < thresh)

    # Write by-radius CSV
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    by_radius = df[['galaxy','type','R_kpc','boundary_kpc','is_outer','Vobs_kms','Vbar_kms','Vgr_kms','gr_percent_close','gr_failing','G_pred','G_required','G_ratio','Vpred_kms','percent_close']].copy()
    if getattr(args, 'model', 'shell') == 'shell':
        by_radius['Model'] = 'shell'
    elif getattr(args, 'model', 'shell') == 'gr':
        by_radius['Model'] = 'gr'
    by_radius_path = out_dir / 'sparc_predictions_by_radius.csv'
    by_radius.to_csv(by_radius_path, index=False)
    # Also write boundaries used for transparency (see README: SPARC workflow)
    boundaries_out = out_dir / 'boundaries.csv'
    boundaries.to_csv(boundaries_out, index=False)

    # Compute human-friendly columns (percent off instead of percent close)
    df['gr_percent_off'] = 100.0 * np.abs(df['Vgr_kms'] - df['Vobs_kms']) / df['Vobs_kms'].replace(0, np.nan)
    df['model_percent_off'] = 100.0 * np.abs(df['Vpred_kms'] - df['Vobs_kms']) / df['Vobs_kms'].replace(0, np.nan)

    # Galaxy-level summary on outer region
    def agg_grp(g: pd.DataFrame):
        g_outer = g[g['is_outer'] & g['percent_close'].notna()]
        g_fail = g[g['gr_failing'] & g['percent_close'].notna()]
        # Use outer region (physically relevant) for G diagnostics
        g_outer_all = g[g['is_outer']]
        return pd.Series({
            'type': g['type'].iloc[0] if len(g)>0 else 'LTG',
            'M_bary_Msun': g['M_bary'].iloc[0] if 'M_bary' in g.columns else np.nan,
            'boundary_kpc': g['boundary_kpc'].iloc[0] if 'boundary_kpc' in g.columns else np.nan,
            'outer_points': int(len(g_outer)),
            'median_percent_close': float(np.nanmedian(g_outer['percent_close'])) if len(g_outer)>0 else np.nan,
            'mean_percent_close': float(np.nanmean(g_outer['percent_close'])) if len(g_outer)>0 else np.nan,
            'avg_gr_percent_off': float(np.nanmean(g_outer['gr_percent_off'])) if len(g_outer)>0 else np.nan,
            'median_gr_percent_off': float(np.nanmedian(g_outer['gr_percent_off'])) if len(g_outer)>0 else np.nan,
            'avg_model_percent_off': float(np.nanmean(g_outer['model_percent_off'])) if len(g_outer)>0 else np.nan,
            'median_model_percent_off': float(np.nanmedian(g_outer['model_percent_off'])) if len(g_outer)>0 else np.nan,
            'gr_failing_points': int(len(g_fail)),
            'median_percent_close_on_gr_failing': float(np.nanmedian(g_fail['percent_close'])) if len(g_fail)>0 else np.nan,
            'mean_percent_close_on_gr_failing': float(np.nanmean(g_fail['percent_close'])) if len(g_fail)>0 else np.nan,
            # G diagnostics (outer region)
            'median_G_Required_Outer': float(np.nanmedian(g_outer_all['G_required'])) if len(g_outer_all)>0 else np.nan,
            'mean_G_Required_Outer': float(np.nanmean(g_outer_all['G_required'])) if len(g_outer_all)>0 else np.nan,
            'median_G_Ratio_Outer': float(np.nanmedian(g_outer_all['G_ratio'])) if len(g_outer_all)>0 else np.nan,
            'mean_G_Ratio_Outer': float(np.nanmean(g_outer_all['G_ratio'])) if len(g_outer_all)>0 else np.nan,
        })
    by_gal = df.groupby('galaxy').apply(agg_grp).reset_index()
    if getattr(args, 'model', 'shell') == 'shell':
        by_gal['Model'] = 'shell'
    elif getattr(args, 'model', 'shell') == 'gr':
        by_gal['Model'] = 'gr'
    by_gal_path = out_dir / 'sparc_predictions_by_galaxy.csv'
    by_gal.to_csv(by_gal_path, index=False)

    # Also write human-friendly per-radius and by-galaxy CSVs with descriptive headers
    human_by_radius = pd.DataFrame({
        'Galaxy': df['galaxy'],
        'Galaxy_Type': df['type'],
        'Galaxy_Class': df['class'],
        'Radius_kpc': df['R_kpc'],
        'Boundary_kpc': df['boundary_kpc'],
        'In_Outer_Region': df['is_outer'],
        'Observed_Speed_km_s': df['Vobs_kms'],
        'Baryonic_Speed_km_s': df['Vbar_kms'],
        'GR_Speed_km_s': df['Vgr_kms'],
        'GR_Percent_Off': df['gr_percent_off'],
        'G_Predicted': df['G_pred'],
        'Predicted_Speed_km_s': df['Vpred_kms'],
        'Model_Percent_Off': df['model_percent_off'],
        'Baryonic_Mass_Msun': df.get('M_bary', np.nan),
    })
    if getattr(args, 'model', 'shell') == 'shell':
        human_by_radius['Model'] = 'shell'
    elif getattr(args, 'model', 'shell') == 'gr':
        human_by_radius['Model'] = 'gr'
    human_by_radius_path = out_dir / 'sparc_human_by_radius.csv'
    human_by_radius.to_csv(human_by_radius_path, index=False)

    def mass_category(m):
        try:
            m = float(m)
        except Exception:
            return ''
        if not np.isfinite(m):
            return ''
        if m < 1e9:
            return 'ultra-light'
        if m < 1e10:
            return 'dwarf'
        if m < 1e11:
            return 'MW-like'
        return 'massive'

    # Merge back a representative class per galaxy
    class_map = df[['galaxy','class']].drop_duplicates().groupby('galaxy').first().reset_index()
    by_gal = by_gal.merge(class_map, on='galaxy', how='left')

    human_by_gal = pd.DataFrame({
        'Galaxy': by_gal['galaxy'],
        'Galaxy_Type': by_gal['type'],
        'Galaxy_Class': by_gal['class'],
        'Baryonic_Mass_Msun': by_gal['M_bary_Msun'],
        'Mass_Category': by_gal['M_bary_Msun'].apply(mass_category),
        'Boundary_kpc': by_gal['boundary_kpc'],
        'Outer_Points_Count': by_gal['outer_points'],
        'Avg_GR_Percent_Off': by_gal['avg_gr_percent_off'],
        'Median_GR_Percent_Off': by_gal['median_gr_percent_off'],
        'Avg_Model_Percent_Off': by_gal['avg_model_percent_off'],
        'Median_Model_Percent_Off': by_gal['median_model_percent_off'],
        'Median_G_Required_Outer': by_gal['median_G_Required_Outer'],
        'Mean_G_Required_Outer': by_gal['mean_G_Required_Outer'],
        'Median_G_Ratio_Outer': by_gal['median_G_Ratio_Outer'],
        'Mean_G_Ratio_Outer': by_gal['mean_G_Ratio_Outer'],
    })
    if getattr(args, 'model', 'shell') == 'shell':
        human_by_gal['Model'] = 'shell'
    elif getattr(args, 'model', 'shell') == 'gr':
        human_by_gal['Model'] = 'gr'
    human_by_gal_path = out_dir / 'sparc_human_by_galaxy.csv'
    human_by_gal.to_csv(human_by_gal_path, index=False)

    # Grouped summary by Galaxy_Type and Mass_Category
    grouped = human_by_gal.groupby(['Galaxy_Type','Galaxy_Class','Mass_Category'], dropna=False).agg(
        Galaxies=('Galaxy','count'),
        Avg_GR_Percent_Off=('Avg_GR_Percent_Off','mean'),
        Median_GR_Percent_Off=('Median_GR_Percent_Off','median'),
        Avg_Model_Percent_Off=('Avg_Model_Percent_Off','mean'),
        Median_Model_Percent_Off=('Median_Model_Percent_Off','median'),
        Median_G_Ratio_Outer=('Median_G_Ratio_Outer','median'),
        Mean_G_Ratio_Outer=('Mean_G_Ratio_Outer','mean'),
    ).reset_index()
    summary_by_type_path = out_dir / 'sparc_summary_by_type.csv'
    grouped.to_csv(summary_by_type_path, index=False)

    # Diagnostics: G_required vs M_bary (outer medians)
    diag = human_by_gal[['Galaxy','Galaxy_Type','Galaxy_Class','Baryonic_Mass_Msun','Median_G_Required_Outer','Mean_G_Required_Outer','Median_G_Ratio_Outer','Mean_G_Ratio_Outer']].copy()
    diag_path = out_dir / 'sparc_greq_vs_mass.csv'
    diag.to_csv(diag_path, index=False)

    # Simple fit: log10(G_required_med) vs log10(M_bary/M0)
    fit_txt = out_dir / 'sparc_greq_mass_fit.txt'
    try:
        x = np.log10(diag['Baryonic_Mass_Msun'] / float(M0_DEFAULT))
        y = np.log10(diag['Median_G_Required_Outer'])
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 2:
            beta_fit, logA_fit = np.polyfit(x[mask], y[mask], 1)
            A_fit = 10 ** logA_fit
            with open(fit_txt, 'w') as fh:
                fh.write(f'A_fit={A_fit}\n')
                fh.write(f'beta_fit={beta_fit}\n')
            print(f'Fitted G_required ~ A*(M/M0)^beta with A_fit={A_fit:.6g}, beta_fit={beta_fit:.6g}')
        else:
            with open(fit_txt, 'w') as fh:
                fh.write('Insufficient data for fit\n')
    except Exception as e:
        with open(fit_txt, 'w') as fh:
            fh.write(f'Fit failed: {e}\n')
        print(f'G_required vs mass fit failed: {e}')

    print(f"Wrote: {by_radius_path}")
    print(f"Wrote: {by_gal_path}")
    print(f"Wrote: {human_by_radius_path}")
    print(f"Wrote: {human_by_gal_path}")
    print(f"Wrote: {summary_by_type_path}")

    # Additional diagnostics for shell model (inner vs outer medians)
    if getattr(args, 'model', 'shell') == 'shell':
        try:
            inner_mask = df['R_kpc'] < 0.5 * df['boundary_kpc']
            outer_mask = df['R_kpc'] >= df['boundary_kpc']
            inner = df.loc[inner_mask]
            outer = df.loc[outer_mask]
            print(f"Inner median G_pred: {np.nanmedian(inner['G_pred']):.3f} (n={int(inner_mask.sum())})")
            print(
                "Outer medians - "
                f"G_required: {np.nanmedian(outer['G_required']):.3f}, "
                f"G_pred: {np.nanmedian(outer['G_pred']):.3f}, "
                f"G_ratio: {np.nanmedian(outer['G_ratio']):.3f}, "
                f"% close: {np.nanmedian(outer['percent_close']):.1f}"
            )
        except Exception as e:
            print(f"[warn] Shell diagnostics skipped: {e}")

    missing = by_gal['M_bary_Msun'].isna().sum()
    if missing:
        print(f"WARNING: {missing} galaxies missing M_bary (fill mass CSV and re-run)")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', type=Path, default=(repo_root() / 'data/sparc_rotmod_ltg.parquet'))
    ap.add_argument('--mass-csv', type=Path, required=False)
    ap.add_argument('--sparc-master-csv', type=Path, required=False,
                    help='Path to SPARC MasterSheet_SPARC.csv. If provided, derive masses and types from this CSV.')
    ap.add_argument('--sparc-mrt', type=Path, required=False,
                    help='Path to SPARC_Lelli2016c.mrt (fixed-width). Used to derive masses and Rd when mass-csv not given. See README.')
    ap.add_argument('--masses-parquet', type=Path, required=False,
                    help='Path to sparc_master_clean.parquet produced by build_sparc_parquet.py (preferred source for M_bary, T, and Rd).')
    ap.add_argument('--boundaries-csv', type=Path)
    ap.add_argument('--boundary-frac', type=float, default=0.5)

    # Model selection and parameters
    ap.add_argument('--model', choices=['shell','gr'], default='shell',
                    help='Select gravity model: shell (default) or gr (no scaling baseline).')
    # Shell-model params
    ap.add_argument('--shell-M-ref', type=float, default=1e10, help='Reference mass for shell model (Msun)')
    ap.add_argument('--shell-mass-exp', type=float, default=-0.35, help='Mass scaling exponent for shell model (negative boosts low-mass galaxies)')
    ap.add_argument('--shell-middle', type=float, default=2.0, help='Middle-shell peak enhancement factor')
    ap.add_argument('--shell-max', type=float, default=5.0, help='Maximum outer-shell enhancement')

    ap.add_argument('--ml-star36', type=float, default=0.5, help='Mass-to-light ratio (Msun/Lsun) at 3.6μm for stars')
    ap.add_argument('--gr-fail-thresh', type=float, default=0.9, help='Threshold for GR percent_close below which a star is considered GR-failing (e.g., 0.9 = 10%% off)')
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()
    run(args)
