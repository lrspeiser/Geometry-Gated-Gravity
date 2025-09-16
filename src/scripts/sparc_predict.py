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
import pandas as pd
import numpy as np

A_DEFAULT = 5.589712133866334
BETA_DEFAULT = -0.6913962885091143
M0_DEFAULT = 1e10  # Msun

# Project-relative defaults (no absolute paths). We locate the repo root assuming this file lives at src/scripts/.
# See README (SPARC workflow) for details.
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR: Path = repo_root() / 'data'


def norm_name(s: str) -> str:
    return str(s).strip().upper().replace(' ', '')


def load_masses(mass_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(mass_csv)
    # Normalize name
    name_col = 'galaxy' if 'galaxy' in df.columns else ('Name' if 'Name' in df.columns else None)
    if name_col is None:
        raise ValueError("Mass CSV must have a 'galaxy' or 'Name' column")
    df['galaxy'] = df[name_col].astype(str)
    # Get type if present
    if 'type' not in df.columns:
        df['type'] = 'LTG'
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', type=Path, default=(repo_root() / 'data/sparc_rotmod_ltg.parquet'))
    ap.add_argument('--mass-csv', type=Path, required=False)
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
    # Log the inverse mass-power law being used (beta < 0 indicates inverse correlation).
    print(f"Using inverse mass-power law: G_pred = A * (M_bary / M0)^beta with A={args.A}, beta={args.beta}, M0={args.M0}")
    print(f"Reading rotmod parquet from: {args.parquet}")
    if getattr(args, 'mass_csv', None):
        print(f"Mass CSV: {args.mass_csv}")
    if getattr(args, 'sparc_mrt', None):
        print(f"SPARC MRT: {args.sparc_mrt}")
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
        from astropy.io import ascii
        from astropy.table import Table
        # Try robust fixed-width first
        try:
            t = ascii.read(path, format='fixed_width', guess=False)
        except Exception:
            t = Table.read(path, format='ascii.cds', guess=False)
        df = t.to_pandas()
        # Normalize columns to str and lower
        cols_lc = {c: str(c).strip() for c in df.columns}
        df.columns = [cols_lc[c] for c in df.columns]
        # Candidate name columns
        name_col = None
        for cand in ['galaxy','name','gal','objname']:
            if cand in df.columns:
                name_col = cand; break
        if name_col is None:
            # fallback: first column as name
            name_col = df.columns[0]
        out = pd.DataFrame()
        out['galaxy'] = df[name_col].astype(str)
        out['galaxy_key'] = out['galaxy'].apply(norm_name)
        # Type if present
        tcol = None
        for cand in ['type','morph','morph_type']:
            if cand in df.columns:
                tcol = cand; break
        out['type'] = df[tcol] if tcol else 'LTG'
        # Disk scale Rd
        rd = None
        for cand in ['rd','r_d','rd_kpc','r_d_kpc']:
            if cand in df.columns:
                rd = df[cand]; break
        # Units may not be kpc; assume table provides kpc; if not present, NaN
        out['Rd_kpc'] = pd.to_numeric(rd, errors='coerce') if rd is not None else np.nan
        # Masses
        # Priority 1: use L[3.6] and MHI to estimate baryonic mass
        Mb = None
        # Identify possible column names for L36 and MHI
        def find_col(cands):
            for c in cands:
                if c in df.columns:
                    return c
            return None
        col_L36 = find_col(['L[3.6]','L_3.6','L36','L3.6'])
        col_MHI = find_col(['MHI','M_HI','MHI_1e9'])
        if col_L36 is not None or col_MHI is not None:
            L36_1e9 = pd.to_numeric(df[col_L36], errors='coerce') if col_L36 is not None else 0.0
            MHI_1e9 = pd.to_numeric(df[col_MHI], errors='coerce') if col_MHI is not None else 0.0
            # Msun
            Mstar = float(getattr(args, 'ml_star36', 0.5)) * L36_1e9 * 1e9
            Mgas = 1.33 * MHI_1e9 * 1e9
            Mb = Mstar + Mgas
        # Fallbacks: direct M_bary or component-based
        if Mb is None:
            for cand in ['m_bary','mbary','mbar','m_baryon']:
                if cand in df.columns:
                    Mb = pd.to_numeric(df[cand], errors='coerce'); break
        if Mb is None:
            # Components (look for linear or log10)
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
            # log variants
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
            # Combine
            zero = pd.Series(0.0, index=df.index)
            Mstar = Mstar if Mstar is not None else zero
            Mbul = Mbul if Mbul is not None else zero
            Mgas = Mgas if Mgas is not None else zero
            Mb = Mstar + Mbul + 1.33*Mgas
        out['M_bary'] = pd.to_numeric(Mb, errors='coerce')
        return out[['galaxy','galaxy_key','type','M_bary','Rd_kpc']]

    if args.mass_csv is not None:
        masses = load_masses(args.mass_csv)
    else:
        if not args.sparc_mrt:
            raise SystemExit("Provide either --mass-csv or --sparc-mrt for masses")
        masses = parse_sparc_mrt(args.sparc_mrt)

    # Build boundaries using hybrid rule if not provided
    # boundary = min(max(3.2*Rd, 0.4*R_last), 0.8*R_last); if Rd missing -> 0.5*R_last
    rlast = rot.groupby('galaxy', as_index=False)['R_kpc'].max().rename(columns={'R_kpc':'R_last_kpc'})
    rot_key = rot[['galaxy','galaxy_key']].drop_duplicates()
    rlast = rlast.merge(rot_key, on='galaxy', how='left')
    if args.boundaries_csv is None:
        rd_map = masses[['galaxy_key','Rd_kpc']]
        bdf = rlast.merge(rd_map, on='galaxy_key', how='left')
        cand = np.maximum(3.2 * bdf['Rd_kpc'].fillna(0.0), 0.4 * bdf['R_last_kpc'])
        hybrid = np.minimum(cand, 0.8 * bdf['R_last_kpc'])
        fallback = 0.5 * bdf['R_last_kpc']
        boundary_kpc = np.where(np.isfinite(bdf['Rd_kpc']), hybrid, fallback)
        boundaries = pd.DataFrame({'galaxy': bdf['galaxy'], 'boundary_kpc': boundary_kpc})
    else:
        boundaries = load_boundaries(args.boundaries_csv, rot, args.boundary_frac)

    # Join type & mass
    df = rot.merge(masses[['galaxy_key','M_bary','type']], on='galaxy_key', how='left')
    # Join boundaries by original name
    df = df.merge(boundaries, on='galaxy', how='left')

    # Compute baryonic velocity and predicted gravitational scaling
    vbar = np.sqrt(np.maximum(0.0, df['Vgas_kms'].fillna(0.0))**2
                   + np.maximum(0.0, df['Vdisk_kms'].fillna(0.0))**2
                   + np.maximum(0.0, df['Vbul_kms'].fillna(0.0))**2)
    df['Vbar_kms'] = vbar

    # Compute G_pred from mass (rows without mass will be NaN)
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = (df['M_bary'] / float(args.M0)) ** float(args.beta)
        df['G_pred'] = float(args.A) * scale

    # Predicted speed assuming velocity scales with sqrt(G)
    df['Vpred_kms'] = np.sqrt(np.maximum(0.0, df['G_pred'])) * df['Vbar_kms']

    # GR predicted speed (pure baryons, no scaling)
    df['Vgr_kms'] = df['Vbar_kms']
    gr_pct = 100.0 * (1.0 - np.abs(df['Vgr_kms'] - df['Vobs_kms']) / df['Vobs_kms'].replace(0, np.nan))
    df['gr_percent_close'] = gr_pct

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
    by_radius = df[['galaxy','type','R_kpc','boundary_kpc','is_outer','Vobs_kms','Vbar_kms','Vgr_kms','gr_percent_close','gr_failing','G_pred','Vpred_kms','percent_close']]
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
        })
    by_gal = df.groupby('galaxy').apply(agg_grp).reset_index()
    by_gal_path = out_dir / 'sparc_predictions_by_galaxy.csv'
    by_gal.to_csv(by_gal_path, index=False)

    # Also write human-friendly per-radius and by-galaxy CSVs with descriptive headers
    human_by_radius = pd.DataFrame({
        'Galaxy': df['galaxy'],
        'Galaxy_Type': df['type'],
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

    human_by_gal = pd.DataFrame({
        'Galaxy': by_gal['galaxy'],
        'Galaxy_Type': by_gal['type'],
        'Baryonic_Mass_Msun': by_gal['M_bary_Msun'],
        'Mass_Category': by_gal['M_bary_Msun'].apply(mass_category),
        'Boundary_kpc': by_gal['boundary_kpc'],
        'Outer_Points_Count': by_gal['outer_points'],
        'Avg_GR_Percent_Off': by_gal['avg_gr_percent_off'],
        'Median_GR_Percent_Off': by_gal['median_gr_percent_off'],
        'Avg_Model_Percent_Off': by_gal['avg_model_percent_off'],
        'Median_Model_Percent_Off': by_gal['median_model_percent_off'],
    })
    human_by_gal_path = out_dir / 'sparc_human_by_galaxy.csv'
    human_by_gal.to_csv(human_by_gal_path, index=False)

    print(f"Wrote: {by_radius_path}")
    print(f"Wrote: {by_gal_path}")
    print(f"Wrote: {human_by_radius_path}")
    print(f"Wrote: {human_by_gal_path}")
    missing = by_gal['M_bary_Msun'].isna().sum()
    if missing:
        print(f"WARNING: {missing} galaxies missing M_bary (fill mass CSV and re-run)")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', type=Path, default=(repo_root() / 'data/sparc_rotmod_ltg.parquet'))
    ap.add_argument('--mass-csv', type=Path, required=False)
    ap.add_argument('--sparc-mrt', type=Path, required=False,
                    help='Path to SPARC_Lelli2016c.mrt (fixed-width). Used to derive masses and Rd when mass-csv not given. See README.')
    ap.add_argument('--boundaries-csv', type=Path)
    ap.add_argument('--boundary-frac', type=float, default=0.5)
    ap.add_argument('--A', type=float, default=A_DEFAULT)
    ap.add_argument('--beta', type=float, default=BETA_DEFAULT)
    ap.add_argument('--M0', type=float, default=M0_DEFAULT)
    ap.add_argument('--ml-star36', type=float, default=0.5, help='Mass-to-light ratio (Msun/Lsun) at 3.6μm for stars')
    ap.add_argument('--gr-fail-thresh', type=float, default=0.9, help='Threshold for GR percent_close below which a star is considered GR-failing (e.g., 0.9 = 10% off)')
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()
    run(args)
