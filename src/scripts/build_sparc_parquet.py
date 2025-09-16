#!/usr/bin/env python3
"""
Build SPARC Parquet files by parsing MRT (machine-readable) tables and combining
into clean dataframes without header/preamble rows.

Outputs:
- data/sparc_all_tables.parquet: union of parsed MRT tables with a source_file column
- data/sparc_master_clean.parquet: per-galaxy essentials + computed M_bary and galaxy_key

This script now auto-derives fixed-width column specs from each MRT file's
"Byte-by-byte" header when present, with robust fallbacks. It filters rows based
on expected numeric columns per table (e.g., T and D for Table1, or R/Vobs for
MassModels) to avoid dropping valid data lines.

Logging is verbose to aid diagnosis. If you change how SPARC MRTs are fetched,
see README.md (SPARC workflow) for details.
"""
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

# Legacy fixed colspecs for Table 1 (Galaxy Sample) as a fallback
# Byte ranges (1-indexed in docs) mapped to 0-indexed python slice bounds
FALLBACK_COLS_TABLE1: List[Tuple[int,int,str]] = [
    (0, 11, 'Galaxy'),
    (11, 13, 'T'),
    (13, 19, 'D'),
    (19, 24, 'e_D'),
    (24, 26, 'f_D'),
    (26, 30, 'Inc'),
    (30, 34, 'e_Inc'),
    (34, 41, 'L36'),
    (41, 48, 'e_L36'),
    (48, 53, 'Reff'),
    (53, 61, 'SBeff'),
    (61, 66, 'Rdisk'),
    (66, 74, 'SBdisk'),
    (74, 81, 'MHI'),
    (81, 86, 'RHI'),
    (86, 91, 'Vflat'),
    (91, 96, 'e_Vflat'),
    (96, 99, 'Q'),
    (99, 113, 'Ref'),
]

CAT_PREFIXES = ('NGC', 'UGC', 'IC', 'ESO', 'PGC')


def norm_name(s: str) -> str:
    if s is None:
        return ''
    t = str(s).strip().upper()
    t = t.replace(' ', '').replace('-', '').replace('_', '')
    for pref in CAT_PREFIXES:
        if t.startswith(pref):
            rest = t[len(pref):]
            m = re.match(r'^(0*)(\d+)(.*)$', rest)
            if m:
                _zeros, digits, tail = m.groups()
                t = f"{pref}{int(digits)}{tail}"
            break
    t = re.sub(r'[^A-Z0-9]', '', t)
    return t


def sanitize_label(label: str) -> str:
    # Map CDS-style labels to our canonical names used elsewhere
    lbl = label.strip()
    # Normalize e.g., 'L[3.6]' -> 'L36', 'e_L[3.6]' -> 'e_L36', 'Ref.' -> 'Ref'
    lbl = lbl.replace('[3.6]', '36').replace('[3.6', '36')
    lbl = lbl.replace('Ref.', 'Ref')
    # Remove any stray characters not alnum or underscore
    lbl = re.sub(r'[^A-Za-z0-9_]+', '', lbl)
    # Map specific alternate names
    if lbl == 'ID':
        return 'Galaxy'  # align MassModels 'ID' with other tables
    if lbl == 'L36' or lbl == 'L3_6':  # guard just in case
        return 'L36'
    if lbl == 'e_L36' or lbl == 'e_L3_6':
        return 'e_L36'
    return lbl


def extract_colspecs_from_header(text: str) -> Optional[List[Tuple[int,int,str]]]:
    """
    Parse a CDS/VizieR 'Byte-by-byte' header into zero-indexed fixed-width colspecs.
    Returns list of (start, end, name) or None if not found.
    """
    lines = text.splitlines()
    # Locate the header block
    # Find the line index for 'Bytes' header
    hdr_idx = None
    for i, line in enumerate(lines):
        if re.search(r'\bBytes\b\s+Format\b', line):
            hdr_idx = i
            break
    if hdr_idx is None:
        return None
    # From hdr_idx forward, skip the next dashed line if present
    spec_start = None
    for j in range(hdr_idx+1, min(hdr_idx+6, len(lines))):
        if re.match(r'^[-=]{5,}$', lines[j].strip('-= ')) or re.match(r'^[-=]{5,}$', lines[j].strip()):
            # the next line after this will be the first spec line
            spec_start = j + 1
            break
    if spec_start is None:
        spec_start = hdr_idx + 1
    # Read spec lines until a dashed line
    colspecs: List[Tuple[int,int,str]] = []
    for k in range(spec_start, len(lines)):
        line = lines[k]
        if re.match(r'^[-=]{5,}$', line.strip('-= ')):
            break
        m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s+\S+\s+\S+\s+([^\s].*?)\s{2,}.*$', line)
        if not m:
            # Sometimes there are blank or malformed lines in the block; skip them
            continue
        start_1, end_1, label = m.groups()
        start = int(start_1) - 1
        end = int(end_1)  # end-exclusive for pandas
        name = sanitize_label(label)
        colspecs.append((start, end, name))
    if not colspecs:
        return None
    # Ensure unique names (in case of duplicates)
    seen = {}
    uniq: List[Tuple[int,int,str]] = []
    for s, e, n in colspecs:
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        uniq.append((s, e, n))
    return uniq


def read_fixed_width(path: Path, colspecs: List[Tuple[int,int,str]]) -> pd.DataFrame:
    names = [c[2] for c in colspecs]
    specs = [(c[0], c[1]) for c in colspecs]
    # Read entire file; we'll filter out header lines afterwards
    df = pd.read_fwf(path, colspecs=specs, names=names, dtype=str, header=None)
    return df


def try_align_table1(path: Path, base_specs: List[Tuple[int,int,str]]) -> Tuple[pd.DataFrame, List[Tuple[int,int,str]], int]:
    """
    Table 1 alignment heuristic: try shifting all non-'Galaxy' columns by -1, 0, +1, +2
    and pick the shift that yields the most non-null D values among rows with plausible T.
    Returns (df, specs_used, shift).
    """
    best_df = pd.DataFrame()
    best_specs = base_specs
    best_shift = 0
    best_score = -1

    # Identify index of 'Galaxy' to keep fixed
    try:
        galaxy_idx = [i for i, (_, _, n) in enumerate(base_specs) if n == 'Galaxy'][0]
    except IndexError:
        galaxy_idx = 0

    for shift in (-1, 0, 1, 2):
        shifted: List[Tuple[int,int,str]] = []
        for i, (s, e, n) in enumerate(base_specs):
            if i == galaxy_idx:
                shifted.append((s, e, n))
            else:
                shifted.append((max(0, s + shift), max(0, e + shift), n))
        try:
            df_try = read_fixed_width(path, shifted)
        except Exception:
            continue
        # Coerce and score
        for c in ('T', 'D'):
            if c in df_try.columns:
                df_try[c] = pd.to_numeric(df_try[c], errors='coerce')
        mask_t = df_try['T'].between(0, 11, inclusive='both') if 'T' in df_try.columns else pd.Series([True]*len(df_try))
        score = int(df_try.loc[mask_t, 'D'].notna().sum()) if 'D' in df_try.columns else 0
        if score > best_score:
            best_score = score
            best_df = df_try
            best_specs = shifted
            best_shift = shift
    return best_df, best_specs, best_shift


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def parse_mrt_file(path: Path) -> pd.DataFrame:
    """
    Parse an MRT file using header-derived colspecs when possible, with robust fallbacks.
    Returns a DataFrame with at least ['galaxy','galaxy_key'] and any parsed columns.
    """
    text = path.read_text(encoding='utf-8', errors='ignore')
    colspecs = extract_colspecs_from_header(text)

    df: pd.DataFrame
    used_fallback = False

    if colspecs is not None:
        try:
            # Try header specs first
            df = read_fixed_width(path, colspecs)
            # If looks like Table1, attempt alignment heuristic if D is mostly NaN
            cols_tmp = set(df.columns)
            if {'T', 'D'}.issubset(cols_tmp):
                d_notna = pd.to_numeric(df['D'], errors='coerce').notna().sum()
                if d_notna < 50:  # heuristic threshold
                    df, colspecs, used_shift = try_align_table1(path, colspecs)
                    if used_shift != 0:
                        print(f"Adjusted Table1 alignment for {path.name} by shift={used_shift}.")
        except Exception as e:
            print(f"Header-derived read failed for {path.name}: {e}\nFalling back to legacy specs if available.")
            colspecs = None

    if colspecs is None:
        # Fallback for Table1-style files; also try alignment shifts
        base_specs = FALLBACK_COLS_TABLE1
        try:
            df0 = pd.read_fwf(
                path,
                colspecs=[(a, b) for a, b, _ in [(c[0], c[1]) for c in base_specs]],
                names=[c[2] for c in base_specs],
                dtype=str,
                header=None,
            )
            # Attempt alignment optimization
            df, colspecs, used_shift = try_align_table1(path, base_specs)
            used_fallback = True
            if used_shift != 0:
                print(f"Adjusted Table1 alignment (fallback) for {path.name} by shift={used_shift}.")
            else:
                # If no improvement, keep df0
                df = df0
        except Exception as e:
            print(f"Fallback read_fwf failed for {path.name}: {e}")
            return pd.DataFrame()

    # Identify table type by presence of key columns
    cols = set(df.columns)
    # Normalize column names (strip spaces/newlines)
    df.columns = [str(c).strip() for c in df.columns]

    # Heuristics: Table 1 has T and D; MassModels has R and Vobs and ID->Galaxy
    if {'T', 'D'}.issubset(cols) or 'T' in cols:
        # Coerce numeric
        df = coerce_numeric(df, ['T', 'D', 'L36', 'MHI', 'Rdisk', 'Reff', 'SBeff', 'SBdisk', 'RHI', 'Vflat', 'e_Vflat'])
        # Keep plausible data rows
        mask = (
            df.get('D').notna()
            & (
                (df.get('T').between(0, 11, inclusive='both'))
                if 'T' in df.columns else True
            )
        )
        df = df[mask]
        # Canonicalize
        if 'Galaxy' not in df.columns:
            # Sometimes sanitized label may not match; try common variants
            for alt in ['ID', 'Name', 'GalaxyName']:
                if alt in df.columns:
                    df = df.rename(columns={alt: 'Galaxy'})
                    break
        df['galaxy'] = df['Galaxy'].astype(str)
        # L36 and e_L36 renaming (if labels carried weird sanitation)
        if 'L36' not in df.columns:
            for alt in ['L36_1', 'L36_2', 'L3_6']:
                if alt in df.columns:
                    df = df.rename(columns={alt: 'L36'})
                    break
        if 'e_L36' not in df.columns:
            for alt in ['e_L36_1', 'e_L36_2', 'e_L3_6']:
                if alt in df.columns:
                    df = df.rename(columns={alt: 'e_L36'})
                    break
    else:
        # Treat as MassModels-style table
        # Ensure 'Galaxy' column exists
        if 'Galaxy' not in df.columns:
            for alt in ['ID', 'Id', 'id']:
                if alt in df.columns:
                    df = df.rename(columns={alt: 'Galaxy'})
                    break
        df = coerce_numeric(df, ['D', 'R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
        mask = df.get('R').notna() & df.get('Vobs').notna()
        df = df[mask]
        df['galaxy'] = df['Galaxy'].astype(str)

    # Normalized key
    df['galaxy_key'] = df['galaxy'].apply(norm_name)

    # Compute baryonic mass if components present
    if 'L36' in df.columns and 'MHI' in df.columns:
        # Units per header: L36 and MHI are in 1e9 solar units
        df['M_bary'] = 0.5 * df['L36'].astype(float) * 1e9 + 1.33 * df['MHI'].astype(float) * 1e9

    # Trim rows with empty keys
    df = df[df['galaxy_key'] != '']

    # Diagnostics
    pre_rows = len(df)
    unique_keys = df['galaxy_key'].nunique() if 'galaxy_key' in df.columns else 0
    print(f"Parsed {pre_rows} rows ({unique_keys} unique keys) from {path.name}{' [fallback]' if used_fallback else ''}.")

    return df


def main():
    repo = Path(__file__).resolve().parents[2]
    data_dir = repo / 'data'
    out_all = data_dir / 'sparc_all_tables.parquet'
    out_clean = data_dir / 'sparc_master_clean.parquet'

    # Candidate MRT files to union
    mrt_files = [
        data_dir / 'SPARC_Lelli2016c.mrt',
        data_dir / 'Rotmod_LTG' / 'MasterSheet_SPARC.mrt',
        data_dir / 'Rotmod_LTG' / 'MassModels_Lelli2016c.mrt',
    ]
    mrt_files = [p for p in mrt_files if p.exists()]
    if not mrt_files:
        print('No MRT files found to parse under data/.')
        sys.exit(1)

    tables = []
    for p in mrt_files:
        df = parse_mrt_file(p)
        if df.empty:
            print(f'WARNING: parsed 0 rows from {p.name}')
        df['source_file'] = p.name
        tables.append(df)

    all_df = pd.concat(tables, axis=0, ignore_index=True, sort=False)
    # Drop any rows with empty galaxy_key (already filtered, but double-guard)
    if 'galaxy_key' in all_df.columns:
        all_df = all_df[all_df['galaxy_key'].astype(str) != '']
    # Write
    all_df.to_parquet(out_all, index=False)
    print(f'Wrote {out_all} with {len(all_df)} rows and {len(all_df.columns)} cols')

    # Master clean subset: one row per galaxy (prefer the first occurrence)
    keep_cols = ['galaxy', 'galaxy_key', 'T', 'D', 'L36', 'MHI', 'Rdisk', 'M_bary', 'source_file']
    clean = all_df.copy()
    for c in keep_cols:
        if c not in clean.columns:
            clean[c] = pd.NA
    clean = clean[keep_cols]
    clean = clean.groupby('galaxy_key', as_index=False).first()
    # Save
    clean.to_parquet(out_clean, index=False)
    print(f'Wrote {out_clean} with {len(clean)} galaxies')

if __name__ == '__main__':
    main()
