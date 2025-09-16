#!/usr/bin/env python3
import sys
import re
from pathlib import Path
import pandas as pd

# NOTE: This test derives and auto-aligns colspecs from the MRT header rather than using hard-coded
# positions. This mirrors the production parser to avoid off-by-one drops. See README (SPARC workflow).

def norm_name(s: str) -> str:
    import re
    if s is None:
        return ''
    t = str(s).strip().upper().replace(' ', '').replace('-', '').replace('_', '')
    for pref in ('NGC','UGC','IC','ESO','PGC'):
        if t.startswith(pref):
            rest = t[len(pref):]
            m = re.match(r'^(0*)(\d+)(.*)$', rest)
            if m:
                zeros, digits, tail = m.groups()
                t = f"{pref}{int(digits)}{tail}"
            break
    t = re.sub(r'[^A-Z0-9]', '', t)
    return t


def sanitize_label(label: str) -> str:
    lbl = label.strip().replace('[3.6]', '36').replace('Ref.', 'Ref')
    lbl = re.sub(r'[^A-Za-z0-9_]+', '', lbl)
    if lbl == 'ID':
        return 'Galaxy'
    return lbl


def extract_colspecs_from_header(text: str):
    lines = text.splitlines()
    hdr_idx = None
    for i, line in enumerate(lines):
        if re.search(r'\bBytes\b\s+Format\b', line):
            hdr_idx = i
            break
    if hdr_idx is None:
        return None
    spec_start = None
    for j in range(hdr_idx+1, min(hdr_idx+6, len(lines))):
        s = lines[j].strip('-= ')
        if re.match(r'^[-=]{5,}$', s) or re.match(r'^[-=]{5,}$', lines[j].strip()):
            spec_start = j+1
            break
    if spec_start is None:
        spec_start = hdr_idx + 1
    colspecs = []
    for k in range(spec_start, len(lines)):
        line = lines[k]
        if re.match(r'^[-=]{5,}$', line.strip('-= ')):
            break
        m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s+\S+\s+\S+\s+([^\s].*?)\s{2,}.*$', line)
        if not m:
            continue
        a, b, label = m.groups()
        name = sanitize_label(label)
        colspecs.append((int(a)-1, int(b), name))
    return colspecs or None


def read_fixed_width(path: Path, colspecs):
    names = [c[2] for c in colspecs]
    specs = [(c[0], c[1]) for c in colspecs]
    return pd.read_fwf(path, colspecs=specs, names=names, dtype=str, header=None)


def try_align_table1(path: Path, base_specs):
    best_df = pd.DataFrame()
    best_shift = 0
    best_score = -1
    # keep Galaxy fixed
    try:
        galaxy_idx = [i for i, (_, _, n) in enumerate(base_specs) if n == 'Galaxy'][0]
    except IndexError:
        galaxy_idx = 0
    for shift in (-1, 0, 1, 2):
        shifted = []
        for i, (s, e, n) in enumerate(base_specs):
            if i == galaxy_idx:
                shifted.append((s, e, n))
            else:
                shifted.append((max(0, s+shift), max(0, e+shift), n))
        try:
            df_try = read_fixed_width(path, shifted)
        except Exception:
            continue
        for c in ('T','D'):
            if c in df_try.columns:
                df_try[c] = pd.to_numeric(df_try[c], errors='coerce')
        score = int(df_try['D'].notna().sum()) if 'D' in df_try.columns else 0
        if score > best_score:
            best_score = score
            best_df = df_try
            best_shift = shift
    return best_df, best_shift


def main():
    repo = Path(__file__).resolve().parents[1]
    data = repo / 'data'
    rot = pd.read_parquet(data/'sparc_rotmod_ltg.parquet')
    rot['galaxy'] = rot['galaxy'].astype(str)
    rot['rot_key'] = rot['galaxy'].apply(norm_name)

    # Derive MRT colspecs from header and auto-align
    mrt_path = data/'SPARC_Lelli2016c.mrt'
    text = mrt_path.read_text(encoding='utf-8', errors='ignore')
    specs = extract_colspecs_from_header(text)
    if specs is None:
        print('Failed to extract colspecs from MRT header', file=sys.stderr)
        sys.exit(1)
    df, used_shift = try_align_table1(mrt_path, specs)

    # Filter plausible rows by T and D
    if 'T' in df.columns:
        df = df[pd.to_numeric(df['T'], errors='coerce').between(0, 11, inclusive='both')]
    if 'D' in df.columns:
        df = df[pd.to_numeric(df['D'], errors='coerce').notna()]

    mrt = pd.DataFrame({'galaxy': df['Galaxy'].astype(str)})
    mrt['mrt_key'] = mrt['galaxy'].apply(norm_name)

    rot_keys = set(rot['rot_key'])
    mrt_keys = set(mrt['mrt_key'])
    common = len(rot_keys & mrt_keys)
    print(f'Rotmod galaxies: {rot["galaxy"].nunique()}  MRT rows: {len(mrt)}  Overlap: {common} (shift={used_shift})')
    # Require decent overlap
    if common < int(0.9 * rot['galaxy'].nunique()):
        # Print samples to diagnose
        miss_rot = rot[['galaxy','rot_key']].drop_duplicates()
        miss_rot = miss_rot[~miss_rot['rot_key'].isin(mrt_keys)].head(20)
        miss_mrt = mrt[['galaxy','mrt_key']][~mrt['mrt_key'].isin(rot_keys)].head(20)
        print('Sample rot-only (first 20):')
        print(miss_rot.to_string(index=False))
        print('Sample mrt-only (first 20):')
        print(miss_mrt.to_string(index=False))
        sys.exit(1)
    print('Name matching check passed.')

if __name__ == '__main__':
    main()
