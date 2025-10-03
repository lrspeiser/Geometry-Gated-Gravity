#!/usr/bin/env python3
"""
Parse Umetsu (2016) VizieR tables and summarize available cluster entries.

This script scans data/clash/umetsu821_116/*.dat and ReadMe, extracts any cluster
identifier columns present, and writes a summary JSON with per-file headers and 
first few rows as a quick inspection artifact. It does NOT produce gas/stars profiles
(yet) because Umetsu (2016) primarily contains lensing mass measurements.

Outputs:
- data/clash/umetsu821_116/summary.json
- data/clash/umetsu821_116/preview_<table>.csv (head of parsed tables)
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / 'data' / 'clash' / 'umetsu821_116'

summary = {
    'files': [],
}

if not IN_DIR.exists():
    print(f"Input directory does not exist: {IN_DIR}")
    raise SystemExit(1)

# crude table parser: whitespace-separated, skip comment lines starting with # or ;
def parse_table(p: Path, max_rows: int = 10):
    rows = []
    header = None
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or line.lstrip().startswith(('#',';','!')):
                continue
            parts = line.split()
            # guess header: if first non-comment line contains any non-numeric tokens
            if header is None:
                # If all tokens numeric, fabricate column names
                if all(any(ch.isdigit() for ch in tok) and not any(c.isalpha() for c in tok) for tok in parts):
                    header = [f'col{i+1}' for i in range(len(parts))]
                    rows.append(parts)
                else:
                    header = parts
            else:
                rows.append(parts)
            if len(rows) >= max_rows:
                break
    return header or [], rows

for p in sorted(IN_DIR.glob('table*.dat')):
    cols, rows = parse_table(p)
    summary['files'].append({
        'name': p.name,
        'columns': cols,
        'preview_rows': rows,
    })
    # write a CSV preview for manual inspection
    out_csv = IN_DIR / f"preview_{p.stem}.csv"
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if cols: w.writerow(cols)
        for r in rows: w.writerow(r)

with (IN_DIR / 'summary.json').open('w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(f"Wrote summary for {len(summary['files'])} tables in {IN_DIR}")