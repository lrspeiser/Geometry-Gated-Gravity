#!/usr/bin/env python3
from __future__ import annotations
"""
Extract published Einstein radius (theta_E) and source redshift z from PDFs.

- Uses pdfminer.six to extract text.
- Searches for patterns like "Einstein radius", "effective Einstein radius",
  Greek thetaE (as words), and unit markers (arcsec/arcseconds/″).
- Captures nearby redshift mentions (z=..., z_s, zsource) around the match.
- Writes a JSONL summary and emits human-readable lines to stdout.

Usage:
  python scripts/extract_pdf_thetaE.py \
    C:\\Users\\henry\\dev\\GravityCalculator\\data\\1607.03462v3.pdf ...

Output files:
  out/papers_text/<basename>.txt   # extracted plain text for audit
  out/papers_text/extracted_thetaE.jsonl  # one JSON per match
"""
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from pdfminer.high_level import extract_text  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
OUTTXT = ROOT / 'out' / 'papers_text'
OUTTXT.mkdir(parents=True, exist_ok=True)

# Regexes
ARCSEC_RX = r"(?:arcsec|arcseconds|\u2033|\"|''|\u2032\u2032)"
NUM_RX = r"([0-9]{1,3}(?:\.[0-9]+)?)"
THETA_TERMS = [
    r"einstein\s*radius",
    r"effective\s*einstein\s*radius",
    r"equivalent\s*einstein\s*radius",
    r"einstein\s*radii",
    r"theta\s*e",
    r"\b\u03b8\s*e\b",
    r"critical\s*radius",
    r"tangential\s*critical\s*radius",
]
Z_RX = r"z\s*[=:\s]\s*([0-9]+(?:\.[0-9]+)?)"

NEAR = 220  # characters context window on each side

PATTERNS = [
    re.compile(fr"({term})[^\n]{{0,180}}?{NUM_RX}\s*{ARCSEC_RX}", re.I)
    for term in THETA_TERMS
]
ZPAT = re.compile(Z_RX, re.I)


def extract_from_text(text: str, pdf_path: Path) -> List[dict]:
    hits = []
    # Pass 1: targeted patterns (term + value + arcsec)
    for pat in PATTERNS:
        for m in pat.finditer(text):
            start, end = m.start(), m.end()
            context_start = max(0, start - NEAR)
            context_end = min(len(text), end + NEAR)
            context = text[context_start:context_end]
            try:
                theta_val = float(m.group(m.lastindex))
            except Exception:
                continue
            # Find redshifts near the match
            z_near: List[float] = []
            for zm in ZPAT.finditer(context):
                try:
                    z_near.append(float(zm.group(1)))
                except Exception:
                    pass
            hits.append({
                'file': str(pdf_path),
                'heuristic': 'pattern',
                'term': m.group(1),
                'theta_E_arcsec': theta_val,
                'z_candidates': z_near,
                'context': context.strip().replace('\n', ' '),
            })
    # Pass 2: fuzzy - look for any line containing 'Einstein' and an arcsec number nearby
    # Build a simpler arcsec-number regex
    arcnum = re.compile(fr"{NUM_RX}\s*{ARCSEC_RX}", re.I)
    einx = re.compile(r"einstein", re.I)
    critx = re.compile(r"tangential|critical\s*curve|critical\s*radius", re.I)
    # Scan sliding windows of text to capture fuzzy matches
    for m in einx.finditer(text):
        start = max(0, m.start() - NEAR)
        end = min(len(text), m.end() + NEAR)
        context = text[start:end]
        an = arcnum.search(context)
        if an:
            try:
                theta_val = float(an.group(1))
            except Exception:
                theta_val = None
            z_near: List[float] = []
            for zm in ZPAT.finditer(context):
                try:
                    z_near.append(float(zm.group(1)))
                except Exception:
                    pass
            if theta_val is not None:
                hits.append({
                    'file': str(pdf_path),
                    'heuristic': 'einstein+arcsec',
                    'term': 'einstein (fuzzy)',
                    'theta_E_arcsec': theta_val,
                    'z_candidates': z_near,
                    'context': context.strip().replace('\n', ' '),
                })
    # Pass 3: fuzzy - critical radius phrasing
    for m in critx.finditer(text):
        start = max(0, m.start() - NEAR)
        end = min(len(text), m.end() + NEAR)
        context = text[start:end]
        an = arcnum.search(context)
        if an:
            try:
                theta_val = float(an.group(1))
            except Exception:
                theta_val = None
            z_near: List[float] = []
            for zm in ZPAT.finditer(context):
                try:
                    z_near.append(float(zm.group(1)))
                except Exception:
                    pass
            if theta_val is not None:
                hits.append({
                    'file': str(pdf_path),
                    'heuristic': 'critical+arcsec',
                    'term': 'critical (fuzzy)',
                    'theta_E_arcsec': theta_val,
                    'z_candidates': z_near,
                    'context': context.strip().replace('\n', ' '),
                })
    return hits


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print('Usage: python scripts/extract_pdf_thetaE.py <pdf1> [pdf2 ...]')
        return 2
    out_jsonl = OUTTXT / 'extracted_thetaE.jsonl'
    if out_jsonl.exists():
        out_jsonl.unlink()
    total = 0
    for ap in argv[1:]:
        pdfp = Path(ap)
        if not pdfp.exists():
            print(f'SKIP (missing): {pdfp}')
            continue
        try:
            text = extract_text(str(pdfp))
        except Exception as e:
            print(f'ERROR reading {pdfp}: {e}')
            continue
        # Save audit text
        (OUTTXT / (pdfp.stem + '.txt')).write_text(text, encoding='utf-8', errors='ignore')
        # Extract matches
        hits = extract_from_text(text, pdfp)
        for h in hits:
            print(f"{pdfp.name}: term='{h['term']}' thetaE~{h['theta_E_arcsec']} arcsec; z candidates={h['z_candidates']}\n  ... {h['context'][:240]} ...")
            with open(out_jsonl, 'a', encoding='utf-8') as fj:
                fj.write(json.dumps(h) + '\n')
        total += len(hits)
    print(f'Found {total} candidate matches across {len(argv)-1} PDFs')
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
