# -*- coding: utf-8 -*-
"""
rigor/scripts/summarize_sparc_cv.py

Read root-m/out/pde_sparc/cv_summary.json (or a provided path) and print a
one-liner headline suitable for paper.md updates.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv_json', default='root-m/out/pde_sparc/cv_summary.json')
    args = ap.parse_args()
    p = Path(args.cv_json)
    if not p.exists():
        raise SystemExit(f'cv_summary.json not found at {p}')
    data = json.loads(p.read_text())
    best = (data or {}).get('best')
    if not best:
        raise SystemExit('No best entry in cv_summary.json')
    cv_med = best.get('cv_median')
    print(f"G³ (axisym) NR=128 CV median = {cv_med:.1f}%")


if __name__ == '__main__':
    main()
