# -*- coding: utf-8 -*-
"""
rigor/scripts/lensing_stacks_compare.py

Compare predicted DeltaSigma(R) stacks from models to observed stacked
measurements. This script expects observed stacks in CSVs and a directory of
PDE field outputs per galaxy or a pre-aggregated model stack. Assembling a true
model stack across many galaxies requires per-galaxy predicted fields and a
stacking scheme (weights, selection), which is project-specific. Therefore this
helper provides scaffolding and will emit a clear message if required model
inputs are missing.

Inputs:
- --stacks_glob: glob for observed stacks CSVs with columns R_kpc,DeltaSigma_obs,DeltaSigma_err
- --pred_csv: data/sparc_predictions_by_radius.csv (for sample definition)
- --pde_field_dir: directory where per-galaxy PDE field summaries live (optional)
- --out_dir: output directory

Outputs:
- For each observed stack, creates an overlay PNG if a corresponding model stack
  is found, else writes a TODO note in out_dir.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stacks_glob', required=True)
    ap.add_argument('--pred_csv', default='data/sparc_predictions_by_radius.csv')
    ap.add_argument('--pde_field_dir', default='root-m/out/pde_sparc_axisym_full')
    ap.add_argument('--out_dir', default='out/lensing_stacks')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stack_files = sorted(glob.glob(args.stacks_glob))
    if not stack_files:
        raise SystemExit(f'No stacks matched {args.stacks_glob}')

    # Currently we do not have a precomputed model stack; emit TODO notes
    for s in stack_files:
        name = Path(s).stem
        (out_dir/f'{name}_TODO.txt').write_text(
            'Model stacking requires per-galaxy PDE field predictions and a stacking scheme.\n'
            'Provide a precomputed model stack or per-galaxy fields and selection to proceed.'
        )


if __name__ == '__main__':
    main()
