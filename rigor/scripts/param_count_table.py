# -*- coding: utf-8 -*-
"""
rigor/scripts/param_count_table.py

Emit a simple Markdown table comparing global parameters of the one-law models
with per-object halo fits.

Usage:
py -u rigor/scripts/param_count_table.py \
  --n_gal 175 --n_cluster 2 --global_params 4 --halo_params_per_obj 2 --profile NFW

Outputs:
- out/param_count.md
"""
from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_gal', type=int, required=True)
    ap.add_argument('--n_cluster', type=int, required=True)
    ap.add_argument('--global_params', type=int, required=True, help='Total global parameters of the one-law model')
    ap.add_argument('--halo_params_per_obj', type=float, required=True, help='Parameters per object for the halo model (e.g., 2 for NFW M200,c)')
    ap.add_argument('--profile', default='NFW')
    ap.add_argument('--out', default='out/param_count.md')
    args = ap.parse_args()

    n_obj = int(args.n_gal + args.n_cluster)
    halos_total = args.halo_params_per_obj * n_obj

    md = []
    md.append('# Parameter Count Comparison\n')
    md.append('')
    md.append('| Model | Scope | Parameters |')
    md.append('|---|---:|---:|')
    md.append(f'| One-law (this work) | global | {args.global_params} |')
    md.append(f'| {args.profile} halos | per object | {args.halo_params_per_obj} × {n_obj} = {halos_total:.0f} |')
    md.append('')
    md.append(f'Total objects counted: {args.n_gal} galaxies + {args.n_cluster} clusters = {n_obj}.')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(md))
    print(f'Wrote {out}')

if __name__ == '__main__':
    main()
