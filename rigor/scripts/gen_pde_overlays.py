#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import subprocess as sp
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

G3 = {
    'S0': '1.4e-4',
    'rc_kpc': '22',
    'rc_gamma': '0.5',
    'rc_ref_kpc': '30',
    'sigma_beta': '0.1',
    'sigma0_Msun_pc2': '150',
    'g0_kms2_per_kpc': '1200',
    'NR': '128',
    'NZ': '128',
}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEL_JSON = ROOT / 'out' / 'analysis' / 'type_breakdown' / 'sparc_example_galaxies.json'
OUTDIR = ROOT / 'root-m' / 'out' / 'pde_sparc_overlays'
RUN = ROOT / 'root-m' / 'pde' / 'run_sparc_pde.py'
ROT = ROOT / 'data' / 'sparc_rotmod_ltg.parquet'
ALL = ROOT / 'data' / 'sparc_all_tables.parquet'
SPARC_IN = ROOT / 'data' / 'sparc_predictions_by_radius.csv'


def run_pde_for_gal(gal: str, rmax_hint: float | None = None) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    Rmax = rmax_hint if rmax_hint is not None else 80.0
    cmd = [
        'py', '-u', str(RUN),
        '--in', str(SPARC_IN),
        '--axisym_maps', '--galaxy', gal,
        '--S0', G3['S0'], '--rc_kpc', G3['rc_kpc'],
        '--rc_gamma', G3['rc_gamma'], '--rc_ref_kpc', G3['rc_ref_kpc'],
        '--sigma_beta', G3['sigma_beta'], '--sigma0_Msun_pc2', G3['sigma0_Msun_pc2'],
        '--g0_kms2_per_kpc', G3['g0_kms2_per_kpc'],
        '--NR', G3['NR'], '--NZ', G3['NZ'],
        '--Rmax', f'{Rmax:.2f}', '--Zmax', f'{Rmax:.2f}',
        '--rotmod_parquet', str(ROT), '--all_tables_parquet', str(ALL),
        '--hz_kpc', '0.3', '--outdir', str(OUTDIR), '--tag', gal,
    ]
    print('[RUN]', ' '.join([str(x) for x in cmd]))
    sp.run(cmd, check=True)
    return OUTDIR / gal / 'rc_pde_predictions.csv'


def plot_overlays(gals: list[str], save_path: Path) -> None:
    n = len(gals)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0*ncols, 3.5*nrows), squeeze=False)

    for i, gal in enumerate(gals):
        r = i // ncols; c = i % ncols
        ax = axes[r][c]
        csv_path = OUTDIR / gal / 'rc_pde_predictions.csv'
        if not csv_path.exists():
            ax.text(0.5, 0.5, f'Missing: {csv_path.name}', transform=ax.transAxes, ha='center')
            continue
        df = pd.read_csv(csv_path)
        R = df['R_kpc'].to_numpy(float)
        Vobs = df['Vobs_kms'].to_numpy(float)
        Vbar = df['Vbar_kms'].to_numpy(float)
        Vpde = df['Vpred_pde_kms'].to_numpy(float)
        # Compute an outer mask similar to SPARC CV (>= 70th percentile of R)
        r_thr = float(np.percentile(R, 70.0))
        mask = (R >= r_thr)
        # Use percent_close if available; else derive
        if 'percent_close' in df.columns:
            pct = df['percent_close'].to_numpy(float)
            outer_pct = float(np.median(pct[mask]))
        else:
            err = 100.0 * np.abs(Vpde - Vobs) / np.maximum(Vobs, 1e-9)
            outer_pct = float(np.median(100.0 - err[mask]))

        ax.plot(R, Vobs, 'k.', ms=3, label='Observed')
        ax.plot(R, Vbar, 'c--', lw=1.2, label='GR (baryons)')
        ax.plot(R, Vpde, 'r-', lw=1.2, label='G³ (disk surrogate)')
        ax.set_title(gal)
        ax.set_xlabel('R [kpc]'); ax.set_ylabel('V [km/s]')
        ax.grid(True, alpha=0.3)
        # Stamp outer median closeness
        ax.text(0.02, 0.95, f'G³ outer ≈ {outer_pct:.1f}%', transform=ax.transAxes,
                ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
        if i == 0:
            ax.legend()

    # Hide any unused axes
    for j in range(i+1, nrows*ncols):
        r = j // ncols; c = j % ncols
        axes[r][c].axis('off')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gal_list', type=str, default=None,
                    help='Comma-separated list of galaxies to overlay; default uses example JSON list')
    ap.add_argument('--save', type=str, default=str(ROOT/'figs'/'rc_overlays_examples_v2.png'))
    args = ap.parse_args()

    if args.gal_list:
        names = [s.strip() for s in args.gal_list.split(',') if s.strip()]
        r_last = {}
    else:
        if DEFAULT_SEL_JSON.exists():
            sel = json.loads(DEFAULT_SEL_JSON.read_text())
            names = sel.get('galaxies', [])
            r_last = sel.get('r_last_kpc', {})
        else:
            names = []
            r_last = {}

    if not names:
        print('[WARN] No galaxies specified for overlays')
        return

    # Run PDE for each galaxy (axisymmetric builder)
    for g in names:
        r = float(r_last.get(g, 60.0))
        Rmax = max(60.0, 1.5*r)
        run_pde_for_gal(g, rmax_hint=Rmax)

    # Build the multi-panel overlays
    plot_overlays(names, Path(args.save))

if __name__ == '__main__':
    main()
    main()