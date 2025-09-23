#!/usr/bin/env python3
from __future__ import annotations
import json
import math
import subprocess as sp
from pathlib import Path

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
SEL_JSON = ROOT / 'out' / 'analysis' / 'type_breakdown' / 'sparc_example_galaxies.json'
OUTDIR = ROOT / 'root-m' / 'out' / 'pde_sparc_overlays'
RUN = ROOT / 'root-m' / 'pde' / 'run_sparc_pde.py'
ROT = ROOT / 'data' / 'sparc_rotmod_ltg.parquet'
ALL = ROOT / 'data' / 'sparc_all_tables.parquet'


def main():
    sel = json.loads(SEL_JSON.read_text())
    names = sel.get('galaxies', [])
    r_last = sel.get('r_last_kpc', {})
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for g in names:
        r = float(r_last.get(g, 60.0))
        Rmax = max(60.0, 1.5*r)
        cmd = [
            'py', '-u', str(RUN),
            '--in', str(ROOT / 'data' / 'sparc_predictions_by_radius.csv'),
            '--axisym_maps', '--galaxy', g,
            '--S0', G3['S0'], '--rc_kpc', G3['rc_kpc'],
            '--rc_gamma', G3['rc_gamma'], '--rc_ref_kpc', G3['rc_ref_kpc'],
            '--sigma_beta', G3['sigma_beta'], '--sigma0_Msun_pc2', G3['sigma0_Msun_pc2'],
            '--g0_kms2_per_kpc', G3['g0_kms2_per_kpc'],
            '--NR', G3['NR'], '--NZ', G3['NZ'],
            '--Rmax', f'{Rmax:.2f}', '--Zmax', f'{Rmax:.2f}',
            '--rotmod_parquet', str(ROT), '--all_tables_parquet', str(ALL),
            '--hz_kpc', '0.3', '--outdir', str(OUTDIR), '--tag', g,
        ]
        print('[RUN]', ' '.join(cmd))
        sp.run(cmd, check=True)

if __name__ == '__main__':
    main()