#!/usr/bin/env python3
"""
Holdout validation (paper variant) using NPZ posterior from PyMC conversion.
- Reads cluster_selection_paper.yaml to select holdouts.
- Calls existing validate_holdout_mass_scaled.py with correct args.
"""
import sys, subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
SG = ROOT / 'projects' / 'SigmaGravity'

SEL = SG / 'config' / 'cluster_selection_paper.yaml'
SET = SG / 'config' / 'paper_settings.yaml'
OUT_CAL = SG / 'output' / 'pymc_mass_scaled'


def main():
    sel = yaml.safe_load(SEL.read_text())
    settings = yaml.safe_load(SET.read_text())

    holdouts = ','.join(sel[sel.get('variant','current_catalog')]['holdout_names'])
    posterior = OUT_CAL / 'flat_samples_from_pymc.npz'
    catalog = ROOT / settings['clusters']['catalog_path']

    cmd = [sys.executable, str(ROOT / 'scripts' / 'validate_holdout_mass_scaled.py'),
           '--posterior', str(posterior), '--catalog', str(catalog), '--clusters', holdouts,
           '--pzs', settings['source_redshift']['mode'], '--outdir', str(SG / 'output' / 'holdout_paper')]
    print('>>>', ' '.join(cmd))
    rc = subprocess.call(cmd)
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
