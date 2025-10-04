#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import compute_cluster

if __name__ == "__main__":
    outdir = ROOT / 'out' / 'diag' / 'a209_beta002'
    outdir.parent.mkdir(parents=True, exist_ok=True)
    # A209 lens redshift 0.206; try beta=0.02 as per paper claim; z_source fixed 2.0
    summary = compute_cluster('ABELL_0209', 0.206, 2.0, outdir, beta=0.02, phi0_km2s2=1.0e4, generate_plots=False, debug=True)
    # Read back debug JSON and print
    dbg = json.loads((outdir / 'debug_realSigma.json').read_text(encoding='utf-8'))
    print('[diagnose] A209 debug:', json.dumps(dbg, indent=2))
