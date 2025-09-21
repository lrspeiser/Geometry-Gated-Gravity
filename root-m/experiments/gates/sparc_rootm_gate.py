# -*- coding: utf-8 -*-
"""
root-m/experiments/gates/sparc_rootm_gate.py

SPARC runner for Root-M tails with rho-aware gating using spherical ρ proxy.
Writes outputs under root-m/out/experiments/gates/sparc/<tag>/.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)

import importlib.util as _ilu, sys as _sys
from pathlib import Path as _P
_pkg_dir = _P(__file__).resolve().parent
_spec = _ilu.spec_from_file_location('gating_common', str(_pkg_dir/'gating_common.py'))
_gm = _ilu.module_from_spec(_spec); _sys.modules['gating_common'] = _gm
_spec.loader.exec_module(_gm)

v_tail2_rootm_rhoaware = _gm.v_tail2_rootm_rhoaware


def spherical_rho_from_vbar(R_kpc, Vbar_kms):
    r = np.asarray(R_kpc, float)
    v = np.asarray(Vbar_kms, float)
    M = (v*v) * r / G
    dMdr = np.gradient(M, r)
    rho = dMdr / (4.0*np.pi * np.maximum(r*r, 1e-12))
    return np.clip(rho, 0.0, None), M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', default='data/sparc_predictions_by_radius.csv')
    ap.add_argument('--outdir', default='root-m/out/experiments/gates/sparc')
    ap.add_argument('--tag', default='rhoaware')
    ap.add_argument('--A_kms', type=float, default=160.0)
    ap.add_argument('--rc_kpc', type=float, default=10.0)
    ap.add_argument('--rho0', type=float, default=1e7)
    ap.add_argument('--q', type=float, default=1.0)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    df = pd.read_csv(in_path)
    needed = ['R_kpc','Vbar_kms','Vobs_kms','galaxy','type']
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {in_path}")

    # compute rho_b spherical proxy and Mb per radius
    rho_b, Mb = spherical_rho_from_vbar(df['R_kpc'], df['Vbar_kms'])
    v2_tail = v_tail2_rootm_rhoaware(df['R_kpc'], Mb, rho_b,
                                     A_kms=args.A_kms, Mref_Msun=6e10, rc_kpc=args.rc_kpc,
                                     rho0_Msun_kpc3=args.rho0, q=args.q, s_kpc=10.0)

    vpred = np.sqrt(np.clip(df['Vbar_kms']**2 + (v2_tail), 0.0, None))
    out = df.copy()
    out['Vpred_rootm_gate_kms'] = vpred
    err_pct = 100.0 * np.abs(vpred - df['Vobs_kms']) / np.maximum(df['Vobs_kms'], 1e-9)
    out['percent_close_rootm_gate'] = 100.0 - err_pct

    od = Path(args.outdir)/args.tag
    od.mkdir(parents=True, exist_ok=True)
    out_path = od/'sparc_predictions_by_radius_rootm_gate.csv'
    out.to_csv(out_path, index=False)

    # per-galaxy summary (outer if is_outer present)
    dfo = out.copy()
    if 'is_outer' in dfo.columns:
        dfo = dfo[dfo['is_outer']==True]
    grp_gal = out.groupby('galaxy', as_index=False).agg(
        galaxy_type=('type','first'),
        N=('galaxy','size'),
        median_percent_close_rootm_gate=('percent_close_rootm_gate','median'),
        mean_percent_close_rootm_gate=('percent_close_rootm_gate','mean'),
    )
    grp_type = dfo.groupby('type', as_index=False).agg(
        galaxies=('galaxy','nunique'),
        points=('galaxy','size'),
        outer_median_percent_close_rootm_gate=('percent_close_rootm_gate','median'),
    )
    summary = {
        'A_kms': args.A_kms,
        'rc_kpc': args.rc_kpc,
        'rho0_Msun_kpc3': args.rho0,
        'q': args.q,
        'global_all_median': float(out['percent_close_rootm_gate'].median()),
        'outer_median': float(dfo['percent_close_rootm_gate'].median()) if len(dfo) else None,
        'files': {
            'predictions_by_radius_rootm_gate': str(out_path),
        }
    }
    (od/'summary_rootm_gate.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
