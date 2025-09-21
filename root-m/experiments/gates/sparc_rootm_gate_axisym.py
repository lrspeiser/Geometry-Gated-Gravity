# -*- coding: utf-8 -*-
"""
root-m/experiments/gates/sparc_rootm_gate_axisym.py

SPARC runner for Root-M tails with rho-aware gating using axisymmetric midplane density
from rotmod parquet (Σ -> ρ midplane). Keeps original code intact.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)

# dynamic import of gating_common
import importlib.util as _ilu, sys as _sys
from pathlib import Path as _P
_pkg_dir = _P(__file__).resolve().parent
_spec = _ilu.spec_from_file_location('gating_common', str(_pkg_dir/'gating_common.py'))
_gm = _ilu.module_from_spec(_spec); _sys.modules['gating_common'] = _gm
_spec.loader.exec_module(_gm)

v_tail2_rootm_rhoaware = _gm.v_tail2_rootm_rhoaware

# dynamic import of pde baryon axisym builder
pde_dir = _pkg_dir.parent.parent / 'pde'
_spec2 = _ilu.spec_from_file_location('baryon_maps_axisym', str(pde_dir/'baryon_maps_axisym.py'))
_axis = _ilu.module_from_spec(_spec2); _sys.modules['baryon_maps_axisym'] = _axis
_spec2.loader.exec_module(_axis)
axisym_map_from_rotmod_parquet = _axis.axisym_map_from_rotmod_parquet


def spherical_M_from_vbar(R_kpc, Vbar_kms):
    r = np.asarray(R_kpc, float)
    v = np.asarray(Vbar_kms, float)
    return (v*v) * r / G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', default='data/sparc_predictions_by_radius.csv')
    ap.add_argument('--rotmod_parquet', default='data/sparc_rotmod_ltg.parquet')
    ap.add_argument('--outdir', default='root-m/out/experiments/gates/sparc_axisym')
    ap.add_argument('--tag', default='rhoaware_axisym')
    ap.add_argument('--A_kms', type=float, default=160.0)
    ap.add_argument('--rc_kpc', type=float, default=10.0)
    ap.add_argument('--rho0', type=float, default=1e7)
    ap.add_argument('--q', type=float, default=1.0)
    ap.add_argument('--hz_kpc', type=float, default=0.3)
    ap.add_argument('--NR', type=int, default=128)
    ap.add_argument('--NZ', type=int, default=128)
    ap.add_argument('--Rmax', type=float, default=80.0)
    ap.add_argument('--Zmax', type=float, default=80.0)
    ap.add_argument('--gal_subset', type=str, default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    df = pd.read_csv(in_path)

    rot = pd.read_parquet(Path(args.rotmod_parquet))
    allowed = set(rot['galaxy'].unique())
    galaxies = sorted([g for g in df['galaxy'].unique() if g in allowed])
    if args.gal_subset:
        req = [s.strip() for s in str(args.gal_subset).split(',') if s.strip()]
        galaxies = [g for g in galaxies if g in set(req)]

    rows = []
    per_gal_summary = []
    for gname in galaxies:
        dfg = df[df['galaxy'] == gname].copy()
        r_eval = np.asarray(dfg['R_kpc'], float)
        vbar = np.asarray(dfg['Vbar_kms'], float)
        vobs = np.asarray(dfg['Vobs_kms'], float)
        Rmax = max(args.Rmax, float(np.max(r_eval))*1.2)
        Zmax = max(args.Zmax, Rmax)
        Zg, Rg, rho = axisym_map_from_rotmod_parquet(Path(args.rotmod_parquet), gname,
                                                     R_max=Rmax, Z_max=Zmax,
                                                     NR=args.NR, NZ=args.NZ, hz_kpc=args.hz_kpc,
                                                     bulge_model='hernquist', bulge_a_fallback_kpc=0.7)
        z0_idx = len(Zg)//2
        rho_mid = rho[z0_idx, :]
        rho_at_eval = np.interp(r_eval, Rg, rho_mid, left=rho_mid[0], right=rho_mid[-1])
        Mb = spherical_M_from_vbar(r_eval, vbar)
        # Direct rho-aware gate at evaluation radii (no extra smoothing to preserve lengths)
        dens_fac = (args.rho0 / (rho_at_eval + args.rho0))**(args.q)
        v2_tail = (args.A_kms**2) * np.sqrt(Mb / 6e10) * dens_fac * (r_eval / (r_eval + args.rc_kpc))
        vpred = np.sqrt(np.clip(vbar**2 + v2_tail, 0.0, None))
        err_pct = 100.0 * np.abs(vpred - vobs) / np.maximum(vobs, 1e-9)
        pct_close = 100.0 - err_pct
        dfg['Vpred_rootm_gate_axisym_kms'] = vpred
        dfg['percent_close_rootm_gate_axisym'] = pct_close
        rows.append(dfg)
        # per-gal summary
        outer = dfg[dfg['is_outer']==True] if 'is_outer' in dfg.columns else dfg
        per_gal_summary.append({
            'galaxy': gname,
            'N': int(len(dfg)),
            'outer_median_percent_close': float(outer['percent_close_rootm_gate_axisym'].median())
        })

    out_all = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    od = Path(args.outdir)/args.tag
    od.mkdir(parents=True, exist_ok=True)
    out_path = od/'sparc_predictions_by_radius_rootm_gate_axisym.csv'
    out_all.to_csv(out_path, index=False)

    dfo = out_all.copy()
    if 'is_outer' in dfo.columns:
        dfo = dfo[dfo['is_outer']==True]
    summary = {
        'A_kms': args.A_kms,
        'rc_kpc': args.rc_kpc,
        'rho0_Msun_kpc3': args.rho0,
        'q': args.q,
        'hz_kpc': args.hz_kpc,
        'outer_median': float(dfo['percent_close_rootm_gate_axisym'].median()) if len(dfo) else None,
        'global_all_median': float(out_all['percent_close_rootm_gate_axisym'].median()) if len(out_all) else None,
        'per_galaxy': per_gal_summary,
        'files': {
            'predictions_by_radius_rootm_gate_axisym': str(out_path),
        }
    }
    (od/'summary_rootm_gate_axisym.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
