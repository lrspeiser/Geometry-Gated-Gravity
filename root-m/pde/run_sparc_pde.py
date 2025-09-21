# -*- coding: utf-8 -*-
"""
root-m/pde/run_sparc_pde.py

End-to-end PDE run for SPARC-like tables (spherical-equivalent prototype):
- Build spherical rho_b(R,z) from Vbar(r) in predictions_by_radius.csv
- Solve ∇·(|∇φ| ∇φ) = - S0 ρ_b
- Predict v(R) from g_phi(R,0) + baryon g_N
- Write CSV + PNG under root-m/out/pde_sparc/<TAG>/

Note: This uses a spherical-equivalent rho_b derived from Vbar(r).
Disc geometry can be incorporated later by building axisymmetric rho_b maps.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt

# Local imports (folder is not a Python package)
import importlib.util as _ilu
from pathlib import Path as _P
_pkg_dir = _P(__file__).resolve().parent

def _load_local(modname: str, filename: str):
    import sys as _sys
    spec = _ilu.spec_from_file_location(modname, str(_pkg_dir/filename))
    mod = _ilu.module_from_spec(spec)
    _sys.modules[modname] = mod  # register for dataclasses and relative references
    spec.loader.exec_module(mod)
    return mod
_solve = _load_local('solve_phi', 'solve_phi.py')
_maps  = _load_local('baryon_maps', 'baryon_maps.py')
_prc   = _load_local('predict_rc', 'predict_rc.py')

SolverParams = _solve.SolverParams
solve_axisym = _solve.solve_axisym
sparc_map_from_predictions = _maps.sparc_map_from_predictions
predict_v_from_phi_equatorial = _prc.predict_v_from_phi_equatorial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', default='data/sparc_predictions_by_radius.csv')
    ap.add_argument('--outdir', default='root-m/out/pde_sparc')
    ap.add_argument('--tag', default='all')
    ap.add_argument('--Rmax', type=float, default=80.0)
    ap.add_argument('--Zmax', type=float, default=80.0)
    ap.add_argument('--NR', type=int, default=128)
    ap.add_argument('--NZ', type=int, default=128)
    ap.add_argument('--S0', type=float, default=1.0e-7)
    ap.add_argument('--rc_kpc', type=float, default=15.0)
    ap.add_argument('--axisym_maps', action='store_true')
    ap.add_argument('--rotmod_parquet', default='data/sparc_rotmod_ltg.parquet')
    ap.add_argument('--galaxy', default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    df = pd.read_csv(in_path)

    if args.axisym_maps:
        # build axisymmetric map for a single galaxy
        if not args.galaxy:
            raise SystemExit('Please supply --galaxy when using --axisym_maps')
        # dynamic load axisym builder
        import importlib.util as _ilu
        pkg = _P(__file__).resolve().parent
        spec = _ilu.spec_from_file_location('baryon_maps_axisym', str(pkg/'baryon_maps_axisym.py'))
        axisym = _ilu.module_from_spec(spec); spec.loader.exec_module(axisym)
        Z, R, rho = axisym.axisym_map_from_rotmod_parquet(Path(args.rotmod_parquet), args.galaxy,
                                                          R_max=args.Rmax, Z_max=args.Zmax, NR=args.NR, NZ=args.NZ, hz_kpc=0.3)
    else:
        # PDE map from spherical-equivalent rho_b
        Z, R, rho = sparc_map_from_predictions(in_path, R_max=args.Rmax, Z_max=args.Zmax, NR=args.NR, NZ=args.NZ)

    # Solve PDE
    params = SolverParams(S0=args.S0, rc_kpc=args.rc_kpc)
    phi, gR, gZ = solve_axisym(R, Z, rho, params)

    # Predict v(R) at observed radii
    if args.axisym_maps:
        dfg = df[df['galaxy'] == args.galaxy].copy()
        if dfg.empty:
            raise SystemExit(f'No rows for galaxy {args.galaxy} in {in_path}')
        r_eval = np.asarray(dfg['R_kpc'], float)
        vbar = np.asarray(dfg['Vbar_kms'], float)
    else:
        r_eval = np.asarray(df['R_kpc'], float)
        vbar = np.asarray(df['Vbar_kms'], float)
    v_pred, gphi, gN = predict_v_from_phi_equatorial(R, gR, r_eval, vbar)

    if args.axisym_maps:
        vobs = np.asarray(dfg['Vobs_kms'], float)
    else:
        vobs = np.asarray(df['Vobs_kms'], float)

    # Logging for diagnostics
    print(f"[PDE] axisym_maps={args.axisym_maps} galaxy={args.galaxy} n_eval={len(r_eval)} n_obs={len(vobs)} n_bar={len(vbar)} grid_NR={R.shape[0]} grid_NZ={Z.shape[0]} rho_shape={rho.shape}")

    err = 100.0 * np.abs(v_pred - vobs) / np.maximum(vobs, 1e-9)
    med = float(np.median(100.0 - err))

    od = Path(args.outdir)/args.tag
    od.mkdir(parents=True, exist_ok=True)
    # save RC table
    out_csv = od/'rc_pde_predictions.csv'
    pd.DataFrame({'R_kpc': r_eval, 'Vobs_kms': vobs, 'Vbar_kms': vbar,
                  'Vpred_pde_kms': v_pred, 'percent_close': (100.0 - err)}).to_csv(out_csv, index=False)
    # metrics
    (od/'summary.json').write_text(json.dumps({'S0': args.S0, 'rc_kpc': args.rc_kpc,
                                               'median_percent_close': med}, indent=2))

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(r_eval, vobs, 'k.', ms=3, label='obs')
    plt.plot(r_eval, vbar, 'c--', lw=1.2, label='baryon')
    plt.plot(r_eval, v_pred, 'r-', lw=1.2, label='PDE pred')
    plt.xlabel('R [kpc]'); plt.ylabel('V [km/s]'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(od/'rc_pde_results.png', dpi=140); plt.close()

if __name__ == '__main__':
    main()
