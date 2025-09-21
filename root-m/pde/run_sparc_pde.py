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
    ap.add_argument('--g0_kms2_per_kpc', type=float, default=1000.0)
    ap.add_argument('--m_exp', type=float, default=1.0)
    ap.add_argument('--m_grid', type=str, default=None, help='Comma-separated m exponents for CV, e.g., 0.7,1.0,1.3')
    ap.add_argument('--axisym_maps', action='store_true')
    ap.add_argument('--rotmod_parquet', default='data/sparc_rotmod_ltg.parquet')
    ap.add_argument('--galaxy', default=None)
    # axisym vertical scale height and CV controls
    ap.add_argument('--hz_kpc', type=float, default=0.3)
    ap.add_argument('--cv', type=int, default=0, help='If >0, perform K-fold CV across galaxies to choose (S0, rc)')
    ap.add_argument('--S0_grid', type=str, default=None, help='Comma-separated S0 grid, e.g., 3e-7,1e-6,3e-6')
    ap.add_argument('--rc_grid', type=str, default=None, help='Comma-separated rc grid in kpc, e.g., 10,15,20')
    ap.add_argument('--gal_subset', type=str, default=None, help='Comma-separated subset of galaxy names for CV')
    ap.add_argument('--cv_max_iter', type=int, default=400, help='Max solver iterations per galaxy during CV')
    ap.add_argument('--cv_tol', type=float, default=5e-5, help='Solver tolerance during CV')
    # New physics knobs
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--Mref', type=float, default=6.0e10)
    ap.add_argument('--kappa', type=float, default=0.0)
    ap.add_argument('--q_slope', type=float, default=1.0)
    ap.add_argument('--chi', type=float, default=0.0)
    ap.add_argument('--h_aniso_kpc', type=float, default=0.3)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    df = pd.read_csv(in_path)

    # Cross-validation mode for axisymmetric maps
    if args.cv:
        if not args.axisym_maps:
            raise SystemExit('CV mode currently requires --axisym_maps to be enabled')
        # dynamic load axisym builder for CV
        pkg = _P(__file__).resolve().parent
        spec = _ilu.spec_from_file_location('baryon_maps_axisym', str(pkg/'baryon_maps_axisym.py'))
        axisym = _ilu.module_from_spec(spec); spec.loader.exec_module(axisym)
        # parse grids
        def _parse_grid(s: str):
            if s is None:
                return None
            vals = []
            for tok in str(s).split(','):
                tok = tok.strip()
                if tok:
                    vals.append(float(tok))
            return vals
        S0s = _parse_grid(args.S0_grid) or [args.S0]
        rcs = _parse_grid(args.rc_grid) or [args.rc_kpc]
        ms  = _parse_grid(args.m_grid)  or [args.m_exp]

        # galaxies available in both CSV and rotmod parquet
        rot = pd.read_parquet(Path(args.rotmod_parquet))
        gals_rot = set(rot['galaxy'].unique())
        gals_df = sorted([g for g in df['galaxy'].unique() if g in gals_rot])
        if args.gal_subset:
            req = [s.strip() for s in str(args.gal_subset).split(',') if s.strip()]
            gals_df = [g for g in gals_df if g in set(req)]
        if not gals_df:
            raise SystemExit('No overlapping galaxies between predictions CSV and rotmod parquet (or subset empty)')

        k = max(2, int(args.cv))
        # round-robin split for determinism
        folds = [gals_df[i::k] for i in range(k)]

        # per-galaxy observed vectors
        per_gal = {}
        for gname in gals_df:
            dfg = df[df['galaxy'] == gname].copy()
            r_eval = np.asarray(dfg['R_kpc'], float)
            vbar = np.asarray(dfg['Vbar_kms'], float)
            vobs = np.asarray(dfg['Vobs_kms'], float)
            is_outer = np.asarray(dfg['is_outer'], bool) if 'is_outer' in dfg.columns else (r_eval >= np.percentile(r_eval, 70))
            per_gal[gname] = {'r': r_eval, 'vbar': vbar, 'vobs': vobs, 'is_outer': is_outer, 'R_last': float(np.max(r_eval))}

        cache = {}  # (gal,S0,rc,R_max,Z_max,NR,NZ,hz) -> median_outer

        grid_results = []
        for S0 in S0s:
            for rc in rcs:
                for mm in ms:
                    fold_meds = []
                    for fi in range(k):
                        val_gals = folds[fi]
                        med_outer_vals = []
                        for gname in val_gals:
                            Rlast = per_gal[gname]['R_last']
                            R_max = args.Rmax if (args.Rmax and args.Rmax > 0) else 3.0*Rlast
                            Z_max = args.Zmax if (args.Zmax and args.Zmax > 0) else R_max
                            key = (gname, S0, rc, mm, R_max, Z_max, args.NR, args.NZ, args.hz_kpc)
                            if key not in cache:
                                Zg, Rg, rho = axisym.axisym_map_from_rotmod_parquet(Path(args.rotmod_parquet), gname,
                                                                                    R_max=R_max, Z_max=Z_max,
                                                                                    NR=args.NR, NZ=args.NZ,
                                                                                    hz_kpc=args.hz_kpc,
                                                                                    bulge_model='hernquist',
                                                                                    bulge_a_fallback_kpc=0.7)
                                params = SolverParams(S0=float(S0), rc_kpc=float(rc), g0_kms2_per_kpc=float(args.g0_kms2_per_kpc), m_exp=float(mm),
                                                       eta=float(args.eta), Mref_Msun=float(args.Mref),
                                                       kappa=float(args.kappa), q_slope=float(args.q_slope),
                                                       chi=float(args.chi), h_aniso_kpc=float(args.h_aniso_kpc),
                                                       max_iter=int(args.cv_max_iter), tol=float(args.cv_tol))
                                phi, gR, gZ = solve_axisym(Rg, Zg, rho, params)
                                r_eval = per_gal[gname]['r']
                                vbar = per_gal[gname]['vbar']
                                vobs = per_gal[gname]['vobs']
                                v_pred, _, _ = predict_v_from_phi_equatorial(Rg, gR, r_eval, vbar)
                                is_outer = per_gal[gname]['is_outer']
                                if not np.any(is_outer):
                                    idx = np.argsort(r_eval)[-3:]
                                else:
                                    idx = np.where(is_outer)[0]
                                err = 100.0 * np.abs(v_pred[idx] - vobs[idx]) / np.maximum(vobs[idx], 1e-9)
                                med_outer = float(np.median(100.0 - err))
                                cache[key] = med_outer
                            med_outer_vals.append(cache[key])
                    if med_outer_vals:
                        fold_meds.append(float(np.median(med_outer_vals)))
                    else:
                        fold_meds.append(float('nan'))
                    # aggregate for this (S0, rc, m)
                    fold_meds_arr = np.array([fm for fm in fold_meds if np.isfinite(fm)], float)
                    agg = float(np.median(fold_meds_arr)) if fold_meds_arr.size else float('nan')
                    grid_results.append({'S0': float(S0), 'rc_kpc': float(rc), 'm_exp': float(mm), 'fold_medians': fold_meds, 'cv_median': agg})
                    val_gals = folds[fi]
                    med_outer_vals = []
                    for gname in val_gals:
                        Rlast = per_gal[gname]['R_last']
                        R_max = args.Rmax if (args.Rmax and args.Rmax > 0) else 3.0*Rlast
                        Z_max = args.Zmax if (args.Zmax and args.Zmax > 0) else R_max
                        key = (gname, S0, rc, R_max, Z_max, args.NR, args.NZ, args.hz_kpc)
                        if key not in cache:
                            Zg, Rg, rho = axisym.axisym_map_from_rotmod_parquet(Path(args.rotmod_parquet), gname,
                                                                                R_max=R_max, Z_max=Z_max,
                                                                                NR=args.NR, NZ=args.NZ,
                                                                                hz_kpc=args.hz_kpc,
                                                                                bulge_model='hernquist',
                                                                                bulge_a_fallback_kpc=0.7)
                            params = SolverParams(S0=float(S0), rc_kpc=float(rc), g0_kms2_per_kpc=float(args.g0_kms2_per_kpc),
                                                  eta=float(args.eta), Mref_Msun=float(args.Mref),
                                                  kappa=float(args.kappa), q_slope=float(args.q_slope),
                                                  chi=float(args.chi), h_aniso_kpc=float(args.h_aniso_kpc),
                                                  max_iter=int(args.cv_max_iter), tol=float(args.cv_tol))
                            phi, gR, gZ = solve_axisym(Rg, Zg, rho, params)
                            r_eval = per_gal[gname]['r']
                            vbar = per_gal[gname]['vbar']
                            vobs = per_gal[gname]['vobs']
                            v_pred, _, _ = predict_v_from_phi_equatorial(Rg, gR, r_eval, vbar)
                            is_outer = per_gal[gname]['is_outer']
                            if not np.any(is_outer):
                                # fallback to last 3 points
                                idx = np.argsort(r_eval)[-3:]
                            else:
                                idx = np.where(is_outer)[0]
                            err = 100.0 * np.abs(v_pred[idx] - vobs[idx]) / np.maximum(vobs[idx], 1e-9)
                            med_outer = float(np.median(100.0 - err))
                            cache[key] = med_outer
                        med_outer_vals.append(cache[key])
                    if med_outer_vals:
                        fold_meds.append(float(np.median(med_outer_vals)))
                    else:
                        fold_meds.append(float('nan'))
                # aggregate across folds
                fold_meds_arr = np.array([fm for fm in fold_meds if np.isfinite(fm)], float)
                agg = float(np.median(fold_meds_arr)) if fold_meds_arr.size else float('nan')
                grid_results.append({'S0': float(S0), 'rc_kpc': float(rc), 'fold_medians': fold_meds, 'cv_median': agg})

        # choose best by highest cv_median
        best = max(grid_results, key=lambda d: (d['cv_median'],)) if grid_results else None

        od = Path(args.outdir)
        od.mkdir(parents=True, exist_ok=True)
        (od/'cv_summary.json').write_text(json.dumps({'k_folds': k,
                                                      'galaxies': gals_df,
                                                      'grid_results': grid_results,
                                                      'best': best}, indent=2))
        print(f"[PDE-CV] Completed CV over {len(S0s)*len(rcs)} combos; best={best}")
        return

    if args.axisym_maps:
        # build axisymmetric map for a single galaxy
        if not args.galaxy:
            raise SystemExit('Please supply --galaxy when using --axisym_maps')
        # dynamic load axisym builder
        pkg = _P(__file__).resolve().parent
        spec = _ilu.spec_from_file_location('baryon_maps_axisym', str(pkg/'baryon_maps_axisym.py'))
        axisym = _ilu.module_from_spec(spec); spec.loader.exec_module(axisym)
        Z, R, rho = axisym.axisym_map_from_rotmod_parquet(Path(args.rotmod_parquet), args.galaxy,
                                                          R_max=args.Rmax, Z_max=args.Zmax, NR=args.NR, NZ=args.NZ,
                                                          hz_kpc=args.hz_kpc, bulge_model='hernquist', bulge_a_fallback_kpc=0.7)
    else:
        # PDE map from spherical-equivalent rho_b
        Z, R, rho = sparc_map_from_predictions(in_path, R_max=args.Rmax, Z_max=args.Zmax, NR=args.NR, NZ=args.NZ)

    # Solve PDE
    params = SolverParams(S0=args.S0, rc_kpc=args.rc_kpc, g0_kms2_per_kpc=args.g0_kms2_per_kpc, m_exp=args.m_exp,
                          eta=float(args.eta), Mref_Msun=float(args.Mref),
                          kappa=float(args.kappa), q_slope=float(args.q_slope),
                          chi=float(args.chi), h_aniso_kpc=float(args.h_aniso_kpc))
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
