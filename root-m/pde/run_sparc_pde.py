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

# Optional: dynamic import of SOG utilities
import importlib.util as _ilu2, sys as _sys
from pathlib import Path as _Path
_sog_path = _Path(__file__).resolve().parents[1] / 'pde' / 'second_order.py'
if _sog_path.exists():
    _spec_sog = _ilu2.spec_from_file_location('second_order', str(_sog_path))
    second_order = _ilu2.module_from_spec(_spec_sog)
    _sys.modules['second_order'] = second_order
    _spec_sog.loader.exec_module(second_order)
else:
    second_order = None

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
    ap.add_argument('--S0', type=float, default=1.4e-4)
    ap.add_argument('--rc_kpc', type=float, default=22.0)
    ap.add_argument('--g0_kms2_per_kpc', type=float, default=1200.0)
    ap.add_argument('--m_exp', type=float, default=1.0)
    ap.add_argument('--m_grid', type=str, default=None, help='Comma-separated m exponents for CV, e.g., 0.7,1.0,1.3')
    # Geometry-aware global knobs
    ap.add_argument('--rc_gamma', type=float, default=0.5)
    ap.add_argument('--rc_ref_kpc', type=float, default=30.0)
    ap.add_argument('--sigma_beta', type=float, default=0.10)
    ap.add_argument('--sigma0_Msun_pc2', type=float, default=150.0)
    # CV grids for geometry-aware knobs
    ap.add_argument('--rc_gamma_grid', type=str, default=None, help='Comma-separated rc_gamma grid for CV, e.g., 0.0,0.25,0.5')
    ap.add_argument('--sigma_beta_grid', type=str, default=None, help='Comma-separated sigma_beta grid for CV, e.g., 0.0,0.1')
    ap.add_argument('--axisym_maps', action='store_true')
    ap.add_argument('--rotmod_parquet', default='data/sparc_rotmod_ltg.parquet')
    ap.add_argument('--galaxy', default=None)
    # axisym vertical scale height and CV controls
    ap.add_argument('--hz_kpc', type=float, default=0.3)
    ap.add_argument('--hz_gamma', type=float, default=0.0)
    ap.add_argument('--all_tables_parquet', default='data/sparc_all_tables.parquet')
    ap.add_argument('--hz_from_rd', action='store_true', help='If set, derive hz≈rd_to_hz * Rd (fallback hz_floor_kpc) from sparc_all_tables.parquet')
    ap.add_argument('--rd_to_hz', type=float, default=0.1)
    ap.add_argument('--hz_floor_kpc', type=float, default=0.3)
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
    ap.add_argument('--bc_robin_lambda', type=float, default=0.0, help='Robin BC strength (1/kpc); 0 disables')
    # NEW A1/A2 global controls
    ap.add_argument('--use_saturating_mobility', action='store_true')
    ap.add_argument('--gsat_kms2_per_kpc', type=float, default=2000.0)
    ap.add_argument('--n_sat', type=float, default=2.0)
    ap.add_argument('--use_ambient_boost', action='store_true')
    ap.add_argument('--beta_env', type=float, default=0.0)
    ap.add_argument('--rho_ref_Msun_per_kpc3', type=float, default=1.0e6)
    ap.add_argument('--env_L_kpc', type=float, default=150.0)
    # Local surface-density screen (category-blind; off by default for backward compatibility)
    ap.add_argument('--use_sigma_screen', action='store_true', help='Enable local surface-density screen of the PDE source')
    ap.add_argument('--sigma_star_Msun_per_pc2', type=float, default=150.0, help='Surface-density threshold Sigma* [Msun/pc^2] for sigma screen')
    ap.add_argument('--alpha_sigma', type=float, default=1.0, help='Strength of sigma screen (alpha)')
    ap.add_argument('--n_sigma', type=float, default=2.0, help='Slope parameter of sigma screen (n)')
    # SOG options (axisym only; default OFF)
    ap.add_argument('--use_sog_fe', action='store_true')
    ap.add_argument('--use_sog_rho2', action='store_true')
    ap.add_argument('--use_sog_rg', action='store_true')
    ap.add_argument('--sog_sigma_star', type=float, default=100.0)
    ap.add_argument('--sog_g_star', type=float, default=1200.0)
    ap.add_argument('--sog_aSigma', type=float, default=2.0)
    ap.add_argument('--sog_ag', type=float, default=2.0)
    ap.add_argument('--sog_fe_lambda', type=float, default=1.0)
    ap.add_argument('--sog_rho2_eta', type=float, default=0.01)
    ap.add_argument('--sog_rg_A', type=float, default=0.8)
    ap.add_argument('--sog_rg_n', type=float, default=1.2)
    ap.add_argument('--sog_rg_g0', type=float, default=5.0)
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
        gammas = _parse_grid(args.rc_gamma_grid) or [args.rc_gamma]
        betas  = _parse_grid(args.sigma_beta_grid) or [args.sigma_beta]

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
                    for gamma in gammas:
                        for beta in betas:
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
                                                                                    hz_kpc=args.hz_kpc, hz_gamma=args.hz_gamma,
                                                                                    bulge_model='hernquist',
                                                                                    bulge_a_fallback_kpc=0.7,
                                                                                    all_tables_parquet=Path(args.all_tables_parquet) if args.all_tables_parquet else None,
                                                                                    hz_from_rd=bool(args.hz_from_rd),
                                                                                    rd_to_hz=float(args.rd_to_hz),
                                                                                    hz_floor_kpc=float(args.hz_floor_kpc))
                                # Geometry scalars from axisym map (spherical M_enc approx)
                                dR = float(np.mean(np.diff(Rg))) if Rg.size > 1 else 1.0
                                dZ = float(np.mean(np.diff(Zg))) if Zg.size > 1 else 1.0
                                R2D = np.broadcast_to(Rg.reshape(1,-1), rho.shape)
                                dV = (2.0 * np.pi) * R2D * dR * dZ
                                r_cell = np.sqrt(R2D*R2D + np.broadcast_to(Zg.reshape(-1,1), rho.shape)**2)
                                mass_cells = np.clip(rho, 0.0, None) * dV
                                r_flat = r_cell.reshape(-1); m_flat = mass_cells.reshape(-1)
                                order = np.argsort(r_flat)
                                M_cum = np.cumsum(m_flat[order])
                                M_tot = float(M_cum[-1]) if M_cum.size else 0.0
                                # r_half via percentile on sorted radii
                                r_sorted = r_flat[order]
                                r_half = float(np.interp(0.5*M_tot, M_cum, r_sorted)) if M_tot > 0 else float(Rg[len(Rg)//4])
                                sigma_bar_kpc2 = (0.5*M_tot) / (np.pi * max(r_half, 1e-9)**2) if r_half > 0 else 0.0
                                sigma_bar_pc2 = sigma_bar_kpc2 / 1.0e6
                                rc_eff = float(rc) * (max(r_half, 1e-12) / max(args.rc_ref_kpc, 1e-12))**float(gamma)
                                S0_eff = float(S0) * (max(args.sigma0_Msun_pc2, 1e-12) / max(sigma_bar_pc2, 1e-12))**float(beta)
                                params = SolverParams(S0=S0_eff, rc_kpc=rc_eff, g0_kms2_per_kpc=float(args.g0_kms2_per_kpc), m_exp=float(mm),
                                                       eta=float(args.eta), Mref_Msun=float(args.Mref),
                                                       kappa=float(args.kappa), q_slope=float(args.q_slope),
                                                       chi=float(args.chi), h_aniso_kpc=float(args.h_aniso_kpc),
                                                       use_saturating_mobility=bool(args.use_saturating_mobility),
                                                       gsat_kms2_per_kpc=float(args.gsat_kms2_per_kpc),
                                                       n_sat=float(args.n_sat),
                                                       use_ambient_boost=bool(args.use_ambient_boost),
                                                       beta_env=float(args.beta_env),
                                                       rho_ref_Msun_per_kpc3=float(args.rho_ref_Msun_per_kpc3),
                                                       env_L_kpc=float(args.env_L_kpc),
                                                       use_sigma_screen=bool(args.use_sigma_screen),
                                                       sigma_star_Msun_per_pc2=float(args.sigma_star_Msun_per_pc2),
                                                       alpha_sigma=float(args.alpha_sigma),
                                                       n_sigma=float(args.n_sigma),
                                                       max_iter=int(args.cv_max_iter), tol=float(args.cv_tol))
                                phi, gR, gZ = solve_axisym(Rg, Zg, rho, params)
                                r_eval = per_gal[gname]['r']
                                vbar = per_gal[gname]['vbar']
                                vobs = per_gal[gname]['vobs']
                                v_pred, gphi_eval, gN_eval = predict_v_from_phi_equatorial(Rg, gR, r_eval, vbar)
                                # Optional SOG augmentation (axisym-only)
                                if second_order is not None and (args.use_sog_fe or args.use_sog_rho2 or args.use_sog_rg):
                                    # Build Sigma proxy from rotmod components
                                    try:
                                        dfr = rot[rot['galaxy'] == gname].sort_values('R_kpc')
                                        Rr = dfr['R_kpc'].to_numpy(float)
                                        Vgas = dfr['Vgas_kms'].to_numpy(float) if 'Vgas_kms' in dfr.columns else np.zeros_like(Rr)
                                        Vdisk = dfr['Vdisk_kms'].to_numpy(float) if 'Vdisk_kms' in dfr.columns else np.zeros_like(Rr)
                                        Gc = 4.300917270e-6
                                        Mgas = (Vgas*Vgas) * Rr / Gc
                                        Mdisk= (Vdisk*Vdisk) * Rr / Gc
                                        Sgas = axisym.sigma_from_M_of_R(Rr, Mgas)
                                        Sdisk= axisym.sigma_from_M_of_R(Rr, Mdisk)
                                        Sigma_sum = Sgas + Sdisk
                                        Sigma_eval = np.interp(r_eval, Rr, Sigma_sum, left=Sigma_sum[0], right=Sigma_sum[-1])
                                    except Exception:
                                        Sigma_eval = np.zeros_like(r_eval)
                                    g1_eval = (vbar*vbar) / np.maximum(r_eval, 1e-9)
                                    gate = {
                                        'Sigma_star': float(args.sog_sigma_star),
                                        'g_star': float(args.sog_g_star),
                                        'aSigma': float(args.sog_aSigma),
                                        'ag': float(args.sog_ag),
                                    }
                                    g2_add = np.zeros_like(g1_eval)
                                    if args.use_sog_fe:
                                        g2_add += second_order.g2_field_energy(r_eval, g1_eval, Sigma_eval, {'lambda': float(args.sog_fe_lambda), **gate})
                                    if args.use_sog_rho2:
                                        # Construct a spherical-equivalent rho proxy from Sigma via rho ~ Sigma/(2h)
                                        # Use h ~ 0.3 kpc as a benign scale height default
                                        rho_proxy = (Sigma_eval * 1e6) / (2.0 * 0.3)
                                        g2_add += second_order.g2_rho2_local(r_eval, rho_proxy, g1_eval, Sigma_eval, {'eta': float(args.sog_rho2_eta), **gate})
                                    if args.use_sog_rg:
                                        g2_add += second_order.g_runningG(r_eval, g1_eval, Sigma_eval, {'A': float(args.sog_rg_A), 'n': float(args.sog_rg_n), 'g0': float(args.sog_rg_g0), **gate})
                                    v2 = np.clip(vbar*vbar + (gphi_eval + g2_add) * r_eval, 0.0, None)
                                    v_pred = np.sqrt(v2)
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
                    # aggregate for this (S0, rc, m, gamma, beta)
                    fold_meds_arr = np.array([fm for fm in fold_meds if np.isfinite(fm)], float)
                    agg = float(np.median(fold_meds_arr)) if fold_meds_arr.size else float('nan')
                    grid_results.append({'S0': float(S0), 'rc_kpc': float(rc), 'm_exp': float(mm), 'rc_gamma': float(gamma), 'sigma_beta': float(beta), 'fold_medians': fold_meds, 'cv_median': agg})
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
                                                  use_sigma_screen=bool(args.use_sigma_screen),
                                                  sigma_star_Msun_per_pc2=float(args.sigma_star_Msun_per_pc2),
                                                  alpha_sigma=float(args.alpha_sigma),
                                                  n_sigma=float(args.n_sigma),
                                                  max_iter=int(args.cv_max_iter), tol=float(args.cv_tol))
                            phi, gR, gZ = solve_axisym(Rg, Zg, rho, params)
                            r_eval = per_gal[gname]['r']
                            vbar = per_gal[gname]['vbar']
                            vobs = per_gal[gname]['vobs']
                            v_pred, gphi_eval, gN_eval = predict_v_from_phi_equatorial(Rg, gR, r_eval, vbar)
                            # Optional SOG augmentation (axisym-only)
                            if second_order is not None and (args.use_sog_fe or args.use_sog_rho2 or args.use_sog_rg):
                                try:
                                    dfr = rot[rot['galaxy'] == gname].sort_values('R_kpc')
                                    Rr = dfr['R_kpc'].to_numpy(float)
                                    Vgas = dfr['Vgas_kms'].to_numpy(float) if 'Vgas_kms' in dfr.columns else np.zeros_like(Rr)
                                    Vdisk = dfr['Vdisk_kms'].to_numpy(float) if 'Vdisk_kms' in dfr.columns else np.zeros_like(Rr)
                                    Gc = 4.300917270e-6
                                    Mgas = (Vgas*Vgas) * Rr / Gc
                                    Mdisk= (Vdisk*Vdisk) * Rr / Gc
                                    Sgas = axisym.sigma_from_M_of_R(Rr, Mgas)
                                    Sdisk= axisym.sigma_from_M_of_R(Rr, Mdisk)
                                    Sigma_sum = Sgas + Sdisk
                                    Sigma_eval = np.interp(r_eval, Rr, Sigma_sum, left=Sigma_sum[0], right=Sigma_sum[-1])
                                except Exception:
                                    Sigma_eval = np.zeros_like(r_eval)
                                g1_eval = (vbar*vbar) / np.maximum(r_eval, 1e-9)
                                gate = {
                                    'Sigma_star': float(args.sog_sigma_star),
                                    'g_star': float(args.sog_g_star),
                                    'aSigma': float(args.sog_aSigma),
                                    'ag': float(args.sog_ag),
                                }
                                g2_add = np.zeros_like(g1_eval)
                                if args.use_sog_fe:
                                    g2_add += second_order.g2_field_energy(r_eval, g1_eval, Sigma_eval, {'lambda': float(args.sog_fe_lambda), **gate})
                                if args.use_sog_rho2:
                                    rho_proxy = (Sigma_eval * 1e6) / (2.0 * 0.3)
                                    g2_add += second_order.g2_rho2_local(r_eval, rho_proxy, g1_eval, Sigma_eval, {'eta': float(args.sog_rho2_eta), **gate})
                                if args.use_sog_rg:
                                    g2_add += second_order.g_runningG(r_eval, g1_eval, Sigma_eval, {'A': float(args.sog_rg_A), 'n': float(args.sog_rg_n), 'g0': float(args.sog_rg_g0), **gate})
                                v2 = np.clip(vbar*vbar + (gphi_eval + g2_add) * r_eval, 0.0, None)
                                v_pred = np.sqrt(v2)
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
                                                          hz_kpc=args.hz_kpc, hz_gamma=args.hz_gamma,
                                                          bulge_model='hernquist', bulge_a_fallback_kpc=0.7,
                                                          all_tables_parquet=Path(args.all_tables_parquet) if args.all_tables_parquet else None,
                                                          hz_from_rd=bool(args.hz_from_rd),
                                                          rd_to_hz=float(args.rd_to_hz),
                                                          hz_floor_kpc=float(args.hz_floor_kpc))
    else:
        # PDE map from spherical-equivalent rho_b
        Z, R, rho = sparc_map_from_predictions(in_path, R_max=args.Rmax, Z_max=args.Zmax, NR=args.NR, NZ=args.NZ)

    # Geometry scalars from map (axisym approx)
    dR = float(np.mean(np.diff(R))) if R.size > 1 else 1.0
    dZ = float(np.mean(np.diff(Z))) if Z.size > 1 else 1.0
    R2D = np.broadcast_to(R.reshape(1,-1), rho.shape)
    dV = (2.0 * np.pi) * R2D * dR * dZ
    r_cell = np.sqrt(R2D*R2D + np.broadcast_to(Z.reshape(-1,1), rho.shape)**2)
    mass_cells = np.clip(rho, 0.0, None) * dV
    r_flat = r_cell.reshape(-1); m_flat = mass_cells.reshape(-1)
    order = np.argsort(r_flat)
    M_cum = np.cumsum(m_flat[order])
    M_tot = float(M_cum[-1]) if M_cum.size else 0.0
    r_sorted = r_flat[order]
    r_half = float(np.interp(0.5*M_tot, M_cum, r_sorted)) if M_tot > 0 else float(R[len(R)//4])
    sigma_bar_kpc2 = (0.5*M_tot) / (np.pi * max(r_half, 1e-9)**2) if r_half > 0 else 0.0
    sigma_bar_pc2 = sigma_bar_kpc2 / 1.0e6
    rc_eff = float(args.rc_kpc) * (max(r_half, 1e-12) / max(args.rc_ref_kpc, 1e-12))**float(args.rc_gamma)
    S0_eff = float(args.S0) * (max(args.sigma0_Msun_pc2, 1e-12) / max(sigma_bar_pc2, 1e-12))**float(args.sigma_beta)

    # Solve PDE
    params = SolverParams(S0=S0_eff, rc_kpc=rc_eff, g0_kms2_per_kpc=args.g0_kms2_per_kpc, m_exp=args.m_exp,
                          eta=float(args.eta), Mref_Msun=float(args.Mref),
                          kappa=float(args.kappa), q_slope=float(args.q_slope),
                          chi=float(args.chi), h_aniso_kpc=float(args.h_aniso_kpc),
                          use_saturating_mobility=bool(args.use_saturating_mobility),
                          gsat_kms2_per_kpc=float(args.gsat_kms2_per_kpc),
                          n_sat=float(args.n_sat),
                          use_ambient_boost=bool(args.use_ambient_boost),
                          beta_env=float(args.beta_env),
                          rho_ref_Msun_per_kpc3=float(args.rho_ref_Msun_per_kpc3),
                          env_L_kpc=float(args.env_L_kpc),
                          use_sigma_screen=bool(args.use_sigma_screen),
                          sigma_star_Msun_per_pc2=float(args.sigma_star_Msun_per_pc2),
                          alpha_sigma=float(args.alpha_sigma),
                          n_sigma=float(args.n_sigma),
                          bc_robin_lambda=float(args.bc_robin_lambda))
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
    # Optional SOG augmentation for single galaxy (axisym_maps only)
    if args.axisym_maps and second_order is not None and (args.use_sog_fe or args.use_sog_rho2 or args.use_sog_rg):
        # Build Sigma proxy from rotmod
        try:
            rot = pd.read_parquet(Path(args.rotmod_parquet))
            dfr = rot[rot['galaxy'] == args.galaxy].sort_values('R_kpc')
            Rr = dfr['R_kpc'].to_numpy(float)
            Vgas = dfr['Vgas_kms'].to_numpy(float) if 'Vgas_kms' in dfr.columns else np.zeros_like(Rr)
            Vdisk = dfr['Vdisk_kms'].to_numpy(float) if 'Vdisk_kms' in dfr.columns else np.zeros_like(Rr)
            Gc = 4.300917270e-6
            Mgas = (Vgas*Vgas) * Rr / Gc
            Mdisk= (Vdisk*Vdisk) * Rr / Gc
            # dynamic import axisym builder to access sigma_from_M_of_R
            pkg = _P(__file__).resolve().parent
            spec = _ilu.spec_from_file_location('baryon_maps_axisym', str(pkg/'baryon_maps_axisym.py'))
            axisym = _ilu.module_from_spec(spec); spec.loader.exec_module(axisym)
            Sgas = axisym.sigma_from_M_of_R(Rr, Mgas)
            Sdisk= axisym.sigma_from_M_of_R(Rr, Mdisk)
            Sigma_sum = Sgas + Sdisk
            Sigma_eval = np.interp(r_eval, Rr, Sigma_sum, left=Sigma_sum[0], right=Sigma_sum[-1])
        except Exception:
            Sigma_eval = np.zeros_like(r_eval)
        g1_eval = (vbar*vbar) / np.maximum(r_eval, 1e-9)
        gate = {
            'Sigma_star': float(args.sog_sigma_star),
            'g_star': float(args.sog_g_star),
            'aSigma': float(args.sog_aSigma),
            'ag': float(args.sog_ag),
        }
        g2_add = np.zeros_like(g1_eval)
        if args.use_sog_fe:
            g2_add += second_order.g2_field_energy(r_eval, g1_eval, Sigma_eval, {'lambda': float(args.sog_fe_lambda), **gate})
        if args.use_sog_rho2:
            rho_proxy = (Sigma_eval * 1e6) / (2.0 * 0.3)
            g2_add += second_order.g2_rho2_local(r_eval, rho_proxy, g1_eval, Sigma_eval, {'eta': float(args.sog_rho2_eta), **gate})
        if args.use_sog_rg:
            g2_add += second_order.g_runningG(r_eval, g1_eval, Sigma_eval, {'A': float(args.sog_rg_A), 'n': float(args.sog_rg_n), 'g0': float(args.sog_rg_g0), **gate})
        v2 = np.clip(vbar*vbar + (gphi + g2_add) * r_eval, 0.0, None)
        v_pred = np.sqrt(v2)

    if args.axisym_maps:
        vobs = np.asarray(dfg['Vobs_kms'], float)
    else:
        vobs = np.asarray(df['Vobs_kms'], float)

    # Logging for diagnostics
    print(f"[PDE] axisym_maps={args.axisym_maps} galaxy={args.galaxy} n_eval={len(r_eval)} n_obs={len(vobs)} n_bar={len(vbar)} grid_NR={R.shape[0]} grid_NZ={Z.shape[0]} rho_shape={rho.shape}")

    err = 100.0 * np.abs(v_pred - vobs) / np.maximum(vobs, 1e-9)
    # Optional scoring mask to exclude tiny inner radii from scoring (keep in plots)
    mask_score = (r_eval >= 0.8)
    if np.any(mask_score):
        med = float(np.median(100.0 - err[mask_score]))
    else:
        med = float(np.median(100.0 - err))

    od = Path(args.outdir)/args.tag
    od.mkdir(parents=True, exist_ok=True)
    # save RC table
    out_csv = od/'rc_pde_predictions.csv'
    pd.DataFrame({'R_kpc': r_eval, 'Vobs_kms': vobs, 'Vbar_kms': vbar,
                  'Vpred_pde_kms': v_pred, 'percent_close': (100.0 - err)}).to_csv(out_csv, index=False)
    # metrics
    (od/'summary.json').write_text(json.dumps({'S0_input': args.S0, 'rc_input_kpc': args.rc_kpc,
                                               'g0_kms2_per_kpc': args.g0_kms2_per_kpc,
                                               'rc_gamma': args.rc_gamma,
                                               'sigma_beta': args.sigma_beta,
                                               'rc_ref_kpc': args.rc_ref_kpc,
                                               'sigma0_Msun_pc2': args.sigma0_Msun_pc2,
                                               'S0_eff': float(params.S0), 'rc_eff_kpc': float(params.rc_kpc),
                                               'r_half_kpc': r_half, 'sigma_bar_Msun_pc2': sigma_bar_pc2,
                                               'use_sigma_screen': bool(args.use_sigma_screen),
                                               'sigma_star_Msun_per_pc2': float(args.sigma_star_Msun_per_pc2),
                                               'alpha_sigma': float(args.alpha_sigma),
                                               'n_sigma': float(args.n_sigma),
                                               'median_percent_close': med}, indent=2))

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(r_eval, vobs, 'k.', ms=3, label='obs')
    plt.plot(r_eval, vbar, 'c--', lw=1.2, label='GR (baryons only)')
    plt.plot(r_eval, v_pred, 'r-', lw=1.2, label='G³ prediction (single global tuple)')
    plt.xlabel('R [kpc]'); plt.ylabel('V [km/s]'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(od/'rc_pde_results.png', dpi=140); plt.close()

if __name__ == '__main__':
    main()
