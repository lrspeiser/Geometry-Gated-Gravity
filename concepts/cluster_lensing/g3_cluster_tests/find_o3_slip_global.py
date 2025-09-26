# -*- coding: utf-8 -*-
"""
Global search for O3 slip parameters across multiple clusters with exact cosmology.
Find a single parameter set minimizing θE error on clusters with observed θE while
avoiding spurious θE on clusters without observed strong lensing.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import simpson
import importlib.util as _ilu, sys as _sys
import argparse

# Import slip
_o3p = Path(__file__).resolve().parent / 'o3_slip.py'
_spec = _ilu.spec_from_file_location('o3_slip', str(_o3p))
o3s = _ilu.module_from_spec(_spec); _sys.modules['o3_slip'] = o3s; _spec.loader.exec_module(o3s)

# Cosmology
try:
    from astropy.cosmology import Planck18 as COSMO
    import astropy.units as u
    ASTROPY_OK = True
except Exception:
    ASTROPY_OK = False

G = 4.300917270e-6  # kpc (km/s)^2 Msun^-1
MU_E = 1.17
M_P_G = 1.67262192369e-24
KPC_CM = 3.0856775814913673e21
MSUN_G = 1.988409870698051e33


def ne_to_rho(ne_cm3: np.ndarray) -> np.ndarray:
    rho_g_cm3 = MU_E * M_P_G * np.asarray(ne_cm3)
    return rho_g_cm3 * (KPC_CM**3) / MSUN_G


def sigma_from_rho_abel(r_arr, rho_arr, R_eval):
    Sig = np.zeros_like(R_eval)
    for j, Rv in enumerate(R_eval):
        mask = r_arr > Rv
        rr = r_arr[mask]
        if rr.size < 2:
            Sig[j] = 0.0
            continue
        integ = 2.0 * rho_arr[mask] * rr / np.sqrt(np.maximum(rr**2 - Rv**2, 1e-20))
        Sig[j] = simpson(integ, rr)
    return Sig


def kappa_bar(Sigma_pc2, R_kpc, z_l):
    D_l = COSMO.angular_diameter_distance(z_l).to(u.kpc).value
    D_s = COSMO.angular_diameter_distance(1.0).to(u.kpc).value
    D_ls = COSMO.angular_diameter_distance_z1z2(z_l, 1.0).to(u.kpc).value
    c_kms = 299792.458
    Sigma_crit_kpc2 = (c_kms**2 / (4.0 * np.pi * G)) * (D_s / (D_l * D_ls))
    Sigma_crit = Sigma_crit_kpc2 / 1e6
    kap = Sigma_pc2 / np.maximum(Sigma_crit, 1e-30)
    kb = np.zeros_like(kap)
    for i in range(1, len(R_kpc)):
        val = simpson(kap[:i+1] * R_kpc[:i+1], R_kpc[:i+1])
        kb[i] = 2.0 * val / (R_kpc[i]**2)
    return kap, kb


def theta_E_kpc(R_kpc, kbar):
    kmax = float(np.nanmax(kbar))
    if kmax >= 1.0:
        idx = np.where(kbar >= 1.0)[0]
        RE = R_kpc[idx[-1]]
        return float(RE)
    elif kmax >= 0.95:
        j = int(np.argmin(np.abs(kbar - 1.0)))
        j0 = max(0, min(len(R_kpc)-1, j-1))
        K1, K2 = kbar[j0], kbar[j]
        R1, R2 = R_kpc[j0], R_kpc[j]
        if abs(K2 - K1) < 1e-9:
            return float(R2)
        return float(R1 + (1.0 - K1) * (R2 - R1) / (K2 - K1))
    else:
        return float('nan')

def theta_E_arcsec_from_kpc(R_E_kpc, z_l):
    if not np.isfinite(R_E_kpc):
        return float('nan')
    D_l = COSMO.angular_diameter_distance(z_l).to(u.kpc).value
    return float((R_E_kpc / D_l) * (180.0/np.pi) * 3600.0)


def load_cluster(name: str, base: Path):
    g = pd.read_csv(base / name / 'gas_profile.csv')
    r_g = np.asarray(g['r_kpc'], float)
    rho_g = np.asarray(g['rho_gas_Msun_per_kpc3'], float) if 'rho_gas_Msun_per_kpc3' in g.columns else ne_to_rho(np.asarray(g['n_e_cm3'], float))
    s_path = base / name / 'stars_profile.csv'
    if s_path.exists():
        s = pd.read_csv(s_path)
        r_s = np.asarray(s['r_kpc'], float)
        rho_s = np.asarray(s['rho_star_Msun_per_kpc3'], float)
    else:
        r_s = r_g; rho_s = np.zeros_like(r_g)
    r = np.union1d(r_g, r_s)
    rho = np.interp(r, r_g, rho_g) + np.interp(r, r_s, rho_s)
    m = np.isfinite(r) & np.isfinite(rho) & (r>0)
    r = r[m]; rho = np.clip(rho[m], 0.0, None)
    i = np.argsort(r)
    return r[i], rho[i]


def _parse_weights(s: str) -> dict:
    default = {
        'ABELL_1689': 0.4,
        'A2029': 0.3,
        'A478': 0.25,
        'A1795': 0.04,
        'ABELL_0426': 0.01,
    }
    if not s:
        return default
    out = {}
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if '=' not in tok:
            continue
        k, v = tok.split('=', 1)
        k = k.strip().upper()
        name = k
        if k in ('A1689', 'ABELL1689'): name = 'ABELL_1689'
        elif k in ('PERSEUS', 'ABELL0426', 'A0426'): name = 'ABELL_0426'
        elif k == 'A2029': name = 'A2029'
        elif k == 'A478': name = 'A478'
        elif k == 'A1795': name = 'A1795'
        else:
            name = k  # allow direct keys
        try:
            out[name] = float(v)
        except Exception:
            pass
    # merge with defaults
    for k, v in default.items():
        out.setdefault(k, v)
    return out

def main():
    if not ASTROPY_OK:
        raise SystemExit('astropy required')
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='', help='e.g. A1689=0.4,A2029=0.3,A478=0.25,A1795=0.04,PERSEUS=0.01')
    ap.add_argument('--kappa_core', type=float, default=1.8)
    ap.add_argument('--Rmin_kpc', type=float, default=30.0)
    ap.add_argument('--kappa_500', type=float, default=0.6)
    ap.add_argument('--overshoot_mult', type=float, default=2.5)
    ap.add_argument('--tau_curv', type=float, default=0.3)
    ap.add_argument('--lowz_veto_z', type=float, default=0.03)
    ap.add_argument('--gamma_sigma', type=float, default=0.5)
    ap.add_argument('--gamma_z', type=float, default=1.0)
    ap.add_argument('--Sigma_star', type=float, default=50.0)
    args = ap.parse_args()
    W = _parse_weights(args.weights)

    base = Path('data/clusters')
    clusters = [
        ('ABELL_1689', 0.184, 47.0),
        ('A2029', 0.0767, 28.0),
        ('A478', 0.0881, 31.0),
        ('A1795', 0.0622, None),
        ('ABELL_0426', 0.0179, None),
    ]
    # precompute Sigma_dyn per cluster
    items = []
    for name, z, theta_obs in clusters:
        try:
            r, rho = load_cluster(name, base)
            Sigma_dyn = sigma_from_rho_abel(r, rho, r) / 1e6
            items.append((name, z, theta_obs, r, rho, Sigma_dyn))
        except Exception as e:
            print('[WARN]', name, e)
    # normalize weight dict with defaults for any missing clusters
    for name, _, _ in clusters:
        W.setdefault(name, 0.1)
    # grid
    A3_list = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0, 100.0]
    Sigma_star_list = [30.0, 50.0, 150.0, 500.0]
    beta_list = [0.7, 1.0, 1.5, 1.8]
    rboost_list = [60.0, 80.0, 120.0, 200.0, 300.0]
    wdec_list = [0.4, 0.6, 0.8, 1.0]
    ell_list = [0.0, 150.0, 300.0, 500.0, 800.0]
    ccurv_list = [0.6, 0.8, 1.0]

    best = None
    for A3 in A3_list:
        for Sstar in Sigma_star_list:
            for beta in beta_list:
                for rb in rboost_list:
                    for wd in wdec_list:
                        for ell in ell_list:
                            for cc in ccurv_list:
                                params = {'A3':A3,'Sigma_star':Sstar,'beta':beta,'r_boost_kpc':rb,'w_dec':wd,'ell_kpc':ell,'c_curv':cc}
                                total_err = 0.0
                                ok_global = True
                                for (name, z, theta_obs, r, rho, Sigma_dyn) in items:
                                    w = float(W.get(name, 1.0))
                                    # Environment + redshift amplitude regulation
                                    mask_band = (r >= 30.0) & (r <= 100.0)
                                    if not np.any(mask_band):
                                        Sigma_mean = float(np.mean(Sigma_dyn))
                                    else:
                                        Sigma_mean = float(np.mean(Sigma_dyn[mask_band]))
                                    A3_eff = A3 * (max(Sigma_mean, 1e-9) / max(args.Sigma_star, 1e-9)) ** (-args.gamma_sigma) * (1.0 + float(z)) ** (-args.gamma_z)
                                    # Low-z curvature veto
                                    logr = np.log(np.maximum(r, 1e-12))
                                    lnS = np.log(np.maximum(Sigma_dyn, 1e-30))
                                    d1 = np.gradient(lnS, logr)
                                    d2 = np.gradient(d1, logr)
                                    curv_band = float(np.min(d2[mask_band])) if np.any(mask_band) else float(np.min(d2))
                                    local_params = dict(params)
                                    if (float(z) < args.lowz_veto_z) and (curv_band > -args.tau_curv):
                                        local_params['A3'] = 0.0
                                    else:
                                        local_params['A3'] = A3_eff
                                    Sigma_lens = o3s.apply_slip(r, Sigma_dyn, rho, local_params)
                                    kap, kbar = kappa_bar(Sigma_lens, r, z)
                                    RE_kpc = theta_E_kpc(r, kbar)
                                    th_arc = theta_E_arcsec_from_kpc(RE_kpc, z)
                                    kmax = float(np.nanmax(kbar)) if np.all(np.isfinite(kbar)) else 0.0
                                    # Core safety (weighted)
                                    if kmax > args.kappa_core:
                                        total_err += w * 7.5
                                    # kappa at 500 kpc guardrail (weighted)
                                    kbar_500 = float(np.interp(500.0, r, kbar))
                                    if kbar_500 > args.kappa_500:
                                        total_err += w * 1.0 * (kbar_500 - args.kappa_500)
                                    if theta_obs is not None:
                                        if not np.isfinite(th_arc):
                                            ok_global = False; total_err += w * 7.5
                                        else:
                                            if th_arc > args.overshoot_mult*theta_obs:
                                                ok_global = False; total_err += w * 10.0
                                            if np.isfinite(RE_kpc):
                                                rel_err = abs(th_arc - theta_obs)/max(theta_obs, 1e-6)
                                                band_pen = 0.0
                                                # Central crossing penalty using innermost crossing (weighted)
                                                idxs = np.where(kbar >= 1.0)[0]
                                                if idxs.size > 0:
                                                    RE_min = float(r[idxs[0]])
                                                    if RE_min < args.Rmin_kpc:
                                                        band_pen += (args.Rmin_kpc - RE_min)/max(args.Rmin_kpc, 1e-6)
                                                if RE_kpc < 50.0:
                                                    band_pen += (50.0 - RE_kpc)/50.0
                                                elif RE_kpc > 150.0:
                                                    band_pen += (RE_kpc - 150.0)/150.0
                                                total_err += w * (2.0 * rel_err + 1.5 * band_pen)
                                            else:
                                                ok_global = False; total_err += w * 7.5
                                    else:
                                        if np.isfinite(th_arc):
                                            # Penalty for spurious θE (weighted), stronger at low-z
                                            spur = 0.2
                                            if float(z) < 0.03:
                                                spur += 0.8
                                            # additional penalty if central crossing occurs very small
                                            idxs = np.where(kbar >= 1.0)[0]
                                            if idxs.size > 0 and float(r[idxs[0]]) < args.Rmin_kpc:
                                                spur += 0.5
                                            total_err += w * spur
                                # Overall score
                                score = total_err if ok_global else (total_err + 10.0)
                                rec = {'params': params, 'score': float(score)}
                                if (best is None) or (score < best['score']):
                                    best = rec
    out = Path('concepts/cluster_lensing/g3_cluster_tests/outputs/o3_slip_global_best.json')
    out.write_text(json.dumps(best, indent=2))
    print('Best global O3 slip params:', best)

if __name__ == '__main__':
    main()
