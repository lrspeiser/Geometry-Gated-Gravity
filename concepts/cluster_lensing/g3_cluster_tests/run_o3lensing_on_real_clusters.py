# -*- coding: utf-8 -*-
"""
Run non-local O3 lensing (o3_lensing.apply_o3_lensing) on real clusters with exact cosmology.
Grid-search parameters and select by θE error (observed clusters) plus guardrails.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import simpson
import importlib.util as _ilu, sys as _sys

# Import non-local O3 lensing
_o3p = Path(__file__).resolve().parent / 'o3_lensing.py'
_spec = _ilu.spec_from_file_location('o3_lensing', str(_o3p))
o3l = _ilu.module_from_spec(_spec); _sys.modules['o3_lensing'] = o3l; _spec.loader.exec_module(o3l)

# Cosmology
from astropy.cosmology import Planck18 as COSMO
import astropy.units as u

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
        return float(R_kpc[idx[-1]])
    elif kmax >= 0.95:
        j = int(np.argmin(np.abs(kbar - 1.0)))
        j0 = max(0, min(len(R_kpc)-1, j-1))
        K1, K2 = kbar[j0], kbar[j]
        R1, R2 = R_kpc[j0], R_kpc[j]
        if abs(K2 - K1) < 1e-9:
            return float(R2)
        return float(R1 + (1.0 - K1) * (R2 - R1) / (K2 - K1))
    return float('nan')


def main():
    base = Path('data/clusters')
    clusters = [
        ('ABELL_1689', 0.184, 47.0),
        ('A2029', 0.0767, 28.0),
        ('A478', 0.0881, 31.0),
        ('A1795', 0.0622, None),
        ('ABELL_0426', 0.0179, None),
    ]
    # Precompute Σ_dyn (baryons-only) and Σ_bary (projected) per cluster
    items = []
    for name, z, theta_obs in clusters:
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
        r = r[i]; rho = rho[i]
        Sigma_dyn_pc2 = sigma_from_rho_abel(r, rho, r) / 1e6
        # Compute Newtonian g_dyn(r) from enclosed mass
        M_enc = 4.0 * np.pi * np.zeros_like(r)
        if r.size > 1:
            integrand = rho * r*r
            M_enc[1:] = 4.0 * np.pi * np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))
        g_N = G * M_enc / np.maximum(r*r, 1e-12)
        g_N[0] = g_N[1] if r.size > 1 else 0.0
        items.append((name, z, theta_obs, r, rho, Sigma_dyn_pc2, g_N))

    # Modest grid for o3_lensing (non-local env boost):
    ell3_list = [300.0, 500.0]
    Sigma_star3_list = [30.0, 50.0]
    beta3_list = [1.0, 1.5]
    r3_list = [80.0, 120.0]
    w3_list = [0.7, 1.0]
    xi3_list = [0.8, 1.2]
    A3_list = [1e-5, 1e-4, 5e-4]
    chi_list = [0.8, 1.0]

    best = None
    for ell3 in ell3_list:
        for S3 in Sigma_star3_list:
            for b3 in beta3_list:
                for r3 in r3_list:
                    for w3 in w3_list:
                        for xi3 in xi3_list:
                            for A3 in A3_list:
                                for chi in chi_list:
                                    params = {
                                        'ell3_kpc': ell3,
                                        'Sigma_star3_Msun_pc2': S3,
                                        'beta3': b3,
                                        'r3_kpc': r3,
                                        'w3_decades': w3,
                                        'xi3': xi3,
                                        'A3': A3,
                                        'chi': chi,
                                        'm_ref_Msun': 1.0,
                                        'm_floor_Msun': 1e-8,
                                    }
                                    total_err = 0.0
                                    ok = True
                                    for (name, z, theta_obs, r, rho, Sigma_dyn_pc2, g_N) in items:
                                        # Use baryon-only Newtonian field as g_dyn baseline
                                        g_lens = o3l.apply_o3_lensing(g_N, r, Sigma_dyn_pc2, params, m_test_Msun=0.0)
                                        # Approximate Σ_lens via proportional scaling of Σ_dyn by g_lens/g_N (avoid divide by zero)
                                        scale = g_lens / np.maximum(g_N, 1e-30)
                                        Sigma_eff_pc2 = Sigma_dyn_pc2 * scale
                                        _, kbar = kappa_bar(Sigma_eff_pc2, r, z)
                                        RE_kpc = theta_E_kpc(r, kbar)
                                        kmax = float(np.nanmax(kbar)) if np.all(np.isfinite(kbar)) else 0.0
                                        if kmax > 2.0:
                                            total_err += 5.0
                                        if theta_obs is not None:
                                            if not np.isfinite(RE_kpc):
                                                ok = False; total_err += 5.0
                                            else:
                                                D_l = COSMO.angular_diameter_distance(z).to(u.kpc).value
                                                theta_pred = (RE_kpc / D_l) * (180.0/np.pi) * 3600.0
                                                rel_err = abs(theta_pred - theta_obs)/max(theta_obs, 1e-6)
                                                band_pen = 0.0
                                                if RE_kpc < 50.0:
                                                    band_pen = (50.0 - RE_kpc)/50.0
                                                elif RE_kpc > 150.0:
                                                    band_pen = (RE_kpc - 150.0)/150.0
                                                total_err += 2.0 * (rel_err + 0.5 * band_pen)
                                        else:
                                            if np.isfinite(RE_kpc):
                                                total_err += 0.5
                                    score = total_err if ok else (total_err + 10.0)
                                    rec = {'params': params, 'score': float(score)}
                                    if (best is None) or (score < best['score']):
                                        best = rec
    out = Path('concepts/cluster_lensing/g3_cluster_tests/outputs/o3_nonlocal_global_best.json')
    out.write_text(json.dumps(best, indent=2))
    print('Best non-local O3 params:', best)

if __name__ == '__main__':
    main()
