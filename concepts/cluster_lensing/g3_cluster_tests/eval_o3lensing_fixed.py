# -*- coding: utf-8 -*-
"""
Evaluate non-local O3 lensing fixed parameters (from o3_nonlocal_global_best.json)
on real clusters and write per-cluster metrics.
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
    best_path = Path('concepts/cluster_lensing/g3_cluster_tests/outputs/o3_nonlocal_global_best.json')
    params = json.loads(best_path.read_text())['params']

    summaries = {}
    for name, z, theta_obs in clusters:
        try:
            g = pd.read_csv(base / name / 'gas_profile.csv')
        except Exception as e:
            print('[WARN]', name, e)
            continue
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
        # Newtonian baseline
        M_enc = 4.0 * np.pi * np.zeros_like(r)
        if r.size > 1:
            integrand = rho * r*r
            M_enc[1:] = 4.0 * np.pi * np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))
        g_N = G * M_enc / np.maximum(r*r, 1e-12)
        g_N[0] = g_N[1] if r.size > 1 else 0.0
        # Apply non-local O3 lensing to get g_lens(R)
        g_lens = o3l.apply_o3_lensing(g_N, r, Sigma_dyn_pc2, params, m_test_Msun=0.0)
        # Derive rho from g_lens: rho = (1/(4πG r^2)) d/dr [ r^2 g(R) ]
        y = r*r * g_lens
        dy_dr = np.gradient(y, r)
        rho_eff = dy_dr / (4.0 * np.pi * G * np.maximum(r*r, 1e-20))
        rho_eff[rho_eff < 0] = 0.0
        # Project to Sigma via Abel
        Sigma_kpc2 = np.zeros_like(r)
        for j, Rv in enumerate(r):
            mask = r > Rv
            rr = r[mask]
            if rr.size < 2:
                Sigma_kpc2[j] = 0.0
                continue
            integrand = 2.0 * rho_eff[mask] * rr / np.sqrt(np.maximum(rr*rr - Rv*Rv, 1e-20))
            Sigma_kpc2[j] = simpson(integrand, rr)
        Sigma_eff_pc2 = Sigma_kpc2 / 1e6
        _, kbar = kappa_bar(Sigma_eff_pc2, r, z)
        RE_kpc = theta_E_kpc(r, kbar)
        D_l = COSMO.angular_diameter_distance(z).to(u.kpc).value
        th_arc = (RE_kpc / D_l) * (180.0/np.pi) * 3600.0 if np.isfinite(RE_kpc) else float('nan')
        kmax = float(np.max(kbar))
        summaries[name] = {
            'theta_E_arcsec': float(th_arc) if np.isfinite(th_arc) else None,
            'theta_E_obs': theta_obs,
            'kappa_max': float(kmax),
            'kappa_50': float(np.interp(50, r, kbar)),
            'kappa_100': float(np.interp(100, r, kbar)),
            'params': params
        }
        print(f"{name}: θE={summaries[name]['theta_E_arcsec']} vs obs={theta_obs}, κ̄max={kmax:.2f}")
    outp = Path('concepts/cluster_lensing/g3_cluster_tests/outputs/o3_nonlocal_fixed_eval.json')
    outp.write_text(json.dumps(summaries, indent=2))
    print('Saved:', outp)

if __name__ == '__main__':
    main()
