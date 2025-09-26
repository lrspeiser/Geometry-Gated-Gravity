# -*- coding: utf-8 -*-
"""
Evaluate O3 slip with a fixed parameter set (from o3_slip_global_best.json) on real clusters.
Print per-cluster θE (arcsec), θE_obs, κ̄max, κ̄@50, κ̄@100 with exact cosmology.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import simpson
import importlib.util as _ilu, sys as _sys

# Import slip
_o3p = Path(__file__).resolve().parent / 'o3_slip.py'
_spec = _ilu.spec_from_file_location('o3_slip', str(_o3p))
o3s = _ilu.module_from_spec(_spec); _sys.modules['o3_slip'] = o3s; _spec.loader.exec_module(o3s)

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


def theta_E(R_kpc, kbar, z_l):
    kmax = float(np.nanmax(kbar))
    if kmax >= 1.0:
        idx = np.where(kbar >= 1.0)[0]
        RE = R_kpc[idx[-1]]
    elif kmax >= 0.95:
        j = int(np.argmin(np.abs(kbar - 1.0)))
        j0 = max(0, min(len(R_kpc)-1, j-1))
        K1, K2 = kbar[j0], kbar[j]
        R1, R2 = R_kpc[j0], R_kpc[j]
        if abs(K2 - K1) < 1e-9:
            RE = R2
        else:
            RE = R1 + (1.0 - K1) * (R2 - R1) / (K2 - K1)
    else:
        return float('nan')
    D_l = COSMO.angular_diameter_distance(z_l).to(u.kpc).value
    return float((RE / D_l) * (180.0/np.pi) * 3600.0)


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


def main():
    base = Path('data/clusters')
    clusters = [
        ('ABELL_1689', 0.184, 47.0),
        ('A2029', 0.0767, 28.0),
        ('A478', 0.0881, 31.0),
        ('A1795', 0.0622, None),
        ('ABELL_0426', 0.0179, None),
    ]
    best_path = Path('concepts/cluster_lensing/g3_cluster_tests/outputs/o3_slip_global_best.json')
    params = json.loads(best_path.read_text())['params']
    summaries = {}
    for name, z, theta_obs in clusters:
        try:
            r, rho = load_cluster(name, base)
        except Exception as e:
            print('[WARN]', name, e)
            continue
        Sigma_dyn = sigma_from_rho_abel(r, rho, r) / 1e6
        Sigma_lens = o3s.apply_slip(r, Sigma_dyn, rho, params)
        _, kbar = kappa_bar(Sigma_lens, r, z)
        th = theta_E(r, kbar, z)
        kmax = float(np.max(kbar))
        summaries[name] = {
            'theta_E_arcsec': float(th) if np.isfinite(th) else None,
            'theta_E_obs': theta_obs,
            'kappa_max': float(kmax),
            'kappa_50': float(np.interp(50, r, kbar)),
            'kappa_100': float(np.interp(100, r, kbar)),
            'params': params
        }
        print(f"{name}: θE={summaries[name]['theta_E_arcsec']} vs obs={theta_obs}, κ̄max={kmax:.2f}")
    outp = Path('concepts/cluster_lensing/g3_cluster_tests/outputs/o3_slip_fixed_eval.json')
    outp.write_text(json.dumps(summaries, indent=2))
    print('Saved:', outp)

if __name__ == '__main__':
    main()
