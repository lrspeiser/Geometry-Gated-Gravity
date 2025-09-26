# -*- coding: utf-8 -*-
"""
Run O3 (third-order lensing-only) on real cluster baryon profiles under exact cosmology.
- Loads data/clusters/<cluster>/{gas_profile.csv,stars_profile.csv}
- Converts n_e to rho_gas when needed; sums with rho_star
- Computes GR baseline (baryons only) kappa_bar(R) and theta_E
- Sweeps a small conservative O3 grid and selects the configuration minimizing |theta_E - theta_E_obs|
- Writes per-cluster JSON summary and prints a compact table

Clusters supported (with literature z and approximate theta_E when available):
- ABELL_1689 (z=0.184, theta_E≈47")
- A2029 (z=0.0767, theta_E≈28")
- A478 (z=0.0881, theta_E≈31")
- A1795 (z=0.0622, theta_E unknown -> optimize for kappa_bar≈1 near 50–100 kpc)
- ABELL_0426 (Perseus; z=0.0179, strong lensing rare -> report kappa_bar)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import simpson

# Import O3 slip helper
import importlib.util as _ilu, sys as _sys
_o3p = Path(__file__).resolve().parent / 'o3_slip.py'
_spec = _ilu.spec_from_file_location('o3_slip', str(_o3p))
o3s = _ilu.module_from_spec(_spec); _sys.modules['o3_slip'] = o3s; _spec.loader.exec_module(o3s)

# Constants
G = 4.300917270e-6  # kpc (km/s)^2 Msun^-1
MU_E = 1.17
M_P_G = 1.67262192369e-24  # g
KPC_CM = 3.0856775814913673e21
MSUN_G = 1.988409870698051e33

# Cosmology
try:
    from astropy.cosmology import Planck18 as COSMO
    import astropy.units as u
    ASTROPY_OK = True
except Exception:
    ASTROPY_OK = False


def ne_to_rho_gas_Msun_kpc3(ne_cm3: np.ndarray) -> np.ndarray:
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
    return Sig  # Msun/kpc^2


def kappa_bar_from_sigma(Sigma_pc2, R_kpc, z_l: float, z_s: float = 1.0):
    if not ASTROPY_OK:
        raise RuntimeError('astropy is required for exact cosmology distances')
    D_l = COSMO.angular_diameter_distance(z_l).to(u.kpc).value
    D_s = COSMO.angular_diameter_distance(z_s).to(u.kpc).value
    D_ls = COSMO.angular_diameter_distance_z1z2(z_l, z_s).to(u.kpc).value
    # Safety: distances must be positive finite and D_ls>0
    if not np.isfinite(D_l) or not np.isfinite(D_s) or not np.isfinite(D_ls) or (D_l <= 0) or (D_s <= 0) or (D_ls <= 0):
        return np.zeros_like(Sigma_pc2), np.zeros_like(Sigma_pc2), float('nan')
    c_kms = 299792.458
    Sigma_crit_kpc2 = (c_kms**2 / (4.0 * np.pi * G)) * (D_s / (D_l * D_ls))
    Sigma_crit = Sigma_crit_kpc2 / 1e6
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.where(np.isfinite(Sigma_crit) & (Sigma_crit > 0), Sigma_pc2 / Sigma_crit, 0.0)
    # Mean kappa
    kb = np.zeros_like(kappa)
    for i in range(1, len(R_kpc)):
        val = simpson(kappa[:i+1] * R_kpc[:i+1], R_kpc[:i+1])
        kb[i] = 2.0 * val / (R_kpc[i]**2)
    return kappa, kb, Sigma_crit


def theta_E_from_kbar(R_kpc, kbar, z_l, z_s=1.0):
    # Robust θE: prefer outermost crossing kbar>=1; else interpolate near the closest point to 1 if max>=0.95
    kmax = float(np.nanmax(kbar))
    if kmax >= 1.0:
        idx = np.where(kbar >= 1.0)[0]
        R_E = R_kpc[idx[-1]]
    elif kmax >= 0.95:
        j = int(np.argmin(np.abs(kbar - 1.0)))
        # pick neighbor with distinct value for linear interp
        if 0 < j < len(R_kpc)-1:
            j0 = j-1 if kbar[j-1] > kbar[j+1] else j+1
        elif j == 0:
            j0 = 1
        else:
            j0 = len(R_kpc)-2
        K1, K2 = kbar[j0], kbar[j]
        R1, R2 = R_kpc[j0], R_kpc[j]
        if abs(K2 - K1) < 1e-9:
            R_E = R2
        else:
            R_E = R1 + (1.0 - K1) * (R2 - R1) / (K2 - K1)
    else:
        return np.nan
    D_l = COSMO.angular_diameter_distance(z_l).to(u.kpc).value
    theta_E_arcsec = (R_E / D_l) * (180.0/np.pi) * 3600.0
    return float(theta_E_arcsec)


def load_cluster_rho(cluster_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    g = pd.read_csv(cluster_dir / 'gas_profile.csv')
    r_g = np.asarray(g['r_kpc'], float)
    if 'rho_gas_Msun_per_kpc3' in g.columns:
        rho_gas = np.asarray(g['rho_gas_Msun_per_kpc3'], float)
    elif 'n_e_cm3' in g.columns:
        rho_gas = ne_to_rho_gas_Msun_kpc3(np.asarray(g['n_e_cm3'], float))
    else:
        raise ValueError('gas_profile.csv missing required columns')
    s_path = cluster_dir / 'stars_profile.csv'
    if s_path.exists():
        s = pd.read_csv(s_path)
        r_s = np.asarray(s['r_kpc'], float)
        rho_star = np.asarray(s['rho_star_Msun_per_kpc3'], float)
    else:
        r_s = r_g
        rho_star = np.zeros_like(r_g)
    # unify grid
    r_all = np.union1d(r_g, r_s)
    rho_total = np.interp(r_all, r_g, rho_gas) + np.interp(r_all, r_s, rho_star)
    # clean
    mask = np.isfinite(r_all) & np.isfinite(rho_total) & (r_all > 0)
    r = r_all[mask]
    rho = np.clip(rho_total[mask], 0.0, None)
    # ensure sorted
    idx = np.argsort(r)
    return r[idx], rho[idx]


def main():
    base = Path('data/clusters')
    clusters = [
        ('ABELL_1689', 0.184, 47.0),
        ('A2029', 0.0767, 28.0),
        ('A478', 0.0881, 31.0),
        ('A1795', 0.0622, None),
        ('ABELL_0426', 0.0179, None),
    ]
    out_dir = Path('concepts/cluster_lensing/g3_cluster_tests/outputs')
    out_dir.mkdir(parents=True, exist_ok=True)

    # conservative O3 grid (retuned safer defaults)
    ell3_list = [200.0, 300.0, 500.0]
    Sigma_star3_list = [50.0, 80.0]
    beta3_list = [0.8, 1.0, 1.3]
    # slip model has no xi3/chi; keep placeholders removed
    A3_list = [0.5, 1.0, 2.0, 3.0]
    rboost_list = [80.0, 100.0, 120.0]
    wdec_list = [0.6, 0.8, 1.0]
    ccurv_list = [0.6, 0.8, 1.0]

    summaries = {}
    for name, z_l, theta_obs in clusters:
        cdir = base / name
        if not (cdir / 'gas_profile.csv').exists():
            continue
        try:
            r, rho = load_cluster_rho(cdir)
        except Exception as e:
            print(f'[WARN] {name}: {e}')
            continue
        # GR baseline: Sigma_dyn from rho via Abel
        Sigma_kpc2 = sigma_from_rho_abel(r, rho, r)
        Sigma_pc2 = Sigma_kpc2 / 1e6
        # Compute GR lensing
        _, kbar_GR, _ = kappa_bar_from_sigma(Sigma_pc2, r, z_l)
        theta_GR = theta_E_from_kbar(r, kbar_GR, z_l)

        best = None
        for ell3 in ell3_list:
            for S3 in Sigma_star3_list:
                for b3 in beta3_list:
                    for A3 in A3_list:
                        for rb in rboost_list:
                            for wd in wdec_list:
                                for cc in ccurv_list:
                                    params = {
                                        'A3': A3,
                                        'Sigma_star': S3,
                                        'beta': b3,
                                        'r_boost_kpc': rb,
                                        'w_dec': wd,
                                        'ell_kpc': ell3,
                                        'c_curv': cc,
                                    }
                                    Sigma_lens_pc2 = o3s.apply_slip(r, Sigma_pc2, rho, params)
                                    # Compute kappa_bar & theta_E
                                    _, kbar_lens, _ = kappa_bar_from_sigma(Sigma_lens_pc2, r, z_l)
                                    theta = theta_E_from_kbar(r, kbar_lens, z_l)
                                    # Guard against runaway solutions
                                    kmax = float(np.nanmax(kbar_lens)) if np.all(np.isfinite(kbar_lens)) else float('inf')
                                    if theta_obs is not None and np.isfinite(theta):
                                        # Hard gate: skip if θE overshoots badly
                                        if theta > 2.0 * theta_obs:
                                            continue
                                        err = abs(theta - theta_obs)
                                    else:
                                        # Unknown θE: target κ̄≈1 in 50–100 kpc band and avoid huge cores
                                        band = (r >= 50) & (r <= 100)
                                        k_med = float(np.median(kbar_lens[band])) if np.any(band) else float('nan')
                                        err = abs(k_med - 1.0) + 0.25 * max(0.0, kmax - 2.0)  # penalize extreme cores
                                    rec = {
                                        'ell3_kpc': ell3, 'Sigma_star3': S3, 'beta3': b3, 'A3': A3,
                                        'r_boost_kpc': rb, 'w_dec': wd, 'c_curv': cc,
                                        'theta_E_arcsec': float(theta) if np.isfinite(theta) else None,
                                        'theta_E_obs': theta_obs,
                                        'err': float(err),
                                        'kappa_max': float(kmax) if np.isfinite(kmax) else None,
                                        'kappa_50': float(np.interp(50, r, kbar_lens)),
                                        'kappa_100': float(np.interp(100, r, kbar_lens)),
                                        'theta_E_GR': float(theta_GR) if np.isfinite(theta_GR) else None,
                                    }
                                    if (best is None) or (err < best['err']):
                                        best = rec
        summaries[name] = best
        with open(out_dir / f'{name}_o3_summary.json', 'w') as f:
            json.dump(best, f, indent=2)
        theta_str = f"{best['theta_E_arcsec']:.2f}" if isinstance(best.get('theta_E_arcsec'), (int, float)) else 'n/a'
        theta_gr_str = f"{best['theta_E_GR']:.2f}" if isinstance(best.get('theta_E_GR'), (int, float)) else 'n/a'
        kmax_str = f"{best['kappa_max']:.2f}" if isinstance(best.get('kappa_max'), (int, float)) else 'n/a'
        print(f"{name}: θE_pred={theta_str} vs θE_obs={best['theta_E_obs']} (GR={theta_gr_str}), κ̄max={kmax_str}")

    with open(out_dir / 'o3_real_clusters_summary.json', 'w') as f:
        json.dump(summaries, f, indent=2)
    print('Saved cluster O3 summary to', out_dir / 'o3_real_clusters_summary.json')


if __name__ == '__main__':
    main()
