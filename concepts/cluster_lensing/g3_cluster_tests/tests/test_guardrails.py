# -*- coding: utf-8 -*-
"""
Guardrail test: ensure core and outer convergence guardrails hold for non-lensing/low-z clusters.
We target ABELL_0426 (Perseus, low-z non-lensing) and A1795 (moderate z, no observed θE).
Asserts that κ̄_max <= κ̄_core and κ̄(500 kpc) <= κ̄_500 under regulated slip with best params.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import simpson
import importlib.util as _ilu, sys as _sys
import pytest

# Import slip
_O3P = Path(__file__).resolve().parents[1] / 'o3_slip.py'
_spec = _ilu.spec_from_file_location('o3_slip', str(_O3P))
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


def _best_params_or_default(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text()).get('params', {})
        except Exception:
            pass
    return {'A3': 0.5, 'Sigma_star': 50.0, 'beta': 1.0, 'r_boost_kpc': 80.0, 'w_dec': 0.6, 'ell_kpc': 0.0, 'c_curv': 0.8}


def _regulated_params(params: dict, r: np.ndarray, Sigma_dyn: np.ndarray, z: float,
                      Sigma_star: float = 50.0, gamma_sigma: float = 0.5, gamma_z: float = 1.0,
                      tau_curv: float = 0.3, lowz_veto_z: float = 0.03) -> dict:
    mask_band = (r >= 30.0) & (r <= 100.0)
    Sigma_mean = float(np.mean(Sigma_dyn[mask_band])) if np.any(mask_band) else float(np.mean(Sigma_dyn))
    A3_eff = float(params.get('A3', 1.0)) * (max(Sigma_mean, 1e-9)/Sigma_star) ** (-gamma_sigma) * (1.0 + float(z)) ** (-gamma_z)
    logr = np.log(np.maximum(r, 1e-12))
    lnS = np.log(np.maximum(Sigma_dyn, 1e-30))
    d1 = np.gradient(lnS, logr)
    d2 = np.gradient(d1, logr)
    curv_band = float(np.min(d2[mask_band])) if np.any(mask_band) else float(np.min(d2))
    veto = (float(z) < lowz_veto_z) and (curv_band > -tau_curv)
    lp = dict(params)
    lp['A3'] = 0.0 if veto else A3_eff
    return lp


def _compute_kbars(name: str, z: float, base: Path, params: dict):
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
    i = np.argsort(r); r = r[i]; rho = rho[i]
    Sigma_dyn = sigma_from_rho_abel(r, rho, r) / 1e6
    lp = _regulated_params(params, r, Sigma_dyn, z)
    Sigma_lens = o3s.apply_slip(r, Sigma_dyn, rho, lp)
    _, kbar = kappa_bar(Sigma_lens, r, z)
    return r, kbar


def test_guardrails_non_lensing_clusters():
    base = Path(__file__).resolve().parents[4] / 'data' / 'clusters'
    params = _best_params_or_default(Path(__file__).resolve().parents[1] / 'outputs' / 'o3_slip_global_best.json')

    # Guardrail thresholds (should mirror defaults used in search)
    kappa_core = 1.8
    kappa_500_cap = 0.6

    samples = [
        ('ABELL_0426', 0.0179),
        ('A1795', 0.0622),
    ]
    for name, z in samples:
        r, kbar = _compute_kbars(name, z, base, params)
        kmax = float(np.nanmax(kbar))
        k500 = float(np.interp(500.0, r, kbar))
        assert kmax <= kappa_core + 1e-6
        assert k500 <= kappa_500_cap + 1e-6
