# -*- coding: utf-8 -*-
"""
root-m/pde/predict_hse.py

Hydrostatic prediction from PDE total acceleration field.
Given: n_e(r) (or rho_gas(r)), and g_tot(r) (km^2 s^-2 / kpc),
compute kT(r) via integral form:
    I(r) = ∫_r^{Rmax} n_g(r') g_eff(r') dr',  with n_g = (MU_E/MU) n_e
    kT = (MU * m_p) * I(r) [converted to keV]

Optionally include a non-thermal pressure fraction f_nt(r) by setting
    g_eff(r) = (1 - f_nt(r)) * g_tot(r)
which reduces the thermal support in the outskirts.

For now we provide a simple spherical-extraction from a (Z,R) field by
sampling along z=0 and treating it as g_tot(r) ~ g_tot(R, z=0).
"""
import numpy as np
import pandas as pd
from pathlib import Path

MU = 0.61
MU_E = 1.17
M_P = 1.67262192369e-27  # kg
J_PER_KEV = 1.602176634e-16
KM2_PER_S2_TO_J_PER_KG = 1.0e6
KMS2_TO_KEV = (MU * M_P * KM2_PER_S2_TO_J_PER_KG) / J_PER_KEV


def reversed_cumtrapz(x, y):
    x = np.asarray(x); y = np.asarray(y)
    S = np.zeros_like(y)
    if x.size >= 2:
        dx = np.diff(x)
        area = 0.5 * (y[1:] + y[:-1]) * dx
        S[:-1] = area[::-1].cumsum()[::-1]
    return S


def fnt_profile(r_kpc: np.ndarray, fnt0: float, n: float, r500_kpc: float, fnt_max: float = 0.3) -> np.ndarray:
    r = np.asarray(r_kpc, float)
    r500 = max(float(r500_kpc), 1e-9)
    prof = float(max(fnt0, 0.0)) * np.power(np.maximum(r, 0.0)/r500, float(n))
    return np.clip(prof, 0.0, float(fnt_max))


def kT_from_ne_and_gtot(r_kpc, n_e_cm3, g_tot_kms2_per_kpc, f_nt: np.ndarray | None = None):
    n_g_cm3 = (MU_E/MU) * np.asarray(n_e_cm3)
    g_tot = np.asarray(g_tot_kms2_per_kpc)
    if f_nt is not None:
        f_nt_cl = np.clip(np.asarray(f_nt), 0.0, 0.9)
        g_eff = (1.0 - f_nt_cl) * g_tot
    else:
        g_eff = g_tot
    I = reversed_cumtrapz(r_kpc, n_g_cm3 * g_eff)
    potential_kms2 = I / np.maximum(n_g_cm3, 1e-99)
    kT_keV = KMS2_TO_KEV * potential_kms2
    return kT_keV
