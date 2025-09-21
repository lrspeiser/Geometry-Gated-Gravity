# -*- coding: utf-8 -*-
"""
root-m/pde/baryon_maps.py

Build baryon density maps for the PDE solver.
Provides spherical builders for clusters (from n_e(r), rho_star(r)), and
spherical-equivalent maps for SPARC (from Vbar(r)).

All maps are returned as (Z,R) grids with rho_b(R,z) in Msun/kpc^3.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)
MU_E = 1.17
M_P_G = 1.67262192369e-24  # g
KPC_CM = 3.0856775814913673e21
MSUN_G = 1.988409870698051e33


def ne_to_rho_gas_Msun_kpc3(ne_cm3: np.ndarray) -> np.ndarray:
    rho_g_cm3 = MU_E * M_P_G * np.asarray(ne_cm3)
    return rho_g_cm3 * (KPC_CM**3) / MSUN_G


def spherical_from_radial_profile(r_kpc: np.ndarray, rho_r: np.ndarray,
                                  R_max: float, Z_max: float,
                                  NR: int, NZ: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a (Z,R) map from a spherical radial density profile rho(r).
    """
    R = np.linspace(0.0, R_max, NR)
    Z = np.linspace(-Z_max, Z_max, NZ)
    RR, ZZ = np.meshgrid(R, Z)
    rr = np.sqrt(RR*RR + ZZ*ZZ)
    rho = np.interp(rr, r_kpc, rho_r, left=rho_r[0], right=rho_r[-1])
    return Z, R, rho


def cluster_map_from_csv(cluster_dir: Path, R_max: float=1500.0, Z_max: float=1500.0,
                         NR: int=128, NZ: int=128,
                         clump: float = 1.0):
    cdir = Path(cluster_dir)
    g = pd.read_csv(cdir/"gas_profile.csv")
    r = np.asarray(g['r_kpc'], float)
    if 'rho_gas_Msun_per_kpc3' in g.columns:
        rho_gas = np.asarray(g['rho_gas_Msun_per_kpc3'], float)
        # apply clumping correction as sqrt(C)
        rho_gas = rho_gas * float(np.sqrt(max(clump, 1.0)))
    else:
        ne = np.asarray(g['n_e_cm3'], float)
        ne_eff = ne * float(np.sqrt(max(clump, 1.0)))
        rho_gas = ne_to_rho_gas_Msun_kpc3(ne_eff)

    # stars (optional)
    rho_star = None
    s_path = cdir/"stars_profile.csv"
    if s_path.exists():
        s = pd.read_csv(s_path)
        rs = np.asarray(s['r_kpc'], float)
        rho_s = np.asarray(s['rho_star_Msun_per_kpc3'], float)
        # align onto r grid for simplicity
        rho_star = np.interp(r, rs, rho_s, left=rho_s[0], right=rho_s[-1])
    rho_b = rho_gas if rho_star is None else (rho_gas + rho_star)

    return spherical_from_radial_profile(r, rho_b, R_max, Z_max, NR, NZ)


def sparc_map_from_predictions(csv_path: Path, R_max: float=80.0, Z_max: float=80.0,
                               NR: int=128, NZ: int=128):
    """
    Build a spherical-equivalent rho_b(r) from Vbar(r) via
        M(<r) = Vbar^2 r / G,  rho(r) = 1/(4π r^2) dM/dr.
    Then map onto (Z,R).
    """
    df = pd.read_csv(csv_path)
    r = np.asarray(df['R_kpc'], float)
    vbar = np.asarray(df['Vbar_kms'], float)
    r, idx = np.unique(r, return_index=True)
    vbar = vbar[idx]
    # smooth derivative via finite diff
    M = (vbar*vbar) * r / G
    dMdr = np.gradient(M, r)
    rho = dMdr / (4.0*np.pi * r*r + 1e-12)
    # clip and smooth
    rho = np.clip(rho, 0.0, None)
    return spherical_from_radial_profile(r, rho, R_max, Z_max, NR, NZ)
