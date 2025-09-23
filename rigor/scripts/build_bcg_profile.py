# -*- coding: utf-8 -*-
"""
rigor/scripts/build_bcg_profile.py

Build a simple 3D stellar density profile (BCG + optional ICL) and write
stars_profile.csv with columns: r_kpc,rho_star_Msun_per_kpc3.

- BCG uses a deprojected Sérsic (Prugniel & Simien) approximation.
- ICL uses a Hernquist distribution for simplicity.

Example:
  py -u rigor/scripts/build_bcg_profile.py \
    --cluster_dir data/clusters/A1795 \
    --Mtot_Msun 7.5e11 --Re_kpc 12.5 --sersic_n 4 \
    --icl_Mtot_Msun 3.0e11 --icl_a_kpc 100
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PI = np.pi


def prugniel_simien_rho(r, Mtot, Re, n):
    # Prugniel & Simien (1997) 3D approximation to deprojected Sérsic
    # rho(r) = rho0 * (r/Re)^(-p) * exp(-b * (r/Re)^(1/n))
    # p(n) ~ 1 - 0.6097/n + 0.05463/n^2; b(n) ~ 2n - 1/3 + 4/(405n) + ...
    n = float(n)
    if n <= 0:
        return np.zeros_like(r)
    p = 1.0 - 0.6097/n + 0.05463/(n*n)
    b = 2.0*n - 1.0/3.0 + 4.0/(405.0*n)
    x = np.maximum(r, 1e-12)/max(Re, 1e-12)
    shape = np.power(x, -p) * np.exp(-b * np.power(x, 1.0/n))
    # Normalize rho0 so that 4π ∫ r^2 rho dr = Mtot over [0,∞)
    # Numerically integrate to get rho0
    r_int = np.geomspace(1e-3*Re, 3e3*Re, 2000)
    shape_int = np.power(r_int/Re, -p) * np.exp(-b * np.power(r_int/Re, 1.0/n))
    integ = 4.0*PI * np.trapz(shape_int * (r_int**2), r_int)
    rho0 = Mtot / max(integ, 1e-30)
    return rho0 * shape


def hernquist_rho(r, Mtot, a):
    # ρ(r) = M a / (2π) * 1/( r (r+a)^3 )
    r = np.asarray(r, float)
    a = float(a)
    return (Mtot * a / (2.0*PI)) / (np.maximum(r, 1e-12) * (r + a)**3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster_dir', required=True)
    ap.add_argument('--Mtot_Msun', type=float, required=True, help='BCG total mass')
    ap.add_argument('--Re_kpc', type=float, required=True, help='BCG effective radius')
    ap.add_argument('--sersic_n', type=float, default=4.0, help='Sérsic index (default 4)')
    ap.add_argument('--icl_Mtot_Msun', type=float, default=0.0, help='Optional ICL mass')
    ap.add_argument('--icl_a_kpc', type=float, default=100.0, help='ICL scale length (Hernquist a)')
    ap.add_argument('--r_min_kpc', type=float, default=0.5)
    ap.add_argument('--r_max_kpc', type=float, default=2000.0)
    ap.add_argument('--N_r', type=int, default=400)
    args = ap.parse_args()

    cdir = Path(args.cluster_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    r = np.geomspace(max(args.r_min_kpc, 1e-3), max(args.r_max_kpc, 10.0), int(args.N_r))
    rho_bcg = prugniel_simien_rho(r, args.Mtot_Msun, args.Re_kpc, args.sersic_n)
    rho_icl = hernquist_rho(r, args.icl_Mtot_Msun, args.icl_a_kpc) if args.icl_Mtot_Msun > 0 else 0.0*r
    rho = rho_bcg + rho_icl

    # Integral check
    M_bcg = 4.0*PI * np.trapz(rho_bcg * (r**2), r)
    M_icl = 4.0*PI * np.trapz(rho_icl * (r**2), r)
    M_tot = 4.0*PI * np.trapz(rho * (r**2), r)
    print(f"[BCG] M_bcg≈{M_bcg:.3e} Msun  M_icl≈{M_icl:.3e} Msun  M_total≈{M_tot:.3e} Msun")

    out = cdir / 'stars_profile.csv'
    pd.DataFrame({'r_kpc': r, 'rho_star_Msun_per_kpc3': rho}).to_csv(out, index=False)
    print(f"[BCG] Wrote {out}")


if __name__ == '__main__':
    main()
