# -*- coding: utf-8 -*-
"""
root-m/pde/tests_pde_sanity.py

Quick sanity checks for the PDE kernel:
1) Compact Gaussian mass (spherical-like) -> expect g_phi ~ 1/R tail and v_tail^2 ~ const at large R.
2) Scaling with S0: plateau of v_tail^2 should grow with S0 (roughly like S0^0.5 for this kernel family).

Outputs simple JSON with plateau metrics so we can iterate quickly.
"""
import json
import numpy as np
from pathlib import Path

# dynamic imports from local files
import importlib.util as _ilu
from pathlib import Path as _P
_pkg = _P(__file__).resolve().parent

import sys as _sys
spec = _ilu.spec_from_file_location('solve_phi', str(_pkg/'solve_phi.py'))
solve_phi = _ilu.module_from_spec(spec)
_sys.modules['solve_phi'] = solve_phi  # register for dataclasses
spec.loader.exec_module(solve_phi)
SolverParams = solve_phi.SolverParams
solve_axisym = solve_phi.solve_axisym

def gaussian_rho(NZ=128, NR=128, Rmax=80.0, Zmax=80.0, Mtot=1.0e11, r0=2.0):
    R = np.linspace(0.0, Rmax, NR)
    Z = np.linspace(-Zmax, Zmax, NZ)
    RR, ZZ = np.meshgrid(R, Z)
    r = np.sqrt(RR*RR + ZZ*ZZ)
    # 3D Gaussian rho(r) ~ exp(-r^2/(2r0^2)) normalized to Mtot
    rho_unn = np.exp(-0.5*(r/r0)**2)
    # normalize: integrate over cylindrical grid approximate volume
    dR = R[1]-R[0]; dZ = Z[1]-Z[0]
    vol = 2*np.pi*RR * dR * dZ
    M_est = float(np.sum(rho_unn * vol))
    rho = rho_unn * (Mtot / max(M_est, 1e-30))
    return Z, R, rho


def plateau_metric(R, gR0, r_inner=20.0, r_outer=60.0):
    mask = (R>=r_inner) & (R<=r_outer)
    if not np.any(mask):
        return None
    v2_extra = R[mask] * gR0[mask]
    return float(np.median(v2_extra)), float(np.std(v2_extra))


def run_once(S0=1e-6, rc=15.0):
    Z, R, rho = gaussian_rho(NZ=128, NR=128, Rmax=80.0, Zmax=80.0, Mtot=1.0e11, r0=2.0)
    params = SolverParams(S0=S0, rc_kpc=rc, max_iter=800, tol=5e-6)
    phi, gR, gZ = solve_axisym(R, Z, rho, params)
    gR0 = np.abs(gR[len(Z)//2, :])
    med, sig = plateau_metric(R, gR0, r_inner=20.0, r_outer=60.0)
    return dict(S0=S0, rc_kpc=rc, plateau_v2_med=med, plateau_v2_std=sig)


def main():
    out = {}
    a = run_once(S0=1e-6, rc=15.0)
    b = run_once(S0=3e-6, rc=15.0)
    out['case_S0_1e-6'] = a
    out['case_S0_3e-6'] = b
    out['ratio_plateau_v2'] = None
    if a['plateau_v2_med'] and b['plateau_v2_med']:
        out['ratio_plateau_v2'] = float(b['plateau_v2_med'] / max(a['plateau_v2_med'], 1e-30))
    Path(_pkg/'tests_pde_sanity.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
