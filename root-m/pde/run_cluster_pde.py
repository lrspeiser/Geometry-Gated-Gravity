# -*- coding: utf-8 -*-
"""
root-m/pde/run_cluster_pde.py

End-to-end PDE run for clusters:
- Build spherical rho_b(R,z) from gas_profile.csv (+ stars)
- Solve ∇·(|∇φ| ∇φ) = - S0 ρ_b on (R,z) grid
- Combine g_phi with baryon Newtonian to get g_tot along z=0
- Predict kT(r) via integral HSE
- Write PNG + JSON metrics under root-m/out/pde_clusters/<CLUSTER>/
"""
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt

# Local imports (folder is not a Python package)
import importlib.util as _ilu
from pathlib import Path as _P
_pkg_dir = _P(__file__).resolve().parent

def _load_local(modname: str, filename: str):
    import sys as _sys
    spec = _ilu.spec_from_file_location(modname, str(_pkg_dir/filename))
    mod = _ilu.module_from_spec(spec)
    _sys.modules[modname] = mod  # register so dataclasses sees module
    spec.loader.exec_module(mod)
    return mod
_solve = _load_local('solve_phi', 'solve_phi.py')
_maps  = _load_local('baryon_maps', 'baryon_maps.py')
_phse  = _load_local('predict_hse', 'predict_hse.py')

SolverParams = _solve.SolverParams
solve_axisym = _solve.solve_axisym
cluster_map_from_csv = _maps.cluster_map_from_csv
kT_from_ne_and_gtot = _phse.kT_from_ne_and_gtot

G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', required=True)
    ap.add_argument('--base', default='data/clusters')
    ap.add_argument('--outdir', default='root-m/out/pde_clusters')
    ap.add_argument('--Rmax', type=float, default=1500.0)
    ap.add_argument('--Zmax', type=float, default=1500.0)
    ap.add_argument('--NR', type=int, default=128)
    ap.add_argument('--NZ', type=int, default=128)
    ap.add_argument('--S0', type=float, default=1.0e-7)
    ap.add_argument('--rc_kpc', type=float, default=15.0)
    ap.add_argument('--g0_kms2_per_kpc', type=float, default=1000.0)
    ap.add_argument('--m_exp', type=float, default=1.0)
    # New physics knobs
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--Mref', type=float, default=6.0e10)
    ap.add_argument('--kappa', type=float, default=0.0)
    ap.add_argument('--q_slope', type=float, default=1.0)
    ap.add_argument('--chi', type=float, default=0.0)
    ap.add_argument('--h_aniso_kpc', type=float, default=0.3)
    args = ap.parse_args()

    cdir = Path(args.base)/args.cluster
    Z, R, rho = cluster_map_from_csv(cdir, R_max=args.Rmax, Z_max=args.Zmax, NR=args.NR, NZ=args.NZ)
    # Guard taper near boundaries to reduce reflection/suppression
    def _taper_1d(x, frac=0.1):
        xmax = float(np.max(np.abs(x)))
        xabs = np.abs(x)
        x0 = (1.0 - frac) * xmax
        w = np.ones_like(xabs)
        mask = xabs > x0
        if np.any(mask):
            xi = (xabs[mask] - x0) / (xmax - x0 + 1e-12)
            w[mask] = 0.5 * (1.0 + np.cos(np.pi * xi))  # cosine to zero at edge
        return w
    import numpy as np
    wZ = _taper_1d(Z, frac=0.1).reshape(-1,1)
    wR = _taper_1d(R, frac=0.1).reshape(1,-1)
    rho = rho * (wZ * wR)

    # Solve PDE
    params = SolverParams(S0=args.S0, rc_kpc=args.rc_kpc, g0_kms2_per_kpc=args.g0_kms2_per_kpc, m_exp=args.m_exp,
                          eta=float(args.eta), Mref_Msun=float(args.Mref),
                          kappa=float(args.kappa), q_slope=float(args.q_slope),
                          chi=float(args.chi), h_aniso_kpc=float(args.h_aniso_kpc))
    phi, gR, gZ = solve_axisym(R, Z, rho, params)

    # Equatorial extraction for g_phi and HSE
    z0_idx = len(Z)//2
    # Additional centripetal acceleration magnitude along midplane
    g_phi_R = np.abs(gR[z0_idx, :])

    # Build g_N along z=0 using M(<R)=∫ 4πr^2ρ dr (spherical approx)
    # integrate rho over spherical shells from cluster CSV (consistent with generator)
    g = pd.read_csv(cdir/'gas_profile.csv')
    r_obs = np.asarray(g['r_kpc'], float)
    # simple cumulative mass from the input spherical rho we used
    ne_to_rho = _maps.ne_to_rho_gas_Msun_kpc3
    if 'rho_gas_Msun_per_kpc3' in g.columns:
        rho_r = np.asarray(g['rho_gas_Msun_per_kpc3'], float)
    else:
        rho_r = ne_to_rho(np.asarray(g['n_e_cm3'], float))
    # Ensure ascending radius for stable integration
    order = np.argsort(r_obs)
    r_obs = r_obs[order]
    rho_r = rho_r[order]
    # cumulative mass integral
    integrand = 4.0*np.pi * (r_obs**2) * rho_r
    M = np.concatenate(([0.0], np.cumsum(0.5*(integrand[1:]+integrand[:-1]) * np.diff(r_obs))))
    M = M[:len(r_obs)]

    # interpolate onto PDE R grid
    M_R = np.interp(R, r_obs, M, left=M[0], right=M[-1])
    g_N_R = G * M_R / np.maximum(R**2, 1e-9)
    g_tot_R = g_N_R + g_phi_R
    # Diagnostics (restrict to observed temperature radii)
    t = pd.read_csv(cdir/'temp_profile.csv')
    rT = np.asarray(t['r_kpc'], float)
    g_phi_on_obs = np.interp(rT, R, g_phi_R)
    g_N_on_obs = np.interp(rT, R, g_N_R)
    med_ratio = float(np.median(g_phi_on_obs / np.maximum(g_N_on_obs, 1e-12)))
    print(f"[PDE-Cluster] {args.cluster}: median g_phi/g_N on kT radii = {med_ratio:.3g}")

    # HSE prediction (interpolate n_e onto R grid)
    if 'n_e_cm3' in g.columns:
        ne_obs = np.asarray(g['n_e_cm3'], float)
    else:
        # if only rho_gas is present, approximate n_e via rho_gas/(mu_e m_p)
        MU_E = 1.17; M_P_G = 1.67262192369e-24; KPC_CM = 3.0856775814913673e21; MSUN_G = 1.988409870698051e33
        ne_obs = (np.asarray(g['rho_gas_Msun_per_kpc3'], float) * MSUN_G / (KPC_CM**3)) / (MU_E * M_P_G)
    # Match ordering to r_obs ascending
    ne_obs = ne_obs[order]
    ne_R = np.interp(R, r_obs, ne_obs, left=ne_obs[0], right=ne_obs[-1])

    kT_pred = kT_from_ne_and_gtot(R, ne_R, g_tot_R)

    # Observational T
    t = pd.read_csv(cdir/'temp_profile.csv')
    rT = np.asarray(t['r_kpc'], float)
    kT = np.asarray(t['kT_keV'], float)
    kT_pred_on_obs = np.interp(rT, R, kT_pred)
    frac = np.median(np.abs(kT_pred_on_obs - kT)/np.maximum(kT, 1e-12))

    # Output
    od = Path(args.outdir)/args.cluster
    od.mkdir(parents=True, exist_ok=True)
    with open(od/'metrics.json','w') as f:
        json.dump({'cluster': args.cluster, 'S0': args.S0, 'rc_kpc': args.rc_kpc,
                   'temp_median_frac_err': float(frac)}, f, indent=2)

    plt.figure(figsize=(11,5))
    # Mass/accel panel
    plt.subplot(1,2,1)
    plt.plot(R, g_phi_R, label='g_phi (PDE)')
    plt.plot(R, g_N_R, label='g_N (baryons)')
    plt.plot(R, g_tot_R, label='g_tot')
    plt.xscale('log'); plt.yscale('symlog')
    plt.xlabel('R [kpc]'); plt.ylabel('g [(km/s)^2/kpc]'); plt.legend(); plt.grid(True, which='both', alpha=0.3)

    # Temperature panel
    plt.subplot(1,2,2)
    plt.semilogx(R, kT_pred, '--', label='kT (PDE+HSE)')
    plt.errorbar(rT, kT, yerr=t.get('kT_err_keV'), fmt='o', ms=3, label='kT (X-ray)')
    plt.xlabel('r [kpc]'); plt.ylabel('kT [keV]'); plt.legend(); plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(od/'cluster_pde_results.png', dpi=140); plt.close()

if __name__ == '__main__':
    main()
