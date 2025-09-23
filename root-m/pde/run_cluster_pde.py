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
    # Geometry-aware global knobs (shape and mild amplitude)
    ap.add_argument('--rc_gamma', type=float, default=0.0, help='Exponent for rc_eff = rc * (r_half/rc_ref_kpc)^rc_gamma')
    ap.add_argument('--rc_ref_kpc', type=float, default=30.0, help='Reference size for rc_gamma scaling')
    ap.add_argument('--sigma_beta', type=float, default=0.0, help='Exponent for S0_eff = S0 * (sigma0/sigma_bar)^sigma_beta')
    ap.add_argument('--sigma0_Msun_pc2', type=float, default=150.0, help='Reference surface density for sigma_beta (Msun/pc^2)')
    # New physics knobs
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--Mref', type=float, default=6.0e10)
    ap.add_argument('--kappa', type=float, default=0.0)
    ap.add_argument('--q_slope', type=float, default=1.0)
    ap.add_argument('--chi', type=float, default=0.0)
    ap.add_argument('--h_aniso_kpc', type=float, default=0.3)
    ap.add_argument('--bc_robin_lambda', type=float, default=0.0, help='Robin BC strength (1/kpc); 0 disables')
    # NEW A1/A2 global controls
    ap.add_argument('--use_saturating_mobility', action='store_true')
    ap.add_argument('--gsat_kms2_per_kpc', type=float, default=2000.0)
    ap.add_argument('--n_sat', type=float, default=2.0)
    ap.add_argument('--use_ambient_boost', action='store_true')
    ap.add_argument('--beta_env', type=float, default=0.0)
    ap.add_argument('--rho_ref_Msun_per_kpc3', type=float, default=1.0e6)
    ap.add_argument('--env_L_kpc', type=float, default=150.0)
    # Optional non-thermal pressure in HSE
    ap.add_argument('--fnt0', type=float, default=0.0, help='Non-thermal pressure fraction at r=r500 (dimensionless); 0 disables')
    ap.add_argument('--fnt_n', type=float, default=0.8, help='Radial power for f_nt(r) = fnt0 * (r/r500)^n')
    ap.add_argument('--r500_kpc', type=float, default=1000.0, help='Characteristic r500 in kpc for f_nt radial scaling')
    ap.add_argument('--fnt_max', type=float, default=0.3, help='Max cap for f_nt (dimensionless)')
    # Optional compactness-aware source modulation (global, density-based)
    ap.add_argument('--use_compactness_source', action='store_true', help='Enable compactness-based screening of the PDE source')
    ap.add_argument('--rho_comp_star_Msun_per_kpc3', type=float, default=1.0e6, help='Reference density for compactness screening')
    ap.add_argument('--alpha_comp', type=float, default=0.2, help='Strength of compactness screening')
    ap.add_argument('--m_comp', type=float, default=2.0, help='Slope of compactness screening')
    # B-input levers
    ap.add_argument('--clump', type=float, default=1.0, help='Uniform gas clumping C; applies n_e -> sqrt(C) * n_e if profile not given')
    ap.add_argument('--clump_profile_csv', type=str, default=None, help='CSV with r_kpc,C for radial clumping; overrides uniform --clump')
    ap.add_argument('--stars_csv', type=str, default=None, help='Optional stars_profile.csv path (r_kpc,rho_star_Msun_per_kpc3); default: data/clusters/<CL>/stars_profile.csv if present')
    args = ap.parse_args()

    cdir = Path(args.base)/args.cluster
    Z, R, rho = cluster_map_from_csv(cdir, R_max=args.Rmax, Z_max=args.Zmax, NR=args.NR, NZ=args.NZ,
                                     clump=float(args.clump),
                                     clump_profile_csv=Path(args.clump_profile_csv) if args.clump_profile_csv else None,
                                     stars_csv=Path(args.stars_csv) if args.stars_csv else None)
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

    # Geometry scalars from TOTAL baryons (gas+stars) after clumping — parity with SPARC
    # Build rho_gas(r) with clumping applied
    g_in = pd.read_csv(cdir/'gas_profile.csv')
    r_in = np.asarray(g_in['r_kpc'], float)
    ne_to_rho = _maps.ne_to_rho_gas_Msun_kpc3
    if 'rho_gas_Msun_per_kpc3' in g_in.columns:
        rho_gas = np.asarray(g_in['rho_gas_Msun_per_kpc3'], float)
    else:
        rho_gas = ne_to_rho(np.asarray(g_in['n_e_cm3'], float))
    # Apply clumping: use radial C(r) if provided; else uniform sqrt(C)
    C_prof = None
    if args.clump_profile_csv:
        try:
            cp = pd.read_csv(Path(args.clump_profile_csv))
            r_cp = np.asarray(cp['r_kpc'], float)
            C_cp = np.asarray(cp['C'], float)
            C_prof = np.interp(r_in, r_cp, C_cp, left=C_cp[0], right=C_cp[-1])
        except Exception:
            C_prof = None
    C_uni = float(np.sqrt(max(float(args.clump), 1.0)))
    if C_prof is not None:
        rho_gas_eff = rho_gas * np.sqrt(np.maximum(C_prof, 1.0))
    else:
        rho_gas_eff = rho_gas * C_uni

    # Optional stars profile (BCG/ICL): interpolate onto r_in if present
    rho_star_eff = np.zeros_like(r_in, dtype=float)
    s_path = Path(args.stars_csv) if args.stars_csv else (cdir/"stars_profile.csv")
    if s_path is not None and Path(s_path).exists():
        try:
            s = pd.read_csv(s_path)
            rs = np.asarray(s['r_kpc'], float)
            rho_s = np.asarray(s['rho_star_Msun_per_kpc3'], float)
            rho_star_eff = np.interp(r_in, rs, rho_s, left=rho_s[0], right=rho_s[-1])
        except Exception:
            rho_star_eff = np.zeros_like(r_in, dtype=float)

    # Total baryon density profile
    rho_tot = rho_gas_eff + rho_star_eff

    # Ensure ascending radius for stable integrations
    order_geom = np.argsort(r_in)
    r_in = r_in[order_geom]
    rho_tot = rho_tot[order_geom]

    # Spherical cumulative mass and geometry scalars
    dr = np.gradient(r_in)
    shell_vol = 4.0 * np.pi * (r_in**2) * dr
    M_shell = rho_tot * shell_vol
    M_cum = np.cumsum(M_shell)
    M_tot = float(M_cum[-1]) if M_cum.size else 0.0
    r_half = float(np.interp(0.5*M_tot, M_cum, r_in)) if M_tot > 0 else float(R[len(R)//4])
    sigma_bar_kpc2 = (0.5*M_tot) / (np.pi * max(r_half, 1e-9)**2) if r_half > 0 else 0.0
    sigma_bar_pc2 = sigma_bar_kpc2 / 1.0e6

    # Effective rc and S0
    rc_eff = float(args.rc_kpc) * (max(r_half, 1e-12) / max(args.rc_ref_kpc, 1e-12))**float(args.rc_gamma)
    S0_eff = float(args.S0) * (max(args.sigma0_Msun_pc2, 1e-12) / max(sigma_bar_pc2, 1e-12))**float(args.sigma_beta)

    # Solve PDE
    params = SolverParams(S0=S0_eff, rc_kpc=rc_eff, g0_kms2_per_kpc=args.g0_kms2_per_kpc, m_exp=args.m_exp,
                          eta=float(args.eta), Mref_Msun=float(args.Mref),
                          kappa=float(args.kappa), q_slope=float(args.q_slope),
                          chi=float(args.chi), h_aniso_kpc=float(args.h_aniso_kpc),
                          use_saturating_mobility=bool(args.use_saturating_mobility),
                          gsat_kms2_per_kpc=float(args.gsat_kms2_per_kpc),
                          n_sat=float(args.n_sat),
                          use_ambient_boost=bool(args.use_ambient_boost),
                          beta_env=float(args.beta_env),
                          rho_ref_Msun_per_kpc3=float(args.rho_ref_Msun_per_kpc3),
                          env_L_kpc=float(args.env_L_kpc),
                          use_compactness_source=bool(args.use_compactness_source),
                          rho_comp_star_Msun_per_kpc3=float(args.rho_comp_star_Msun_per_kpc3),
                          alpha_comp=float(args.alpha_comp),
                          m_comp=float(args.m_comp),
                          bc_robin_lambda=float(args.bc_robin_lambda))
    phi, gR, gZ = solve_axisym(R, Z, rho, params)

    # Spherical radial projection of PDE field onto shells r = const
    NZ, NR = gR.shape
    dR = float(np.mean(np.diff(R))) if R.size > 1 else 1.0
    dZ = float(np.mean(np.diff(Z))) if Z.size > 1 else 1.0
    R2D = np.broadcast_to(R.reshape(1,-1), (NZ, NR))
    Z2D = np.broadcast_to(Z.reshape(-1,1), (NZ, NR))
    r_cell = np.sqrt(R2D*R2D + Z2D*Z2D)
    # radial unit components
    r_hat_R = np.divide(R2D, np.maximum(r_cell, 1e-9))
    r_hat_Z = np.divide(Z2D, np.maximum(r_cell, 1e-9))
    g_r_field = gR * r_hat_R + gZ * r_hat_Z
    g_phi_R = np.zeros_like(R)
    dr = 0.5*max(dR, dZ)
    for i, rsh in enumerate(R):
        if rsh <= 0:
            # Small ball around origin
            mask = (r_cell <= max(dr, 1e-6))
        else:
            mask = (np.abs(r_cell - rsh) <= dr)
        if np.any(mask):
            # Use mean of absolute radial component on the shell
            g_phi_R[i] = float(np.mean(np.abs(g_r_field[mask])))
        else:
            # Fallback to midplane value at same cylindrical R
            z0_idx = len(Z)//2
            g_phi_R[i] = float(np.abs(gR[z0_idx, i]))

    # Build g_N along r using M(<r)=∫ 4πr^2ρ dr (spherical approx)
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

    # Non-thermal pressure fraction profile (optional)
    f_nt = None
    if float(args.fnt0) > 0.0:
        r_abs = np.asarray(R, float)
        # simple power-law in radius relative to r500, capped at fnt_max
        f_nt = np.clip(float(args.fnt0) * np.power(np.maximum(r_abs, 0.0) / max(float(args.r500_kpc), 1e-9), float(args.fnt_n)), 0.0, float(args.fnt_max))
    kT_pred = kT_from_ne_and_gtot(R, ne_R, g_tot_R, f_nt)

    # Save field summary for downstream lensing overlays
    import pandas as _pd
    _pd.DataFrame({'R_kpc': R, 'g_phi_R': g_phi_R, 'g_N_R': g_N_R, 'g_tot_R': g_tot_R}).to_csv((Path(args.outdir)/args.cluster/'field_summary.csv'), index=False)

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
        json.dump({'cluster': args.cluster,
                   'S0_input': args.S0,
                   'rc_input_kpc': args.rc_kpc,
                   'S0_eff': S0_eff,
                   'rc_eff_kpc': rc_eff,
                   'r_half_kpc': r_half,
                   'sigma_bar_Msun_pc2': sigma_bar_pc2,
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
    # Annotate global tuple and median fractional error
    txt = (f"G³ global: S0={args.S0}, rc={args.rc_kpc} kpc, b3={args.rc_gamma}, b2={args.sigma_beta}, g0={args.g0_kms2_per_kpc}\n"
           f"median |ΔT|/T = {frac:.3f}")
    plt.gcf().text(0.02, 0.02, txt, fontsize=8, ha='left', va='bottom')
    plt.tight_layout(); plt.savefig(od/'cluster_pde_results.png', dpi=140); plt.close()

if __name__ == '__main__':
    main()
