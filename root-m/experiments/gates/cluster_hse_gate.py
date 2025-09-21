# -*- coding: utf-8 -*-
"""
root-m/experiments/gates/cluster_hse_gate.py

Cluster HSE runner for Root-M tail variants with thermodynamic gating.
Writes metrics and plots under root-m/out/experiments/gates/<model>/<CLUSTER>/.

This keeps original code intact.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

import importlib.util as _ilu, sys as _sys
from pathlib import Path as _P
_pkg_dir = _P(__file__).resolve().parent
_spec = _ilu.spec_from_file_location('gating_common', str(_pkg_dir/'gating_common.py'))
_gm = _ilu.module_from_spec(_spec); _sys.modules['gating_common'] = _gm
_spec.loader.exec_module(_gm)

v_tail2_rootm_rhoaware = _gm.v_tail2_rootm_rhoaware
v_tail2_rootm_gradaware = _gm.v_tail2_rootm_gradaware
v_tail2_rootm_pressaware = _gm.v_tail2_rootm_pressaware

# constants
G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)
MU = 0.61
MU_E = 1.17
M_P = 1.67262192369e-27
J_PER_KEV = 1.602176634e-16
KM2_PER_S2_TO_J_PER_KG = 1.0e6
KMS2_TO_KEV = (MU * M_P * KM2_PER_S2_TO_J_PER_KG) / J_PER_KEV


def ne_to_rho_gas_Msun_kpc3(n_e_cm3, mu_e=MU_E):
    kpc_in_cm = 3.0856775814913673e21
    Msun_in_g = 1.988409870698051e33
    g_per_cm3 = mu_e * 1.67262192369e-24 * np.asarray(n_e_cm3)
    return g_per_cm3 * (kpc_in_cm**3) / Msun_in_g


def enclosed_mass(r_kpc, rho_Msun_kpc3):
    r = np.asarray(r_kpc); rho = np.asarray(rho_Msun_kpc3)
    integrand = 4.0 * np.pi * (r**2) * rho
    M = np.zeros_like(r)
    if r.size >= 2:
        dr = np.diff(r)
        area = 0.5 * (integrand[1:] + integrand[:-1]) * dr
        M[1:] = np.cumsum(area)
    return M


def reversed_cumtrapz(x, y):
    x = np.asarray(x); y = np.asarray(y)
    S = np.zeros_like(y)
    if x.size >= 2:
        dx = np.diff(x)
        area = 0.5 * (y[1:] + y[:-1]) * dx
        S[:-1] = area[::-1].cumsum()[::-1]
    return S


def hydrostatic_kT_keV(r_kpc, n_e_cm3, g_tot_kms2_per_kpc):
    n_g_cm3 = (MU_E / MU) * np.asarray(n_e_cm3)
    I = reversed_cumtrapz(r_kpc, n_g_cm3 * g_tot_kms2_per_kpc)
    potential_kms2 = I / np.maximum(n_g_cm3, 1e-12)
    kT_keV = KMS2_TO_KEV * potential_kms2
    return kT_keV, potential_kms2


def run_cluster(cluster: str, base: Path, outbase: Path, model: str,
                A_kms: float, rc_kpc: float,
                rho0: float, q: float, qg: float, m: float, eta: float, smooth: float):
    cdir = Path(base)/cluster
    g = pd.read_csv(cdir/'gas_profile.csv')
    t = pd.read_csv(cdir/'temp_profile.csv')

    # unified radial grid
    r = np.unique(np.concatenate([g['r_kpc'].values, t['r_kpc'].values]))
    r = r[(r > 0) & np.isfinite(r)]; r = np.sort(r)

    ne = np.interp(r, g['r_kpc'], g['n_e_cm3'])
    rho_gas = ne_to_rho_gas_Msun_kpc3(ne)

    s_path = cdir/'stars_profile.csv'
    if s_path.exists():
        s = pd.read_csv(s_path)
        rho_star = np.interp(r, s['r_kpc'], s['rho_star_Msun_per_kpc3'])
    else:
        rho_star = np.zeros_like(r)

    rho_b = rho_gas + rho_star
    # ensure rho_b and Mb align to same r sampling for tails
    Mb = enclosed_mass(r, rho_b)
    # resample rho_b on r where Mb defined (already same length), but guard any edge effects
    rho_b = np.interp(r, r, rho_b)

    # Choose gating model
    if model == 'rhoaware':
        v2_tail = v_tail2_rootm_rhoaware(r, Mb, rho_b, A_kms=A_kms, Mref_Msun=6e10,
                                         rc_kpc=rc_kpc, rho0_Msun_kpc3=rho0, q=q, s_kpc=smooth)
    elif model == 'gradaware':
        v2_tail = v_tail2_rootm_gradaware(r, Mb, rho_b, A_kms=A_kms, Mref_Msun=6e10,
                                          rc_kpc=rc_kpc, qg=qg, m=m, s_kpc=smooth)
    elif model == 'pressaware':
        kT = np.interp(r, t['r_kpc'], t['kT_keV'])
        v2_tail = v_tail2_rootm_pressaware(r, Mb, ne, kT, A_kms=A_kms, Mref_Msun=6e10,
                                           rc_kpc=rc_kpc, eta=eta, s_kpc=smooth)
    else:
        raise SystemExit(f"Unknown model {model}")

    g_b = G * Mb / np.maximum(r**2, 1e-12)
    g_tail = v2_tail / np.maximum(r, 1e-12)
    g_tot = g_b + g_tail

    kT_pred, _ = hydrostatic_kT_keV(r, ne, g_tot)
    kT_obs = np.interp(r, t['r_kpc'], t['kT_keV'])
    mask = np.isfinite(kT_obs) & (kT_obs > 0)
    frac = np.abs(kT_pred[mask] - kT_obs[mask]) / np.maximum(kT_obs[mask], 1e-12)
    med_frac = float(np.median(frac)) if mask.any() else float('nan')

    od = Path(outbase)/model/cluster/cluster
    od = Path(outbase)/model/cluster
    od.mkdir(parents=True, exist_ok=True)
    with open(od/'metrics.json','w') as f:
        json.dump({'cluster': cluster, 'model': model, 'A_kms': A_kms, 'rc_kpc': rc_kpc,
                   'rho0': rho0, 'q': q, 'qg': qg, 'm': m, 'eta': eta, 'smooth_kpc': smooth,
                   'temp_median_frac_err': med_frac}, f, indent=2)

    # plot
    plt.figure(figsize=(11,5))
    plt.subplot(1,2,1)
    plt.loglog(r, Mb, label='Baryons (gas+stars)')
    M_pred = r*r * g_tot / G
    plt.loglog(r, M_pred, ls='--', label='Total (gated)')
    plt.xlabel('r [kpc]'); plt.ylabel('M(<r) [Msun]'); plt.legend(); plt.grid(True, which='both', alpha=0.3)

    plt.subplot(1,2,2)
    plt.semilogx(r, kT_pred, ls='--', label='kT (pred)')
    plt.errorbar(t['r_kpc'], t['kT_keV'], yerr=t.get('kT_err_keV'), fmt='o', ms=3, label='kT (X-ray)')
    plt.xlabel('r [kpc]'); plt.ylabel('kT [keV]'); plt.legend(); plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig(od/'cluster_gate_results.png', dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', required=True)
    ap.add_argument('--base', default='data/clusters')
    ap.add_argument('--outbase', default='root-m/out/experiments/gates')
    ap.add_argument('--model', choices=['rhoaware','gradaware','pressaware'], default='rhoaware')
    ap.add_argument('--A_kms', type=float, default=160.0)
    ap.add_argument('--rc_kpc', type=float, default=10.0)
    ap.add_argument('--rho0', type=float, default=1e7)
    ap.add_argument('--q', type=float, default=1.0)
    ap.add_argument('--qg', type=float, default=0.8)
    ap.add_argument('--m', type=float, default=1.0)
    ap.add_argument('--eta', type=float, default=0.5)
    ap.add_argument('--smooth', type=float, default=100.0)
    args = ap.parse_args()

    run_cluster(cluster=args.cluster, base=Path(args.base), outbase=Path(args.outbase), model=args.model,
                A_kms=args.A_kms, rc_kpc=args.rc_kpc, rho0=args.rho0, q=args.q, qg=args.qg, m=args.m, eta=args.eta, smooth=args.smooth)

if __name__ == '__main__':
    main()
