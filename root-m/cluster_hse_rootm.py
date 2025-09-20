# -*- coding: utf-8 -*-
import json, math, numpy as np, pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from root_m.common import v_tail2_rootm_soft as v_tail2_soft

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
    integrand = 4.0 * math.pi * (r**2) * rho
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
    potential_kms2 = I / np.maximum(n_g_cm3, 1e-99)
    kT_keV = KMS2_TO_KEV * potential_kms2
    return kT_keV, potential_kms2


def g_total_rootm_soft(r_kpc, M_b_enclosed_Msun, A_kms=140.0, Mref_Msun=6.0e10, rc_kpc=15.0):
    r = np.asarray(r_kpc)
    Mb = np.asarray(M_b_enclosed_Msun)
    g_b = G * Mb / np.maximum(r**2, 1e-12)
    v_tail2 = v_tail2_soft(r, Mb, A_kms=A_kms, Mref=Mref_Msun, rc_kpc=rc_kpc)
    g_tail = v_tail2 / np.maximum(r, 1e-12)
    return g_b + g_tail


def run_cluster(cluster, base='data/clusters', outdir='root-m/out/clusters', A_kms=140.0, Mref_Msun=6.0e10, rc_kpc=15.0):
    cdir = Path(base)/cluster
    g = pd.read_csv(cdir/'gas_profile.csv')
    t = pd.read_csv(cdir/'temp_profile.csv')

    # grid
    r = np.unique(np.concatenate([g['r_kpc'].values, t['r_kpc'].values]))
    r = r[(r>0) & np.isfinite(r)]
    r = np.sort(r)
    if r.size < 2:
        raise ValueError('Not enough radial points')

    ne = np.interp(r, g['r_kpc'], g['n_e_cm3'])
    rho_gas = ne_to_rho_gas_Msun_kpc3(ne)

    s_path = cdir/'stars_profile.csv'
    if s_path.exists():
        s = pd.read_csv(s_path)
        rho_star = np.interp(r, s['r_kpc'], s['rho_star_Msun_per_kpc3'])
    else:
        rho_star = np.zeros_like(r)

    rho_b = rho_gas + rho_star
    Mb = enclosed_mass(r, rho_b)
    g_t = g_total_rootm_soft(r, Mb, A_kms=A_kms, Mref_Msun=Mref_Msun, rc_kpc=rc_kpc)

    kT_pred, phi_eff = hydrostatic_kT_keV(r, ne, g_t)
    M_pred = (r**2) * g_t / G

    kT_obs = np.interp(r, t['r_kpc'], t['kT_keV'])
    mask = np.isfinite(kT_obs) & (kT_obs>0)
    frac_err = np.abs(kT_pred[mask]-kT_obs[mask]) / np.maximum(kT_obs[mask], 1e-12)
    med_frac = float(np.median(frac_err)) if mask.any() else float('nan')

    od = Path(outdir)/cluster
    od.mkdir(parents=True, exist_ok=True)
    with open(od/'cluster_rootm_metrics.json','w') as f:
        json.dump({
            'cluster': cluster,
            'A_kms': A_kms,
            'Mref_Msun': Mref_Msun,
            'r_kpc_min': float(r.min()),
            'r_kpc_max': float(r.max()),
'n_r': int(len(r)),
            'temp_median_frac_err': med_frac,
            'rc_kpc': rc_kpc
        }, f, indent=2)

    # plot
    plt.figure(figsize=(11,5))
    plt.subplot(1,2,1)
    plt.loglog(r, Mb, label='Baryons (gas+stars)')
    plt.loglog(r, M_pred, ls='--', label='Total (Root-M)')
    lm_path = cdir/'lensing_mass.csv'
    if lm_path.exists():
        lm = pd.read_csv(lm_path)
        plt.errorbar(lm['r_kpc'], lm['M_enclosed_Msun'], yerr=lm.get('M_err_Msun'), fmt='o', ms=3, label='Lensing')
    plt.xlabel('r [kpc]'); plt.ylabel('M(<r) [Msun]'); plt.title(f'{cluster}: enclosed mass')
    plt.legend(); plt.grid(True, which='both', alpha=0.3)

    plt.subplot(1,2,2)
    plt.semilogx(r, kT_pred, ls='--', label='kT (pred, HSE+Root-M)')
    plt.errorbar(t['r_kpc'], t['kT_keV'], yerr=t.get('kT_err_keV'), fmt='o', ms=3, label='kT (X-ray)')
    plt.xlabel('r [kpc]'); plt.ylabel('kT [keV]'); plt.title(f'{cluster}: temperature (HSE)')
    plt.legend(); plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(od/'cluster_rootm_results.png', dpi=140)
    plt.close()


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', required=True)
    ap.add_argument('--base', default='data/clusters')
    ap.add_argument('--outdir', default='root-m/out/clusters')
    ap.add_argument('--A_kms', type=float, default=140.0)
    ap.add_argument('--Mref', type=float, default=6.0e10)
    ap.add_argument('--rc_kpc', type=float, default=15.0)
    args = ap.parse_args()
    run_cluster(cluster=args.cluster, base=args.base, outdir=args.outdir, A_kms=args.A_kms, Mref_Msun=args.Mref, rc_kpc=args.rc_kpc)

if __name__ == '__main__':
    cli()
