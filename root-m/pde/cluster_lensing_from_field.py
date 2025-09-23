# -*- coding: utf-8 -*-
"""
root-m/pde/cluster_lensing_from_field.py

Build M(<r) (and optionally DeltaSigma overlays) from the saved PDE field
summary in root-m/out/pde_clusters/<CLUSTER>/field_summary.csv.

Inputs:
- --cluster: name (e.g., ABELL_0426)
- --field_dir: default root-m/out/pde_clusters/<cluster>
- --out_dir: default same as field_dir
- Optional observed lensing:
  - data/clusters/<cluster>/lensing_mass.csv with r_kpc, M_enclosed_Msun, M_err_Msun

Outputs:
- Saves M(<r) overlay plot if observed lensing mass file exists.
- Writes JSON summary with simple amplitude ratios at a few radii if lensing mass provided.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', required=True)
    ap.add_argument('--field_dir', default=None)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    field_dir = Path(args.field_dir) if args.field_dir else (Path('root-m/out/pde_clusters')/args.cluster)
    out_dir = Path(args.out_dir) if args.out_dir else Path('figs/lensing')
    out_dir.mkdir(parents=True, exist_ok=True)

    fsum = pd.read_csv(field_dir/'field_summary.csv')
    R = fsum['R_kpc'].to_numpy(float)
    g_tot = fsum['g_tot_R'].to_numpy(float)
    M_pred = (R**2) * g_tot / G

    # Try load observed lensing mass
    cdir = Path('data/clusters')/args.cluster
    lm_path = cdir/'lensing_mass.csv'
    if lm_path.exists():
        lm = pd.read_csv(lm_path)
        r_obs = lm['r_kpc'].to_numpy(float)
        M_obs = lm['M_enclosed_Msun'].to_numpy(float)
        M_err = lm['M_err_Msun'].to_numpy(float) if 'M_err_Msun' in lm.columns else None
        # Interpolate model to observed radii for simple amplitude ratios
        M_model_on_obs = np.interp(r_obs, R, M_pred)
        ratios = M_model_on_obs / np.maximum(M_obs, 1e-12)
        with open(out_dir/'lensing_mass_compare.json','w') as f:
            json.dump({'r_kpc': r_obs.tolist(), 'M_model': M_model_on_obs.tolist(), 'M_obs': M_obs.tolist(), 'ratio': ratios.tolist()}, f, indent=2)
        # Plot overlay
        plt.figure(figsize=(6,4))
        plt.loglog(R, M_pred, label='PDE (from g_tot)')
        if M_err is not None:
            plt.errorbar(r_obs, M_obs, yerr=M_err, fmt='o', ms=3, label='Lensing')
        else:
            plt.loglog(r_obs, M_obs, 'o', ms=3, label='Lensing')
        plt.xlabel('r [kpc]'); plt.ylabel('M(<r) [Msun]'); plt.title(args.cluster)
        plt.grid(True, which='both', alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir/(f"{args.cluster}_lensing_overlay.png"), dpi=140); plt.close()
    else:
        # No observed lensing mass provided; save the model mass curve and plot
        pd.DataFrame({'R_kpc': R, 'M_model_Msun': M_pred}).to_csv(out_dir/'M_model_only.csv', index=False)
        plt.figure(figsize=(6,4))
        plt.loglog(R, M_pred, 'r-', lw=1.6, label='PDE (from g_tot)')
        plt.xlabel('r [kpc]'); plt.ylabel('M(<r) [Msun]'); plt.title(args.cluster)
        plt.grid(True, which='both', alpha=0.3); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir/(f"{args.cluster}_M_model_only.png"), dpi=140); plt.close()


if __name__ == '__main__':
    main()
