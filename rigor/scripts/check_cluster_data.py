# -*- coding: utf-8 -*-
"""
rigor/scripts/check_cluster_data.py

Validate cluster input CSVs and print simple integrals.

Usage:
  py -u rigor/scripts/check_cluster_data.py --clusters ABELL_0426,ABELL_1689,A1795,A478,A2029
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def is_monotonic_increasing(x: np.ndarray) -> bool:
    return np.all(np.diff(x) > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='data/clusters')
    ap.add_argument('--clusters', type=str, required=True)
    args = ap.parse_args()

    base = Path(args.base)
    clusters = [s.strip() for s in str(args.clusters).split(',') if s.strip()]

    for cl in clusters:
        print(f"\n[CHECK] {cl}")
        cdir = base / cl
        try:
            g = load_csv(cdir/'gas_profile.csv')
            if 'r_kpc' not in g.columns:
                raise ValueError('gas_profile.csv missing r_kpc')
            r_g = g['r_kpc'].to_numpy(float)
            if ('rho_gas_Msun_per_kpc3' not in g.columns) and ('n_e_cm3' not in g.columns):
                raise ValueError('gas_profile.csv must include rho_gas_Msun_per_kpc3 or n_e_cm3')
            if not is_monotonic_increasing(r_g):
                print('[WARN] gas_profile radii not strictly increasing')
        except Exception as e:
            print(f'[ERR] gas_profile.csv: {e}')

        try:
            t = load_csv(cdir/'temp_profile.csv')
            if not {'r_kpc','kT_keV'}.issubset(set(t.columns)):
                raise ValueError('temp_profile.csv must include r_kpc,kT_keV[,kT_err_keV]')
            r_t = t['r_kpc'].to_numpy(float)
            if not is_monotonic_increasing(r_t):
                print('[WARN] temp_profile radii not strictly increasing')
        except Exception as e:
            print(f'[ERR] temp_profile.csv: {e}')

        try:
            c = load_csv(cdir/'clump_profile.csv')
            if not {'r_kpc','C'}.issubset(set(c.columns)):
                raise ValueError('clump_profile.csv must include r_kpc,C')
        except Exception as e:
            print(f'[ERR] clump_profile.csv: {e}')

        try:
            s = load_csv(cdir/'stars_profile.csv')
            if not {'r_kpc','rho_star_Msun_per_kpc3'}.issubset(set(s.columns)):
                raise ValueError('stars_profile.csv must include r_kpc,rho_star_Msun_per_kpc3')
            r_s = s['r_kpc'].to_numpy(float)
            if not is_monotonic_increasing(r_s):
                print('[WARN] stars_profile radii not strictly increasing')
            rho_s = s['rho_star_Msun_per_kpc3'].to_numpy(float)
            if np.any(rho_s < 0):
                print('[WARN] negative stellar density values found')
            # quick integrals
            integ_s = 4.0*np.pi * (r_s**2) * rho_s
            M_star_100 = float(np.interp(100.0, r_s, np.concatenate(([0.0], np.cumsum(0.5*(integ_s[1:]+integ_s[:-1]) * np.diff(r_s))))))
            M_star_500 = float(np.interp(500.0, r_s, np.concatenate(([0.0], np.cumsum(0.5*(integ_s[1:]+integ_s[:-1]) * np.diff(r_s))))))
            print(f"[INT] M_star(<100)={M_star_100:.2e} Msun  M_star(<500)={M_star_500:.2e} Msun")
        except Exception as e:
            print(f'[ERR] stars_profile.csv: {e}')

    print('\n[CHECK] Completed')

if __name__ == '__main__':
    main()
