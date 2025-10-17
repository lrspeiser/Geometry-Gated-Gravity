#!/usr/bin/env python3
"""
convert_accept_profiles.py

Convert ACCEPT-like X-ray profiles (.dat.txt) to projected gas surface density Σ_gas(R)
CSV files usable by baryon_loader (data/baryon_profiles/{SANITIZED}.csv).

Input columns (header present): Rin [Mpc], Rout [Mpc], nelec [cm^-3]
Assumptions: piecewise-constant ρ in shells; μ_e = 1.17, m_p = 1.6726e-24 g.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.cluster_overrides import normalize_cluster_name

MSUN_G = 1.98847e33
KPC_CM = 3.085677581491367e21
MU_E   = 1.17
M_P_G  = 1.6726219e-24


def load_accept_like(path: Path) -> pd.DataFrame:
    # Read whitespace-delimited with comments starting '#'
    df = pd.read_csv(path, delim_whitespace=True, comment='#', header=None)
    # Infer columns by known order (Name Rin Rout nelec ...)
    # Column 0: Name, 1 Rin [Mpc], 2 Rout [Mpc], 3 nelec [cm^-3]
    df = df[[1,2,3]].copy()
    df.columns = ['Rin_Mpc','Rout_Mpc','nelec_cm3']
    return df


def project_sigma_gas(df: pd.DataFrame, R_max_kpc: float = 2000.0, n_R: int = 200) -> pd.DataFrame:
    # Shell midpoints and densities
    r_in_kpc  = df['Rin_Mpc'].values * 1000.0
    r_out_kpc = df['Rout_Mpc'].values * 1000.0
    ne_cm3    = df['nelec_cm3'].values
    rho_g_cm3 = MU_E * M_P_G * ne_cm3  # g/cm^3
    rho_Msun_kpc3 = rho_g_cm3 / MSUN_G * (KPC_CM**3)  # Msun/kpc^3

    # R grid
    R = np.linspace(1.0, R_max_kpc, n_R)
    Sigma = np.zeros_like(R)

    # For each shell, add LOS path length contribution where R < r_out
    for rin, rout, rho in zip(r_in_kpc, r_out_kpc, rho_Msun_kpc3):
        # valid region where R < r_out
        mask = R < rout
        if not np.any(mask):
            continue
        Rm = R[mask]
        # Path length through shell at projected R: 2*(sqrt(rout^2-R^2) - sqrt(max(rin^2-R^2,0)))
        l_out = np.sqrt(np.maximum(rout**2 - Rm**2, 0.0))
        l_in  = np.sqrt(np.maximum(rin**2  - Rm**2, 0.0))
        path = 2.0 * (l_out - l_in)  # kpc
        Sigma[mask] += rho * path    # Msun/kpc^2

    return pd.DataFrame({'R_kpc': R, 'Sigma_baryon': Sigma})


def main():
    ap = argparse.ArgumentParser(description='Convert ACCEPT-like profiles to Σ(R) CSV')
    ap.add_argument('--input', required=True, help='Path to *_profiles.dat.txt')
    ap.add_argument('--cluster', required=True, help='Cluster name for output file sanitization')
    ap.add_argument('--outdir', required=True, help='Output directory (e.g., data/baryon_profiles)')
    ap.add_argument('--rmax', type=float, default=2000.0)
    ap.add_argument('--nR', type=int, default=200)
    args = ap.parse_args()

    df = load_accept_like(Path(args.input))
    prof = project_sigma_gas(df, R_max_kpc=args.rmax, n_R=args.nR)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    key = normalize_cluster_name(args.cluster)
    out = outdir / f'{key}.csv'
    prof.to_csv(out, index=False)
    print(f'Wrote {out}')

if __name__ == '__main__':
    main()
