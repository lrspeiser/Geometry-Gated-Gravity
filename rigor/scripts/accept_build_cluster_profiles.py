# -*- coding: utf-8 -*-
"""
rigor/scripts/accept_build_cluster_profiles.py

One-shot helper to standardize cluster inputs for the PDE pipeline.

Emits the four tables our runners accept under data/clusters/<CL>:
- gas_profile.csv         (r_kpc, n_e_cm3)
- temp_profile.csv        (r_kpc, kT_keV[, kT_err_keV])
- clump_profile.csv       (r_kpc, C)
- stars_profile.csv       (r_kpc, rho_star_Msun_per_kpc3)

Notes:
- If gas_profile.csv already exists, we use its r_kpc grid to build clump/stars.
- If gas/temp are missing and an ACCEPT profiles file is provided, we parse it.
- ACCEPT format assumed: columns [Name, Rin_Mpc, Rout_Mpc, ne_cm3, neerr, ..., Tx_keV, Txerr_keV, ...]
  This matches Cavagnolo+09/ACCEPT per-cluster tables.

CLI examples:
  py -u rigor\scripts\accept_build_cluster_profiles.py --cluster ABELL_0426 \
     --clump_inner 1.05 --clump_outer 3.0 --clump_r0_kpc 500 --clump_alpha 2 \
     --stars_Mtot_Msun 8e11 --stars_a_kpc 20

  py -u rigor\scripts\accept_build_cluster_profiles.py --cluster ABELL_1689 \
     --clump_inner 1.0 --clump_outer 1.5 --clump_r0_kpc 800 --clump_alpha 2 \
     --stars_Mtot_Msun 1.5e12 --stars_a_kpc 30
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def parse_accept_profiles(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Parse an ACCEPT-style per-cluster table into r_kpc, n_e(cm^-3), kT(keV), kT_err(keV or None).

    Expected whitespace-separated rows with columns:
      0: Name (string)
      1: Rin [Mpc]
      2: Rout [Mpc]
      3: nelec [cm^-3]
      4: neerr [cm^-3]
      ...
      13: Tx [keV]
      14: Txerr [keV]
    Lines starting with '#' are ignored.
    """
    r_kpc = []
    ne = []
    kT = []
    kT_err = []
    with open(path, 'r') as f:
        for line in f:
            ls = line.strip()
            if not ls or ls.startswith('#') or ls.startswith('###'):
                continue
            parts = ls.split()
            if len(parts) < 15:
                # not enough columns; skip
                continue
            try:
                Rin = float(parts[1])
                Rout = float(parts[2])
                r_mid_kpc = 0.5 * (Rin + Rout) * 1000.0  # Mpc -> kpc
                ne_val = float(parts[3])
                kT_val = float(parts[13])
                # Txerr may be missing or non-numeric; handle robustly
                try:
                    kT_err_val = float(parts[14])
                except Exception:
                    kT_err_val = np.nan
                r_kpc.append(r_mid_kpc)
                ne.append(ne_val)
                kT.append(kT_val)
                kT_err.append(kT_err_val)
            except Exception:
                # Skip malformed lines
                continue
    r_kpc = np.asarray(r_kpc, float)
    ne = np.asarray(ne, float)
    kT = np.asarray(kT, float)
    kT_err = np.asarray(kT_err, float)
    if not np.any(np.isfinite(kT_err)):
        kT_err = None
    return r_kpc, ne, kT, kT_err


def write_gas_temp(outdir: Path, r_kpc: np.ndarray, ne: np.ndarray, kT: np.ndarray, kT_err: np.ndarray | None, overwrite: bool=False) -> None:
    gp = outdir / 'gas_profile.csv'
    tp = outdir / 'temp_profile.csv'
    if (not gp.exists()) or overwrite:
        pd.DataFrame({'r_kpc': r_kpc, 'n_e_cm3': ne}).to_csv(gp, index=False)
        print(f"[accept_build] wrote {gp}")
    else:
        print(f"[accept_build] exists, skip gas: {gp}")
    if (not tp.exists()) or overwrite:
        df = pd.DataFrame({'r_kpc': r_kpc, 'kT_keV': kT})
        if kT_err is not None:
            df['kT_err_keV'] = kT_err
        df.to_csv(tp, index=False)
        print(f"[accept_build] wrote {tp}")
    else:
        print(f"[accept_build] exists, skip temp: {tp}")


def smooth_step_C(r_kpc: np.ndarray, C_inner: float, C_outer: float, r0_kpc: float, alpha: float) -> np.ndarray:
    x = np.power(np.maximum(r_kpc, 0.0) / max(r0_kpc, 1e-9), alpha)
    return C_inner + (C_outer - C_inner) * (x / (1.0 + x))


def write_clump(outdir: Path, r_kpc: np.ndarray, C_inner: float, C_outer: float, r0_kpc: float, alpha: float, overwrite: bool=False) -> None:
    cp = outdir / 'clump_profile.csv'
    if cp.exists() and not overwrite:
        print(f"[accept_build] exists, skip clump: {cp}")
        return
    C = smooth_step_C(r_kpc, C_inner=C_inner, C_outer=C_outer, r0_kpc=r0_kpc, alpha=alpha)
    pd.DataFrame({'r_kpc': r_kpc, 'C': C}).to_csv(cp, index=False)
    print(f"[accept_build] wrote {cp}")


def hernquist_rho(r_kpc: np.ndarray, Mtot_Msun: float, a_kpc: float) -> np.ndarray:
    r = np.asarray(r_kpc)
    return (Mtot_Msun * a_kpc) / (2.0 * np.pi * np.maximum(r, 1e-12) * np.power(r + a_kpc, 3))


def write_stars(outdir: Path, r_kpc: np.ndarray, Mtot_Msun: float, a_kpc: float,
                overwrite: bool=False,
                M2_Msun: float=0.0, a2_kpc: float=0.0) -> None:
    sp = outdir / 'stars_profile.csv'
    if sp.exists() and not overwrite:
        print(f"[accept_build] exists, skip stars: {sp}")
        return
    rho = hernquist_rho(r_kpc, Mtot_Msun=Mtot_Msun, a_kpc=a_kpc)
    if (M2_Msun is not None) and (a2_kpc is not None) and (float(M2_Msun) > 0.0) and (float(a2_kpc) > 0.0):
        rho = rho + hernquist_rho(r_kpc, Mtot_Msun=float(M2_Msun), a_kpc=float(a2_kpc))
    pd.DataFrame({'r_kpc': r_kpc, 'rho_star_Msun_per_kpc3': rho}).to_csv(sp, index=False)
    print(f"[accept_build] wrote {sp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', required=True, help='Cluster name (e.g., ABELL_0426, ABELL_1689)')
    ap.add_argument('--base', default='data/clusters', help='Base directory for cluster folders')
    ap.add_argument('--accept_path', default=None, help='Optional path to ACCEPT *_profiles.dat[.txt] for parsing gas/temp')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing CSVs')
    # Clumping params
    ap.add_argument('--clump_inner', type=float, default=1.0)
    ap.add_argument('--clump_outer', type=float, default=1.5)
    ap.add_argument('--clump_r0_kpc', type=float, default=800.0)
    ap.add_argument('--clump_alpha', type=float, default=2.0)
    # Stars params
    ap.add_argument('--stars_Mtot_Msun', type=float, default=1.5e12)
    ap.add_argument('--stars_a_kpc', type=float, default=30.0)
    ap.add_argument('--stars2_Mtot_Msun', type=float, default=0.0, help='Optional second Hernquist component mass (ICL)')
    ap.add_argument('--stars2_a_kpc', type=float, default=0.0, help='Optional second Hernquist component scale radius (kpc)')
    args = ap.parse_args()

    outdir = Path(args.base) / args.cluster
    outdir.mkdir(parents=True, exist_ok=True)

    # Determine r_kpc grid
    gas_csv = outdir / 'gas_profile.csv'
    have_gas_temp = gas_csv.exists() and (outdir / 'temp_profile.csv').exists()

    r_kpc = None
    ne = None
    kT = None
    kT_err = None

    if (not have_gas_temp) and args.accept_path:
        acc = Path(args.accept_path)
        if not acc.exists():
            print(f"[accept_build] ERROR: accept_path not found: {acc}", file=sys.stderr)
            sys.exit(2)
        r_kpc, ne, kT, kT_err = parse_accept_profiles(acc)
        write_gas_temp(outdir, r_kpc, ne, kT, kT_err, overwrite=args.overwrite)
        have_gas_temp = True

    if have_gas_temp:
        try:
            r_kpc = pd.read_csv(gas_csv)['r_kpc'].astype(float).values
        except Exception as e:
            print(f"[accept_build] ERROR reading r_kpc from {gas_csv}: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        # Could not find gas CSV or parse ACCEPT: create a generic log-spaced r grid
        r_kpc = np.geomspace(1.0, 1500.0, 200)
        print("[accept_build] WARN: no gas/temp found; generating generic r grid for clump/stars only")

    # Write clump and stars
    write_clump(outdir, r_kpc, C_inner=args.clump_inner, C_outer=args.clump_outer,
                r0_kpc=args.clump_r0_kpc, alpha=args.clump_alpha, overwrite=args.overwrite)
    write_stars(outdir, r_kpc, Mtot_Msun=args.stars_Mtot_Msun, a_kpc=args.stars_a_kpc,
                overwrite=args.overwrite,
                M2_Msun=float(args.stars2_Mtot_Msun), a2_kpc=float(args.stars2_a_kpc))

if __name__ == '__main__':
    main()
