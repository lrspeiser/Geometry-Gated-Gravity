#!/usr/bin/env python3
"""
precompute_thetaE_grids.py

Precompute per-cluster theta_E grids over (A_c, ell0_kpc, q_plane, q_LOS, kappa_ext),
saving to NPZ files with an index.json manifest.

Backwards-compatible: if only A_c and ell0 axes are requested, geometry is fixed to
(q_plane, q_LOS)=(1,1) and kappa_ext=0, reproducing the earlier 2D grids.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Optional, Tuple

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_mass_scaled_hierarchical_inference import (
    load_cluster_catalog,
    compute_theta_E_triaxial,
    compute_baryon_surface_density,
)
from scripts.lensing_utils import effective_distances
from scripts.cluster_overrides import load_cluster_override, normalize_cluster_name


def _build_axes(
    args: argparse.Namespace
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct grid axes from CLI.

    Priority:
    - If --grid counts are provided, build linspace between min/max for each axis.
    - Otherwise, use step-based A_c and ell0; geometry fixed to length-1; kappa fixed to 0.
    """
    # A_c and ell0
    if args.grid:
        try:
            nA, nL0, nQP, nQL, nK = [int(x) for x in args.grid.split(',')]
        except Exception as e:
            raise ValueError(f"--grid must be 5 integers, got {args.grid}") from e
        A_vals = np.linspace(args.ac_min, args.ac_max, nA)
        L0_vals = np.linspace(args.l0_min, args.l0_max, nL0)
        q_plane_vals = np.linspace(args.qplane_min, args.qplane_max, max(nQP, 1))
        q_los_vals = np.linspace(args.qlos_min, args.qlos_max, max(nQL, 1))
        kappa_vals = np.linspace(args.kappa_min, args.kappa_max, max(nK, 1))
    else:
        A_vals = np.arange(args.ac_min, args.ac_max + 1e-9, args.ac_step)
        L0_vals = np.arange(args.l0_min, args.l0_max + 1e-9, args.l0_step)
        q_plane_vals = np.array([1.0])
        q_los_vals = np.array([1.0])
        kappa_vals = np.array([0.0])
    return A_vals.astype(float), L0_vals.astype(float), q_plane_vals.astype(float), q_los_vals.astype(float), kappa_vals.astype(float)


def precompute_for_cluster(
    row: pd.Series,
    A_vals: np.ndarray,
    L0_vals: np.ndarray,
    q_plane_vals: np.ndarray,
    q_los_vals: np.ndarray,
    kappa_vals: np.ndarray,
) -> dict:
    name = row['cluster_name']
    R_kpc = np.linspace(1, 2000, 200)
    override = load_cluster_override(name)
    Sigma_bar = compute_baryon_surface_density(row, R_kpc, override=override)
    D_lens, D_source, D_LS = effective_distances(row['z_lens'], row.get('z_source', None), override=override)

    shape = (len(A_vals), len(L0_vals), len(q_plane_vals), len(q_los_vals), len(kappa_vals))
    theta = np.zeros(shape, dtype=float)

    for ia, A in enumerate(A_vals):
        for il, L0 in enumerate(L0_vals):
            for iqpl, qpl in enumerate(q_plane_vals):
                for iql, ql in enumerate(q_los_vals):
                    for ik, kap in enumerate(kappa_vals):
                        theta[ia, il, iqpl, iql, ik] = compute_theta_E_triaxial(
                            Sigma_bar, R_kpc, float(A), float(L0),
                            float(ql), float(qpl), float(kap),
                            D_lens, D_source, D_LS
                        )

    return {
        'A_grid': A_vals,
        'L0_grid': L0_vals,
        'q_plane_grid': q_plane_vals,
        'q_LOS_grid': q_los_vals,
        'kappa_grid': kappa_vals,
        'thetaE': theta,
        'meta': {
            'cluster_name': name,
            'z_lens': float(row['z_lens']),
            'R500_Mpc': float(row['R500_Mpc'])
        }
    }


def main():
    ap = argparse.ArgumentParser(description='Precompute theta_E grids')
    ap.add_argument('--catalog', default=None)
    ap.add_argument('--tiers', default='1,2')
    ap.add_argument('--exclude', default=None)
    ap.add_argument('--include', default=None)
    # A_c axis
    ap.add_argument('--ac-min', type=float, default=2.0)
    ap.add_argument('--ac-max', type=float, default=12.0)
    ap.add_argument('--ac-step', type=float, default=0.25)
    # ell0 axis
    ap.add_argument('--l0-min', type=float, default=120.0)
    ap.add_argument('--l0-max', type=float, default=320.0)
    ap.add_argument('--l0-step', type=float, default=10.0)
    # geometry axes
    ap.add_argument('--qplane-min', type=float, default=0.7)
    ap.add_argument('--qplane-max', type=float, default=1.4)
    ap.add_argument('--qlos-min', type=float, default=0.7)
    ap.add_argument('--qlos-max', type=float, default=1.4)
    # kappa axis
    ap.add_argument('--kappa-min', type=float, default=-0.12)
    ap.add_argument('--kappa-max', type=float, default=0.12)
    # counts per axis (A, L0, q_plane, q_LOS, kappa)
    ap.add_argument('--grid', type=str, default=None, help='Comma-separated counts, e.g., 21,21,11,11,9')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    tiers = [int(t) for t in args.tiers.split(',')]
    exclude = args.exclude.split(',') if args.exclude else None
    include = args.include.split(',') if args.include else None
    catalog_path = Path(args.catalog) if args.catalog else None

    df = load_cluster_catalog(tiers, exclude, catalog_path, include)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    A_vals, L0_vals, qpl_vals, ql_vals, k_vals = _build_axes(args)

    index = []
    for _, row in df.iterrows():
        data = precompute_for_cluster(row, A_vals, L0_vals, qpl_vals, ql_vals, k_vals)
        key = normalize_cluster_name(row['cluster_name'])
        np.savez_compressed(outdir / f'{key}.npz', **data)
        index.append({
            'cluster_name': row['cluster_name'],
            'file': f'{key}.npz',
            'shape': [int(len(A_vals)), int(len(L0_vals)), int(len(qpl_vals)), int(len(ql_vals)), int(len(k_vals))]
        })
        print(f'Wrote grid for {row["cluster_name"]} -> {key}.npz')

    with open(outdir / 'index.json', 'w') as f:
        json.dump(index, f, indent=2)


if __name__ == '__main__':
    main()
