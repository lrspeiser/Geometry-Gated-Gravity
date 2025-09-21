# -*- coding: utf-8 -*-
"""
root-m/pde/baryon_maps_axisym.py

Axisymmetric baryon map builders for SPARC using rotmod components.

We approximate component surface densities from the enclosed mass curves:
  M(<R) = V_comp(R)^2 * R / G  =>  Sigma(R) = (1/(2πR)) dM/dR
and place mass vertically with a single exponential scale height hz_kpc:
  rho(R,z) = Sigma(R) / (2 hz_kpc) * exp(-|z|/hz_kpc)

This preserves the surface density by construction and provides a simple
axisymmetric ρ(R,z) for the PDE. Bulges are treated via the same Sigma-derivative
route for simplicity; a Hernquist option can be added later.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)


def sigma_from_M_of_R(R_kpc: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
    R = np.asarray(R_kpc, float)
    M = np.asarray(M_enc, float)
    dM_dR = np.gradient(M, R)
    with np.errstate(divide='ignore', invalid='ignore'):
        Sigma = (dM_dR) / (2.0 * np.pi * np.maximum(R, 1e-12))
    Sigma = np.clip(Sigma, 0.0, None)
    return Sigma


def axisym_rho_from_sigma(R_grid: np.ndarray, Z_grid: np.ndarray, R_samp: np.ndarray,
                          Sigma_R: np.ndarray, hz_kpc: float) -> np.ndarray:
    """Build rho(R,z) = Sigma(R)/(2hz) * exp(-|z|/hz) on (Z,R) grid from tabulated Sigma(R)."""
    R = np.asarray(R_grid)
    Z = np.asarray(Z_grid)
    Sigma_on_grid = np.interp(R, R_samp, Sigma_R, left=Sigma_R[0], right=Sigma_R[-1])
    # shape to 2D
    Sigma2D = np.broadcast_to(Sigma_on_grid.reshape(1, -1), (Z.size, R.size))
    Z2D = np.broadcast_to(Z.reshape(-1, 1), (Z.size, R.size))
    hz = max(hz_kpc, 1e-3)
    rho = (Sigma2D / (2.0 * hz)) * np.exp(-np.abs(Z2D) / hz)
    return rho


def axisym_map_from_rotmod_parquet(parquet_path: Path, galaxy: str,
                                   R_max: float=80.0, Z_max: float=80.0,
                                   NR: int=128, NZ: int=128,
                                   hz_kpc: float=0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build axisymmetric rho_b(R,z) for a single SPARC galaxy from rotmod components in a parquet file.
    The parquet is expected to have: galaxy, R_kpc, Vgas_kms, Vdisk_kms, Vbul_kms
    """
    df = pd.read_parquet(parquet_path)
    dfg = df[df['galaxy'] == galaxy].copy()
    if dfg.empty:
        raise ValueError(f"Galaxy {galaxy} not found in {parquet_path}")
    # ensure unique radii
    dfg = dfg.sort_values('R_kpc')
    R = dfg['R_kpc'].to_numpy(float)
    Vgas = dfg['Vgas_kms'].to_numpy(float) if 'Vgas_kms' in dfg.columns else np.zeros_like(R)
    Vdisk = dfg['Vdisk_kms'].to_numpy(float) if 'Vdisk_kms' in dfg.columns else np.zeros_like(R)
    Vbul  = dfg['Vbul_kms'].to_numpy(float) if 'Vbul_kms' in dfg.columns else np.zeros_like(R)

    # component masses
    Mgas = (Vgas*Vgas) * R / G
    Mdisk= (Vdisk*Vdisk) * R / G
    Mbul = (Vbul*Vbul)  * R / G

    # surface densities
    Sgas = sigma_from_M_of_R(R, Mgas)
    Sdisk= sigma_from_M_of_R(R, Mdisk)
    Sbul = sigma_from_M_of_R(R, Mbul)
    Ssum = Sgas + Sdisk + Sbul

    # grid
    Rg = np.linspace(0.0, R_max, NR)
    Zg = np.linspace(-Z_max, Z_max, NZ)

    rho = axisym_rho_from_sigma(Rg, Zg, R, Ssum, hz_kpc)
    return Zg, Rg, rho
