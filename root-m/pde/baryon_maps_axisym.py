# -*- coding: utf-8 -*-
"""
root-m/pde/baryon_maps_axisym.py

Axisymmetric baryon map builders for SPARC using rotmod components.

We approximate component surface densities from the enclosed mass curves:
  M(<R) = V_comp(R)^2 * R / G  =>  Sigma(R) = (1/(2πR)) dM/dR
and place mass vertically with a single exponential scale height hz_kpc:
  rho(R,z) = Sigma(R) / (2 hz_kpc) * exp(-|z|/hz_kpc)

This preserves the surface density by construction and provides a simple
axisymmetric ρ(R,z) for the PDE. Bulges default to a spherical Hernquist model
with scale radius a = Re/1.8153 when an effective radius Re is available; if
not, a global fallback a_fallback is used. If a reliable total bulge mass is
not given in the parquet, it is estimated as max_R M_bulge(<R) from V_bul(R).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

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


def _hernquist_rho(r_kpc: np.ndarray, Mtot_Msun: float, a_kpc: float) -> np.ndarray:
    r = np.asarray(r_kpc, float)
    a = max(float(a_kpc), 1e-6)
    # ρ(r) = M a / [2π r (r+a)^3]
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = (Mtot_Msun * a) / (2.0*np.pi * np.maximum(r, 1e-12) * (r + a)**3)
    return rho


def _infer_bulge_params(dfg: pd.DataFrame,
                        fallback_a_kpc: float = 0.7) -> tuple[float, float]:
    """Return (M_bulge_Msun, a_kpc). Try to use explicit mass/size if present; else estimate.
    Column name guesses for Re: ['Re_kpc','R_e_kpc','Re_bulge_kpc','R_e_bulge_kpc','bulge_Re_kpc']
    Column name guesses for M:  ['Mbulge_Msun','M_bulge_Msun','Mb_Msun','Mbul_Msun']
    """
    # Effective radius -> Hernquist scale a
    Re_cols = ['Re_kpc','R_e_kpc','Re_bulge_kpc','R_e_bulge_kpc','bulge_Re_kpc']
    a_kpc: Optional[float] = None
    for c in Re_cols:
        if c in dfg.columns and pd.notnull(dfg[c].iloc[0]):
            try:
                a_kpc = float(dfg[c].iloc[0]) / 1.8153
                break
            except Exception:
                pass
    if a_kpc is None:
        a_kpc = float(fallback_a_kpc)

    # Bulge mass Mtot
    M_cols = ['Mbulge_Msun','M_bulge_Msun','Mb_Msun','Mbul_Msun']
    M_bul: Optional[float] = None
    for c in M_cols:
        if c in dfg.columns and pd.notnull(dfg[c].iloc[0]):
            try:
                M_bul = float(dfg[c].iloc[0])
                break
            except Exception:
                pass
    if M_bul is None:
        # fallback: estimate from Vbul(R) curve via max_R M(<R)
        if 'Vbul_kms' in dfg.columns:
            R = dfg['R_kpc'].to_numpy(float)
            Vbul = dfg['Vbul_kms'].to_numpy(float)
            Menc = (Vbul*Vbul) * R / G
            M_bul = float(np.nanmax(np.clip(Menc, 0.0, None)))
        else:
            M_bul = 0.0
    return M_bul, a_kpc


def axisym_map_from_rotmod_parquet(parquet_path: Path, galaxy: str,
                                   R_max: float=80.0, Z_max: float=80.0,
                                   NR: int=128, NZ: int=128,
                                   hz_kpc: float=0.3,
                                   bulge_model: str = 'hernquist',
                                   bulge_a_fallback_kpc: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build axisymmetric rho_b(R,z) for a single SPARC galaxy from rotmod components in a parquet file.
    The parquet is expected to have: galaxy, R_kpc, Vgas_kms, Vdisk_kms, Vbul_kms
    If bulge_model == 'hernquist', add a spherical Hernquist bulge using Re if available (a=Re/1.8153),
    else a=bulge_a_fallback_kpc. Bulge mass M_bulge is read from parquet if present or estimated from Vbul.
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

    # surface densities for disc+gas via Sigma-derivative route
    Sgas = sigma_from_M_of_R(R, Mgas)
    Sdisk= sigma_from_M_of_R(R, Mdisk)
    Ssum = Sgas + Sdisk

    # grid
    Rg = np.linspace(0.0, R_max, NR)
    Zg = np.linspace(-Z_max, Z_max, NZ)

    rho = axisym_rho_from_sigma(Rg, Zg, R, Ssum, hz_kpc)

    # Optional spherical Hernquist bulge
    if bulge_model.lower() == 'hernquist':
        M_bul, a_kpc = _infer_bulge_params(dfg, fallback_a_kpc=bulge_a_fallback_kpc)
        if M_bul > 0.0:
            # build spherical rho on grid and add
            Z2D = np.broadcast_to(Zg.reshape(-1,1), (Zg.size, Rg.size))
            R2D = np.broadcast_to(Rg.reshape(1,-1), (Zg.size, Rg.size))
            r2D = np.sqrt(R2D*R2D + Z2D*Z2D)
            rho_bul = _hernquist_rho(r2D, M_bul, a_kpc)
            rho = rho + rho_bul

    return Zg, Rg, rho
