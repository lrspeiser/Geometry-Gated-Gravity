# -*- coding: utf-8 -*-
"""
O3 lensing module: third-order, lensing-only booster that is
- non-local and geometry-aware via a long-range kernel over Sigma(R)
- tracer-mass selective (photons get largest boost)

This module deliberately modifies only the lensing leg. Dynamics remain g_dyn.

Inputs use repo-wide units:
- R in kpc
- Sigma in Msun/pc^2
- g in (km/s)^2 / kpc
"""
from __future__ import annotations
import numpy as np


def smootherstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x*x*x*(x*(x*6.0 - 15.0) + 10.0)


def gaussian_weights(R: np.ndarray, ell_kpc: float) -> np.ndarray:
    """Return an (N,N) matrix W for Gaussian smoothing over R with width ell.
    Each row i contains weights centered at R[i]. Rows are normalized to sum to 1.
    """
    R = np.asarray(R, float)
    N = R.size
    if N == 0:
        return np.zeros((0, 0))
    ell = max(float(ell_kpc), 1e-6)
    # Broadcasted pairwise distances
    d = R.reshape(-1, 1) - R.reshape(1, -1)
    W = np.exp(-0.5 * (d / ell)**2)
    # Normalize rows
    W /= (W.sum(axis=1, keepdims=True) + 1e-30)
    return W


def nonlocal_env_boost(
    R_kpc: np.ndarray,
    Sigma_Msun_pc2: np.ndarray,
    ell3_kpc: float = 400.0,
    Sigma_star3_Msun_pc2: float = 30.0,
    beta3: float = 1.0,
    r3_kpc: float = 80.0,
    w3_decades: float = 1.0,
    xi3: float = 1.2,
) -> np.ndarray:
    """Compute the non-local environment boost B_env(R).

    Steps:
    - Smooth Sigma(R) with a Gaussian in R of width ell3_kpc to get Sigma_L(R).
    - Convert to [0,1] gate via x = Sigma_L/(Sigma_star3 + Sigma_L) and raise by beta3.
    - Apply a C^2 radial smootherstep that transitions over w3_decades around r3_kpc.

    Returns B_env >= 1.
    """
    R = np.asarray(R_kpc, float)
    Sigma = np.asarray(Sigma_Msun_pc2, float)
    W = gaussian_weights(R, ell3_kpc)
    Sigma_L = W @ Sigma
    x = Sigma_L / (np.maximum(Sigma_star3_Msun_pc2, 1e-12) + np.maximum(Sigma_L, 0.0))
    gate_env = np.power(np.clip(x, 0.0, 1.0), float(beta3))
    # Radial scale smoother: map log10(R/r3) across [-w/2, +w/2]
    w = max(float(w3_decades), 1e-6)
    u = (np.log10(np.maximum(R, 1e-12)) - np.log10(max(r3_kpc, 1e-12))) / w + 0.5
    t = smootherstep(u)
    return 1.0 + float(xi3) * gate_env * t


def apply_o3_lensing(
    g_dyn_R: np.ndarray,
    R_kpc: np.ndarray,
    Sigma_Msun_pc2: np.ndarray,
    params: dict,
    m_test_Msun: float = 0.0,
) -> np.ndarray:
    """Apply O3 multiplicative boost to the lensing field.

    g_lens(R) = g_dyn(R) * B_env(R) * B_test(m_test)

    params keys:
    - ell3_kpc, Sigma_star3_Msun_pc2, beta3, r3_kpc, w3_decades, xi3
    - A3, chi, m_ref_Msun, m_floor_Msun
    """
    g_dyn = np.asarray(g_dyn_R, float)
    R = np.asarray(R_kpc, float)
    Sigma = np.asarray(Sigma_Msun_pc2, float)

    B_env = nonlocal_env_boost(
        R, Sigma,
        ell3_kpc=float(params.get('ell3_kpc', 400.0)),
        Sigma_star3_Msun_pc2=float(params.get('Sigma_star3_Msun_pc2', 30.0)),
        beta3=float(params.get('beta3', 1.0)),
        r3_kpc=float(params.get('r3_kpc', 80.0)),
        w3_decades=float(params.get('w3_decades', 1.0)),
        xi3=float(params.get('xi3', 1.2)),
    )
    A3 = float(params.get('A3', 0.05))
    chi = float(params.get('chi', 1.0))
    m_ref = float(params.get('m_ref_Msun', 1.0))
    m_floor = float(params.get('m_floor_Msun', 1e-8))
    B_test = 1.0 + A3 * (m_ref / (float(m_test_Msun) + m_floor))**chi
    return g_dyn * (B_env * B_test)
