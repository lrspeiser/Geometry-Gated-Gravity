# -*- coding: utf-8 -*-
"""
O3 slip model: lensing-only multiplicative amplification of the dynamical surface density.

Sigma_lens(R) = [1 + alpha_env(R)] * (Sigma_dyn(R) convolved with optional Gaussian kernel K_ell)

alpha_env(R) = A3 * S_sigma_low(Σ; Σ*, β) * S_R(R; r_boost, w_dec) * S_curv_low(|d^2 ln ρ/d ln r^2|; c_curv)

Design goals:
- Engage in weak-field, low-surface-density, low-curvature environments (cluster outskirts),
  while tapering to zero in dense, high-curvature cores to avoid MOND/DM-like inner boosts.
- Strictly lensing-only: dynamics remain g_dyn; photons (m≈0) see the largest effect at the final leg.

All gates are C^2 smoothersteps or rational forms in [0,1].
Units:
- R in kpc
- Sigma in Msun/pc^2
- rho in Msun/kpc^3
"""
from __future__ import annotations
import numpy as np
from typing import Dict


def smootherstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x*x*x*(x*(x*6.0 - 15.0) + 10.0)


def S_sigma(Sigma_pc2: np.ndarray, Sigma_star: float, beta: float) -> np.ndarray:
    """Monotone decreasing gate in Σ: ~1 at low Σ, →0 at high Σ.
    S_sigma_low = (Σ* / (Σ + Σ*))^β
    """
    Sigma = np.maximum(Sigma_pc2, 0.0)
    return (max(Sigma_star, 1e-12) / (Sigma + max(Sigma_star, 1e-12))) ** float(beta)


def S_R(R_kpc: np.ndarray, r_boost_kpc: float, w_decades: float) -> np.ndarray:
    # Map log10 R around r_boost across +/- (w_decades/2)
    R = np.maximum(R_kpc, 1e-9)
    t = (np.log10(R) - np.log10(max(r_boost_kpc, 1e-12))) / max(w_decades, 1e-6) + 0.5
    return smootherstep(t)


def curvature_proxy(logr: np.ndarray, logrho: np.ndarray) -> np.ndarray:
    d1 = np.gradient(logrho, logr)
    d2 = np.gradient(d1, logr)
    return np.abs(d2)


def S_curv(kappa: np.ndarray, c_curv: float) -> np.ndarray:
    """Monotone decreasing gate in curvature: ~1 at low curvature, →0 at high curvature."""
    x = np.maximum(kappa, 0.0) / max(c_curv, 1e-6)
    return 1.0 - smootherstep(x)


def gaussian_kernel_1d(R_kpc: np.ndarray, ell_kpc: float) -> np.ndarray:
    # Build a row-normalized Gaussian kernel for 1D convolution on non-uniform R grid.
    # For simplicity, we approximate by using pairwise distances on R grid.
    R = np.asarray(R_kpc, float)
    ell = float(max(ell_kpc, 1e-6))
    d = np.abs(R.reshape(-1,1) - R.reshape(1,-1))
    W = np.exp(-0.5 * (d/ell)**2)
    W /= (W.sum(axis=1, keepdims=True) + 1e-30)
    return W


def apply_slip(
    R_kpc: np.ndarray,
    Sigma_dyn_pc2: np.ndarray,
    rho_dyn_kpc3: np.ndarray,
    params: Dict[str, float],
) -> np.ndarray:
    """Return Sigma_lens_pc2 given Sigma_dyn_pc2 and rho_dyn_kpc3 (for curvature proxy).
    params: {'A3','Sigma_star','beta','r_boost_kpc','w_dec','ell_kpc','c_curv'}
    """
    R = np.asarray(R_kpc, float)
    Sigma = np.asarray(Sigma_dyn_pc2, float)
    rho = np.asarray(rho_dyn_kpc3, float)
    # Optional Gaussian smoothing of Sigma_dyn before gating
    ell = float(params.get('ell_kpc', 0.0))
    if ell > 0.0:
        W = gaussian_kernel_1d(R, ell)
        Sigma_sm = W @ Sigma
    else:
        Sigma_sm = Sigma
    # Curvature proxy from rho
    logr = np.log(np.maximum(R, 1e-12))
    logrho = np.log(np.maximum(rho, 1e-30))
    kappa_curv = curvature_proxy(logr, logrho)
    # Gates (favor low Σ, large R, low curvature)
    sS = S_sigma(Sigma_sm, float(params.get('Sigma_star', 50.0)), float(params.get('beta', 1.0)))
    sR = S_R(R, float(params.get('r_boost_kpc', 400.0)), float(params.get('w_dec', 1.0)))
    sC = S_curv(kappa_curv, float(params.get('c_curv', 0.8)))
    A3 = float(params.get('A3', 0.1))
    alpha = A3 * sS * sR * sC
    # Final lensing Sigma
    Sigma_lens = (1.0 + alpha) * Sigma_sm
    return Sigma_lens
