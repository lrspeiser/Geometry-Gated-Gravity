#!/usr/bin/env python3
"""
Cooperative Response: Additive Piggybacking for Cluster Lensing
================================================================

Implements geometry-gated, non-local mass "export" from dense cores to sparse
outskirts via a positive response kernel. This creates the **cooperative amplification**
where baryons in dense regions enhance deflection in low-Σ halos.

Physics:
    Σ_eff(R) = Σ_baryon(R) + Σ_resp(R)
    
    Σ_resp(R) = A_resp · w_rec(R) · ∫ K(|R-R'|) · w_don(R') · Σ(R') 2πR' dR'

Where:
    - w_don(R'): donor gate (upweights dense cores)
    - w_rec(R): recipient gate (upweights sparse outskirts)
    - K: power-law kernel with scale λ and tail index ν
    - A_resp: amplitude predicted from baryon geometry (ε, M_core)

Universal Rules (from 3-cluster training):
    A_resp = 8.0 · ε^0.5 · (M_core/10^13)^0.3
    λ = 1.3 · R_edge
    ν = 2.0

Author: AI Assistant + User
Date: 2025-01-10
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Optional
import warnings

def logistic(x: np.ndarray, x0: float = 0.3, w: float = 0.3) -> np.ndarray:
    """Smooth logistic function for gating."""
    return 1.0 / (1.0 + np.exp(-(x - x0) / max(w, 1e-6)))


def mean_sigma_inside_R(R: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Compute mean surface density Σ̄(<R) inside radius R.
    
    Σ̄(<R) = M(<R) / (π R²)
    
    Parameters
    ----------
    R : np.ndarray
        Radii in kpc (must be sorted ascending)
    Sigma : np.ndarray
        Surface density in Msun/kpc²
        
    Returns
    -------
    np.ndarray
        Mean surface density in Msun/kpc²
    """
    from scipy.integrate import cumulative_trapezoid
    
    area = np.pi * R**2
    M_enc = cumulative_trapezoid(Sigma * 2.0 * np.pi * R, R, initial=0.0)
    return M_enc / np.maximum(area, 1e-30)


def donor_recipient_gates(R: np.ndarray, 
                          Sigma: np.ndarray,
                          x0: float = 0.3, 
                          w: float = 0.3,
                          Sigma0_pc2: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute donor and recipient gates from baryon surface density.
    
    Dense regions (high Σ̄) are donors; sparse regions (low Σ̄) are recipients.
    
    Gates are based on log-density relative to Σ₀ = 100 Msun/pc²:
        Ŝ = log₁₀(Σ̄(<R) / Σ₀)
        w_don = σ(Ŝ; x₀, w)      # cores donate
        w_rec = 1 - σ(Ŝ; x₀, w)  # outskirts receive
    
    Parameters
    ----------
    R : np.ndarray
        Radii in kpc
    Sigma : np.ndarray
        Surface density in Msun/kpc²
    x0 : float
        Logistic center (log-density threshold)
    w : float
        Logistic width
    Sigma0_pc2 : float
        Reference density in Msun/pc²
        
    Returns
    -------
    w_don : np.ndarray
        Donor weights (0 to 1)
    w_rec : np.ndarray
        Recipient weights (0 to 1)
    """
    Sigma_bar_pc2 = mean_sigma_inside_R(R, Sigma) / 1e6  # Convert to Msun/pc²
    Shat = np.log10(np.maximum(Sigma_bar_pc2, 1e-12) / Sigma0_pc2)
    
    w_don = logistic(Shat, x0=x0, w=w)          # High Σ̄ → w_don ≈ 1
    w_rec = 1.0 - logistic(Shat, x0=x0, w=w)    # Low Σ̄ → w_rec ≈ 1
    
    return w_don, w_rec


def power_kernel(delta_R: np.ndarray, 
                lam: float, 
                nu: float = 2.0, 
                eps: float = 1e-6) -> np.ndarray:
    """
    Normalized positive power-law kernel for mass export.
    
    K(ΔR) = 1 / (1 + ΔR/λ)^ν
    
    Parameters
    ----------
    delta_R : np.ndarray
        Radial separation |R - R'| in kpc
    lam : float
        Export scale in kpc
    nu : float
        Power-law index (2.0 gives 1/r² tail)
    eps : float
        Small offset to avoid division by zero
        
    Returns
    -------
    np.ndarray
        Kernel values (dimensionless)
    """
    return 1.0 / (1.0 + np.maximum(delta_R, eps) / max(lam, eps))**nu


def cooperative_response_sigma(R: np.ndarray,
                               Sigma: np.ndarray,
                               A_resp: float,
                               lam: float,
                               nu: float = 2.0,
                               x0: float = 0.3,
                               w: float = 0.3,
                               conserve_mass: bool = False,
                               debug: bool = False) -> np.ndarray:
    """
    Compute response surface density Σ_resp(R) from cooperative piggybacking.
    
    Σ_resp(R) = A_resp · w_rec(R) · ∫ K(|R-R'|) · w_don(R') · Σ(R') 2πR' dR'
    
    This is the **additive** component that spreads influence from dense cores
    to sparse outskirts via a non-local kernel.
    
    Parameters
    ----------
    R : np.ndarray
        Radii in kpc (sorted ascending)
    Sigma : np.ndarray
        Baryon surface density in Msun/kpc²
    A_resp : float
        Response amplitude (dimensionless)
    lam : float
        Export scale in kpc
    nu : float
        Kernel tail index
    x0, w : float
        Logistic gate parameters
    conserve_mass : bool
        If True, subtract from donors to keep ∫Σ_resp dA ≈ 0
    debug : bool
        If True, print diagnostic info
        
    Returns
    -------
    np.ndarray
        Response surface density Σ_resp in Msun/kpc²
    """
    R = np.asarray(R, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    
    if len(R) != len(Sigma):
        raise ValueError(f"R and Sigma must have same length: {len(R)} vs {len(Sigma)}")
    
    dR = np.gradient(R)
    w_don, w_rec = donor_recipient_gates(R, Sigma, x0=x0, w=w)
    
    # Build kernel matrix K_ij = K(|R_i - R_j|)
    dR_matrix = np.abs(R[:, None] - R[None, :])
    K = power_kernel(dR_matrix, lam=lam, nu=nu)
    
    # Donor weighting with area factor (mass ring elements)
    donor_vec = w_don * Sigma * (2.0 * np.pi * R * dR)  # Msun
    
    # Response via matrix multiply: Σ_resp(R_i) = Σ_j K_ij · donor_vec_j / (2π R_i dR_i)
    Sigma_resp_unnorm = K @ donor_vec
    Sigma_resp = A_resp * w_rec * Sigma_resp_unnorm / (2.0 * np.pi * R * np.maximum(dR, 1e-30))
    
    if conserve_mass:
        # Remove mass from donors so ∫ Σ_resp dA ≈ 0 (pure reweighting)
        total_added = 2.0 * np.pi * np.trapz(Sigma_resp * R, R)
        donor_total = 2.0 * np.pi * np.trapz(w_don * Sigma * R, R)
        alpha = np.clip(total_added / np.maximum(donor_total, 1e-30), 0.0, 0.5)
        Sigma_resp -= alpha * w_don * Sigma
        Sigma_resp = np.maximum(Sigma_resp, 0.0)
        
        if debug:
            print(f"  Mass conservation: removed {alpha:.3f} × donor mass")
    
    if debug:
        print(f"  Σ_resp range: [{Sigma_resp.min():.3e}, {Sigma_resp.max():.3e}] Msun/kpc²")
        print(f"  w_don range: [{w_don.min():.3f}, {w_don.max():.3f}]")
        print(f"  w_rec range: [{w_rec.min():.3f}, {w_rec.max():.3f}]")
        M_resp_total = 2.0 * np.pi * np.trapz(Sigma_resp * R, R)
        M_baryon_total = 2.0 * np.pi * np.trapz(Sigma * R, R)
        print(f"  M_resp / M_baryon = {M_resp_total / M_baryon_total:.3f}")
    
    return Sigma_resp


def cooperative_sigma_eff(R: np.ndarray,
                         Sigma: np.ndarray,
                         A_resp: float,
                         lam: float,
                         nu: float = 2.0,
                         x0: float = 0.3,
                         w: float = 0.3,
                         conserve_mass: bool = False,
                         debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute effective surface density including cooperative response.
    
    Σ_eff = Σ_baryon + Σ_resp
    
    Parameters
    ----------
    R : np.ndarray
        Radii in kpc
    Sigma : np.ndarray
        Baryon surface density in Msun/kpc²
    A_resp : float
        Response amplitude
    lam : float
        Export scale in kpc
    nu : float
        Kernel tail index
    x0, w : float
        Gate parameters
    conserve_mass : bool
        Mass conservation flag
    debug : bool
        Debug output flag
        
    Returns
    -------
    Sigma_eff : np.ndarray
        Effective surface density in Msun/kpc²
    Sigma_resp : np.ndarray
        Response component in Msun/kpc²
    """
    Sigma_resp = cooperative_response_sigma(
        R, Sigma, A_resp, lam, nu, x0, w, conserve_mass, debug
    )
    
    Sigma_eff = Sigma + Sigma_resp
    
    return Sigma_eff, Sigma_resp


def predict_response_hyperparams(features: Dict) -> Tuple[float, float, float, float]:
    """
    Predict response hyperparameters from universal baryon scaling laws.
    
    Universal rules (learned from 3-cluster training):
        A_resp = 8.0 · ε^0.5 · (M_core/10^13)^0.3
        λ = 1.3 · R_edge
        ν = 2.0
        β_bandpass = 0.6 if merger (n_peaks > 1 or c_out < -0.2) else 0.0
    
    Parameters
    ----------
    features : dict
        Must contain:
            'edge_sharp': ε (edge sharpness)
            'core_mass': M_core in Msun
            'R_edge': R_edge in kpc
            'n_peaks': number of peaks (optional, default 1)
            'c_out': curvature at edge (optional, default 0)
            
    Returns
    -------
    A_resp : float
        Response amplitude (dimensionless)
    lam : float
        Export scale in kpc
    nu : float
        Kernel tail index
    beta_bandpass : float
        Merger bandpass coefficient (0 for relaxed, 0.6 for mergers)
    """
    eps = features['edge_sharp']
    M_core = features['core_mass']
    R_edge = features['R_edge']
    n_peaks = features.get('n_peaks', 1)
    c_out = features.get('c_out', 0.0)
    
    # Universal ε₀ rule
    A_resp = 8.0 * (max(eps, 0.01)**0.5) * (max(M_core, 1e10) / 1e13)**0.3
    
    # Universal R_a rule
    lam = 1.3 * max(R_edge, 1.0)
    
    # Fixed tail index
    nu = 2.0
    
    # Merger flag
    beta_bandpass = 0.6 if (n_peaks > 1 or c_out < -0.2) else 0.0
    
    return float(A_resp), float(lam), float(nu), float(beta_bandpass)


def bandpass_response_dog(R: np.ndarray,
                         Sigma: np.ndarray,
                         beta: float,
                         lam1: float = 75.0,
                         lam2: float = 240.0,
                         nu1: float = 1.8,
                         nu2: float = 1.8,
                         **gate_kwargs) -> np.ndarray:
    """
    Difference-of-Gaussians bandpass response for mergers.
    
    Captures multi-scale structure in merging systems.
    
    Σ_resp_dog = β · w_rec · [(K_λ2 * Σ) - (K_λ1 * Σ)]
    
    Parameters
    ----------
    R : np.ndarray
        Radii in kpc
    Sigma : np.ndarray
        Surface density in Msun/kpc²
    beta : float
        Bandpass amplitude
    lam1, lam2 : float
        Inner and outer kernel scales in kpc
    nu1, nu2 : float
        Kernel tail indices
    gate_kwargs : dict
        Passed to donor_recipient_gates
        
    Returns
    -------
    np.ndarray
        Bandpass response in Msun/kpc²
    """
    if beta < 1e-6:
        return np.zeros_like(R)
    
    dR = np.gradient(R)
    _, w_rec = donor_recipient_gates(R, Sigma, **gate_kwargs)
    
    # Build two kernels
    dR_matrix = np.abs(R[:, None] - R[None, :])
    K1 = power_kernel(dR_matrix, lam=lam1, nu=nu1)
    K2 = power_kernel(dR_matrix, lam=lam2, nu=nu2)
    
    # Convolution with Sigma
    area_vec = Sigma * (2.0 * np.pi * R * dR)
    conv1 = K1 @ area_vec
    conv2 = K2 @ area_vec
    
    # Difference of Gaussians
    dog = (conv2 - conv1) / (2.0 * np.pi * R * np.maximum(dR, 1e-30))
    
    Sigma_resp_dog = beta * w_rec * dog
    
    return Sigma_resp_dog


# ==================== TESTING / DEMO ====================

if __name__ == "__main__":
    print("Testing cooperative response module...\n")
    
    # Mock cluster with power-law baryon profile
    R = np.logspace(np.log10(0.1), np.log10(1000.0), 500)
    
    # Core + declining envelope (typical cluster)
    R_c = 50.0  # core scale
    Sigma_0 = 1e9  # central density
    Sigma = Sigma_0 / (1.0 + (R / R_c)**2)**1.5  # β-model-like
    
    # Mock features
    features = {
        'edge_sharp': 2.0,
        'core_mass': 1.5e13,
        'R_edge': 150.0,
        'n_peaks': 1,
        'c_out': -0.1
    }
    
    # Predict hyperparams
    A_resp, lam, nu, beta = predict_response_hyperparams(features)
    print(f"Universal predictions:")
    print(f"  A_resp = {A_resp:.3f}")
    print(f"  λ = {lam:.1f} kpc")
    print(f"  ν = {nu:.1f}")
    print(f"  β_bandpass = {beta:.1f}\n")
    
    # Compute cooperative response
    Sigma_eff, Sigma_resp = cooperative_sigma_eff(
        R, Sigma, A_resp, lam, nu, debug=True
    )
    
    # Diagnostics
    f_resp = Sigma_resp / np.maximum(Sigma_eff, 1e-30)
    B_kappa = Sigma_eff / np.maximum(Sigma, 1e-30)
    
    print(f"\nPiggybacking metrics:")
    print(f"  f_resp(R=100 kpc) = {np.interp(100.0, R, f_resp):.3f}")
    print(f"  f_resp(R=200 kpc) = {np.interp(200.0, R, f_resp):.3f}")
    print(f"  B_κ(R=100 kpc) = {np.interp(100.0, R, B_kappa):.3f}")
    print(f"  B_κ(R=200 kpc) = {np.interp(200.0, R, B_kappa):.3f}")
    
    print("\n✓ Cooperative response module test complete")
