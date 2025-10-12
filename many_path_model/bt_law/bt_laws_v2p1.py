#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended B/T Laws v2.1: Enhanced with Lambda Two-Predictor Law & Ring Concentration
=====================================================================================

Adds two critical fixes to v2:
1. Lambda_ring now depends on BOTH B/T and shear S (prevents "too long coherence")
2. Ring term has radial Gaussian concentration (prevents broad outer overshoot)

Predictors:
1. B/T (bulge-to-total) - sphericity/inner structure
2. Sigma0 (surface density) - compactness gating for eta, M_max
3. Shear S - coherence control AND lambda scaling
4. Coherence factor kappa - decoherence of ring winding

Key improvements over v2:
- Lambda is now suppressed for high-shear systems (S > 1) → fixes overshoot
- Ring amplitude concentrated radially at R_ring(B/T) → fixes broad boost
"""
import json
import numpy as np


def law_value(B, lo, hi, gamma):
    """
    Monotonic law: y(B) = lo + (hi - lo) * (1 - B)**gamma
    """
    B = np.clip(B, 0.0, 1.0)
    gamma = np.maximum(gamma, 1e-3)
    return lo + (hi - lo) * (1.0 - B) ** gamma


def compactness_gate(Sigma0, Sigma_ref=100.0, gamma_Sigma=0.8):
    """
    Smooth compactness gating: f(Σ) = (Σ/Σ_ref)^γ / (1 + (Σ/Σ_ref)^γ)
    
    Returns value in [0, 1]:
    - Low Σ (LSB dwarfs) → 0
    - High Σ (HSB spirals) → 1
    """
    if Sigma0 is None or np.isnan(Sigma0) or Sigma0 <= 0:
        return 0.5  # Neutral default
    
    x = (Sigma0 / Sigma_ref) ** gamma_Sigma
    return x / (1.0 + x)


def shear_gate(S, S0=0.8, n_s=2.0):
    """
    Shear gating for coherence: f(S) = 1 / (1 + (|S|/S0)^n_s)
    
    Returns value in [0, 1]:
    - Low shear (rising curves, S < 1) → 1 (strong coherence)
    - High shear (declining curves, S > 1) → 0 (weak coherence)
    """
    if S is None or np.isnan(S):
        return 0.5  # Neutral default
    
    return 1.0 / (1.0 + (abs(S) / S0) ** n_s)


def lambda_two_predictor(B, S, lambda_min, lambda_max, 
                         gamma_bulge=4.0, gamma_shear=3.0, S0=0.8):
    """
    Lambda depends on BOTH B/T and shear S.
    
    λ(B, S) = λ_min + (λ_max - λ_min) * (1 - B)^γ_b * (1 - g_shear(S))^γ_s
    
    where g_shear(S) = (S/S0)^n / (1 + (S/S0)^n) saturates at high shear.
    
    Physical interpretation:
    - More disk (low B) → longer λ (more coherence)
    - Higher shear (S >> S0) → shorter λ (dephasing kills long loops)
    
    This directly addresses the "red overshoot" failure mode where V2
    picked λ too long for high-shear intermediate spirals.
    """
    # Bulge suppression (disk-dominated → longer λ)
    f_bulge = (1.0 - np.clip(B, 0, 1)) ** gamma_bulge
    
    # Shear suppression (high shear → shorter λ)
    if S is None or np.isnan(S):
        g_shear_val = 0.5
    else:
        # Use a soft saturation function
        x = np.maximum(abs(S) / S0, 1e-6)
        g_shear_val = (x**2) / (1.0 + x**2)  # Goes from 0 at S=0 to ~1 at S>>S0
    
    f_shear = (1.0 - g_shear_val) ** gamma_shear
    
    lam = lambda_min + (lambda_max - lambda_min) * f_bulge * f_shear
    
    return np.clip(lam, lambda_min, lambda_max)


def ring_radial_envelope(R, R_ring, sigma_ring=0.5):
    """
    Radial concentration envelope for ring term.
    
    A_ring(R) = exp(-(R/R_ring)^2 / (2*sigma_ring^2))
    
    Concentrates ring boost near R_ring (typically 2-4 R_d for spirals).
    Prevents the broad outer overshoot seen in V2.
    
    Args:
        R: Radius (kpc), can be array
        R_ring: Concentration radius (kpc)
        sigma_ring: Width parameter (dimensionless)
    
    Returns:
        Envelope factor in [0, 1]
    """
    x = R / np.maximum(R_ring, 1e-3)
    return np.exp(-0.5 * (x / sigma_ring)**2)


def compute_R_ring(B, R_d, b0=2.5, b1=1.2):
    """
    Compute ring concentration radius as function of B/T.
    
    R_ring(B) = (b0 - b1*B) * R_d
    
    Physical interpretation:
    - Late types (B=0): R_ring ≈ 2.5 R_d (arms in outer disk)
    - Early types (B=0.7): R_ring ≈ 1.6 R_d (tighter, more nuclear)
    
    Args:
        B: Bulge-to-total fraction
        R_d: Disk scale length (kpc)
        b0, b1: Scaling coefficients
    
    Returns:
        R_ring in kpc
    """
    factor = b0 - b1 * np.clip(B, 0, 1)
    R_ring = factor * R_d if R_d is not None and R_d > 0 else 8.0
    return np.clip(R_ring, 1.0, 30.0)  # Sanity bounds


def compute_kappa(B, Sigma0, S, 
                  kappa_min=0.3, kappa_max=1.0,
                  Sigma_ref=100.0, S0=0.8):
    """
    Coherence factor κ ∈ [kappa_min, kappa_max] for ring term.
    
    κ → 1 for clean, thin, HSB disks with low shear
    κ → kappa_min for LSB dwarfs with high shear
    
    Uses multiplicative gates:
    κ = kappa_min + (kappa_max - kappa_min) * f_Σ * f_S * (1 - B)
    """
    # Bulge suppression
    bulge_factor = (1.0 - np.clip(B, 0, 1)) ** 2.0
    
    # Compactness boost
    comp_gate = compactness_gate(Sigma0, Sigma_ref, gamma_Sigma=0.6)
    
    # Shear suppression  
    shear_g = shear_gate(S, S0, n_s=2.0)
    
    # Combined
    kappa = kappa_min + (kappa_max - kappa_min) * bulge_factor * comp_gate * shear_g
    
    return np.clip(kappa, kappa_min, kappa_max)


def eval_all_laws_v2p1(B, theta, Sigma0=None, R_d=None, shear=None):
    """
    Evaluate enhanced V2.1 laws with two-predictor lambda and ring concentration.
    
    Args:
        B: Bulge-to-total fraction [0, 1]
        theta: Dictionary with law parameters and scaling coefficients
        Sigma0: Disk central surface density (M_sun/pc^2)
        R_d: Disk scale length (kpc)
        shear: Shear S = -d(ln Ω)/d(ln R) at ~2.2 R_d
    
    Returns:
        Parameter dictionary for many-path model
    """
    def extract_law_params(law_dict):
        return law_dict['lo'], law_dict['hi'], law_dict['gamma']
    
    # Base B/T laws (for eta, ring_amp, M_max only—lambda uses two-predictor law)
    eta_base = float(law_value(B, *extract_law_params(theta["eta"])))
    ring_amp_base = float(law_value(B, *extract_law_params(theta["ring_amp"])))
    M_max_base = float(law_value(B, *extract_law_params(theta["M_max"])))
    
    # --- NEW: Two-predictor lambda law ---
    lambda_params = theta["lambda_ring"]
    lambda_ring = float(lambda_two_predictor(
        B, shear,
        lambda_min=lambda_params['lo'],
        lambda_max=lambda_params['hi'],
        gamma_bulge=lambda_params.get('gamma_bulge', 4.0),
        gamma_shear=lambda_params.get('gamma_shear', 3.0),
        S0=theta.get('S0_lambda', 0.8)
    ))
    
    # --- Compactness gating for eta and M_max ---
    if Sigma0 is not None and 'Sigma_ref' in theta:
        comp_gate = compactness_gate(Sigma0,
                                      Sigma_ref=theta.get('Sigma_ref', 100.0),
                                      gamma_Sigma=theta.get('gamma_Sigma', 0.8))
        # Apply with baseline
        eta_min_frac = theta.get('eta_min_fraction', 0.3)
        eta = eta_base * (eta_min_frac + (1.0 - eta_min_frac) * comp_gate)
        
        Mmax_min_frac = theta.get('Mmax_min_fraction', 0.5)
        M_max = M_max_base * (Mmax_min_frac + (1.0 - Mmax_min_frac) * comp_gate)
    else:
        eta = eta_base
        M_max = M_max_base
    
    # --- Shear gating for ring_amp (amplitude only, not lambda anymore) ---
    if shear is not None and 'S0' in theta:
        shear_g = shear_gate(shear, 
                             S0=theta.get('S0', 0.8),
                             n_s=theta.get('n_shear', 2.0))
        ring_amp_frac = theta.get('ring_min_fraction', 0.2)
        ring_amp = ring_amp_base * (ring_amp_frac + (1.0 - ring_amp_frac) * shear_g)
    else:
        ring_amp = ring_amp_base
    
    # --- Coherence factor kappa ---
    kappa = compute_kappa(B, Sigma0, shear,
                          kappa_min=theta.get('kappa_min', 0.3),
                          kappa_max=theta.get('kappa_max', 1.0),
                          Sigma_ref=theta.get('Sigma_ref', 100.0),
                          S0=theta.get('S0', 0.8))
    
    # --- Size scaling ---
    R_d_eff = R_d if R_d is not None and not np.isnan(R_d) else 2.5
    
    a0 = theta.get('a0', 2.0)
    a1 = theta.get('a1', 28.0)
    
    R0_scaled = a0 * R_d_eff
    R1_scaled = a1 * R_d_eff
    
    # --- NEW: Ring concentration radius ---
    R_ring = compute_R_ring(B, R_d_eff,
                           b0=theta.get('b0_ring', 2.5),
                           b1=theta.get('b1_ring', 1.2))
    
    sigma_ring = theta.get('sigma_ring', 0.5)
    
    pars = {
        "eta": eta,
        "ring_amp": ring_amp,
        "M_max": M_max,
        "lambda_ring": lambda_ring,
        "kappa": kappa,
        # Size-scaled radial parameters
        "R0": R0_scaled,
        "R1": R1_scaled,
        # NEW: Ring concentration
        "R_ring": R_ring,
        "sigma_ring": sigma_ring,
        # Fixed shape parameters
        "q": 3.5, "p": 2.0, "k_an": 1.4,
    }
    
    return pars


def default_theta_v2p1():
    """
    Default V2.1 theta with enhanced lambda law and ring concentration.
    """
    return {
        # Base B/T laws
        "eta":          {"lo": 0.01, "hi": 1.0,  "gamma": 4.0},
        "ring_amp":     {"lo": 0.37, "hi": 3.0,  "gamma": 4.0},
        "M_max":        {"lo": 1.2,  "hi": 3.0,  "gamma": 4.0},
        
        # Two-predictor lambda law
        "lambda_ring":  {
            "lo": 8.0,           # Shorter minimum (tighter for bulge-dominated)
            "hi": 22.0,          # Shorter maximum (prevents runaway)
            "gamma_bulge": 4.0,  # Bulge suppression steepness
            "gamma_shear": 3.0,  # Shear suppression steepness
        },
        
        # Compactness gates
        "Sigma_ref": 100.0,
        "gamma_Sigma": 0.8,
        "eta_min_fraction": 0.3,
        "Mmax_min_fraction": 0.5,
        
        # Shear gates
        "S0": 0.8,                 # Shear gate for ring_amp
        "S0_lambda": 0.8,          # Shear gate for lambda (can be different)
        "n_shear": 2.0,
        "ring_min_fraction": 0.2,
        
        # Coherence factor
        "kappa_min": 0.3,
        "kappa_max": 1.0,
        
        # Size scaling
        "a0": 2.0,   # R0 = a0 * R_d
        "a1": 28.0,  # R1 = a1 * R_d
        
        # Ring concentration (NEW)
        "b0_ring": 2.5,      # R_ring at B=0
        "b1_ring": 1.2,      # R_ring slope vs B
        "sigma_ring": 0.5,   # Ring width (in units of R_ring)
    }


def save_theta(path, theta_dict):
    with open(path, "w") as f:
        json.dump(theta_dict, f, indent=2)


def load_theta(path):
    with open(path, "r") as f:
        return json.load(f)
