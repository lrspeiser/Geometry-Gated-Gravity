#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended B/T Laws v2: Multi-Predictor Scaling
==============================================

Incorporates:
1. B/T (bulge-to-total) - sphericity/inner structure
2. Sigma0 (surface density) - compactness gating for eta, M_max
3. Shear S - coherence control for ring_amp, lambda_ring
4. Coherence factor kappa - decoherence of ring winding

This version uses monotone, bounded functions for all scalings.
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
    bulge_factor = (1.0 - B) ** 2.0
    
    # Compactness boost
    comp_gate = compactness_gate(Sigma0, Sigma_ref, gamma_Sigma=0.6)
    
    # Shear suppression  
    shear_g = shear_gate(S, S0, n_s=2.0)
    
    # Combined
    kappa = kappa_min + (kappa_max - kappa_min) * bulge_factor * comp_gate * shear_g
    
    return np.clip(kappa, kappa_min, kappa_max)


def eval_all_laws_v2(B, theta, Sigma0=None, R_d=None, shear=None, compactness=None):
    """
    Evaluate extended B/T laws with multi-predictor gating.
    
    Args:
        B: Bulge-to-total fraction [0, 1]
        theta: Dictionary with law parameters and scaling coefficients
        Sigma0: Disk central surface density (M_sun/pc^2)
        R_d: Disk scale length (kpc)
        shear: Shear S = -d(ln Ω)/d(ln R) at ~2.2 R_d
        compactness: Dimensionless compactness proxy
    
    Returns:
        Parameter dictionary for many-path model
    """
    def extract_law_params(law_dict):
        return law_dict['lo'], law_dict['hi'], law_dict['gamma']
    
    # Base B/T laws
    eta_base = float(law_value(B, *extract_law_params(theta["eta"])))
    ring_amp_base = float(law_value(B, *extract_law_params(theta["ring_amp"])))
    M_max_base = float(law_value(B, *extract_law_params(theta["M_max"])))
    lambda_base = float(law_value(B, *extract_law_params(theta["lambda_ring"])))
    
    # --- Compactness gating for eta and M_max ---
    if Sigma0 is not None and 'Sigma_ref' in theta:
        comp_gate = compactness_gate(Sigma0,
                                      Sigma_ref=theta.get('Sigma_ref', 100.0),
                                      gamma_Sigma=theta.get('gamma_Sigma', 0.8))
        # Apply with baseline: eta_min at low Σ, eta_base at high Σ
        eta_min_frac = theta.get('eta_min_fraction', 0.3)  # Keep 30% even for LSB
        eta = eta_base * (eta_min_frac + (1.0 - eta_min_frac) * comp_gate)
        
        # M_max also scales with compactness (less saturation cap for LSB)
        Mmax_min_frac = theta.get('Mmax_min_fraction', 0.5)
        M_max = M_max_base * (Mmax_min_frac + (1.0 - Mmax_min_frac) * comp_gate)
    else:
        eta = eta_base
        M_max = M_max_base
    
    # --- Shear gating for ring_amp and lambda_ring ---
    if shear is not None and 'S0' in theta:
        shear_g = shear_gate(shear, 
                             S0=theta.get('S0', 0.8),
                             n_s=theta.get('n_shear', 2.0))
        # Reduce ring coherence for high shear
        ring_amp_frac = theta.get('ring_min_fraction', 0.2)
        ring_amp = ring_amp_base * (ring_amp_frac + (1.0 - ring_amp_frac) * shear_g)
        
        # Lambda also shrinks with high shear
        lambda_ring = lambda_base * (0.5 + 0.5 * shear_g)
    else:
        ring_amp = ring_amp_base
        lambda_ring = lambda_base
    
    # --- Coherence factor kappa ---
    kappa = compute_kappa(B, Sigma0, shear,
                          kappa_min=theta.get('kappa_min', 0.3),
                          kappa_max=theta.get('kappa_max', 1.0),
                          Sigma_ref=theta.get('Sigma_ref', 100.0),
                          S0=theta.get('S0', 0.8))
    
    # --- Size scaling (simple, MW-normalized) ---
    R_d_eff = R_d if R_d is not None and not np.isnan(R_d) else 2.5
    R_d_norm = R_d_eff / 2.5  # Normalize to MW
    
    # Scale radial parameters (use conservative MW-like defaults)
    a0 = theta.get('a0', 2.0)
    a1 = theta.get('a1', 28.0)
    
    R0_scaled = a0 * R_d_eff
    R1_scaled = a1 * R_d_eff
    
    pars = {
        "eta": eta,
        "ring_amp": ring_amp,
        "M_max": M_max,
        "lambda_ring": lambda_ring,
        "kappa": kappa,  # NEW: coherence factor for ring term
        # Size-scaled radial parameters
        "R0": R0_scaled,
        "R1": R1_scaled,
        # Fixed shape parameters
        "q": 3.5, "p": 2.0, "k_an": 1.4,
    }
    
    return pars


def default_theta_v2():
    """
    Default extended theta with gate parameters.
    """
    return {
        # Base B/T laws (same as before)
        "eta":          {"lo": 0.01, "hi": 1.0,  "gamma": 4.0},
        "ring_amp":     {"lo": 0.37, "hi": 3.0,  "gamma": 4.0},
        "M_max":        {"lo": 1.2,  "hi": 3.0,  "gamma": 4.0},
        "lambda_ring":  {"lo": 10.0, "hi": 25.0, "gamma": 3.0},
        
        # Compactness gates
        "Sigma_ref": 100.0,        # Reference surface density (M_sun/pc^2)
        "gamma_Sigma": 0.8,        # Steepness of Sigma gating
        "eta_min_fraction": 0.3,   # Minimum eta as fraction of base (for LSB)
        "Mmax_min_fraction": 0.5,  # Minimum M_max as fraction of base
        
        # Shear gates
        "S0": 0.8,                 # Reference shear
        "n_shear": 2.0,            # Steepness of shear gating
        "ring_min_fraction": 0.2,  # Minimum ring_amp as fraction of base
        
        # Coherence factor
        "kappa_min": 0.3,          # Minimum coherence (decorrelated)
        "kappa_max": 1.0,          # Maximum coherence (perfect winding)
        
        # Size scaling
        "a0": 2.0,   # R0 = a0 * R_d
        "a1": 28.0,  # R1 = a1 * R_d
    }


def save_theta(path, theta_dict):
    with open(path, "w") as f:
        json.dump(theta_dict, f, indent=2)


def load_theta(path):
    with open(path, "r") as f:
        return json.load(f)
