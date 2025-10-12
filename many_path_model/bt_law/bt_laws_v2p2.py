#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended B/T Laws v2.2: Bar Gating Integration
==============================================

Adds bar suppression to V2.1 enhanced laws:
- V2.1 features: Two-predictor lambda λ(B/T, S), radial ring concentration
- V2.2 addition: Bar gating g_bar multiplies ring_amp and lambda

Predictors:
1. B/T (bulge-to-total) - sphericity/inner structure
2. Sigma0 (surface density) - compactness gating for eta, M_max
3. Shear S - coherence control AND lambda scaling
4. Bar strength g_bar - suppresses ring term in barred systems
5. Coherence factor kappa - decoherence of ring winding

Physical interpretation:
Bars introduce non-axisymmetric torques that destroy long, phase-coherent
azimuthal loops. Ring amplitude and lambda should be suppressed in SB/SAB systems.
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


def bar_gate(g_bar_input, use_gate=True):
    """
    Bar gating factor for ring suppression.
    
    Args:
        g_bar_input: Pre-computed bar gate value in [0, 1]
                     (from extract_bar_classification.py)
        use_gate: If False, return 1.0 (no suppression)
    
    Returns:
        float: Bar gating factor
    """
    if not use_gate:
        return 1.0
    
    if g_bar_input is None or np.isnan(g_bar_input):
        return 0.8  # Neutral default (slight suppression)
    
    return np.clip(g_bar_input, 0.4, 1.0)


def lambda_two_predictor(B, S, lambda_min, lambda_max, 
                         gamma_bulge=4.0, gamma_shear=3.0, S0=0.8):
    """
    Lambda depends on BOTH B/T and shear S.
    
    λ(B, S) = λ_min + (λ_max - λ_min) * (1 - B)^γ_b * (1 - g_shear(S))^γ_s
    
    Physical interpretation:
    - More disk (low B) → longer λ (more coherence)
    - Higher shear (S >> S0) → shorter λ (dephasing kills long loops)
    """
    # Bulge suppression (disk-dominated → longer λ)
    f_bulge = (1.0 - np.clip(B, 0, 1)) ** gamma_bulge
    
    # Shear suppression (high shear → shorter λ)
    if S is None or np.isnan(S):
        g_shear_val = 0.5
    else:
        x = np.maximum(abs(S) / S0, 1e-6)
        g_shear_val = (x**2) / (1.0 + x**2)
    
    f_shear = (1.0 - g_shear_val) ** gamma_shear
    
    lam = lambda_min + (lambda_max - lambda_min) * f_bulge * f_shear
    
    return np.clip(lam, lambda_min, lambda_max)


def ring_radial_envelope(R, R_ring, sigma_ring=0.5):
    """
    Radial concentration envelope for ring term.
    
    A_ring(R) = exp(-(R/R_ring)^2 / (2*sigma_ring^2))
    
    Concentrates ring boost near R_ring (typically 2-4 R_d for spirals).
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
    """
    factor = b0 - b1 * np.clip(B, 0, 1)
    R_ring = factor * R_d if R_d is not None and R_d > 0 else 8.0
    return np.clip(R_ring, 1.0, 30.0)


def compute_kappa(B, Sigma0, S, 
                  kappa_min=0.3, kappa_max=1.0,
                  Sigma_ref=100.0, S0=0.8):
    """
    Coherence factor κ ∈ [kappa_min, kappa_max] for ring term.
    
    κ → 1 for clean, thin, HSB disks with low shear
    κ → kappa_min for LSB dwarfs with high shear
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


def eval_all_laws_v2p2(B, theta, Sigma0=None, R_d=None, shear=None, g_bar_in=None):
    """
    Evaluate V2.2 laws with bar gating.
    
    Args:
        B: Bulge-to-total fraction [0, 1]
        theta: Dictionary with law parameters and scaling coefficients
        Sigma0: Disk central surface density (M_sun/pc^2)
        R_d: Disk scale length (kpc)
        shear: Shear S = -d(ln Ω)/d(ln R) at ~2.2 R_d
        g_bar_in: Pre-computed bar gate value from extract_bar_classification.py
    
    Returns:
        Parameter dictionary for many-path model
    """
    def extract_law_params(law_dict):
        return law_dict['lo'], law_dict['hi'], law_dict['gamma']
    
    # Base B/T laws (for eta, ring_amp, M_max only)
    eta_base = float(law_value(B, *extract_law_params(theta["eta"])))
    ring_amp_base = float(law_value(B, *extract_law_params(theta["ring_amp"])))
    M_max_base = float(law_value(B, *extract_law_params(theta["M_max"])))
    
    # --- Two-predictor lambda law (from V2.1) ---
    lambda_params = theta["lambda_ring"]
    lambda_base = float(lambda_two_predictor(
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
        eta_min_frac = theta.get('eta_min_fraction', 0.3)
        eta = eta_base * (eta_min_frac + (1.0 - eta_min_frac) * comp_gate)
        
        Mmax_min_frac = theta.get('Mmax_min_fraction', 0.5)
        M_max = M_max_base * (Mmax_min_frac + (1.0 - Mmax_min_frac) * comp_gate)
    else:
        eta = eta_base
        M_max = M_max_base
    
    # --- Shear gating for ring_amp ---
    if shear is not None and 'S0' in theta:
        shear_g = shear_gate(shear, 
                             S0=theta.get('S0', 0.8),
                             n_s=theta.get('n_shear', 2.0))
        ring_amp_frac = theta.get('ring_min_fraction', 0.2)
        ring_amp = ring_amp_base * (ring_amp_frac + (1.0 - ring_amp_frac) * shear_g)
    else:
        ring_amp = ring_amp_base
    
    # --- NEW: Bar gating (V2.2) ---
    g_bar = bar_gate(g_bar_in, use_gate=theta.get('use_bar_gate', True))
    
    # Apply bar suppression to ring amplitude and lambda
    ring_amp = ring_amp * g_bar
    lambda_ring = lambda_base * g_bar
    
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
    
    # --- Ring concentration radius ---
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
        "g_bar": g_bar,  # Store for diagnostics
        # Size-scaled radial parameters
        "R0": R0_scaled,
        "R1": R1_scaled,
        # Ring concentration
        "R_ring": R_ring,
        "sigma_ring": sigma_ring,
        # Fixed shape parameters
        "q": 3.5, "p": 2.0, "k_an": 1.4,
    }
    
    return pars


def default_theta_v2p2():
    """
    Default V2.2 theta with bar gating.
    """
    return {
        # Base B/T laws
        "eta":          {"lo": 0.01, "hi": 1.0,  "gamma": 4.0},
        "ring_amp":     {"lo": 0.37, "hi": 3.0,  "gamma": 4.0},
        "M_max":        {"lo": 1.2,  "hi": 3.0,  "gamma": 4.0},
        
        # Two-predictor lambda law
        "lambda_ring":  {
            "lo": 8.0,
            "hi": 22.0,
            "gamma_bulge": 4.0,
            "gamma_shear": 3.0,
        },
        
        # Compactness gates
        "Sigma_ref": 100.0,
        "gamma_Sigma": 0.8,
        "eta_min_fraction": 0.3,
        "Mmax_min_fraction": 0.5,
        
        # Shear gates
        "S0": 0.8,
        "S0_lambda": 0.8,
        "n_shear": 2.0,
        "ring_min_fraction": 0.2,
        
        # Coherence factor
        "kappa_min": 0.3,
        "kappa_max": 1.0,
        
        # Size scaling
        "a0": 2.0,
        "a1": 28.0,
        
        # Ring concentration
        "b0_ring": 2.5,
        "b1_ring": 1.2,
        "sigma_ring": 0.5,
        
        # Bar gating (NEW in V2.2)
        "use_bar_gate": True,  # Set to False to disable bar suppression
    }


def save_theta(path, theta_dict):
    with open(path, "w") as f:
        json.dump(theta_dict, f, indent=2)


def load_theta(path):
    with open(path, "r") as f:
        return json.load(f)
