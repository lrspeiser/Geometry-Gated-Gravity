
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to learn and apply smooth B/T (bulge-to-total) laws for
the many-path gravity parameters.
"""
import json
import numpy as np

# --- Morphology -> B/T mapping (rough priors; can be refined with SPARC table) ---
MORPH_TO_BT = {
    # very late / bulgeless
    "Im": 0.00, "IBm": 0.00, "Sm": 0.03, "Irr": 0.00,
    # late spirals
    "Sd": 0.06, "Scd": 0.08, "Sc": 0.15,
    # intermediate
    "Sbc": 0.30, "Sb": 0.40,
    # early with large bulges
    "Sab": 0.50, "Sa": 0.60, "S0": 0.70,
}

def morph_to_bt(hubble_type: str, fallback_group: str = None) -> float:
    """
    Map a Hubble type string to an approximate B/T in [0, 0.7].
    If unknown, use the type_group ('late'->0.08, 'intermediate'->0.25, 'early'->0.50).
    """
    if hubble_type in MORPH_TO_BT:
        return MORPH_TO_BT[hubble_type]
    if fallback_group is not None:
        m = fallback_group.lower()
        if m == "late": return 0.08
        if m == "intermediate": return 0.25
        if m == "early": return 0.50
    return 0.25  # neutral default

# --- Law family and helpers ----------------------------------------------------
def law_value(B, lo, hi, gamma):
    """
    Monotonic law: y(B) = lo + (hi - lo) * (1 - B)**gamma
    B: bulge-to-total in [0, 1]
    gamma: > 0; larger gamma steepens the drop toward bulge-dominated systems.
    """
    B = np.clip(B, 0.0, 1.0)
    gamma = np.maximum(gamma, 1e-3)
    return lo + (hi - lo) * (1.0 - B) ** gamma

def compactness_gate(Sigma0, Sigma_ref=100.0, alpha=0.8):
    """
    Smooth compactness gating function: saturates at high Σ, suppresses at low Σ.
    
    Args:
        Sigma0: Central surface density (M_sun/pc^2)
        Sigma_ref: Reference surface density (M_sun/pc^2)
        alpha: Steepness of transition
    
    Returns:
        Gating factor in [0, 1]
    """
    if Sigma0 is None or np.isnan(Sigma0) or Sigma0 <= 0:
        return 0.5  # Neutral default
    
    x = (Sigma0 / Sigma_ref) ** alpha
    return x / (1.0 + x)


def eval_all_laws(B, theta, Sigma0=None, R_d=None):
    """
    Evaluate all parameter laws at bulge fraction B with size and compactness scaling.
    
    Args:
        B: Bulge-to-total fraction [0, 1]
        theta: Dictionary with parameter laws and scaling coefficients
        Sigma0: Optional disk central surface density (M_sun/pc^2) for compactness scaling
        R_d: Optional disk scale length (kpc) for size scaling
    
    Returns:
        Parameter dictionary ready for many-path model
    """
    def extract_law_params(law_dict):
        """Extract only lo, hi, gamma from law dict (ignoring loss)"""
        return law_dict['lo'], law_dict['hi'], law_dict['gamma']
    
    # Base B/T laws
    eta_base = float(law_value(B, *extract_law_params(theta["eta"])))
    ring_amp_base = float(law_value(B, *extract_law_params(theta["ring_amp"])))
    M_max = float(law_value(B, *extract_law_params(theta["M_max"])))
    lambda_base = float(law_value(B, *extract_law_params(theta["lambda_ring"])))
    
    # Size scaling for radial parameters (use MW defaults if R_d not available)
    R_d_eff = R_d if R_d is not None and not np.isnan(R_d) else 2.5  # MW-like default
    
    # Radial scale coefficients (default to MW-centric if not in theta)
    a0 = theta.get('a0', 2.0)  # R0 = a0 * R_d
    a1 = theta.get('a1', 28.0)  # R1 = a1 * R_d  
    a4 = theta.get('a4', 8.0)  # lambda_ring = a4 * R_d (additional scaling)
    
    R0_scaled = a0 * R_d_eff
    R1_scaled = a1 * R_d_eff
    lambda_ring_scaled = lambda_base * (1.0 + theta.get('lambda_Rd_slope', 0.0) * (R_d_eff / 2.5 - 1.0))
    
    # Compactness gating for amplitudes
    if Sigma0 is not None and 'Sigma_ref' in theta:
        comp_gate = compactness_gate(Sigma0, 
                                      Sigma_ref=theta.get('Sigma_ref', 100.0),
                                      alpha=theta.get('Sigma_alpha', 0.8))
        # Apply compactness gate to disk-dominated terms
        eta = eta_base * (0.3 + 0.7 * comp_gate)  # Keep some baseline even for LSB
        ring_amp = ring_amp_base * comp_gate
    else:
        eta = eta_base
        ring_amp = ring_amp_base
    
    pars = {
        "eta": eta,
        "ring_amp": ring_amp,
        "M_max": M_max,
        "lambda_ring": lambda_ring_scaled,
        # Size-scaled radial parameters
        "R0": R0_scaled,
        "R1": R1_scaled,
        # Fixed shape parameters
        "q": 3.5, "p": 2.0, "k_an": 1.4,
    }
    return pars

def default_theta():
    """
    Safe priors for bounds based on your clustering ranges.
    These can be used to initialize fitting.
    """
    return {
        "eta":          {"lo": 0.02, "hi": 1.8,  "gamma": 1.2},
        "ring_amp":     {"lo": 0.00, "hi": 12.0, "gamma": 1.5},
        "M_max":        {"lo": 0.8,  "hi": 5.0,  "gamma": 1.2},
        "lambda_ring":  {"lo": 6.0,  "hi": 50.0, "gamma": 1.0},
    }

def fit_one_law(B, y, lo_bounds, hi_bounds, gamma_bounds=(0.3, 4.0), n_trials=5000, seed=7, weights=None):
    """
    Fit a single law y(B) = lo + (hi-lo)*(1-B)**gamma by random search
    with robust (Huber-like) loss.  No SciPy dependency.
    """
    rng = np.random.default_rng(seed)
    lo_lo, lo_hi = lo_bounds
    hi_lo, hi_hi = hi_bounds
    g_lo,  g_hi  = gamma_bounds

    if weights is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)

    def huber(resid, delta=1.0):
        a = np.abs(resid)
        return np.where(a <= delta, 0.5*a*a, delta*(a - 0.5*delta))

    best = None
    # Global random sampling
    for _ in range(n_trials):
        lo = rng.uniform(lo_lo, lo_hi)
        hi = rng.uniform(hi_lo, hi_hi)
        # enforce hi >= lo
        if hi < lo:
            lo, hi = hi, lo
        gamma = rng.uniform(g_lo, g_hi)
        yhat = law_value(B, lo, hi, gamma)
        loss = np.sum(w * huber(y - yhat))
        if (best is None) or (loss < best[0]):
            best = (loss, lo, hi, gamma)

    # Local refinement around the best
    loss, lo, hi, gamma = best
    scales = np.array([0.2*(lo_hi-lo_lo), 0.2*(hi_hi-hi_lo), 0.2*(g_hi-g_lo)])
    for _ in range(1000):
        cand = np.array([lo, hi, gamma]) + rng.normal(scale=scales)
        lo_c, hi_c, g_c = cand
        # clamp to bounds
        lo_c = np.clip(lo_c, lo_lo, lo_hi)
        hi_c = np.clip(hi_c, hi_lo, hi_hi)
        if hi_c < lo_c:
            lo_c, hi_c = hi_c, lo_c
        g_c  = np.clip(g_c, g_lo, g_hi)
        yhat = law_value(B, lo_c, hi_c, g_c)
        cand_loss = np.sum(w * huber(y - yhat))
        if cand_loss < loss:
            loss, lo, hi, gamma = cand_loss, lo_c, hi_c, g_c

    return {"lo": float(lo), "hi": float(hi), "gamma": float(gamma), "loss": float(loss)}

def save_theta(path, theta_dict):
    with open(path, "w") as f:
        json.dump(theta_dict, f, indent=2)

def load_theta(path):
    with open(path, "r") as f:
        return json.load(f)
