#!/usr/bin/env python3
"""
B/T Laws V2.3 - Radius-Dependent Suppression
============================================

Extends V2.2 with:
1. Radius-dependent bar/shear tapers (addresses Sc/Sbc overshoot beyond bars)
2. Shear-coupled sigma_ring (narrows ring envelope in high-shear disks)
3. Dynamic M_max(S, B/T) (prevents red roofs in high-shear/bulgy systems)

Mathematical Form:
    ring_term = ring_amp * coherence * envelope * bulge_gate * bar_taper * shear_taper
    
Where:
    bar_taper(R) = [1 + tanh((R_bar - R)/w_bar)]^gamma_bar  (SB/SAB only)
    shear_taper(R) = [1 + tanh((R_shear - R)/w_sh)]^gamma_sh  (high-shear only)
    
Key Insight:
    Bars and shear dephase long azimuthal paths *beyond* specific radii.
    Global suppression (g_bar) says "how much", radial tapers say "where".
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def load_theta(params_file: Path) -> dict:
    """Load V2.3 B/T law parameters from JSON."""
    with open(params_file, 'r') as f:
        return json.load(f)


def logistic_gate(x: float, x_ref: float, gamma: float) -> float:
    """
    Smooth logistic suppression gate.
    
    Args:
        x: Input variable (e.g. B/T, Sigma0)
        x_ref: Reference value
        gamma: Steepness
    
    Returns:
        Gate value in [0, 1]
    """
    return 1.0 / (1.0 + (x / x_ref)**gamma)


def shear_gate(shear: float, S0: float, n: float = 2.0) -> float:
    """
    Smooth shear suppression gate.
    
    Args:
        shear: Shear value
        S0: Reference shear
        n: Power law exponent
    
    Returns:
        Gate value g_shear in [0, 1]
    """
    if shear is None:
        return 1.0
    ratio = max(shear / S0, 0.0)
    return 1.0 / (1.0 + ratio**n)


def eval_eta_law(B_T: float, theta: dict, Sigma0: Optional[float] = None) -> float:
    """
    Evaluate eta law with optional Sigma0 modulation.
    
    eta = eta_lo + (eta_hi - eta_lo) * (1 - B/T)^gamma
    
    With optional Sigma0 gate to reduce eta in LSBs.
    """
    eta_lo = theta['eta']['lo']
    eta_hi = theta['eta']['hi']
    gamma = theta['eta']['gamma']
    
    # Base B/T law
    eta = eta_lo + (eta_hi - eta_lo) * (1.0 - B_T)**gamma
    
    # Optional Sigma0 modulation (reduce eta in LSBs)
    if Sigma0 is not None and 'Sigma_ref' in theta:
        Sigma_ref = theta['Sigma_ref']
        gamma_Sigma = theta.get('gamma_Sigma', 0.8)
        eta_min_frac = theta.get('eta_min_fraction', 0.3)
        
        # Smooth gate: eta -> eta * [eta_min_frac + (1 - eta_min_frac) * gate]
        gate = logistic_gate(Sigma_ref, Sigma0, gamma_Sigma)
        eta = eta * (eta_min_frac + (1.0 - eta_min_frac) * gate)
    
    return eta


def eval_ring_amp_law(B_T: float, theta: dict) -> float:
    """
    Evaluate ring amplitude law.
    
    ring_amp = ring_lo + (ring_hi - ring_lo) * (1 - B/T)^gamma
    """
    ring_lo = theta['ring_amp']['lo']
    ring_hi = theta['ring_amp']['hi']
    gamma = theta['ring_amp']['gamma']
    
    return ring_lo + (ring_hi - ring_lo) * (1.0 - B_T)**gamma


def eval_Mmax_law_v2p3(B_T: float, theta: dict, shear: Optional[float] = None) -> float:
    """
    V2.3: Evaluate M_max law with dynamic shear and B/T dependence.
    
    M_max = M_min + (M_max_disk - M_min) * (1 - B/T)^gamma_B * (1 - g_shear(S))^gamma_S
    
    Rationale:
        - High B/T (bulgy systems): lower ceiling
        - High shear (chaotic disks): lower ceiling
        - Pure disks with low shear: allow highest M_max
    """
    M_min = theta['M_max']['lo']
    M_max_disk = theta['M_max']['hi']
    gamma_bulge = theta['M_max'].get('gamma_bulge', 4.0)
    gamma_shear_Mmax = theta['M_max'].get('gamma_shear', 2.0)
    
    # Base B/T scaling
    bulge_factor = (1.0 - B_T)**gamma_bulge
    
    # Shear scaling (new in V2.3)
    if shear is not None:
        S0 = theta.get('S0', 0.8)
        n_shear = theta.get('n_shear', 2.0)
        g_s = shear_gate(shear, S0, n_shear)
        shear_factor = (1.0 - g_s)**gamma_shear_Mmax
    else:
        shear_factor = 1.0
    
    M_max = M_min + (M_max_disk - M_min) * bulge_factor * shear_factor
    
    return M_max


def eval_lambda_law_v2p3(B_T: float, theta: dict, 
                         Sigma0: Optional[float] = None,
                         R_d: Optional[float] = None,
                         shear: Optional[float] = None) -> float:
    """
    V2.3: Two-predictor lambda law with shear and bulge dependence.
    
    lambda = lambda_lo + (lambda_hi - lambda_lo) * f(B/T, shear)
    
    Where:
        f(B/T, shear) = [(1-B/T)^gamma_b] * [(1 - g_shear(S))^gamma_s]
    """
    lambda_lo = theta['lambda_ring']['lo']
    lambda_hi = theta['lambda_ring']['hi']
    gamma_bulge = theta['lambda_ring'].get('gamma_bulge', 4.0)
    gamma_shear = theta['lambda_ring'].get('gamma_shear', 3.0)
    
    # Bulge suppression (shorter lambda in early types)
    bulge_factor = (1.0 - B_T)**gamma_bulge
    
    # Shear suppression (shorter lambda in high-shear disks)
    if shear is not None:
        S0_lambda = theta.get('S0_lambda', 0.8)
        n_shear = theta.get('n_shear', 2.0)
        g_s = shear_gate(shear, S0_lambda, n_shear)
        shear_factor = (1.0 - g_s)**gamma_shear
    else:
        shear_factor = 1.0
    
    combined_factor = bulge_factor * shear_factor
    lambda_ring = lambda_lo + (lambda_hi - lambda_lo) * combined_factor
    
    return lambda_ring


def eval_sigma_ring_v2p3(theta: dict, R_d: Optional[float] = None,
                         shear: Optional[float] = None) -> float:
    """
    V2.3: Shear-coupled sigma_ring.
    
    sigma_ring = sigma_min + (sigma_max - sigma_min) * (1 - g_shear(S))^gamma_sigma
    
    Rationale:
        High-shear disks get BOTH shorter lambda AND narrower envelope.
        This prevents broad overshoot in Sc/Sbc systems.
    """
    sigma_min = theta.get('sigma_ring_min', 0.3)
    sigma_max = theta.get('sigma_ring_max', 0.8)
    gamma_sigma = theta.get('gamma_sigma_shear', 1.0)
    
    if shear is not None:
        S0 = theta.get('S0', 0.8)
        n_shear = theta.get('n_shear', 2.0)
        g_s = shear_gate(shear, S0, n_shear)
        shear_factor = (1.0 - g_s)**gamma_sigma
    else:
        shear_factor = 1.0
    
    sigma = sigma_min + (sigma_max - sigma_min) * shear_factor
    
    return sigma


def ring_radial_envelope(r: np.ndarray, R_ring: float, sigma_ring: float) -> np.ndarray:
    """
    Gaussian radial envelope centered at R_ring with width sigma_ring.
    
    envelope(R) = exp[-(R - R_ring)^2 / (2 * sigma_ring^2)]
    """
    return np.exp(-0.5 * ((r - R_ring) / sigma_ring)**2)


def bar_radial_taper(r: np.ndarray, R_bar: float, w_bar: float, 
                     gamma_bar: float) -> np.ndarray:
    """
    V2.3: Radius-dependent bar taper.
    
    taper(R) = [1 + tanh((R_bar - R) / w_bar)]^gamma_bar
    
    Behavior:
        - R << R_bar: taper ≈ 2^gamma_bar (allow ring inside bar)
        - R >> R_bar: taper → 0 (suppress ring outside bar)
        
    Applied only to SB/SAB galaxies.
    """
    z = (R_bar - r) / w_bar
    taper = (1.0 + np.tanh(z))**gamma_bar
    return taper


def shear_radial_taper(r: np.ndarray, R_shear: float, w_shear: float,
                       gamma_shear: float) -> np.ndarray:
    """
    V2.3: Radius-dependent shear taper.
    
    taper(R) = [1 + tanh((R_shear - R) / w_shear)]^gamma_shear
    
    Behavior:
        - R << R_shear: taper ≈ 2^gamma_shear (allow ring in spiral region)
        - R >> R_shear: taper → 0 (suppress ring beyond spiral dominance)
        
    Applied to high-shear galaxies.
    """
    z = (R_shear - r) / w_shear
    taper = (1.0 + np.tanh(z))**gamma_shear
    return taper


def eval_all_laws_v2p3(B_T: float, theta: dict,
                       Sigma0: Optional[float] = None,
                       R_d: Optional[float] = None,
                       shear: Optional[float] = None,
                       g_bar_in: Optional[float] = None,
                       bar_class: Optional[str] = None) -> dict:
    """
    Evaluate all V2.3 B/T laws for a given galaxy.
    
    Args:
        B_T: Bulge-to-total ratio
        theta: Parameter dictionary
        Sigma0: Central surface density (M_sun/pc^2)
        R_d: Disk scale length (kpc)
        shear: Shear at 2.2 R_d
        g_bar_in: Bar gating factor (optional, can use bar_class instead)
        bar_class: Bar classification (SA/SAB/SB/S)
    
    Returns:
        Dictionary of model parameters
    """
    # Evaluate base laws
    eta = eval_eta_law(B_T, theta, Sigma0)
    ring_amp = eval_ring_amp_law(B_T, theta)
    M_max = eval_Mmax_law_v2p3(B_T, theta, shear)
    lambda_ring = eval_lambda_law_v2p3(B_T, theta, Sigma0, R_d, shear)
    
    # Ring concentration parameters
    a0 = theta.get('a0', 2.0)
    a1 = theta.get('a1', 28.0)
    b0_ring = theta.get('b0_ring', 2.5)
    b1_ring = theta.get('b1_ring', 1.2)
    
    R_ring = a0 + a1 * (1.0 - B_T)
    
    # V2.3: Shear-coupled sigma_ring
    sigma_ring = eval_sigma_ring_v2p3(theta, R_d, shear)
    
    # Coherence factor kappa
    kappa_min = theta.get('kappa_min', 0.3)
    kappa_max = theta.get('kappa_max', 1.0)
    kappa_span = kappa_max - kappa_min
    kappa = kappa_min + kappa_span * (1.0 - B_T)**2
    
    # Bar gating (global suppression)
    if g_bar_in is not None:
        g_bar = g_bar_in
    elif bar_class is not None:
        # Default bar gates
        bar_gates = {
            'SA': 1.0,      # No suppression
            'SAB': 0.65,    # Moderate suppression
            'SB': 0.45,     # Strong suppression
            'S': 0.73       # Default (unknown/no bar)
        }
        g_bar = bar_gates.get(bar_class, 0.73)
    else:
        g_bar = 1.0  # No bar information
    
    # V2.3: Radius-dependent bar/shear taper parameters
    # These are applied in the forward model, not here
    use_bar_taper = theta.get('use_bar_taper', True) and bar_class in ['SAB', 'SB']
    use_shear_taper = theta.get('use_shear_taper', True) and shear is not None
    
    # V2.3b: Differentiated bar taper parameters for SAB vs SB
    if use_bar_taper and R_d is not None:
        # Check for differentiated parameters (V2.3b)
        if bar_class == 'SAB' and 'R_bar_factor_SAB' in theta:
            # Weak bar: moderate suppression at larger radius
            R_bar_factor = theta.get('R_bar_factor_SAB', 2.0)
            w_bar_factor = theta.get('w_bar_factor_SAB', 0.3)
            gamma_bar_taper = theta.get('gamma_bar_taper_SAB', 1.5)
        elif bar_class == 'SB' and 'R_bar_factor_SB' in theta:
            # Strong bar: aggressive suppression at corotation (~1.5 R_d)
            R_bar_factor = theta.get('R_bar_factor_SB', 1.5)
            w_bar_factor = theta.get('w_bar_factor_SB', 0.2)
            gamma_bar_taper = theta.get('gamma_bar_taper_SB', 2.5)
        else:
            # Fallback to V2.3 unified parameters
            R_bar_factor = theta.get('R_bar_factor', 2.0)
            w_bar_factor = theta.get('w_bar_factor', 0.3)
            gamma_bar_taper = theta.get('gamma_bar_taper', 1.5)
        
        R_bar = R_bar_factor * R_d
        w_bar = w_bar_factor * R_d
    else:
        R_bar = None
        w_bar = None
        gamma_bar_taper = 0.0
    
    # Shear taper parameters (applied to high-shear galaxies)
    if use_shear_taper and R_d is not None and shear is not None:
        shear_threshold = theta.get('shear_threshold', 0.5)
        
        if shear > shear_threshold:
            R_shear_factor = theta.get('R_shear_factor', 2.5)
            w_shear_factor = theta.get('w_shear_factor', 0.5)
            gamma_shear_taper = theta.get('gamma_shear_taper', 1.0)
            
            R_shear = R_shear_factor * R_d
            w_shear = w_shear_factor * R_d
        else:
            R_shear = None
            w_shear = None
            gamma_shear_taper = 0.0
    else:
        R_shear = None
        w_shear = None
        gamma_shear_taper = 0.0
    
    return {
        'eta': eta,
        'ring_amp': ring_amp * g_bar,  # Global bar suppression
        'M_max': M_max,
        'lambda_ring': lambda_ring,
        'kappa': kappa,
        'R_ring': R_ring,
        'sigma_ring': sigma_ring,  # Now shear-dependent
        'g_bar': g_bar,
        'bulge_gate_power': theta.get('bulge_gate_power', 2.0),
        # V2.3: Radial taper parameters
        'R_bar': R_bar,
        'w_bar': w_bar,
        'gamma_bar_taper': gamma_bar_taper,
        'R_shear': R_shear,
        'w_shear': w_shear,
        'gamma_shear_taper': gamma_shear_taper,
        # Fixed parameters
        'R0': theta.get('R0', 5.0),
        'R1': theta.get('R1', 70.0),
        'p': theta.get('p', 2.0),
        'q': theta.get('q', 3.5)
    }


def main():
    """Demo: evaluate V2.3 laws for a few test cases."""
    
    # Load test parameters
    params_file = Path(__file__).parent / 'bt_law_params_v2p3_initial.json'
    
    if not params_file.exists():
        print(f"[!] Parameter file not found: {params_file}")
        print("    Run the initialization script first.")
        return
    
    theta = load_theta(params_file)
    
    print("="*80)
    print("V2.3 B/T LAW EVALUATION")
    print("  (With Radius-Dependent Bar/Shear Tapers)")
    print("="*80)
    
    # Test cases
    test_cases = [
        {'name': 'Early SA (E/S0)', 'B_T': 0.5, 'Sigma0': 1000, 'R_d': 3.0, 
         'shear': 0.6, 'bar_class': 'SA'},
        {'name': 'Intermediate SAB (Sab)', 'B_T': 0.2, 'Sigma0': 500, 'R_d': 4.0,
         'shear': 1.0, 'bar_class': 'SAB'},
        {'name': 'Intermediate SB (Sbc)', 'B_T': 0.1, 'Sigma0': 400, 'R_d': 5.0,
         'shear': 1.2, 'bar_class': 'SB'},
        {'name': 'Late Sc (high shear)', 'B_T': 0.05, 'Sigma0': 100, 'R_d': 4.0,
         'shear': 1.5, 'bar_class': 'S'},
        {'name': 'Late Sd (LSB)', 'B_T': 0.02, 'Sigma0': 30, 'R_d': 5.0,
         'shear': 0.8, 'bar_class': 'S'}
    ]
    
    print("\nTest Cases:")
    print("-"*80)
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"  B/T={case['B_T']:.2f}, Σ₀={case['Sigma0']:.0f} M☉/pc², "
              f"R_d={case['R_d']:.1f} kpc, S={case['shear']:.2f}, "
              f"bar={case['bar_class']}")
        
        params = eval_all_laws_v2p3(case['B_T'], theta,
                                     Sigma0=case['Sigma0'],
                                     R_d=case['R_d'],
                                     shear=case['shear'],
                                     bar_class=case['bar_class'])
        
        print(f"  → η={params['eta']:.3f}, A_ring={params['ring_amp']:.2f}, "
              f"M_max={params['M_max']:.2f}")
        print(f"  → λ={params['lambda_ring']:.1f} kpc, κ={params['kappa']:.2f}, "
              f"σ_ring={params['sigma_ring']:.2f} R_d")
        print(f"  → R_ring={params['R_ring']:.1f} kpc, g_bar={params['g_bar']:.2f}")
        
        if params['R_bar'] is not None:
            print(f"  → BAR TAPER: R_bar={params['R_bar']:.1f} kpc, "
                  f"w_bar={params['w_bar']:.2f} kpc, γ={params['gamma_bar_taper']:.2f}")
        
        if params['R_shear'] is not None:
            print(f"  → SHEAR TAPER: R_shear={params['R_shear']:.1f} kpc, "
                  f"w_shear={params['w_shear']:.2f} kpc, γ={params['gamma_shear_taper']:.2f}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
