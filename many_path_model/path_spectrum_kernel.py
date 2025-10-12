#!/usr/bin/env python3
"""
Path-Spectrum Kernel: Physics-Grounded Alternative to Empirical Tapers
======================================================================

Instead of stacking empirical gates (ring_amp × g_bar × bar_taper × shear_taper),
this module derives the ring enhancement from **stationary-phase path accumulation**
with a single coherence length that shrinks with bars, shear, and bulges.

Physical Motivation:
-------------------
Long azimuthal "loop" paths exist in disk galaxies and contribute additional
gravitational coupling beyond the direct radial path. These loops:
  - Form at galactic scales (R > 0.5 kpc)
  - Survive in coherent disk regions
  - Dephase due to bars, shear, and bulges

Mathematical Form:
-----------------
Conservative potential with path-family kernel:

    Φ(x) = -G ∫ ρ(x') × [1 + K(x, x')] / |x - x'| d³x'

where K is a sum over azimuthal loop families:

    K = Σ_m A_m(R) × exp(-L_m / ℓ_coh)

    m: azimuthal winding number (m=0: direct, m=±1,±2,...: loops)
    L_m: effective path length ~ m × R  
    A_m: stationary-phase weighting (decays with m)
    ℓ_coh: coherence length shrinks with bars/shear/bulge

Coherence Length:
----------------
    ℓ_coh = ℓ_0 × (1-B/T)^α × g_shear(S)^β × g_bar^γ

where:
    ℓ_0: base coherence length (fit parameter)
    α: bulge suppression exponent (fit)
    β: shear suppression exponent (fit)
    γ: bar suppression exponent (fit)

Key Advantages:
--------------
1. **Unifies** V2.2's ring_amp, lambda_ring, and bar/shear gates
2. **Reduces** parameter count from ~15 knobs to 4-5 universal hypers
3. **Interpretable**: coherence length has clear physical meaning
4. **Falsifiable**: predicts ℓ_coh vs. morphology relationship
5. **Conservative**: stays at potential level (∇ × a ≈ 0)

Connection to V2.2 Ring Term:
----------------------------
V2.2: ring_term = ring_amp × exp(-2πR/λ) / (1 - κ×exp(-2πR/λ))

This is approximately the m=1 term of our kernel with:
    - ring_amp ↔ A_1 × morphology gates
    - λ ↔ ℓ_coh / (2π)
    - κ ↔ coherence feedback

So the path-spectrum kernel is a **physical grounding** of the V2.2 ring term,
not a replacement of the underlying physics.

Implementation Strategy:
-----------------------
1. Start with m=1 only (single loop family)
2. Fit {ℓ_0, α, β, γ} on 80% stratified SPARC sample
3. Validate on 20% holdout
4. Compare to V2.2 baseline (median 23.07%)
5. Run ablation: drop m=1 term, verify Δχ² matches V2.2 ablation

Guard Rails:
-----------
- Keep hard saturation (ablation Δχ² = +292 proves it's essential)
- Use multi-objective loss (rotation χ² + outer slope + vertical lag)
- Maintain stratified train/test split
- Require monotone coherence length (no wiggles)
"""

import numpy as np
from typing import Tuple, Dict, Optional


def coherence_length(B_T: float, shear: Optional[float], bar_class: str,
                     l0: float, alpha: float, beta: float, gamma: float,
                     S0: float = 0.8) -> float:
    """
    Compute coherence length for azimuthal path families.
    
    Args:
        B_T: Bulge-to-total ratio
        shear: Shear at 2.2 R_d (optional)
        bar_class: Bar classification (SA/SAB/SB/S)
        l0: Base coherence length (kpc)
        alpha: Bulge suppression exponent
        beta: Shear suppression exponent
        gamma: Bar suppression exponent
        S0: Reference shear for normalization
    
    Returns:
        Coherence length in kpc
    """
    # Bulge suppression (more bulge → shorter coherence)
    bulge_factor = (1.0 - B_T)**alpha
    
    # Shear suppression (higher shear → shorter coherence)
    if shear is not None:
        S_norm = max(shear / S0, 0.0)
        g_shear = 1.0 / (1.0 + S_norm**2)  # Smooth decay
        shear_factor = g_shear**beta
    else:
        shear_factor = 1.0
    
    # Bar suppression (stronger bar → shorter coherence)
    bar_gates = {
        'SA': 1.0,      # No suppression
        'SAB': 0.75,    # Moderate suppression
        'SB': 0.55,     # Strong suppression
        'S': 0.85,      # Default (unknown/no bar)
        'unknown': 0.85
    }
    g_bar = bar_gates.get(bar_class, 0.85)
    bar_factor = g_bar**gamma
    
    # Combined coherence length
    l_coh = l0 * bulge_factor * shear_factor * bar_factor
    
    return l_coh


def path_kernel_m1(R: np.ndarray, z: np.ndarray, l_coh: float,
                    A1: float = 1.0, n: float = 1.0) -> np.ndarray:
    """
    Compute m=1 path-family kernel (single azimuthal loop).
    
    Args:
        R: Cylindrical radius (kpc)
        z: Height above midplane (kpc)
        l_coh: Coherence length (kpc)
        A1: Amplitude for m=1 family
        n: Decay exponent for vertical extent
    
    Returns:
        K_1(R, z) kernel values
    """
    # Effective path length for m=1 loop
    # Rough estimate: one loop around a circle of radius R
    L1 = 2.0 * np.pi * R
    
    # Exponential suppression when path exceeds coherence length
    coherence_factor = np.exp(-L1 / l_coh)
    
    # Vertical suppression (loops are more coherent near midplane)
    # Use scale height h ~ 0.3 kpc for thin disk
    h_disk = 0.3
    vertical_factor = np.exp(-(np.abs(z) / h_disk)**n)
    
    # Combined kernel
    K1 = A1 * coherence_factor * vertical_factor
    
    return K1


def path_kernel_full(R: np.ndarray, z: np.ndarray, l_coh: float,
                      max_m: int = 2, A_decay: float = 2.0) -> np.ndarray:
    """
    Compute full path-spectrum kernel with multiple winding families.
    
    Args:
        R: Cylindrical radius (kpc)
        z: Height above midplane (kpc)
        l_coh: Coherence length (kpc)
        max_m: Maximum winding number to include
        A_decay: Amplitude decay rate (A_m ∝ m^(-A_decay))
    
    Returns:
        K(R, z) = Σ_m K_m(R, z)
    """
    K_total = np.zeros_like(R)
    
    for m in range(1, max_m + 1):
        # Amplitude decays with winding number
        # (higher loops less likely to contribute coherently)
        A_m = 1.0 / (m**A_decay)
        
        # Path length scales with winding number
        L_m = 2.0 * np.pi * R * m
        
        # Coherence factor
        coherence_factor = np.exp(-L_m / l_coh)
        
        # Vertical suppression
        h_disk = 0.3
        vertical_factor = np.exp(-(np.abs(z) / h_disk)**2)
        
        K_m = A_m * coherence_factor * vertical_factor
        K_total += K_m
    
    return K_total


def eval_path_kernel_params(B_T: float, theta: Dict,
                             shear: Optional[float] = None,
                             bar_class: str = 'S',
                             R_d: Optional[float] = None) -> Dict:
    """
    Evaluate path-spectrum parameters for a given galaxy.
    
    Args:
        B_T: Bulge-to-total ratio
        theta: Parameter dictionary with {l0, alpha, beta, gamma, ...}
        shear: Shear at 2.2 R_d
        bar_class: Bar classification
        R_d: Disk scale length (for ring concentration)
    
    Returns:
        Dictionary of parameters for forward model
    """
    # Coherence length (replaces ring_amp, lambda_ring, and gates)
    l0 = theta.get('l0', 20.0)
    alpha = theta.get('alpha', 2.0)
    beta = theta.get('beta', 1.5)
    gamma = theta.get('gamma', 1.0)
    
    l_coh = coherence_length(B_T, shear, bar_class, l0, alpha, beta, gamma)
    
    # Base eta (still needed for baseline coupling)
    eta_lo = theta.get('eta_lo', 0.01)
    eta_hi = theta.get('eta_hi', 1.0)
    eta_gamma = theta.get('eta_gamma', 4.0)
    eta = eta_lo + (eta_hi - eta_lo) * (1.0 - B_T)**eta_gamma
    
    # M_max (hard saturation - essential per ablation)
    M_max_lo = theta.get('M_max_lo', 1.2)
    M_max_hi = theta.get('M_max_hi', 3.0)
    M_max_gamma = theta.get('M_max_gamma', 4.0)
    M_max = M_max_lo + (M_max_hi - M_max_lo) * (1.0 - B_T)**M_max_gamma
    
    # Ring concentration (radial placement)
    # Use V2.1 formula: R_ring = a0 + a1 * (1 - B/T)
    a0 = theta.get('a0', 2.0)
    a1 = theta.get('a1', 28.0)
    R_ring = a0 + a1 * (1.0 - B_T)
    
    # Ring width (could be constant or vary with morphology)
    sigma_ring = theta.get('sigma_ring', 0.5)
    
    # Amplitude for m=1 family
    A1 = theta.get('A1', 1.0)
    
    return {
        'eta': eta,
        'M_max': M_max,
        'l_coh': l_coh,
        'R_ring': R_ring,
        'sigma_ring': sigma_ring,
        'A1': A1,
        # Pass through for context
        'B_T': B_T,
        'shear': shear,
        'bar_class': bar_class
    }


def main():
    """Demo: compare path-spectrum coherence lengths across morphologies."""
    
    print("="*80)
    print("PATH-SPECTRUM KERNEL DEMO")
    print("="*80)
    
    # Test coherence lengths
    test_cases = [
        {'name': 'Early SA (E/S0)', 'B_T': 0.5, 'shear': 0.6, 'bar': 'SA'},
        {'name': 'Intermediate SAB (Sab)', 'B_T': 0.2, 'shear': 1.0, 'bar': 'SAB'},
        {'name': 'Intermediate SB (Sbc)', 'B_T': 0.1, 'shear': 1.2, 'bar': 'SB'},
        {'name': 'Late Sc (high shear)', 'B_T': 0.05, 'shear': 1.5, 'bar': 'S'},
        {'name': 'Late Sd (LSB)', 'B_T': 0.02, 'shear': 0.8, 'bar': 'S'},
    ]
    
    # Initial guess for hyper-parameters
    l0 = 20.0    # Base coherence ~ 20 kpc
    alpha = 2.0  # Bulge suppression
    beta = 1.5   # Shear suppression
    gamma = 1.0  # Bar suppression
    
    print(f"\nHyper-parameters: l0={l0:.1f} kpc, α={alpha:.1f}, β={beta:.1f}, γ={gamma:.1f}")
    print("\nCoherence Lengths by Morphology:")
    print("-"*80)
    
    for case in test_cases:
        l_coh = coherence_length(case['B_T'], case['shear'], case['bar'],
                                 l0, alpha, beta, gamma)
        
        # Approximate ring wavelength (for comparison to V2.2 lambda_ring)
        lambda_equiv = 2.0 * np.pi * l_coh
        
        print(f"{case['name']:30s}  ℓ_coh={l_coh:5.1f} kpc  (λ≈{lambda_equiv:5.1f} kpc)")
    
    print("\n" + "="*80)
    print("Compare to V2.2 lambda_ring values:")
    print("  Early types:  λ ~ 8-10 kpc   → ℓ_coh ~ 1.3-1.6 kpc")
    print("  Late types:   λ ~ 18-22 kpc  → ℓ_coh ~ 2.9-3.5 kpc")
    print("="*80)
    
    # Demo: kernel values at different radii
    print("\nKernel values K_1(R, z=0) for Late Sc:")
    R_test = np.array([1, 2, 5, 10, 20, 30])
    z_test = np.zeros_like(R_test)
    
    l_coh_lateSc = coherence_length(0.05, 1.5, 'S', l0, alpha, beta, gamma)
    K1_vals = path_kernel_m1(R_test, z_test, l_coh_lateSc, A1=1.0)
    
    print(f"  ℓ_coh = {l_coh_lateSc:.2f} kpc")
    for r, k in zip(R_test, K1_vals):
        print(f"  R={r:2.0f} kpc  →  K_1={k:.4f}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
