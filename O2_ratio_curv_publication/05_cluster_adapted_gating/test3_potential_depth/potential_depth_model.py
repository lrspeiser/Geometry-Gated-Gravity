"""
potential_depth_model.py

Extended O2 model with gravitational potential depth gating.

Model form:
    fX = (x^2 / 2) * (1 / denom) * amplification
    
    denom = a - b * Σ̂ - d * |∇ln Σ|
    amplification = exp(β * |Φ| / Φ₀)  or  (|Φ| / Φ₀)^γ
    
Where:
    - a, b, d: O2 baseline parameters (fixed from galaxy fitting)
    - β or γ: potential depth coefficient (new parameter)
    - Φ: gravitational potential |Φ| = ∫ g(r) dr from r to ∞
    - Φ₀: normalization constant (10^4 km²/s²)

Physical motivation:
    - Deeper potential wells → stronger gravitational effects
    - Clusters: |Φ| ~ 10^5-10^6 km²/s² at R ~ 100-500 kpc
    - Galaxies: |Φ| ~ 10^4 km²/s² at R ~ 10 kpc
    - This is a scale-free GR-motivated quantity
    - Should amplify effects in deep wells (clusters)
    
Author: Henry Speiser
Date: October 2, 2025
"""

import numpy as np
from typing import Tuple, Literal


def compute_gravitational_potential(
    R_kpc: np.ndarray,
    g_total_kms2: np.ndarray
) -> np.ndarray:
    """
    Compute gravitational potential from acceleration profile.
    
    Φ(R) = -∫_R^∞ g(r) dr
    
    We approximate as:
    Φ(R) ≈ -Σ g(R_i) * ΔR_i for R_i > R
    
    Parameters
    ----------
    R_kpc : np.ndarray
        Radii in kpc (must be sorted ascending)
    
    g_total_kms2 : np.ndarray
        Total gravitational acceleration in km/s²
        (from observations or Newtonian prediction)
    
    Returns
    -------
    Phi_km2s2 : np.ndarray
        Gravitational potential in km²/s²
        Negative values (convention: Φ(∞) = 0)
    
    Notes
    -----
    - We use absolute value |Φ| for gating
    - Larger |Φ| means deeper well
    """
    # Check inputs
    if not np.all(np.diff(R_kpc) > 0):
        raise ValueError("R_kpc must be sorted ascending")
    
    if len(R_kpc) != len(g_total_kms2):
        raise ValueError("R_kpc and g_total_kms2 must have same length")
    
    # Convert kpc to km for integration
    R_km = R_kpc * 3.086e16  # 1 kpc = 3.086e16 km
    
    # Compute Φ by integrating from R to infinity
    # Φ(R) = -∫_R^∞ g(r) dr
    # Approximate: integrate from R to R_max, assume g→0 beyond
    
    n = len(R_km)
    Phi_km2s2 = np.zeros(n)
    
    for i in range(n):
        # Integrate from R[i] to R[-1] using trapezoidal rule
        if i < n - 1:
            Phi_km2s2[i] = -np.trapz(g_total_kms2[i:], R_km[i:])
        else:
            # At outermost point, assume Φ ≈ -g*R/2 (rough extrapolation)
            Phi_km2s2[i] = -0.5 * g_total_kms2[i] * R_km[i]
    
    return Phi_km2s2


def fX_ratio_curv_potential_exp(
    params: Tuple[float, float, float, float],
    x: float,
    Sigma_hat: float,
    grad_ln_Sigma: float,
    Phi_km2s2: float
) -> float:
    """
    Compute fX with exponential potential depth gating.
    
    Model: fX = fX_base * exp(β * |Φ| / Φ₀)
    
    Parameters
    ----------
    params : tuple of (a, b, d, β)
        a : baseline denominator constant (O2 value: 0.6687)
        b : surface density coefficient (O2 value: 0.1401)
        d : gradient coefficient (O2 value: 0.0871)
        β : potential depth coefficient (NEW, to be fitted)
    
    x : float
        Dimensionless radius (r / r_turnaround)
    
    Sigma_hat : float
        Normalized surface density, log10(Σ / Σ_crit)
    
    grad_ln_Sigma : float
        Absolute value of logarithmic surface density gradient
    
    Phi_km2s2 : float
        Gravitational potential in km²/s²
        Use absolute value |Φ| (deeper wells = larger |Φ|)
    
    Returns
    -------
    fX : float
        Lensing amplification factor
        Returns np.nan if denominator is negative (unphysical)
    """
    a, b, d, beta = params
    
    # Base O2 denominator (unchanged)
    denom = a - b * Sigma_hat - d * abs(grad_ln_Sigma)
    
    if denom <= 0:
        return np.nan
    
    # Base fX
    fX_base = (x**2 / 2.0) / denom
    
    # Potential depth amplification
    Phi_0 = 1e4  # km²/s² (normalization, roughly galaxy scale)
    Phi_abs = abs(Phi_km2s2)
    
    # Exponential amplification
    amplification = np.exp(beta * Phi_abs / Phi_0)
    
    # Final fX
    fX = fX_base * amplification
    
    return fX


def fX_ratio_curv_potential_power(
    params: Tuple[float, float, float, float],
    x: float,
    Sigma_hat: float,
    grad_ln_Sigma: float,
    Phi_km2s2: float
) -> float:
    """
    Compute fX with power-law potential depth gating.
    
    Model: fX = fX_base * (|Φ| / Φ₀)^γ
    
    Parameters
    ----------
    params : tuple of (a, b, d, γ)
        a : baseline denominator constant (O2 value: 0.6687)
        b : surface density coefficient (O2 value: 0.1401)
        d : gradient coefficient (O2 value: 0.0871)
        γ : potential depth power-law exponent (NEW, to be fitted)
    
    x : float
        Dimensionless radius (r / r_turnaround)
    
    Sigma_hat : float
        Normalized surface density, log10(Σ / Σ_crit)
    
    grad_ln_Sigma : float
        Absolute value of logarithmic surface density gradient
    
    Phi_km2s2 : float
        Gravitational potential in km²/s²
        Use absolute value |Φ| (deeper wells = larger |Φ|)
    
    Returns
    -------
    fX : float
        Lensing amplification factor
        Returns np.nan if denominator is negative (unphysical)
    """
    a, b, d, gamma = params
    
    # Base O2 denominator (unchanged)
    denom = a - b * Sigma_hat - d * abs(grad_ln_Sigma)
    
    if denom <= 0:
        return np.nan
    
    # Base fX
    fX_base = (x**2 / 2.0) / denom
    
    # Potential depth amplification (power-law)
    Phi_0 = 1e4  # km²/s² (normalization)
    Phi_abs = abs(Phi_km2s2)
    
    # Avoid division by zero
    if Phi_abs < 1.0:
        Phi_abs = 1.0
    
    # Power-law amplification
    amplification = (Phi_abs / Phi_0) ** gamma
    
    # Final fX
    fX = fX_base * amplification
    
    return fX


def estimate_typical_potential_depth(
    M_vir_Msun: float,
    R_vir_kpc: float,
    system_type: Literal['galaxy', 'cluster'] = 'galaxy'
) -> float:
    """
    Estimate typical gravitational potential depth for a system.
    
    Rough approximation: |Φ| ~ G * M / R
    
    Parameters
    ----------
    M_vir_Msun : float
        Virial mass in solar masses
    
    R_vir_kpc : float
        Virial radius in kpc
    
    system_type : str
        'galaxy' or 'cluster'
    
    Returns
    -------
    Phi_typical_km2s2 : float
        Typical potential depth in km²/s²
    
    Examples
    --------
    Milky Way:
        M ~ 1e12 Msun, R ~ 200 kpc
        |Φ| ~ 4e4 km²/s²
    
    Massive cluster:
        M ~ 1e15 Msun, R ~ 2000 kpc
        |Φ| ~ 4e5 km²/s²
    """
    # Constants
    G_SI = 6.674e-11  # m³ kg⁻¹ s⁻²
    Msun_kg = 1.989e30  # kg
    kpc_m = 3.086e19  # m
    
    # Convert to SI
    M_kg = M_vir_Msun * Msun_kg
    R_m = R_vir_kpc * kpc_m
    
    # Φ ~ G M / R (rough estimate)
    Phi_SI = G_SI * M_kg / R_m  # m²/s²
    
    # Convert to km²/s²
    Phi_km2s2 = Phi_SI / 1e6
    
    return Phi_km2s2


if __name__ == "__main__":
    # Quick sanity check
    print("Gravitational Potential Depth Model - Sanity Check")
    print("=" * 70)
    
    # Test potential computation
    print("\nTest 1: Compute potential from g(R)")
    R_test = np.array([1, 2, 5, 10, 20, 50])  # kpc
    g_test = 1e-9 * np.array([100, 80, 50, 30, 15, 5])  # km/s²
    
    Phi_test = compute_gravitational_potential(R_test, g_test)
    
    print(f"R [kpc]:     {R_test}")
    print(f"g [km/s²]:   {g_test}")
    print(f"|Φ| [km²/s²]: {np.abs(Phi_test)}")
    
    # Test typical potential depths
    print("\nTest 2: Typical potential depths")
    systems = [
        ("Dwarf galaxy", 1e9, 10, 'galaxy'),
        ("Milky Way", 1e12, 200, 'galaxy'),
        ("Massive spiral", 5e12, 300, 'galaxy'),
        ("Poor cluster", 1e14, 1000, 'cluster'),
        ("Rich cluster", 1e15, 2000, 'cluster'),
    ]
    
    for name, M, R, sys_type in systems:
        Phi = estimate_typical_potential_depth(M, R, sys_type)
        print(f"{name:20s}: M = {M:.1e} Msun, R = {R:4.0f} kpc → |Φ| = {Phi:.2e} km²/s²")
    
    # Test amplification models
    print("\nTest 3: Amplification factors")
    params_baseline = (0.6687, 0.1401, 0.0871, 0.0)  # β or γ = 0 → no amplification
    
    # Exponential model with β = 0.1
    params_exp = (0.6687, 0.1401, 0.0871, 0.1)
    
    # Power-law model with γ = 0.5
    params_pow = (0.6687, 0.1401, 0.0871, 0.5)
    
    x = 10.0
    Sigma_hat = -1.5
    grad_ln_Sigma = 0.3
    
    Phi_galaxy = 4e4  # km²/s²
    Phi_cluster = 4e5  # km²/s²
    
    # Baseline
    fX_base_gal = fX_ratio_curv_potential_exp(params_baseline, x, Sigma_hat, grad_ln_Sigma, Phi_galaxy)
    fX_base_clu = fX_ratio_curv_potential_exp(params_baseline, x, Sigma_hat, grad_ln_Sigma, Phi_cluster)
    
    # Exponential
    fX_exp_gal = fX_ratio_curv_potential_exp(params_exp, x, Sigma_hat, grad_ln_Sigma, Phi_galaxy)
    fX_exp_clu = fX_ratio_curv_potential_exp(params_exp, x, Sigma_hat, grad_ln_Sigma, Phi_cluster)
    
    # Power-law
    fX_pow_gal = fX_ratio_curv_potential_power(params_pow, x, Sigma_hat, grad_ln_Sigma, Phi_galaxy)
    fX_pow_clu = fX_ratio_curv_potential_power(params_pow, x, Sigma_hat, grad_ln_Sigma, Phi_cluster)
    
    print("\nBaseline (no potential gating):")
    print(f"  Galaxy:  fX = {fX_base_gal:.2f}")
    print(f"  Cluster: fX = {fX_base_clu:.2f}")
    print(f"  Ratio: {fX_base_clu / fX_base_gal:.2f}×")
    
    print("\nExponential model (β = 0.1):")
    print(f"  Galaxy:  fX = {fX_exp_gal:.2f} ({fX_exp_gal/fX_base_gal:.2f}× amplification)")
    print(f"  Cluster: fX = {fX_exp_clu:.2f} ({fX_exp_clu/fX_base_clu:.2f}× amplification)")
    print(f"  Cluster/Galaxy boost ratio: {(fX_exp_clu/fX_base_clu) / (fX_exp_gal/fX_base_gal):.2f}×")
    
    print("\nPower-law model (γ = 0.5):")
    print(f"  Galaxy:  fX = {fX_pow_gal:.2f} ({fX_pow_gal/fX_base_gal:.2f}× amplification)")
    print(f"  Cluster: fX = {fX_pow_clu:.2f} ({fX_pow_clu/fX_base_clu:.2f}× amplification)")
    print(f"  Cluster/Galaxy boost ratio: {(fX_pow_clu/fX_base_clu) / (fX_pow_gal/fX_base_gal):.2f}×")
    
    print("\n✅ Model loaded successfully")
    print("\nKey insight:")
    print("  - Galaxy |Φ| ~ 4e4 km²/s²")
    print("  - Cluster |Φ| ~ 4e5 km²/s² (10× deeper)")
    print("  - Exponential gating: exp(β * 10) can provide large boost")
    print("  - Power-law gating: (10)^γ provides controlled boost")
