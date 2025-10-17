"""
BCG/ICL Stellar Mass Profiles
==============================

Adds central galaxy stellar mass contribution to cluster lensing models.
Uses Hernquist or Sérsic profiles with empirical M_BCG-M_500 scaling relations.

References:
- Huang et al. 2018 (ApJ 856, 2) - BCG mass-halo mass relation
- Kluge et al. 2020 (ApJS 247, 43) - Frontier Fields BCG photometry
- Hernquist 1990 (ApJ 356, 359) - Hernquist profile

Author: GravityCalculator
Date: 2025-01-19
"""

import numpy as np

def hernquist_density(r, M_star, r_eff):
    """
    Hernquist (1990) density profile for BCG stellar mass.
    
    ρ(r) = (M_star / 2π) * (a / r) * 1/(r + a)^3
    
    where a = r_eff / 1.8153 (matches Sérsic n=4 effective radius)
    
    Parameters:
    -----------
    r : array_like
        3D radius [kpc]
    M_star : float
        Total stellar mass [Msun]
    r_eff : float
        Effective (half-light) radius [kpc]
    
    Returns:
    --------
    rho : array_like
        3D density [Msun/kpc^3]
    """
    a = r_eff / 1.8153  # Hernquist scale radius
    rho_prefactor = M_star / (2 * np.pi)
    
    r = np.asarray(r)
    rho = np.zeros_like(r, dtype=float)
    
    mask = r > 0
    rho[mask] = rho_prefactor * (a / r[mask]) / (r[mask] + a)**3
    
    # Handle r=0 analytically (diverges, but integrable)
    # For numerical stability, set small-r behavior
    rho[~mask] = rho_prefactor / a**3
    
    return rho

def hernquist_enclosed_mass(r, M_star, r_eff):
    """
    Enclosed mass for Hernquist profile.
    
    M(<r) = M_star * r^2 / (r + a)^2
    
    Parameters:
    -----------
    r : array_like
        3D radius [kpc]
    M_star : float
        Total stellar mass [Msun]
    r_eff : float
        Effective radius [kpc]
    
    Returns:
    --------
    M_enc : array_like
        Enclosed mass [Msun]
    """
    a = r_eff / 1.8153
    r = np.asarray(r)
    return M_star * r**2 / (r + a)**2

def hernquist_projected_density(R, M_star, r_eff):
    """
    Projected (surface) density for Hernquist profile via Abel transform.
    
    Σ(R) = M_star / (2π) * 1/(R^2 - a^2) * [stuff involving elliptic integrals]
    
    For R << a: Σ ~ M_star/(2πa^2)
    For R >> a: Σ ~ M_star/(2πR^2)
    
    We use the exact analytic solution from Hernquist (1990), Eq. 33.
    
    Parameters:
    -----------
    R : array_like
        Projected radius [kpc]
    M_star : float
        Total stellar mass [Msun]
    r_eff : float
        Effective radius [kpc]
    
    Returns:
    --------
    Sigma : array_like
        Surface density [Msun/kpc^2]
    """
    a = r_eff / 1.8153
    R = np.asarray(R)
    x = R / a  # Dimensionless radius
    
    Sigma = np.zeros_like(R, dtype=float)
    
    # Small R: use series expansion to avoid divergence
    mask_small = x < 1e-3
    if np.any(mask_small):
        Sigma[mask_small] = M_star / (2 * np.pi * a**2) * (1 - x[mask_small]**2 / 3)
    
    # x < 1 (inside core)
    mask_inner = (x >= 1e-3) & (x < 1.0)
    if np.any(mask_inner):
        x_inner = x[mask_inner]
        # Exact formula from Hernquist 1990, Eq. 33
        F = (2 + x_inner**2) * np.sqrt(1 - x_inner**2) + 3 * np.arcsin(x_inner)
        Sigma[mask_inner] = M_star / (2 * np.pi * a**2) * F / (1 - x_inner**2)**2
    
    # x = 1 (exactly at scale radius)
    mask_unity = np.isclose(x, 1.0, atol=1e-6)
    if np.any(mask_unity):
        Sigma[mask_unity] = M_star / (2 * np.pi * a**2) * (2 * np.pi - 3)
    
    # x > 1 (outside core)
    mask_outer = x > 1.0
    if np.any(mask_outer):
        x_outer = x[mask_outer]
        # Exact formula, Eq. 33
        F = (2 + x_outer**2) * np.sqrt(x_outer**2 - 1) - 3 * np.arccosh(x_outer)
        Sigma[mask_outer] = M_star / (2 * np.pi * a**2) * F / (x_outer**2 - 1)**2
    
    return Sigma

def estimate_bcg_mass(M_500, z_lens):
    """
    Empirical BCG stellar mass from M_500 using scaling relations.
    
    Based on Huang et al. 2018, ApJ 856, 2 (Figure 6):
    log10(M_BCG / Msun) ≈ 0.4 * log10(M_500 / 1e14 Msun) + 11.9
    
    with ~0.15 dex scatter
    
    Parameters:
    -----------
    M_500 : float
        Cluster mass M_500 [Msun]
    z_lens : float
        Cluster redshift (for evolutionary correction if needed)
    
    Returns:
    --------
    M_BCG : float
        BCG stellar mass [Msun]
    r_eff : float
        Effective radius [kpc] (typical 10-20 kpc)
    """
    # Huang+2018 relation
    log_M_BCG = 0.4 * np.log10(M_500 / 1e14) + 11.9
    M_BCG = 10**log_M_BCG
    
    # Typical BCG effective radius: 10-20 kpc (weakly mass-dependent)
    # Kluge+2020 Frontier Fields: R_e ~ 15 ± 5 kpc
    r_eff = 15.0 + 5.0 * np.log10(M_500 / 1e15)  # kpc
    r_eff = np.clip(r_eff, 10.0, 25.0)
    
    return M_BCG, r_eff

# Example usage
if __name__ == "__main__":
    print("BCG/ICL Stellar Mass Utilities")
    print("=" * 60)
    
    # Test case: MACS0416
    M_500_test = 1.15e15  # Msun
    z_test = 0.396
    
    M_BCG, r_eff = estimate_bcg_mass(M_500_test, z_test)
    
    print(f"\nTest: MACS0416 (M_500 = {M_500_test:.2e} Msun, z = {z_test:.3f})")
    print(f"  Estimated M_BCG: {M_BCG:.2e} Msun")
    print(f"  Effective radius: {r_eff:.1f} kpc")
    
    # Compute projected density profile
    R = np.logspace(0, 2.5, 100)  # 1-300 kpc
    Sigma_BCG = hernquist_projected_density(R, M_BCG, r_eff)
    
    # Central surface density
    print(f"  Σ(R=1 kpc): {Sigma_BCG[0]:.2e} Msun/kpc^2")
    print(f"  Σ(R=10 kpc): {hernquist_projected_density(10.0, M_BCG, r_eff):.2e} Msun/kpc^2")
    print(f"  Σ(R=100 kpc): {hernquist_projected_density(100.0, M_BCG, r_eff):.2e} Msun/kpc^2")
    
    # Enclosed stellar mass
    print(f"\n  M_star(<10 kpc): {hernquist_enclosed_mass(10.0, M_BCG, r_eff):.2e} Msun")
    print(f"  M_star(<100 kpc): {hernquist_enclosed_mass(100.0, M_BCG, r_eff):.2e} Msun")
    
    # Contribution to Einstein radius (rough estimate)
    # For MACS0416, θ_E ~ 30" ~ 180 kpc at z=0.396
    R_E_kpc = 180.0
    M_BCG_enc = hernquist_enclosed_mass(R_E_kpc, M_BCG, r_eff)
    frac_BCG = M_BCG_enc / (M_500_test * 0.5)  # Rough: M(<R_E) ~ 0.5*M_500
    
    print(f"\n  M_BCG(<R_E={R_E_kpc:.0f} kpc): {M_BCG_enc:.2e} Msun")
    print(f"  Fractional contribution to M(<R_E): {frac_BCG*100:.1f}%")
    
    print("\n" + "=" * 60)
