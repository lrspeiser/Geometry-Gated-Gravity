"""
NFW Mass Definition Conversions
================================

Converts between different halo mass definitions (M_200c, M_500c, etc.)
using the NFW profile and iterative root-finding.

Based on:
- Hu & Kravtsov 2003 (ApJ 584, 702)
- Wright & Brainerd 2000 (ApJ 534, 34)

Author: GravityCalculator
Date: 2025-01-19
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

# Cosmology (Planck 2018)
H0 = 70.0  # km/s/Mpc
OMEGA_M = 0.3
OMEGA_L = 0.7
RHO_CRIT_0 = 2.775e11  # Msun/Mpc^3 at z=0 for h=0.7

def rho_crit(z):
    """Critical density at redshift z [Msun/Mpc^3]."""
    E_z = np.sqrt(OMEGA_M * (1+z)**3 + OMEGA_L)
    return RHO_CRIT_0 * E_z**2

def nfw_profile(r, r_s, rho_s):
    """NFW density profile [Msun/Mpc^3]."""
    x = r / r_s
    return rho_s / (x * (1 + x)**2)

def nfw_enclosed_mass(r, r_s, rho_s):
    """Enclosed mass within radius r for NFW profile [Msun]."""
    x = r / r_s
    mass_prefactor = 4 * np.pi * rho_s * r_s**3
    f_x = np.log(1 + x) - x / (1 + x)
    return mass_prefactor * f_x

def nfw_mean_density(r, r_s, rho_s):
    """Mean density within radius r [Msun/Mpc^3]."""
    M_enc = nfw_enclosed_mass(r, r_s, rho_s)
    V = (4./3.) * np.pi * r**3
    return M_enc / V

def compute_r_s_from_concentration(M_Delta, c_Delta, Delta, z):
    """
    Compute NFW scale radius r_s from M_Delta and c_Delta.
    
    Parameters:
    -----------
    M_Delta : float
        Mass at overdensity Delta [Msun]
    c_Delta : float
        Concentration at overdensity Delta
    Delta : float
        Overdensity (e.g., 200 for M_200c)
    z : float
        Redshift
    
    Returns:
    --------
    r_s : float
        Scale radius [Mpc]
    """
    rho_c = rho_crit(z)
    r_Delta = (3 * M_Delta / (4 * np.pi * Delta * rho_c))**(1./3.)  # Mpc
    r_s = r_Delta / c_Delta
    return r_s

def compute_rho_s_from_concentration(M_Delta, c_Delta, Delta, z):
    """
    Compute NFW characteristic density rho_s.
    
    Returns:
    --------
    rho_s : float
        Characteristic density [Msun/Mpc^3]
    """
    r_s = compute_r_s_from_concentration(M_Delta, c_Delta, Delta, z)
    rho_c = rho_crit(z)
    
    # From M_Delta = M_enc(r_Delta), solve for rho_s
    # M_Delta = 4*pi*rho_s*r_s^3 * [ln(1+c) - c/(1+c)]
    c = c_Delta
    f_c = np.log(1 + c) - c / (1 + c)
    
    rho_s = M_Delta / (4 * np.pi * r_s**3 * f_c)
    return rho_s

def convert_mass_definition(M_in, c_in, Delta_in, Delta_out, z, tol=1e-6, max_iter=100):
    """
    Convert NFW halo mass from one overdensity definition to another.
    
    Example: M_200c, c_200c → M_500c, c_500c
    
    Parameters:
    -----------
    M_in : float
        Input mass [Msun]
    c_in : float
        Input concentration
    Delta_in : float
        Input overdensity (e.g., 200 for M_200c)
    Delta_out : float
        Output overdensity (e.g., 500 for M_500c)
    z : float
        Redshift
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    
    Returns:
    --------
    M_out : float
        Output mass [Msun]
    c_out : float
        Output concentration
    r_out : float
        Output radius [Mpc]
    """
    # Compute NFW parameters from input mass definition
    r_s = compute_r_s_from_concentration(M_in, c_in, Delta_in, z)
    rho_s = compute_rho_s_from_concentration(M_in, c_in, Delta_in, z)
    
    rho_c = rho_crit(z)
    target_rho = Delta_out * rho_c
    
    # Find r_out such that mean density within r_out = Delta_out * rho_crit
    def residual(r):
        return nfw_mean_density(r, r_s, rho_s) - target_rho
    
    # Bracket the solution
    r_in = compute_r_s_from_concentration(M_in, c_in, Delta_in, z) * c_in
    r_search_min = r_in * 0.1
    r_search_max = r_in * 2.0
    
    # Check if solution exists within bracket
    if residual(r_search_min) * residual(r_search_max) > 0:
        # Try wider bracket
        r_search_min = r_s * 0.1
        r_search_max = r_in * 5.0
        
        if residual(r_search_min) * residual(r_search_max) > 0:
            raise ValueError(f"Cannot bracket solution for Delta_out={Delta_out}")
    
    r_out = brentq(residual, r_search_min, r_search_max, xtol=tol)
    M_out = nfw_enclosed_mass(r_out, r_s, rho_s)
    c_out = r_out / r_s
    
    return M_out, c_out, r_out

def M200c_to_M500c(M_200c, c_200c, z):
    """
    Convert M_200c to M_500c for NFW halo.
    
    Parameters:
    -----------
    M_200c : float
        Halo mass at 200 × rho_crit [Msun]
    c_200c : float
        Concentration at 200 × rho_crit
    z : float
        Redshift
    
    Returns:
    --------
    M_500c : float
        Halo mass at 500 × rho_crit [Msun]
    R_500c : float
        Radius at 500 × rho_crit [kpc]
    c_500c : float
        Concentration at 500 × rho_crit
    """
    M_500c, c_500c, r_500c_Mpc = convert_mass_definition(
        M_200c, c_200c, Delta_in=200, Delta_out=500, z=z
    )
    
    R_500c_kpc = r_500c_Mpc * 1000.0  # Mpc → kpc
    
    return M_500c, R_500c_kpc, c_500c

# Example usage and tests
if __name__ == "__main__":
    print("NFW Mass Conversion Utilities")
    print("=" * 60)
    
    # Test case: MACS0416 from Umetsu+2016
    M_200c_test = 1.074e15  # Msun
    c_200c_test = 2.9
    z_test = 0.396
    
    print(f"\nTest: MACS0416")
    print(f"  Input:  M_200c = {M_200c_test:.3e} Msun, c_200c = {c_200c_test:.1f}, z = {z_test:.3f}")
    
    M_500c, R_500c, c_500c = M200c_to_M500c(M_200c_test, c_200c_test, z_test)
    
    print(f"  Output: M_500c = {M_500c:.3e} Msun, R_500c = {R_500c:.1f} kpc, c_500c = {c_500c:.2f}")
    
    # Test case: RXJ1347 (massive, high-c cluster)
    M_200c_test2 = 3.425e15
    c_200c_test2 = 3.2
    z_test2 = 0.451
    
    print(f"\nTest: RXJ1347")
    print(f"  Input:  M_200c = {M_200c_test2:.3e} Msun, c_200c = {c_200c_test2:.1f}, z = {z_test2:.3f}")
    
    M_500c2, R_500c2, c_500c2 = M200c_to_M500c(M_200c_test2, c_200c_test2, z_test2)
    
    print(f"  Output: M_500c = {M_500c2:.3e} Msun, R_500c = {R_500c2:.1f} kpc, c_500c = {c_500c2:.2f}")
    
    print("\n" + "=" * 60)
