#!/usr/bin/env python3
"""
Triaxial Lensing Module
=======================

Implements triaxial geometry transformations for cluster lensing.

Key Concepts:
-------------
- Transform spherical density rho(r) → triaxial rho(x,y,z)
- Ellipsoidal radius: m² = x² + (y/q_plane)² + (z/q_LOS)²
- Volume element correction: dV_triaxial = q_plane × q_LOS × dV_spherical
- Projection: integrate along line-of-sight (z-axis) to get Sigma(R)

Physical Interpretation:
------------------------
- q_plane < 1: Oblate (flattened in sky plane), like a disk seen face-on
- q_plane = 1: Circular in sky plane
- q_LOS < 1: Flattened along line-of-sight (appears more concentrated)
- q_LOS > 1: Elongated along line-of-sight (bimodal merger, prolate)
- q_LOS = 1: Spherical along LOS

This allows per-cluster geometry to vary while keeping kernel universal.

References:
-----------
- Jing & Suto 2002: Triaxial NFW halos and lensing
- Oguri+ 2005: Triaxial modeling of cluster strong lensing
- Sereno+ 2013: Geometry effects on Einstein radii

Author: GravityCalculator
Date: 2025-01-14
"""

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
from typing import Callable, Tuple, Optional


def rotate_coordinates(
    x: np.ndarray,
    y: np.ndarray, 
    z: np.ndarray,
    phi: float = 0.0,
    theta: float = 0.0,
    psi: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Euler rotation to coordinates.
    
    Parameters
    ----------
    x, y, z : array
        Original coordinates
    phi : float
        Rotation about z-axis (in-plane, radians)
    theta : float
        Inclination from z-axis (radians)
    psi : float
        Rotation about final z-axis (radians)
    
    Returns
    -------
    x_rot, y_rot, z_rot : array
        Rotated coordinates
    """
    # For now, implement simple case (aligned with axes)
    # Full Euler angles can be added later if needed
    if phi == 0 and theta == 0 and psi == 0:
        return x, y, z
    
    # Rotation matrix: R = R_z(phi) × R_y(theta) × R_z(psi)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    
    # R_z(phi)
    x1 = cos_phi * x - sin_phi * y
    y1 = sin_phi * x + cos_phi * y
    z1 = z
    
    # R_y(theta)
    x2 = cos_theta * x1 + sin_theta * z1
    y2 = y1
    z2 = -sin_theta * x1 + cos_theta * z1
    
    # R_z(psi)
    x_rot = cos_psi * x2 - sin_psi * y2
    y_rot = sin_psi * x2 + cos_psi * y2
    z_rot = z2
    
    return x_rot, y_rot, z_rot


def ellipsoidal_radius(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    q_plane: float = 1.0,
    q_LOS: float = 1.0
) -> np.ndarray:
    """
    Compute ellipsoidal radius m from Cartesian coordinates.
    
    m² = x² + (y/q_plane)² + (z/q_LOS)²
    
    This is the "equivalent spherical radius" for a triaxial ellipsoid
    with semi-axes (a, b=q_plane×a, c=q_LOS×a).
    
    Parameters
    ----------
    x, y, z : array
        Cartesian coordinates [kpc]
    q_plane : float
        In-plane axis ratio b/a (0.6-1.0)
    q_LOS : float
        Line-of-sight axis ratio c/a (0.6-1.6)
    
    Returns
    -------
    m : array
        Ellipsoidal radius [kpc]
    """
    return np.sqrt(x**2 + (y/q_plane)**2 + (z/q_LOS)**2)


def spherical_to_triaxial_density(
    rho_spherical: Callable[[np.ndarray], np.ndarray],
    q_plane: float = 1.0,
    q_LOS: float = 1.0,
    phi: float = 0.0,
    theta: float = 0.0
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Transform spherical density rho(r) to triaxial density rho(x,y,z).
    
    The transformation preserves total mass via volume element correction:
    
    rho_triaxial(x,y,z) = rho_spherical(m) / (q_plane × q_LOS)
    
    where m is the ellipsoidal radius.
    
    Physical meaning:
    - The density at a given ellipsoidal radius m is the same as the
      spherical density at radius m
    - The volume element changes by factor (q_plane × q_LOS), so we
      divide by this to preserve integrated mass
    
    Parameters
    ----------
    rho_spherical : callable
        Spherical density function rho(r) [Msun/kpc³]
    q_plane : float
        In-plane axis ratio b/a (default: 1.0 = circular)
    q_LOS : float
        Line-of-sight axis ratio c/a (default: 1.0 = spherical)
    phi, theta : float
        Euler angles for orientation (default: aligned with axes)
    
    Returns
    -------
    rho_triaxial : callable
        Triaxial density function rho(x,y,z) [Msun/kpc³]
    
    Examples
    --------
    >>> # Hernquist profile
    >>> rho_sph = lambda r: M / (2*np.pi) * a / (r * (r+a)**3)
    >>> rho_tri = spherical_to_triaxial_density(rho_sph, q_plane=0.8, q_LOS=1.2)
    >>> # Now rho_tri(x,y,z) gives triaxial density
    """
    def rho_triaxial(x, y, z):
        # Apply rotation if needed
        if phi != 0 or theta != 0:
            x_rot, y_rot, z_rot = rotate_coordinates(x, y, z, phi, theta)
        else:
            x_rot, y_rot, z_rot = x, y, z
        
        # Compute ellipsoidal radius
        m = ellipsoidal_radius(x_rot, y_rot, z_rot, q_plane, q_LOS)
        
        # Evaluate spherical density at m, apply volume correction
        # Volume element: dV_tri = q_plane × q_LOS × dV_sph
        # To conserve mass: rho_tri × dV_tri = rho_sph × dV_sph
        # Therefore: rho_tri = rho_sph / (q_plane × q_LOS)
        return rho_spherical(m) / (q_plane * q_LOS)
    
    return rho_triaxial


def project_triaxial_to_surface_density_simple(
    rho_triaxial: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    R_proj: np.ndarray,
    z_max: float = 5000.0,
    n_z: int = 200
) -> np.ndarray:
    """
    Project triaxial 3D density along line-of-sight to get Sigma(R).
    
    For spherically symmetric case:
    Sigma(R) = 2 ∫[R to ∞] rho(r) × r / sqrt(r² - R²) dr
    
    For triaxial case (aligned with z-axis):
    Sigma(R) = ∫[-∞ to ∞] rho(R, 0, z) dz
    
    We integrate along z-axis through point (R, 0, z).
    
    Parameters
    ----------
    rho_triaxial : callable
        Triaxial density function rho(x,y,z)
    R_proj : array
        Projected radii [kpc]
    z_max : float
        Integration limit along LOS [kpc]
    n_z : int
        Number of integration points
    
    Returns
    -------
    Sigma : array
        Surface density [Msun/kpc²]
    """
    Sigma = np.zeros_like(R_proj)
    z_grid = np.linspace(-z_max, z_max, n_z)
    
    for i, R in enumerate(R_proj):
        # Integrate rho(R, 0, z) along z
        integrand = rho_triaxial(R, 0.0, z_grid)
        Sigma[i] = np.trapz(integrand, z_grid)
    
    return Sigma


def project_triaxial_to_surface_density_accurate(
    rho_triaxial: Callable[[float, float, float], float],
    R_proj: np.ndarray,
    z_max: float = 5000.0
) -> np.ndarray:
    """
    Project triaxial density with accurate integration (slower but precise).
    
    Uses scipy.integrate.quad for each projected radius.
    
    Parameters
    ----------
    rho_triaxial : callable
        Triaxial density rho(x,y,z) - must accept scalar inputs
    R_proj : array
        Projected radii [kpc]
    z_max : float
        Integration limit [kpc]
    
    Returns
    -------
    Sigma : array
        Surface density [Msun/kpc²]
    """
    Sigma = np.zeros_like(R_proj)
    
    for i, R in enumerate(R_proj):
        # Integrate along z-axis through (R, 0, z)
        integrand = lambda z: rho_triaxial(R, 0.0, z)
        Sigma[i], _ = quad(integrand, -z_max, z_max, limit=100)
    
    return Sigma


def compute_enclosed_mass_triaxial(
    rho_triaxial: Callable,
    R_max: float = 3000.0,
    n_r: int = 50,
    n_theta: int = 30,
    n_phi: int = 30
) -> float:
    """
    Compute total enclosed mass for triaxial density (diagnostic).
    
    M_enc = ∫∫∫ rho(x,y,z) dx dy dz
    
    Uses spherical coordinates with ellipsoidal radius.
    
    Parameters
    ----------
    rho_triaxial : callable
        Triaxial density rho(x,y,z)
    R_max : float
        Maximum radius for integration [kpc]
    n_r, n_theta, n_phi : int
        Number of integration points in (r, theta, phi)
    
    Returns
    -------
    M_enc : float
        Enclosed mass [Msun]
    """
    # Use log-spaced radial grid for better sampling
    r_grid = np.geomspace(0.1, R_max, n_r)
    theta_grid = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2*np.pi, n_phi)
    
    # Volume element in spherical coords: dV = r² sin(theta) dr dtheta dphi
    M_enc = 0.0
    
    for ir in range(len(r_grid)-1):
        r = 0.5 * (r_grid[ir] + r_grid[ir+1])
        dr = r_grid[ir+1] - r_grid[ir]
        
        for it in range(len(theta_grid)-1):
            theta = 0.5 * (theta_grid[it] + theta_grid[it+1])
            dtheta = theta_grid[it+1] - theta_grid[it]
            
            for ip in range(len(phi_grid)-1):
                phi = 0.5 * (phi_grid[ip] + phi_grid[ip+1])
                dphi = phi_grid[ip+1] - phi_grid[ip]
                
                # Convert to Cartesian
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                
                # Evaluate density
                rho = rho_triaxial(x, y, z)
                
                # Add contribution (volume element)
                dV = r**2 * np.sin(theta) * dr * dtheta * dphi
                M_enc += rho * dV
    
    return M_enc


def test_mass_conservation():
    """
    Unit test: triaxial transformation should conserve total mass.
    """
    print("Testing mass conservation for triaxial transformation...")
    print("="*60)
    
    # Simple power-law profile: rho(r) = rho_0 / (1 + r/r_s)^3
    rho_0 = 1e6  # Msun/kpc^3
    r_s = 100.0  # kpc
    
    def rho_spherical(r):
        return rho_0 / (1 + r/r_s)**3
    
    # Analytic enclosed mass (integrate 4π r² rho(r) dr)
    # For (1+r/r_s)^-3: M(R) = 4π rho_0 r_s³ × [ln(1+R/r_s) - R/(R+r_s)]
    def M_spherical_analytic(R):
        x = R / r_s
        return 4 * np.pi * rho_0 * r_s**3 * (np.log(1+x) - x/(1+x))
    
    R_test = 1000.0  # kpc
    M_analytic = M_spherical_analytic(R_test)
    
    print(f"\nSpherical case (q_plane=1, q_LOS=1):")
    print(f"  Analytic M(<{R_test} kpc) = {M_analytic:.3e} Msun")
    
    # Test triaxial with different q values
    test_cases = [
        (1.0, 1.0, "Spherical"),
        (0.8, 1.0, "Oblate in-plane"),
        (1.0, 0.7, "Oblate LOS"),
        (1.0, 1.4, "Prolate LOS"),
        (0.85, 1.2, "Mixed")
    ]
    
    for q_plane, q_LOS, description in test_cases:
        rho_tri = spherical_to_triaxial_density(
            rho_spherical, q_plane=q_plane, q_LOS=q_LOS
        )
        
        # Compute mass (numerical, slower)
        M_tri = compute_enclosed_mass_triaxial(
            rho_tri, R_max=R_test, n_r=60, n_theta=40, n_phi=40
        )
        
        frac_error = abs(M_tri - M_analytic) / M_analytic
        status = "PASS" if frac_error < 0.05 else "FAIL"
        
        print(f"\n{description} (q_plane={q_plane}, q_LOS={q_LOS}):")
        print(f"  Numerical M(<{R_test} kpc) = {M_tri:.3e} Msun")
        print(f"  Fractional error = {frac_error*100:.1f}% [{status}]")
    
    print("\n" + "="*60)
    print("Mass conservation test complete!\n")


if __name__ == '__main__':
    print("Triaxial Lensing Module")
    print("="*60)
    print("\nThis module implements geometry transformations for")
    print("triaxial cluster lensing. Run test_mass_conservation()")
    print("to verify mass is conserved under triaxial transformation.")
    print()
    
    # Run unit test
    test_mass_conservation()
    
    print("\nUsage example:")
    print("-"*60)
    print("# 1. Start with spherical density")
    print("rho_sph = lambda r: M / (2*np.pi*a) / (r * (r+a)**3)")
    print()
    print("# 2. Transform to triaxial")
    print("rho_tri = spherical_to_triaxial_density(")
    print("    rho_sph, q_plane=0.85, q_LOS=1.15")
    print(")")
    print()
    print("# 3. Project to surface density")
    print("R_proj = np.geomspace(10, 1500, 100)")
    print("Sigma = project_triaxial_to_surface_density_simple(")
    print("    rho_tri, R_proj")
    print(")")
    print()
    print("# 4. Use in lensing computation")
    print("# (integrate Sigma with 3D shell kernel)")
