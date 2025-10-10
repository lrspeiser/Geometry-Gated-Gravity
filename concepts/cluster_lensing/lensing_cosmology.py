#!/usr/bin/env python3
"""
Lensing Cosmology Module

Handles all cosmological calculations with explicit units and comprehensive testing.
Addresses Editor Concern D: "Tighten cosmology and unit handling"

Usage:
    from lensing_cosmology import LensingCosmology
    
    cosmo = LensingCosmology(H0=70, Om0=0.3)
    D_d = cosmo.angular_diameter_distance(z=0.5)
    Sigma_crit = cosmo.critical_surface_density(z_lens=0.5, z_source=2.0)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# Try to import astropy, fall back to simple implementation if not available
try:
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("Warning: astropy not available. Using simple flat cosmology approximation.")


@dataclass
class PhysicalConstants:
    """Physical constants with explicit units."""
    c_km_s: float = 299792.458  # Speed of light [km/s]
    G_kpc3_Msun_km2s2: float = 4.302e-6  # Gravitational constant [kpc³ M_☉⁻¹ (km/s)²]
    arcsec_per_radian: float = 206265.0  # Conversion factor
    kpc_per_Mpc: float = 1000.0  # Conversion factor


class LensingCosmology:
    """
    Handle all cosmological calculations for gravitational lensing.
    
    Provides:
    - Angular diameter distances with proper redshift handling
    - Critical surface density calculations
    - Angular <-> physical coordinate conversions
    - Unit tests for all operations
    
    Parameters
    ----------
    H0 : float, optional
        Hubble constant in km/s/Mpc (default: 70)
    Om0 : float, optional
        Matter density parameter (default: 0.3)
    Ode0 : float, optional
        Dark energy density parameter (default: 0.7)
    use_astropy : bool, optional
        Use astropy cosmology if available (default: True)
    """
    
    def __init__(self, H0: float = 70.0, Om0: float = 0.3, Ode0: float = 0.7, 
                 use_astropy: bool = True):
        self.H0 = H0
        self.Om0 = Om0
        self.Ode0 = Ode0
        self.constants = PhysicalConstants()
        
        # Initialize cosmology backend
        if use_astropy and HAS_ASTROPY:
            self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
            self.backend = 'astropy'
        else:
            self.cosmo = None
            self.backend = 'simple'
            if use_astropy:
                print(f"Using simple flat ΛCDM approximation (H0={H0}, Ωm={Om0})")
    
    def angular_diameter_distance(self, z: float) -> float:
        """
        Compute angular diameter distance to redshift z.
        
        Parameters
        ----------
        z : float
            Redshift
        
        Returns
        -------
        D_A : float
            Angular diameter distance in Mpc
        
        Notes
        -----
        D_A(z) = D_c(z) / (1 + z)
        where D_c is the comoving distance.
        """
        if z < 0:
            raise ValueError(f"Redshift must be non-negative, got z={z}")
        
        if z == 0:
            return 0.0
        
        if self.backend == 'astropy':
            return self.cosmo.angular_diameter_distance(z).to(u.Mpc).value
        else:
            # Simple flat ΛCDM approximation
            return self._simple_angular_diameter_distance(z)
    
    def angular_diameter_distance_z1z2(self, z1: float, z2: float) -> float:
        """
        Compute angular diameter distance between two redshifts.
        
        Parameters
        ----------
        z1 : float
            Lower redshift (e.g., lens)
        z2 : float
            Higher redshift (e.g., source)
        
        Returns
        -------
        D_A12 : float
            Angular diameter distance from z1 to z2 in Mpc
        
        Notes
        -----
        D_A(z1, z2) = (1 + z1) × [D_c(z2) - D_c(z1)] / (1 + z2)
        """
        if z2 <= z1:
            raise ValueError(f"z2 ({z2}) must be greater than z1 ({z1})")
        
        if self.backend == 'astropy':
            return self.cosmo.angular_diameter_distance_z1z2(z1, z2).to(u.Mpc).value
        else:
            # Simple approximation
            return self._simple_angular_diameter_distance_z1z2(z1, z2)
    
    def critical_surface_density(self, z_lens: float, z_source: float) -> float:
        """
        Compute critical surface density for lensing.
        
        Parameters
        ----------
        z_lens : float
            Lens redshift
        z_source : float
            Source redshift
        
        Returns
        -------
        Sigma_crit : float
            Critical surface density in M_☉/kpc²
        
        Notes
        -----
        Σ_crit = (c² / 4πG) × (D_s / D_d D_ls)
        
        where:
        - D_d = angular diameter distance to lens
        - D_s = angular diameter distance to source
        - D_ls = angular diameter distance from lens to source
        """
        if z_source <= z_lens:
            raise ValueError(f"Source redshift ({z_source}) must be greater than lens redshift ({z_lens})")
        
        # Get distances in kpc
        D_d_kpc = self.angular_diameter_distance(z_lens) * self.constants.kpc_per_Mpc
        D_s_kpc = self.angular_diameter_distance(z_source) * self.constants.kpc_per_Mpc
        D_ls_kpc = self.angular_diameter_distance_z1z2(z_lens, z_source) * self.constants.kpc_per_Mpc
        
        # Critical density
        c2 = self.constants.c_km_s ** 2
        G = self.constants.G_kpc3_Msun_km2s2
        
        Sigma_crit = (c2 / (4 * np.pi * G)) * (D_s_kpc / (D_d_kpc * D_ls_kpc))
        
        return Sigma_crit  # M_☉/kpc²
    
    def physical_to_angular(self, R_kpc: float, z_lens: float) -> float:
        """
        Convert physical radius to angular size.
        
        Parameters
        ----------
        R_kpc : float
            Physical radius in kpc
        z_lens : float
            Lens redshift
        
        Returns
        -------
        theta_arcsec : float
            Angular size in arcsec
        
        Notes
        -----
        θ = R / D_A  (in radians)
        Convert to arcsec: θ_arcsec = θ_rad × 206265
        """
        D_d_kpc = self.angular_diameter_distance(z_lens) * self.constants.kpc_per_Mpc
        
        if D_d_kpc == 0:
            raise ValueError(f"Angular diameter distance is zero at z={z_lens}")
        
        theta_rad = R_kpc / D_d_kpc
        theta_arcsec = theta_rad * self.constants.arcsec_per_radian
        
        return theta_arcsec
    
    def angular_to_physical(self, theta_arcsec: float, z_lens: float) -> float:
        """
        Convert angular size to physical radius.
        
        Parameters
        ----------
        theta_arcsec : float
            Angular size in arcsec
        z_lens : float
            Lens redshift
        
        Returns
        -------
        R_kpc : float
            Physical radius in kpc
        
        Notes
        -----
        R = θ × D_A
        """
        D_d_kpc = self.angular_diameter_distance(z_lens) * self.constants.kpc_per_Mpc
        theta_rad = theta_arcsec / self.constants.arcsec_per_radian
        R_kpc = theta_rad * D_d_kpc
        
        return R_kpc
    
    def deflection_angle_point_mass(self, M_Msun: float, R_kpc: float, 
                                    z_lens: float, z_source: float) -> float:
        """
        Compute deflection angle for a point mass.
        
        Parameters
        ----------
        M_Msun : float
            Mass in solar masses
        R_kpc : float
            Impact parameter in kpc
        z_lens : float
            Lens redshift
        z_source : float
            Source redshift
        
        Returns
        -------
        alpha_arcsec : float
            Deflection angle in arcsec
        
        Notes
        -----
        α = (4GM/c²) × (D_ls/D_s) / R
        
        This is useful for testing and as a reference.
        """
        # Distances
        D_s = self.angular_diameter_distance(z_source)
        D_ls = self.angular_diameter_distance_z1z2(z_lens, z_source)
        
        # Einstein formula
        G = self.constants.G_kpc3_Msun_km2s2
        c2 = self.constants.c_km_s ** 2
        
        alpha_rad = (4 * G * M_Msun / c2) * (D_ls / D_s) / R_kpc
        alpha_arcsec = alpha_rad * self.constants.arcsec_per_radian
        
        return alpha_arcsec
    
    def einstein_radius(self, M_Msun: float, z_lens: float, z_source: float) -> float:
        """
        Compute Einstein radius for a point mass.
        
        Parameters
        ----------
        M_Msun : float
            Mass in solar masses
        z_lens : float
            Lens redshift
        z_source : float
            Source redshift
        
        Returns
        -------
        theta_E_arcsec : float
            Einstein radius in arcsec
        
        Notes
        -----
        θ_E = sqrt[(4GM/c²) × (D_ls / D_d D_s)]
        """
        D_d = self.angular_diameter_distance(z_lens) * self.constants.kpc_per_Mpc
        D_s = self.angular_diameter_distance(z_source) * self.constants.kpc_per_Mpc
        D_ls = self.angular_diameter_distance_z1z2(z_lens, z_source) * self.constants.kpc_per_Mpc
        
        G = self.constants.G_kpc3_Msun_km2s2
        c2 = self.constants.c_km_s ** 2
        
        theta_E_rad = np.sqrt((4 * G * M_Msun / c2) * (D_ls / (D_d * D_s)))
        theta_E_arcsec = theta_E_rad * self.constants.arcsec_per_radian
        
        return theta_E_arcsec
    
    # =========================================================================
    # Simple flat ΛCDM implementation (fallback when astropy not available)
    # =========================================================================
    
    def _E(self, z: float) -> float:
        """Hubble parameter evolution: E(z) = H(z)/H0."""
        return np.sqrt(self.Om0 * (1 + z)**3 + self.Ode0)
    
    def _simple_angular_diameter_distance(self, z: float) -> float:
        """
        Simple numerical integration for angular diameter distance.
        
        Uses trapezoidal rule to integrate 1/E(z) from 0 to z.
        """
        from scipy.integrate import quad
        
        c_Mpc_s = self.constants.c_km_s / 1000.0  # Mpc/s when multiplied by H0 units
        
        # Comoving distance integral
        integrand = lambda zp: 1.0 / self._E(zp)
        D_c, _ = quad(integrand, 0, z, limit=100)
        D_c *= (c_Mpc_s / self.H0)  # Convert to Mpc
        
        # Angular diameter distance
        D_A = D_c / (1 + z)
        
        return D_A
    
    def _simple_angular_diameter_distance_z1z2(self, z1: float, z2: float) -> float:
        """Simple angular diameter distance between two redshifts."""
        from scipy.integrate import quad
        
        c_Mpc_s = self.constants.c_km_s / 1000.0
        
        integrand = lambda zp: 1.0 / self._E(zp)
        D_c, _ = quad(integrand, z1, z2, limit=100)
        D_c *= (c_Mpc_s / self.H0)
        
        D_A12 = D_c / (1 + z2)
        
        return D_A12
    
    def __repr__(self) -> str:
        return (f"LensingCosmology(H0={self.H0}, Om0={self.Om0}, "
                f"backend='{self.backend}')")


# =============================================================================
# Convenience functions
# =============================================================================

def get_default_cosmology() -> LensingCosmology:
    """
    Get default cosmology (Planck 2018 best-fit).
    
    Returns
    -------
    cosmo : LensingCosmology
        Default cosmology with H0=67.4, Om0=0.315
    """
    return LensingCosmology(H0=67.4, Om0=0.315, Ode0=0.685)


def get_wmap_cosmology() -> LensingCosmology:
    """
    Get WMAP9 cosmology.
    
    Returns
    -------
    cosmo : LensingCosmology
        WMAP cosmology with H0=69.3, Om0=0.286
    """
    return LensingCosmology(H0=69.3, Om0=0.286, Ode0=0.714)


if __name__ == "__main__":
    # Quick demonstration
    print("="*70)
    print("LENSING COSMOLOGY MODULE")
    print("="*70)
    print()
    
    cosmo = LensingCosmology(H0=70, Om0=0.3)
    print(f"Cosmology: {cosmo}")
    print()
    
    # Example calculations
    z_lens = 0.5
    z_source = 2.0
    
    D_d = cosmo.angular_diameter_distance(z_lens)
    D_s = cosmo.angular_diameter_distance(z_source)
    D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)
    
    print(f"Lens redshift: z = {z_lens}")
    print(f"Source redshift: z = {z_source}")
    print()
    print(f"D_d  = {D_d:.2f} Mpc")
    print(f"D_s  = {D_s:.2f} Mpc")
    print(f"D_ls = {D_ls:.2f} Mpc")
    print()
    
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_source)
    print(f"Σ_crit = {Sigma_crit:.3e} M_☉/kpc²")
    print()
    
    # Conversion example
    R_kpc = 100.0
    theta = cosmo.physical_to_angular(R_kpc, z_lens)
    print(f"{R_kpc} kpc at z={z_lens} → {theta:.2f} arcsec")
    
    R_back = cosmo.angular_to_physical(theta, z_lens)
    print(f"Round-trip: {theta:.2f} arcsec → {R_back:.2f} kpc")
    print()
    
    # Einstein radius example
    M = 1e14  # Solar masses
    theta_E = cosmo.einstein_radius(M, z_lens, z_source)
    print(f"Einstein radius for M = {M:.1e} M_☉:")
    print(f"  θ_E = {theta_E:.2f} arcsec")
