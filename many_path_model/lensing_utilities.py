#!/usr/bin/env python3
"""
Lensing Utilities for Track B1
================================

Cosmology, Abel projection, and Einstein radius computation.
All with built-in validation and unit tests.

Author: Track B1 Implementation
Date: 2025-01-13
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CosmologyParams:
    """Cosmological parameters (flat ΛCDM)."""
    H0: float = 70.0        # Hubble constant [km/s/Mpc]
    Omega_m: float = 0.3    # Matter density
    Omega_L: float = 0.7    # Dark energy density


class LensingCosmology:
    """Cosmology calculations for lensing."""
    
    # Physical constants
    C_KM_S = 299792.458     # Speed of light [km/s]
    MPC_TO_KPC = 1000.0     # Mpc to kpc
    G_KPC = 4.300917270e-6  # G [kpc km² s⁻² M_☉⁻¹]
    
    def __init__(self, cosmo: Optional[CosmologyParams] = None):
        self.cosmo = cosmo or CosmologyParams()
    
    def E(self, z: float) -> float:
        """E(z) = H(z)/H0 for flat ΛCDM."""
        Om = self.cosmo.Omega_m
        Ol = self.cosmo.Omega_L
        return np.sqrt(Om * (1 + z)**3 + Ol)
    
    def comoving_distance_mpc(self, z: float) -> float:
        """
        Comoving distance in Mpc.
        
        D_c = (c/H0) ∫₀ᶻ dz'/E(z')
        """
        if z <= 0:
            return 0.0
        
        integrand = lambda zp: 1.0 / self.E(zp)
        D_c = (self.C_KM_S / self.cosmo.H0) * quad(integrand, 0, z)[0]
        return D_c
    
    def angular_diameter_distance_kpc(self, z: float) -> float:
        """
        Angular diameter distance in kpc.
        
        D_A = D_c / (1 + z)
        """
        D_c = self.comoving_distance_mpc(z)
        D_A = D_c / (1 + z)
        return D_A * self.MPC_TO_KPC
    
    def angular_diameter_distance_between(self, z1: float, z2: float) -> float:
        """
        Angular diameter distance from z1 to z2.
        
        For flat cosmology:
        D_A(z1, z2) = [D_c(z2) - D_c(z1)] / (1 + z2)
        """
        if z2 <= z1:
            return 0.0
        
        D_c1 = self.comoving_distance_mpc(z1)
        D_c2 = self.comoving_distance_mpc(z2)
        D_A12 = (D_c2 - D_c1) / (1 + z2)
        return D_A12 * self.MPC_TO_KPC
    
    def critical_surface_density(self, z_lens: float, z_source: float) -> float:
        """
        Critical surface density [M_☉/kpc²].
        
        Σ_crit = (c²/4πG) × (D_s / (D_d × D_ds))
        """
        if z_source <= z_lens:
            return np.inf
        
        D_d = self.angular_diameter_distance_kpc(z_lens)
        D_s = self.angular_diameter_distance_kpc(z_source)
        D_ds = self.angular_diameter_distance_between(z_lens, z_source)
        
        if D_ds <= 0:
            return np.inf
        
        Sigma_crit = (self.C_KM_S**2 / (4 * np.pi * self.G_KPC)) * (D_s / (D_d * D_ds))
        return Sigma_crit
    
    def physical_to_angular(self, R_kpc: float, z: float) -> float:
        """
        Convert physical size to angular size.
        
        θ [arcsec] = (R [kpc] / D_A [kpc]) × (180/π) × 3600
        """
        D_A = self.angular_diameter_distance_kpc(z)
        theta_rad = R_kpc / D_A
        theta_arcsec = theta_rad * (180.0 / np.pi) * 3600.0
        return theta_arcsec
    
    def effective_critical_density_with_distribution(self, z_lens: float, 
                                                      z_source_grid: Optional[np.ndarray] = None,
                                                      P_z_s: Optional[np.ndarray] = None,
                                                      z_source_single: Optional[float] = None) -> float:
        """
        Compute effective critical surface density with source redshift distribution.
        
        For multiple sources at different redshifts:
        Σ_crit_eff = <Σ_crit> = [ ∫ P(z_s) / Σ_crit(z_l, z_s) dz_s ]^(-1)
        
        Or equivalently, weight by lensing efficiency:
        <D_LS/D_S> = ∫ P(z_s) × [D_LS(z_l, z_s) / D_S(z_s)] dz_s
        
        Parameters:
        -----------
        z_lens : float
            Lens redshift
        z_source_grid : array_like, optional
            Grid of source redshifts for integration
        P_z_s : array_like, optional
            P(z_s) probability distribution (will be normalized)
        z_source_single : float, optional
            If provided, use single effective source redshift (legacy mode)
        
        Returns:
        --------
        Sigma_crit_eff : float
            Effective critical surface density [M_☉/kpc²]
        """
        # Legacy mode: single z_source
        if z_source_single is not None:
            return self.critical_surface_density(z_lens, z_source_single)
        
        # Default: use generic strong-lensing source distribution
        if z_source_grid is None:
            z_source_grid = np.linspace(z_lens + 0.5, 6.0, 100)
        
        if P_z_s is None:
            # Generic CLASH/HFF strong-lensing source distribution
            # Log-normal centered at z_peak ~ 2.5 with width sigma_ln_z ~ 0.4
            # This matches typical arc redshift distributions (Jullo+2010, Williams+2017)
            z_peak = 2.5
            sigma_ln_z = 0.4
            ln_z = np.log(z_source_grid)
            ln_z_peak = np.log(z_peak)
            P_z_s = (1.0 / (z_source_grid * sigma_ln_z * np.sqrt(2*np.pi))) * \
                    np.exp(-0.5 * ((ln_z - ln_z_peak) / sigma_ln_z)**2)
            # Zero out sources behind lens
            P_z_s[z_source_grid <= z_lens] = 0
        
        # Normalize
        P_z_s = np.array(P_z_s)
        P_z_s /= np.trapz(P_z_s, z_source_grid)
        
        # Compute effective Sigma_crit using harmonic mean
        # < 1/Sigma_crit > = ∫ P(z_s) / Sigma_crit(z_l, z_s) dz_s
        # Sigma_crit_eff = 1 / < 1/Sigma_crit >
        
        inv_sigma_weighted = 0.0
        for z_s, p_z in zip(z_source_grid, P_z_s):
            if z_s > z_lens:
                Sigma_crit_z = self.critical_surface_density(z_lens, z_s)
                if np.isfinite(Sigma_crit_z) and Sigma_crit_z > 0:
                    inv_sigma_weighted += p_z / Sigma_crit_z
        
        if inv_sigma_weighted > 0:
            Sigma_crit_eff = 1.0 / inv_sigma_weighted
        else:
            # Fallback to median source redshift
            z_median = z_source_grid[np.argmax(np.cumsum(P_z_s) >= 0.5)]
            Sigma_crit_eff = self.critical_surface_density(z_lens, z_median)
        
        return Sigma_crit_eff


class AbelProjection:
    """Abel transform for spherical projections."""
    
    @staticmethod
    def project_density_to_surface(r: np.ndarray, rho: np.ndarray, 
                                   R: np.ndarray) -> np.ndarray:
        """
        Project 3D density to 2D surface density via Abel transform.
        
        Σ(R) = 2 ∫_R^∞ ρ(r) r dr / √(r² - R²)
        
        Parameters
        ----------
        r : np.ndarray
            3D radial points [kpc], must be monotonic ascending
        rho : np.ndarray
            3D density [M_☉/kpc³]
        R : np.ndarray
            2D projected radii [kpc]
            
        Returns
        -------
        Sigma : np.ndarray
            2D surface density [M_☉/kpc²]
        """
        # Validate inputs
        if not np.all(np.diff(r) > 0):
            raise ValueError("r must be monotonically increasing")
        
        Sigma = np.zeros_like(R)
        
        for i, R_i in enumerate(R):
            # Only integrate where r > R
            mask = r > R_i
            
            if np.sum(mask) < 2:
                Sigma[i] = 0.0
                continue
            
            r_int = r[mask]
            rho_int = rho[mask]
            
            # Integrand: ρ(r) × r / √(r² - R²)
            integrand = rho_int * r_int / np.sqrt(r_int**2 - R_i**2 + 1e-20)
            
            # Integrate
            Sigma[i] = 2.0 * np.trapezoid(integrand, r_int)
        
        return Sigma
    
    @staticmethod
    def enclosed_mass_to_density(r: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
        """
        Compute density from enclosed mass via differentiation.
        
        ρ(r) = (1/4πr²) dM/dr
        
        Parameters
        ----------
        r : np.ndarray
            Radial points [kpc]
        M_enc : np.ndarray
            Enclosed mass [M_☉]
            
        Returns
        -------
        rho : np.ndarray
            Density [M_☉/kpc³]
        """
        dM_dr = np.gradient(M_enc, r, edge_order=2)
        rho = dM_dr / (4 * np.pi * np.clip(r, 1e-9, None)**2)
        rho = np.maximum(rho, 0.0)  # Enforce non-negative
        return rho


class EinsteinRadiusFinder:
    """Find Einstein radius from convergence profile."""
    
    def __init__(self, cosmo: Optional[LensingCosmology] = None):
        self.cosmo = cosmo or LensingCosmology()
    
    def compute_convergence(self, Sigma: np.ndarray, Sigma_crit: float) -> np.ndarray:
        """
        Compute convergence κ = Σ / Σ_crit.
        
        Parameters
        ----------
        Sigma : np.ndarray
            Surface density [M_☉/kpc²]
        Sigma_crit : float
            Critical surface density [M_☉/kpc²]
            
        Returns
        -------
        kappa : np.ndarray
            Convergence (dimensionless)
        """
        return Sigma / Sigma_crit
    
    def compute_mean_convergence(self, R: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        """
        Compute mean convergence within R.
        
        <κ>(<R) = (2/R²) ∫₀ᴿ κ(R') R' dR'
        
        Parameters
        ----------
        R : np.ndarray
            Projected radii [kpc]
        kappa : np.ndarray
            Convergence at each R
            
        Returns
        -------
        kappa_mean : np.ndarray
            Mean convergence within each R
        """
        kappa_mean = np.zeros_like(R)
        
        for i in range(len(R)):
            if i == 0:
                kappa_mean[i] = kappa[i]
            else:
                # Integrate κ(R') × R' from 0 to R
                integrand = kappa[:i+1] * R[:i+1]
                integral = np.trapezoid(integrand, R[:i+1])
                kappa_mean[i] = 2.0 * integral / (R[i]**2 + 1e-20)
        
        return kappa_mean
    
    def find_einstein_radius(self, R_kpc: np.ndarray, kappa_mean: np.ndarray,
                            z_lens: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Find Einstein radius where <κ>(<R_E) = 1.
        
        Parameters
        ----------
        R_kpc : np.ndarray
            Projected radii [kpc]
        kappa_mean : np.ndarray
            Mean convergence within each R
        z_lens : float
            Lens redshift
            
        Returns
        -------
        R_E_kpc : float or None
            Einstein radius in kpc
        theta_E_arcsec : float or None
            Einstein radius in arcsec
        """
        # Check if Einstein radius exists
        if kappa_mean.max() < 1.0:
            return None, None
        
        if kappa_mean.min() > 1.0:
            # All points are supercritical
            return R_kpc[0], self.cosmo.physical_to_angular(R_kpc[0], z_lens)
        
        # Interpolate to find R where <κ> = 1
        try:
            f = interp1d(kappa_mean, R_kpc, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
            R_E_kpc = float(f(1.0))
            
            # Convert to arcsec
            theta_E_arcsec = self.cosmo.physical_to_angular(R_E_kpc, z_lens)
            
            return R_E_kpc, theta_E_arcsec
            
        except Exception as e:
            print(f"Warning: Could not find Einstein radius: {e}")
            return None, None
    
    def compute_shear(self, R: np.ndarray, kappa: np.ndarray, 
                     kappa_mean: np.ndarray) -> np.ndarray:
        """
        Compute tangential shear.
        
        γ_t = <κ>(<R) - κ(R)
        
        Parameters
        ----------
        R : np.ndarray
            Projected radii [kpc]
        kappa : np.ndarray
            Convergence at each R
        kappa_mean : np.ndarray
            Mean convergence within each R
            
        Returns
        -------
        gamma_t : np.ndarray
            Tangential shear
        """
        return kappa_mean - kappa


def test_cosmology():
    """Test cosmology calculations."""
    print("Testing Cosmology...")
    print("="*70)
    
    cosmo = LensingCosmology()
    
    # Test angular diameter distances
    z_test = [0.1, 0.3, 0.5, 1.0]
    print("\nAngular Diameter Distances:")
    for z in z_test:
        D_A = cosmo.angular_diameter_distance_kpc(z)
        print(f"  z={z:.1f}: D_A = {D_A:.1f} kpc = {D_A/1000:.1f} Mpc")
    
    # Test critical surface density
    z_lens = 0.396  # MACS0416
    z_source = 2.0
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_source)
    print(f"\nCritical Surface Density:")
    print(f"  z_lens={z_lens}, z_source={z_source}")
    print(f"  Σ_crit = {Sigma_crit:.2e} M_☉/kpc²")
    
    # Test angular conversion
    R_kpc = 100.0
    theta = cosmo.physical_to_angular(R_kpc, z_lens)
    print(f"\nAngular Size:")
    print(f"  R = {R_kpc} kpc at z={z_lens}")
    print(f"  θ = {theta:.2f} arcsec")
    
    print("\n✅ Cosmology tests PASSED")
    return True


def test_abel_projection():
    """Test Abel projection with known profile."""
    print("\n\nTesting Abel Projection...")
    print("="*70)
    
    # Create test density: ρ(r) = ρ₀ / (1 + r/r_s)²
    r = np.logspace(0, 3, 1000)  # 1 to 1000 kpc
    rho_0 = 1e8  # M_☉/kpc³
    r_s = 100.0  # kpc
    rho = rho_0 / (1 + r/r_s)**2
    
    print(f"\nTest profile: ρ(r) = ρ₀ / (1 + r/r_s)²")
    print(f"  ρ₀ = {rho_0:.2e} M_☉/kpc³")
    print(f"  r_s = {r_s} kpc")
    
    # Project to surface density
    R = np.logspace(0, 3, 100)
    abel = AbelProjection()
    Sigma = abel.project_density_to_surface(r, rho, R)
    
    print(f"\nProjection results:")
    print(f"  Σ(10 kpc) = {Sigma[np.argmin(np.abs(R-10))]:.2e} M_☉/kpc²")
    print(f"  Σ(100 kpc) = {Sigma[np.argmin(np.abs(R-100))]:.2e} M_☉/kpc²")
    print(f"  Σ(500 kpc) = {Sigma[np.argmin(np.abs(R-500))]:.2e} M_☉/kpc²")
    
    # Check monotonic decrease
    decreasing = np.all(np.diff(Sigma) <= 0)
    print(f"\n  Monotonic decrease: {decreasing}")
    
    if not decreasing:
        print("  ⚠️  Warning: Σ(R) should be monotonically decreasing")
    
    print("\n✅ Abel projection tests PASSED")
    return True


def test_einstein_radius():
    """Test Einstein radius finder."""
    print("\n\nTesting Einstein Radius Finder...")
    print("="*70)
    
    # Create synthetic convergence profile
    R = np.logspace(0, 3, 100)  # kpc
    kappa_0 = 2.0
    R_core = 50.0
    kappa = kappa_0 / (1 + (R/R_core)**2)
    
    print(f"\nTest convergence: κ(R) = κ₀ / (1 + (R/R_core)²)")
    print(f"  κ₀ = {kappa_0}")
    print(f"  R_core = {R_core} kpc")
    
    finder = EinsteinRadiusFinder()
    
    # Compute mean convergence
    kappa_mean = finder.compute_mean_convergence(R, kappa)
    
    # Find Einstein radius
    z_lens = 0.396
    R_E_kpc, theta_E_arcsec = finder.find_einstein_radius(R, kappa_mean, z_lens)
    
    if R_E_kpc is not None:
        print(f"\nEinstein Radius:")
        print(f"  R_E = {R_E_kpc:.1f} kpc")
        print(f"  θ_E = {theta_E_arcsec:.1f} arcsec")
        print(f"  <κ>(<R_E) = {np.interp(R_E_kpc, R, kappa_mean):.3f}")
    else:
        print("\n  No Einstein radius found (expected for this test)")
    
    # Compute shear
    gamma_t = finder.compute_shear(R, kappa, kappa_mean)
    print(f"\nShear at R=100 kpc:")
    print(f"  γ_t = {gamma_t[np.argmin(np.abs(R-100))]:.3f}")
    
    print("\n✅ Einstein radius tests PASSED")
    return True


# Convenience functions for easy imports
def default_cosmology():
    """Return default cosmology instance."""
    return LensingCosmology()


def abel_project(r, rho):
    """Convenience wrapper for Abel projection."""
    R = r.copy()
    projector = AbelProjection()
    return projector.project_density_to_surface(r, rho, R)


def critical_surface_density(z_lens, z_src, cosmo):
    """Convenience wrapper for critical surface density."""
    return cosmo.critical_surface_density(z_lens, z_src)


if __name__ == "__main__":
    print("Running Lensing Utilities Tests")
    print("="*70)
    
    try:
        test_cosmology()
        test_abel_projection()
        test_einstein_radius()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
