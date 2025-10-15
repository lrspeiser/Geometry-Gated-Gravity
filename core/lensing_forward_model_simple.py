#!/usr/bin/env python3
"""
Simple Lensing Forward Model for Hierarchical Inference
========================================================

Simplified forward model that connects:
1. Cluster parameters (M_500, R_500, z) → Baryon profiles
2. Kernel parameters (ℓ₀, A_c) → Boosted surface density
3. Surface density → Einstein radius

This is a streamlined version for hierarchical inference.
For full physics, see cluster_kernel_3d_shell.py.

Author: GravityCalculator
Date: 2025-01-19
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import warnings


class SimpleLensingForwardModel:
    """
    Simplified lensing forward model for hierarchical inference.
    
    Workflow:
    1. Build baryon profiles from cluster parameters
    2. Project to surface density Σ(R)
    3. Apply kernel boost: Σ_eff = Σ × (1 + K_Σ)
    4. Compute Einstein radius from κ_eff
    
    Simplifications vs full model:
    - Uses 2D projected kernel (not 3D shell integral)
    - Assumes spherical symmetry (no triaxiality)
    - Simple NFW+gNFW profiles (no detailed gas physics)
    - Mock cosmology for quick evaluation
    
    For publication-quality results, replace with full model.
    """
    
    def __init__(self):
        """Initialize forward model with mock cosmology."""
        # Mock cosmology (Planck 2018)
        self.H0 = 67.4  # km/s/Mpc
        self.Om0 = 0.315
        self.h = self.H0 / 100.0
        
    def critical_surface_density(self, z_lens: float, z_source: float) -> float:
        """
        Compute critical surface density Σ_crit [Msun/kpc²].
        
        Simplified formula (valid for z_l << z_s):
        Σ_crit ≈ (c² / 4πG) × (D_s / (D_l × D_ls))
        
        Parameters
        ----------
        z_lens : float
            Lens redshift
        z_source : float
            Source redshift
        
        Returns
        -------
        Sigma_crit : float
            Critical surface density [Msun/kpc²]
        """
        # Angular diameter distances (Mpc, comoving)
        # Simplified approximation for low-z
        D_l = 3000.0 * z_lens / self.h  # Mpc (comoving)
        D_s = 3000.0 * z_source / self.h
        D_ls = D_s - D_l
        
        # c²/(4πG) in units of Msun/kpc when distances in Mpc
        # (this is a dimensional conversion factor)
        conversion = 1.663e9  # Msun/kpc² per (1/Mpc)
        
        Sigma_crit = conversion * D_s / (D_l * D_ls * (1 + z_lens)**2)
        
        return Sigma_crit
    
    def physical_to_angular(self, R_kpc: float, z: float) -> float:
        """
        Convert physical radius to angular size.
        
        Parameters
        ----------
        R_kpc : float
            Physical radius [kpc]
        z : float
            Redshift
        
        Returns
        -------
        theta : float
            Angular size [arcsec]
        """
        # Angular diameter distance (Mpc, comoving)
        D_A = 3000.0 * z / self.h / (1 + z)  # Mpc (proper)
        
        # Convert kpc to Mpc
        R_Mpc = R_kpc / 1000.0
        
        # Angle in radians
        theta_rad = R_Mpc / D_A
        
        # Convert to arcsec
        theta_arcsec = theta_rad * 206265.0
        
        return theta_arcsec
    
    def build_simple_baryon_profile(
        self,
        r_grid: np.ndarray,
        M_500: float,
        R_500: float,
        fgas: float = 0.11
    ) -> np.ndarray:
        """
        Build simplified baryon density profile.
        
        Uses gNFW for gas + Hernquist for stars.
        
        Parameters
        ----------
        r_grid : ndarray
            Radial grid [kpc]
        M_500 : float
            Total mass [Msun]
        R_500 : float
            R_500 [kpc]
        fgas : float
            Gas fraction
        
        Returns
        -------
        rho_baryon : ndarray
            Total baryon density [Msun/kpc³]
        """
        # Gas: gNFW profile (Arnaud+ 2010)
        # ρ_gas(r) = ρ_0 / [(r/r_c)^γ × (1 + (r/r_s)^α)^((β-γ)/α)]
        r_c = 0.12 * R_500  # Core radius
        r_s = R_500  # Scale radius
        alpha, beta, gamma = 1.05, 5.49, 0.31  # gNFW shape parameters
        
        # Profile shape (not normalized)
        denominator = (r_grid / r_c)**gamma * (1 + (r_grid / r_s)**alpha)**((beta - gamma) / alpha)
        rho_gas_shape = 1.0 / (denominator + 1e-30)
        
        # Normalize to f_gas × M_500
        M_gas_target = fgas * M_500
        integrand_gas = 4 * np.pi * r_grid**2 * rho_gas_shape
        M_gas_integral = trapezoid(integrand_gas, r_grid)
        rho_gas = rho_gas_shape * (M_gas_target / M_gas_integral)
        
        # Stars: Hernquist profile (BCG + ICL combined)
        # ρ_star(r) = M_star / (2π) × a / [r(r+a)³]
        M_star = 0.015 * M_500  # Stellar fraction ~1.5%
        a_star = 0.15 * R_500  # Scale radius
        
        rho_star = M_star / (2 * np.pi) * a_star / (r_grid * (r_grid + a_star)**3 + 1e-30)
        
        # Total baryons
        rho_baryon = rho_gas + rho_star
        
        return rho_baryon
    
    def project_to_surface_density(
        self,
        r_grid: np.ndarray,
        rho_3d: np.ndarray,
        R_grid: np.ndarray
    ) -> np.ndarray:
        """
        Abel projection: 3D density → 2D surface density.
        
        Σ(R) = 2 ∫_R^∞ ρ(r) × r dr / √(r² - R²)
        
        Parameters
        ----------
        r_grid : ndarray
            3D radial grid [kpc]
        rho_3d : ndarray
            3D density [Msun/kpc³]
        R_grid : ndarray
            2D projected grid [kpc]
        
        Returns
        -------
        Sigma : ndarray
            Surface density [Msun/kpc²]
        """
        Sigma = np.zeros_like(R_grid)
        
        for i, R in enumerate(R_grid):
            if R >= r_grid[-1]:
                Sigma[i] = 0.0
                continue
            
            # Integration from R to r_max
            mask = r_grid >= R
            r_int = r_grid[mask]
            rho_int = rho_3d[mask]
            
            # Abel integrand
            integrand = rho_int * r_int / np.sqrt(r_int**2 - R**2 + 1e-30)
            
            # Integrate
            Sigma[i] = 2.0 * trapezoid(integrand, r_int)
        
        return Sigma
    
    def apply_kernel_boost(
        self,
        R_grid: np.ndarray,
        Sigma: np.ndarray,
        ell0: float,
        A_c: float,
        R_500: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply simplified kernel boost.
        
        K_Σ(R) = A_c × W(R; ℓ₀)
        
        where W is a radial window function.
        
        Parameters
        ----------
        R_grid : ndarray
            Projected radii [kpc]
        Sigma : ndarray
            Baseline surface density [Msun/kpc²]
        ell0 : float
            Coherence length [kpc]
        A_c : float
            Coherence amplitude (dimensionless)
        R_500 : float
            Cluster scale [kpc]
        
        Returns
        -------
        Sigma_eff : ndarray
            Boosted surface density [Msun/kpc²]
        K_Sigma : ndarray
            Boost factor (dimensionless)
        """
        # Simple radial window (power-law falloff)
        # W(R) = 1 / [1 + (R/ℓ₀)²]^n_coh
        n_coh = 2.0
        W = 1.0 / (1.0 + (R_grid / ell0)**2)**n_coh
        
        # Boost factor
        K_Sigma = A_c * W
        
        # Large-scale taper (prevent boost beyond R_500)
        taper = 1.0 / (1.0 + (R_grid / (1.2 * R_500))**4)
        K_Sigma *= taper
        
        # Effective surface density
        Sigma_eff = Sigma * (1.0 + K_Sigma)
        
        return Sigma_eff, K_Sigma
    
    def compute_einstein_radius(
        self,
        R_grid: np.ndarray,
        Sigma_eff: np.ndarray,
        Sigma_crit: float
    ) -> float:
        """
        Compute Einstein radius from mean convergence.
        
        Find R where ⟨κ⟩(R) = 1, where:
        ⟨κ⟩(R) = M_2D(<R) / (π R² Σ_crit)
        
        Parameters
        ----------
        R_grid : ndarray
            Projected radii [kpc]
        Sigma_eff : ndarray
            Effective surface density [Msun/kpc²]
        Sigma_crit : float
            Critical surface density [Msun/kpc²]
        
        Returns
        -------
        R_E : float
            Einstein radius [kpc], or 0 if not found
        """
        # Cumulative mass
        area = np.pi * R_grid**2
        darea = np.diff(area)
        Sigma_avg = (Sigma_eff[1:] + Sigma_eff[:-1]) / 2.0
        M_2D_cumulative = np.cumsum(Sigma_avg * darea)
        
        # Mean convergence
        mean_kappa = np.zeros_like(R_grid)
        mean_kappa[1:] = M_2D_cumulative / (area[1:] * Sigma_crit)
        mean_kappa[0] = mean_kappa[1]
        
        # Find where ⟨κ⟩ = 1
        idx = np.where(mean_kappa >= 1.0)[0]
        if idx.size > 0:
            return R_grid[idx[-1]]
        else:
            return 0.0
    
    def predict_einstein_radius(
        self,
        M_500: float,
        R_500: float,
        z_lens: float,
        z_source: float,
        ell0: float,
        A_c: float,
        fgas: float = 0.11
    ) -> Tuple[float, Dict]:
        """
        End-to-end forward model: cluster params + kernel → Einstein radius.
        
        Parameters
        ----------
        M_500 : float
            Total mass [Msun]
        R_500 : float
            R_500 [kpc]
        z_lens : float
            Lens redshift
        z_source : float
            Source redshift
        ell0 : float
            Coherence length [kpc]
        A_c : float
            Coherence amplitude
        fgas : float
            Gas fraction
        
        Returns
        -------
        theta_E_arcsec : float
            Einstein radius [arcsec]
        diagnostics : dict
            Intermediate results
        """
        # 1. Build baryon profile
        r_grid = np.logspace(0, np.log10(3 * R_500), 500)
        rho_baryon = self.build_simple_baryon_profile(r_grid, M_500, R_500, fgas)
        
        # 2. Project to surface density
        R_grid = np.linspace(0, 2 * R_500, 500)
        Sigma = self.project_to_surface_density(r_grid, rho_baryon, R_grid)
        
        # 3. Apply kernel boost
        Sigma_eff, K_Sigma = self.apply_kernel_boost(R_grid, Sigma, ell0, A_c, R_500)
        
        # 4. Critical surface density
        Sigma_crit = self.critical_surface_density(z_lens, z_source)
        
        # 5. Einstein radius
        R_E_kpc = self.compute_einstein_radius(R_grid, Sigma_eff, Sigma_crit)
        theta_E_arcsec = self.physical_to_angular(R_E_kpc, z_lens)
        
        # Diagnostics
        diagnostics = {
            'R_E_kpc': R_E_kpc,
            'Sigma_crit': Sigma_crit,
            'K_Sigma_at_RE': np.interp(R_E_kpc, R_grid, K_Sigma),
            'M_baryon_R500': trapezoid(4 * np.pi * r_grid**2 * rho_baryon, r_grid) / M_500,
            'boost_factor_mean': np.mean(1 + K_Sigma[R_grid < R_500])
        }
        
        return theta_E_arcsec, diagnostics


# Quick test
if __name__ == "__main__":
    print("="*70)
    print("Simple Lensing Forward Model Test")
    print("="*70)
    
    model = SimpleLensingForwardModel()
    
    # MACS0416 parameters
    M_500 = 1.15e15  # Msun
    R_500 = 1200  # kpc
    z_lens = 0.396
    z_source = 2.0
    
    # Test with different kernel parameters
    print("\nTest 1: No boost (A_c = 0)")
    theta_E, diag = model.predict_einstein_radius(
        M_500, R_500, z_lens, z_source,
        ell0=200, A_c=0.0
    )
    print(f"  θ_E = {theta_E:.2f}\" (baseline)")
    
    print("\nTest 2: Moderate boost (A_c = 10, ℓ₀ = 200 kpc)")
    theta_E, diag = model.predict_einstein_radius(
        M_500, R_500, z_lens, z_source,
        ell0=200, A_c=10.0
    )
    print(f"  θ_E = {theta_E:.2f}\"")
    print(f"  R_E = {diag['R_E_kpc']:.0f} kpc")
    print(f"  K_Σ at R_E = {diag['K_Sigma_at_RE']:.3f}")
    
    print("\nTest 3: Mass-scaled ℓ₀ (smaller cluster → smaller ℓ₀)")
    R_500_small = 800  # kpc
    M_500_small = 0.5e15  # Msun
    ell0_scaled = 200 * (R_500_small / 1000.0)**0.5  # γ = 0.5
    theta_E, diag = model.predict_einstein_radius(
        M_500_small, R_500_small, z_lens, z_source,
        ell0=ell0_scaled, A_c=10.0
    )
    print(f"  R_500 = {R_500_small} kpc → ℓ₀ = {ell0_scaled:.1f} kpc")
    print(f"  θ_E = {theta_E:.2f}\"")
    
    print("\n✅ Forward model test complete")
    print("\nNext: Integrate with hierarchical inference")
