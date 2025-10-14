#!/usr/bin/env python3
"""
Physical Gas Profile Models for Clusters
=========================================

Implements realistic ICM models:
- Double-β (Vikhlinin-style with core + extended components)
- gNFW (generalized NFW with universal pressure profile)
- Clumping corrections
- BCG + ICL stellar mass
- Normalization to f_gas priors

Physical motivation:
Single-β profiles from X-ray fits are too steep and underestimate
extended ICM by factor ~3-4. Double-β or gNFW with proper normalization
to f_gas ~ 0.11 ± 0.02 at R_500 provides realistic mass distribution.

Author: Cluster Lensing Physics Upgrade
Date: 2025-01-13
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar


# Physical constants
MU_E = 1.167  # Mean molecular weight per electron
MU_GAS = 0.59  # Mean molecular weight for gas
M_PROTON_KG = 1.6726219e-27  # kg
M_SUN_KG = 1.98847e30  # kg
CM3_PER_KPC3 = (3.085677581e21)**3  # cm^3 per kpc^3


@dataclass
class DoubleBetaParams:
    """Parameters for double-β gas profile."""
    n01: float  # Central density component 1 [cm^-3]
    rc1: float  # Core radius component 1 [kpc]
    beta1: float  # Slope parameter component 1
    n02: float  # Central density component 2 [cm^-3]
    rc2: float  # Core radius component 2 [kpc]
    beta2: float  # Slope parameter component 2


@dataclass
class GNFWParams:
    """Parameters for generalized NFW gas profile."""
    rho0: float  # Central density [Msun/kpc^3]
    rs: float  # Scale radius [kpc]
    alpha: float  # Intermediate slope
    beta: float  # Outer slope
    gamma: float  # Inner slope


def ne_to_rho_gas(ne_cm3: np.ndarray) -> np.ndarray:
    """
    Convert electron density to gas mass density.
    
    Parameters
    ----------
    ne_cm3 : ndarray
        Electron density in cm^-3
    
    Returns
    -------
    rho_gas : ndarray
        Gas mass density in Msun/kpc^3
    """
    # ρ_gas = μ_e × m_p × n_e
    rho_kg_cm3 = ne_cm3 * MU_E * M_PROTON_KG
    # Convert kg/cm^3 to Msun/kpc^3
    rho_msun_kpc3 = rho_kg_cm3 * CM3_PER_KPC3 / M_SUN_KG
    return rho_msun_kpc3


def rho_gas_double_beta(r: np.ndarray, params: DoubleBetaParams) -> np.ndarray:
    """
    Double-β gas density profile (Vikhlinin-style).
    
    n_e(r) = n_01 (1 + (r/rc1)²)^(-3β1/2) + n_02 (1 + (r/rc2)²)^(-3β2/2)
    
    The first component represents the core, the second the extended halo.
    Typical: rc1 ~ 20-80 kpc, rc2 ~ 0.2-0.6 R_500
            beta1 ~ 0.6-0.9, beta2 ~ 1.0-1.5
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc
    params : DoubleBetaParams
        Model parameters
    
    Returns
    -------
    rho_gas : ndarray
        Gas mass density in Msun/kpc^3
    """
    ne1 = params.n01 * (1 + (r/params.rc1)**2)**(-1.5*params.beta1)
    ne2 = params.n02 * (1 + (r/params.rc2)**2)**(-1.5*params.beta2)
    ne_total = ne1 + ne2
    return ne_to_rho_gas(ne_total)


def rho_gas_gnfw(r: np.ndarray, params: GNFWParams) -> np.ndarray:
    """
    Generalized NFW gas density profile.
    
    ρ(r) = ρ₀ / [(r/rs)^γ × (1 + (r/rs)^α)^((β-γ)/α)]
    
    This is flexible enough to match both cusped and cored profiles.
    Universal pressure profile (Arnaud+ 2010) uses similar form.
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc
    params : GNFWParams
        Model parameters
    
    Returns
    -------
    rho_gas : ndarray
        Gas mass density in Msun/kpc^3
    """
    x = r / params.rs
    denominator = x**params.gamma * (1 + x**params.alpha)**((params.beta - params.gamma)/params.alpha)
    rho = params.rho0 / (denominator + 1e-30)  # avoid division by zero
    return rho


def integrate_mass_spherical(r: np.ndarray, rho: np.ndarray) -> float:
    """
    Integrate mass: M = 4π ∫ r² ρ(r) dr
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc (must be monotonic)
    rho : ndarray
        Density in Msun/kpc^3
    
    Returns
    -------
    M_total : float
        Total mass in Msun
    """
    integrand = 4 * np.pi * r**2 * rho
    return trapezoid(integrand, r)


def rescale_to_fgas(r: np.ndarray, rho_gas: np.ndarray, 
                    M_500: float, R_500: float, 
                    fgas_target: float) -> np.ndarray:
    """
    Rescale gas density to match target f_gas at R_500.
    
    This enforces the physical prior: f_gas(<R_500) = M_gas(<R_500) / M_500
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc
    rho_gas : ndarray
        Gas density in Msun/kpc^3
    M_500 : float
        Total mass at R_500 in Msun (from literature)
    R_500 : float
        Radius in kpc corresponding to overdensity 500
    fgas_target : float
        Target gas fraction (typically 0.10-0.12)
    
    Returns
    -------
    rho_gas_scaled : ndarray
        Rescaled gas density
    """
    # Integrate gas mass within R_500
    mask = r <= R_500
    if not np.any(mask):
        return rho_gas
    
    M_gas_current = integrate_mass_spherical(r[mask], rho_gas[mask])
    M_gas_target = fgas_target * M_500
    
    scale_factor = M_gas_target / max(M_gas_current, 1e-30)
    
    return rho_gas * scale_factor


def apply_clumping_correction(r: np.ndarray, rho_gas: np.ndarray,
                              C0: float = 0.3, eta: float = 2.0,
                              R_200: Optional[float] = None) -> np.ndarray:
    """
    Apply clumping correction to gas density.
    
    X-ray observations underestimate density by factor sqrt(C) due to
    unresolved clumping (X-ray ∝ n_e² sees clumps as overdense).
    
    C(r) = 1 + C₀ (r/R_200)^η
    ρ_true = sqrt(C(r)) × ρ_X-ray
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc
    rho_gas : ndarray
        Gas density from X-ray in Msun/kpc^3
    C0 : float
        Clumping amplitude (typically 0.2-0.5)
    eta : float
        Radial power law (typically 1.5-2.5)
    R_200 : float, optional
        Overdensity radius in kpc (if None, no correction applied)
    
    Returns
    -------
    rho_gas_corrected : ndarray
        Corrected gas density
    """
    if R_200 is None or C0 == 0:
        return rho_gas
    
    C_r = 1 + C0 * (r / R_200)**eta
    return rho_gas * np.sqrt(C_r)


def rho_hernquist(r: np.ndarray, M_star: float, a: float) -> np.ndarray:
    """
    Hernquist profile for BCG.
    
    ρ(r) = M/(2π) × a/(r(r+a)³)
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc
    M_star : float
        Total stellar mass in Msun
    a : float
        Scale length in kpc
    
    Returns
    -------
    rho_star : ndarray
        Stellar density in Msun/kpc^3
    """
    return (M_star / (2*np.pi)) * a / (r * (r + a)**3 + 1e-30)


def rho_jaffe(r: np.ndarray, M_star: float, r_j: float) -> np.ndarray:
    """
    Jaffe profile for BCG (alternative to Hernquist, slightly steeper).
    
    ρ(r) = M/(4π r_j³) × 1/[(r/r_j)² (1 + r/r_j)²]
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc
    M_star : float
        Total stellar mass in Msun
    r_j : float
        Scale radius in kpc
    
    Returns
    -------
    rho_star : ndarray
        Stellar density in Msun/kpc^3
    """
    x = r / r_j
    return (M_star / (4*np.pi*r_j**3)) / (x**2 * (1 + x)**2 + 1e-30)


def rho_icl_exponential(r: np.ndarray, M_icl: float, r_s: float) -> np.ndarray:
    """
    Exponential profile for extended ICL.
    
    ρ(r) = M/(8π rs³) × exp(-r/rs)
    
    Normalized so ∫ 4πr² ρ dr = M_icl
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc
    M_icl : float
        Total ICL mass in Msun
    r_s : float
        Scale radius in kpc
    
    Returns
    -------
    rho_icl : ndarray
        ICL density in Msun/kpc^3
    """
    return (M_icl / (8*np.pi*r_s**3)) * np.exp(-r/r_s)


def default_double_beta_params(R_500: float) -> DoubleBetaParams:
    """
    Sensible default parameters for double-β profile.
    
    Parameters
    ----------
    R_500 : float
        Cluster R_500 in kpc
    
    Returns
    -------
    params : DoubleBetaParams
        Default parameters scaled to cluster size
    """
    return DoubleBetaParams(
        n01=4e-4,  # cm^-3, core component (further reduced)
        rc1=100.0,  # kpc, even larger core
        beta1=0.60,  # shallower
        n02=8e-4,  # cm^-3, extended component (DOMINANT now)
        rc2=0.70 * R_500,  # kpc, VERY extended (was 0.50)
        beta2=0.90  # MUCH shallower outer slope (was 1.05)
    )


def default_gnfw_params(R_500: float) -> GNFWParams:
    """
    Sensible default parameters for gNFW profile.
    
    Parameters
    ----------
    R_500 : float
        Cluster R_500 in kpc
    
    Returns
    -------
    params : GNFWParams
        Default parameters
    """
    return GNFWParams(
        rho0=1e8,  # Msun/kpc^3, to be rescaled by f_gas
        rs=0.3 * R_500,
        alpha=1.5,
        beta=4.5,
        gamma=0.5
    )


def build_cluster_density_profile(
    r: np.ndarray,
    M_500: float,
    R_500: float,
    fgas_target: float = 0.11,
    M_bcg: float = 2e12,
    a_bcg: float = 25.0,
    M_icl: float = 8e11,
    rs_icl: float = 150.0,
    C0_clump: float = 0.3,
    eta_clump: float = 2.0,
    R_200: Optional[float] = None,
    use_gnfw: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build complete cluster baryon density profile.
    
    Combines: Gas (double-β or gNFW) + BCG + ICL + clumping
    
    Parameters
    ----------
    r : ndarray
        Radial grid in kpc
    M_500, R_500 : float
        Cluster mass and radius
    fgas_target : float
        Target f_gas at R_500
    M_bcg, a_bcg : float
        BCG mass and scale
    M_icl, rs_icl : float
        ICL mass and scale
    C0_clump, eta_clump : float
        Clumping parameters
    R_200 : float, optional
        For clumping correction
    use_gnfw : bool
        If True, use gNFW; else use double-β
    
    Returns
    -------
    rho_gas, rho_bcg, rho_icl, rho_total : ndarray
        Density components in Msun/kpc^3
    """
    # Gas profile
    if use_gnfw:
        params_gas = default_gnfw_params(R_500)
        rho_gas = rho_gas_gnfw(r, params_gas)
    else:
        params_gas = default_double_beta_params(R_500)
        rho_gas = rho_gas_double_beta(r, params_gas)
    
    # Rescale to target f_gas
    rho_gas = rescale_to_fgas(r, rho_gas, M_500, R_500, fgas_target)
    
    # Apply clumping
    rho_gas = apply_clumping_correction(r, rho_gas, C0_clump, eta_clump, R_200)
    
    # Stellar components
    rho_bcg = rho_jaffe(r, M_bcg, a_bcg)
    rho_icl = rho_icl_exponential(r, M_icl, rs_icl)
    
    # Total
    rho_total = rho_gas + rho_bcg + rho_icl
    
    return rho_gas, rho_bcg, rho_icl, rho_total


if __name__ == '__main__':
    # Quick test
    r = np.logspace(-1, 3.5, 1000)  # 0.1 to 3000 kpc
    
    M_500 = 1.15e15  # Msun
    R_500 = 1200.0  # kpc
    
    rho_gas, rho_bcg, rho_icl, rho_total = build_cluster_density_profile(
        r, M_500, R_500, fgas_target=0.11
    )
    
    # Check f_gas
    mask = r <= R_500
    M_gas = integrate_mass_spherical(r[mask], rho_gas[mask])
    fgas = M_gas / M_500
    
    print(f"Test cluster profile:")
    print(f"  M_500 = {M_500:.2e} Msun")
    print(f"  R_500 = {R_500:.1f} kpc")
    print(f"  M_gas(<R_500) = {M_gas:.2e} Msun")
    print(f"  f_gas = {fgas:.3f} (target: 0.110)")
    print(f"  ✓ Profile built successfully!")
