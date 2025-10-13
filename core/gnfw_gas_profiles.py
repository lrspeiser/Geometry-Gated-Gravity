#!/usr/bin/env python3
"""
Universal gNFW Gas Profile (Arnaud+ 2010)
==========================================

Implements the universal pressure profile from Arnaud+ 2010 (A&A 517, A92):
- Empirically calibrated from REXCESS X-ray cluster sample
- Minimal intrinsic scatter (~15%)
- Self-similar scaling with M_500
- Proper normalization to f_gas ~ 0.11

This is the physics-based replacement for ad-hoc double-β profiles.

Key features:
- Universal shape parameters: (P₀, c₅₀₀, γ, α, β) from Arnaud+ 2010
- Self-similar pressure scaling: P ∝ M_500^(2/3) × E(z)^(8/3)
- Hydrostatic normalization to f_gas(R_500) = 0.11 ± 0.01
- Proper handling of pressure → density → mass conversion

Physical motivation:
Standard X-ray-derived single-β profiles underestimate extended ICM.
The universal profile captures both core and outskirts correctly.

References:
- Arnaud+ 2010, A&A 517, A92 (universal profile)
- Planck 2013, A&A 550, A131 (validation + updates)

Author: GravityCalculator Physics Upgrade
Date: 2025-01-13
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar

# Physical constants
K_BOLTZMANN_CGS = 1.380649e-16  # erg/K
MU_E = 1.167  # Mean molecular weight per electron (fully ionized primordial gas)
MU_GAS = 0.59  # Mean molecular weight for gas
M_PROTON_KG = 1.6726219e-27  # kg
M_PROTON_CGS = 1.6726219e-24  # g
M_SUN_KG = 1.98847e30  # kg
M_SUN_CGS = 1.98847e33  # g
CM3_PER_KPC3 = (3.085677581e21)**3  # cm^3 per kpc^3
KEV_TO_K = 1.16045e7  # K per keV

# Cosmology (Planck 2018)
H0_KM_S_MPC = 67.66  # km/s/Mpc
H0_S_INV = H0_KM_S_MPC * 1e3 / (3.085677581e22)  # s^-1
OMEGA_M = 0.3111
OMEGA_L = 0.6889

# Critical density at z=0 (in Msun/kpc^3)
RHO_CRIT_0_CGS = (3 * H0_S_INV**2) / (8 * np.pi * 6.67430e-8)  # g/cm^3
RHO_CRIT_0 = RHO_CRIT_0_CGS * CM3_PER_KPC3 / M_SUN_CGS  # Msun/kpc^3


@dataclass
class ArnaudParams:
    """Universal pressure profile parameters from Arnaud+ 2010."""
    P0: float  # Normalization
    c500: float  # Concentration parameter
    gamma: float  # Inner slope
    alpha: float  # Intermediate steepness
    beta: float  # Outer slope


def arnaud_universal_params() -> ArnaudParams:
    """
    Return universal pressure profile parameters from Arnaud+ 2010.
    
    These are calibrated from REXCESS sample (N=33 local clusters).
    Minimal intrinsic scatter: ~15% at 0.5 R_500.
    
    Returns
    -------
    params : ArnaudParams
        Universal parameters (P₀, c₅₀₀, γ, α, β)
    """
    return ArnaudParams(
        P0=8.403,  # h₇₀^(-3/2) (dimensionless normalization)
        c500=1.177,  # Concentration
        gamma=0.3081,  # Inner slope
        alpha=1.0510,  # Intermediate steepness
        beta=5.4905  # Outer slope
    )


def hubble_parameter(z: float) -> float:
    """
    Hubble parameter E(z) = H(z)/H₀.
    
    E(z)² = Ω_m(1+z)³ + Ω_Λ
    
    Parameters
    ----------
    z : float
        Redshift
    
    Returns
    -------
    E_z : float
        E(z) = H(z)/H₀
    """
    return np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_L)


def gnfw_pressure_profile(
    r: np.ndarray,
    R_500: float,
    M_500: float,
    z: float,
    params: Optional[ArnaudParams] = None,
    h70: float = 0.967  # H₀/(70 km/s/Mpc) for Planck 2018
) -> np.ndarray:
    """
    Universal pressure profile from Arnaud+ 2010.
    
    P(x) = P₀ × P_500 × h(x)
    
    where:
      x = r / R_500
      h(x) = 1 / [(c₅₀₀×x)^γ × (1 + (c₅₀₀×x)^α)^((β-γ)/α)]
      P_500 = 1.65e-3 × h(z)^(8/3) × [M_500/(3e14 h₇₀^-1 Msun)]^(2/3) keV cm^-3
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc
    R_500 : float
        R_500 in kpc
    M_500 : float
        M_500 in Msun
    z : float
        Redshift
    params : ArnaudParams, optional
        Profile parameters (default: universal from Arnaud+ 2010)
    h70 : float
        H₀ / (70 km/s/Mpc), for unit conversions
    
    Returns
    -------
    P : ndarray
        Pressure in keV/cm³
    """
    if params is None:
        params = arnaud_universal_params()
    
    # Dimensionless radius
    x = r / R_500
    
    # Profile shape function
    cx = params.c500 * x
    h_x = 1.0 / (cx**params.gamma * (1 + cx**params.alpha)**((params.beta - params.gamma) / params.alpha))
    
    # Self-similar pressure normalization
    Ez = hubble_parameter(z)
    M_pivot = 3e14 / h70  # Msun (pivot mass in h₇₀^-1 Msun units)
    
    P_500 = 1.65e-3 * Ez**(8/3) * (M_500 / M_pivot)**(2/3)  # keV cm^-3
    
    # Full pressure profile
    P = params.P0 * h70**(3/2) * P_500 * h_x
    
    return P


def pressure_to_ne(P_keV_cm3: np.ndarray, T_keV: float) -> np.ndarray:
    """
    Convert pressure to electron number density assuming ideal gas.
    
    P = n_e k_B T  (in CGS: erg/cm³)
    
    Parameters
    ----------
    P_keV_cm3 : ndarray
        Pressure in keV/cm³
    T_keV : float
        Temperature in keV (assumed constant for simplicity, or profile)
    
    Returns
    -------
    n_e : ndarray
        Electron number density in cm^-3
    """
    # Convert keV/cm³ to erg/cm³
    P_erg_cm3 = P_keV_cm3 * 1.60218e-9  # 1 keV = 1.60218e-9 erg
    
    # Temperature in K
    T_K = T_keV * KEV_TO_K
    
    # n_e = P / (k_B T)
    n_e = P_erg_cm3 / (K_BOLTZMANN_CGS * T_K)
    
    return n_e


def ne_to_rho_gas(ne_cm3: np.ndarray) -> np.ndarray:
    """
    Convert electron density to gas mass density.
    
    ρ_gas = μ_e × m_p × n_e
    
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
    rho_cgs = ne_cm3 * MU_E * M_PROTON_CGS  # g/cm³
    
    # Convert g/cm³ to Msun/kpc³
    rho_msun_kpc3 = rho_cgs * CM3_PER_KPC3 / M_SUN_CGS
    
    return rho_msun_kpc3


def rho_gas_from_pressure(
    r: np.ndarray,
    R_500: float,
    M_500: float,
    z: float,
    T_keV: Optional[float] = None,
    params: Optional[ArnaudParams] = None
) -> np.ndarray:
    """
    Compute gas density from universal pressure profile.
    
    Steps:
    1. Compute P(r) from Arnaud+ 2010
    2. Assume temperature profile T(r) (or constant T)
    3. Convert P → n_e via ideal gas law
    4. Convert n_e → ρ_gas
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc
    R_500 : float
        R_500 in kpc
    M_500 : float
        M_500 in Msun
    z : float
        Redshift
    T_keV : float, optional
        Temperature in keV (if None, estimated from M-T relation)
    params : ArnaudParams, optional
        Profile parameters
    
    Returns
    -------
    rho_gas : ndarray
        Gas mass density in Msun/kpc^3
    """
    # Estimate temperature if not provided (M-T relation)
    if T_keV is None:
        # Approximate M-T relation: T ≈ 5 keV × (M_500 / 1e14 Msun)^0.6
        T_keV = 5.0 * (M_500 / 1e14)**0.6
    
    # Compute pressure profile
    P_keV_cm3 = gnfw_pressure_profile(r, R_500, M_500, z, params)
    
    # Convert pressure to electron density
    n_e = pressure_to_ne(P_keV_cm3, T_keV)
    
    # Convert electron density to gas mass density
    rho_gas = ne_to_rho_gas(n_e)
    
    return rho_gas


def integrate_gas_mass(r: np.ndarray, rho_gas: np.ndarray, R_max: float) -> float:
    """
    Integrate gas mass within R_max.
    
    M_gas = 4π ∫₀^R_max r² ρ_gas(r) dr
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc
    rho_gas : ndarray
        Gas density in Msun/kpc³
    R_max : float
        Maximum radius in kpc
    
    Returns
    -------
    M_gas : float
        Gas mass in Msun
    """
    mask = r <= R_max
    integrand = 4 * np.pi * r[mask]**2 * rho_gas[mask]
    return trapezoid(integrand, r[mask])


def normalize_to_fgas(
    r: np.ndarray,
    rho_gas: np.ndarray,
    R_500: float,
    M_500: float,
    fgas_target: float = 0.11
) -> np.ndarray:
    """
    Rescale gas density to match target f_gas at R_500.
    
    This enforces physical prior from X-ray and SZ observations:
    f_gas(<R_500) = 0.11 ± 0.01 (cosmic baryon fraction)
    
    Parameters
    ----------
    r : ndarray
        Radii in kpc
    rho_gas : ndarray
        Gas density in Msun/kpc³
    R_500 : float
        R_500 in kpc
    M_500 : float
        M_500 in Msun
    fgas_target : float
        Target gas fraction (default: 0.11)
    
    Returns
    -------
    rho_gas_scaled : ndarray
        Rescaled gas density
    """
    # Integrate current gas mass
    M_gas_current = integrate_gas_mass(r, rho_gas, R_500)
    
    # Target gas mass
    M_gas_target = fgas_target * M_500
    
    # Scale factor
    scale = M_gas_target / max(M_gas_current, 1e-30)
    
    return rho_gas * scale


def build_gnfw_gas_profile(
    r: np.ndarray,
    R_500: float,
    M_500: float,
    z: float = 0.2,
    fgas_target: float = 0.11,
    T_keV: Optional[float] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Build gNFW gas profile with Arnaud+ 2010 universal pressure profile.
    
    This is the primary interface for cluster gas modeling.
    
    Parameters
    ----------
    r : ndarray
        Radial grid in kpc
    R_500 : float
        R_500 in kpc
    M_500 : float
        M_500 in Msun
    z : float
        Redshift (default: 0.2)
    fgas_target : float
        Target f_gas at R_500 (default: 0.11)
    T_keV : float, optional
        Temperature in keV (if None, estimated from M-T relation)
    verbose : bool
        Print diagnostics
    
    Returns
    -------
    rho_gas : ndarray
        Gas density in Msun/kpc³, normalized to f_gas target
    info : dict
        Diagnostic information (M_gas, f_gas, scale_factor, etc.)
    """
    # Universal parameters
    params = arnaud_universal_params()
    
    # Estimate temperature if needed
    if T_keV is None:
        T_keV = 5.0 * (M_500 / 1e14)**0.6
    
    # Compute raw density from pressure profile
    rho_gas_raw = rho_gas_from_pressure(r, R_500, M_500, z, T_keV, params)
    
    # Check raw f_gas
    M_gas_raw = integrate_gas_mass(r, rho_gas_raw, R_500)
    fgas_raw = M_gas_raw / M_500
    
    # Normalize to target f_gas
    rho_gas_norm = normalize_to_fgas(r, rho_gas_raw, R_500, M_500, fgas_target)
    
    # Verify normalized f_gas
    M_gas_norm = integrate_gas_mass(r, rho_gas_norm, R_500)
    fgas_norm = M_gas_norm / M_500
    
    # Diagnostics
    info = {
        'M_500': M_500,
        'R_500': R_500,
        'z': z,
        'T_keV': T_keV,
        'fgas_target': fgas_target,
        'M_gas_raw': M_gas_raw,
        'fgas_raw': fgas_raw,
        'scale_factor': fgas_target / fgas_raw,
        'M_gas_normalized': M_gas_norm,
        'fgas_normalized': fgas_norm,
        'params': params
    }
    
    if verbose:
        print(f"gNFW Gas Profile (Arnaud+ 2010):")
        print(f"  M_500 = {M_500:.2e} Msun")
        print(f"  R_500 = {R_500:.1f} kpc")
        print(f"  z = {z:.3f}")
        print(f"  T = {T_keV:.1f} keV")
        print(f"  f_gas (raw) = {fgas_raw:.4f}")
        print(f"  f_gas (normalized) = {fgas_norm:.4f} (target: {fgas_target:.3f})")
        print(f"  Scale factor = {info['scale_factor']:.3f}")
    
    return rho_gas_norm, info


if __name__ == '__main__':
    # Test on MACS0416-like cluster
    print("=" * 60)
    print("Testing gNFW Universal Pressure Profile (Arnaud+ 2010)")
    print("=" * 60)
    print()
    
    # Radial grid
    r = np.logspace(-1, 3.5, 2000)  # 0.1 to ~3000 kpc
    
    # MACS0416 parameters (from literature)
    M_500 = 1.15e15  # Msun (Jauzac+ 2015)
    R_500 = 1200.0  # kpc
    z = 0.396
    
    # Build profile
    rho_gas, info = build_gnfw_gas_profile(
        r, R_500, M_500, z, fgas_target=0.11, verbose=True
    )
    
    # Additional diagnostics at key radii
    print()
    print("Density at key radii:")
    radii_test = [10, 50, 100, 180, 500, R_500]
    for r_test in radii_test:
        idx = np.argmin(np.abs(r - r_test))
        print(f"  ρ_gas({r_test:.0f} kpc) = {rho_gas[idx]:.2e} Msun/kpc³")
    
    # Surface density at Einstein radius (~180 kpc for MACS0416)
    R_E = 180.0  # kpc
    # Project to surface density (simple Abel integral approximation)
    mask_proj = r >= R_E
    integrand = rho_gas[mask_proj] * r[mask_proj] / np.sqrt(r[mask_proj]**2 - R_E**2)
    Sigma_E = 2 * trapezoid(integrand, r[mask_proj])
    
    print()
    print(f"Surface density at R_E = {R_E:.0f} kpc:")
    print(f"  Σ_gas(R_E) = {Sigma_E:.2e} Msun/kpc²")
    print()
    print("✓ gNFW profile test complete!")
