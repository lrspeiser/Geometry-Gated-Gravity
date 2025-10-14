#!/usr/bin/env python3
"""
Physically-Calibrated Cluster Baryon Model Builder
===================================================

Constructs 3D baryon density profiles for galaxy clusters using:
1. gNFW gas profiles (Arnaud+ 2010) normalized to f_gas(R_500) = 0.11
2. BCG stellar profile (de Vaucouleurs, r_eff ~ 25 kpc)
3. ICL (Intracluster Light) - Sersic profile (r_s ~ 150 kpc)
4. Radial clumping correction (Simionescu+ 2011)

This is the standard baryon model for blind cluster validation.

Physical Motivation:
- Universal gNFW captures both core and outskirts ICM correctly
- BCG mass ~1-3e12 Msun from typical L* scaling
- ICL mass ~0.5-1.5e12 Msun from deep photometry
- Total f_baryon ~ 0.11-0.12 at R_500

Key Design Principles:
- NO free parameters per cluster
- Templates scaled by observed M_500, z, and photometry priors
- Clumping correction physically motivated, not tuned

Author: GravityCalculator
Date: 2025-01-14
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from scipy.integrate import trapezoid
import sys
import os

# Add core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from gnfw_gas_profiles import build_gnfw_gas_profile, M_SUN_CGS


@dataclass
class BaryonComponents:
    """Container for 3D baryon density components."""
    r: np.ndarray  # Radial grid [kpc]
    rho_gas: np.ndarray  # Gas density [Msun/kpc^3]
    rho_bcg: np.ndarray  # BCG stellar density [Msun/kpc^3]
    rho_icl: np.ndarray  # ICL density [Msun/kpc^3]
    rho_total: np.ndarray  # Total baryon density [Msun/kpc^3]
    clumping_factor: np.ndarray  # C(r) clumping correction
    info: Dict  # Diagnostic information


@dataclass
class ClusterBaryonParams:
    """Parameters for baryon model construction."""
    # Required cluster properties
    M_500: float  # Total mass [Msun]
    R_500: float  # R_500 [kpc]
    z: float  # Redshift
    
    # Gas parameters
    fgas_target: float = 0.11  # Target f_gas at R_500
    T_keV: Optional[float] = None  # Temperature [keV], auto-estimated if None
    
    # BCG parameters
    M_BCG: Optional[float] = None  # BCG mass [Msun], auto-scaled if None
    r_eff_BCG: float = 25.0  # Effective radius [kpc]
    
    # ICL parameters  
    M_ICL: Optional[float] = None  # ICL mass [Msun], auto-scaled if None
    r_s_ICL: float = 150.0  # Scale radius [kpc]
    n_sersic_ICL: float = 1.5  # Sersic index
    
    # Clumping parameters (Simionescu+ 2011)
    C0: float = 1.3  # Clumping at core
    eta: float = 2.0  # Radial exponent (C ~ r^eta)
    C_max: float = 2.5  # Maximum clumping in outskirts


def bcg_de_vaucouleurs(r: np.ndarray, M_BCG: float, r_eff: float) -> np.ndarray:
    """
    BCG stellar profile using de Vaucouleurs (n=4 Sersic).
    
    ρ(r) = ρ₀ exp{-7.67[(r/r_eff)^(1/4) - 1]}
    
    Normalized such that ∫4πr²ρ dr = M_BCG
    
    Parameters
    ----------
    r : ndarray
        Radii [kpc]
    M_BCG : float
        Total BCG stellar mass [Msun]
    r_eff : float
        Effective radius [kpc]
    
    Returns
    -------
    rho_bcg : ndarray
        BCG density [Msun/kpc^3]
    """
    b_n = 7.67  # de Vaucouleurs constant (n=4 Sersic)
    
    # Profile shape
    x = (r / r_eff)**0.25
    profile = np.exp(-b_n * (x - 1))
    
    # Normalize to total mass
    integrand = 4 * np.pi * r**2 * profile
    M_integral = trapezoid(integrand, r)
    rho_0 = M_BCG / M_integral
    
    return rho_0 * profile


def icl_sersic(
    r: np.ndarray,
    M_ICL: float,
    r_s: float,
    n: float = 1.5
) -> np.ndarray:
    """
    ICL (Intracluster Light) profile using Sersic.
    
    ρ(r) = ρ₀ exp{-b_n[(r/r_s)^(1/n) - 1]}
    
    Parameters
    ----------
    r : ndarray
        Radii [kpc]
    M_ICL : float
        Total ICL mass [Msun]
    r_s : float
        Scale radius [kpc]
    n : float
        Sersic index (default: 1.5, between exponential and de Vauc)
    
    Returns
    -------
    rho_icl : ndarray
        ICL density [Msun/kpc^3]
    """
    # Sersic constant (approximate for n=1.5)
    b_n = 1.82 * n - 0.33
    
    # Profile shape
    x = (r / r_s)**(1.0 / n)
    profile = np.exp(-b_n * (x - 1))
    
    # Normalize to total mass
    integrand = 4 * np.pi * r**2 * profile
    M_integral = trapezoid(integrand, r)
    rho_0 = M_ICL / M_integral
    
    return rho_0 * profile


def clumping_profile(
    r: np.ndarray,
    R_500: float,
    C0: float = 1.3,
    eta: float = 2.0,
    C_max: float = 2.5
) -> np.ndarray:
    """
    Radial clumping correction from Simionescu+ 2011.
    
    C(r) = C₀ + (C_max - C₀) × (r/R_500)^η
    
    Clumping increases with radius due to:
    - Substructure in outskirts
    - Incomplete mixing
    - Projection effects
    
    Parameters
    ----------
    r : ndarray
        Radii [kpc]
    R_500 : float
        R_500 [kpc]
    C0 : float
        Core clumping (default: 1.3)
    eta : float
        Radial exponent (default: 2.0)
    C_max : float
        Maximum clumping at R_500 (default: 2.5)
    
    Returns
    -------
    C : ndarray
        Clumping factor C(r)
    """
    x = r / R_500
    C = C0 + (C_max - C0) * x**eta
    C = np.clip(C, C0, C_max)
    return C


def estimate_bcg_mass(M_500: float) -> float:
    """
    Estimate BCG stellar mass from cluster mass.
    
    Scaling from Gonzalez+ 2013:
    M_BCG ~ 2e12 Msun × (M_500 / 1e15 Msun)^0.4
    
    Parameters
    ----------
    M_500 : float
        Cluster M_500 [Msun]
    
    Returns
    -------
    M_BCG : float
        Estimated BCG mass [Msun]
    """
    M_pivot = 1e15
    M_BCG_pivot = 2e12
    alpha = 0.4
    
    return M_BCG_pivot * (M_500 / M_pivot)**alpha


def estimate_icl_mass(M_500: float) -> float:
    """
    Estimate ICL mass from cluster mass.
    
    ICL typically ~40% of BCG mass (Morishita+ 2017; Zhang+ 2019).
    Some scatter: 30-60% depending on dynamical state.
    
    Parameters
    ----------
    M_500 : float
        Cluster M_500 [Msun]
    
    Returns
    -------
    M_ICL : float
        Estimated ICL mass [Msun]
    """
    M_BCG = estimate_bcg_mass(M_500)
    return 0.4 * M_BCG


def build_cluster_baryon_model(
    r: np.ndarray,
    params: ClusterBaryonParams,
    apply_clumping: bool = True,
    verbose: bool = False
) -> BaryonComponents:
    """
    Build complete 3D baryon density model for cluster.
    
    This is the main interface for cluster baryon construction.
    
    Parameters
    ----------
    r : ndarray
        Radial grid [kpc]
    params : ClusterBaryonParams
        Cluster parameters and model settings
    apply_clumping : bool
        Apply gas clumping correction (default: True)
    verbose : bool
        Print diagnostics
    
    Returns
    -------
    components : BaryonComponents
        Full baryon model with all components
    """
    # 1. Gas profile (gNFW Arnaud+ 2010)
    rho_gas, gas_info = build_gnfw_gas_profile(
        r=r,
        R_500=params.R_500,
        M_500=params.M_500,
        z=params.z,
        fgas_target=params.fgas_target,
        T_keV=params.T_keV,
        verbose=verbose
    )
    
    # 2. BCG stellar profile
    M_BCG = params.M_BCG if params.M_BCG is not None else estimate_bcg_mass(params.M_500)
    rho_bcg = bcg_de_vaucouleurs(r, M_BCG, params.r_eff_BCG)
    
    # 3. ICL profile
    M_ICL = params.M_ICL if params.M_ICL is not None else estimate_icl_mass(params.M_500)
    rho_icl = icl_sersic(r, M_ICL, params.r_s_ICL, params.n_sersic_ICL)
    
    # 4. Clumping correction
    if apply_clumping:
        C_factor = clumping_profile(r, params.R_500, params.C0, params.eta, params.C_max)
        # Apply to gas: n_e -> sqrt(C) × n_e  =>  rho_gas -> C × rho_gas (squared)
        # Actually: clumping affects density inference, so rho_gas_corrected = rho_gas / C
        # Wait, convention matters. Simionescu: C = <n_e^2>/<n_e>^2
        # So measured n_e is biased high by sqrt(C), true n_e = measured / sqrt(C)
        # Thus rho_gas_true = rho_gas_measured / sqrt(C)
        # Let's use the standard convention: apply sqrt(C) to ne, which squares for mass
        rho_gas_corrected = rho_gas / np.sqrt(C_factor)
    else:
        C_factor = np.ones_like(r)
        rho_gas_corrected = rho_gas
    
    # 5. Total baryons
    rho_total = rho_gas_corrected + rho_bcg + rho_icl
    
    # 6. Compute total masses at R_500
    idx_R500 = np.argmin(np.abs(r - params.R_500))
    mask_R500 = r <= params.R_500
    
    M_gas_R500 = trapezoid(4 * np.pi * r[mask_R500]**2 * rho_gas_corrected[mask_R500], r[mask_R500])
    M_bcg_total = trapezoid(4 * np.pi * r**2 * rho_bcg, r)
    M_icl_total = trapezoid(4 * np.pi * r**2 * rho_icl, r)
    M_baryon_R500 = M_gas_R500 + M_bcg_total + M_icl_total
    
    fgas_R500 = M_gas_R500 / params.M_500
    fbaryon_R500 = M_baryon_R500 / params.M_500
    
    # 7. Diagnostic info
    info = {
        'M_500': params.M_500,
        'R_500': params.R_500,
        'z': params.z,
        'M_gas_R500': M_gas_R500,
        'M_BCG': M_BCG,
        'M_ICL': M_ICL,
        'M_baryon_R500': M_baryon_R500,
        'fgas_R500': fgas_R500,
        'fbaryon_R500': fbaryon_R500,
        'apply_clumping': apply_clumping,
        'gas_info': gas_info,
        'r_eff_BCG': params.r_eff_BCG,
        'r_s_ICL': params.r_s_ICL
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Cluster Baryon Model Summary")
        print(f"{'='*60}")
        print(f"M_500 = {params.M_500:.2e} Msun")
        print(f"R_500 = {params.R_500:.1f} kpc")
        print(f"z = {params.z:.3f}")
        print(f"\nComponent Masses:")
        print(f"  M_gas(<R_500) = {M_gas_R500:.2e} Msun")
        print(f"  M_BCG = {M_BCG:.2e} Msun")
        print(f"  M_ICL = {M_ICL:.2e} Msun")
        print(f"  M_baryon(<R_500) = {M_baryon_R500:.2e} Msun")
        print(f"\nBaryon Fractions:")
        print(f"  f_gas(R_500) = {fgas_R500:.4f}")
        print(f"  f_baryon(R_500) = {fbaryon_R500:.4f}")
        print(f"\nClumping: {'Applied' if apply_clumping else 'Off'}")
        if apply_clumping:
            print(f"  C(core) = {C_factor[0]:.2f}")
            print(f"  C(R_500) = {C_factor[idx_R500]:.2f}")
        print(f"{'='*60}\n")
    
    return BaryonComponents(
        r=r,
        rho_gas=rho_gas_corrected,
        rho_bcg=rho_bcg,
        rho_icl=rho_icl,
        rho_total=rho_total,
        clumping_factor=C_factor,
        info=info
    )


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Physically-Calibrated Cluster Baryon Model")
    print("=" * 70)
    print()
    
    # Radial grid
    r = np.logspace(-1, 3.5, 2000)  # 0.1 to ~3000 kpc
    
    # Test on MACS0416
    print("Test Case: MACS0416")
    print("-" * 70)
    params = ClusterBaryonParams(
        M_500=1.15e15,
        R_500=1200.0,
        z=0.396,
        fgas_target=0.11,
        T_keV=10.5
    )
    
    components = build_cluster_baryon_model(r, params, verbose=True)
    
    # Check densities at key radii
    print("Density checks at key radii:")
    print(f"{'Radius [kpc]':<15} {'ρ_gas':<15} {'ρ_BCG':<15} {'ρ_ICL':<15} {'ρ_total':<15}")
    print("-" * 75)
    
    radii_test = [10, 50, 100, 180, 500, 1000]
    for r_test in radii_test:
        idx = np.argmin(np.abs(r - r_test))
        print(f"{r_test:<15.0f} "
              f"{components.rho_gas[idx]:<15.2e} "
              f"{components.rho_bcg[idx]:<15.2e} "
              f"{components.rho_icl[idx]:<15.2e} "
              f"{components.rho_total[idx]:<15.2e}")
    
    print()
    print("✓ Baryon model builder test complete!")
    print()
