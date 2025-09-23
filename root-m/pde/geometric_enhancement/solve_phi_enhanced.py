#!/usr/bin/env python3
"""
solve_phi_enhanced.py

Enhanced PDE solver with geometric enhancement based on baryon density gradients.
This module attempts to enhance gravity in regions of low baryon density while
preserving the core baryon-only approach.

The key innovation is a geometric enhancement factor that scales with:
1. Relative density gradients (faster density falloff → stronger enhancement)
2. Absolute density thresholds (lower density → stronger enhancement) 
3. Radial position (optional radial scaling)

This avoids explicit dark matter while producing effective gravity that can
match both cluster (extended) and galaxy (compact) observations.
"""

import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings

# Constants
G = 4.301e-6  # km^2/s^2 * kpc / Msun
c_km_s = 299792.458  # km/s


@dataclass
class EnhancedSolverParams:
    """Parameters for the enhanced PDE solver."""
    # Base coupling strength
    S0_base: float = 1.4e-4  # Base source strength
    rc_kpc: float = 22.0  # Core radius for density suppression
    g0_kms2_per_kpc: float = 1200.0  # Reference acceleration scale
    
    # Geometric enhancement parameters
    use_geometric_enhancement: bool = True
    lambda0: float = 0.5  # Enhancement strength (0=none, 1=max)
    alpha_grad: float = 1.5  # Gradient sensitivity exponent
    rho_crit_Msun_kpc3: float = 1e6  # Critical density below which enhancement activates
    r_enhance_kpc: float = 50.0  # Radial scale for enhancement activation
    
    # Mobility saturation (optional)
    use_saturating_mobility: bool = False
    gsat_kms2_per_kpc: float = 2500.0  # Saturation acceleration scale
    n_sat: float = 2.0  # Saturation exponent
    
    # Numerical parameters
    max_iter: int = 2000
    rtol: float = 1e-8
    verbose: bool = True


def compute_geometric_enhancement(R: np.ndarray, Z: np.ndarray, 
                                 rho: np.ndarray,
                                 params: EnhancedSolverParams) -> np.ndarray:
    """
    Compute the geometric enhancement factor based on density gradients.
    
    The enhancement factor Lambda(R,Z) is designed to:
    - Be ~1 in high-density regions (no enhancement)
    - Increase in low-density regions with steep gradients
    - Provide smooth transitions
    
    Returns:
        Lambda: 2D array of enhancement factors >= 1.0
    """
    NZ, NR = rho.shape
    dR = R[1] - R[0] if len(R) > 1 else 1.0
    dZ = Z[1] - Z[0] if len(Z) > 1 else 1.0
    
    # Create 2D coordinate grids
    R2D = np.broadcast_to(R.reshape(1, -1), (NZ, NR))
    Z2D = np.broadcast_to(Z.reshape(-1, 1), (NZ, NR))
    r_sph = np.sqrt(R2D**2 + Z2D**2)
    
    # Compute density gradients
    grad_rho_R = np.gradient(rho, axis=1) / dR
    grad_rho_Z = np.gradient(rho, axis=0) / dZ
    grad_mag = np.sqrt(grad_rho_R**2 + grad_rho_Z**2)
    
    # Relative gradient (dimensionless)
    rel_grad = grad_mag / np.maximum(rho, params.rho_crit_Msun_kpc3 * 0.01)
    
    # Density suppression factor (0 to 1)
    rho_supp = np.exp(-rho / params.rho_crit_Msun_kpc3)
    
    # Radial enhancement factor (optional)
    if params.r_enhance_kpc > 0:
        radial_factor = 1.0 + (r_sph / params.r_enhance_kpc)**2
    else:
        radial_factor = 1.0
    
    # Combined enhancement
    # Lambda = 1 + lambda0 * (relative_gradient)^alpha * density_suppression * radial_factor
    Lambda = 1.0 + params.lambda0 * (rel_grad**params.alpha_grad) * rho_supp * radial_factor
    
    # Smooth and bound the enhancement
    Lambda = np.minimum(Lambda, 10.0)  # Cap maximum enhancement
    Lambda = np.maximum(Lambda, 1.0)   # Ensure >= 1
    
    # Apply smoothing to avoid numerical noise
    from scipy.ndimage import gaussian_filter
    Lambda = gaussian_filter(Lambda, sigma=1.0, mode='nearest')
    
    return Lambda


def solve_axisym_enhanced(R: np.ndarray, Z: np.ndarray,
                          rho: np.ndarray,
                          params: EnhancedSolverParams,
                          phi_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the enhanced PDE for the scalar field phi with geometric enhancement.
    
    The PDE is:
        ∇·[D(|∇φ|²) ∇φ] = -S(ρ) * Lambda(ρ,∇ρ)
    
    where:
        D(|∇φ|²) = mobility function (possibly saturating)
        S(ρ) = source term from baryon density
        Lambda = geometric enhancement factor
    
    Args:
        R, Z: coordinate arrays
        rho: baryon density in Msun/kpc^3
        params: solver parameters
        phi_init: initial guess for phi
        
    Returns:
        phi: scalar field solution
        gR, gZ: acceleration components -∇φ in km/s^2/kpc
    """
    
    NZ, NR = len(Z), len(R)
    dR = R[1] - R[0] if NR > 1 else 1.0
    dZ = Z[1] - Z[0] if NZ > 1 else 1.0
    
    # Ensure rho has correct shape
    if rho.shape != (NZ, NR):
        raise ValueError(f"rho shape {rho.shape} doesn't match grid ({NZ}, {NR})")
    
    # Compute geometric enhancement if enabled
    if params.use_geometric_enhancement:
        Lambda = compute_geometric_enhancement(R, Z, rho, params)
        if params.verbose:
            print(f"  Geometric enhancement: min={np.min(Lambda):.2f}, max={np.max(Lambda):.2f}, mean={np.mean(Lambda):.2f}")
    else:
        Lambda = np.ones_like(rho)
    
    # Build the source term with enhancement
    # S = S0 * rho / (1 + rho/rho_c) with rho_c from rc
    rho_c = params.g0_kms2_per_kpc / (4 * np.pi * G * params.rc_kpc**2)
    # Scale S0 to get reasonable phi values (should give accelerations ~ g0)
    # The natural scale is: S0 ~ g0 / R_max where R_max is the domain size
    R_scale = np.max(R) 
    S0_scaled = params.S0_base * params.g0_kms2_per_kpc / R_scale
    source = S0_scaled * rho / (1.0 + rho / rho_c)
    source = source * Lambda  # Apply enhancement
    
    if params.verbose:
        print(f"  Source term: max={np.max(source):.2e}, mean={np.mean(source):.2e}")
    
    # Handle cylindrical singularity at R=0
    R2D = np.broadcast_to(R.reshape(1, -1), (NZ, NR))
    R_reg = np.maximum(R2D, dR * 0.1)
    
    # Initialize phi
    if phi_init is not None:
        phi = phi_init.copy()
    else:
        phi = np.zeros((NZ, NR))
    
    # Iterative solver (simple Jacobi relaxation for nonlinear PDE)
    omega = 0.8  # relaxation parameter (increased for faster convergence)
    for iteration in range(params.max_iter):
        phi_old = phi.copy()
        
        # Simple update using 5-point stencil
        phi_new = np.zeros_like(phi)
        
        for iz in range(1, NZ-1):
            for ir in range(1, NR-1):
                # Central differences
                phi_R_plus = phi[iz, ir+1] if ir < NR-1 else 0.0
                phi_R_minus = phi[iz, ir-1] if ir > 0 else 0.0
                phi_Z_plus = phi[iz+1, ir] if iz < NZ-1 else 0.0
                phi_Z_minus = phi[iz-1, ir] if iz > 0 else 0.0
                
                # Cylindrical Laplacian 
                r = max(R[ir], dR * 0.1)
                laplacian = (phi_R_plus - 2*phi[iz,ir] + phi_R_minus) / dR**2
                laplacian += (phi_R_plus - phi_R_minus) / (2*dR*r)  # 1/r dφ/dr term
                laplacian += (phi_Z_plus - 2*phi[iz,ir] + phi_Z_minus) / dZ**2
                
                # Update with source term
                phi_new[iz, ir] = phi[iz, ir] + omega * (laplacian + source[iz, ir])
        
        # Apply boundary conditions
        phi_new[0, :] = 0.0   # Bottom
        phi_new[-1, :] = 0.0  # Top
        phi_new[:, 0] = 0.0   # Left (axis)
        phi_new[:, -1] = 0.0  # Right
        
        phi = phi_new
        
        # Check convergence
        residual = np.max(np.abs(phi - phi_old)) / (np.max(np.abs(phi)) + 1e-12)
        if iteration % 100 == 0 and params.verbose:
            print(f"  Iteration {iteration}: residual={residual:.2e}, max|phi|={np.max(np.abs(phi)):.2e}")
        if residual < params.rtol:
            if params.verbose:
                print(f"  Converged at iteration {iteration}, residual={residual:.2e}")
            break
    
    # Compute final accelerations
    gR = -np.gradient(phi, axis=1) / dR
    gZ = -np.gradient(phi, axis=0) / dZ
    
    # Apply enhancement to accelerations if using field-level enhancement
    # (This is an alternative approach where enhancement modifies the field gradient)
    # gR = gR * np.sqrt(Lambda)
    # gZ = gZ * np.sqrt(Lambda)
    
    return phi, gR, gZ


def predict_temperature_hse(r: np.ndarray, ne: np.ndarray, g_tot: np.ndarray,
                           f_nt: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Predict temperature from HSE given density and total acceleration.
    
    kT(r) = ∫_r^∞ μ m_p g_tot / (1 - f_nt) * (r'/r)² * (ne(r)/ne(r')) dr'
    
    Args:
        r: radius array in kpc
        ne: electron density in cm^-3
        g_tot: total acceleration in km/s^2/kpc
        f_nt: non-thermal pressure fraction (optional)
        
    Returns:
        kT: temperature in keV
    """
    # Constants
    mu = 0.61  # mean molecular weight
    m_p_keV = 938272.0  # proton mass in keV/c^2
    c_km_s = 299792.458
    kpc_to_cm = 3.086e21
    
    # Default f_nt
    if f_nt is None:
        f_nt = np.zeros_like(r)
    
    # Initialize temperature array
    kT = np.zeros_like(r)
    
    # Integrate from outside in
    for i in range(len(r)-2, -1, -1):
        dr = r[i+1] - r[i]
        
        # Average values in interval
        g_avg = 0.5 * (g_tot[i] + g_tot[i+1])
        fnt_avg = 0.5 * (f_nt[i] + f_nt[i+1])
        ne_ratio = ne[i] / (ne[i+1] + 1e-30)
        r_ratio = r[i+1] / (r[i] + 1e-10)
        
        # Temperature gradient from HSE
        dT = (mu * m_p_keV / c_km_s**2) * g_avg * dr * kpc_to_cm / (1 - fnt_avg)
        
        # Update temperature with geometric factors
        kT[i] = kT[i+1] * ne_ratio * r_ratio**2 + dT
    
    return np.maximum(kT, 0.0)