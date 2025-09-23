#!/usr/bin/env python3
"""
solve_phi_v2.py

Improved PDE solver for the scalar field with geometric enhancement.
This version includes:
- Better numerical methods (SOR instead of Jacobi)
- Proper source term calibration
- Adaptive relaxation parameter
- Better boundary conditions
"""

import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve, cg
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings

# Constants
G = 4.301e-6  # km^2/s^2 * kpc / Msun
c_km_s = 299792.458  # km/s


@dataclass
class EnhancedSolverParams:
    """Parameters for the enhanced PDE solver."""
    # Physical parameters
    gamma: float = 0.5  # Coupling strength (dimensionless)
    beta: float = 1.0   # Length scale parameter
    
    # Geometric enhancement parameters
    use_geometric_enhancement: bool = True
    lambda0: float = 0.5  # Enhancement strength (0=none, 1=max)
    alpha_grad: float = 1.5  # Gradient sensitivity exponent
    rho_crit_factor: float = 0.01  # Enhancement activates when rho < rho_crit_factor * rho_max
    r_enhance_kpc: float = 50.0  # Radial scale for enhancement activation
    
    # Numerical parameters
    max_iter: int = 10000
    rtol: float = 1e-6
    omega_init: float = 1.5  # Initial SOR relaxation parameter
    adaptive_omega: bool = True
    verbose: bool = True
    smooth_source: bool = True  # Apply smoothing to source term
    

def compute_geometric_enhancement_v2(R: np.ndarray, Z: np.ndarray, 
                                    rho: np.ndarray,
                                    params: EnhancedSolverParams) -> np.ndarray:
    """
    Compute the geometric enhancement factor based on density gradients.
    Version 2 with improved scaling.
    """
    NZ, NR = rho.shape
    dR = R[1] - R[0] if len(R) > 1 else 1.0
    dZ = Z[1] - Z[0] if len(Z) > 1 else 1.0
    
    # Create 2D coordinate grids
    R2D = np.broadcast_to(R.reshape(1, -1), (NZ, NR))
    Z2D = np.broadcast_to(Z.reshape(-1, 1), (NZ, NR))
    r_sph = np.sqrt(R2D**2 + Z2D**2)
    
    # Compute density gradients (with smoothing to reduce noise)
    rho_smooth = gaussian_filter(rho, sigma=1.0, mode='nearest')
    grad_rho_R = np.gradient(rho_smooth, axis=1) / dR
    grad_rho_Z = np.gradient(rho_smooth, axis=0) / dZ
    grad_mag = np.sqrt(grad_rho_R**2 + grad_rho_Z**2)
    
    # Dynamic critical density based on current maximum
    rho_max = np.max(rho)
    rho_crit = params.rho_crit_factor * rho_max
    
    # Relative gradient (dimensionless)
    rel_grad = grad_mag / np.maximum(rho, rho_crit * 0.1)
    
    # Density suppression factor (0 to 1)
    # More gradual transition
    rho_norm = rho / rho_crit
    rho_supp = np.exp(-rho_norm) * (1.0 - np.tanh(rho_norm - 1.0))/2.0
    
    # Radial enhancement factor (optional)
    if params.r_enhance_kpc > 0:
        radial_factor = 1.0 + 0.5 * np.tanh((r_sph - params.r_enhance_kpc) / params.r_enhance_kpc)
    else:
        radial_factor = 1.0
    
    # Combined enhancement with bounded growth
    Lambda = 1.0 + params.lambda0 * np.tanh(rel_grad**params.alpha_grad) * rho_supp * radial_factor
    
    # Apply smoothing to avoid numerical noise
    Lambda = gaussian_filter(Lambda, sigma=1.5, mode='nearest')
    
    # Ensure bounds
    Lambda = np.minimum(Lambda, 5.0)  # More conservative cap
    Lambda = np.maximum(Lambda, 1.0)
    
    return Lambda


def calibrate_source_strength(rho: np.ndarray, R: np.ndarray, 
                             target_g0: float = 1e-3) -> float:
    """
    Calibrate the source strength to produce reasonable accelerations.
    
    Args:
        rho: density array
        R: radius array
        target_g0: target acceleration scale in km/s^2/kpc
    
    Returns:
        S0: calibrated source strength
    """
    # Estimate the characteristic density and length scales
    rho_char = np.median(rho[rho > 0])
    R_char = np.median(R[R > 0])
    
    # For Poisson-like equation: ∇²φ ~ S0 * rho
    # Dimensional analysis: φ/R² ~ S0 * rho
    # For acceleration g ~ φ/R: g ~ S0 * rho * R
    # So S0 ~ g / (rho * R)
    
    S0 = target_g0 / (rho_char * R_char)
    
    return S0


def solve_axisym_sor(R: np.ndarray, Z: np.ndarray,
                     rho: np.ndarray,
                     params: EnhancedSolverParams,
                     phi_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the enhanced PDE using Successive Over-Relaxation (SOR).
    
    More efficient than Jacobi for elliptic PDEs.
    """
    
    NZ, NR = len(Z), len(R)
    dR = R[1] - R[0] if NR > 1 else 1.0
    dZ = Z[1] - Z[0] if NZ > 1 else 1.0
    
    # Ensure rho has correct shape
    if rho.shape != (NZ, NR):
        raise ValueError(f"rho shape {rho.shape} doesn't match grid ({NZ}, {NR})")
    
    # Calibrate source strength
    rho_scale = np.max(rho)
    R_scale = np.max(R)
    
    # Use dimensionless variables internally
    rho_norm = rho / rho_scale
    
    # Compute geometric enhancement if enabled
    if params.use_geometric_enhancement:
        Lambda = compute_geometric_enhancement_v2(R, Z, rho, params)
        if params.verbose:
            print(f"  Geometric enhancement: min={np.min(Lambda):.2f}, max={np.max(Lambda):.2f}, mean={np.mean(Lambda):.2f}")
    else:
        Lambda = np.ones_like(rho)
    
    # Build the source term
    # Calibrated coupling: S = gamma * sqrt(G) * rho
    S0 = params.gamma * np.sqrt(G)
    source = S0 * rho_norm * Lambda
    
    # Apply smoothing to source if requested
    if params.smooth_source:
        source = gaussian_filter(source, sigma=0.5, mode='nearest')
    
    if params.verbose:
        source_scale = np.max(np.abs(source))
        print(f"  Source scale: {source_scale:.2e}")
    
    # Initialize phi
    if phi_init is not None:
        phi = phi_init.copy()
    else:
        # Better initial guess based on source
        phi = np.zeros((NZ, NR))
        # Set interior to average expected value
        phi_est = np.mean(source) * min(R_scale, 100.0)**2 / 4.0
        phi[1:-1, 1:-1] = phi_est
    
    # SOR iteration
    omega = params.omega_init
    converged = False
    
    for iteration in range(params.max_iter):
        phi_old = phi.copy()
        max_update = 0.0
        
        # SOR sweep
        for iz in range(1, NZ-1):
            for ir in range(1, NR-1):
                # Get neighboring values
                phi_R_plus = phi[iz, ir+1]
                phi_R_minus = phi[iz, ir-1]
                phi_Z_plus = phi[iz+1, ir]
                phi_Z_minus = phi[iz-1, ir]
                
                # Cylindrical coordinates: include 1/r d/dr term
                r = max(R[ir], dR * 0.1)
                
                # Discretized Laplacian in cylindrical coordinates
                laplacian_R = (phi_R_plus - 2*phi[iz,ir] + phi_R_minus) / dR**2
                laplacian_R += (phi_R_plus - phi_R_minus) / (2*dR*r)
                laplacian_Z = (phi_Z_plus - 2*phi[iz,ir] + phi_Z_minus) / dZ**2
                
                # Gauss-Seidel update
                phi_gs = (laplacian_R + laplacian_Z + source[iz, ir]) / (2/dR**2 + 2/dZ**2)
                phi_gs *= (2/dR**2 + 2/dZ**2)
                phi_gs = (phi_R_plus/dR**2 + phi_R_minus/dR**2 + 
                         phi_Z_plus/dZ**2 + phi_Z_minus/dZ**2 +
                         (phi_R_plus - phi_R_minus)/(2*dR*r) + source[iz, ir])
                phi_gs /= (2/dR**2 + 2/dZ**2)
                
                # SOR update
                phi_new = (1 - omega) * phi[iz, ir] + omega * phi_gs
                max_update = max(max_update, abs(phi_new - phi[iz, ir]))
                phi[iz, ir] = phi_new
        
        # Apply boundary conditions (Dirichlet: phi=0)
        phi[0, :] = 0.0   # Bottom
        phi[-1, :] = 0.0  # Top
        phi[:, 0] = 0.0   # Axis
        phi[:, -1] = 0.0  # Right
        
        # Check convergence
        residual = np.max(np.abs(phi - phi_old)) / (np.max(np.abs(phi)) + 1e-12)
        
        # Adaptive omega (Chebyshev acceleration)
        if params.adaptive_omega and iteration > 10:
            if iteration == 11:
                # Estimate spectral radius
                rho_jacobi = residual  # Approximation
                omega_opt = 2.0 / (1.0 + np.sqrt(1.0 - rho_jacobi**2))
                omega = min(max(omega_opt, 1.0), 1.95)
                if params.verbose:
                    print(f"  Adjusted omega to {omega:.3f}")
        
        if iteration % 100 == 0 and params.verbose:
            print(f"  Iteration {iteration}: residual={residual:.2e}, max|phi|={np.max(np.abs(phi)):.2e}")
        
        if residual < params.rtol:
            converged = True
            if params.verbose:
                print(f"  Converged at iteration {iteration}, residual={residual:.2e}")
            break
    
    if not converged and params.verbose:
        print(f"  Warning: Did not converge after {params.max_iter} iterations")
    
    # Rescale phi back to physical units
    phi = phi * rho_scale * R_scale**2
    
    # Compute accelerations
    gR = -np.gradient(phi, axis=1) / dR
    gZ = -np.gradient(phi, axis=0) / dZ
    
    return phi, gR, gZ


def solve_with_multigrid(R: np.ndarray, Z: np.ndarray,
                        rho: np.ndarray,
                        params: EnhancedSolverParams,
                        n_levels: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve using a simple multigrid approach for faster convergence.
    """
    # Start with coarsest grid
    NZ, NR = len(Z), len(R)
    
    # Create grid hierarchy
    grids = []
    sources = []
    
    # Coarsen
    rho_current = rho.copy()
    R_current = R.copy()
    Z_current = Z.copy()
    
    for level in range(n_levels):
        grids.append((Z_current, R_current))
        sources.append(rho_current)
        
        if level < n_levels - 1:
            # Coarsen by factor of 2
            NZ_coarse = (len(Z_current) + 1) // 2
            NR_coarse = (len(R_current) + 1) // 2
            
            Z_current = Z_current[::2]
            R_current = R_current[::2]
            
            # Restrict density (simple averaging)
            rho_coarse = np.zeros((NZ_coarse, NR_coarse))
            for iz in range(NZ_coarse):
                for ir in range(NR_coarse):
                    iz_fine = min(2*iz, len(sources[-1])-1)
                    ir_fine = min(2*ir, len(sources[-1][0])-1)
                    rho_coarse[iz, ir] = sources[-1][iz_fine, ir_fine]
            
            rho_current = rho_coarse
    
    # Solve on coarsest grid
    phi = None
    
    # V-cycle
    for level in range(n_levels-1, -1, -1):
        Z_level, R_level = grids[level]
        rho_level = sources[level]
        
        if level == n_levels - 1:
            # Coarsest level - solve from scratch
            phi, _, _ = solve_axisym_sor(R_level, Z_level, rho_level, params)
        else:
            # Interpolate from coarser level
            NZ_fine = len(Z_level)
            NR_fine = len(R_level)
            phi_fine = np.zeros((NZ_fine, NR_fine))
            
            NZ_coarse, NR_coarse = phi.shape
            for iz in range(NZ_fine):
                for ir in range(NR_fine):
                    iz_coarse = min(iz // 2, NZ_coarse - 1)
                    ir_coarse = min(ir // 2, NR_coarse - 1)
                    phi_fine[iz, ir] = phi[iz_coarse, ir_coarse]
            
            # Smooth on fine level
            phi, _, _ = solve_axisym_sor(R_level, Z_level, rho_level, params, phi_init=phi_fine)
    
    # Final solve on finest grid
    phi, gR, gZ = solve_axisym_sor(R, Z, rho, params, phi_init=phi)
    
    return phi, gR, gZ


if __name__ == "__main__":
    """Quick test of the improved solver."""
    
    print("\nTesting improved PDE solver...")
    
    # Create a simple test case
    NR, NZ = 32, 32
    R = np.linspace(0, 100, NR)
    Z = np.linspace(-100, 100, NZ)
    
    # Gaussian blob density
    R2D, Z2D = np.meshgrid(R, Z)
    r = np.sqrt(R2D**2 + Z2D**2)
    rho = 1e8 * np.exp(-r**2 / (2 * 20**2))
    
    # Test parameters
    params = EnhancedSolverParams(
        gamma=0.5,
        beta=1.0,
        use_geometric_enhancement=True,
        lambda0=0.5,
        alpha_grad=1.5,
        max_iter=1000,
        rtol=1e-5,
        verbose=True
    )
    
    print("\nSolving with SOR...")
    phi, gR, gZ = solve_axisym_sor(R, Z, rho, params)
    
    print(f"\nResults:")
    print(f"  max|phi| = {np.max(np.abs(phi)):.2e}")
    print(f"  max|gR| = {np.max(np.abs(gR)):.2e}")
    print(f"  max|gZ| = {np.max(np.abs(gZ)):.2e}")
    
    # Compare with Newtonian
    M_total = np.sum(rho) * (R[1]-R[0]) * (Z[1]-Z[0]) * 2 * np.pi * R[R>0].mean()
    g_newton_est = G * M_total / (50**2)  # At r=50 kpc
    print(f"  Newtonian g (r=50kpc) ~ {g_newton_est:.2e}")
    
    print("\nSolver test complete!")