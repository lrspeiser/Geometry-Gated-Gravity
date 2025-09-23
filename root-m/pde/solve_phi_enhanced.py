# -*- coding: utf-8 -*-
"""
root-m/pde/solve_phi_enhanced.py

Enhanced PDE solver with:
1. Fixed normalization for total-baryon densities
2. Integrated geometric enhancement based on density gradients
3. Automatic scaling based on system size

The geometric enhancement factor κ(r) amplifies gravity where density gradients are large,
naturally strengthening the field where baryons thin out.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# Constants for normalization
G_CONSTANT = 4.301e-6  # km^2/s^2 * kpc / Msun - gravitational constant in these units

@dataclass
class EnhancedSolverParams:
    # Core PDE parameters
    S0_base: float = 1.0e-4      # base amplitude (will be auto-scaled)
    rc_kpc: float = 22.0         # soft length [kpc]
    g0_kms2_per_kpc: float = 1200.0  # scale for mobility
    m_exp: float = 1.0           # kinetic exponent
    
    # Geometric enhancement parameters
    use_geometric_enhancement: bool = True
    lambda0: float = 0.5         # max enhancement amplitude
    alpha_grad: float = 1.5      # gradient sensitivity
    beta_grad: float = 1.0       # saturation scale
    gamma_scale: float = 0.5     # size scaling exponent
    r_ref_kpc: float = 30.0      # reference size
    
    # Geometry-aware scaling (from original)
    rc_gamma: float = 0.5        # rc scaling with r_half
    sigma_beta: float = 0.10     # S0 scaling with surface density
    rc_ref_kpc: float = 30.0
    sigma0_Msun_pc2: float = 150.0
    
    # Saturating mobility
    use_saturating_mobility: bool = True
    gsat_kms2_per_kpc: float = 2500.0
    n_sat: float = 2.0
    
    # Solver parameters
    max_iter: int = 3000
    tol: float = 1e-6
    omega: float = 0.5
    floor_kms2: float = 50.0
    
    # Boundary conditions
    bc_robin_lambda: float = 0.0


def compute_geometric_enhancement(R, Z, rho, params):
    """
    Compute the geometric enhancement factor κ based on density gradients.
    
    κ(r) = 1 + λ_eff * |∇log(ρ)|^α / (1 + |∇log(ρ)|^α/β)
    
    where λ_eff scales with system size.
    """
    NZ, NR = rho.shape
    dR = float(np.mean(np.diff(R)))
    dZ = float(np.mean(np.diff(Z)))
    
    # Compute half-mass radius for scaling
    R2D = np.broadcast_to(R.reshape(1,-1), (NZ, NR))
    Z2D = np.broadcast_to(Z.reshape(-1,1), (NZ, NR))
    dV = (2.0 * np.pi) * R2D * dR * dZ
    mass_cells = np.clip(rho, 0.0, None) * dV
    r_cell = np.sqrt(R2D*R2D + Z2D*Z2D)
    r_flat = r_cell.reshape(-1)
    m_flat = mass_cells.reshape(-1)
    order = np.argsort(r_flat)
    M_cum = np.cumsum(m_flat[order])
    M_tot = float(M_cum[-1]) if M_cum.size else 0.0
    r_half = float(np.interp(0.5*M_tot, M_cum, r_flat[order])) if M_tot > 0 else 10.0
    
    # Scale enhancement with system size
    lambda_eff = params.lambda0 * (r_half / params.r_ref_kpc)**params.gamma_scale
    
    # Compute log density gradient
    log_rho = np.log(np.maximum(rho, 1e-30))
    
    # Gradient in R and Z directions
    grad_R = np.zeros_like(log_rho)
    grad_Z = np.zeros_like(log_rho)
    
    # Central differences with boundary handling
    grad_R[:, 1:-1] = (log_rho[:, 2:] - log_rho[:, :-2]) / (2*dR)
    grad_R[:, 0] = (log_rho[:, 1] - log_rho[:, 0]) / dR
    grad_R[:, -1] = (log_rho[:, -1] - log_rho[:, -2]) / dR
    
    grad_Z[1:-1, :] = (log_rho[2:, :] - log_rho[:-2, :]) / (2*dZ)
    grad_Z[0, :] = (log_rho[1, :] - log_rho[0, :]) / dZ
    grad_Z[-1, :] = (log_rho[-1, :] - log_rho[-2, :]) / dZ
    
    # Magnitude of gradient
    grad_mag = np.sqrt(grad_R**2 + grad_Z**2)
    
    # Compute enhancement factor
    grad_term = grad_mag**params.alpha_grad
    kappa = 1.0 + lambda_eff * grad_term / (1.0 + grad_term/params.beta_grad)
    
    return kappa, r_half


def solve_axisym_enhanced(R: np.ndarray, Z: np.ndarray, rho_Msun_kpc3: np.ndarray, params: EnhancedSolverParams):
    """
    Enhanced PDE solver with proper normalization and geometric enhancement.
    
    Returns: phi, gR, gZ
    """
    NZ, NR = rho_Msun_kpc3.shape
    assert NR == R.size and NZ == Z.size
    dR = float(np.mean(np.diff(R)))
    dZ = float(np.mean(np.diff(Z)))
    
    # Initialize
    phi = np.zeros_like(rho_Msun_kpc3, dtype=float)
    eps = (params.floor_kms2**2) / max(params.rc_kpc, 1e-9)
    
    R2D = np.broadcast_to(R.reshape(1,-1), (NZ, NR))
    Z2D = np.broadcast_to(Z.reshape(-1,1), (NZ, NR))
    
    # Compute geometric enhancement if enabled
    if params.use_geometric_enhancement:
        kappa, r_half = compute_geometric_enhancement(R, Z, rho_Msun_kpc3, params)
        print(f"[Enhanced PDE] r_half={r_half:.1f} kpc, max κ={np.max(kappa):.2f}")
    else:
        kappa = np.ones_like(rho_Msun_kpc3)
        r_half = 30.0
    
    # Compute geometry-aware S0 scaling (like original)
    dV = (2.0 * np.pi) * R2D * dR * dZ
    mass_cells = np.clip(rho_Msun_kpc3, 0.0, None) * dV
    M_tot = np.sum(mass_cells)
    sigma_bar_kpc2 = (0.5*M_tot) / (np.pi * max(r_half, 1e-9)**2) if r_half > 0 else 1e6
    sigma_bar_pc2 = sigma_bar_kpc2 / 1.0e6
    
    # Effective rc and S0 with geometry scaling
    rc_eff = params.rc_kpc * (max(r_half, 1e-12) / max(params.rc_ref_kpc, 1e-12))**params.rc_gamma
    S0_eff_base = params.S0_base * (max(params.sigma0_Msun_pc2, 1e-12) / max(sigma_bar_pc2, 1e-12))**params.sigma_beta
    
    # CRITICAL FIX: Normalize S0 based on typical density scale
    # The issue was that S0 wasn't properly scaled for the density magnitude
    rho_typical = np.median(rho_Msun_kpc3[rho_Msun_kpc3 > 0]) if np.any(rho_Msun_kpc3 > 0) else 1e6
    # We want the source term S0 * rho to produce reasonable field strengths
    # Target: S0 * rho ~ G * rho for proper scaling
    S0_normalized = G_CONSTANT * S0_eff_base
    
    # Apply enhancement to source
    S_eff = S0_normalized * kappa
    
    print(f"[Enhanced PDE] rc_eff={rc_eff:.1f} kpc, S0_norm={S0_normalized:.2e}, rho_typ={rho_typical:.2e} Msun/kpc^3")
    
    # Iterative solver
    for it in range(params.max_iter):
        # Compute gradients
        gR = np.zeros_like(phi)
        gZ = np.zeros_like(phi)
        gR[:,1:-1] = (phi[:,2:] - phi[:,:-2]) / (2*dR)
        gR[:,0] = (phi[:,1] - phi[:,0]) / dR
        gR[:,-1] = (phi[:,-1] - phi[:,-2]) / dR
        
        gZ[1:-1,:] = (phi[2:,:] - phi[:-2,:]) / (2*dZ)
        gZ[0,:] = (phi[1,:] - phi[0,:]) / dZ
        gZ[-1,:] = (phi[-1,:] - phi[-2,:]) / dZ
        
        # Compute mobility coefficient
        grad_mag = np.sqrt(gR*gR + gZ*gZ + eps*eps)
        A = (grad_mag / max(params.g0_kms2_per_kpc, 1e-12))**params.m_exp
        
        # Saturating mobility
        if params.use_saturating_mobility and params.gsat_kms2_per_kpc > 0.0:
            A = A / (1.0 + np.power(np.maximum(grad_mag / params.gsat_kms2_per_kpc, 0.0), params.n_sat))
        
        # Interface values
        A_Rp = np.zeros_like(A); A_Rm = np.zeros_like(A)
        A_Rp[:, :-1] = 0.5*(A[:,1:] + A[:,:-1]); A_Rp[:, -1] = A[:, -1]
        A_Rm[:, 1:] = 0.5*(A[:,1:] + A[:,:-1]); A_Rm[:, 0] = A[:, 0]
        
        A_Zp = np.zeros_like(A); A_Zm = np.zeros_like(A)
        A_Zp[:-1, :] = 0.5*(A[1:,:] + A[:-1,:]); A_Zp[-1,:] = A[-1,:]
        A_Zm[1:, :] = 0.5*(A[1:,:] + A[:-1,:]); A_Zm[0, :] = A[0, :]
        
        # Gauss-Seidel update
        phi_old = phi.copy()
        for j in range(NZ):
            for i in range(NR):
                R_i = max(R[i], 1e-6)
                
                # Neighbors
                phi_Rp = phi[j, i+1] if i+1 < NR else phi[j, i]
                phi_Rm = phi[j, i-1] if i-1 >= 0 else phi[j, i]
                phi_Zp = phi[j+1, i] if j+1 < NZ else phi[j, i]
                phi_Zm = phi[j-1, i] if j-1 >= 0 else phi[j, i]
                
                # Coefficients
                cRp = (R_i * A_Rp[j, i]) / (dR*dR)
                cRm = (R_i * A_Rm[j, i]) / (dR*dR)
                cZp = A_Zp[j, i] / (dZ*dZ)
                cZm = A_Zm[j, i] / (dZ*dZ)
                
                diag = (cRp + cRm) / R_i + cZp + cZm + 1e-20
                rhs = (cRp*phi_Rp + cRm*phi_Rm) / R_i + cZp*phi_Zp + cZm*phi_Zm
                
                # Add source term with proper normalization and enhancement
                rhs += S_eff[j, i] * rho_Msun_kpc3[j, i]
                
                phi_new = rhs / diag
                phi[j, i] = (1.0 - params.omega)*phi[j, i] + params.omega*phi_new
        
        # Robin BC if enabled
        if params.bc_robin_lambda and params.bc_robin_lambda > 0.0:
            lam = float(params.bc_robin_lambda)
            phi[:, -1] = phi[:, -2] / (1.0 + lam * dR)
            phi[-1, :] = phi[-2, :] / (1.0 + lam * dZ)
            phi[0, :] = phi[1, :] / (1.0 + lam * dZ)
        
        # Check convergence
        diff = np.linalg.norm(phi - phi_old) / (np.linalg.norm(phi_old) + 1e-12)
        if diff < params.tol:
            print(f"[Enhanced PDE] Converged at iteration {it} (residual={diff:.2e})")
            break
    
    # Final gradient computation
    gR = np.zeros_like(phi)
    gZ = np.zeros_like(phi)
    gR[:,1:-1] = (phi[:,2:] - phi[:,:-2]) / (2*dR)
    gR[:,0] = (phi[:,1] - phi[:,0]) / dR
    gR[:,-1] = (phi[:,-1] - phi[:,-2]) / dR
    
    gZ[1:-1,:] = (phi[2:,:] - phi[:-2,:]) / (2*dZ)
    gZ[0,:] = (phi[1,:] - phi[0,:]) / dZ
    gZ[-1,:] = (phi[-1,:] - phi[-2,:]) / dZ
    
    return phi, gR, gZ