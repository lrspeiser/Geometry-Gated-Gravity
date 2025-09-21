# -*- coding: utf-8 -*-
"""
root-m/pde/solve_phi.py

Finite-difference solver for the nonlinear PDE:
    ∇·( A(|∇φ|) ∇φ ) = - S0 * ρ_b(x),
with A(|∇φ|) ∝ |∇φ| (k-mouflage-like kinetic nonlinearity).

Axisymmetric (R,z) grid:
    div = (1/R) ∂/∂R ( R A φ_R ) + ∂/∂z ( A φ_z )

We solve with nonlinear Gauss–Seidel / fixed-point iterations.
This is a protected, experimental implementation intended for Root‑M tests.

Units:
- R,z in kpc
- ρ_b in Msun/kpc^3
- φ is a potential whose gradient has units of (km/s)^2/kpc
- S0 [km^2 s^-2 kpc Msun^-1] controls the amplitude (global);
  choose S0 to recover galaxy-scale amplitudes from SPARC CV.

Regularizer:
- rc_kpc enters the A() computation as |∇φ|_eff = sqrt(|∇φ|^2 + (ε)^2)
  with ε ≈ v_floor^2/rc to avoid singular coefficients near zero slopes.

This file exposes a single entry point: solve_axisym(r_grid, z_grid, rho, S0, rc_kpc, ...)
which returns φ, g_R, g_z.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class SolverParams:
    S0: float = 1.0e-7   # amplitude scaling [km^2 s^-2 kpc Msun^-1]
    rc_kpc: float = 15.0 # global soft length [kpc]
    max_iter: int = 2000
    tol: float = 1e-5
    omega: float = 0.6   # relaxation
    floor_kms2: float = 50.0  # sets ε via ε = floor_kms2^2/rc


def _grad_RZ(phi: np.ndarray, dR: float, dZ: float):
    # central differences, Neumann at boundaries (zero-gradient)
    gR = np.zeros_like(phi)
    gZ = np.zeros_like(phi)
    gR[:,1:-1] = (phi[:,2:] - phi[:,:-2]) / (2*dR)
    gR[:,0]    = (phi[:,1] - phi[:,0]) / dR
    gR[:,-1]   = (phi[:,-1] - phi[:,-2]) / dR

    gZ[1:-1,:] = (phi[2:,:] - phi[:-2,:]) / (2*dZ)
    gZ[0,:]    = (phi[1,:] - phi[0,:]) / dZ
    gZ[-1,:]   = (phi[-1,:] - phi[-2,:]) / dZ
    return gR, gZ


def solve_axisym(R: np.ndarray, Z: np.ndarray, rho_Msun_kpc3: np.ndarray, params: SolverParams):
    """
    Solve ∇·( A ∇φ ) = - S0 ρ on an axisymmetric (z,R) grid.

    Inputs:
      R: 1D array of R (kpc), length NR
      Z: 1D array of z (kpc), length NZ
      rho_Msun_kpc3: 2D array (NZ, NR)
      params: SolverParams

    Returns:
      phi (NZ, NR), gR (NZ, NR), gZ (NZ, NR)
    """
    NZ, NR = rho_Msun_kpc3.shape
    assert NR == R.size and NZ == Z.size
    dR = float(np.mean(np.diff(R)))
    dZ = float(np.mean(np.diff(Z)))

    phi = np.zeros_like(rho_Msun_kpc3, dtype=float)
    eps = (params.floor_kms2**2) / max(params.rc_kpc, 1e-9)

    R2D = np.broadcast_to(R.reshape(1,-1), (NZ, NR))

    for it in range(params.max_iter):
        # compute gradients and coefficient A = |∇φ|_eff
        gR, gZ = _grad_RZ(phi, dR, dZ)
        grad_mag = np.sqrt(gR*gR + gZ*gZ + eps*eps)
        A = grad_mag

        # precompute interface A at half indices
        A_Rp = np.zeros_like(A); A_Rm = np.zeros_like(A)
        A_Rp[:, :-1] = 0.5*(A[:,1:] + A[:,:-1]);  A_Rp[:, -1] = A[:, -1]
        A_Rm[:, 1:]  = 0.5*(A[:,1:] + A[:,:-1]);  A_Rm[:, 0]  = A[:, 0]

        A_Zp = np.zeros_like(A); A_Zm = np.zeros_like(A)
        A_Zp[:-1, :] = 0.5*(A[1:,:] + A[:-1,:]); A_Zp[-1,:] = A[-1,:]
        A_Zm[1:,  :] = 0.5*(A[1:,:] + A[:-1,:]); A_Zm[0, :] = A[0, :]

        # nonlinear Gauss–Seidel update
        phi_old = phi.copy()
        for j in range(NZ):
            for i in range(NR):
                R_i = max(R[i], 1e-6)
                # neighbors
                phi_Rp = phi[j, i+1] if i+1 < NR else phi[j, i]
                phi_Rm = phi[j, i-1] if i-1 >= 0 else phi[j, i]
                phi_Zp = phi[j+1, i] if j+1 < NZ else phi[j, i]
                phi_Zm = phi[j-1, i] if j-1 >= 0 else phi[j, i]

                # coefficients (cylindrical divergence)
                cRp = (R_i * A_Rp[j, i]) / (dR*dR)
                cRm = (R_i * A_Rm[j, i]) / (dR*dR)
                cZp = A_Zp[j, i] / (dZ*dZ)
                cZm = A_Zm[j, i] / (dZ*dZ)

                diag = (cRp + cRm) / R_i + cZp + cZm + 1e-20
                rhs  = (cRp*phi_Rp + cRm*phi_Rm) / R_i + cZp*phi_Zp + cZm*phi_Zm - params.S0 * rho_Msun_kpc3[j, i]

                phi_new = rhs / diag
                phi[j, i] = (1.0 - params.omega)*phi[j, i] + params.omega*phi_new

        # check residual (L2 norm of update)
        diff = np.linalg.norm(phi - phi_old) / (np.linalg.norm(phi_old) + 1e-12)
        if diff < params.tol:
            break

    gR, gZ = _grad_RZ(phi, dR, dZ)
    return phi, gR, gZ
