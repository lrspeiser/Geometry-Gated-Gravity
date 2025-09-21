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
    g0_kms2_per_kpc: float = 1000.0 # dimensional scale for A(|∇φ|)=|∇φ|/g0 to fix units
    m_exp: float = 1.0   # exponent in A(|∇φ|)=(|∇φ|/g0)^m
    # Mass-aware source (A1)
    eta: float = 0.0     # global mass-coupling exponent; if 0, disabled
    Mref_Msun: float = 6.0e10
    # Curvature-aware mobility (A2)
    kappa: float = 0.0   # weight for |d ln rho / d ln r|; if 0, disabled
    q_slope: float = 1.0 # exponent on curvature term
    # Anisotropic in-plane mobility (A3)
    chi: float = 0.0     # in-plane boost amplitude near mid-plane; if 0, disabled
    h_aniso_kpc: float = 0.3  # exponential decay scale in z for anisotropy
    # Solver
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
    Z2D = np.broadcast_to(Z.reshape(-1,1), (NZ, NR))

    # Precompute mass-aware source factor S_eff (A1)
    if params.eta and params.eta != 0.0:
        # Cell volumes for axisym: dV = 2π R dR dZ
        dV = (2.0 * np.pi) * R2D * dR * dZ
        mass_cells = np.clip(rho_Msun_kpc3, 0.0, None) * dV
        r_cell = np.sqrt(R2D*R2D + Z2D*Z2D).reshape(-1)
        mass_flat = mass_cells.reshape(-1)
        order = np.argsort(r_cell)
        mass_cum = np.cumsum(mass_flat[order])
        # Menc at each cell via cumulative mass up to its radius
        Menc_flat = np.zeros_like(mass_flat)
        Menc_flat[order] = mass_cum
        Menc = Menc_flat.reshape(NZ, NR)
        S_eff = params.S0 * (1.0 + np.power(np.clip(Menc / max(params.Mref_Msun, 1e-30), 1e-30, None), params.eta))
    else:
        S_eff = np.full((NZ, NR), params.S0, dtype=float)

    # Precompute curvature factor s = r * |d ln rho / dr| (A2)
    if params.kappa and params.kappa != 0.0:
        rho_floor = 1e-30
        ln_rho = np.log(np.clip(rho_Msun_kpc3, rho_floor, None))
        # gradients of ln rho
        dln_dR, dln_dZ = _grad_RZ(ln_rho, dR, dZ)
        r_mag = np.sqrt(R2D*R2D + Z2D*Z2D)
        r_hat_R = np.divide(R2D, np.maximum(r_mag, 1e-9))
        r_hat_Z = np.divide(Z2D, np.maximum(r_mag, 1e-9))
        dln_dr = np.abs(dln_dR * r_hat_R + dln_dZ * r_hat_Z)
        s_dimless = r_mag * dln_dr
    else:
        s_dimless = None

    for it in range(params.max_iter):
        # compute gradients and coefficient A = |∇φ|_eff
        gR, gZ = _grad_RZ(phi, dR, dZ)
        grad_mag = np.sqrt(gR*gR + gZ*gZ + eps*eps)
        A = (grad_mag / max(params.g0_kms2_per_kpc, 1e-12))**max(params.m_exp, 1e-6)
        if s_dimless is not None:
            A = A * np.power(1.0 + params.kappa * s_dimless, params.q_slope)

        # precompute interface A at half indices
        A_Rp = np.zeros_like(A); A_Rm = np.zeros_like(A)
        A_Rp[:, :-1] = 0.5*(A[:,1:] + A[:,:-1]);  A_Rp[:, -1] = A[:, -1]
        A_Rm[:, 1:]  = 0.5*(A[:,1:] + A[:,:-1]);  A_Rm[:, 0]  = A[:, 0]

        A_Zp = np.zeros_like(A); A_Zm = np.zeros_like(A)
        A_Zp[:-1, :] = 0.5*(A[1:,:] + A[:-1,:]); A_Zp[-1,:] = A[-1,:]
        A_Zm[1:,  :] = 0.5*(A[1:,:] + A[:-1,:]); A_Zm[0, :] = A[0, :]

        # Anisotropic in-plane boost for R-fluxes (A3)
        if params.chi and params.chi != 0.0:
            B = 1.0 + params.chi * np.exp(-np.abs(Z2D) / max(params.h_aniso_kpc, 1e-6))
            B_Rp = np.zeros_like(B); B_Rm = np.zeros_like(B)
            B_Rp[:, :-1] = 0.5*(B[:,1:] + B[:,:-1]); B_Rp[:, -1] = B[:, -1]
            B_Rm[:, 1:]  = 0.5*(B[:,1:] + B[:,:-1]); B_Rm[:, 0]  = B[:, 0]
            A_Rp = A_Rp * B_Rp
            A_Rm = A_Rm * B_Rm

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
                rhs  = (cRp*phi_Rp + cRm*phi_Rm) / R_i + cZp*phi_Zp + cZm*phi_Zm + S_eff[j, i] * rho_Msun_kpc3[j, i]

                phi_new = rhs / diag
                phi[j, i] = (1.0 - params.omega)*phi[j, i] + params.omega*phi_new

        # check residual (L2 norm of update)
        diff = np.linalg.norm(phi - phi_old) / (np.linalg.norm(phi_old) + 1e-12)
        if diff < params.tol:
            break

    gR, gZ = _grad_RZ(phi, dR, dZ)
    return phi, gR, gZ
