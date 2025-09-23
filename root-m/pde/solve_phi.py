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

# Optional smoothing for ambient-density boost
try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:  # pragma: no cover
    gaussian_filter = None

@dataclass
class SolverParams:
    S0: float = 1.0e-7   # amplitude scaling [km^2 s^-2 kpc Msun^-1]
    rc_kpc: float = 15.0 # global soft length [kpc]
    g0_kms2_per_kpc: float = 1000.0 # dimensional scale for A(|∇φ|)=(|∇φ|/g0)^m
    m_exp: float = 1.0   # exponent in A(|∇φ|)=(|∇φ|/g0)^m

    # Mass-aware source (existing)
    eta: float = 0.0     # global mass-coupling exponent; if 0, disabled
    Mref_Msun: float = 6.0e10

    # Curvature-aware mobility (existing)
    kappa: float = 0.0   # weight for |d ln rho / d ln r|; if 0, disabled
    q_slope: float = 1.0 # exponent on curvature term

    # Anisotropic in-plane mobility (existing)
    chi: float = 0.0     # in-plane boost amplitude near mid-plane; if 0, disabled
    h_aniso_kpc: float = 0.3  # exponential decay scale in z for anisotropy

    # NEW A1: saturating mobility cap
    use_saturating_mobility: bool = False
    gsat_kms2_per_kpc: float = 2000.0
    n_sat: float = 2.0

    # NEW A2: ambient-density boost (multiplies S0 locally)
    use_ambient_boost: bool = False
    beta_env: float = 0.0
    rho_ref_Msun_per_kpc3: float = 1.0e6
    env_L_kpc: float = 150.0

    # NEW: local surface-density screen (multiplies source; uses Sigma_loc from rho grid)
    use_sigma_screen: bool = False
    sigma_star_Msun_per_pc2: float = 150.0
    alpha_sigma: float = 1.0
    n_sigma: float = 2.0

    # NEW: compactness-aware source modulation (global, density-based screening)
    use_compactness_source: bool = False
    rho_comp_star_Msun_per_kpc3: float = 1.0e6
    alpha_comp: float = 0.2
    m_comp: float = 2.0

    # Solver
    max_iter: int = 2000
    tol: float = 1e-5
    omega: float = 0.6   # relaxation
    floor_kms2: float = 50.0  # sets ε via ε = floor_kms2^2/rc

    # Optional Robin boundary condition strength (1/kpc). If 0, disabled.
    bc_robin_lambda: float = 0.0


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

    # Precompute mass-aware source factor S_eff (A1) and/or compactness needs Menc
    need_Menc = bool(params.eta and params.eta != 0.0) or bool(params.use_compactness_source)
    if need_Menc:
        # Cell volumes for axisym: dV = 2π R dR dZ
        dV = (2.0 * np.pi) * R2D * dR * dZ
        mass_cells = np.clip(rho_Msun_kpc3, 0.0, None) * dV
        r_cell = np.sqrt(R2D*R2D + Z2D*Z2D)
        r_flat = r_cell.reshape(-1)
        mass_flat = mass_cells.reshape(-1)
        order = np.argsort(r_flat)
        mass_cum = np.cumsum(mass_flat[order])
        # Menc at each cell via cumulative mass up to its radius
        Menc_flat = np.zeros_like(mass_flat)
        Menc_flat[order] = mass_cum
        Menc = Menc_flat.reshape(NZ, NR)
    # Base source factor from mass-aware term (if enabled)
    if params.eta and params.eta != 0.0:
        S_eff = params.S0 * (1.0 + np.power(np.clip(Menc / max(params.Mref_Msun, 1e-30), 1e-30, None), params.eta))
    else:
        S_eff = np.full((NZ, NR), params.S0, dtype=float)

    # Ambient-density boost to source (A2-new)
    if params.use_ambient_boost and params.beta_env != 0.0:
        if gaussian_filter is not None:
            # sigma specified in grid cells; array is (NZ, NR) so (sigma_Z, sigma_R)
            sigma_Z = max(params.env_L_kpc / max(dZ, 1e-9), 1.0)
            sigma_R = max(params.env_L_kpc / max(dR, 1e-9), 1.0)
            rho_env = gaussian_filter(rho_Msun_kpc3, sigma=(sigma_Z, sigma_R), mode="nearest")
            boost = 1.0 + params.beta_env * np.sqrt(np.clip(rho_env / max(params.rho_ref_Msun_per_kpc3, 1e-30), 0.0, None))
        else:
            boost = 1.0
        S_eff = S_eff * boost

    # Local surface-density screen (sigma screen)
    if params.use_sigma_screen:
        # Sigma_loc(R) = ∫ rho(R,z) dz [Msun/kpc^2]; convert to Msun/pc^2
        try:
            Sigma_loc_kpc2 = np.trapz(np.clip(rho_Msun_kpc3, 0.0, None), Z, axis=0)
        except Exception:
            # Fallback: assume uniform spacing in Z if integration fails
            Sigma_loc_kpc2 = np.sum(np.clip(rho_Msun_kpc3, 0.0, None), axis=0) * max(dZ, 1e-9)
        Sigma_loc_pc2 = Sigma_loc_kpc2 / 1.0e6
        denom = max(params.sigma_star_Msun_per_pc2, 1e-30)
        m = max(params.n_sigma, 1e-9)
        ratio = np.maximum(Sigma_loc_pc2 / denom, 0.0)
        S_R = np.power(1.0 + np.power(ratio, m), -params.alpha_sigma / m)
        # Broadcast across Z to (NZ, NR)
        S_2D = np.broadcast_to(S_R.reshape(1, -1), S_eff.shape)
        S_eff = S_eff * S_2D
        # Diagnostics
        try:
            q = np.quantile(Sigma_loc_pc2, [0.0, 0.5, 0.95])
            print(f"[PDE] Sigma-screen ON: Sigma*={params.sigma_star_Msun_per_pc2:.3g} Msun/pc^2, alpha={params.alpha_sigma}, n={params.n_sigma}; Sigma_loc pc2 min/med/p95 = {q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g}")
        except Exception:
            print(f"[PDE] Sigma-screen ON: Sigma*={params.sigma_star_Msun_per_pc2:.3g} Msun/pc^2, alpha={params.alpha_sigma}, n={params.n_sigma}")

    # Compactness-aware modulation of source (density-based screening)
    if params.use_compactness_source:
        # Use spherical mean enclosed density at each cell radius
        # rho_bar = 3 Menc / (4π r^3); avoid r=0 singularity with small floor
        r_eps = 1e-9
        r_cell = np.sqrt(R2D*R2D + Z2D*Z2D)
        rho_bar = (3.0 * Menc) / (4.0 * np.pi * np.maximum(r_cell, r_eps)**3)
        x = rho_bar / max(params.rho_comp_star_Msun_per_kpc3, 1e-30)
        m = max(params.m_comp, 1e-9)
        f_comp = np.power(1.0 + np.power(np.clip(x, 0.0, None), m), -params.alpha_comp / m)
        S_eff = S_eff * f_comp

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
        # NEW A1: saturating mobility cap
        if params.use_saturating_mobility and params.gsat_kms2_per_kpc > 0.0:
            A = A / (1.0 + np.power(np.maximum(grad_mag / params.gsat_kms2_per_kpc, 0.0), params.n_sat))
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

        # Optional Robin boundary update (∂n φ + λ φ = 0) on outer box
        if params.bc_robin_lambda and params.bc_robin_lambda > 0.0:
            lam = float(params.bc_robin_lambda)
            # R = Rmax (right edge)
            phi[:, -1] = phi[:, -2] / (1.0 + lam * dR)
            # Z = +Zmax (top)
            phi[-1, :] = phi[-2, :] / (1.0 + lam * dZ)
            # Z = -Zmax (bottom)
            phi[0, :]  = phi[1, :]  / (1.0 + lam * dZ)

        # check residual (L2 norm of update)
        diff = np.linalg.norm(phi - phi_old) / (np.linalg.norm(phi_old) + 1e-12)
        if diff < params.tol:
            break

    gR, gZ = _grad_RZ(phi, dR, dZ)
    return phi, gR, gZ
