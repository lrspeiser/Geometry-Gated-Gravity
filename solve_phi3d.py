
"""
solve_phi3d.py — Reference 3D solver for the G³ field (Geometry‑Gated Gravity)

This module solves the nonlinear elliptic PDE in Cartesian 3D:

    ∇ · [ μ(|∇φ|, ρ; params) ∇φ ] = S0_eff * 4π G * ρ_b(x, y, z)        (1)

where
  - φ(x,y,z) is the G³ auxiliary potential whose gradient adds a "tail"
    acceleration g_φ = −∇φ to the Newtonian baryon acceleration g_N.
  - ρ_b is the *total* baryon density [Msun/kpc^3] including gas×√C(r) and stars.
  - μ is a geometry/field‑dependent mobility that can include:
        (a) saturating mobility in |∇φ|/g0  (avoids blowups in dense cores),
        (b) a local surface‑density screen Σ_loc (suppresses sources when Σ >> Σ⋆).
  - S0_eff and r_c_eff are geometry‑aware global scalings set from the baryon map.

The PDE is solved on a regular 3D grid with spacing (dx, dy, dz) using a
nonlinear multigrid V‑cycle with red‑black Gauss‑Seidel smoothing.
Robin boundary conditions are supported on all faces:

    ∂φ/∂n + φ / λ = 0                                                    (2)

with λ→∞ reducing to Neumann (∂φ/∂n = 0). Dirichlet (φ = 0) is available too.

Outputs:
  - φ field, g_φ components, and bookkeeping (rc_eff, S0_eff, r_half, Σ̄).

Unit conventions (consistent with the rest of your repo):
  - Distances in kpc, time in s, velocity in km/s.
  - Density ρ_b in Msun/kpc^3.
  - Gravitational constant G_kpc = 4.300917270e-6  (kpc km^2 s^-2 Msun^-1).

Author: (you)
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import json
import math

G_KPC = 4.300917270e-6  # [kpc km^2 s^-2 Msun^-1]


# ---- Utility: geometry scalars from the same 3D rho grid --------------------

def geometry_scalars_from_rho(
    rho: np.ndarray, dx: float, dy: float, dz: float, origin: Tuple[float, float, float]
) -> Dict[str, float]:
    """
    Compute global geometry scalars from the provided total-baryon density grid.

    Returns:
        {
          'Mtot_Msun': float,
          'r_half_kpc': float,
          'sigma_bar_Msun_pc2': float
        }
    Notes:
        - r_half is the radius of the sphere enclosing half the mass.
        - Σ̄ is an *approximate* mean surface density defined by projecting the
          mass onto the mid‑plane (z = const) and dividing by the geometric area.
          This is sufficient as a scalar gate; per‑cell Σ_loc is handled by the
          sigma screen in the mobility.
    """
    nx, ny, nz = rho.shape
    x0, y0, z0 = origin

    # Cell centers
    xs = (np.arange(nx) + 0.5) * dx - x0
    ys = (np.arange(ny) + 0.5) * dy - y0
    zs = (np.arange(nz) + 0.5) * dz - z0

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2)

    dV = dx * dy * dz  # [kpc^3]
    mass = np.sum(rho) * dV  # [Msun]
    if mass <= 0:
        return {'Mtot_Msun': 0.0, 'r_half_kpc': 0.0, 'sigma_bar_Msun_pc2': 0.0}

    # Cumulative M(<r) on logarithmic shells (fast approximate histogram)
    r_flat = r.reshape(-1)
    m_flat = (rho * dV).reshape(-1)
    order = np.argsort(r_flat)
    r_sorted = r_flat[order]
    m_sorted = m_flat[order]
    M_cum = np.cumsum(m_sorted)
    # r_half at 0.5 Mtot
    idx = np.searchsorted(M_cum, 0.5 * mass)
    r_half = float(r_sorted[min(idx, len(r_sorted)-1)])

    # project onto z-plane area box as a naive Σ̄ = Mtot / (Lx*Ly)
    Lx = nx * dx
    Ly = ny * dy
    area_kpc2 = Lx * Ly
    sigma_bar_Msun_per_kpc2 = mass / max(area_kpc2, 1e-30)
    # convert to Msun/pc^2
    sigma_bar_Msun_per_pc2 = sigma_bar_Msun_per_kpc2 / (1e6)

    return {
        'Mtot_Msun': float(mass),
        'r_half_kpc': float(r_half),
        'sigma_bar_Msun_pc2': float(sigma_bar_Msun_per_pc2),
    }


# ---- Mobility models --------------------------------------------------------

@dataclass
class MobilityParams:
    g0_kms2_per_kpc: float = 1200.0
    use_saturating_mobility: bool = True
    g_sat_kms2_per_kpc: float = 2500.0
    n_sat: float = 2.0
    # Sigma screen
    use_sigma_screen: bool = False
    sigma_star_Msun_per_pc2: float = 150.0
    alpha_sigma: float = 1.0
    n_sigma: float = 2.0


def mobility_mu(
    grad_phi: Tuple[np.ndarray, np.ndarray, np.ndarray],
    rho: np.ndarray,
    params: MobilityParams,
    sigma_loc_pc2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute μ = μ_sat * μ_sigma  (elementwise).

    μ_sat(|∇φ|) = 1 / (1 + (|∇φ| / g_sat)^n_sat)      [optional]

    μ_sigma(Σ_loc) = [ 1 + (Σ_loc / Σ⋆)^n_sigma ]^(−α_sigma/n_sigma)   [optional]

    Both factors are ≥ 0 and dimensionless.
    """
    gx, gy, gz = grad_phi
    abs_g = np.sqrt(gx*gx + gy*gy + gz*gz) + 1e-30

    mu_sat = 1.0
    if params.use_saturating_mobility:
        mu_sat = 1.0 / (1.0 + (abs_g / params.g_sat_kms2_per_kpc) ** params.n_sat)

    mu_sigma = 1.0
    if params.use_sigma_screen and sigma_loc_pc2 is not None:
        mu_sigma = (1.0 + (sigma_loc_pc2 / max(params.sigma_star_Msun_per_pc2, 1e-12))**params.n_sigma) ** (
            -params.alpha_sigma / max(params.n_sigma, 1e-12)
        )

    return mu_sat * mu_sigma


# ---- Σ_loc estimator (local column density) --------------------------------

def sigma_local_pc2(rho: np.ndarray, dz: float, window_cells: int = 8) -> np.ndarray:
    """
    Estimate per‑cell Σ_loc by integrating rho along z in a local window:
        Σ_loc ≈ ∑ rho[i,j,k'] dz over k' in [k−w, k+w]
    Returns Msun/kpc^2, converted to Msun/pc^2.

    This is a symmetric top‑hat window; for disks you may prefer a Gaussian.
    """
    nz = rho.shape[2]
    w = int(max(1, window_cells))
    # cumulative along z for O(1) window sums
    csum = np.cumsum(rho, axis=2)
    # pad a leading zero plane for easy differences
    csum = np.concatenate([np.zeros_like(rho[:, :, :1]), csum], axis=2)

    out = np.zeros_like(rho)
    for k in range(nz):
        k0 = max(0, k - w)
        k1 = min(nz - 1, k + w)
        # convert to indices in csum (shifted by +1)
        v = (csum[:, :, k1 + 1] - csum[:, :, k0]) * dz  # Msun/kpc^2
        out[:, :, k] = v

    return out / 1e6  # Msun/pc^2


# ---- G³ globals & effective scalings ---------------------------------------

@dataclass
class G3Globals:
    S0: float = 1.4e-4
    rc_kpc: float = 22.0
    g0_kms2_per_kpc: float = 1200.0
    rc_gamma: float = 0.5
    rc_ref_kpc: float = 30.0
    sigma_beta: float = 0.10
    sigma0_Msun_pc2: float = 150.0

    def effective(self, r_half_kpc: float, sigma_bar_pc2: float) -> Dict[str, float]:
        rc_eff = self.rc_kpc * (max(r_half_kpc, 1e-6) / max(self.rc_ref_kpc, 1e-12)) ** self.rc_gamma
        S0_eff = self.S0 * (max(self.sigma0_Msun_pc2, 1e-12) / max(sigma_bar_pc2, 1e-12)) ** self.sigma_beta
        return {'rc_eff_kpc': rc_eff, 'S0_eff': S0_eff}


# ---- Discretization helpers -------------------------------------------------

def gradient_3d(phi: np.ndarray, dx: float, dy: float, dz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second‑order central differences for ∇φ with Neumann at the boundary (copy edge)."""
    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)
    gz = np.zeros_like(phi)

    gx[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2 * dx)
    gy[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dy)
    gz[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dz)

    gx[0, :, :] = gx[1, :, :]
    gx[-1, :, :] = gx[-2, :, :]
    gy[:, 0, :] = gy[:, 1, :]
    gy[:, -1, :] = gy[:, -2, :]
    gz[:, :, 0] = gz[:, :, 1]
    gz[:, :, -1] = gz[:, :, -2]
    return gx, gy, gz


def divergence_3d(
    Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray, dx: float, dy: float, dz: float
) -> np.ndarray:
    """Second‑order central differences for ∇·F with zero‑flux at the outermost faces."""
    div = np.zeros_like(Fx)

    div[1:-1, :, :] += (Fx[2:, :, :] - Fx[:-2, :, :]) / (2 * dx)
    div[:, 1:-1, :] += (Fy[:, 2:, :] - Fy[:, :-2, :]) / (2 * dy)
    div[:, :, 1:-1] += (Fz[:, :, 2:] - Fz[:, :, :-2]) / (2 * dz)

    # Neumann: copy adjacent interior derivative
    div[0, :, :] += (Fx[1, :, :] - Fx[0, :, :]) / dx
    div[-1, :, :] += (Fx[-1, :, :] - Fx[-2, :, :]) / dx
    div[:, 0, :] += (Fy[:, 1, :] - Fy[:, 0, :]) / dy
    div[:, -1, :] += (Fy[:, -1, :] - Fy[:, -2, :]) / dy
    div[:, :, 0] += (Fz[:, :, 1] - Fz[:, :, 0]) / dz
    div[:, :, -1] += (Fz[:, :, -1] - Fz[:, :, -2]) / dz

    return div


# ---- Core nonlinear operator -----------------------------------------------

@dataclass
class BC:
    kind: str = "robin"  # 'robin', 'neumann', 'dirichlet'
    lambda_kpc: float = 1e6  # large ~ Neumann; small -> Dirichlet‑like


def apply_operator(
    phi: np.ndarray,
    rho: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    g3: G3Globals,
    mobil: MobilityParams,
    origin: Tuple[float, float, float],
    bc: BC,
    sigma_loc_pc2: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute the residual R = L[φ] − RHS.
    L[φ] = ∇ · ( μ ∇φ )
    RHS  = S0_eff * 4π G * ρ

    Returns residual array and the dict of scalars used (rc_eff, S0_eff, r_half, Σ̄).
    """
    # geometry scalars + effective params
    scalars = geometry_scalars_from_rho(rho, dx, dy, dz, origin)
    eff = g3.effective(scalars['r_half_kpc'], scalars['sigma_bar_Msun_pc2'])
    rc_eff = eff['rc_eff_kpc']
    S0_eff = eff['S0_eff']

    # gradient and mobility
    gx, gy, gz = gradient_3d(phi, dx, dy, dz)
    mu = mobility_mu((gx, gy, gz), rho, mobil, sigma_loc_pc2=sigma_loc_pc2)

    # fluxes
    Fx = mu * gx
    Fy = mu * gy
    Fz = mu * gz

    # core operator
    Lphi = divergence_3d(Fx, Fy, Fz, dx, dy, dz)

    # RHS
    RHS = (S0_eff * 4.0 * math.pi * G_KPC) * rho

    # residual
    R = Lphi - RHS

    # Robin boundary condition weak enforcement:
    if bc.kind == "robin" and bc.lambda_kpc > 0:
        lam = bc.lambda_kpc
        # Implement φ/λ + ∂φ/∂n = 0 ⇒ penalize boundary values
        penalty = 1.0 / max(lam, 1e-12)
        # Add simple penalty to residual at faces
        R[0, :, :] += penalty * phi[0, :, :]
        R[-1, :, :] += penalty * phi[-1, :, :]
        R[:, 0, :] += penalty * phi[:, 0, :]
        R[:, -1, :] += penalty * phi[:, -1, :]
        R[:, :, 0] += penalty * phi[:, :, 0]
        R[:, :, -1] += penalty * phi[:, :, -1]

    scalars.update(eff)
    return R, scalars


# ---- Nonlinear multigrid solver --------------------------------------------

def gauss_seidel_relax(
    phi: np.ndarray,
    rho: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    g3: G3Globals,
    mobil: MobilityParams,
    origin: Tuple[float, float, float],
    bc: BC,
    sigma_loc_pc2: Optional[np.ndarray] = None,
    n_sweeps: int = 2,
) -> None:
    """
    Red‑black Gauss‑Seidel nonlinear relaxation (a few sweeps).
    We apply a damped Newton step on the diagonal approximation.
    """
    nx, ny, nz = phi.shape
    for sweep in range(n_sweeps):
        # two colors
        for color in (0, 1):
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    # Color slicing along k for cache efficiency
                    kslice = slice(1, nz-1)
                    if (i + j) % 2 != color:
                        continue

                    # local stencil (7‑point) for Laplacian‑like operator with μ frozen
                    # compute μ at current iterate
                    gx, gy, gz = gradient_3d(phi, dx, dy, dz)
                    mu = mobility_mu((gx, gy, gz), rho, mobil, sigma_loc_pc2=sigma_loc_pc2)

                    # neighbors
                    phi_c = phi[i, j, 1:-1]
                    phi_xm = phi[i-1, j, 1:-1]
                    phi_xp = phi[i+1, j, 1:-1]
                    phi_ym = phi[i, j-1, 1:-1]
                    phi_yp = phi[i, j+1, 1:-1]
                    phi_zm = phi[i, j, :-2]
                    phi_zp = phi[i, j, 2:]

                    # local μ averaged to faces (simple arithmetic average)
                    mu_xm = 0.5 * (mu[i, j, 1:-1] + mu[i-1, j, 1:-1])
                    mu_xp = 0.5 * (mu[i, j, 1:-1] + mu[i+1, j, 1:-1])
                    mu_ym = 0.5 * (mu[i, j, 1:-1] + mu[i, j-1, 1:-1])
                    mu_yp = 0.5 * (mu[i, j, 1:-1] + mu[i, j+1, 1:-1])
                    mu_zm = 0.5 * (mu[i, j, 1:-1] + mu[i, j, :-2])
                    mu_zp = 0.5 * (mu[i, j, 1:-1] + mu[i, j, 2:])

                    # discrete divergence with μ frozen (linearized step)
                    diag = (mu_xm + mu_xp) / (dx*dx) + (mu_ym + mu_yp) / (dy*dy) + (mu_zm + mu_zp) / (dz*dz)

                    # RHS with effective params
                    scalars = geometry_scalars_from_rho(rho, dx, dy, dz, origin)
                    eff = g3.effective(scalars['r_half_kpc'], scalars['sigma_bar_Msun_pc2'])
                    RHS = (eff['S0_eff'] * 4.0 * math.pi * G_KPC) * rho[i, j, 1:-1]

                    # 7‑point Laplacian (with variable μ on faces)
                    lap = (
                        mu_xp * phi_xp + mu_xm * phi_xm
                    ) / (dx*dx) + (
                        mu_yp * phi_yp + mu_ym * phi_ym
                    ) / (dy*dy) + (
                        mu_zp * phi_zp + mu_zm * phi_zm
                    ) / (dz*dz)

                    new_phi = (RHS + lap) / np.maximum(diag, 1e-30)
                    # Damping for stability
                    phi[i, j, 1:-1] = 0.6 * new_phi + 0.4 * phi_c

    # simple Robin update on faces (diagonal penalty)
    if bc.kind == "robin" and bc.lambda_kpc > 0:
        lam = bc.lambda_kpc
        w = 0.5
        phi[0, :, :] *= (1.0 - w / max(lam, 1e-12))
        phi[-1, :, :] *= (1.0 - w / max(lam, 1e-12))
        phi[:, 0, :] *= (1.0 - w / max(lam, 1e-12))
        phi[:, -1, :] *= (1.0 - w / max(lam, 1e-12))
        phi[:, :, 0] *= (1.0 - w / max(lam, 1e-12))
        phi[:, :, -1] *= (1.0 - w / max(lam, 1e-12))


def restrict_2x(arr: np.ndarray) -> np.ndarray:
    """Full‑weighting restriction by factors of 2 in each dimension."""
    return 0.125 * (
        arr[0::2, 0::2, 0::2] + arr[1::2, 0::2, 0::2] + arr[0::2, 1::2, 0::2] + arr[0::2, 0::2, 1::2] +
        arr[1::2, 1::2, 0::2] + arr[1::2, 0::2, 1::2] + arr[0::2, 1::2, 1::2] + arr[1::2, 1::2, 1::2]
    )


def prolong_2x(arr: np.ndarray, shape_out: Tuple[int, int, int]) -> np.ndarray:
    """Trilinear prolongation back to a finer grid."""
    nx, ny, nz = shape_out
    out = np.zeros(shape_out, dtype=arr.dtype)
    out[0::2, 0::2, 0::2] = arr
    out[1::2, 0::2, 0::2] = arr
    out[0::2, 1::2, 0::2] = arr
    out[0::2, 0::2, 1::2] = arr
    out[1::2, 1::2, 0::2] = arr
    out[1::2, 0::2, 1::2] = arr
    out[0::2, 1::2, 1::2] = arr
    out[1::2, 1::2, 1::2] = arr
    return out[:, :ny, :nz]  # crop in case of odd sizes


def v_cycle(
    phi: np.ndarray,
    rho: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    levels: int,
    g3: G3Globals,
    mobil: MobilityParams,
    origin: Tuple[float, float, float],
    bc: BC,
    sigma_loc_pc2: Optional[np.ndarray] = None,
) -> None:
    """One multigrid V‑cycle with a few relaxations per level."""
    # pre‑smooth
    gauss_seidel_relax(phi, rho, dx, dy, dz, g3, mobil, origin, bc, sigma_loc_pc2, n_sweeps=2)

    if levels > 1 and min(phi.shape) >= 4:
        # residual on fine grid
        R, _ = apply_operator(phi, rho, dx, dy, dz, g3, mobil, origin, bc, sigma_loc_pc2)
        # restrict residual and rho to coarse (operator‑induced coarse RHS)
        R_c = restrict_2x(R)
        rho_c = restrict_2x(rho)
        phi_c = np.zeros_like(R_c)

        # coarse grid spacings
        v_cycle(phi_c, rho_c, 2*dx, 2*dy, 2*dz, levels-1, g3, mobil, origin, bc, sigma_loc_pc2=None)

        # prolongate correction and add
        phi += prolong_2x(phi_c, phi.shape)

    # post‑smooth
    gauss_seidel_relax(phi, rho, dx, dy, dz, g3, mobil, origin, bc, sigma_loc_pc2, n_sweeps=2)


def solve_g3_3d(
    rho: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    g3: Optional[G3Globals] = None,
    mobil: Optional[MobilityParams] = None,
    bc: Optional[BC] = None,
    levels: Optional[int] = None,
    sigma_window_cells: int = 8,
    use_sigma_screen: bool = False,
    max_cycles: int = 20,
    tol: float = 1e-5,
    verbose: bool = True,
) -> Dict[str, np.ndarray | float | Dict[str, float]]:
    """
    Solve the G³ PDE for the given baryon density grid.

    Returns a dict with:
        phi, gx, gy, gz, g_abs, scalars_dict
    """
    if g3 is None:
        g3 = G3Globals()
    if mobil is None:
        mobil = MobilityParams(use_sigma_screen=use_sigma_screen)
    if bc is None:
        bc = BC(kind="robin", lambda_kpc=1e6)

    if levels is None:
        # choose depth so that the coarsest grid is >= 8^3
        n_min = min(rho.shape)
        lev = 1
        while n_min >= 16 and (n_min % 2) == 0 and lev < 10:
            lev += 1
            n_min //= 2
        levels = lev

    nx, ny, nz = rho.shape
    phi = np.zeros_like(rho, dtype=np.float64)

    sigma_loc_pc2 = None
    if mobil.use_sigma_screen:
        sigma_loc_pc2 = sigma_local_pc2(rho, dz, window_cells=sigma_window_cells)

    last_resid = None
    for cyc in range(max_cycles):
        v_cycle(phi, rho, dx, dy, dz, levels, g3, mobil, origin, bc, sigma_loc_pc2=sigma_loc_pc2)
        R, scalars = apply_operator(phi, rho, dx, dy, dz, g3, mobil, origin, bc, sigma_loc_pc2=sigma_loc_pc2)
        resid = float(np.linalg.norm(R) / max(np.linalg.norm((4.0 * math.pi * G_KPC) * rho), 1e-30))
        if verbose:
            print(f"[solve_g3_3d] cycle {cyc+1:02d}  residual={resid:.3e}  rc_eff={scalars['rc_eff_kpc']:.2f} kpc  S0_eff={scalars['S0_eff']:.3e}")
        if last_resid is not None and abs(resid - last_resid) < tol * max(1.0, last_resid):
            break
        last_resid = resid

    gx, gy, gz = gradient_3d(phi, dx, dy, dz)
    g_abs = np.sqrt(gx*gx + gy*gy + gz*gz)

    return {
        'phi': phi,
        'gx': gx,
        'gy': gy,
        'gz': gz,
        'g_abs': g_abs,
        'scalars': scalars,
        'globals': asdict(g3),
        'mobility': asdict(mobil),
        'bc': asdict(bc)
    }


# ---- Simple self‑test -------------------------------------------------------

def _gaussian_blob(nx=64, ny=64, nz=64, dx=2.0, dy=2.0, dz=2.0, mass=1e11, r0=10.0):
    """Return a normalized 3D Gaussian density blob with total mass 'mass'."""
    xs = (np.arange(nx) + 0.5) * dx - 0.5 * nx * dx
    ys = (np.arange(ny) + 0.5) * dy - 0.5 * ny * dy
    zs = (np.arange(nz) + 0.5) * dz - 0.5 * nz * dz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    r2 = X**2 + Y**2 + Z**2
    rho = np.exp(-r2 / (2*r0*r0))
    # normalize to requested mass
    dV = dx * dy * dz
    rho *= mass / (rho.sum() * dV)
    return rho, dx, dy, dz, (0.0, 0.0, 0.0)


if __name__ == "__main__":
    # Smoke test: solve for a single Gaussian blob
    rho, dx, dy, dz, origin = _gaussian_blob()
    g3 = G3Globals()
    mobil = MobilityParams(use_saturating_mobility=True, g_sat_kms2_per_kpc=2500.0, n_sat=2.0,
                           use_sigma_screen=False)
    bc = BC(kind="robin", lambda_kpc=1e6)

    out = solve_g3_3d(rho, dx, dy, dz, origin, g3, mobil, bc, verbose=True, max_cycles=10)
    print("Scalars:", json.dumps(out['scalars'], indent=2))
    print("Globals:", json.dumps(out['globals'], indent=2))
