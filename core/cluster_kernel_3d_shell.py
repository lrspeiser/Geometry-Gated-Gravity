#!/usr/bin/env python3
"""
3D Shell Integral Kernel for Cluster Lensing
=============================================

Implements full 3D shell integration for path-spectrum boost, accounting for:
- Interior chord families (through-core paths)
- Exterior shell contributions (up-and-over arcs)
- Off-axis scattering (not just radial rings)

This is the Phase 2, Step 2.1 upgrade from PHYSICS_ROADMAP.md.

Physical Motivation
-------------------
In the path-integral picture, gravitational lensing receives contributions from
ALL matter in the cluster, not just matter in the direct line-of-sight ring at
radius R. Specifically:

1. **Interior chords**: Matter at r < R_E contributes via paths passing through
   or near the dense core, with significant impact at the Einstein radius.

2. **Exterior shells**: Matter at r > R_E contributes via paths that curve up
   and over the core, accumulating phase along extended trajectories.

3. **Full 3D geometry**: Coherence between paths depends on 3D separation and
   density contrast, not just projected distance.

Mathematical Framework
---------------------
For lensing at projected radius R, the effective surface density boost is:

    K_Σ(R) = ∫ K_3D(R, r_shell) × ρ(r_shell) × W(R, r_shell) dr_shell

where:
- K_3D(R, r_shell): 3D kernel encoding coherence between field point and source
- W(R, r_shell): Geometric projection weight (interior vs exterior)
- ρ(r_shell): 3D density at source shell radius

For r_shell < R (interior):
    W_interior ∝ chord_length(R, r_shell) × coherence_damping

For r_shell > R (exterior):
    W_exterior ∝ path_curvature(R, r_shell) × coherence_damping

Implementation Strategy
-----------------------
1. Shell-by-shell integration over 3D density profile
2. Separate handling of interior (r < R) and exterior (r > R) regimes
3. Coherence damping with cluster-scale ℓ₀ ~ 100-300 kpc
4. Density-dependent constructive interference

References
----------
- PHYSICS_ROADMAP.md: Phase 2 implementation plan
- Arnaud+ 2010: gNFW gas profile (used with this kernel)
- Feynman & Hibbs: Path integrals in quantum mechanics (analogy)

Author: GravityCalculator Phase 2 Upgrade
Date: 2025-01-13
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import interp1d


@dataclass
class Shell3DKernelParams:
    """Parameters for 3D shell integral kernel."""
    A_c: float = 10.0       # Cluster amplitude
    r_gate: float = 5.0     # Gate radius [kpc] - Newtonian preservation
    n_gate: int = 4         # Gate steepness
    ell0: float = 180.0     # Coherence length [kpc] - cluster scale
    p_density: float = 1.2  # Density-dependent constructive interference exponent
    L1: float = 1200.0      # Large-scale taper [kpc]
    q_taper: float = 2.0    # Taper steepness
    
    # NEW: Interior/exterior path family weights
    w_interior: float = 1.0  # Weight for interior chord families
    w_exterior: float = 1.0  # Weight for exterior arc families
    
    # Coherence damping mode
    coherence_mode: str = 'power_law'  # 'exponential' or 'power_law'
    n_coh: float = 1.5      # Coherence damping exponent (for power_law mode)


def chord_length_through_sphere(R: float, r_shell: float) -> float:
    """
    Compute chord length through a spherical shell at radius r_shell
    viewed from projected radius R.
    
    For R < r_shell (exterior view):
        Returns 0 (no intersection)
    
    For R > r_shell (interior view):
        Chord length = 2 × sqrt(r_shell² - R²)
    
    Parameters
    ----------
    R : float
        Projected radius [kpc]
    r_shell : float
        3D shell radius [kpc]
    
    Returns
    -------
    L_chord : float
        Chord length [kpc]
    """
    if R > r_shell:
        return 0.0
    else:
        return 2.0 * np.sqrt(r_shell**2 - R**2)


def exterior_path_curvature_weight(R: float, r_shell: float, ell0: float) -> float:
    """
    Weight for exterior shell contributions (r_shell > R).
    
    Paths from exterior shells must curve "up and over" the dense core,
    accumulating extra geometric phase. The weight decreases with distance
    and coherence length.
    
    Physical interpretation:
    - Nearby shells (r ~ R): strong contribution, short path deviation
    - Distant shells (r >> R): weak contribution, long incoherent paths
    
    Parameters
    ----------
    R : float
        Projected radius [kpc]
    r_shell : float
        3D shell radius [kpc]
    ell0 : float
        Coherence length [kpc]
    
    Returns
    -------
    W_ext : float
        Geometric weight (dimensionless)
    """
    if r_shell <= R:
        return 0.0
    
    # Geometric weight: falls off with separation
    # Simple model: inverse distance weighted by coherence
    dr = r_shell - R
    W = r_shell / (r_shell**2 + ell0**2)  # Peaks near R, tapers with ell0
    
    return W


def coherence_damping(
    separation: float,
    ell0: float,
    mode: str = 'power_law',
    n_coh: float = 1.5
) -> float:
    """
    Coherence damping factor for path contributions.
    
    Two modes:
    1. 'exponential': exp(-separation/ell0)
       - Strong damping, ~95% suppression at 3×ell0
       - Good for cold, coherent systems
    
    2. 'power_law': (ell0/(ell0 + separation))^n_coh
       - Gentler damping, tunable with n_coh
       - Better for hot, turbulent clusters
       - n_coh=1: 50% at ell0, 9% at 10×ell0
       - n_coh=2: 25% at ell0, 0.8% at 10×ell0
    
    Parameters
    ----------
    separation : float
        Path length or effective separation [kpc]
    ell0 : float
        Coherence length [kpc]
    mode : str
        'exponential' or 'power_law'
    n_coh : float
        Exponent for power_law mode
    
    Returns
    -------
    C : float
        Coherence factor in [0, 1]
    """
    if mode == 'exponential':
        return np.exp(-separation / ell0)
    elif mode == 'power_law':
        return (ell0 / (ell0 + separation))**n_coh
    else:
        raise ValueError(f"Unknown coherence mode: {mode}")


def interior_contribution(
    R: float,
    r_shells_interior: np.ndarray,
    rho_interior: np.ndarray,
    params: Shell3DKernelParams
) -> float:
    """
    Compute interior chord family contribution to K_Σ(R).
    
    Integrates over all shells with r < R, weighting by:
    - Chord length through shell
    - Density at shell
    - Coherence damping
    
    Parameters
    ----------
    R : float
        Projected radius [kpc]
    r_shells_interior : ndarray
        Shell radii with r < R [kpc]
    rho_interior : ndarray
        Density at interior shells [Msun/kpc³]
    params : Shell3DKernelParams
        Kernel parameters
    
    Returns
    -------
    K_interior : float
        Interior contribution (dimensionless)
    """
    if len(r_shells_interior) == 0:
        return 0.0
    
    integrand = np.zeros_like(r_shells_interior)
    
    for i, r_s in enumerate(r_shells_interior):
        # Chord length
        L_chord = chord_length_through_sphere(R, r_s)
        
        # Coherence damping (based on chord length as effective path)
        C_damp = coherence_damping(
            L_chord, params.ell0, params.coherence_mode, params.n_coh
        )
        
        # Density weighting (normalized, dimensionless)
        rho_factor = rho_interior[i]**params.p_density
        
        # Dimensionless weight: chord length normalized by coherence
        # This gives ~O(1) contributions per shell
        weight = (L_chord / params.ell0) * C_damp * rho_factor
        
        # Shell surface area factor (for proper weighting, not integration measure)
        # Normalized by R² to make dimensionless
        area_factor = (r_s / R)**2
        
        # Accumulate (dimensionless integrand)
        integrand[i] = weight * area_factor
    
    # Integrate over interior shells (dimensionless dr/R)
    if len(integrand) > 1:
        # Dimensionless radial coordinate
        r_norm = r_shells_interior / R
        K_int = simpson(integrand, x=r_norm)
    else:
        K_int = 0.0
    
    return params.w_interior * K_int


def exterior_contribution(
    R: float,
    r_shells_exterior: np.ndarray,
    rho_exterior: np.ndarray,
    params: Shell3DKernelParams
) -> float:
    """
    Compute exterior shell contribution to K_Σ(R).
    
    Integrates over all shells with r > R, weighting by:
    - Path curvature weight (up-and-over arcs)
    - Density at shell
    - Coherence damping
    
    Parameters
    ----------
    R : float
        Projected radius [kpc]
    r_shells_exterior : ndarray
        Shell radii with r > R [kpc]
    rho_exterior : ndarray
        Density at exterior shells [Msun/kpc³]
    params : Shell3DKernelParams
        Kernel parameters
    
    Returns
    -------
    K_exterior : float
        Exterior contribution (dimensionless)
    """
    if len(r_shells_exterior) == 0:
        return 0.0
    
    integrand = np.zeros_like(r_shells_exterior)
    
    for i, r_s in enumerate(r_shells_exterior):
        # Separation for coherence damping
        dr = r_s - R
        C_damp = coherence_damping(
            dr, params.ell0, params.coherence_mode, params.n_coh
        )
        
        # Density weighting (normalized, dimensionless)
        rho_factor = rho_exterior[i]**params.p_density
        
        # Dimensionless weight: separation normalized by coherence
        weight = (dr / params.ell0) * C_damp * rho_factor
        
        # Shell area factor (dimensionless)
        area_factor = (r_s / R)**2
        
        # Accumulate (dimensionless integrand)
        integrand[i] = weight * area_factor
    
    # Integrate over exterior shells (dimensionless)
    if len(integrand) > 1:
        # Dimensionless radial coordinate
        r_norm = r_shells_exterior / R
        K_ext = simpson(integrand, x=r_norm)
    else:
        K_ext = 0.0
    
    return params.w_exterior * K_ext


def K_Sigma_3D_shell(
    R: np.ndarray,
    r_grid: np.ndarray,
    rho_3d: np.ndarray,
    params: Shell3DKernelParams,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute surface density boost K_Σ(R) via full 3D shell integration.
    
    This is the main interface for the upgraded cluster kernel.
    
    Parameters
    ----------
    R : ndarray
        Projected radii [kpc] where K_Σ is evaluated
    r_grid : ndarray
        3D radial grid [kpc] for density profile
    rho_3d : ndarray
        3D density profile [Msun/kpc³] on r_grid
    params : Shell3DKernelParams
        Kernel parameters
    normalize : bool
        If True, normalize by a characteristic density scale
    
    Returns
    -------
    K_Sigma : ndarray
        Surface density boost K_Σ(R) (dimensionless)
        Usage: Σ_eff(R) = Σ(R) × [1 + K_Σ(R)]
    
    Notes
    -----
    The boost is computed as:
        K_Σ(R) = A_c × gate(R) × [K_interior(R) + K_exterior(R)] × taper(R)
    
    where gate and taper preserve Newtonian limits and prevent divergence.
    """
    K_Sigma = np.zeros_like(R)
    
    # Normalization density scale (for dimensionless weighting)
    if normalize:
        rho_norm = np.median(rho_3d[rho_3d > 0]) if np.any(rho_3d > 0) else 1.0
    else:
        rho_norm = 1.0
    
    # Interpolate density for smooth queries
    rho_interp = interp1d(r_grid, rho_3d, kind='linear', 
                          bounds_error=False, fill_value=0.0)
    
    for i, R_eval in enumerate(R):
        # Split into interior and exterior shells
        mask_interior = r_grid < R_eval
        mask_exterior = r_grid >= R_eval
        
        r_int = r_grid[mask_interior]
        rho_int = rho_3d[mask_interior] / rho_norm
        
        r_ext = r_grid[mask_exterior]
        rho_ext = rho_3d[mask_exterior] / rho_norm
        
        # Compute contributions
        K_int = interior_contribution(R_eval, r_int, rho_int, params)
        K_ext = exterior_contribution(R_eval, r_ext, rho_ext, params)
        
        # Total boost (before gates/tapers)
        K_raw = K_int + K_ext
        
        # Apply gates and tapers
        # Small-radius gate (preserves Newtonian limit)
        gate = (R_eval / params.r_gate)**params.n_gate / (
            1.0 + (R_eval / params.r_gate)**params.n_gate
        )
        
        # Large-scale taper (prevents divergence)
        taper = 1.0 / (1.0 + (R_eval / params.L1)**params.q_taper)
        
        # Final boost
        K_Sigma[i] = params.A_c * gate * K_raw * taper
    
    return K_Sigma


def lensing_profiles_3d_shell(
    R: np.ndarray,
    z_lens: float,
    z_src: float,
    r_grid: np.ndarray,
    rho_3d: np.ndarray,
    params: Shell3DKernelParams,
    cosmo,
    verbose: bool = False
) -> Dict:
    """
    Compute full lensing profiles with 3D shell kernel.
    
    Parameters
    ----------
    R : ndarray
        Projected radii [kpc]
    z_lens : float
        Cluster redshift
    z_src : float
        Source redshift
    r_grid : ndarray
        3D radial grid [kpc]
    rho_3d : ndarray
        3D baryon density [Msun/kpc³]
    params : Shell3DKernelParams
        Kernel parameters
    cosmo : object
        Cosmology with critical_surface_density and physical_to_angular
    verbose : bool
        Print diagnostics
    
    Returns
    -------
    profiles : dict
        - R: projected radii [kpc]
        - Sigma: baseline surface density [Msun/kpc²]
        - K_Sigma: boost factor (dimensionless)
        - Sigma_eff: effective surface density [Msun/kpc²]
        - kappa: convergence
        - mean_kappa: mean convergence
        - gamma_t: tangential shear
        - theta_E_arcsec: Einstein radius [arcsec]
    """
    from many_path_model.lensing_utilities import AbelProjection
    
    # Project baseline surface density
    projector = AbelProjection()
    Sigma_on_rgrid = projector.project_density_to_surface(r_grid, rho_3d, r_grid)
    
    # Interpolate to R grid
    f_Sigma = interp1d(r_grid, Sigma_on_rgrid, kind='linear',
                       bounds_error=False, fill_value=0.0)
    Sigma = f_Sigma(R)
    
    # Compute 3D shell boost
    K_Sigma = K_Sigma_3D_shell(R, r_grid, rho_3d, params)
    
    # Effective surface density
    Sigma_eff = Sigma * (1.0 + K_Sigma)
    
    # Critical surface density
    from many_path_model.lensing_utilities import critical_surface_density
    Sigma_crit = critical_surface_density(z_lens, z_src, cosmo)
    
    # Convergence
    kappa = Sigma_eff / Sigma_crit
    
    # Mean convergence (cumulative average)
    area = np.pi * R**2
    cum = np.cumsum((Sigma_eff[1:] + Sigma_eff[:-1]) / 2.0 * np.diff(area))
    mean_kappa = np.zeros_like(kappa)
    mean_kappa[1:] = (cum / area[1:]) / Sigma_crit
    mean_kappa[0] = mean_kappa[1]
    
    # Tangential shear
    gamma_t = mean_kappa - kappa
    
    # Einstein radius
    idx = np.where(mean_kappa >= 1.0)[0]
    theta_E_arcsec = 0.0
    if idx.size > 0:
        i = idx[-1]
        theta_E_arcsec = cosmo.physical_to_angular(R[i], z_lens)
    
    if verbose:
        print(f"3D Shell Kernel Lensing:")
        print(f"  Einstein radius: {theta_E_arcsec:.2f} arcsec")
        print(f"  Max ⟨κ⟩: {np.max(mean_kappa):.3f}")
        print(f"  K_Σ at R_E: {K_Sigma[i] if idx.size > 0 else 0:.2f}")
    
    return {
        'R': R,
        'Sigma': Sigma,
        'K_Sigma': K_Sigma,
        'Sigma_eff': Sigma_eff,
        'kappa': kappa,
        'mean_kappa': mean_kappa,
        'gamma_t': gamma_t,
        'theta_E_arcsec': theta_E_arcsec,
        'Sigma_crit': Sigma_crit
    }


if __name__ == '__main__':
    # Test the 3D shell kernel
    print("=" * 70)
    print("3D Shell Integral Kernel Test")
    print("=" * 70)
    print()
    
    # Create test cluster profile
    r_test = np.logspace(0, 3.5, 500)  # 1 to ~3000 kpc
    
    # Simple NFW-like profile
    r_s = 300.0  # kpc
    rho_0 = 1e7  # Msun/kpc³
    rho_test = rho_0 / ((r_test/r_s) * (1 + r_test/r_s)**2)
    
    # Kernel parameters
    params = Shell3DKernelParams(
        A_c=10.0,
        r_gate=5.0,
        n_gate=4,
        ell0=180.0,
        p_density=1.2,
        L1=1200.0,
        w_interior=1.0,
        w_exterior=1.0,
        coherence_mode='power_law',
        n_coh=1.5
    )
    
    print("Kernel parameters:")
    print(f"  A_c = {params.A_c}")
    print(f"  ell0 = {params.ell0} kpc")
    print(f"  Coherence mode = {params.coherence_mode}")
    print()
    
    # Compute K_Sigma
    R_eval = np.array([50, 100, 180, 300, 500])
    K_Sigma = K_Sigma_3D_shell(R_eval, r_test, rho_test, params)
    
    print("Surface density boost K_Σ(R):")
    for i, R_val in enumerate(R_eval):
        print(f"  R = {R_val:4.0f} kpc: K_Σ = {K_Sigma[i]:.3f}")
    
    print()
    print("✓ 3D shell kernel test complete!")
    print()
    print("Next: Test on MACS0416 with gNFW gas profile")
