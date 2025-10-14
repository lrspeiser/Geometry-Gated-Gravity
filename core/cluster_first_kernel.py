# core/cluster_first_kernel.py
"""
Cluster-first, isotropic path-spectrum boost for lensing.

This module implements a clean cluster-specific kernel that does NOT reuse
galaxy-tuned parameters. The key differences from galaxy kernels:
  - Isotropic (no disk anisotropy)
  - Long coherence length ell0 ~ 100-300 kpc (vs ~5 kpc for galaxies)
  - Hot, pressure-supported system assumptions
  - Boost cannot collapse to zero by construction

The 3D boost kernel is:
    K_3D(r) = A_c * gate(r) * growth(r) * taper(r)

Where:
  - gate: Local Newtonian preservation (r_g ~ 5-10 kpc)
  - growth: Path accumulation over coherence length ell0
  - taper: Large-scale saturation beyond L1 ~ Mpc

Physical motivation: In a multi-path picture, clusters with large dynamical
scales and hot ICM should accumulate many stationary-phase loop paths over
their ~100-1000 kpc extent, producing measurable lensing boost.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from many_path_model.lensing_utilities import abel_project, critical_surface_density


def K3D_isotropic(
    r: np.ndarray,
    A_c: float = 8.0,
    r_gate: float = 5.0,
    n_gate: int = 4,
    ell0: float = 150.0,
    p: float = 1.2,
    L1: float = 1200.0,
    q: float = 2.0
) -> np.ndarray:
    """
    3D cluster boost kernel (dimensionless).
    
    Parameters
    ----------
    r : array_like
        Radial distance in kpc
    A_c : float
        Cluster amplitude (dimensionless), controls overall boost strength
    r_gate : float
        Gate radius in kpc, preserves Newtonian limit at small scales
    n_gate : int
        Gate steepness parameter (higher = sharper turn-on)
    ell0 : float
        Coherence length in kpc (cluster scale: 100-300 kpc typical)
    p : float
        Growth power (controls how quickly paths accumulate with radius)
    L1 : float
        Large-scale taper length in kpc (~Mpc scale)
    q : float
        Taper steepness
    
    Returns
    -------
    K_3D : ndarray
        Dimensionless 3D boost factor at each radius
    
    Notes
    -----
    The kernel structure ensures:
    - K_3D → 0 as r → 0 (Newtonian limit via gate)
    - K_3D grows with r over scale ell0 (path accumulation)
    - K_3D saturates/tapers at r >> L1 (prevents divergence)
    - K_3D cannot be identically zero unless A_c = 0
    """
    r = np.asarray(r, dtype=float)
    
    # Gate: turns on above r_gate, preserving Newtonian regime
    gate = (r / r_gate)**n_gate / (1.0 + (r / r_gate)**n_gate)
    
    # Growth: rises as (r/ell0)^p at small r, approaches 1 at large r
    # This captures path accumulation over coherence length
    growth = ((1.0 + (r/ell0)**p) - 1.0) / (1.0 + (r/ell0)**p)
    
    # Taper: gentle saturation at large scales
    taper = 1.0 / (1.0 + (r/L1)**q)
    
    return A_c * gate * growth * taper


def project_boost_Sigma(
    R: np.ndarray,
    r_grid: np.ndarray,
    rho_r: np.ndarray,
    K3D_params: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute projected boost and surface densities via Abel transform.
    
    Parameters
    ----------
    R : ndarray
        Cylindrical radii in kpc where profiles are evaluated
    r_grid : ndarray
        Monotonic 3D radii in kpc for density profile
    rho_r : ndarray
        Baryon density profile in Msun/kpc^3 (BCG + ICM + galaxies)
    K3D_params : dict
        Parameters for K3D_isotropic function
    
    Returns
    -------
    K_Sigma : ndarray
        Projected boost factor K_Sigma(R) = Sigma_K / Sigma (dimensionless)
    Sigma : ndarray
        Baseline projected surface density in Msun/kpc^2
    Sigma_K : ndarray
        Boosted surface density component in Msun/kpc^2
    
    Notes
    -----
    The effective surface density for lensing is:
        Sigma_eff(R) = (1 + K_Sigma(R)) * Sigma(R)
    
    This comes from projecting the 3D boosted density:
        Sigma_K(R) = 2 * integral_R^infty [rho(r) * K_3D(r) * r * dr / sqrt(r^2 - R^2)]
    """
    from many_path_model.lensing_utilities import AbelProjection
    from scipy.interpolate import interp1d
    
    projector = AbelProjection()
    
    # Base surface density Sigma(R) projected on the r_grid, then interpolate to R
    Sigma_on_rgrid = projector.project_density_to_surface(r_grid, rho_r, r_grid)
    f_Sigma = interp1d(r_grid, Sigma_on_rgrid, kind='linear', 
                       bounds_error=False, fill_value=0.0)
    Sigma = f_Sigma(R)
    
    # Boosted component: project rho(r) * K_3D(r)
    K_r = K3D_isotropic(r_grid, **K3D_params)
    Sigma_K_on_rgrid = projector.project_density_to_surface(r_grid, rho_r * K_r, r_grid)
    f_Sigma_K = interp1d(r_grid, Sigma_K_on_rgrid, kind='linear',
                         bounds_error=False, fill_value=0.0)
    Sigma_K = f_Sigma_K(R)
    
    # Relative boost factor (avoid divide by zero at very small R)
    K_Sigma = np.where(Sigma > 0, Sigma_K / Sigma, 0.0)
    
    return K_Sigma, Sigma, Sigma_K


def lensing_profiles(
    R: np.ndarray,
    z_lens: float,
    z_src: float,
    r_grid: np.ndarray,
    rho_r: np.ndarray,
    K3D_params: Dict,
    cosmo
) -> Dict:
    """
    Compute full lensing profiles with cluster-first boost kernel.
    
    Parameters
    ----------
    R : ndarray
        Cylindrical radii in kpc for profile evaluation
    z_lens : float
        Cluster redshift
    z_src : float
        Source redshift (for critical surface density)
    r_grid : ndarray
        3D radii in kpc for density profile
    rho_r : ndarray
        Total baryon density in Msun/kpc^3
    K3D_params : dict
        Cluster kernel parameters
    cosmo : object
        Cosmology instance with kpc_to_arcsec method
    
    Returns
    -------
    profiles : dict
        Dictionary containing:
        - R: cylindrical radii (kpc)
        - Sigma: baseline surface density (Msun/kpc^2)
        - K_Sigma: projected boost factor (dimensionless)
        - Sigma_eff: effective surface density (Msun/kpc^2)
        - kappa: convergence κ(R)
        - mean_kappa: mean convergence ⟨κ⟩(<R)
        - gamma_t: tangential shear γ_t = ⟨κ⟩ - κ
        - theta_E_arcsec: Einstein radius in arcsec (0 if none found)
        - Sigma_crit: critical surface density (Msun/kpc^2)
    
    Notes
    -----
    Einstein radius is defined by ⟨κ⟩(θ_E) = 1.
    This implementation finds the outermost crossing point.
    """
    # Project boosts
    K_Sigma, Sigma, Sigma_K = project_boost_Sigma(R, r_grid, rho_r, K3D_params)
    Sigma_eff = (1.0 + K_Sigma) * Sigma
    
    # Critical surface density
    Sigma_crit = critical_surface_density(z_lens=z_lens, z_src=z_src, cosmo=cosmo)
    
    # Convergence κ(R) = Σ_eff(R) / Σ_crit
    kappa = Sigma_eff / Sigma_crit
    
    # Mean convergence ⟨κ⟩(<R) via cumulative average in area
    area = np.pi * R**2
    # Trapezoid integration in area
    cum = np.cumsum((Sigma_eff[1:] + Sigma_eff[:-1]) / 2.0 * np.diff(area))
    mean_kappa = np.zeros_like(kappa)
    mean_kappa[1:] = (cum / area[1:]) / Sigma_crit
    mean_kappa[0] = mean_kappa[1]  # extrapolate to avoid singularity
    
    # Tangential shear γ_t = ⟨κ⟩ - κ
    gamma_t = mean_kappa - kappa
    
    # Einstein radius: find where ⟨κ⟩(R) = 1
    idx = np.where(mean_kappa >= 1.0)[0]
    theta_E_arcsec = 0.0
    if idx.size > 0:
        # Take outermost crossing for robustness
        i = idx[-1]
        theta_E_arcsec = cosmo.physical_to_angular(R[i], z_lens)
    
    return dict(
        R=R,
        Sigma=Sigma,
        K_Sigma=K_Sigma,
        Sigma_eff=Sigma_eff,
        kappa=kappa,
        mean_kappa=mean_kappa,
        gamma_t=gamma_t,
        theta_E_arcsec=theta_E_arcsec,
        Sigma_crit=Sigma_crit
    )
