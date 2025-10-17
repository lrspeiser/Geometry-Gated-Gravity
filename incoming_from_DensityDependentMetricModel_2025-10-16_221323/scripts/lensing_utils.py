#!/usr/bin/env python3
"""
lensing_utils.py

Cosmology utilities for lensing distance calculations, including
support for effective source redshift distributions P(z_s).
"""
from __future__ import annotations

from typing import Optional, Dict, List, Tuple
import numpy as np

try:
    from astropy.cosmology import Planck18
    HAS_ASTROPY = True
except Exception:
    HAS_ASTROPY = False


def angular_diameter_distance(z: float) -> float:
    """Planck18 cosmology angular diameter distance in Mpc."""
    if not HAS_ASTROPY:
        raise ImportError("astropy is required for cosmology distances. Install with: pip install astropy")
    return float(Planck18.angular_diameter_distance(z).value)


def angular_diameter_distance_between(z1: float, z2: float) -> float:
    """Angular diameter distance between two redshifts in Mpc."""
    if not HAS_ASTROPY:
        raise ImportError("astropy is required for cosmology distances. Install with: pip install astropy")
    D1 = Planck18.angular_diameter_distance(z1).value
    D2 = Planck18.angular_diameter_distance(z2).value
    D12 = (D2 * (1 + z2) - D1 * (1 + z1)) / (1 + z2)
    return float(D12)


def _mixture_normal_pdf(z: np.ndarray, components: List[Dict], weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Evaluate a mixture of Gaussians PDF at z with optional weights array."""
    pdf = np.zeros_like(z, dtype=float)
    K = len(components)
    if weights is None:
        # use component-provided weights (may be unnormalized)
        weights = np.array([max(float(c.get('weight', 0.0)), 0.0) for c in components], dtype=float)
        if np.sum(weights) <= 0:
            weights = np.ones(K, dtype=float)
        weights = weights / np.sum(weights)
    for k, comp in enumerate(components):
        w = float(weights[k])
        mu = float(comp.get('mu', 1.0))
        sigma = float(comp.get('sigma', 0.1))
        if sigma <= 0 or w <= 0:
            continue
        pdf = pdf + w * (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((z - mu)/sigma)**2)
    # Normalize to integral 1 over the grid
    norm = np.trapz(pdf, z)
    if norm > 0:
        pdf = pdf / norm
    return pdf


def _beta_eff(z_lens: float, z_grid: np.ndarray, p_z: np.ndarray) -> float:
    """Compute effective beta = <D_ls/D_s> over the source redshift PDF p_z."""
    mask = z_grid > z_lens
    if not np.any(mask):
        return 0.0
    z_use = z_grid[mask]
    p_use = p_z[mask]

    D_ls = np.array([angular_diameter_distance_between(z_lens, zs) for zs in z_use])
    D_s = np.array([angular_diameter_distance(zs) for zs in z_use])
    beta = D_ls / D_s

    num = np.trapz(beta * p_use, z_use)
    den = np.trapz(p_use, z_use)
    return float(num / den) if den > 0 else 0.0


def effective_distances(z_lens: float, z_source: Optional[float] = None, override: Optional[Dict] = None) -> Tuple[float, float, float]:
    """
    Return (D_lens, D_source_eff, D_LS_eff) in Mpc.

    If override includes a source_distribution, compute an effective lensing efficiency
    beta_eff = <D_ls/D_s> over P(z_s) and set D_source_eff=1, D_LS_eff=beta_eff.
    Otherwise, use the single provided z_source.
    """
    D_lens = angular_diameter_distance(z_lens)

    # If override provides explicit single source redshift
    if override and 'z_source' in override and isinstance(override['z_source'], (int, float)):
        z_s = float(override['z_source'])
        D_source = angular_diameter_distance(z_s)
        D_LS = angular_diameter_distance_between(z_lens, z_s)
        return D_lens, D_source, D_LS

    # Source distribution handling
    if override and 'source_distribution' in override:
        sd = override['source_distribution']
        z_min = float(sd.get('z_min', max(0.01, z_lens + 0.01)))
        z_max = float(sd.get('z_max', 6.0))
        n_grid = int(sd.get('n_grid', 400))
        z_grid = np.linspace(z_min, z_max, n_grid)

        sd_type = str(sd.get('type', 'mixture_normal')).lower()
        if sd_type == 'mixture_normal':
            components = sd.get('components', [])
            # Optional Dirichlet weight sampling
            weights = None
            if 'dirichlet_alpha' in sd:
                alpha = np.array(sd['dirichlet_alpha'], dtype=float)
                if alpha.ndim == 0:
                    alpha = np.full(len(components), float(alpha))
                if len(alpha) != len(components):
                    alpha = np.ones(len(components), dtype=float)
                w = np.random.dirichlet(alpha)
                weights = w
            p_z = _mixture_normal_pdf(z_grid, components, weights=weights)
        else:
            # Fallback: delta at provided z_source or narrow Gaussian above z_lens
            z0 = float(z_source if z_source is not None else (z_lens + 0.5))
            p_z = _mixture_normal_pdf(z_grid, [{"weight": 1.0, "mu": z0, "sigma": 0.1}])

        beta = _beta_eff(z_lens, z_grid, p_z)
        # Effective distances: D_source_eff/D_LS_eff = 1/beta
        return D_lens, 1.0, beta

    # Default single source redshift
    if z_source is None:
        raise ValueError("z_source must be provided if no source_distribution override is present")
    D_source = angular_diameter_distance(z_source)
    D_LS = angular_diameter_distance_between(z_lens, z_source)
    return D_lens, D_source, D_LS
