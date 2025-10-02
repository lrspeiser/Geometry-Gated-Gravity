"""
hot_gas_model.py

Extended O2 model with hot gas fraction gating.

Model form:
    fX = (x^2 / 2) * (1 / denom)
    
    denom = a - b * Σ̂ - d * |∇ln Σ| - f * fgas
    
Where:
    - a, b, d: O2 baseline parameters (fixed from galaxy fitting)
    - f: hot gas fraction coefficient (new parameter)
    - fgas: hot gas mass fraction (0-1, from observations)

Physical motivation:
    - Galaxy clusters have 10-15% baryonic mass in hot X-ray gas
    - Galaxies have ~0% hot gas (cold ISM instead)
    - Hot gas doesn't contribute to lensing surface density
    - Should reduce predicted lensing strength for clusters
    
Author: Henry Speiser
Date: October 2, 2025
"""

import numpy as np
from typing import Tuple


def fX_ratio_curv_fgas(
    params: Tuple[float, float, float, float],
    x: float,
    Sigma_hat: float,
    grad_ln_Sigma: float,
    fgas: float
) -> float:
    """
    Compute fX with hot gas fraction gating.
    
    Parameters
    ----------
    params : tuple of (a, b, d, f)
        a : baseline denominator constant (O2 value: 0.6687)
        b : surface density coefficient (O2 value: 0.1401)
        d : gradient coefficient (O2 value: 0.0871)
        f : hot gas fraction coefficient (NEW, to be fitted)
    
    x : float
        Dimensionless radius (r / r_turnaround)
    
    Sigma_hat : float
        Normalized surface density, log10(Σ / Σ_crit)
    
    grad_ln_Sigma : float
        Absolute value of logarithmic surface density gradient
    
    fgas : float
        Hot gas mass fraction (0-1)
        - Clusters: ~0.10 to 0.15 (10-15%)
        - Galaxies: ~0.00 (no hot gas)
    
    Returns
    -------
    fX : float
        Lensing amplification factor
        Returns np.nan if denominator is negative (unphysical)
    """
    a, b, d, f = params
    
    # Denominator with hot gas gating
    denom = a - b * Sigma_hat - d * abs(grad_ln_Sigma) - f * fgas
    
    if denom <= 0:
        # Model breaks down (unphysical regime)
        return np.nan
    
    # Standard O2 numerator
    fX = (x**2 / 2.0) / denom
    
    return fX


def fX_ratio_curv_fgas_vectorized(
    params: Tuple[float, float, float, float],
    x: np.ndarray,
    Sigma_hat: np.ndarray,
    grad_ln_Sigma: np.ndarray,
    fgas: np.ndarray
) -> np.ndarray:
    """
    Vectorized version for batch processing.
    
    Parameters
    ----------
    params : tuple of (a, b, d, f)
        Model parameters
    
    x, Sigma_hat, grad_ln_Sigma, fgas : np.ndarray
        Arrays of input values (must be same shape)
    
    Returns
    -------
    fX : np.ndarray
        Lensing amplification factors
        NaN where denominator is negative
    """
    a, b, d, f = params
    
    # Compute denominator
    denom = a - b * Sigma_hat - d * np.abs(grad_ln_Sigma) - f * fgas
    
    # Standard numerator
    numerator = (x**2 / 2.0)
    
    # Compute fX, set to NaN where unstable
    fX = np.where(denom > 0, numerator / denom, np.nan)
    
    return fX


# Physical constants and typical values
TYPICAL_FGAS = {
    'galaxy': 0.00,       # Galaxies have no hot gas
    'poor_cluster': 0.08,  # Lower mass clusters
    'rich_cluster': 0.12,  # Typical massive clusters
    'very_rich': 0.15      # Most massive clusters
}


def estimate_fgas_from_mass(M200: float) -> float:
    """
    Estimate hot gas fraction from cluster mass.
    
    Empirical scaling from X-ray observations:
    fgas increases with cluster mass, roughly as M^0.2
    
    Parameters
    ----------
    M200 : float
        Cluster mass in solar masses (M☉)
    
    Returns
    -------
    fgas : float
        Estimated hot gas fraction
    
    Notes
    -----
    This is a rough approximation. Use actual X-ray measurements
    when available.
    
    References:
    - Pratt+09: fgas ~ 0.10-0.14 for M500 > 1e14 Msun
    - Sun+09: fgas correlates with cluster mass
    """
    # Reference: 1e14 Msun → fgas ~ 0.10
    M_ref = 1e14  # solar masses
    fgas_ref = 0.10
    
    # Scaling exponent (weak)
    beta = 0.2
    
    fgas = fgas_ref * (M200 / M_ref) ** beta
    
    # Clip to physical range
    fgas = np.clip(fgas, 0.0, 0.20)
    
    return fgas


if __name__ == "__main__":
    # Quick sanity check
    print("Hot Gas Fraction Model - Sanity Check")
    print("=" * 60)
    
    # O2 baseline params + test hot gas coefficient
    params_baseline = (0.6687, 0.1401, 0.0871, 0.0)  # f=0 → no hot gas effect
    params_test = (0.6687, 0.1401, 0.0871, 0.5)      # f=0.5 → test value
    
    # Test point: cluster outskirt
    x = 10.0
    Sigma_hat = -1.5
    grad_ln_Sigma = 0.3
    fgas_cluster = 0.12  # 12% hot gas (typical cluster)
    fgas_galaxy = 0.0    # No hot gas (galaxy)
    
    # Compute fX
    fX_baseline_cluster = fX_ratio_curv_fgas(params_baseline, x, Sigma_hat, grad_ln_Sigma, fgas_cluster)
    fX_test_cluster = fX_ratio_curv_fgas(params_test, x, Sigma_hat, grad_ln_Sigma, fgas_cluster)
    fX_test_galaxy = fX_ratio_curv_fgas(params_test, x, Sigma_hat, grad_ln_Sigma, fgas_galaxy)
    
    print(f"\nCluster (fgas={fgas_cluster}):")
    print(f"  fX baseline (f=0):    {fX_baseline_cluster:.2f}")
    print(f"  fX with fgas (f=0.5): {fX_test_cluster:.2f}")
    print(f"  Amplification:        {fX_test_cluster / fX_baseline_cluster:.2f}×")
    
    print(f"\nGalaxy (fgas={fgas_galaxy}):")
    print(f"  fX with fgas (f=0.5): {fX_test_galaxy:.2f}")
    print(f"  Change from baseline: {(fX_test_galaxy / fX_baseline_cluster - 1) * 100:.1f}%")
    
    print("\n✅ Model loaded successfully")
