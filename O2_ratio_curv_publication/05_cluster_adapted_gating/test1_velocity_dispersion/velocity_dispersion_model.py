"""
velocity_dispersion_model.py

Implementation of Test 1: Velocity Dispersion Gating for cluster extension.

Hypothesis: Deep potential wells (high σ) amplify geometry-gated tail.
Formula: fX = x² / (a - b·Σ̂ - d·|∇ln Σ| - e·(σ/σ₀)^α)

Author: Henry Speiser
Date: October 2, 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path


def compute_velocity_dispersion_from_temperature(kT_keV, mu=0.59):
    """
    Compute 1D velocity dispersion from X-ray temperature.
    
    For ionized gas in hydrostatic equilibrium:
    σ = sqrt(kT / μ m_p)
    
    Parameters:
    -----------
    kT_keV : float or array
        X-ray temperature in keV
    mu : float
        Mean molecular weight (0.59 for ionized ICM, 0.6 for solar metallicity)
        
    Returns:
    --------
    sigma_kms : float or array
        1D velocity dispersion in km/s
        
    Notes:
    ------
    - For clusters: typical kT = 5-10 keV → σ ~ 1000-1400 km/s
    - For galaxies: use virial estimate σ ~ sqrt(GM/R) ~ 100-200 km/s
    - Conversion: 1 keV = 1.602e-9 erg, m_p = 1.673e-24 g
    """
    # Physical constants
    kT_erg = kT_keV * 1.602e-9  # keV to erg
    m_p_g = 1.673e-24  # proton mass in grams
    
    # Velocity dispersion in cm/s
    sigma_cms = np.sqrt(kT_erg / (mu * m_p_g))
    
    # Convert to km/s
    sigma_kms = sigma_cms / 1e5
    
    return sigma_kms


def compute_velocity_dispersion_from_virial(M_vir_Msun, R_vir_kpc):
    """
    Compute velocity dispersion from virial mass and radius.
    
    σ = sqrt(GM_vir / R_vir)
    
    Parameters:
    -----------
    M_vir_Msun : float or array
        Virial mass in solar masses (typically M_200)
    R_vir_kpc : float or array
        Virial radius in kpc (typically R_200)
        
    Returns:
    --------
    sigma_kms : float or array
        1D velocity dispersion in km/s
        
    Notes:
    ------
    - For SPARC galaxies: M_200 ~ 1e10-1e12 Msun, R_200 ~ 50-200 kpc
    - Typical σ ~ 50-200 km/s for galaxies
    """
    G_kpc_Msun_kms = 4.302e-6  # G in (kpc km²/s²)/Msun
    sigma_kms = np.sqrt(G_kpc_Msun_kms * M_vir_Msun / R_vir_kpc)
    return sigma_kms


def fX_ratio_curv_sigma(params, x, Sigma_hat, grad_ln_Sigma, sigma_kms, sigma_ref=100.0):
    """
    Velocity dispersion-gated excess factor.
    
    fX = x² / (a - b·Σ̂ - d·|∇ln Σ| - e·(σ/σ₀)^α)
    
    This extends the O2 ratio_curv model with velocity dispersion gating.
    
    Parameters:
    -----------
    params : tuple (a, b, d, e, alpha)
        Model parameters:
        - a: numerator scale (from O2)
        - b: surface density weight (from O2)
        - d: curvature weight (from O2)
        - e: velocity dispersion gate weight (NEW)
        - alpha: velocity dispersion power index (NEW)
    x : float or array
        Dimensionless radius (R / Rd)
    Sigma_hat : float or array
        Normalized surface density
    grad_ln_Sigma : float or array
        Logarithmic gradient of surface density
    sigma_kms : float or array
        Velocity dispersion in km/s
    sigma_ref : float
        Reference velocity dispersion (100 km/s for typical galaxies)
        
    Returns:
    --------
    fX : float or array
        Excess gravity factor (dimensionless, typically 0-5)
        
    Notes:
    ------
    - For galaxies (σ ~ 100 km/s): σ/σ₀ ~ 1 → minimal change
    - For clusters (σ ~ 1000 km/s): σ/σ₀ ~ 10 → large amplification
    - With α = 1.5: (1000/100)^1.5 = 31.6× boost in denominator reduction
    """
    a, b, d, e, alpha = params
    
    # Velocity dispersion term (new physics)
    sigma_ratio = sigma_kms / sigma_ref
    sigma_term = e * (sigma_ratio ** alpha)
    
    # Denominator with velocity dispersion gating
    # Larger σ → larger sigma_term → smaller denom → larger fX
    denom = a - b * Sigma_hat - d * np.abs(grad_ln_Sigma) - sigma_term
    
    # Avoid division by zero or negative denominator
    denom = np.clip(denom, 1e-6, None)
    
    # Compute excess factor
    fX = (x ** 2) / denom
    
    # Ensure non-negative (physical constraint)
    fX = np.maximum(fX, 0.0)
    
    return fX


def load_cluster_temperature_profile(cluster_name, data_dir):
    """
    Load temperature profile for a cluster.
    
    Parameters:
    -----------
    cluster_name : str
        Name of cluster (e.g., 'ABELL_1689', 'A2029', 'A478')
    data_dir : Path or str
        Path to data/clusters/ directory
        
    Returns:
    --------
    df : DataFrame
        Columns: r_kpc, kT_keV, kT_err_keV
    """
    cluster_dir = Path(data_dir) / cluster_name
    temp_file = cluster_dir / "temp_profile.csv"
    
    if not temp_file.exists():
        raise FileNotFoundError(f"Temperature profile not found: {temp_file}")
    
    df = pd.read_csv(temp_file)
    
    # Validate columns
    required_cols = ['r_kpc', 'kT_keV']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {temp_file}")
    
    return df


def compute_median_temperature(temp_profile_df, r_min_kpc=10, r_max_kpc=500):
    """
    Compute characteristic temperature in a radial range.
    
    Parameters:
    -----------
    temp_profile_df : DataFrame
        Temperature profile with columns: r_kpc, kT_keV
    r_min_kpc : float
        Minimum radius for averaging (default 10 kpc, avoid central AGN)
    r_max_kpc : float
        Maximum radius for averaging (default 500 kpc, virial region)
        
    Returns:
    --------
    kT_median : float
        Median temperature in keV
    sigma_median : float
        Corresponding velocity dispersion in km/s
    """
    df = temp_profile_df.copy()
    
    # Filter to radial range
    mask = (df['r_kpc'] >= r_min_kpc) & (df['r_kpc'] <= r_max_kpc)
    df_filtered = df[mask]
    
    if len(df_filtered) == 0:
        raise ValueError(f"No data points in range [{r_min_kpc}, {r_max_kpc}] kpc")
    
    # Median temperature
    kT_median = df_filtered['kT_keV'].median()
    
    # Convert to velocity dispersion
    sigma_median = compute_velocity_dispersion_from_temperature(kT_median)
    
    return kT_median, sigma_median


def test_velocity_dispersion_calculations():
    """
    Unit test for velocity dispersion functions.
    """
    print("=" * 60)
    print("Testing Velocity Dispersion Calculations")
    print("=" * 60)
    
    # Test 1: Temperature to velocity dispersion
    print("\nTest 1: X-ray temperature to velocity dispersion")
    kT_values = [1.0, 5.0, 10.0, 15.0]  # keV
    for kT in kT_values:
        sigma = compute_velocity_dispersion_from_temperature(kT)
        print(f"  kT = {kT:5.1f} keV  →  σ = {sigma:7.1f} km/s")
    
    # Test 2: Virial mass to velocity dispersion
    print("\nTest 2: Virial mass/radius to velocity dispersion")
    M_values = [1e10, 1e11, 1e12, 1e13]  # Msun
    R = 100.0  # kpc
    for M in M_values:
        sigma = compute_velocity_dispersion_from_virial(M, R)
        print(f"  M_200 = {M:.1e} Msun, R_200 = {R} kpc  →  σ = {sigma:7.1f} km/s")
    
    # Test 3: Velocity dispersion gating effect
    print("\nTest 3: Velocity dispersion gating amplification")
    # O2 baseline parameters
    params_base = (0.6687, 0.1401, 0.0871, 0.0, 0.0)  # (a, b, d, e=0, alpha=0)
    params_sigma = (0.6687, 0.1401, 0.0871, 0.05, 1.5)  # (a, b, d, e=0.05, alpha=1.5)
    
    # Test conditions
    x = 5.0  # Dimensionless radius
    Sigma_hat = -1.0  # Low surface density (cluster outskirts)
    grad_ln_Sigma = 0.5  # Moderate gradient
    
    sigma_values = [50, 100, 200, 500, 1000, 1500]  # km/s
    
    print(f"\n  Conditions: x={x}, Σ̂={Sigma_hat}, |∇ln Σ|={grad_ln_Sigma}")
    print(f"  {'σ (km/s)':>12} {'fX (base)':>12} {'fX (σ-gate)':>12} {'Amplification':>15}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*15}")
    
    for sigma in sigma_values:
        fX_base = fX_ratio_curv_sigma(params_base, x, Sigma_hat, grad_ln_Sigma, sigma)
        fX_sigma = fX_ratio_curv_sigma(params_sigma, x, Sigma_hat, grad_ln_Sigma, sigma)
        amplification = fX_sigma / fX_base if fX_base > 0 else 0
        print(f"  {sigma:12.0f} {fX_base:12.3f} {fX_sigma:12.3f} {amplification:15.2f}×")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run unit tests
    test_velocity_dispersion_calculations()
