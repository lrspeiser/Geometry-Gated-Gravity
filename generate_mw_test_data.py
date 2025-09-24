#!/usr/bin/env python3
"""
Generate synthetic Milky Way test data
=======================================

Creates a realistic Gaia-like dataset for testing the MW G³ optimizer.
Replace this with your actual 144k-star Gaia DR3 data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_mw_test_data(n_stars=10000, output_dir="data"):
    """
    Generate synthetic MW rotation curve data
    
    Creates realistic distribution of stars with:
    - Exponential radial distribution
    - Gaussian vertical distribution  
    - Realistic velocity scatter
    - Measurement uncertainties
    """
    
    np.random.seed(42)
    
    # Radial distribution (exponential disk)
    R_min, R_max = 3.0, 20.0  # kpc
    R_scale = 3.0  # Scale length
    
    # Generate R with exponential distribution
    R_uniform = np.random.uniform(0, 1, n_stars)
    R_kpc = -R_scale * np.log(1 - R_uniform * (1 - np.exp(-(R_max-R_min)/R_scale)))
    R_kpc += R_min
    
    # Vertical distribution (Gaussian)
    z_scale = 0.3  # kpc (thin disk)
    z_kpc = np.random.normal(0, z_scale, n_stars)
    
    # True rotation curve (MW-like with dark matter for synthetic data)
    # This is what we're trying to fit without assuming dark matter
    v_disk = 220.0  # Flat rotation speed
    v_bulge = 100.0 * np.exp(-R_kpc/1.0)  # Bulge contribution
    v_true = np.sqrt(v_disk**2 * (1 - np.exp(-R_kpc/2.0)) + v_bulge**2)
    
    # Add realistic scatter
    v_scatter = 10.0 + 0.5 * R_kpc  # Velocity dispersion increases with R
    vphi_kms = v_true + np.random.normal(0, v_scatter, n_stars)
    
    # Measurement uncertainties
    # Closer stars have better measurements
    vphi_err_base = 2.0
    vphi_err_kms = vphi_err_base * (1 + 0.1 * R_kpc) * (1 + 0.5 * np.abs(z_kpc))
    vphi_err_kms *= np.random.lognormal(0, 0.3, n_stars)  # Log-normal error distribution
    
    # Create DataFrame
    df = pd.DataFrame({
        'R_kpc': R_kpc,
        'z_kpc': z_kpc,
        'vphi_kms': vphi_kms,
        'vphi_err_kms': vphi_err_kms
    })
    
    # Sort by R for easier visualization
    df = df.sort_values('R_kpc').reset_index(drop=True)
    
    # Save to CSV
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    output_file = Path(output_dir) / "gaia_mw_test.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Generated {n_stars} synthetic MW stars")
    print(f"Saved to: {output_file}")
    print("\nData statistics:")
    print(f"  R range: {df['R_kpc'].min():.1f} - {df['R_kpc'].max():.1f} kpc")
    print(f"  |z| range: {df['z_kpc'].abs().min():.2f} - {df['z_kpc'].abs().max():.2f} kpc")
    print(f"  vphi range: {df['vphi_kms'].min():.0f} - {df['vphi_kms'].max():.0f} km/s")
    print(f"  Median error: {df['vphi_err_kms'].median():.1f} km/s")
    
    # Also create a surface density table
    R_table = np.linspace(1, 30, 30)
    # Realistic MW surface density profile
    Sigma_disk = 800 * np.exp(-R_table/2.6)  # Stellar disk
    Sigma_gas = 150 * np.exp(-R_table/7.0)   # Gas disk
    Sigma_total = Sigma_disk + Sigma_gas
    
    sigma_df = pd.DataFrame({
        'R_kpc': R_table,
        'Sigma_Msun_pc2': Sigma_total
    })
    
    sigma_file = Path(output_dir) / "mw_sigma_disk.csv"
    sigma_df.to_csv(sigma_file, index=False)
    print(f"\nSurface density table saved to: {sigma_file}")
    
    return df, sigma_df

if __name__ == "__main__":
    # Generate test data
    stars_df, sigma_df = generate_mw_test_data(n_stars=10000)
    
    print("\n" + "="*60)
    print("TEST DATA READY")
    print("="*60)
    print("\nTo run the optimizer:")
    print("python mw_g3_gpu_opt.py \\")
    print("  --stars_csv data/gaia_mw_test.csv \\")
    print("  --sigma_csv data/mw_sigma_disk.csv \\")
    print("  --out_dir out/mw_g3_gpu \\")
    print("  --popsize 64 \\")
    print("  --z_cut 0.8")
    print("\n(Replace with your actual 144k-star Gaia file for real results)")