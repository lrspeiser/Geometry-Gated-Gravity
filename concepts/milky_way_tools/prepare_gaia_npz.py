#!/usr/bin/env python3
"""
Prepare Gaia MW data in NPZ format for the GPU orchestrator
"""

import numpy as np
import pandas as pd

def prepare_mw_npz():
    # Load the real Gaia MW data
    print("Loading Gaia MW data...")
    df = pd.read_csv('data/gaia_mw_real.csv')
    
    # Load surface density model
    print("Loading surface density...")
    sigma_df = pd.read_csv('data/mw_sigma_disk.csv')
    
    # Interpolate surface density to star positions
    from scipy.interpolate import interp1d
    sigma_interp = interp1d(sigma_df['R_kpc'], sigma_df['Sigma_Msun_pc2'], 
                            bounds_error=False, fill_value='extrapolate')
    Sigma_loc = sigma_interp(df['R_kpc'].values)
    
    # Compute Newtonian acceleration from baryon model
    # Using Miyamoto-Nagai disk + Hernquist bulge
    R = df['R_kpc'].values
    z = df['z_kpc'].values
    
    # Disk parameters (stellar + gas)
    M_disk = 6e10  # Msun
    a_disk = 3.5   # kpc
    b_disk = 0.3   # kpc
    
    # Bulge parameters
    M_bulge = 1e10  # Msun
    a_bulge = 0.7   # kpc
    
    # Miyamoto-Nagai disk acceleration
    G = 4.3e-6  # km^2/s^2 kpc/Msun
    s = np.sqrt(R**2 + (a_disk + np.sqrt(z**2 + b_disk**2))**2)
    gR_disk = G * M_disk * R / s**3
    
    # Hernquist bulge acceleration
    r = np.sqrt(R**2 + z**2)
    gR_bulge = G * M_bulge * R / (r * (r + a_bulge)**2)
    
    # Total Newtonian
    gN = gR_disk + gR_bulge  # km^2/s^2/kpc
    
    # Save as NPZ
    output_path = 'data/mw_gaia_144k.npz'
    np.savez(output_path,
             R_kpc=R,
             z_kpc=z,
             v_obs_kms=df['vphi'].values,
             v_err_kms=df['vphi_err'].values if 'vphi_err' in df else np.ones_like(R) * 10.0,
             gN_kms2_per_kpc=gN,
             Sigma_loc_Msun_pc2=Sigma_loc)
    
    print(f"Saved {len(R)} stars to {output_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  R range: {R.min():.1f} - {R.max():.1f} kpc")
    print(f"  z range: {z.min():.2f} - {z.max():.2f} kpc")
    print(f"  vphi range: {df['vphi'].min():.0f} - {df['vphi'].max():.0f} km/s")
    print(f"  Median gN: {np.median(gN):.1f} km^2/s^2/kpc")
    print(f"  Median Sigma: {np.median(Sigma_loc):.1f} Msun/pc^2")

if __name__ == '__main__':
    prepare_mw_npz()