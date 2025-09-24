#!/usr/bin/env python3
"""
Convert raw Gaia DR3 data to Milky Way coordinates for G³ optimizer.
Converts parallax/proper motions to R, z, vphi in the Galactic frame.
"""

import numpy as np
import pandas as pd
import argparse

# Constants
R_SUN = 8.178  # kpc - Sun's distance from GC (Gravity Collaboration 2021)
V_SUN = 248.0  # km/s - Sun's circular velocity
U_SUN = 11.1   # km/s - Sun's motion toward GC
V_SUN_PEC = 12.24  # km/s - Sun's motion in rotation direction
W_SUN = 7.25   # km/s - Sun's motion toward NGP

def gaia_to_galactocentric(df):
    """
    Convert Gaia observables to Galactocentric coordinates.
    """
    # Convert parallax to distance
    d_kpc = 1.0 / df['parallax'].values  # parallax in mas -> distance in kpc
    
    # Galactic coordinates
    l_rad = np.deg2rad(df['l'].values)
    b_rad = np.deg2rad(df['b'].values)
    
    # Heliocentric Cartesian coordinates
    x_h = d_kpc * np.cos(b_rad) * np.cos(l_rad)
    y_h = d_kpc * np.cos(b_rad) * np.sin(l_rad)
    z_h = d_kpc * np.sin(b_rad)
    
    # Galactocentric Cartesian coordinates
    x_gc = R_SUN - x_h
    y_gc = -y_h
    z_gc = z_h
    
    # Cylindrical coordinates
    R_kpc = np.sqrt(x_gc**2 + y_gc**2)
    phi = np.arctan2(y_gc, x_gc)
    z_kpc = z_gc
    
    # Convert proper motions to velocities
    # pm in mas/yr, d in kpc -> v in km/s
    k = 4.74047  # conversion factor
    vl = k * df['pmra'].values * d_kpc  # velocity in l direction
    vb = k * df['pmdec'].values * d_kpc  # velocity in b direction
    vr = df['radial_velocity'].values  # already in km/s
    
    # Handle missing RVs
    vr = np.where(np.isnan(vr), 0.0, vr)  # assume 0 if missing
    
    # Transform to Galactocentric velocities
    # This is approximate - proper transformation requires full matrix
    cos_b = np.cos(b_rad)
    sin_b = np.sin(b_rad)
    cos_l = np.cos(l_rad)
    sin_l = np.sin(l_rad)
    
    # Heliocentric Cartesian velocities
    vx_h = vr * cos_b * cos_l - vl * sin_l - vb * sin_b * cos_l
    vy_h = vr * cos_b * sin_l + vl * cos_l - vb * sin_b * sin_l
    vz_h = vr * sin_b + vb * cos_b
    
    # Add solar motion
    vx_gc = vx_h - U_SUN
    vy_gc = vy_h - (V_SUN + V_SUN_PEC)
    vz_gc = vz_h - W_SUN
    
    # Cylindrical velocities
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    vR = -vx_gc * cos_phi - vy_gc * sin_phi
    vphi = vx_gc * sin_phi - vy_gc * cos_phi
    vz = vz_gc
    
    # Add uncertainties (simplified)
    # Propagate parallax error to distance error
    d_err = df['parallax_error'].values / (df['parallax'].values**2)
    vphi_err = np.sqrt(
        (k * df['pmra_error'].values * d_kpc)**2 + 
        (k * df['pmra'].values * d_err)**2 +
        25.0  # systematic uncertainty
    )
    
    return pd.DataFrame({
        'R_kpc': R_kpc,
        'z_kpc': z_kpc,
        'vphi': vphi,
        'vR': vR,
        'vz': vz,
        'vphi_err': vphi_err,
        'l': df['l'].values,
        'b': df['b'].values,
        'source_id': df['source_id'].values
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input Gaia CSV')
    parser.add_argument('--output', required=True, help='Output MW CSV')
    parser.add_argument('--quality_cuts', action='store_true', 
                       help='Apply quality cuts')
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} stars loaded")
    
    # Quality cuts
    if args.quality_cuts:
        print("Applying quality cuts...")
        # Good parallax
        mask = (df['parallax'] > 0.1) & (df['parallax_error'] < 0.2)
        # Good astrometry
        mask &= (df['ruwe'] < 1.4) if 'ruwe' in df.columns else True
        # Has proper motions
        mask &= ~(df['pmra'].isna() | df['pmdec'].isna())
        df = df[mask]
        print(f"  {len(df)} stars after quality cuts")
    
    print("Converting to Galactocentric coordinates...")
    mw_df = gaia_to_galactocentric(df)
    
    # Additional filters
    print("Applying MW disk cuts...")
    mask = (mw_df['R_kpc'] > 3.0) & (mw_df['R_kpc'] < 20.0)
    mask &= np.abs(mw_df['z_kpc']) < 1.0
    mask &= mw_df['vphi'] > 50.0  # Remove counter-rotating stars
    mw_df = mw_df[mask]
    print(f"  {len(mw_df)} stars in MW disk")
    
    print(f"Saving to {args.output}...")
    mw_df.to_csv(args.output, index=False)
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  R range: {mw_df['R_kpc'].min():.1f} - {mw_df['R_kpc'].max():.1f} kpc")
    print(f"  z range: {mw_df['z_kpc'].min():.2f} - {mw_df['z_kpc'].max():.2f} kpc")
    print(f"  vphi range: {mw_df['vphi'].min():.0f} - {mw_df['vphi'].max():.0f} km/s")
    print(f"  median vphi: {mw_df['vphi'].median():.0f} km/s")

if __name__ == '__main__':
    main()