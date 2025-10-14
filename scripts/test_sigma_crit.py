#!/usr/bin/env python3
"""
Quick diagnostic: verify Σ_crit calculation for MACS0416.

Expected: Σ_crit ~ few ×10⁹ M☉/kpc² for z_l=0.396, z_s=2.0
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from many_path_model.lensing_utilities import LensingCosmology

def main():
    cosmo = LensingCosmology()
    
    # MACS0416 parameters
    z_lens = 0.396
    z_source = 2.0
    
    # Angular diameter distances
    D_d = cosmo.angular_diameter_distance_kpc(z_lens)
    D_s = cosmo.angular_diameter_distance_kpc(z_source)
    D_ds = cosmo.angular_diameter_distance_between(z_lens, z_source)
    
    # Critical surface density
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_source)
    
    print("\n" + "="*70)
    print("Σ_crit DIAGNOSTIC FOR MACS0416")
    print("="*70)
    print(f"\nCosmology: H0={cosmo.cosmo.H0} km/s/Mpc, Ωm={cosmo.cosmo.Omega_m}, ΩΛ={cosmo.cosmo.Omega_L}")
    print(f"\nLens redshift:   z_l = {z_lens}")
    print(f"Source redshift: z_s = {z_source}")
    print(f"\nAngular Diameter Distances:")
    print(f"  D_d  (observer → lens):   {D_d:.2e} kpc = {D_d/1000:.2f} Mpc")
    print(f"  D_s  (observer → source): {D_s:.2e} kpc = {D_s/1000:.2f} Mpc")
    print(f"  D_ds (lens → source):     {D_ds:.2e} kpc = {D_ds/1000:.2f} Mpc")
    print(f"\nCritical Surface Density:")
    print(f"  Σ_crit = {Sigma_crit:.4e} M☉/kpc²")
    print(f"  Σ_crit = {Sigma_crit/1e9:.2f} × 10⁹ M☉/kpc²")
    
    # Sanity check
    expected_range = (2e9, 5e9)
    if expected_range[0] < Sigma_crit < expected_range[1]:
        print(f"\n✅ PASS: Σ_crit is in expected range {expected_range[0]/1e9:.1f}-{expected_range[1]/1e9:.1f} ×10⁹ M☉/kpc²")
    else:
        print(f"\n❌ WARNING: Σ_crit outside expected range!")
    
    # Physical interpretation
    print(f"\nPhysical Interpretation:")
    print(f"  To produce θ_E ~ 35 arcsec (R_E ~ 180 kpc):")
    R_E = 180.0  # kpc
    M_needed = np.pi * R_E**2 * Sigma_crit
    print(f"  Need M(<R_E) ~ {M_needed:.2e} M☉ = {M_needed/1e14:.2f} × 10¹⁴ M☉")
    print(f"  This is the total projected mass inside Einstein radius")
    print("="*70 + "\n")

if __name__ == '__main__':
    import numpy as np
    main()
