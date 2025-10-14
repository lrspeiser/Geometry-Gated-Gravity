#!/usr/bin/env python3
"""
Debug: Why is there no Einstein radius despite K_Σ ~ 5-6?

Trace through the lensing calculation step by step.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from many_path_model.cluster_data_loader import load_cluster_profile
from many_path_model.lensing_utilities import default_cosmology
from core.cluster_first_kernel import lensing_profiles

def main():
    # Parameters
    cluster = 'MACSJ0416'
    z_src = 2.0
    
    K3D_params = {
        'A_c': 10.0,
        'r_gate': 5.0,
        'n_gate': 4,
        'ell0': 180.0,
        'p': 1.2,
        'L1': 1200.0,
        'q': 2.0
    }
    
    # Load data
    cosmo = default_cosmology()
    z_lens, r_grid, rho_r = load_cluster_profile(cluster)
    
    print("\n" + "="*70)
    print(f"DEBUGGING EINSTEIN RADIUS FOR {cluster}")
    print("="*70)
    
    # Total baryon mass
    M_baryon_total = np.trapezoid(4*np.pi*r_grid**2*rho_r, r_grid)
    print(f"\n1. Total baryon mass: {M_baryon_total:.2e} M☉")
    
    # Sigma_crit
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_src)
    print(f"2. Σ_crit = {Sigma_crit:.2e} M☉/kpc²")
    
    # Run lensing calculation
    R = np.geomspace(2.0, 1500.0, 200)
    prof = lensing_profiles(R, z_lens, z_src, r_grid, rho_r, K3D_params, cosmo)
    
    # Check at R=180 kpc (expected Einstein radius)
    idx_180 = np.argmin(np.abs(R - 180))
    R_test = R[idx_180]
    
    print(f"\n3. At R = {R_test:.1f} kpc:")
    print(f"   Σ_baryon = {prof['Sigma'][idx_180]:.2e} M☉/kpc²")
    print(f"   K_Σ = {prof['K_Sigma'][idx_180]:.3f}")
    print(f"   Σ_eff = {prof['Sigma_eff'][idx_180]:.2e} M☉/kpc²")
    print(f"   κ(R) = {prof['kappa'][idx_180]:.3f}")
    print(f"   ⟨κ⟩(<R) = {prof['mean_kappa'][idx_180]:.3f}")
    
    # Projected mass inside R=180
    # M_2D = π R² Σ_eff  (this is what lensing sees)
    M_2D_baryon = np.pi * R_test**2 * prof['Sigma'][idx_180]
    M_2D_eff = np.pi * R_test**2 * prof['Sigma_eff'][idx_180]
    M_needed = np.pi * R_test**2 * Sigma_crit
    
    print(f"\n4. Projected mass inside R={R_test:.1f} kpc:")
    print(f"   M_2D (baryons only) = {M_2D_baryon:.2e} M☉")
    print(f"   M_2D (with boost) = {M_2D_eff:.2e} M☉")
    print(f"   M_needed (for ⟨κ⟩=1) = {M_needed:.2e} M☉")
    print(f"   Ratio (effective/needed) = {M_2D_eff/M_needed:.3f}")
    
    # Find maximum mean_kappa
    max_idx = np.argmax(prof['mean_kappa'])
    print(f"\n5. Maximum ⟨κ⟩:")
    print(f"   ⟨κ⟩_max = {prof['mean_kappa'][max_idx]:.3f} at R = {R[max_idx]:.1f} kpc")
    
    # Predict Einstein radius
    if prof['theta_E_arcsec'] > 0:
        print(f"\n6. Einstein radius: θ_E = {prof['theta_E_arcsec']:.2f} arcsec ✅")
    else:
        print(f"\n6. Einstein radius: NONE FOUND ❌")
        print(f"   Need to increase boost or baryons by factor ~{1.0/prof['mean_kappa'][idx_180]:.2f}×")
    
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
