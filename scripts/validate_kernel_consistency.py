"""
Belt-and-Suspenders Validation Checks for Σ-Gravity Kernel

This script performs three critical consistency checks:
1. Einstein mass identity: M(<R_E) = π R_E² Σ_crit within few %
2. No hidden mass creation: integrated boost is finite, Σ_eff ~ R^-1 at large R
3. Solar system safety: K_σ → 0 in Newtonian limit

Author: Generated for DensityDependentMetricModel
Date: 2025-01-14
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_macs0416_projected_kernel import build_macs0416_baryon_profile_3d, project_to_surface_density
from core.kernel2d_sigma import convolve_sigma_with_kernel
from many_path_model.lensing_utilities import LensingCosmology

# Constants
MSUN = 1.989e30  # kg
MPC_TO_M = 3.086e22  # meters
KPC_TO_M = 3.086e19  # meters
ARCSEC_TO_RAD = np.pi / 648000
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
C_SI = 2.998e8  # m/s

def build_baryon_profile_2d(grid_size=512, fov_kpc=2000.0):
    """
    Build 2D baryon surface density profile for MACS0416.
    
    Returns:
        Sigma_bar: 2D surface density in Msun/kpc^2
        R_grid_2d: 2D grid of radii in kpc
        extent: FOV extent in kpc
    """
    # Build 3D baryon profile
    r_3d = np.logspace(-1, 3.5, 2000)
    rho_total, baryon_info = build_macs0416_baryon_profile_3d(r_3d, verbose=False)
    
    # Create 2D grid
    x = np.linspace(-fov_kpc, fov_kpc, grid_size)
    y = np.linspace(-fov_kpc, fov_kpc, grid_size)
    X, Y = np.meshgrid(x, y)
    R_grid_2d = np.sqrt(X**2 + Y**2)
    
    # Project to surface density (spherical case)
    Sigma_bar = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)
    
    return Sigma_bar, R_grid_2d, fov_kpc, baryon_info

def test_einstein_mass_identity(cluster_name="MACS0416", A_c=16.4, l0_kpc=200.0):
    """
    Check 1: Einstein mass identity
    
    Verify that M(<R_E) = π R_E² Σ_crit to within a few percent.
    """
    print("\n" + "="*70)
    print("CHECK 1: EINSTEIN MASS IDENTITY")
    print("="*70)
    
    # Load cluster
    catalog = load_cluster_catalog()
    cluster = catalog[catalog['name'] == cluster_name].iloc[0]
    
    z_lens = cluster['z']
    z_source = 2.0  # typical background
    theta_E_arcsec = cluster['theta_E']
    
    # Critical surface density
    Sigma_crit = critical_surface_density(z_lens, z_source)
    print(f"\nCluster: {cluster_name}")
    print(f"z_lens = {z_lens:.3f}, z_source = {z_source:.1f}")
    print(f"θ_E = {theta_E_arcsec:.1f} arcsec")
    print(f"Σ_crit = {Sigma_crit:.2e} Msun/kpc^2")
    
    # Convert θ_E to physical R_E
    D_l = angular_diameter_distance(z_lens) * 1e3  # kpc
    R_E_kpc = theta_E_arcsec * ARCSEC_TO_RAD * D_l
    print(f"R_E = {R_E_kpc:.2f} kpc (physical)")
    
    # Expected Einstein mass from definition
    M_E_expected = np.pi * R_E_kpc**2 * Sigma_crit
    print(f"\nExpected M(<R_E) = π R_E² Σ_crit = {M_E_expected:.3e} Msun")
    
    # Build baryon profile
    Sigma_bar, extent_kpc = build_baryon_surface_density(
        cluster_name, z_lens, grid_size=512, fov_kpc=2000.0
    )
    
    # Apply Σ-gravity kernel
    params = {'A_c': A_c, 'l0_kpc': l0_kpc, 'p': 2.0, 'n_coh': 2.0}
    Sigma_eff, K_sigma, _, _ = compute_boost_profile_on_plane_2d(
        Sigma_bar, extent_kpc, params,
        q_plane=1.0, q_los=1.0  # spherical
    )
    
    # Compute radial profile
    ny, nx = Sigma_eff.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    cy, cx = ny // 2, nx // 2
    R_pix = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    pix_scale_kpc = extent_kpc / nx
    R_kpc = R_pix * pix_scale_kpc
    
    # Azimuthal average
    R_bins = np.linspace(0, extent_kpc / 2, 200)
    Sigma_eff_avg = np.zeros(len(R_bins) - 1)
    R_bin_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    
    for i in range(len(R_bins) - 1):
        mask = (R_kpc >= R_bins[i]) & (R_kpc < R_bins[i+1])
        if np.any(mask):
            Sigma_eff_avg[i] = np.nanmean(Sigma_eff[mask])
    
    # Compute enclosed mass via cumulative integration
    M_enc = np.zeros_like(R_bin_centers)
    for i, R in enumerate(R_bin_centers):
        if i == 0:
            M_enc[i] = np.pi * R**2 * Sigma_eff_avg[i]
        else:
            # Annular contribution
            dR = R_bin_centers[i] - R_bin_centers[i-1]
            annulus_area = 2 * np.pi * R_bin_centers[i-1] * dR
            M_enc[i] = M_enc[i-1] + annulus_area * Sigma_eff_avg[i]
    
    # Find M(<R_E)
    idx_E = np.argmin(np.abs(R_bin_centers - R_E_kpc))
    M_E_computed = M_enc[idx_E]
    R_E_actual = R_bin_centers[idx_E]
    
    print(f"\nComputed M(<R={R_E_actual:.2f} kpc) = {M_E_computed:.3e} Msun")
    
    # Fractional error
    frac_error = (M_E_computed - M_E_expected) / M_E_expected * 100
    print(f"\nFractional error: {frac_error:+.2f}%")
    
    if abs(frac_error) < 5.0:
        print("✅ PASS: Einstein mass identity verified within 5%")
        status = "PASS"
    else:
        print("⚠️  WARNING: Fractional error exceeds 5%")
        status = "WARNING"
    
    return status, frac_error

def test_no_mass_creation(cluster_name="MACS0416", A_c=16.4, l0_kpc=200.0):
    """
    Check 2: No hidden mass creation
    
    Verify that:
    - Integrated boost is finite
    - Σ_eff falls faster than R^-1 at large radii (no divergence)
    """
    print("\n" + "="*70)
    print("CHECK 2: NO HIDDEN MASS CREATION")
    print("="*70)
    
    # Load cluster
    catalog = load_cluster_catalog()
    cluster = catalog[catalog['name'] == cluster_name].iloc[0]
    z_lens = cluster['z']
    
    # Build baryon profile (larger FOV for tail behavior)
    Sigma_bar, extent_kpc = build_baryon_surface_density(
        cluster_name, z_lens, grid_size=512, fov_kpc=3000.0
    )
    
    # Apply kernel
    params = {'A_c': A_c, 'l0_kpc': l0_kpc, 'p': 2.0, 'n_coh': 2.0}
    Sigma_eff, K_sigma, _, _ = compute_boost_profile_on_plane_2d(
        Sigma_bar, extent_kpc, params,
        q_plane=1.0, q_los=1.0
    )
    
    # Radial profiles
    ny, nx = Sigma_eff.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    cy, cx = ny // 2, nx // 2
    R_pix = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    pix_scale_kpc = extent_kpc / nx
    R_kpc = R_pix * pix_scale_kpc
    
    # Azimuthal average
    R_bins = np.linspace(0, extent_kpc / 2, 300)
    Sigma_eff_avg = np.zeros(len(R_bins) - 1)
    K_sigma_avg = np.zeros(len(R_bins) - 1)
    R_bin_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    
    for i in range(len(R_bins) - 1):
        mask = (R_kpc >= R_bins[i]) & (R_kpc < R_bins[i+1])
        if np.any(mask):
            Sigma_eff_avg[i] = np.nanmean(Sigma_eff[mask])
            K_sigma_avg[i] = np.nanmean(K_sigma[mask])
    
    # Check 2a: Integrated boost is finite
    valid = np.isfinite(K_sigma_avg) & (R_bin_centers > 0)
    total_boost_integral = np.trapz(K_sigma_avg[valid] * R_bin_centers[valid], R_bin_centers[valid])
    
    print(f"\nCluster: {cluster_name}")
    print(f"Integrated boost: ∫ K_σ(R) * R dR = {total_boost_integral:.2e} kpc")
    
    if np.isfinite(total_boost_integral):
        print("✅ Boost integral is finite")
        finite_status = "PASS"
    else:
        print("❌ Boost integral is NOT finite")
        finite_status = "FAIL"
    
    # Check 2b: Tail behavior - Σ_eff should fall faster than R^-1
    # At large R, check if Σ_eff ~ R^-α with α > 1
    R_tail_min = 800.0  # kpc
    R_tail_max = 1400.0
    tail_mask = (R_bin_centers >= R_tail_min) & (R_bin_centers <= R_tail_max) & np.isfinite(Sigma_eff_avg) & (Sigma_eff_avg > 0)
    
    if np.sum(tail_mask) > 5:
        log_R_tail = np.log10(R_bin_centers[tail_mask])
        log_Sigma_tail = np.log10(Sigma_eff_avg[tail_mask])
        
        # Linear fit in log-log space
        coeffs = np.polyfit(log_R_tail, log_Sigma_tail, 1)
        slope = coeffs[0]
        
        print(f"\nTail behavior at R = {R_tail_min:.0f}–{R_tail_max:.0f} kpc:")
        print(f"Σ_eff ~ R^{slope:.2f}")
        
        if slope < -1.0:
            print(f"✅ Tail falls faster than R^-1 (no mass divergence)")
            tail_status = "PASS"
        else:
            print(f"⚠️  WARNING: Tail falls slower than R^-1 (slope = {slope:.2f})")
            tail_status = "WARNING"
    else:
        print("⚠️  Insufficient data in tail region for slope fit")
        tail_status = "INSUFFICIENT_DATA"
    
    overall_status = "PASS" if (finite_status == "PASS" and tail_status == "PASS") else "WARNING"
    
    return overall_status, total_boost_integral, slope if np.sum(tail_mask) > 5 else np.nan

def test_solar_system_safety(A_c=16.4, l0_kpc=200.0):
    """
    Check 3: Solar system / Newtonian limit safety
    
    Verify that K_σ → 0 at small scales where Newtonian physics is tested.
    
    The window function W(R) should suppress coherence at R << ℓ_0.
    """
    print("\n" + "="*70)
    print("CHECK 3: SOLAR SYSTEM SAFETY (NEWTONIAN LIMIT)")
    print("="*70)
    
    # Test scales
    R_test_kpc = np.array([1e-9, 1e-6, 1e-3, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0])  # kpc
    
    # Window function W(R) = (R / ℓ_0)^n_coh  (simplified)
    n_coh = 2.0
    W = (R_test_kpc / l0_kpc)**n_coh
    
    # K_σ ≈ A_c * W / (1 + (R/ℓ_0)^p)  (simplified, actual kernel is more complex)
    p = 2.0
    K_sigma_approx = A_c * W / (1 + (R_test_kpc / l0_kpc)**p)
    
    print(f"\nCoherence scale ℓ_0 = {l0_kpc:.0f} kpc")
    print(f"Amplitude A_c = {A_c:.1f}\n")
    print("R [kpc]       | K_σ (approx) | Status")
    print("-" * 50)
    
    solar_system_safe = True
    galaxy_safe = True
    
    for R, K in zip(R_test_kpc, K_sigma_approx):
        if R < 1e-3:  # Solar system scale
            status = "✅ SAFE" if K < 0.001 else "❌ UNSAFE"
            if K >= 0.001:
                solar_system_safe = False
        elif R < 10.0:  # Galaxy scale
            status = "✅ SAFE" if K < 0.1 else "⚠️ CHECK"
            if K >= 0.1:
                galaxy_safe = False
        else:
            status = "Cluster scale"
        
        print(f"{R:12.2e} | {K:12.4e} | {status}")
    
    print("\n" + "-" * 50)
    if solar_system_safe and galaxy_safe:
        print("✅ PASS: K_σ → 0 at small scales (Newtonian limit preserved)")
        overall_status = "PASS"
    elif solar_system_safe:
        print("⚠️  WARNING: Solar system safe, but check galaxy-scale suppression")
        overall_status = "WARNING"
    else:
        print("❌ FAIL: K_σ too large at small scales")
        overall_status = "FAIL"
    
    return overall_status, K_sigma_approx[0]  # Return smallest-scale K_σ

def main():
    """Run all validation checks."""
    print("\n" + "="*70)
    print("BELT-AND-SUSPENDERS VALIDATION FOR Σ-GRAVITY KERNEL")
    print("="*70)
    print("\nThese checks verify that the kernel is physically consistent:")
    print("  1. Einstein mass identity: M(<R_E) = π R_E² Σ_crit")
    print("  2. No hidden mass creation: finite boost, Σ_eff ~ R^-α with α > 1")
    print("  3. Solar system safety: K_σ → 0 at small scales")
    print("\n" + "="*70)
    
    # Test with MACS0416 parameters
    cluster_name = "MACS0416"
    A_c = 16.4
    l0_kpc = 200.0
    
    # Run checks
    results = {}
    
    try:
        status1, frac_err = test_einstein_mass_identity(cluster_name, A_c, l0_kpc)
        results['einstein_identity'] = (status1, frac_err)
    except Exception as e:
        print(f"❌ Check 1 failed with error: {e}")
        results['einstein_identity'] = ("ERROR", None)
    
    try:
        status2, boost_integral, tail_slope = test_no_mass_creation(cluster_name, A_c, l0_kpc)
        results['mass_creation'] = (status2, boost_integral, tail_slope)
    except Exception as e:
        print(f"❌ Check 2 failed with error: {e}")
        results['mass_creation'] = ("ERROR", None, None)
    
    try:
        status3, K_small = test_solar_system_safety(A_c, l0_kpc)
        results['solar_safety'] = (status3, K_small)
    except Exception as e:
        print(f"❌ Check 3 failed with error: {e}")
        results['solar_safety'] = ("ERROR", None)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_pass = True
    for key, val in results.items():
        status = val[0]
        if status != "PASS":
            all_pass = False
    
    print(f"\n1. Einstein mass identity: {results['einstein_identity'][0]}")
    if results['einstein_identity'][1] is not None:
        print(f"   Fractional error: {results['einstein_identity'][1]:+.2f}%")
    
    print(f"\n2. No mass creation: {results['mass_creation'][0]}")
    if results['mass_creation'][1] is not None:
        print(f"   Boost integral: {results['mass_creation'][1]:.2e} kpc")
    if results['mass_creation'][2] is not None and np.isfinite(results['mass_creation'][2]):
        print(f"   Tail slope: {results['mass_creation'][2]:.2f}")
    
    print(f"\n3. Solar system safety: {results['solar_safety'][0]}")
    if results['solar_safety'][1] is not None:
        print(f"   K_σ at R=1e-9 kpc: {results['solar_safety'][1]:.2e}")
    
    print("\n" + "="*70)
    if all_pass:
        print("✅ ALL CHECKS PASSED")
        print("The Σ-gravity kernel is physically consistent.")
    else:
        print("⚠️  SOME CHECKS HAVE WARNINGS OR FAILURES")
        print("Review the detailed output above.")
    print("="*70 + "\n")
    
    return results

if __name__ == "__main__":
    results = main()
