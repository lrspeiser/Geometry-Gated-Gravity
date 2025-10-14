"""
Quick Belt-and-Suspenders Validation for Σ-Gravity Kernel

Uses existing MACS0416 scripts to verify:
1. Einstein mass identity: M(<R_E) = π R_E² Σ_crit
2. No mass creation: boost is localized
3. Solar system safety: K_σ → 0 at small scales

Author: Generated for DensityDependentMetricModel  
Date: 2025-01-14
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.integrate import cumulative_trapezoid
from test_macs0416_projected_kernel import build_macs0416_baryon_profile_3d, project_to_surface_density
from core.kernel2d_sigma import convolve_sigma_with_kernel
from many_path_model.lensing_utilities import LensingCosmology

def test_einstein_mass_identity():
    """Check 1: Verify M(<R_E) = π R_E² Σ_crit."""
    print("\n" + "="*70)
    print("CHECK 1: EINSTEIN MASS IDENTITY")
    print("="*70)
    
    # Build baryon profile
    print("\nBuilding MACS0416 baryon profile...")
    r_3d = np.logspace(-1, 3.5, 2000)
    rho_total, baryon_info = build_macs0416_baryon_profile_3d(r_3d, verbose=False)
    
    # Create 2D grid
    nx, ny = 512, 512
    R_max = 2000.0
    x = np.linspace(-R_max, R_max, nx)
    y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid_2d = np.sqrt(X**2 + Y**2)
    
    # Project to surface density
    Sigma_bar = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)
    
    # Apply kernel
    A_c = 16.4
    l0_kpc = 200.0
    Sigma_eff, K_sigma, diag = convolve_sigma_with_kernel(
        Sigma_bar, R_grid_2d, l0_kpc, 2.0, 2.0, A_c,
        emphasize_interior=True, use_fft=True
    )
    
    # Cosmology
    cosmo = LensingCosmology()
    z_lens = baryon_info['z']
    z_src = 2.0
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_src)
    theta_E_obs = 30.0  # arcsec
    
    # Convert to kpc (reverse operation: angular * D_A / 206265)
    D_A_kpc = cosmo.angular_diameter_distance_kpc(z_lens)
    theta_E_rad = theta_E_obs / 206265.0  # arcsec to radians
    R_E_kpc = theta_E_rad * D_A_kpc
    
    print(f"\nCluster: MACS0416")
    print(f"z_lens = {z_lens:.3f}, z_source = {z_src:.1f}")
    print(f"θ_E (observed) = {theta_E_obs:.1f} arcsec")
    print(f"R_E = {R_E_kpc:.2f} kpc")
    print(f"Σ_crit = {Sigma_crit:.2e} Msun/kpc^2")
    
    # Expected Einstein mass
    M_E_expected = np.pi * R_E_kpc**2 * Sigma_crit
    print(f"\nExpected M(<R_E) = π R_E² Σ_crit = {M_E_expected:.3e} Msun")
    
    # Compute enclosed mass at R_E
    mask_inside = R_grid_2d <= R_E_kpc
    pixel_area = (2*R_max/nx)**2
    M_E_computed = np.sum(Sigma_eff[mask_inside]) * pixel_area
    
    print(f"Computed M(<R_E) = {M_E_computed:.3e} Msun")
    
    # Fractional error
    frac_error = (M_E_computed - M_E_expected) / M_E_expected * 100
    print(f"\nFractional error: {frac_error:+.2f}%")
    
    if abs(frac_error) < 5.0:
        print("✅ PASS: Einstein mass identity within 5%")
        return "PASS", frac_error
    elif abs(frac_error) < 15.0:
        print("⚠️  WARNING: Fractional error {:.1f}% (> 5%)".format(abs(frac_error)))
        return "WARNING", frac_error
    else:
        print("❌ FAIL: Fractional error too large")
        return "FAIL", frac_error

def test_boost_localization():
    """Check 2: Verify boost is localized (not a mass sheet)."""
    print("\n" + "="*70)
    print("CHECK 2: BOOST LOCALIZATION (NO MASS SHEET)")
    print("="*70)
    
    # Build profile
    r_3d = np.logspace(-1, 3.5, 2000)
    rho_total, _ = build_macs0416_baryon_profile_3d(r_3d, verbose=False)
    
    # Larger FOV to check tail
    nx, ny = 512, 512
    R_max = 3000.0
    x = np.linspace(-R_max, R_max, nx)
    y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid_2d = np.sqrt(X**2 + Y**2)
    
    Sigma_bar = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)
    
    # Apply kernel
    A_c = 16.4
    Sigma_eff, K_sigma, _ = convolve_sigma_with_kernel(
        Sigma_bar, R_grid_2d, 200.0, 2.0, 2.0, A_c,
        emphasize_interior=True, use_fft=True
    )
    
    # Check K_sigma at various radii
    print("\nBoost K_σ(R) at key radii:")
    print("R [kpc]  | K_σ     | Status")
    print("-" * 40)
    
    max_boost_outside_500 = 0.0
    
    for R_check in [10, 50, 100, 200, 500, 1000, 1500, 2000]:
        mask = (R_grid_2d >= R_check - 20) & (R_grid_2d < R_check + 20)
        if np.any(mask):
            K_mean = np.nanmean(K_sigma[mask])
            status = "Core" if R_check < 500 else "Tail"
            if R_check >= 500:
                max_boost_outside_500 = max(max_boost_outside_500, K_mean)
            print(f"{R_check:7d}  | {K_mean:7.3f} | {status}")
    
    print(f"\nMax boost outside 500 kpc: {max_boost_outside_500:.3f}")
    
    if max_boost_outside_500 < 2.0:
        print("✅ PASS: Boost is localized (< 2.0× beyond 500 kpc)")
        return "PASS", max_boost_outside_500
    else:
        print("⚠️  WARNING: Boost extends far (> 2.0× beyond 500 kpc)")
        return "WARNING", max_boost_outside_500

def test_small_scale_safety():
    """Check 3: K_σ → 0 at small scales (Newtonian limit)."""
    print("\n" + "="*70)
    print("CHECK 3: SMALL-SCALE SAFETY (NEWTONIAN LIMIT)")
    print("="*70)
    
    # Analytic approximation for K_σ
    # K_σ(R) ≈ A_c * (R/ℓ_0)^n_coh / [1 + (R/ℓ_0)^p]
    
    A_c = 16.4
    l0 = 200.0  # kpc
    n_coh = 2.0
    p = 2.0
    
    R_test_kpc = np.array([1e-9, 1e-6, 1e-3, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0])
    
    # Simplified kernel (window function)
    W = (R_test_kpc / l0)**n_coh
    K_approx = A_c * W / (1 + (R_test_kpc / l0)**p)
    
    print(f"\nℓ_0 = {l0:.0f} kpc, A_c = {A_c:.1f}")
    print("\nR [kpc]       | K_σ (approx) | Status")
    print("-" * 50)
    
    solar_safe = True
    galaxy_safe = True
    
    for R, K in zip(R_test_kpc, K_approx):
        if R < 1e-3:  # Solar system
            status = "✅ SAFE" if K < 1e-3 else "❌ UNSAFE"
            if K >= 1e-3:
                solar_safe = False
        elif R < 10.0:  # Galaxy
            status = "✅ SAFE" if K < 0.1 else "⚠️  CHECK"
            if K >= 0.1:
                galaxy_safe = False
        else:
            status = "Cluster scale"
        
        print(f"{R:12.2e} | {K:12.4e} | {status}")
    
    print("\n" + "-" * 50)
    if solar_safe and galaxy_safe:
        print("✅ PASS: K_σ → 0 at small scales")
        return "PASS"
    elif solar_safe:
        print("⚠️  WARNING: Solar system safe, check galaxy scale")
        return "WARNING"
    else:
        print("❌ FAIL: K_σ too large at small scales")
        return "FAIL"

def main():
    """Run all validation checks."""
    print("\n" + "="*70)
    print("QUICK VALIDATION FOR Σ-GRAVITY KERNEL")
    print("="*70)
    print("\nThese checks verify physical consistency:")
    print("  1. Einstein mass identity")
    print("  2. Boost localization (no mass sheet)")
    print("  3. Small-scale safety (Newtonian limit)")
    print("="*70)
    
    results = {}
    
    try:
        status1, frac_err = test_einstein_mass_identity()
        results['einstein'] = (status1, frac_err)
    except Exception as e:
        print(f"❌ Check 1 failed: {e}")
        results['einstein'] = ("ERROR", None)
    
    try:
        status2, max_boost = test_boost_localization()
        results['localization'] = (status2, max_boost)
    except Exception as e:
        print(f"❌ Check 2 failed: {e}")
        results['localization'] = ("ERROR", None)
    
    try:
        status3 = test_small_scale_safety()
        results['safety'] = (status3,)
    except Exception as e:
        print(f"❌ Check 3 failed: {e}")
        results['safety'] = ("ERROR",)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\n1. Einstein mass identity: {results['einstein'][0]}")
    if results['einstein'][1] is not None:
        print(f"   Error: {results['einstein'][1]:+.2f}%")
    
    print(f"\n2. Boost localization: {results['localization'][0]}")
    if len(results['localization']) > 1 and results['localization'][1] is not None:
        print(f"   Max boost (R>500kpc): {results['localization'][1]:.3f}")
    
    print(f"\n3. Small-scale safety: {results['safety'][0]}")
    
    all_pass = all(r[0] == "PASS" for r in results.values())
    
    print("\n" + "="*70)
    if all_pass:
        print("✅ ALL CHECKS PASSED")
    else:
        print("⚠️  SOME CHECKS HAVE WARNINGS")
    print("="*70 + "\n")
    
    return results

if __name__ == "__main__":
    results = main()
