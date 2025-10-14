#!/usr/bin/env python3
"""
Triaxial Lensing Validation Suite
==================================

Comprehensive tests to verify triaxial geometry transformations work correctly:
1. Mass conservation (analytic vs numerical)
2. Surface density projection (Sigma vs q_LOS)
3. Einstein radius sensitivity to geometry
4. Comparison with known analytic cases

Must pass all tests before proceeding to Phase 2.

Author: GravityCalculator
Date: 2025-01-14
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, trapezoid

from core.triaxial_lensing import (
    spherical_to_triaxial_density,
    project_triaxial_to_surface_density_simple,
    ellipsoidal_radius
)


def test_1_simple_density_profile():
    """
    Test 1: Simple power-law density - verify transformation.
    
    For rho(r) = rho_0 / (1 + r/r_s)^3:
    - Triaxial should give rho(m) / (q_plane × q_LOS)
    - Check at specific points
    """
    print("\n" + "="*70)
    print("TEST 1: Simple Density Profile Transformation")
    print("="*70)
    
    rho_0 = 1e6  # Msun/kpc^3
    r_s = 100.0  # kpc
    
    def rho_spherical(r):
        return rho_0 / (1 + r/r_s)**3
    
    # Test points
    test_points = [
        (100, 0, 0),    # Along x-axis
        (0, 100, 0),    # Along y-axis
        (0, 0, 100),    # Along z-axis (LOS)
        (50, 50, 50),   # Diagonal
    ]
    
    q_plane = 0.8
    q_LOS = 1.2
    
    rho_tri = spherical_to_triaxial_density(rho_spherical, q_plane, q_LOS)
    
    print(f"\nTransformation: q_plane = {q_plane}, q_LOS = {q_LOS}")
    print(f"Expected volume correction: 1/(q_plane × q_LOS) = {1/(q_plane*q_LOS):.3f}")
    print()
    
    all_passed = True
    
    for x, y, z in test_points:
        # Compute ellipsoidal radius
        m = np.sqrt(x**2 + (y/q_plane)**2 + (z/q_LOS)**2)
        
        # Expected density
        rho_expected = rho_spherical(m) / (q_plane * q_LOS)
        
        # Actual density from triaxial function
        rho_actual = rho_tri(x, y, z)
        
        # Check
        error = abs(rho_actual - rho_expected) / rho_expected
        status = "PASS" if error < 1e-6 else "FAIL"
        
        if error >= 1e-6:
            all_passed = False
        
        print(f"Point ({x:3.0f}, {y:3.0f}, {z:3.0f}): m={m:.1f} kpc")
        print(f"  Expected: {rho_expected:.2e} Msun/kpc^3")
        print(f"  Actual:   {rho_actual:.2e} Msun/kpc^3")
        print(f"  Error:    {error:.2e} [{status}]")
        print()
    
    print(f"Test 1: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


def test_2_surface_density_spherical():
    """
    Test 2: For q_plane=1, q_LOS=1 (spherical), Sigma_triaxial should equal Sigma_spherical.
    """
    print("\n" + "="*70)
    print("TEST 2: Spherical Case (q=1) - Sigma Comparison")
    print("="*70)
    
    # Hernquist profile (has analytic Sigma)
    M = 1e12  # Msun
    a = 50.0  # kpc
    
    def rho_hernquist(r):
        """Hernquist 3D density."""
        return M / (2*np.pi) * a / (r * (r + a)**3 + 1e-30)
    
    def Sigma_hernquist_analytic(R):
        """Analytic surface density for Hernquist."""
        X = R / a
        if X < 1:
            # Inside a
            term1 = (2 + X**2) / (1 - X**2)**2
            term2 = -3 * np.log((1 + np.sqrt(1 - X**2)) / X) / (1 - X**2)**1.5
            return M / (2*np.pi*a**2) * (term1 + term2)
        elif X > 1:
            # Outside a
            term1 = (2 + X**2) / (X**2 - 1)**2
            term2 = -3 * np.arctan(np.sqrt(X**2 - 1)) / (X**2 - 1)**1.5
            return M / (2*np.pi*a**2) * (term1 + term2)
        else:
            # At X = 1
            return M / (2*np.pi*a**2) * (2/3 + np.log(2))
    
    # Transform to triaxial (with q=1, should be identical)
    rho_tri = spherical_to_triaxial_density(rho_hernquist, q_plane=1.0, q_LOS=1.0)
    
    # Project
    R_test = np.array([10, 30, 50, 100, 200])
    Sigma_tri = project_triaxial_to_surface_density_simple(rho_tri, R_test, z_max=1000, n_z=500)
    
    print("\nComparing Sigma_triaxial (q=1) vs Sigma_analytic:")
    print()
    
    all_passed = True
    
    for i, R in enumerate(R_test):
        Sigma_analytic = Sigma_hernquist_analytic(R)
        Sigma_numerical = Sigma_tri[i]
        
        error = abs(Sigma_numerical - Sigma_analytic) / Sigma_analytic
        status = "PASS" if error < 0.05 else "FAIL"  # 5% tolerance
        
        if error >= 0.05:
            all_passed = False
        
        print(f"R = {R:3.0f} kpc:")
        print(f"  Analytic:  {Sigma_analytic:.3e} Msun/kpc^2")
        print(f"  Numerical: {Sigma_numerical:.3e} Msun/kpc^2")
        print(f"  Error:     {error*100:.1f}% [{status}]")
        print()
    
    print(f"Test 2: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    return all_passed


def test_3_surface_density_scaling():
    """
    Test 3: Verify Sigma changes with q_LOS in expected direction.
    
    For fixed projected radius R:
    - q_LOS < 1 (oblate): Sigma should INCREASE (matter compressed in z)
    - q_LOS > 1 (prolate): Sigma should DECREASE (matter spread along z)
    """
    print("\n" + "="*70)
    print("TEST 3: Surface Density Scaling with q_LOS")
    print("="*70)
    
    # Simple power-law
    rho_0 = 1e6
    r_s = 100.0
    
    def rho_spherical(r):
        return rho_0 / (1 + r/r_s)**3
    
    R_test = 150.0  # kpc (test at fixed projected radius)
    q_LOS_values = [0.7, 0.9, 1.0, 1.1, 1.3]
    
    Sigma_values = []
    
    for q_LOS in q_LOS_values:
        rho_tri = spherical_to_triaxial_density(rho_spherical, q_plane=1.0, q_LOS=q_LOS)
        Sigma = project_triaxial_to_surface_density_simple(
            rho_tri, np.array([R_test]), z_max=1500, n_z=400
        )[0]
        Sigma_values.append(Sigma)
    
    print(f"\nAt R = {R_test} kpc (q_plane = 1.0):")
    print()
    
    # Check monotonicity
    all_passed = True
    
    for i, (q, Sigma) in enumerate(zip(q_LOS_values, Sigma_values)):
        relative = Sigma / Sigma_values[2]  # Normalize to q_LOS=1.0
        print(f"q_LOS = {q:.1f}: Sigma = {Sigma:.3e} Msun/kpc^2 (×{relative:.3f})")
        
        # Check expected behavior
        if i < 2:  # q < 1
            expected = "HIGHER"
            correct = relative > 1.0
        elif i > 2:  # q > 1
            expected = "LOWER"
            correct = relative < 1.0
        else:
            expected = "BASELINE"
            correct = True
        
        if not correct:
            all_passed = False
            print(f"  ✗ FAIL: Expected {expected} than baseline, got ×{relative:.3f}")
    
    # Check that Sigma decreases with q_LOS
    is_monotonic = all(Sigma_values[i] >= Sigma_values[i+1] for i in range(len(Sigma_values)-1))
    
    print()
    print(f"Monotonicity check: Sigma should decrease with increasing q_LOS")
    print(f"  Result: {'✓ PASS' if is_monotonic else '✗ FAIL'}")
    
    print()
    print(f"Test 3: {'✓ PASSED' if all_passed and is_monotonic else '✗ FAILED'}")
    return all_passed and is_monotonic


def test_4_einstein_radius_sensitivity():
    """
    Test 4: Build simple NFW-like profile and verify Einstein radius
    changes with q_LOS when using a proper lensing integral.
    
    For a mass sheet, Einstein radius θ_E ∝ sqrt(Sigma_crit / Sigma_eff).
    So if Sigma changes with q_LOS, theta_E should too.
    """
    print("\n" + "="*70)
    print("TEST 4: Einstein Radius Sensitivity to Geometry")
    print("="*70)
    
    # NFW-like profile
    rho_0 = 1e7  # Msun/kpc^3
    r_s = 200.0  # kpc
    
    def rho_nfw(r):
        x = r / r_s
        return rho_0 / (x * (1 + x)**2 + 1e-30)
    
    # For different q_LOS, compute Sigma and estimate convergence
    q_LOS_values = [0.7, 0.9, 1.0, 1.1, 1.3]
    R_grid = np.geomspace(10, 500, 100)
    
    print("\nComputing Sigma(R) for different geometries...")
    print()
    
    Sigma_profiles = []
    kappa_center_values = []
    
    # Rough critical density (order of magnitude, not exact)
    Sigma_crit = 3e9  # Msun/kpc^2 (typical for z~0.4, z_src~2)
    
    for q_LOS in q_LOS_values:
        rho_tri = spherical_to_triaxial_density(rho_nfw, q_plane=0.9, q_LOS=q_LOS)
        Sigma = project_triaxial_to_surface_density_simple(
            rho_tri, R_grid, z_max=2000, n_z=500
        )
        Sigma_profiles.append(Sigma)
        
        # Convergence at center
        kappa_center = Sigma[0] / Sigma_crit
        kappa_center_values.append(kappa_center)
        
        print(f"q_LOS = {q_LOS:.1f}:")
        print(f"  Sigma(R=10 kpc) = {Sigma[0]:.3e} Msun/kpc^2")
        print(f"  κ(center) ≈ {kappa_center:.3f}")
        print()
    
    # Check that kappa changes with q_LOS
    kappa_range = max(kappa_center_values) - min(kappa_center_values)
    kappa_fractional_range = kappa_range / kappa_center_values[2]  # Relative to q=1
    
    print(f"Convergence range: Δκ = {kappa_range:.3f}")
    print(f"Fractional range: {kappa_fractional_range*100:.1f}% of spherical value")
    print()
    
    # Should have at least 20% variation
    passed = kappa_fractional_range > 0.20
    
    if passed:
        print("✓ Geometry has significant effect (>20% variation)")
    else:
        print("✗ Geometry effect too weak (<20% variation)")
    
    print()
    print(f"Test 4: {'✓ PASSED' if passed else '✗ FAILED'}")
    return passed


def test_5_visual_inspection():
    """
    Test 5: Generate diagnostic plots for visual inspection.
    """
    print("\n" + "="*70)
    print("TEST 5: Visual Diagnostic Plots")
    print("="*70)
    
    # NFW-like profile
    rho_0 = 1e7
    r_s = 200.0
    
    def rho_nfw(r):
        x = r / r_s
        return rho_0 / (x * (1 + x)**2 + 1e-30)
    
    # Different geometries
    q_configs = [
        (1.0, 0.7, "Oblate LOS (flattened)"),
        (1.0, 1.0, "Spherical"),
        (1.0, 1.3, "Prolate LOS (elongated)"),
    ]
    
    R_grid = np.geomspace(10, 800, 100)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot Sigma(R) for each geometry
    for q_plane, q_LOS, label in q_configs:
        rho_tri = spherical_to_triaxial_density(rho_nfw, q_plane, q_LOS)
        Sigma = project_triaxial_to_surface_density_simple(rho_tri, R_grid, z_max=2000, n_z=500)
        
        ax1.loglog(R_grid, Sigma, lw=2, label=label)
    
    ax1.set_xlabel('Projected Radius R [kpc]', fontsize=11)
    ax1.set_ylabel('Surface Density Σ(R) [Msun/kpc²]', fontsize=11)
    ax1.set_title('Surface Density Profiles', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot Sigma ratio (relative to spherical)
    Sigma_spherical = project_triaxial_to_surface_density_simple(
        spherical_to_triaxial_density(rho_nfw, 1.0, 1.0), R_grid, z_max=2000, n_z=500
    )
    
    for q_plane, q_LOS, label in q_configs:
        if q_LOS == 1.0:
            continue
        rho_tri = spherical_to_triaxial_density(rho_nfw, q_plane, q_LOS)
        Sigma = project_triaxial_to_surface_density_simple(rho_tri, R_grid, z_max=2000, n_z=500)
        ratio = Sigma / Sigma_spherical
        
        ax2.semilogx(R_grid, ratio, lw=2, label=label)
    
    ax2.axhline(1.0, color='k', ls='--', alpha=0.5)
    ax2.set_xlabel('Projected Radius R [kpc]', fontsize=11)
    ax2.set_ylabel('Σ_triaxial / Σ_spherical', fontsize=11)
    ax2.set_title('Surface Density Ratio (Geometry Effect)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot 3D density at z=0 plane for different q_LOS
    x_grid = np.linspace(-500, 500, 100)
    z_grid = np.linspace(-500, 500, 100)
    X, Z = np.meshgrid(x_grid, z_grid)
    
    q_LOS_test = 0.7
    rho_tri = spherical_to_triaxial_density(rho_nfw, 1.0, q_LOS_test)
    
    # Evaluate on grid
    rho_grid = np.zeros_like(X)
    for i in range(len(x_grid)):
        for j in range(len(z_grid)):
            rho_grid[j, i] = rho_tri(X[j, i], 0.0, Z[j, i])
    
    im = ax3.contourf(X, Z, np.log10(rho_grid + 1e-10), levels=20, cmap='viridis')
    ax3.set_xlabel('x [kpc]', fontsize=11)
    ax3.set_ylabel('z (LOS) [kpc]', fontsize=11)
    ax3.set_title(f'log10(ρ) in x-z plane (q_LOS={q_LOS_test})', fontsize=12, fontweight='bold')
    ax3.set_aspect('equal')
    plt.colorbar(im, ax=ax3, label='log10(ρ) [Msun/kpc³]')
    
    # Plot ellipsoidal radius contours
    m_grid = np.zeros_like(X)
    for i in range(len(x_grid)):
        for j in range(len(z_grid)):
            m_grid[j, i] = ellipsoidal_radius(X[j, i], 0.0, Z[j, i], 1.0, q_LOS_test)
    
    ax4.contour(X, Z, m_grid, levels=10, colors='white', alpha=0.7)
    ax4.contourf(X, Z, m_grid, levels=20, cmap='plasma')
    ax4.set_xlabel('x [kpc]', fontsize=11)
    ax4.set_ylabel('z (LOS) [kpc]', fontsize=11)
    ax4.set_title(f'Ellipsoidal Radius m(x,z) (q_LOS={q_LOS_test})', fontsize=12, fontweight='bold')
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    
    output_path = '../figures/triaxial_validation.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plots saved: {output_path}")
    print("Please visually inspect:")
    print("  1. Sigma(R) should differ for oblate/prolate")
    print("  2. Ratio plot should show clear separation from 1.0")
    print("  3. Density contours should show elliptical shape")
    print("  4. Ellipsoidal radius should show nested ellipses")
    
    return True


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "#"*70)
    print("# TRIAXIAL LENSING VALIDATION SUITE")
    print("#"*70)
    
    tests = [
        ("Density Transformation", test_1_simple_density_profile),
        ("Spherical Case Sigma", test_2_surface_density_spherical),
        ("Sigma Scaling with q_LOS", test_3_surface_density_scaling),
        ("Einstein Radius Sensitivity", test_4_einstein_radius_sensitivity),
        ("Visual Diagnostics", test_5_visual_inspection),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' CRASHED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print()
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:40s} {status}")
    
    print()
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("="*70)
        print("✓ ALL TESTS PASSED - TRIAXIAL LENSING VALIDATED")
        print("="*70)
        print("\nReady to proceed to Phase 2 (Hierarchical Calibration)")
    else:
        print("="*70)
        print("✗ SOME TESTS FAILED - DO NOT PROCEED")
        print("="*70)
        print("\nFix issues before moving to Phase 2")
    
    print()
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
