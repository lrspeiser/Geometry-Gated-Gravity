#!/usr/bin/env python3
"""
Unit Test: Uniform Sphere Interior Chord Normalization
=======================================================

Tests that interior chords contribute correctly by comparing against
an analytic uniform sphere solution.

Physical Setup
--------------
Consider a uniform sphere of radius R_sphere with constant density ρ_0.
For a field point at projected radius R < R_sphere:

- Interior shells (r < R): Contribute via chords of length L = 2√(r² - R²)
- Exterior shells (r > R): Contribute via arc paths

With NO coherence damping (ℓ₀ → ∞), the 3D shell integral should reduce
to the standard Abel projection, and we can verify the interior chord
contribution analytically.

Expected Behavior
-----------------
For R ~ 0.5 R_sphere:
- Interior mass should dominate K_Σ (dense core sampled by chords)
- With coherence off, 3D shell ≈ 2D ring projection (< 1% error)

Test Cases
----------
1. Coherence off: 3D shell vs 2D ring (should agree)
2. Interior only: w_interior=1, w_exterior=0
3. Exterior only: w_interior=0, w_exterior=1
4. Both families: Interior should dominate at R < R_sphere/2

Success Criteria
----------------
- Interior chords contribute > 50% for R < R_sphere/2
- 3D shell ≈ 2D ring when ℓ₀ → ∞ (< 1% error)
- Total K_Σ has correct sign and magnitude

Author: GravityCalculator Cluster Kernel Fix
Date: 2025-01-13
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.integrate import trapezoid

# Import 3D shell kernel
from core.cluster_kernel_3d_shell import (
    Shell3DKernelParams,
    K_Sigma_3D_shell,
    interior_contribution,
    exterior_contribution
)


def uniform_sphere_profile(r: np.ndarray, R_sphere: float, rho_0: float) -> np.ndarray:
    """
    Uniform sphere density profile.
    
    ρ(r) = ρ_0  for r < R_sphere
    ρ(r) = 0    for r ≥ R_sphere
    """
    rho = np.zeros_like(r)
    rho[r < R_sphere] = rho_0
    return rho


def abel_project_uniform_sphere(R: float, R_sphere: float, rho_0: float) -> float:
    """
    Analytic Abel projection of uniform sphere.
    
    Σ(R) = 2 ∫_R^R_sphere ρ_0 × r/√(r² - R²) dr
         = 2 ρ_0 √(R_sphere² - R²)    for R < R_sphere
         = 0                           for R ≥ R_sphere
    
    Parameters
    ----------
    R : float
        Projected radius [kpc]
    R_sphere : float
        Sphere radius [kpc]
    rho_0 : float
        Constant density [Msun/kpc³]
    
    Returns
    -------
    Sigma : float
        Surface density [Msun/kpc²]
    """
    if R >= R_sphere:
        return 0.0
    else:
        return 2.0 * rho_0 * np.sqrt(R_sphere**2 - R**2)


def test_coherence_off():
    """
    Test 1: With coherence off (ℓ₀ → ∞), 3D shell should match 2D ring.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Coherence Off (3D shell vs 2D ring projection)")
    print("=" * 70)
    print()
    
    # Uniform sphere
    R_sphere = 300.0  # kpc
    rho_0 = 1e6  # Msun/kpc³
    
    r_grid = np.linspace(0.1, R_sphere * 2, 1000)
    rho_3d = uniform_sphere_profile(r_grid, R_sphere, rho_0)
    
    # Test at various projected radii
    R_test = np.array([50, 100, 150, 200, 250])
    
    print(f"Uniform sphere: R_sphere = {R_sphere} kpc, ρ_0 = {rho_0:.0e} Msun/kpc³")
    print()
    
    # 3D shell kernel with very large ℓ₀ (coherence off)
    params = Shell3DKernelParams(
        A_c=1.0,  # Amplitude = 1 (no boost)
        r_gate=0.1,  # Tiny gate (no suppression)
        n_gate=1,
        ell0=1e6,  # Very large (no coherence damping)
        p_density=1.0,  # Linear density weighting
        L1=1e6,  # Very large (no taper)
        q_taper=1,
        w_interior=1.0,
        w_exterior=1.0,
        coherence_mode='power_law',
        n_coh=0.1  # Very weak damping
    )
    
    print("3D Shell Kernel vs Analytic Abel Projection:")
    print(f"{'R [kpc]':<10} {'Σ_analytic':<15} {'K_Σ (3D)':<15} {'Ratio':<10}")
    print("─" * 70)
    
    for R in R_test:
        # Analytic surface density
        Sigma_analytic = abel_project_uniform_sphere(R, R_sphere, rho_0)
        
        # 3D shell kernel boost (should be ~0 when coherence off)
        K_Sigma_3d = K_Sigma_3D_shell(np.array([R]), r_grid, rho_3d, params, normalize=False)[0]
        
        # With no boost, surface density from numerical projection
        # (We're testing the kernel normalization, not the projection itself)
        
        print(f"{R:<10.0f} {Sigma_analytic:<15.2e} {K_Sigma_3d:<15.3f} {K_Sigma_3d:.3f}")
    
    print()
    print("Note: With A_c=1 and ℓ₀→∞, K_Σ should be small (<<1)")
    print("      Testing kernel normalization, not total projection")
    print()


def test_interior_exterior_split():
    """
    Test 2: Interior vs exterior contributions for uniform sphere.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Interior vs Exterior Chord/Arc Contributions")
    print("=" * 70)
    print()
    
    # Uniform sphere
    R_sphere = 300.0  # kpc
    rho_0 = 1e6  # Msun/kpc³
    
    r_grid = np.linspace(0.1, R_sphere * 2, 1000)
    rho_3d = uniform_sphere_profile(r_grid, R_sphere, rho_0)
    
    # Kernel parameters
    params = Shell3DKernelParams(
        A_c=1.0,
        r_gate=5.0,
        n_gate=4,
        ell0=150.0,  # Cluster-scale coherence
        p_density=1.0,
        L1=1000.0,
        q_taper=2.0,
        w_interior=1.0,
        w_exterior=1.0,
        coherence_mode='power_law',
        n_coh=1.5
    )
    
    # Test at R = R_sphere/2 (interior chords should dominate)
    R_test = R_sphere / 2
    
    print(f"Testing at R = {R_test:.0f} kpc (R_sphere/2)")
    print(f"Coherence length ℓ₀ = {params.ell0} kpc")
    print()
    
    # Split grid into interior and exterior
    mask_int = r_grid < R_test
    mask_ext = r_grid >= R_test
    
    r_int = r_grid[mask_int]
    rho_int = rho_3d[mask_int]  # ACTUAL density, not normalized!
    
    r_ext = r_grid[mask_ext]
    rho_ext = rho_3d[mask_ext]  # ACTUAL density, not normalized!
    
    # Compute baseline surface density at R_test
    Sigma_baseline = abel_project_uniform_sphere(R_test, R_sphere, rho_0)
    
    # Compute contributions (now with Sigma_baseline)
    K_int = interior_contribution(R_test, r_int, rho_int, params, Sigma_baseline)
    K_ext = exterior_contribution(R_test, r_ext, rho_ext, params, Sigma_baseline)
    K_total = K_int + K_ext
    
    print(f"Interior chord contribution: K_int = {K_int:.6f}")
    print(f"Exterior arc contribution:   K_ext = {K_ext:.6f}")
    print(f"Total:                       K_tot = {K_total:.6f}")
    print()
    
    # Fractions
    if K_total > 0:
        frac_int = K_int / K_total
        frac_ext = K_ext / K_total
        print(f"Interior fraction: {frac_int*100:.1f}%")
        print(f"Exterior fraction: {frac_ext*100:.1f}%")
        print()
        
        if frac_int > 0.5:
            print("✓ PASS: Interior chords dominate at R = R_sphere/2")
        else:
            print("✗ FAIL: Interior chords should dominate here!")
    else:
        print("✗ FAIL: Total contribution is zero!")
    print()


def test_ablation_study():
    """
    Test 3: Ablation study - interior only, exterior only, both.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Ablation Study (Interior vs Exterior)")
    print("=" * 70)
    print()
    
    # Uniform sphere
    R_sphere = 300.0  # kpc
    rho_0 = 1e6  # Msun/kpc³
    
    r_grid = np.linspace(0.1, R_sphere * 2, 1000)
    rho_3d = uniform_sphere_profile(r_grid, R_sphere, rho_0)
    
    # Test radii
    R_test = np.array([100, 150, 200])  # kpc
    
    configs = [
        {'name': 'Interior only', 'w_int': 1.0, 'w_ext': 0.0},
        {'name': 'Exterior only', 'w_int': 0.0, 'w_ext': 1.0},
        {'name': 'Both (full)', 'w_int': 1.0, 'w_ext': 1.0},
    ]
    
    print(f"{'R [kpc]':<10} {'Interior':<15} {'Exterior':<15} {'Both':<15}")
    print("─" * 70)
    
    for R in R_test:
        results = []
        for config in configs:
            params = Shell3DKernelParams(
                A_c=1.0,
                r_gate=5.0,
                n_gate=4,
                ell0=150.0,
                p_density=1.0,
                L1=1000.0,
                w_interior=config['w_int'],
                w_exterior=config['w_ext'],
                coherence_mode='power_law',
                n_coh=1.5
            )
            K = K_Sigma_3D_shell(np.array([R]), r_grid, rho_3d, params)[0]
            results.append(K)
        
        print(f"{R:<10.0f} {results[0]:<15.6f} {results[1]:<15.6f} {results[2]:<15.6f}")
    
    print()
    print("Expected: Both > Interior-only, Both > Exterior-only")
    print()


def run_all_tests():
    """Run all uniform sphere unit tests."""
    print("\n" + "=" * 70)
    print("UNIFORM SPHERE KERNEL UNIT TESTS")
    print("Testing Interior Chord Normalization")
    print("=" * 70)
    
    test_coherence_off()
    test_interior_exterior_split()
    test_ablation_study()
    
    print("\n" + "=" * 70)
    print("UNIT TESTS COMPLETE")
    print("=" * 70)
    print()
    print("Next: Fix interior chord weighting if tests fail")
    print("      Then run test_macs0416_full_physics.py again")
    print()


if __name__ == '__main__':
    run_all_tests()
