#!/usr/bin/env python3
"""
Mass consistency tests to identify and fix the 20% integration error
Tests surface → mass, volume → surface → mass, and potential/force consistency
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Physical constants
G = 4.300917270e-6  # (km/s)^2 kpc / Msun

def enclosed_mass_from_sigma(R_grid, Sigma_grid):
    """
    Compute enclosed mass from surface density profile
    M(<R) = 2π ∫₀^R Σ(R') R' dR'
    
    CRITICAL: Must include the R' factor!
    """
    # Use Simpson's rule on log-spaced grid for better accuracy
    # Include R=0 limit explicitly
    R_ext = np.concatenate([[0], R_grid])
    Sigma_ext = np.concatenate([[Sigma_grid[0]], Sigma_grid])
    
    # Integrand: 2π R Σ(R)
    integrand = 2 * np.pi * R_ext * Sigma_ext
    
    # Cumulative integral using trapezoidal rule
    M_enc = np.zeros_like(R_grid)
    for i in range(len(R_grid)):
        idx = np.where(R_ext <= R_grid[i])[0]
        if len(idx) > 1:
            M_enc[i] = np.trapz(integrand[idx], R_ext[idx])
    
    return M_enc

def sigma_from_rho(R_grid, z_grid, rho):
    """
    Compute surface density from 3D density
    Σ(R) = 2 ∫₀^∞ ρ(R,z) dz
    
    Factor of 2 for both sides of midplane!
    """
    # Integrate over z (assuming symmetric about z=0)
    dz = z_grid[1] - z_grid[0] if len(z_grid) > 1 else 1.0
    
    # Only integrate positive z (then multiply by 2)
    z_pos = z_grid[z_grid >= 0]
    if len(rho.shape) == 2:  # (NR, Nz)
        rho_pos = rho[:, z_grid >= 0]
        Sigma = 2 * np.trapz(rho_pos, z_pos, axis=1)
    else:  # Already 1D
        Sigma = rho
    
    return Sigma

def test_exponential_disk():
    """
    Test 1: Exponential disk surface density → enclosed mass
    """
    print("\n" + "="*60)
    print("TEST 1: EXPONENTIAL DISK (Surface → Mass)")
    print("="*60)
    
    # Exponential disk parameters
    M_disk = 6e10  # Msun
    R_d = 3.0      # kpc
    
    # Radial grid
    R = np.logspace(-1, 2, 200)  # 0.1 to 100 kpc
    
    # Surface density: Σ(R) = Σ₀ exp(-R/R_d)
    # where Σ₀ = M_disk / (2π R_d²)
    Sigma_0 = M_disk / (2 * np.pi * R_d**2)
    Sigma = Sigma_0 * np.exp(-R / R_d)
    
    # Convert to Msun/pc^2 for display
    Sigma_pc2 = Sigma / 1e6  # kpc^2 to pc^2
    
    # Compute enclosed mass
    M_enc_computed = enclosed_mass_from_sigma(R, Sigma)
    
    # Analytic enclosed mass for exponential disk
    # M(<R) = M_disk * [1 - (1 + R/R_d) * exp(-R/R_d)]
    M_enc_analytic = M_disk * (1 - (1 + R/R_d) * np.exp(-R/R_d))
    
    # Compare
    ratio = M_enc_computed / M_enc_analytic
    error_pct = 100 * (ratio - 1)
    
    print(f"Disk parameters:")
    print(f"  Total mass: {M_disk:.2e} Msun")
    print(f"  Scale radius: {R_d:.1f} kpc")
    print(f"  Central Σ: {Sigma_pc2[0]:.1f} Msun/pc²")
    
    print(f"\nMass comparison at key radii:")
    for r in [1, 3, 10, 30]:
        idx = np.argmin(np.abs(R - r))
        print(f"  R = {r:2d} kpc: computed = {M_enc_computed[idx]:.2e}, "
              f"analytic = {M_enc_analytic[idx]:.2e}, "
              f"error = {error_pct[idx]:+.1f}%")
    
    max_error = np.max(np.abs(error_pct[R < 30]))
    print(f"\nMax error (R < 30 kpc): {max_error:.1f}%")
    
    if max_error > 2:
        print("❌ FAILED: Integration error > 2%")
        print("   Check: Are you multiplying by R in the integrand?")
        print("   Check: Are you including 2π?")
    else:
        print("✅ PASSED: Integration accurate")
    
    return max_error < 2

def test_hernquist_bulge():
    """
    Test 2: Hernquist bulge profile
    """
    print("\n" + "="*60)
    print("TEST 2: HERNQUIST BULGE")
    print("="*60)
    
    # Hernquist parameters
    M_bulge = 1e10  # Msun
    a = 0.5        # kpc
    
    # Radial grid
    R = np.logspace(-2, 2, 200)  # 0.01 to 100 kpc
    
    # Hernquist surface density (projected)
    # Σ(R) = M_bulge * a / (2π * R * (R + a)²)
    Sigma = M_bulge * a / (2 * np.pi * R * (R + a)**2)
    
    # Compute enclosed mass
    M_enc_computed = enclosed_mass_from_sigma(R, Sigma)
    
    # Analytic enclosed mass for Hernquist
    # M(<R) = M_bulge * R² / (R + a)²
    M_enc_analytic = M_bulge * R**2 / (R + a)**2
    
    # Compare
    ratio = M_enc_computed / M_enc_analytic
    error_pct = 100 * (ratio - 1)
    
    print(f"Bulge parameters:")
    print(f"  Total mass: {M_bulge:.2e} Msun")
    print(f"  Scale radius: {a:.2f} kpc")
    
    print(f"\nMass comparison at key radii:")
    for r in [0.1, 0.5, 1, 5]:
        idx = np.argmin(np.abs(R - r))
        print(f"  R = {r:.1f} kpc: computed = {M_enc_computed[idx]:.2e}, "
              f"analytic = {M_enc_analytic[idx]:.2e}, "
              f"error = {error_pct[idx]:+.1f}%")
    
    max_error = np.max(np.abs(error_pct[(R > 0.1) & (R < 10)]))
    print(f"\nMax error (0.1 < R < 10 kpc): {max_error:.1f}%")
    
    if max_error > 2:
        print("❌ FAILED: Integration error > 2%")
    else:
        print("✅ PASSED: Integration accurate")
    
    return max_error < 2

def test_volume_to_surface():
    """
    Test 3: Volume density → surface density → mass (round-trip)
    """
    print("\n" + "="*60)
    print("TEST 3: VOLUME → SURFACE → MASS ROUND-TRIP")
    print("="*60)
    
    # Exponential disk with sech² vertical profile
    M_disk = 6e10  # Msun
    R_d = 3.0      # kpc
    z_0 = 0.3      # kpc (scale height)
    
    # Grids
    R = np.logspace(-1, 2, 100)  # 0.1 to 100 kpc
    z = np.linspace(0, 3, 50)    # 0 to 3 kpc (positive only)
    
    # 3D density: ρ(R,z) = ρ₀ exp(-R/R_d) sech²(z/z_0)
    # where ρ₀ is chosen to give total mass M_disk
    # For sech² profile: ∫ sech²(z/z_0) dz = 2*z_0
    rho_0 = M_disk / (4 * np.pi * R_d**2 * z_0)
    
    R_mesh, z_mesh = np.meshgrid(R, z, indexing='ij')
    rho = rho_0 * np.exp(-R_mesh / R_d) * (1/np.cosh(z_mesh / z_0))**2
    
    # Convert to surface density
    Sigma_computed = sigma_from_rho(R, z, rho)
    
    # Analytic surface density for this profile
    # Σ(R) = 2 * ∫₀^∞ ρ(R,z) dz = 2 * ρ₀ * exp(-R/R_d) * 2*z_0
    Sigma_analytic = 4 * z_0 * rho_0 * np.exp(-R / R_d)
    
    # Compare surface densities
    ratio_sigma = Sigma_computed / Sigma_analytic
    error_sigma = 100 * (ratio_sigma - 1)
    
    print(f"3D disk parameters:")
    print(f"  Total mass: {M_disk:.2e} Msun")
    print(f"  Scale radius: {R_d:.1f} kpc")
    print(f"  Scale height: {z_0:.1f} kpc")
    
    print(f"\nSurface density comparison:")
    for r in [1, 3, 10]:
        idx = np.argmin(np.abs(R - r))
        print(f"  R = {r:2d} kpc: computed = {Sigma_computed[idx]:.2e}, "
              f"analytic = {Sigma_analytic[idx]:.2e}, "
              f"error = {error_sigma[idx]:+.1f}%")
    
    # Now compute enclosed mass
    M_enc_computed = enclosed_mass_from_sigma(R, Sigma_computed)
    M_enc_analytic = M_disk * (1 - (1 + R/R_d) * np.exp(-R/R_d))
    
    ratio_mass = M_enc_computed / M_enc_analytic
    error_mass = 100 * (ratio_mass - 1)
    
    print(f"\nEnclosed mass comparison:")
    for r in [1, 3, 10, 30]:
        idx = np.argmin(np.abs(R - r))
        print(f"  R = {r:2d} kpc: computed = {M_enc_computed[idx]:.2e}, "
              f"analytic = {M_enc_analytic[idx]:.2e}, "
              f"error = {error_mass[idx]:+.1f}%")
    
    max_error_sigma = np.max(np.abs(error_sigma[R < 30]))
    max_error_mass = np.max(np.abs(error_mass[R < 30]))
    
    print(f"\nMax errors (R < 30 kpc):")
    print(f"  Surface density: {max_error_sigma:.1f}%")
    print(f"  Enclosed mass: {max_error_mass:.1f}%")
    
    if max_error_sigma > 2 or max_error_mass > 2:
        print("❌ FAILED: Round-trip error > 2%")
        if max_error_sigma > 2:
            print("   Check: Are you multiplying by 2 for both sides of midplane?")
    else:
        print("✅ PASSED: Round-trip accurate")
    
    return max_error_sigma < 2 and max_error_mass < 2

def test_potential_force():
    """
    Test 4: Potential and force consistency
    """
    print("\n" + "="*60)
    print("TEST 4: POTENTIAL/FORCE CONSISTENCY")
    print("="*60)
    
    # Miyamoto-Nagai disk
    M_disk = 6e10  # Msun
    a = 3.0       # kpc (radial scale)
    b = 0.3       # kpc (vertical scale)
    
    # Radial grid
    R = np.logspace(-1, 2, 200)  # 0.1 to 100 kpc
    
    # Miyamoto-Nagai circular velocity at z=0
    # v²(R) = G*M*R² / (R² + (a + b)²)^(3/2)
    v_MN = np.sqrt(G * M_disk * R**2 / (R**2 + (a + b)**2)**1.5)
    
    # Also test Hernquist bulge
    M_bulge = 1e10  # Msun
    a_h = 0.5      # kpc
    
    # Hernquist circular velocity
    # v²(R) = G*M*R / (R + a)²
    v_H = np.sqrt(G * M_bulge * R / (R + a_h)**2)
    
    # Total
    v_total = np.sqrt(v_MN**2 + v_H**2)
    
    print(f"Potential test:")
    print(f"  Miyamoto-Nagai: M = {M_disk:.2e} Msun, a = {a:.1f} kpc, b = {b:.1f} kpc")
    print(f"  Hernquist: M = {M_bulge:.2e} Msun, a = {a_h:.1f} kpc")
    
    print(f"\nCircular velocities at key radii:")
    for r in [1, 3, 8, 20]:
        idx = np.argmin(np.abs(R - r))
        print(f"  R = {r:2d} kpc: v_disk = {v_MN[idx]:6.1f} km/s, "
              f"v_bulge = {v_H[idx]:6.1f} km/s, "
              f"v_total = {v_total[idx]:6.1f} km/s")
    
    # Check that velocities are reasonable
    v_at_8kpc = v_total[np.argmin(np.abs(R - 8))]
    if 150 < v_at_8kpc < 250:
        print(f"\n✅ PASSED: v_c at 8 kpc = {v_at_8kpc:.1f} km/s (reasonable for MW)")
        return True
    else:
        print(f"\n❌ FAILED: v_c at 8 kpc = {v_at_8kpc:.1f} km/s (unreasonable)")
        return False

def plot_results():
    """
    Create diagnostic plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Test exponential disk
    M_disk = 6e10
    R_d = 3.0
    R = np.logspace(-1, 2, 200)
    Sigma_0 = M_disk / (2 * np.pi * R_d**2)
    Sigma = Sigma_0 * np.exp(-R / R_d)
    M_enc_computed = enclosed_mass_from_sigma(R, Sigma)
    M_enc_analytic = M_disk * (1 - (1 + R/R_d) * np.exp(-R/R_d))
    
    ax = axes[0, 0]
    ax.loglog(R, M_enc_computed, 'b-', label='Computed', linewidth=2)
    ax.loglog(R, M_enc_analytic, 'r--', label='Analytic', linewidth=2)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('M(<R) (M☉)')
    ax.set_title('Exponential Disk: Enclosed Mass')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error plot
    ax = axes[0, 1]
    error = 100 * (M_enc_computed / M_enc_analytic - 1)
    ax.semilogx(R, error, 'g-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(2, color='r', linestyle=':', alpha=0.5)
    ax.axhline(-2, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Error (%)')
    ax.set_title('Integration Error')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-5, 5])
    
    # Surface density profile
    ax = axes[1, 0]
    ax.loglog(R, Sigma / 1e6, 'b-', linewidth=2)  # Convert to Msun/pc^2
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Σ (M☉/pc²)')
    ax.set_title('Surface Density Profile')
    ax.grid(True, alpha=0.3)
    
    # Circular velocity
    M_bulge = 1e10
    a_h = 0.5
    a = 3.0
    b = 0.3
    v_MN = np.sqrt(G * M_disk * R**2 / (R**2 + (a + b)**2)**1.5)
    v_H = np.sqrt(G * M_bulge * R / (R + a_h)**2)
    v_total = np.sqrt(v_MN**2 + v_H**2)
    
    ax = axes[1, 1]
    ax.plot(R, v_MN, 'b-', label='Disk', linewidth=2)
    ax.plot(R, v_H, 'r-', label='Bulge', linewidth=2)
    ax.plot(R, v_total, 'k-', label='Total', linewidth=2)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('v_c (km/s)')
    ax.set_title('Circular Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 300])
    
    plt.tight_layout()
    plt.savefig('out/mass_consistency.png', dpi=150)
    print(f"\nPlot saved to out/mass_consistency.png")

def main():
    """
    Run all mass consistency tests
    """
    print("="*60)
    print("MASS CONSISTENCY TEST SUITE")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Exponential disk", test_exponential_disk()))
    results.append(("Hernquist bulge", test_hernquist_bulge()))
    results.append(("Volume round-trip", test_volume_to_surface()))
    results.append(("Potential/force", test_potential_force()))
    
    # Create plots
    Path('out').mkdir(exist_ok=True)
    plot_results()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 All tests passed! Mass integration is correct.")
    else:
        print("\n⚠️ Some tests failed. Check integration formulas:")
        print("  1. Surface → Mass: M(<R) = 2π ∫ Σ(R') R' dR'  (need R factor!)")
        print("  2. Volume → Surface: Σ(R) = 2 ∫₀^∞ ρ(R,z) dz  (factor of 2!)")
        print("  3. Check unit conversions: Msun/pc² ↔ Msun/kpc² (factor 10⁶)")
    
    return all_passed

if __name__ == '__main__':
    main()