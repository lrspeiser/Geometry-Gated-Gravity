#!/usr/bin/env python3
"""
Debug MACS0416 Surface Density Issue
====================================

Investigate why the Einstein radius is always zero even with large A_c values.

Check:
1. Surface density magnitude after projection
2. Critical surface density value
3. Convergence κ and mean_kappa profiles
4. Whether mean_kappa ever exceeds 1.0

Author: Debug Investigation
Date: 2025-01-14
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from test_macs0416_projected_kernel import (
    build_macs0416_baryon_profile_3d,
    project_to_surface_density,
    compute_lensing_from_sigma_eff
)
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from many_path_model.lensing_utilities import LensingCosmology


def diagnose_surface_density():
    """Diagnose surface density and lensing calculation."""
    
    print("=" * 70)
    print("MACS0416 SURFACE DENSITY DIAGNOSTIC")
    print("=" * 70)
    print()
    
    # Build 3D baryon profile
    print("Step 1: Building 3D baryon profile...")
    r_3d = np.logspace(-1, 3.5, 2000)
    rho_total, baryon_info = build_macs0416_baryon_profile_3d(r_3d, verbose=False)
    
    print(f"  M_baryon(R_500) = {baryon_info['M_total']:.2e} Msun")
    print(f"  f_gas(R_500) = {baryon_info['fgas']:.3f}")
    print()
    
    # Create 2D grid
    print("Step 2: Creating 2D projection grid...")
    nx, ny = 512, 512
    R_max = 2500.0
    x = np.linspace(-R_max, R_max, nx)
    y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid_2d = np.sqrt(X**2 + Y**2)
    pixel_area = (2*R_max/nx)**2  # kpc^2
    print(f"  Grid: {nx}x{ny}, R_max = {R_max:.0f} kpc")
    print(f"  Pixel area = {pixel_area:.2f} kpc^2")
    print()
    
    # Project to surface density
    print("Step 3: Projecting to surface density...")
    Sigma_triax = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)
    
    print(f"  Sigma min = {np.min(Sigma_triax):.2e} Msun/kpc^2")
    print(f"  Sigma max = {np.max(Sigma_triax):.2e} Msun/kpc^2")
    print(f"  Sigma at R=100 kpc = {Sigma_triax[ny//2, nx//2 + 10]:.2e} Msun/kpc^2")
    
    # Total projected mass
    total_proj_mass = np.sum(Sigma_triax) * pixel_area
    print(f"  Total projected mass = {total_proj_mass:.2e} Msun")
    print()
    
    # Apply kernel with moderate A_c
    print("Step 4: Applying kernel (A_c = 50)...")
    A_c = 50.0
    ell0 = 200.0
    Sigma_eff_2d, K_sigma_2d, kernel_diag = convolve_sigma_with_kernel(
        Sigma_triax, R_grid_2d, ell0, 2.0, 2.0, A_c,
        emphasize_interior=True, use_fft=True
    )
    
    print(f"  <K_sigma> = {kernel_diag['K_sigma_mean']:.4f}")
    print(f"  <Boost> = {kernel_diag['boost_factor_mean']:.4f}")
    print(f"  Sigma_eff max = {np.max(Sigma_eff_2d):.2e} Msun/kpc^2")
    print()
    
    # Azimuthal average
    print("Step 5: Azimuthal averaging...")
    R_bins = np.linspace(0, 2000, 201)
    R_prof, Sigma_prof, _ = azimuthal_average(Sigma_triax, R_grid_2d, R_bins)
    _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff_2d, R_grid_2d, R_bins)
    
    # Show profile at key radii
    for R_check in [100, 200, 500, 1000]:
        idx = np.argmin(np.abs(R_prof - R_check))
        print(f"  R = {R_prof[idx]:.0f} kpc: Sigma = {Sigma_prof[idx]:.2e}, Sigma_eff = {Sigma_eff_prof[idx]:.2e}")
    print()
    
    # Compute lensing
    print("Step 6: Computing lensing observables...")
    cosmo = LensingCosmology()
    z_lens = baryon_info['z']
    z_src = 2.0
    
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_src)
    print(f"  z_lens = {z_lens}")
    print(f"  z_src = {z_src}")
    print(f"  Sigma_crit = {Sigma_crit:.2e} Msun/kpc^2")
    print()
    
    # Convergence
    kappa = Sigma_eff_prof / Sigma_crit
    print(f"  kappa at R=100 kpc: {kappa[np.argmin(np.abs(R_prof - 100))]:.4f}")
    print(f"  kappa at R=200 kpc: {kappa[np.argmin(np.abs(R_prof - 200))]:.4f}")
    print(f"  kappa max: {np.max(kappa):.4f}")
    print()
    
    # Cumulative mass and mean convergence
    from scipy.integrate import cumulative_trapezoid
    M_cum = cumulative_trapezoid(2.0 * np.pi * R_prof * Sigma_eff_prof, R_prof, initial=0.0)
    mean_kappa = M_cum / (np.pi * R_prof**2 * Sigma_crit)
    mean_kappa[0] = kappa[0]
    
    print(f"  M_cum at R=100 kpc: {M_cum[np.argmin(np.abs(R_prof - 100))]:.2e} Msun")
    print(f"  M_cum at R=500 kpc: {M_cum[np.argmin(np.abs(R_prof - 500))]:.2e} Msun")
    print(f"  mean_kappa at R=100 kpc: {mean_kappa[np.argmin(np.abs(R_prof - 100))]:.4f}")
    print(f"  mean_kappa at R=200 kpc: {mean_kappa[np.argmin(np.abs(R_prof - 200))]:.4f}")
    print(f"  mean_kappa max: {np.max(mean_kappa):.4f}")
    print()
    
    # Check where mean_kappa >= 1
    idx_above_1 = np.where(mean_kappa >= 1.0)[0]
    if len(idx_above_1) > 0:
        print(f"  ✓ mean_kappa >= 1.0 at {len(idx_above_1)} radii")
        print(f"    First: R = {R_prof[idx_above_1[0]]:.1f} kpc")
        print(f"    Last (R_E): R = {R_prof[idx_above_1[-1]]:.1f} kpc")
    else:
        print(f"  ❌ mean_kappa NEVER reaches 1.0!")
        print(f"  --> Einstein radius cannot be determined")
    print()
    
    # Diagnostic plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # (0, 0): 3D density profile
    ax = axes[0, 0]
    ax.loglog(r_3d, rho_total, 'b-', lw=2)
    ax.set_xlabel('3D Radius r [kpc]')
    ax.set_ylabel('Density ρ [Msun/kpc³]')
    ax.set_title('3D Baryon Density')
    ax.grid(alpha=0.3)
    
    # (0, 1): Surface density profiles
    ax = axes[0, 1]
    ax.loglog(R_prof, Sigma_prof, 'b-', lw=2, label='Σ_baryon')
    ax.loglog(R_prof, Sigma_eff_prof, 'r-', lw=2, label='Σ_eff')
    ax.axhline(Sigma_crit, color='green', ls='--', lw=2, label=f'Σ_crit')
    ax.set_xlabel('Projected Radius R [kpc]')
    ax.set_ylabel('Surface Density [Msun/kpc²]')
    ax.set_title('Surface Density Profiles')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # (0, 2): Convergence
    ax = axes[0, 2]
    ax.loglog(R_prof, kappa, 'b-', lw=2, label='κ(R)')
    ax.loglog(R_prof, mean_kappa, 'r-', lw=2, label='<κ>(<R)')
    ax.axhline(1.0, color='k', ls='--', lw=1, label='κ = 1')
    ax.set_xlabel('Projected Radius R [kpc]')
    ax.set_ylabel('Convergence')
    ax.set_title('Convergence Profiles')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # (1, 0): Cumulative mass
    ax = axes[1, 0]
    ax.loglog(R_prof, M_cum, 'b-', lw=2)
    ax.set_xlabel('Projected Radius R [kpc]')
    ax.set_ylabel('Cumulative Mass [Msun]')
    ax.set_title('Enclosed Projected Mass')
    ax.grid(alpha=0.3)
    
    # (1, 1): Boost kernel
    ax = axes[1, 1]
    _, K_prof, _ = azimuthal_average(K_sigma_2d, R_grid_2d, R_bins)
    ax.semilogx(R_prof, K_prof, 'g-', lw=2)
    ax.set_xlabel('Projected Radius R [kpc]')
    ax.set_ylabel('K_σ(R)')
    ax.set_title(f'Boost Kernel (A_c={A_c})')
    ax.grid(alpha=0.3)
    
    # (1, 2): Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    issue_text = ""
    if len(idx_above_1) == 0:
        issue_text = f"""
        ISSUE IDENTIFIED:
        ================
        
        mean_kappa never reaches 1.0!
        
        max(mean_kappa) = {np.max(mean_kappa):.4f}
        
        This means the projected surface
        density is too low to create a
        strong enough gravitational lens.
        
        Possible causes:
        1. Projection loses too much mass
        2. Clumping reduces effective Σ
        3. Critical Σ is too high
        4. Kernel boost is insufficient
        
        Need to increase boost or check
        baryon normalization.
        """
    else:
        issue_text = f"""
        Einstein radius found!
        ======================
        
        R_E = {R_prof[idx_above_1[-1]]:.1f} kpc
        
        mean_kappa(R_E) = {mean_kappa[idx_above_1[-1]]:.4f}
        
        This should translate to
        θ_E ≈ {cosmo.physical_to_angular(R_prof[idx_above_1[-1]], z_lens):.2f}\"
        """
    
    ax.text(0.1, 0.5, issue_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow' if len(idx_above_1)==0 else 'lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('../results/macs0416_sigma_diagnostic.png', dpi=150, bbox_inches='tight')
    print("Diagnostic plot saved: ../results/macs0416_sigma_diagnostic.png")
    print()
    
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    
    return {
        'Sigma_crit': Sigma_crit,
        'max_Sigma': np.max(Sigma_prof),
        'max_Sigma_eff': np.max(Sigma_eff_prof),
        'max_kappa': np.max(kappa),
        'max_mean_kappa': np.max(mean_kappa),
        'R_E_kpc': R_prof[idx_above_1[-1]] if len(idx_above_1) > 0 else 0.0
    }


if __name__ == '__main__':
    results = diagnose_surface_density()
    
    print()
    print("Key Values:")
    print(f"  Sigma_crit = {results['Sigma_crit']:.2e} Msun/kpc^2")
    print(f"  max(Sigma) = {results['max_Sigma']:.2e} Msun/kpc^2")
    print(f"  max(Sigma_eff) = {results['max_Sigma_eff']:.2e} Msun/kpc^2")
    print(f"  max(kappa) = {results['max_kappa']:.6f}")
    print(f"  max(mean_kappa) = {results['max_mean_kappa']:.6f}")
    print(f"  R_E = {results['R_E_kpc']:.1f} kpc")
    print()
    
    # Calculate required boost
    if results['max_mean_kappa'] < 1.0:
        required_boost = 1.0 / results['max_mean_kappa']
        print(f"REQUIRED BOOST to reach mean_kappa = 1:")
        print(f"  Current max(mean_kappa) = {results['max_mean_kappa']:.6f}")
        print(f"  Need boost factor of {required_boost:.2f}x")
        print(f"  --> Try K_sigma ≈ {required_boost - 1:.2f}")
        print(f"  --> Estimate A_c ≈ {50 * (required_boost - 1) / 0.088:.0f}")
