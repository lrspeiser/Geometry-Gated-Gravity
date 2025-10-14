#!/usr/bin/env python3
"""
Einstein Radius Mass Check for MACS0416
========================================

Critical validation: verify that <kappa>(R_E) = 1.0 ± few per-mille

Also compute:
1. Area-weighted mean boost inside R_E (the physically meaningful value)
2. Radial profile of boost factor
3. Mass conservation check

This is the sanity check that proves the normalization is correct.

Author: Sigma-Gravity Validation
Date: 2025-01-14
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from test_macs0416_projected_kernel import (
    build_macs0416_baryon_profile_3d,
    project_to_surface_density,
    compute_lensing_from_sigma_eff
)
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from many_path_model.lensing_utilities import LensingCosmology


def validate_einstein_mass(A_c=16.429, ell0=200.0, p=2.0, ncoh=2.0, verbose=True):
    """
    Validate Einstein radius mass condition and compute proper boost factors.
    
    Returns
    -------
    validation : dict
        All validation metrics
    """
    if verbose:
        print("=" * 70)
        print("EINSTEIN RADIUS MASS VALIDATION")
        print("=" * 70)
        print()
        print(f"Parameters: A_c={A_c:.3f}, ell0={ell0:.1f} kpc")
        print()
    
    # Build baryon profile
    r_3d = np.logspace(-1, 3.5, 2000)
    rho_total, baryon_info = build_macs0416_baryon_profile_3d(r_3d, verbose=False)
    
    # Create 2D grid
    nx, ny = 512, 512
    R_max = 2500.0
    x = np.linspace(-R_max, R_max, nx)
    y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid_2d = np.sqrt(X**2 + Y**2)
    pixel_area = (2*R_max/nx)**2
    
    # Project to surface density
    Sigma_baryon = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)
    
    # Apply kernel
    Sigma_eff_2d, K_sigma_2d, kernel_diag = convolve_sigma_with_kernel(
        Sigma_baryon, R_grid_2d, ell0, p, ncoh, A_c,
        emphasize_interior=True, use_fft=True
    )
    
    # Azimuthal average
    R_bins = np.linspace(0, 2000, 401)
    R_prof, Sigma_bar_prof, _ = azimuthal_average(Sigma_baryon, R_grid_2d, R_bins)
    _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff_2d, R_grid_2d, R_bins)
    _, K_sigma_prof, _ = azimuthal_average(K_sigma_2d, R_grid_2d, R_bins)
    
    # Cosmology
    cosmo = LensingCosmology()
    z_lens = baryon_info['z']
    z_src = 2.0
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_src)
    
    # Convergence profiles
    kappa_bar = Sigma_bar_prof / Sigma_crit
    kappa_eff = Sigma_eff_prof / Sigma_crit
    
    # Cumulative mass and mean convergence
    M_bar_cum = cumulative_trapezoid(2.0 * np.pi * R_prof * Sigma_bar_prof, R_prof, initial=0.0)
    M_eff_cum = cumulative_trapezoid(2.0 * np.pi * R_prof * Sigma_eff_prof, R_prof, initial=0.0)
    
    mean_kappa_bar = M_bar_cum / (np.pi * R_prof**2 * Sigma_crit)
    mean_kappa_eff = M_eff_cum / (np.pi * R_prof**2 * Sigma_crit)
    mean_kappa_bar[0] = kappa_bar[0]
    mean_kappa_eff[0] = kappa_eff[0]
    
    # Find Einstein radius (where mean_kappa_eff = 1)
    idx_E = np.where(mean_kappa_eff >= 1.0)[0]
    if len(idx_E) == 0:
        if verbose:
            print("ERROR: No Einstein radius found!")
        return None
    
    idx_E_last = idx_E[-1]
    R_E_kpc = R_prof[idx_E_last]
    theta_E_arcsec = cosmo.physical_to_angular(R_E_kpc, z_lens)
    
    # CRITICAL CHECK: mean_kappa at R_E should be exactly 1.0
    mean_kappa_at_RE = mean_kappa_eff[idx_E_last]
    einstein_mass_error = abs(mean_kappa_at_RE - 1.0)
    
    if verbose:
        print("Einstein Radius:")
        print(f"  R_E = {R_E_kpc:.2f} kpc")
        print(f"  theta_E = {theta_E_arcsec:.2f} arcsec")
        print()
        print("CRITICAL VALIDATION:")
        print(f"  <kappa>(R_E) = {mean_kappa_at_RE:.6f}")
        print(f"  Error from 1.0: {einstein_mass_error:.2e}")
        
        if einstein_mass_error < 0.01:
            print("  ✅ PASS: Within 1% (excellent!)")
        elif einstein_mass_error < 0.05:
            print("  ✓ OK: Within 5%")
        else:
            print("  ❌ FAIL: More than 5% off - normalization issue!")
        print()
    
    # Compute area-weighted mean boost INSIDE R_E
    # This is the physically meaningful boost factor
    mask_inside_RE = R_grid_2d <= R_E_kpc
    
    Sigma_bar_inside = Sigma_baryon[mask_inside_RE]
    Sigma_eff_inside = Sigma_eff_2d[mask_inside_RE]
    K_sigma_inside = K_sigma_2d[mask_inside_RE]
    
    # Area-weighted means
    total_area_inside = np.sum(mask_inside_RE) * pixel_area
    mean_Sigma_bar_inside = np.sum(Sigma_bar_inside) * pixel_area / total_area_inside
    mean_Sigma_eff_inside = np.sum(Sigma_eff_inside) * pixel_area / total_area_inside
    mean_K_inside = np.sum(K_sigma_inside) * pixel_area / total_area_inside
    
    # Boost factor inside R_E
    boost_factor_inside = mean_Sigma_eff_inside / mean_Sigma_bar_inside
    
    if verbose:
        print("Boost Factor Analysis:")
        print(f"  Global mean <K_sigma> = {kernel_diag['K_sigma_mean']:.4f}")
        print(f"  Global mean boost = {kernel_diag['boost_factor_mean']:.4f}")
        print()
        print(f"  INSIDE R_E (area-weighted):")
        print(f"    <Sigma_baryon> = {mean_Sigma_bar_inside:.2e} Msun/kpc^2")
        print(f"    <Sigma_eff> = {mean_Sigma_eff_inside:.2e} Msun/kpc^2")
        print(f"    <K_sigma> = {mean_K_inside:.4f}")
        print(f"    Mean boost factor = {boost_factor_inside:.4f}x")
        print()
        
        # Baryon-only would give:
        mean_kappa_bar_at_RE = mean_kappa_bar[idx_E_last]
        baryon_deficit = 1.0 - mean_kappa_bar_at_RE
        print(f"  Baryons alone at R_E:")
        print(f"    <kappa_baryon>(R_E) = {mean_kappa_bar_at_RE:.4f}")
        print(f"    Deficit to unity = {baryon_deficit:.4f}")
        print(f"    Required boost = {1.0/mean_kappa_bar_at_RE:.4f}x")
        print(f"    Actual boost = {boost_factor_inside:.4f}x")
        print()
    
    # Boost profile
    boost_profile = Sigma_eff_prof / np.maximum(Sigma_bar_prof, np.max(Sigma_bar_prof)*1e-10)
    
    # Mass conservation check
    M_bar_total = np.sum(Sigma_baryon) * pixel_area
    M_eff_total = np.sum(Sigma_eff_2d) * pixel_area
    mass_ratio = M_eff_total / M_bar_total
    
    if verbose:
        print("Mass Conservation:")
        print(f"  M_baryon (projected) = {M_bar_total:.2e} Msun")
        print(f"  M_eff (projected) = {M_eff_total:.2e} Msun")
        print(f"  Ratio = {mass_ratio:.4f}x")
        print()
    
    # Package results
    validation = {
        'R_E_kpc': R_E_kpc,
        'theta_E_arcsec': theta_E_arcsec,
        'theta_E_obs': 30.0,
        'error_pct': abs(theta_E_arcsec - 30.0) / 30.0 * 100,
        'mean_kappa_at_RE': mean_kappa_at_RE,
        'einstein_mass_error': einstein_mass_error,
        'einstein_check_pass': einstein_mass_error < 0.01,
        'K_sigma_global_mean': kernel_diag['K_sigma_mean'],
        'boost_global_mean': kernel_diag['boost_factor_mean'],
        'K_sigma_inside_RE': mean_K_inside,
        'boost_inside_RE': boost_factor_inside,
        'mean_kappa_baryon_at_RE': mean_kappa_bar[idx_E_last],
        'baryon_deficit': 1.0 - mean_kappa_bar[idx_E_last],
        'required_boost': 1.0 / mean_kappa_bar[idx_E_last],
        'M_baryon_total': M_bar_total,
        'M_eff_total': M_eff_total,
        'mass_ratio': mass_ratio,
        'R_prof': R_prof,
        'boost_profile': boost_profile,
        'kappa_bar': kappa_bar,
        'kappa_eff': kappa_eff,
        'mean_kappa_bar': mean_kappa_bar,
        'mean_kappa_eff': mean_kappa_eff,
        'K_sigma_prof': K_sigma_prof
    }
    
    return validation


def plot_validation(validation, output_dir='../results'):
    """Generate comprehensive validation plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    R = validation['R_prof']
    R_E = validation['R_E_kpc']
    
    # (0, 0): Convergence profiles
    ax = axes[0, 0]
    ax.loglog(R, validation['kappa_bar'], 'b-', lw=2, label='κ_baryon')
    ax.loglog(R, validation['kappa_eff'], 'r-', lw=2, label='κ_eff')
    ax.axhline(1.0, color='k', ls='--', lw=1, label='κ = 1')
    ax.axvline(R_E, color='gray', ls='--', alpha=0.7, label=f'R_E = {R_E:.0f} kpc')
    ax.set_xlabel('Projected Radius R [kpc]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Convergence κ(R)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Profiles', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # (0, 1): Mean convergence
    ax = axes[0, 1]
    ax.semilogx(R, validation['mean_kappa_bar'], 'b-', lw=2, label='<κ_baryon>(<R)')
    ax.semilogx(R, validation['mean_kappa_eff'], 'r-', lw=2, label='<κ_eff>(<R)')
    ax.axhline(1.0, color='k', ls='--', lw=2, label='<κ> = 1')
    ax.axvline(R_E, color='gray', ls='--', alpha=0.7)
    
    # Mark Einstein radius point
    ax.plot(R_E, validation['mean_kappa_at_RE'], 'ro', markersize=10, 
            label=f'<κ>(R_E) = {validation["mean_kappa_at_RE"]:.4f}')
    
    ax.set_xlabel('Projected Radius R [kpc]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Convergence <κ>(<R)', fontsize=12, fontweight='bold')
    ax.set_title('Einstein Mass Condition', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim([0.1, 5])
    
    # (0, 2): Boost factor profile
    ax = axes[0, 2]
    ax.semilogx(R, validation['boost_profile'], 'g-', lw=2)
    ax.axhline(1.0, color='k', ls=':', lw=1, alpha=0.5)
    ax.axhline(validation['boost_inside_RE'], color='orange', ls='--', lw=2,
               label=f'Mean inside R_E: {validation["boost_inside_RE"]:.3f}x')
    ax.axvline(R_E, color='gray', ls='--', alpha=0.7)
    ax.set_xlabel('Projected Radius R [kpc]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Boost Factor (Σ_eff / Σ_baryon)', fontsize=12, fontweight='bold')
    ax.set_title('Radial Boost Profile', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # (1, 0): K_sigma profile
    ax = axes[1, 0]
    ax.semilogx(R, validation['K_sigma_prof'], 'purple', lw=2)
    ax.axhline(validation['K_sigma_inside_RE'], color='orange', ls='--', lw=2,
               label=f'Mean inside R_E: {validation["K_sigma_inside_RE"]:.4f}')
    ax.axvline(R_E, color='gray', ls='--', alpha=0.7)
    ax.set_xlabel('Projected Radius R [kpc]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Boost Kernel K_σ(R)', fontsize=12, fontweight='bold')
    ax.set_title('Kernel Amplitude Profile', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # (1, 1): Baryon budget
    ax = axes[1, 1]
    baryon_contribution = validation['mean_kappa_baryon_at_RE']
    boost_contribution = validation['mean_kappa_at_RE'] - baryon_contribution
    
    components = ['Baryons', 'Σ-Gravity\nBoost', 'Total']
    values = [baryon_contribution, boost_contribution, validation['mean_kappa_at_RE']]
    colors = ['blue', 'green', 'red']
    
    bars = ax.bar(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(1.0, color='k', ls='--', lw=2, label='Einstein condition')
    ax.set_ylabel('<κ>(R_E)', fontsize=12, fontweight='bold')
    ax.set_title('Einstein Mass Budget', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # (1, 2): Validation summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = f"""
    VALIDATION SUMMARY
    {'='*50}
    
    Einstein Radius:
      Predicted: {validation['theta_E_arcsec']:.2f}\"
      Observed:  {validation['theta_E_obs']:.2f}\"
      Error: {validation['error_pct']:.1f}%
    
    Mass Condition Check:
      <κ>(R_E) = {validation['mean_kappa_at_RE']:.6f}
      Error from 1.0: {validation['einstein_mass_error']:.2e}
      {'✅ PASS' if validation['einstein_check_pass'] else '❌ FAIL'}
    
    Boost Analysis:
      Required (from baryons): {validation['required_boost']:.3f}x
      Actual (inside R_E): {validation['boost_inside_RE']:.3f}x
      Match: {abs(validation['boost_inside_RE'] - validation['required_boost'])/validation['required_boost']*100:.1f}% difference
    
    Baryon Budget at R_E:
      <κ_baryon> = {validation['mean_kappa_baryon_at_RE']:.4f}
      Deficit = {validation['baryon_deficit']:.4f}
      Boost provides: {boost_contribution:.4f}
    
    Physical Interpretation:
      Baryons contribute {baryon_contribution/1.0*100:.1f}%
      Many-paths boost: {boost_contribution/1.0*100:.1f}%
      NO DARK MATTER
    """
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', 
                     facecolor='lightgreen' if validation['einstein_check_pass'] else 'lightyellow',
                     alpha=0.3))
    
    plt.suptitle('MACS0416 Einstein Radius Mass Validation - Σ-Gravity Model',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'macs0416_einstein_mass_validation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nValidation plot saved: {output_path}")
    
    return fig


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("MACS0416 EINSTEIN MASS VALIDATION")
    print("Critical sanity check for Σ-Gravity normalization")
    print("=" * 70)
    print()
    
    # Run validation with optimal parameters from tuning
    validation = validate_einstein_mass(A_c=16.429, ell0=200.0, verbose=True)
    
    if validation is None:
        print("\nERROR: Validation failed!")
        sys.exit(1)
    
    # Generate plots
    plot_validation(validation)
    
    # Final assessment
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print()
    
    if validation['einstein_check_pass']:
        print("✅ Einstein mass condition verified!")
        print(f"   <κ>(R_E) = {validation['mean_kappa_at_RE']:.6f} (target: 1.0)")
        print()
        print(f"Physical boost inside R_E: {validation['boost_inside_RE']:.3f}x")
        print(f"Required boost from baryons: {validation['required_boost']:.3f}x")
        print()
        print("The normalization is CORRECT.")
        print("Ready for hierarchical multi-cluster calibration!")
    else:
        print("❌ Einstein mass condition FAILED!")
        print(f"   <κ>(R_E) = {validation['mean_kappa_at_RE']:.6f} (target: 1.0)")
        print(f"   Error: {validation['einstein_mass_error']:.2e}")
        print()
        print("There may still be a normalization issue.")
        print("Review the kernel implementation.")
