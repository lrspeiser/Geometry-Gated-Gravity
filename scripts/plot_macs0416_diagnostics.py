"""
Diagnostic plotting for MACS0416 Einstein mass validation.

Generates:
1. Radial convergence profiles (baryon vs effective)
2. Boost factor profile (K_sigma vs R)
3. 2D convergence maps (baryon, effective, boost)
4. Cumulative mass curves with Einstein radius marker
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.integrate import cumulative_trapezoid

from test_macs0416_projected_kernel import build_macs0416_baryon_profile_3d, project_to_surface_density
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from many_path_model.lensing_utilities import LensingCosmology

print("="*70)
print("MACS0416 DIAGNOSTIC PLOTS")
print("="*70)

# Setup output directory
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'macs0416_diagnostics')
os.makedirs(output_dir, exist_ok=True)
print(f"\nOutput directory: {output_dir}")

# Build baryon profile
print("\nBuilding baryon profile...")
r_3d = np.logspace(-1, 3.5, 2000)
rho_total, baryon_info = build_macs0416_baryon_profile_3d(r_3d, verbose=False)

# Create 2D grid
nx, ny = 512, 512
R_max = 2500.0
x = np.linspace(-R_max, R_max, nx)
y = np.linspace(-R_max, R_max, ny)
X, Y = np.meshgrid(x, y)
R_grid_2d = np.sqrt(X**2 + Y**2)

# Project to surface density
print("Projecting to surface density...")
Sigma_baryon = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)

# Apply kernel with optimal parameters
A_c = 16.429
ell0 = 200.0
p = 2.0
ncoh = 2.0

print(f"\nApplying kernel (A_c = {A_c:.3f}, ℓ_0 = {ell0:.1f} kpc)...")
Sigma_eff_2d, K_sigma_2d, kernel_diag = convolve_sigma_with_kernel(
    Sigma_baryon, R_grid_2d, ell0, p, ncoh, A_c,
    emphasize_interior=True, use_fft=True
)

# Cosmology
cosmo = LensingCosmology()
z_lens = baryon_info['z']
z_src = 2.0
Sigma_crit = cosmo.critical_surface_density(z_lens, z_src)

# Convergence maps
kappa_bar_2d = Sigma_baryon / Sigma_crit
kappa_eff_2d = Sigma_eff_2d / Sigma_crit

# Azimuthal profiles
print("Computing radial profiles...")
R_bins = np.linspace(0, 2000, 401)
R_prof, Sigma_bar_prof, _ = azimuthal_average(Sigma_baryon, R_grid_2d, R_bins)
_, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff_2d, R_grid_2d, R_bins)
_, K_sigma_prof, K_sigma_std = azimuthal_average(K_sigma_2d, R_grid_2d, R_bins)

# Remove NaNs
valid_mask = ~(np.isnan(Sigma_bar_prof) | np.isnan(Sigma_eff_prof))
R_prof = R_prof[valid_mask]
Sigma_bar_prof = Sigma_bar_prof[valid_mask]
Sigma_eff_prof = Sigma_eff_prof[valid_mask]
K_sigma_prof = K_sigma_prof[valid_mask]
K_sigma_std = K_sigma_std[valid_mask]

# Convergence profiles
kappa_bar_prof = Sigma_bar_prof / Sigma_crit
kappa_eff_prof = Sigma_eff_prof / Sigma_crit

# Boost profile
boost_prof = (1.0 + K_sigma_prof)

# Cumulative mass
M_bar_cum = cumulative_trapezoid(2.0 * np.pi * R_prof * Sigma_bar_prof, R_prof, initial=0.0)
M_eff_cum = cumulative_trapezoid(2.0 * np.pi * R_prof * Sigma_eff_prof, R_prof, initial=0.0)

# Mean convergence
mean_kappa_bar = M_bar_cum / (np.pi * R_prof**2 * Sigma_crit)
mean_kappa_eff = M_eff_cum / (np.pi * R_prof**2 * Sigma_crit)
mean_kappa_bar[0] = kappa_bar_prof[0]
mean_kappa_eff[0] = kappa_eff_prof[0]

# Find Einstein radius
idx_E = np.where(mean_kappa_eff >= 1.0)[0]
if len(idx_E) > 0:
    idx_E_last = idx_E[-1]
    R_E_kpc = R_prof[idx_E_last]
    theta_E_arcsec = cosmo.physical_to_angular(R_E_kpc, z_lens)
    print(f"\nEinstein Radius: R_E = {R_E_kpc:.2f} kpc, θ_E = {theta_E_arcsec:.2f}\"")
else:
    R_E_kpc = None
    theta_E_arcsec = None
    print("\nWarning: No Einstein radius found!")

# ============================================================================
# PLOT 1: Radial Convergence Profiles
# ============================================================================
print("\nGenerating Plot 1: Convergence Profiles...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Point convergence κ(R)
ax1.plot(R_prof, kappa_bar_prof, 'b-', linewidth=2, label='Baryon κ(R)')
ax1.plot(R_prof, kappa_eff_prof, 'r-', linewidth=2, label='Effective κ(R)')
if R_E_kpc:
    ax1.axvline(R_E_kpc, color='gray', linestyle='--', alpha=0.7, label=f'R_E = {R_E_kpc:.1f} kpc')
ax1.axhline(1.0, color='k', linestyle=':', alpha=0.5)
ax1.set_xlabel('Radius (kpc)', fontsize=12)
ax1.set_ylabel('Convergence κ(R)', fontsize=12)
ax1.set_title('Point Convergence Profile', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 1000)
ax1.set_ylim(0, 5)

# Right: Mean convergence <κ>(<R)
ax2.plot(R_prof, mean_kappa_bar, 'b-', linewidth=2, label='Baryon <κ>(<R)')
ax2.plot(R_prof, mean_kappa_eff, 'r-', linewidth=2, label='Effective <κ>(<R)')
ax2.axhline(1.0, color='k', linestyle='-', linewidth=2, alpha=0.7, label='Einstein condition')
if R_E_kpc:
    ax2.axvline(R_E_kpc, color='gray', linestyle='--', alpha=0.7, label=f'R_E = {R_E_kpc:.1f} kpc')
    ax2.plot(R_E_kpc, 1.0, 'ro', markersize=10, zorder=10)
ax2.set_xlabel('Radius (kpc)', fontsize=12)
ax2.set_ylabel('Mean Convergence <κ>(<R)', fontsize=12)
ax2.set_title('Mean Convergence (Einstein Mass)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 500)
ax2.set_ylim(0, 2)

plt.tight_layout()
plot1_path = os.path.join(output_dir, 'convergence_profiles.png')
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot1_path}")
plt.close()

# ============================================================================
# PLOT 2: Boost Factor Profile
# ============================================================================
print("Generating Plot 2: Boost Profile...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: K_sigma kernel
ax1.plot(R_prof, K_sigma_prof, 'purple', linewidth=2, label='K_σ(R)')
ax1.fill_between(R_prof, K_sigma_prof - K_sigma_std, K_sigma_prof + K_sigma_std,
                  alpha=0.3, color='purple', label='±1σ')
ax1.axhline(0, color='k', linestyle=':', alpha=0.5)
ax1.axhline(A_c, color='gray', linestyle='--', alpha=0.7, label=f'A_c = {A_c:.2f}')
if R_E_kpc:
    ax1.axvline(R_E_kpc, color='gray', linestyle='--', alpha=0.7, label=f'R_E = {R_E_kpc:.1f} kpc')
ax1.set_xlabel('Radius (kpc)', fontsize=12)
ax1.set_ylabel('Kernel K_σ(R)', fontsize=12)
ax1.set_title('Sigma-Gravity Kernel', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 1000)

# Right: Boost factor (1 + K_sigma)
ax2.plot(R_prof, boost_prof, 'green', linewidth=2, label='Boost = 1 + K_σ(R)')
ax2.axhline(1.0, color='k', linestyle='-', linewidth=2, alpha=0.7, label='Newtonian (no boost)')
if R_E_kpc:
    ax2.axvline(R_E_kpc, color='gray', linestyle='--', alpha=0.7, label=f'R_E = {R_E_kpc:.1f} kpc')
    boost_at_RE = boost_prof[idx_E_last]
    ax2.plot(R_E_kpc, boost_at_RE, 'go', markersize=10, zorder=10)
    ax2.text(R_E_kpc + 20, boost_at_RE, f'{boost_at_RE:.2f}×', fontsize=11, va='center')
ax2.set_xlabel('Radius (kpc)', fontsize=12)
ax2.set_ylabel('Boost Factor (1 + K_σ)', fontsize=12)
ax2.set_title('Gravitational Boost Profile', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 1000)
ax2.set_ylim(0.8, max(boost_prof[R_prof < 1000]) * 1.1)

plt.tight_layout()
plot2_path = os.path.join(output_dir, 'boost_profile.png')
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot2_path}")
plt.close()

# ============================================================================
# PLOT 3: 2D Convergence Maps
# ============================================================================
print("Generating Plot 3: 2D Convergence Maps...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Extent for images
extent = [-R_max, R_max, -R_max, R_max]

# Map 1: Baryon convergence
im1 = axes[0].imshow(kappa_bar_2d, origin='lower', extent=extent, cmap='Blues',
                      norm=LogNorm(vmin=1e-4, vmax=1.0))
axes[0].set_title('Baryon Convergence κ_bar', fontsize=13, fontweight='bold')
axes[0].set_xlabel('x (kpc)', fontsize=11)
axes[0].set_ylabel('y (kpc)', fontsize=11)
if R_E_kpc:
    circle = plt.Circle((0, 0), R_E_kpc, color='red', fill=False, linewidth=2, linestyle='--')
    axes[0].add_patch(circle)
plt.colorbar(im1, ax=axes[0], label='κ_bar')

# Map 2: Effective convergence
im2 = axes[1].imshow(kappa_eff_2d, origin='lower', extent=extent, cmap='Reds',
                      norm=LogNorm(vmin=1e-4, vmax=10.0))
axes[1].set_title('Effective Convergence κ_eff', fontsize=13, fontweight='bold')
axes[1].set_xlabel('x (kpc)', fontsize=11)
axes[1].set_ylabel('y (kpc)', fontsize=11)
if R_E_kpc:
    circle = plt.Circle((0, 0), R_E_kpc, color='white', fill=False, linewidth=2, linestyle='--')
    axes[1].add_patch(circle)
plt.colorbar(im2, ax=axes[1], label='κ_eff')

# Map 3: Boost factor
boost_2d = 1.0 + K_sigma_2d
im3 = axes[2].imshow(boost_2d, origin='lower', extent=extent, cmap='viridis',
                      vmin=1.0, vmax=np.percentile(boost_2d, 99))
axes[2].set_title('Boost Factor (1 + K_σ)', fontsize=13, fontweight='bold')
axes[2].set_xlabel('x (kpc)', fontsize=11)
axes[2].set_ylabel('y (kpc)', fontsize=11)
if R_E_kpc:
    circle = plt.Circle((0, 0), R_E_kpc, color='white', fill=False, linewidth=2, linestyle='--')
    axes[2].add_patch(circle)
plt.colorbar(im3, ax=axes[2], label='Boost')

plt.tight_layout()
plot3_path = os.path.join(output_dir, 'convergence_maps_2d.png')
plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot3_path}")
plt.close()

# ============================================================================
# PLOT 4: Cumulative Mass Curves
# ============================================================================
print("Generating Plot 4: Cumulative Mass...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Enclosed mass
ax1.plot(R_prof, M_bar_cum / 1e12, 'b-', linewidth=2, label='M_bar(<R)')
ax1.plot(R_prof, M_eff_cum / 1e12, 'r-', linewidth=2, label='M_eff(<R)')
if R_E_kpc:
    ax1.axvline(R_E_kpc, color='gray', linestyle='--', alpha=0.7, label=f'R_E = {R_E_kpc:.1f} kpc')
    M_E = M_eff_cum[idx_E_last] / 1e12
    ax1.plot(R_E_kpc, M_E, 'ro', markersize=10, zorder=10)
    ax1.text(R_E_kpc + 20, M_E, f'M(R_E) = {M_E:.2f}×10¹² M_☉', fontsize=10, va='bottom')
ax1.set_xlabel('Radius (kpc)', fontsize=12)
ax1.set_ylabel('Enclosed Mass (10¹² M_☉)', fontsize=12)
ax1.set_title('Cumulative Mass Profile', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 1000)

# Right: Mass ratio (avoid division by zero at R=0)
mass_ratio = np.divide(M_eff_cum, M_bar_cum, out=np.ones_like(M_eff_cum), where=M_bar_cum>0)
ax2.plot(R_prof, mass_ratio, 'purple', linewidth=2, label='M_eff / M_bar')
ax2.axhline(1.0, color='k', linestyle=':', alpha=0.5)
if R_E_kpc:
    ax2.axvline(R_E_kpc, color='gray', linestyle='--', alpha=0.7, label=f'R_E = {R_E_kpc:.1f} kpc')
    ratio_at_RE = mass_ratio[idx_E_last]
    ax2.plot(R_E_kpc, ratio_at_RE, 'ro', markersize=10, zorder=10)
    ax2.text(R_E_kpc + 20, ratio_at_RE, f'{ratio_at_RE:.2f}×', fontsize=11, va='center')
ax2.set_xlabel('Radius (kpc)', fontsize=12)
ax2.set_ylabel('Mass Ratio M_eff / M_bar', fontsize=12)
ax2.set_title('Cumulative Boost Factor', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 1000)
# Safe ylim calculation with valid data
mask_plot = R_prof < 1000
valid_ratios = mass_ratio[mask_plot & np.isfinite(mass_ratio)]
if len(valid_ratios) > 0:
    ax2.set_ylim(0.9, max(valid_ratios) * 1.1)
else:
    ax2.set_ylim(0.9, 20)

plt.tight_layout()
plot4_path = os.path.join(output_dir, 'cumulative_mass.png')
plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot4_path}")
plt.close()

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

if R_E_kpc:
    print(f"\nEinstein Radius:")
    print(f"  R_E = {R_E_kpc:.2f} kpc")
    print(f"  θ_E = {theta_E_arcsec:.2f}\" (observed: 30.00\")")
    print(f"  Error = {abs(theta_E_arcsec - 30.0):.2f}\" ({abs(theta_E_arcsec - 30.0)/30.0*100:.1f}%)")
    print(f"\nMean Convergence at R_E:")
    print(f"  <κ>(R_E) = {mean_kappa_eff[idx_E_last]:.6f}")
    print(f"  <κ>_bar(R_E) = {mean_kappa_bar[idx_E_last]:.6f}")
    print(f"\nBoost at R_E:")
    print(f"  Boost factor = {boost_prof[idx_E_last]:.3f}×")
    print(f"  K_σ(R_E) = {K_sigma_prof[idx_E_last]:.3f}")
    print(f"\nEnclosed Mass at R_E:")
    print(f"  M_bar(R_E) = {M_bar_cum[idx_E_last]/1e12:.3f} × 10¹² M_☉")
    print(f"  M_eff(R_E) = {M_eff_cum[idx_E_last]/1e12:.3f} × 10¹² M_☉")
    print(f"  Ratio = {mass_ratio[idx_E_last]:.3f}×")

print(f"\nKernel Statistics:")
print(f"  A_c = {A_c:.3f}")
print(f"  ℓ_0 = {ell0:.1f} kpc")
print(f"  <K_σ> = {kernel_diag['K_sigma_mean']:.3f}")
print(f"  max K_σ = {kernel_diag['K_sigma_max']:.3f}")
print(f"  <boost> = {kernel_diag['boost_factor_mean']:.3f}")

print("\n" + "="*70)
print("ALL DIAGNOSTIC PLOTS GENERATED SUCCESSFULLY")
print("="*70)
print(f"\nLocation: {output_dir}")
print("\nFiles:")
print("  1. convergence_profiles.png - κ(R) and <κ>(<R) profiles")
print("  2. boost_profile.png - K_σ(R) and boost factor profiles")
print("  3. convergence_maps_2d.png - 2D maps of κ_bar, κ_eff, boost")
print("  4. cumulative_mass.png - M(<R) and mass ratio profiles")
print("\n" + "="*70)
