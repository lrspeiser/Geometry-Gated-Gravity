"""
Parameter Sensitivity Study: Coherence Amplitude A_c

Systematically vary A_c while holding other parameters fixed to determine:
1. Sensitivity of Einstein radius to A_c
2. Range of A_c that produces physically reasonable results
3. Gradient dθ_E/dA_c for uncertainty propagation

This provides the foundation for hierarchical calibration across the 12-cluster catalog.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from test_macs0416_projected_kernel import build_macs0416_baryon_profile_3d, project_to_surface_density
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from many_path_model.lensing_utilities import LensingCosmology

print("="*70)
print("PARAMETER SENSITIVITY STUDY: A_c (Coherence Amplitude)")
print("="*70)

# Setup output
output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'parameter_sensitivity')
os.makedirs(output_dir, exist_ok=True)
print(f"\nOutput directory: {output_dir}")

# Build baryon profile (once)
print("\nBuilding MACS0416 baryon profile...")
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

# Cosmology
cosmo = LensingCosmology()
z_lens = baryon_info['z']
z_src = 2.0
Sigma_crit = cosmo.critical_surface_density(z_lens, z_src)

# Fixed parameters
ell0 = 200.0
p = 2.0
ncoh = 2.0
print(f"\nFixed parameters: ℓ_0 = {ell0:.1f} kpc, p = {p:.1f}, n_coh = {ncoh:.1f}")

# A_c range to explore
# Based on optimal A_c = 16.429, explore ±50% range
A_c_optimal = 16.429
A_c_values = np.linspace(10, 25, 31)  # Fine sampling around optimal
print(f"\nTesting A_c range: [{A_c_values[0]:.1f}, {A_c_values[-1]:.1f}], N = {len(A_c_values)}")
print(f"Optimal value: A_c = {A_c_optimal:.3f}")

# Storage for results
results = {
    'A_c': [],
    'theta_E': [],
    'R_E': [],
    'mean_kappa_at_RE': [],
    'kappa_max': [],
    'boost_max': [],
    'boost_at_RE': [],
    'M_eff_at_RE': []
}

print("\nRunning parameter sweep...")
R_bins = np.linspace(0, 2000, 401)

for i, A_c in enumerate(A_c_values):
    if (i+1) % 5 == 0:
        print(f"  Progress: {i+1}/{len(A_c_values)} (A_c = {A_c:.2f})")
    
    # Apply kernel
    Sigma_eff_2d, K_sigma_2d, kernel_diag = convolve_sigma_with_kernel(
        Sigma_baryon, R_grid_2d, ell0, p, ncoh, A_c,
        emphasize_interior=True, use_fft=True
    )
    
    # Convergence
    kappa_eff_2d = Sigma_eff_2d / Sigma_crit
    
    # Azimuthal average
    R_prof, Sigma_bar_prof, _ = azimuthal_average(Sigma_baryon, R_grid_2d, R_bins)
    _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff_2d, R_grid_2d, R_bins)
    
    # Remove NaNs
    valid_mask = ~(np.isnan(Sigma_bar_prof) | np.isnan(Sigma_eff_prof))
    R_prof_clean = R_prof[valid_mask]
    Sigma_bar_prof_clean = Sigma_bar_prof[valid_mask]
    Sigma_eff_prof_clean = Sigma_eff_prof[valid_mask]
    
    # Cumulative mass and mean convergence
    M_bar_cum = cumulative_trapezoid(2.0 * np.pi * R_prof_clean * Sigma_bar_prof_clean, 
                                      R_prof_clean, initial=0.0)
    M_eff_cum = cumulative_trapezoid(2.0 * np.pi * R_prof_clean * Sigma_eff_prof_clean, 
                                      R_prof_clean, initial=0.0)
    
    kappa_bar_prof = Sigma_bar_prof_clean / Sigma_crit
    kappa_eff_prof = Sigma_eff_prof_clean / Sigma_crit
    
    mean_kappa_eff = M_eff_cum / (np.pi * R_prof_clean**2 * Sigma_crit)
    mean_kappa_eff[0] = kappa_eff_prof[0]
    
    # Find Einstein radius
    idx_E = np.where(mean_kappa_eff >= 1.0)[0]
    
    if len(idx_E) > 0:
        idx_E_last = idx_E[-1]
        R_E_kpc = R_prof_clean[idx_E_last]
        theta_E_arcsec = cosmo.physical_to_angular(R_E_kpc, z_lens)
        mean_kappa_at_RE = mean_kappa_eff[idx_E_last]
        
        # Boost at R_E (area-weighted inside R_E)
        mask_inside_RE = R_grid_2d <= R_E_kpc
        Sigma_bar_inside = Sigma_baryon[mask_inside_RE]
        Sigma_eff_inside = Sigma_eff_2d[mask_inside_RE]
        boost_at_RE = np.mean(Sigma_eff_inside) / np.mean(Sigma_bar_inside)
        
        M_eff_at_RE = M_eff_cum[idx_E_last]
    else:
        # No Einstein radius found
        R_E_kpc = np.nan
        theta_E_arcsec = np.nan
        mean_kappa_at_RE = np.nan
        boost_at_RE = np.nan
        M_eff_at_RE = np.nan
    
    # Store results
    results['A_c'].append(A_c)
    results['theta_E'].append(theta_E_arcsec)
    results['R_E'].append(R_E_kpc)
    results['mean_kappa_at_RE'].append(mean_kappa_at_RE)
    results['kappa_max'].append(np.max(kappa_eff_2d))
    results['boost_max'].append(1.0 + np.max(K_sigma_2d))
    results['boost_at_RE'].append(boost_at_RE)
    results['M_eff_at_RE'].append(M_eff_at_RE)

# Convert to arrays
for key in results:
    results[key] = np.array(results[key])

print("\nParameter sweep complete!")

# ============================================================================
# Analysis
# ============================================================================
print("\n" + "="*70)
print("SENSITIVITY ANALYSIS")
print("="*70)

# Find valid Einstein radius entries
valid_mask = np.isfinite(results['theta_E'])
n_valid = np.sum(valid_mask)
print(f"\nEinstein radius found for {n_valid}/{len(A_c_values)} parameter values")

if n_valid > 0:
    A_c_valid = results['A_c'][valid_mask]
    theta_E_valid = results['theta_E'][valid_mask]
    
    # Gradient at optimal point
    idx_opt = np.argmin(np.abs(results['A_c'] - A_c_optimal))
    if idx_opt > 0 and idx_opt < len(A_c_values) - 1:
        dtheta_dAc = (results['theta_E'][idx_opt+1] - results['theta_E'][idx_opt-1]) / \
                      (results['A_c'][idx_opt+1] - results['A_c'][idx_opt-1])
        print(f"\nGradient at optimal A_c = {A_c_optimal:.3f}:")
        print(f"  dθ_E/dA_c ≈ {dtheta_dAc:.4f} arcsec per unit A_c")
        print(f"  1% change in A_c → {abs(dtheta_dAc * A_c_optimal * 0.01):.3f}\" change in θ_E")
    
    # Observed Einstein radius
    theta_E_obs = 30.0
    
    # Find A_c that gives exact match
    residuals = np.abs(theta_E_valid - theta_E_obs)
    idx_best = np.argmin(residuals)
    A_c_best = A_c_valid[idx_best]
    theta_E_best = theta_E_valid[idx_best]
    error_best = theta_E_best - theta_E_obs
    
    print(f"\nBest-fit A_c:")
    print(f"  A_c = {A_c_best:.3f}")
    print(f"  θ_E = {theta_E_best:.2f}\" (observed: {theta_E_obs:.2f}\")")
    print(f"  Error = {error_best:.2f}\" ({abs(error_best)/theta_E_obs*100:.2f}%)")
    
    # Acceptable range (within 5% error)
    tolerance = 0.05 * theta_E_obs
    acceptable_mask = np.abs(theta_E_valid - theta_E_obs) <= tolerance
    if np.sum(acceptable_mask) > 0:
        A_c_acceptable = A_c_valid[acceptable_mask]
        print(f"\nAcceptable A_c range (θ_E error < 5%):")
        print(f"  A_c ∈ [{np.min(A_c_acceptable):.3f}, {np.max(A_c_acceptable):.3f}]")
        print(f"  Width: ΔA_c = {np.max(A_c_acceptable) - np.min(A_c_acceptable):.3f}")

# ============================================================================
# PLOT 1: Einstein Radius vs A_c
# ============================================================================
print("\nGenerating plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: θ_E(A_c)
ax = axes[0, 0]
ax.plot(results['A_c'], results['theta_E'], 'b-', linewidth=2, label='θ_E(A_c)')
ax.axhline(30.0, color='k', linestyle='--', linewidth=2, alpha=0.7, label='Observed (30")')
ax.axhline(30.0*1.05, color='gray', linestyle=':', alpha=0.5)
ax.axhline(30.0*0.95, color='gray', linestyle=':', alpha=0.5, label='±5% tolerance')
ax.axvline(A_c_optimal, color='red', linestyle='--', alpha=0.7, label=f'Optimal A_c = {A_c_optimal:.2f}')
if n_valid > 0 and idx_best is not None:
    ax.plot(A_c_best, theta_E_best, 'ro', markersize=10, zorder=10)
ax.set_xlabel('Coherence Amplitude A_c', fontsize=12)
ax.set_ylabel('Einstein Radius θ_E (arcsec)', fontsize=12)
ax.set_title('Einstein Radius Sensitivity', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 2: Mean κ at R_E
ax = axes[0, 1]
ax.plot(results['A_c'], results['mean_kappa_at_RE'], 'purple', linewidth=2, label='<κ>(R_E)')
ax.axhline(1.0, color='k', linestyle='--', linewidth=2, alpha=0.7, label='Einstein condition')
ax.axhline(1.05, color='gray', linestyle=':', alpha=0.5)
ax.axhline(0.95, color='gray', linestyle=':', alpha=0.5, label='±5% tolerance')
ax.axvline(A_c_optimal, color='red', linestyle='--', alpha=0.7, label=f'Optimal A_c')
ax.set_xlabel('Coherence Amplitude A_c', fontsize=12)
ax.set_ylabel('Mean Convergence <κ>(R_E)', fontsize=12)
ax.set_title('Einstein Mass Condition', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 3: Boost at R_E
ax = axes[1, 0]
ax.plot(results['A_c'], results['boost_at_RE'], 'green', linewidth=2, label='Boost at R_E')
ax.axhline(1.0, color='k', linestyle=':', alpha=0.5)
ax.axvline(A_c_optimal, color='red', linestyle='--', alpha=0.7, label=f'Optimal A_c')
ax.set_xlabel('Coherence Amplitude A_c', fontsize=12)
ax.set_ylabel('Boost Factor (Σ_eff / Σ_bar)', fontsize=12)
ax.set_title('Boost Inside Einstein Radius', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel 4: Max convergence
ax = axes[1, 1]
ax.plot(results['A_c'], results['kappa_max'], 'orange', linewidth=2, label='max κ_eff')
ax.axhline(1.0, color='k', linestyle=':', alpha=0.5)
ax.axvline(A_c_optimal, color='red', linestyle='--', alpha=0.7, label=f'Optimal A_c')
ax.set_xlabel('Coherence Amplitude A_c', fontsize=12)
ax.set_ylabel('Maximum Convergence κ_max', fontsize=12)
ax.set_title('Peak Convergence', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(output_dir, 'sensitivity_Ac_all_panels.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path}")
plt.close()

# ============================================================================
# PLOT 2: Detailed Einstein Radius Zoom
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(results['A_c'], results['theta_E'], 'b-', linewidth=3, label='Model θ_E(A_c)')
ax.axhline(30.0, color='k', linestyle='-', linewidth=2, alpha=0.7, label='Observed: 30"')
ax.fill_between([10, 25], [30.0*0.95]*2, [30.0*1.05]*2, 
                 alpha=0.2, color='gray', label='±5% tolerance')
ax.axvline(A_c_optimal, color='red', linestyle='--', linewidth=2, alpha=0.7, 
           label=f'Optimal: A_c = {A_c_optimal:.3f}')

if n_valid > 0 and idx_best is not None:
    ax.plot(A_c_best, theta_E_best, 'ro', markersize=12, zorder=10,
            label=f'Best fit: {theta_E_best:.2f}" at A_c={A_c_best:.3f}')

ax.set_xlabel('Coherence Amplitude A_c', fontsize=14)
ax.set_ylabel('Einstein Radius θ_E (arcsec)', fontsize=14)
ax.set_title('MACS0416: Einstein Radius Sensitivity to A_c', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(10, 25)
ax.set_ylim(20, 40)

plt.tight_layout()
plot_path2 = os.path.join(output_dir, 'sensitivity_Ac_zoom.png')
plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot_path2}")
plt.close()

# ============================================================================
# Save results to file
# ============================================================================
results_file = os.path.join(output_dir, 'sensitivity_Ac_results.txt')
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("PARAMETER SENSITIVITY STUDY: A_c (Coherence Amplitude)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Cluster: MACS0416-2403\n")
    f.write(f"Fixed parameters: ℓ_0 = {ell0:.1f} kpc, p = {p:.1f}, n_coh = {ncoh:.1f}\n")
    f.write(f"A_c range: [{A_c_values[0]:.1f}, {A_c_values[-1]:.1f}], N = {len(A_c_values)}\n\n")
    
    f.write("Results Table:\n")
    f.write("-"*70 + "\n")
    f.write(f"{'A_c':>6s} {'θ_E':>8s} {'R_E':>8s} {'<κ>(R_E)':>10s} {'Boost@R_E':>10s} {'κ_max':>8s}\n")
    f.write(f"{'':>6s} {'(arcsec)':>8s} {'(kpc)':>8s} {'':>10s} {'':>10s} {'':>8s}\n")
    f.write("-"*70 + "\n")
    
    for i in range(len(results['A_c'])):
        f.write(f"{results['A_c'][i]:6.2f} ")
        if np.isfinite(results['theta_E'][i]):
            f.write(f"{results['theta_E'][i]:8.2f} ")
            f.write(f"{results['R_E'][i]:8.1f} ")
            f.write(f"{results['mean_kappa_at_RE'][i]:10.4f} ")
            f.write(f"{results['boost_at_RE'][i]:10.3f} ")
            f.write(f"{results['kappa_max'][i]:8.2f}\n")
        else:
            f.write(f"{'---':>8s} {'---':>8s} {'---':>10s} {'---':>10s} {'---':>8s}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("SUMMARY\n")
    f.write("="*70 + "\n\n")
    
    if n_valid > 0:
        f.write(f"Valid Einstein radii: {n_valid}/{len(A_c_values)}\n\n")
        f.write(f"Best-fit A_c: {A_c_best:.3f}\n")
        f.write(f"  θ_E = {theta_E_best:.2f}\" (observed: 30.00\")\n")
        f.write(f"  Error = {error_best:.2f}\" ({abs(error_best)/30.0*100:.2f}%)\n\n")
        
        if idx_opt > 0 and idx_opt < len(A_c_values) - 1:
            f.write(f"Gradient at optimal A_c = {A_c_optimal:.3f}:\n")
            f.write(f"  dθ_E/dA_c ≈ {dtheta_dAc:.4f} arcsec per unit A_c\n")
            f.write(f"  1% change in A_c → {abs(dtheta_dAc * A_c_optimal * 0.01):.3f}\" change in θ_E\n\n")
        
        if np.sum(acceptable_mask) > 0:
            f.write(f"Acceptable A_c range (θ_E error < 5%):\n")
            f.write(f"  A_c ∈ [{np.min(A_c_acceptable):.3f}, {np.max(A_c_acceptable):.3f}]\n")
            f.write(f"  Width: ΔA_c = {np.max(A_c_acceptable) - np.min(A_c_acceptable):.3f}\n")

print(f"  Saved: {results_file}")

print("\n" + "="*70)
print("SENSITIVITY STUDY COMPLETE")
print("="*70)
