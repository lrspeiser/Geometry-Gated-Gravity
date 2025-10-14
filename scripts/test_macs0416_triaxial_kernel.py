"""
MACS0416 Triaxial Projection + Sigma-Gravity Kernel Test
==========================================================

This script tests the full pipeline:
1. Build 3D baryon density rho(r)
2. Transform to triaxial rho(x,y,z) with axis ratios (q_plane, q_LOS)
3. Project to surface density Sigma_triax(R)
4. Apply 2D Sigma-Gravity kernel: Sigma_eff = Sigma_triax × (1 + K_sigma)
5. Compute Einstein radius and compare to observations

Goal: Validate that triaxial geometry signal is preserved through the kernel
and affects the Einstein radius predictions as expected (~20% sensitivity).

Author: GravityCalculator
Date: 2025-01-14
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Import triaxial lensing
from core.triaxial_lensing import (
    spherical_to_triaxial_density,
    project_triaxial_to_surface_density_simple
)

# Import baryon model
from test_macs0416_projected_kernel import build_macs0416_baryon_profile_3d

# Import Sigma-Gravity kernel
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average

# Import cosmology
from many_path_model.lensing_utilities import LensingCosmology

print("="*70)
print("MACS0416: TRIAXIAL PROJECTION + SIGMA-GRAVITY KERNEL TEST")
print("="*70)

# =============================================================================
# 1. Build Baryon Profile
# =============================================================================
print("\n[1/5] Building MACS0416 baryon profile...")

r_3d = np.logspace(-1, 3.5, 2000)
rho_spherical_array, baryon_info = build_macs0416_baryon_profile_3d(r_3d, verbose=False)

# Create interpolation function for triaxial transform
from scipy.interpolate import interp1d
rho_spherical_func = interp1d(r_3d, rho_spherical_array, kind='linear',
                                bounds_error=False, fill_value=0.0)

print(f"  M_500 = {baryon_info['M_500']:.2e} Msun")
print(f"  R_500 = {baryon_info['R_500']:.1f} kpc")
print(f"  z = {baryon_info['z']:.3f}")
print(f"  M_baryon(R_500) = {baryon_info['M_total']:.2e} Msun")

# =============================================================================
# 2. Test Different Triaxial Configurations
# =============================================================================
print("\n[2/5] Testing triaxial configurations...")

# Configurations to test
configs = [
    {"name": "Spherical", "q_plane": 1.0, "q_LOS": 1.0, "color": "blue"},
    {"name": "Oblate (in-plane)", "q_plane": 0.80, "q_LOS": 1.0, "color": "green"},
    {"name": "Oblate (LOS)", "q_plane": 1.0, "q_LOS": 0.75, "color": "orange"},
    {"name": "Prolate (LOS)", "q_plane": 1.0, "q_LOS": 1.3, "color": "purple"},
    {"name": "Mixed", "q_plane": 0.85, "q_LOS": 1.15, "color": "red"}
]

print(f"  Testing {len(configs)} configurations:")
for cfg in configs:
    print(f"    - {cfg['name']}: q_plane={cfg['q_plane']:.2f}, q_LOS={cfg['q_LOS']:.2f}")

# =============================================================================
# 3. Project to Surface Density
# =============================================================================
print("\n[3/5] Projecting to surface density...")

# Radial grid for projection
R_proj_1d = np.geomspace(1.0, 2000, 200)  # 1D radial profile

results = {}

for cfg in configs:
    name = cfg['name']
    q_plane = cfg['q_plane']
    q_LOS = cfg['q_LOS']
    
    print(f"  Processing {name}...")
    
    # Transform to triaxial
    rho_triax = spherical_to_triaxial_density(
        rho_spherical_func,
        q_plane=q_plane,
        q_LOS=q_LOS,
        normalize_to_mass=baryon_info['M_total'],
        R_norm=baryon_info['R_500']
    )
    
    # Project to surface density
    Sigma_triax_1d = project_triaxial_to_surface_density_simple(
        rho_triax,
        R_proj_1d,
        z_max=5000.0,
        n_z=300
    )
    
    results[name] = {
        'Sigma_triax': Sigma_triax_1d,
        'q_plane': q_plane,
        'q_LOS': q_LOS,
        'color': cfg['color']
    }

# =============================================================================
# 4. Apply Sigma-Gravity Kernel
# =============================================================================
print("\n[4/5] Applying Sigma-Gravity kernel...")

# Optimal kernel parameters from validation
A_c = 16.429
ell0 = 200.0
p = 2.0
ncoh = 2.0

print(f"  Kernel parameters:")
print(f"    A_c = {A_c:.3f}")
print(f"    ℓ_0 = {ell0:.1f} kpc")

# For kernel, need 2D grid
nx, ny = 512, 512
R_max = 2500.0
x = np.linspace(-R_max, R_max, nx)
y = np.linspace(-R_max, R_max, ny)
X, Y = np.meshgrid(x, y)
R_grid_2d = np.sqrt(X**2 + Y**2)

# Cosmology
cosmo = LensingCosmology()
z_lens = baryon_info['z']
z_src = 2.0
Sigma_crit = cosmo.critical_surface_density(z_lens, z_src)

print(f"  Sigma_crit = {Sigma_crit:.2e} Msun/kpc^2")

# Apply kernel to each configuration
for name, data in results.items():
    print(f"  Processing {name} with kernel...")
    
    # Interpolate 1D profile to 2D grid
    Sigma_triax_interp = np.interp(R_grid_2d, R_proj_1d, data['Sigma_triax'])
    
    # Apply kernel
    Sigma_eff_2d, K_sigma_2d, kernel_diag = convolve_sigma_with_kernel(
        Sigma_triax_interp, R_grid_2d, ell0, p, ncoh, A_c,
        emphasize_interior=True, use_fft=True
    )
    
    # Azimuthal average back to 1D
    R_bins = np.linspace(0, 2000, 401)
    R_prof, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff_2d, R_grid_2d, R_bins)
    
    # Remove NaNs
    valid_mask = ~np.isnan(Sigma_eff_prof)
    R_prof_clean = R_prof[valid_mask]
    Sigma_eff_prof_clean = Sigma_eff_prof[valid_mask]
    
    # Store results
    data['Sigma_eff'] = Sigma_eff_prof_clean
    data['R_eff'] = R_prof_clean
    data['K_sigma_2d'] = K_sigma_2d
    data['kernel_diag'] = kernel_diag

# =============================================================================
# 5. Compute Einstein Radii
# =============================================================================
print("\n[5/5] Computing Einstein radii...")

theta_E_obs = 30.0  # arcsec, observed

print(f"\n{'Configuration':<25} {'θ_E (arcsec)':<15} {'Error (%)':<12} {'Sensitivity'}")
print("-" * 70)

baseline_theta_E = None

for name, data in results.items():
    R_prof = data['R_eff']
    Sigma_eff = data['Sigma_eff']
    
    # Cumulative mass
    M_eff_cum = cumulative_trapezoid(2.0 * np.pi * R_prof * Sigma_eff, R_prof, initial=0.0)
    
    # Mean convergence
    kappa_eff = Sigma_eff / Sigma_crit
    mean_kappa_eff = M_eff_cum / (np.pi * R_prof**2 * Sigma_crit)
    mean_kappa_eff[0] = kappa_eff[0]
    
    # Find Einstein radius
    idx_E = np.where(mean_kappa_eff >= 1.0)[0]
    
    if len(idx_E) > 0:
        R_E_kpc = R_prof[idx_E[-1]]
        theta_E = cosmo.physical_to_angular(R_E_kpc, z_lens)
        error_pct = (theta_E - theta_E_obs) / theta_E_obs * 100
        
        data['theta_E'] = theta_E
        data['R_E'] = R_E_kpc
        data['mean_kappa_at_RE'] = mean_kappa_eff[idx_E[-1]]
        
        # Compute sensitivity relative to spherical case
        if name == "Spherical":
            baseline_theta_E = theta_E
            sensitivity = "---"
        else:
            if baseline_theta_E is not None:
                delta_theta = theta_E - baseline_theta_E
                sensitivity_pct = delta_theta / baseline_theta_E * 100
                sensitivity = f"{sensitivity_pct:+.1f}%"
            else:
                sensitivity = "---"
        
        print(f"{name:<25} {theta_E:>7.2f}          {error_pct:>6.1f}%        {sensitivity}")
    else:
        print(f"{name:<25} {'No Einstein radius found'}")
        data['theta_E'] = np.nan
        data['R_E'] = np.nan

print("-" * 70)
print(f"{'Observed':<25} {theta_E_obs:>7.2f}          {'---':>6}       {'---'}")

# =============================================================================
# Generate Plots
# =============================================================================
print("\n[6/6] Generating plots...")

output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'triaxial_kernel_test')
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Surface Density Profiles
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Triaxial Sigma
for name, data in results.items():
    ax1.loglog(R_proj_1d, data['Sigma_triax'], linewidth=2, 
               label=name, color=data['color'])

ax1.set_xlabel('Projected Radius R (kpc)', fontsize=12)
ax1.set_ylabel('Surface Density Σ(R) [M☉/kpc²]', fontsize=12)
ax1.set_title('Triaxial Surface Density', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3, which='both')

# Effective Sigma (after kernel)
for name, data in results.items():
    if 'Sigma_eff' in data:
        ax2.loglog(data['R_eff'], data['Sigma_eff'], linewidth=2,
                   label=name, color=data['color'])

ax2.set_xlabel('Projected Radius R (kpc)', fontsize=12)
ax2.set_ylabel('Effective Surface Density Σ_eff(R) [M☉/kpc²]', fontsize=12)
ax2.set_title('After Sigma-Gravity Kernel', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, which='both')

plt.tight_layout()
plot1_path = os.path.join(output_dir, 'triaxial_surface_density_profiles.png')
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot1_path}")
plt.close()

# Plot 2: Einstein Radius Comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

theta_E_values = []
names_sorted = []
colors_sorted = []
q_LOS_values = []

for name, data in results.items():
    if not np.isnan(data.get('theta_E', np.nan)):
        theta_E_values.append(data['theta_E'])
        names_sorted.append(name)
        colors_sorted.append(data['color'])
        q_LOS_values.append(data['q_LOS'])

# Sort by q_LOS for nice ordering
sort_idx = np.argsort(q_LOS_values)
theta_E_values = [theta_E_values[i] for i in sort_idx]
names_sorted = [names_sorted[i] for i in sort_idx]
colors_sorted = [colors_sorted[i] for i in sort_idx]

x_pos = np.arange(len(names_sorted))
bars = ax.bar(x_pos, theta_E_values, color=colors_sorted, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add observed line
ax.axhline(theta_E_obs, color='red', linestyle='--', linewidth=2, label='Observed (30")')

# Add error band
ax.fill_between([-0.5, len(names_sorted)-0.5], 
                [theta_E_obs*0.95]*2, [theta_E_obs*1.05]*2,
                alpha=0.2, color='gray', label='±5% tolerance')

ax.set_xticks(x_pos)
ax.set_xticklabels(names_sorted, rotation=15, ha='right')
ax.set_ylabel('Einstein Radius θ_E (arcsec)', fontsize=12)
ax.set_title('MACS0416: Einstein Radius vs Triaxial Geometry', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Annotate values
for i, (x, y) in enumerate(zip(x_pos, theta_E_values)):
    ax.text(x, y + 0.5, f'{y:.1f}"', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plot2_path = os.path.join(output_dir, 'triaxial_einstein_radius_comparison.png')
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot2_path}")
plt.close()

# Plot 3: Geometry Sensitivity
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Extract q_LOS and theta_E for plotting
q_LOS_plot = []
theta_E_plot = []
names_plot = []
colors_plot = []

for name, data in results.items():
    if not np.isnan(data.get('theta_E', np.nan)):
        q_LOS_plot.append(data['q_LOS'])
        theta_E_plot.append(data['theta_E'])
        names_plot.append(name)
        colors_plot.append(data['color'])

# Sort by q_LOS
sort_idx = np.argsort(q_LOS_plot)
q_LOS_plot = [q_LOS_plot[i] for i in sort_idx]
theta_E_plot = [theta_E_plot[i] for i in sort_idx]
names_plot = [names_plot[i] for i in sort_idx]
colors_plot = [colors_plot[i] for i in sort_idx]

ax.plot(q_LOS_plot, theta_E_plot, 'o-', linewidth=2, markersize=10, color='blue', label='Model')

# Color code points
for i, (q, theta, name, color) in enumerate(zip(q_LOS_plot, theta_E_plot, names_plot, colors_plot)):
    ax.plot(q, theta, 'o', markersize=12, color=color, zorder=10)
    ax.text(q + 0.02, theta, name, fontsize=9, va='center')

ax.axhline(theta_E_obs, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Observed (30")')
ax.set_xlabel('Line-of-Sight Axis Ratio q_LOS', fontsize=12)
ax.set_ylabel('Einstein Radius θ_E (arcsec)', fontsize=12)
ax.set_title('Geometry Sensitivity: θ_E vs q_LOS', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plot3_path = os.path.join(output_dir, 'triaxial_geometry_sensitivity.png')
plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot3_path}")
plt.close()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nGeometry Signal Preserved:")
if baseline_theta_E is not None:
    # Compute range of theta_E variations
    theta_E_all = [data['theta_E'] for data in results.values() if not np.isnan(data.get('theta_E', np.nan))]
    theta_E_range = max(theta_E_all) - min(theta_E_all)
    sensitivity_pct = theta_E_range / baseline_theta_E * 100
    
    print(f"  Baseline (spherical): θ_E = {baseline_theta_E:.2f}\"")
    print(f"  Range across geometries: Δθ_E = {theta_E_range:.2f}\" ({sensitivity_pct:.1f}%)")
    
    if sensitivity_pct > 5:
        print(f"  ✅ Triaxial geometry signal PRESERVED (>5% variation)")
    else:
        print(f"  ⚠️ Geometry signal weak (<5% variation)")

print("\nBest Match to Observations:")
best_name = None
best_error = np.inf

for name, data in results.items():
    if not np.isnan(data.get('theta_E', np.nan)):
        error = abs(data['theta_E'] - theta_E_obs)
        if error < best_error:
            best_error = error
            best_name = name

if best_name:
    best_data = results[best_name]
    print(f"  Configuration: {best_name}")
    print(f"  q_plane = {best_data['q_plane']:.2f}, q_LOS = {best_data['q_LOS']:.2f}")
    print(f"  θ_E = {best_data['theta_E']:.2f}\" (observed: {theta_E_obs:.2f}\")")
    print(f"  Error = {best_error:.2f}\" ({best_error/theta_E_obs*100:.1f}%)")

print("\n" + "="*70)
print("TRIAXIAL KERNEL TEST COMPLETE")
print("="*70)
