import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.integrate import cumulative_trapezoid
from test_macs0416_projected_kernel import build_macs0416_baryon_profile_3d, project_to_surface_density
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from many_path_model.lensing_utilities import LensingCosmology

print("="*70)
print("SIMPLE EINSTEIN MASS CHECK - MACS0416")
print("="*70)

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
pixel_area = (2*R_max/nx)**2

print(f"Grid: {nx}x{ny}, R_max={R_max} kpc, pixel_area={pixel_area:.2f} kpc^2")

# Project to surface density
print("\nProjecting to surface density...")
Sigma_baryon = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)
print(f"Max Sigma_baryon = {np.max(Sigma_baryon):.2e} Msun/kpc^2")

# Apply kernel with optimal A_c
A_c = 16.429
print(f"\nApplying kernel with A_c = {A_c}...")
Sigma_eff_2d, K_sigma_2d, kernel_diag = convolve_sigma_with_kernel(
    Sigma_baryon, R_grid_2d, 200.0, 2.0, 2.0, A_c,
    emphasize_interior=True, use_fft=True
)

print(f"Max K_sigma = {np.max(K_sigma_2d):.4f}")
print(f"Max Sigma_eff = {np.max(Sigma_eff_2d):.2e} Msun/kpc^2")
print(f"Mean boost (global) = {kernel_diag['boost_factor_mean']:.4f}")

# Cosmology
cosmo = LensingCosmology()
z_lens = baryon_info['z']
z_src = 2.0
Sigma_crit = cosmo.critical_surface_density(z_lens, z_src)
print(f"\nSigma_crit = {Sigma_crit:.2e} Msun/kpc^2")

# Check 2D convergence directly
kappa_eff_2d = Sigma_eff_2d / Sigma_crit
print(f"Max kappa_eff (2D) = {np.max(kappa_eff_2d):.4f}")
print(f"Center kappa_eff = {kappa_eff_2d[ny//2, nx//2]:.4f}")

# Check for NaNs before averaging
print(f"\nChecking for NaNs BEFORE averaging:")
print(f"  NaNs in Sigma_baryon: {np.sum(np.isnan(Sigma_baryon))}")
print(f"  NaNs in Sigma_eff_2d: {np.sum(np.isnan(Sigma_eff_2d))}")
print(f"  NaNs in R_grid_2d: {np.sum(np.isnan(R_grid_2d))}")

# Azimuthal average
print("\nAzimuthal averaging...")
R_bins = np.linspace(0, 2000, 401)
R_prof, Sigma_bar_prof, _ = azimuthal_average(Sigma_baryon, R_grid_2d, R_bins)
_, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff_2d, R_grid_2d, R_bins)

print(f"\nChecking for NaNs AFTER averaging:")
print(f"  NaNs in Sigma_bar_prof: {np.sum(np.isnan(Sigma_bar_prof))}")
print(f"  NaNs in Sigma_eff_prof: {np.sum(np.isnan(Sigma_eff_prof))}")

print(f"R_prof: {len(R_prof)} points from {R_prof[0]:.2f} to {R_prof[-1]:.2f} kpc")

# Remove NaN values from profiles before processing
valid_mask = ~(np.isnan(Sigma_bar_prof) | np.isnan(Sigma_eff_prof))
R_prof = R_prof[valid_mask]
Sigma_bar_prof = Sigma_bar_prof[valid_mask]
Sigma_eff_prof = Sigma_eff_prof[valid_mask]

print(f"Valid profile points: {len(R_prof)}")
print(f"Max Sigma_eff_prof = {np.max(Sigma_eff_prof):.2e}")

# Convergence profiles
kappa_bar_prof = Sigma_bar_prof / Sigma_crit
kappa_eff_prof = Sigma_eff_prof / Sigma_crit

print(f"Max kappa_eff_prof = {np.max(kappa_eff_prof):.4f}")

# Cumulative mass and mean convergence
print("\nComputing cumulative mass...")
M_bar_cum = cumulative_trapezoid(2.0 * np.pi * R_prof * Sigma_bar_prof, R_prof, initial=0.0)
M_eff_cum = cumulative_trapezoid(2.0 * np.pi * R_prof * Sigma_eff_prof, R_prof, initial=0.0)

mean_kappa_bar = M_bar_cum / (np.pi * R_prof**2 * Sigma_crit)
mean_kappa_eff = M_eff_cum / (np.pi * R_prof**2 * Sigma_crit)
mean_kappa_bar[0] = kappa_bar_prof[0]
mean_kappa_eff[0] = kappa_eff_prof[0]

print(f"Max mean_kappa_bar = {np.max(mean_kappa_bar):.4f}")
print(f"Max mean_kappa_eff = {np.max(mean_kappa_eff):.4f}")

# Find Einstein radius
idx_E = np.where(mean_kappa_eff >= 1.0)[0]

print(f"\n{'='*70}")
print("EINSTEIN RADIUS RESULTS")
print(f"{'='*70}")

if len(idx_E) == 0:
    print("\nERROR: No Einstein radius found!")
    print("\nChecking mean_kappa at key radii:")
    for r in [50, 100, 150, 200, 250, 300, 500]:
        idx = np.argmin(np.abs(R_prof - r))
        print(f"  R={r:3d} kpc: mean_kappa_bar={mean_kappa_bar[idx]:.4f}, mean_kappa_eff={mean_kappa_eff[idx]:.4f}")
else:
    idx_E_last = idx_E[-1]
    R_E_kpc = R_prof[idx_E_last]
    theta_E_arcsec = cosmo.physical_to_angular(R_E_kpc, z_lens)
    mean_kappa_at_RE = mean_kappa_eff[idx_E_last]
    
    print(f"\nEinstein Radius FOUND!")
    print(f"  R_E = {R_E_kpc:.2f} kpc")
    print(f"  theta_E = {theta_E_arcsec:.2f} arcsec (observed: 30.00)")
    print(f"  Error = {abs(theta_E_arcsec - 30.0):.2f} arcsec ({abs(theta_E_arcsec - 30.0)/30.0*100:.1f}%)")
    print(f"\nCRITICAL CHECK:")
    print(f"  <kappa>(R_E) = {mean_kappa_at_RE:.6f}")
    print(f"  Error from 1.0: {abs(mean_kappa_at_RE - 1.0):.2e}")
    
    if abs(mean_kappa_at_RE - 1.0) < 0.01:
        print(f"  Status: PASS (within 1%)")
    elif abs(mean_kappa_at_RE - 1.0) < 0.05:
        print(f"  Status: OK (within 5%)")
    else:
        print(f"  Status: FAIL (more than 5% off)")
    
    # Compute area-weighted boost inside R_E
    mask_inside_RE = R_grid_2d <= R_E_kpc
    Sigma_bar_inside = Sigma_baryon[mask_inside_RE]
    Sigma_eff_inside = Sigma_eff_2d[mask_inside_RE]
    
    mean_Sigma_bar_inside = np.mean(Sigma_bar_inside)
    mean_Sigma_eff_inside = np.mean(Sigma_eff_inside)
    boost_inside = mean_Sigma_eff_inside / mean_Sigma_bar_inside
    
    print(f"\nBOOST INSIDE R_E (area-weighted):")
    print(f"  Boost factor = {boost_inside:.3f}x")
    print(f"  Baryon <kappa> at R_E = {mean_kappa_bar[idx_E_last]:.4f}")
    print(f"  Required boost = {1.0/mean_kappa_bar[idx_E_last]:.3f}x")
    print(f"  Match = {abs(boost_inside - 1.0/mean_kappa_bar[idx_E_last])//(1.0/mean_kappa_bar[idx_E_last])*100:.1f}% difference")

print(f"\n{'='*70}")
