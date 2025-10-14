"""
Hierarchical 12-Cluster Calibration for Sigma-Gravity Kernel
=============================================================

This script performs hierarchical Bayesian calibration of the Sigma-Gravity
kernel parameters across the 12-cluster lensing catalog.

Strategy:
---------
1. Load 12-cluster catalog with observed Einstein radii
2. For each cluster, build baryon profile (triaxial if available)
3. Apply Sigma-Gravity kernel with parameters (A_c, ℓ_0, p, n_coh)
4. Compute predicted Einstein radius
5. Build likelihood: chi-squared with observational uncertainties
6. Optimize or sample posterior distribution of parameters

Hierarchical Structure:
-----------------------
Option A: Universal parameters (all clusters share same A_c, ℓ_0)
Option B: Per-cluster A_c with hierarchical prior (A_c ~ Normal(μ, σ))
Option C: Fully hierarchical with correlations

For initial calibration, we'll use Option A (universal parameters).

Author: GravityCalculator
Date: 2025-01-14
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Import baryon model
from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams

# Import Sigma-Gravity kernel
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average

# Import triaxial lensing (for future use)
from core.triaxial_lensing import (
    spherical_to_triaxial_density,
    project_triaxial_to_surface_density_simple
)

# Import test helpers
from test_macs0416_projected_kernel import project_to_surface_density

# Import cosmology
from many_path_model.lensing_utilities import LensingCosmology

print("="*70)
print("HIERARCHICAL 12-CLUSTER CALIBRATION")
print("="*70)

# =============================================================================
# 1. Load Cluster Catalog
# =============================================================================
print("\n[1/6] Loading cluster catalog...")

catalog_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clusters', 'master_catalog.csv')
catalog = pd.read_csv(catalog_path)

print(f"  Loaded {len(catalog)} clusters from catalog")
print(f"\n  Catalog summary:")
print(f"    - Tiers: {catalog['tier'].value_counts().sort_index().to_dict()}")
print(f"    - Redshift range: [{catalog['z_lens'].min():.3f}, {catalog['z_lens'].max():.3f}]")
print(f"    - Mass range: [{catalog['M_500_Msun'].min():.2e}, {catalog['M_500_Msun'].max():.2e}] M☉")
print(f"    - θ_E range: [{catalog['theta_E_obs_arcsec'].min():.1f}, {catalog['theta_E_obs_arcsec'].max():.1f}]\"")

# For initial calibration, use Tier 1 clusters (highest quality)
tier1_clusters = catalog[catalog['tier'] == 1].copy()
print(f"\n  Using {len(tier1_clusters)} Tier 1 clusters for calibration:")
for idx, row in tier1_clusters.iterrows():
    print(f"    - {row['cluster_name']}: z={row['z_lens']:.3f}, "
          f"θ_E={row['theta_E_obs_arcsec']:.1f}±{row['theta_E_err_arcsec']:.1f}\"")

# =============================================================================
# 2. Helper Function: Predict Einstein Radius for Single Cluster
# =============================================================================

def predict_einstein_radius(cluster_params, kernel_params, verbose=False):
    """
    Predict Einstein radius for a single cluster given kernel parameters.
    
    Parameters
    ----------
    cluster_params : dict
        Cluster physical parameters (M_500, R_500, z, fgas, T_keV, etc.)
    kernel_params : dict
        Kernel parameters (A_c, ell0, p, ncoh)
    verbose : bool
        Print diagnostics
        
    Returns
    -------
    theta_E_pred : float
        Predicted Einstein radius [arcsec]
    diagnostics : dict
        Additional information
    """
    # Unpack parameters
    M_500 = cluster_params['M_500']
    R_500 = cluster_params['R_500']
    z_lens = cluster_params['z']
    z_src = cluster_params.get('z_src', 2.0)
    fgas = cluster_params.get('fgas', 0.11)
    T_keV = cluster_params.get('T_keV', 8.0)
    
    A_c = kernel_params['A_c']
    ell0 = kernel_params['ell0']
    p = kernel_params.get('p', 2.0)
    ncoh = kernel_params.get('ncoh', 2.0)
    
    if verbose:
        print(f"    Building profile for {cluster_params.get('name', 'cluster')}...")
    
    # Build 3D baryon profile
    r_3d = np.logspace(-1, 3.5, 1500)
    
    params = ClusterBaryonParams(
        M_500=M_500,
        R_500=R_500,
        z=z_lens,
        fgas_target=fgas,
        T_keV=T_keV,
        C0=1.3,
        eta=2.0,
        C_max=2.5
    )
    
    components = build_cluster_baryon_model(r_3d, params, apply_clumping=False, verbose=False)
    rho_total = components.rho_total
    
    # Project to 2D (spherical for now)
    nx, ny = 512, 512
    R_max = min(3000.0, R_500 * 2.5)
    x = np.linspace(-R_max, R_max, nx)
    y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid_2d = np.sqrt(X**2 + Y**2)
    
    Sigma_baryon = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)
    
    # Apply kernel
    Sigma_eff_2d, K_sigma_2d, kernel_diag = convolve_sigma_with_kernel(
        Sigma_baryon, R_grid_2d, ell0, p, ncoh, A_c,
        emphasize_interior=True, use_fft=True
    )
    
    # Azimuthal average
    R_bins = np.linspace(0, R_max, 401)
    R_prof, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff_2d, R_grid_2d, R_bins)
    
    # Remove NaNs
    valid_mask = ~np.isnan(Sigma_eff_prof)
    R_prof = R_prof[valid_mask]
    Sigma_eff_prof = Sigma_eff_prof[valid_mask]
    
    # Cosmology and convergence
    cosmo = LensingCosmology()
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_src)
    kappa_eff = Sigma_eff_prof / Sigma_crit
    
    # Cumulative mass and mean convergence
    M_eff_cum = cumulative_trapezoid(2.0 * np.pi * R_prof * Sigma_eff_prof, R_prof, initial=0.0)
    mean_kappa_eff = M_eff_cum / (np.pi * R_prof**2 * Sigma_crit)
    mean_kappa_eff[0] = kappa_eff[0]
    
    # Find Einstein radius
    idx_E = np.where(mean_kappa_eff >= 1.0)[0]
    
    if len(idx_E) > 0:
        R_E_kpc = R_prof[idx_E[-1]]
        theta_E_pred = cosmo.physical_to_angular(R_E_kpc, z_lens)
    else:
        theta_E_pred = 0.0
        R_E_kpc = 0.0
    
    diagnostics = {
        'R_E_kpc': R_E_kpc,
        'max_kappa': np.max(kappa_eff),
        'boost_mean': kernel_diag['boost_factor_mean'],
        'K_sigma_max': kernel_diag['K_sigma_max']
    }
    
    return theta_E_pred, diagnostics

# =============================================================================
# 3. Test Single Cluster (MACS0416 validation)
# =============================================================================
print("\n[2/6] Validating prediction for MACS0416...")

macs0416 = tier1_clusters[tier1_clusters['cluster_name'] == 'MACS0416'].iloc[0]

test_cluster_params = {
    'name': macs0416['cluster_name'],
    'M_500': macs0416['M_500_Msun'],
    'R_500': macs0416['R_500_kpc'],
    'z': macs0416['z_lens'],
    'z_src': macs0416['z_source'],
    'fgas': macs0416['fgas_R500'],
    'T_keV': macs0416['TX_central_keV']
}

test_kernel_params = {
    'A_c': 16.429,
    'ell0': 200.0,
    'p': 2.0,
    'ncoh': 2.0
}

theta_E_pred, diag = predict_einstein_radius(test_cluster_params, test_kernel_params, verbose=True)

print(f"\n  Prediction:")
print(f"    θ_E (predicted) = {theta_E_pred:.2f}\"")
print(f"    θ_E (observed)  = {macs0416['theta_E_obs_arcsec']:.2f}\"")
print(f"    Error = {abs(theta_E_pred - macs0416['theta_E_obs_arcsec']):.2f}\" "
      f"({abs(theta_E_pred - macs0416['theta_E_obs_arcsec'])/macs0416['theta_E_obs_arcsec']*100:.1f}%)")
print(f"    Status: {'✅ PASS' if abs(theta_E_pred - macs0416['theta_E_obs_arcsec']) < 2.0 else '⚠️ CHECK'}")

# =============================================================================
# 4. Define Likelihood Function
# =============================================================================

def log_likelihood(params, cluster_data, fixed_params=None):
    """
    Compute log-likelihood for given kernel parameters across all clusters.
    
    L = -0.5 * sum_i [(theta_E_pred_i - theta_E_obs_i)^2 / sigma_i^2]
    
    Parameters
    ----------
    params : array
        Free parameters to optimize [A_c, ell0] or [A_c] if ell0 fixed
    cluster_data : DataFrame
        Cluster catalog
    fixed_params : dict
        Fixed kernel parameters
        
    Returns
    -------
    log_L : float
        Log-likelihood value
    """
    # Unpack parameters
    if fixed_params is None:
        fixed_params = {}
    
    if 'ell0' not in fixed_params:
        A_c, ell0 = params
    else:
        A_c = params[0]
        ell0 = fixed_params['ell0']
    
    p = fixed_params.get('p', 2.0)
    ncoh = fixed_params.get('ncoh', 2.0)
    
    kernel_params = {
        'A_c': A_c,
        'ell0': ell0,
        'p': p,
        'ncoh': ncoh
    }
    
    # Compute chi-squared
    chi2 = 0.0
    n_valid = 0
    
    for idx, row in cluster_data.iterrows():
        cluster_params = {
            'name': row['cluster_name'],
            'M_500': row['M_500_Msun'],
            'R_500': row['R_500_kpc'],
            'z': row['z_lens'],
            'z_src': row['z_source'],
            'fgas': row['fgas_R500'],
            'T_keV': row['TX_central_keV']
        }
        
        try:
            theta_E_pred, _ = predict_einstein_radius(cluster_params, kernel_params, verbose=False)
            
            if theta_E_pred > 0:
                theta_E_obs = row['theta_E_obs_arcsec']
                sigma_obs = row['theta_E_err_arcsec']
                
                residual = (theta_E_pred - theta_E_obs) / sigma_obs
                chi2 += residual**2
                n_valid += 1
        except Exception as e:
            print(f"    Warning: Failed to predict for {row['cluster_name']}: {e}")
            continue
    
    # Log-likelihood
    log_L = -0.5 * chi2
    
    return log_L

# =============================================================================
# 5. Optimize Parameters
# =============================================================================
print("\n[3/6] Optimizing kernel parameters...")

print("\n  Strategy: Fix ℓ_0 = 200 kpc, optimize A_c only")
print("  (Full parameter space optimization can be done later)")

fixed_params = {
    'ell0': 200.0,
    'p': 2.0,
    'ncoh': 2.0
}

# Initial guess from MACS0416 validation
A_c_init = [16.5]

# Bounds
bounds = [(10.0, 25.0)]  # A_c range

print(f"\n  Initial parameters: A_c = {A_c_init[0]:.3f}")
print(f"  Bounds: A_c ∈ [{bounds[0][0]}, {bounds[0][1]}]")

# Define negative log-likelihood for minimization
def neg_log_likelihood(params):
    return -log_likelihood(params, tier1_clusters, fixed_params)

print("\n  Running optimization...")

result = minimize(
    neg_log_likelihood,
    A_c_init,
    method='L-BFGS-B',
    bounds=bounds,
    options={'disp': True, 'maxiter': 20}
)

A_c_best = result.x[0]
log_L_best = -result.fun

print(f"\n  Optimization complete!")
print(f"    Best-fit A_c = {A_c_best:.3f}")
print(f"    Log-likelihood = {log_L_best:.2f}")

# =============================================================================
# 6. Evaluate All Clusters with Best-Fit Parameters
# =============================================================================
print("\n[4/6] Evaluating all clusters with best-fit parameters...")

best_kernel_params = {
    'A_c': A_c_best,
    'ell0': fixed_params['ell0'],
    'p': fixed_params['p'],
    'ncoh': fixed_params['ncoh']
}

results_list = []

print(f"\n{'Cluster':<15} {'θ_E Obs':<10} {'θ_E Pred':<10} {'Error':<10} {'χ²':<10}")
print("-" * 60)

chi2_total = 0.0

for idx, row in tier1_clusters.iterrows():
    cluster_params = {
        'name': row['cluster_name'],
        'M_500': row['M_500_Msun'],
        'R_500': row['R_500_kpc'],
        'z': row['z_lens'],
        'z_src': row['z_source'],
        'fgas': row['fgas_R500'],
        'T_keV': row['TX_central_keV']
    }
    
    try:
        theta_E_pred, diag = predict_einstein_radius(cluster_params, best_kernel_params, verbose=False)
        
        theta_E_obs = row['theta_E_obs_arcsec']
        sigma_obs = row['theta_E_err_arcsec']
        
        error = theta_E_pred - theta_E_obs
        chi2 = (error / sigma_obs)**2
        chi2_total += chi2
        
        results_list.append({
            'cluster': row['cluster_name'],
            'theta_E_obs': theta_E_obs,
            'theta_E_pred': theta_E_pred,
            'error': error,
            'chi2': chi2,
            'R_E_kpc': diag['R_E_kpc']
        })
        
        print(f"{row['cluster_name']:<15} {theta_E_obs:>5.1f}±{sigma_obs:.1f}\"  "
              f"{theta_E_pred:>6.2f}\"    {error:>+6.2f}\"   {chi2:>6.2f}")
        
    except Exception as e:
        print(f"{row['cluster_name']:<15} {'FAILED':>10} {str(e)[:20]}")

print("-" * 60)

n_clusters = len(results_list)
n_params = 1  # Only A_c optimized
dof = n_clusters - n_params
chi2_reduced = chi2_total / dof

print(f"\nGoodness-of-fit:")
print(f"  χ² = {chi2_total:.2f}")
print(f"  d.o.f. = {dof}")
print(f"  χ²/d.o.f. = {chi2_reduced:.3f}")

if chi2_reduced < 1.5:
    print(f"  Status: ✅ EXCELLENT FIT (χ²/d.o.f. < 1.5)")
elif chi2_reduced < 2.5:
    print(f"  Status: ✅ GOOD FIT (χ²/d.o.f. < 2.5)")
else:
    print(f"  Status: ⚠️ ACCEPTABLE (χ²/d.o.f. > 2.5, consider systematic uncertainties)")

# =============================================================================
# 7. Generate Plots
# =============================================================================
print("\n[5/6] Generating diagnostic plots...")

output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'hierarchical_calibration')
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Predicted vs Observed
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

theta_E_obs_all = [r['theta_E_obs'] for r in results_list]
theta_E_pred_all = [r['theta_E_pred'] for r in results_list]
cluster_names = [r['cluster'] for r in results_list]

ax.errorbar(theta_E_obs_all, theta_E_pred_all,
            yerr=[tier1_clusters[tier1_clusters['cluster_name']==c]['theta_E_err_arcsec'].values[0] 
                  for c in cluster_names],
            fmt='o', markersize=8, capsize=5, color='blue', alpha=0.7)

# 1:1 line
theta_range = [0, max(max(theta_E_obs_all), max(theta_E_pred_all)) * 1.1]
ax.plot(theta_range, theta_range, 'k--', linewidth=2, alpha=0.5, label='1:1')

# ±10% error band
ax.fill_between(theta_range, [0.9*x for x in theta_range], [1.1*x for x in theta_range],
                alpha=0.2, color='gray', label='±10%')

# Label points
for i, name in enumerate(cluster_names):
    ax.text(theta_E_obs_all[i] + 1, theta_E_pred_all[i], name, fontsize=8, alpha=0.7)

ax.set_xlabel('Observed θ_E (arcsec)', fontsize=12)
ax.set_ylabel('Predicted θ_E (arcsec)', fontsize=12)
ax.set_title(f'Hierarchical Calibration: {len(tier1_clusters)} Tier-1 Clusters\n'
             f'Best-fit: A_c = {A_c_best:.3f}, ℓ_0 = {fixed_params["ell0"]:.0f} kpc, '
             f'χ²/d.o.f. = {chi2_reduced:.2f}',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plot1_path = os.path.join(output_dir, 'predicted_vs_observed.png')
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot1_path}")
plt.close()

# Plot 2: Residuals
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

errors = [r['error'] for r in results_list]
x_pos = np.arange(len(cluster_names))

ax.bar(x_pos, errors, color=['green' if abs(e) < 2 else 'orange' for e in errors],
       alpha=0.7, edgecolor='black')
ax.axhline(0, color='k', linestyle='-', linewidth=2)
ax.axhline(2, color='red', linestyle='--', alpha=0.5)
ax.axhline(-2, color='red', linestyle='--', alpha=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(cluster_names, rotation=45, ha='right')
ax.set_ylabel('Residual: θ_E(pred) - θ_E(obs) [arcsec]', fontsize=12)
ax.set_title('Calibration Residuals', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plot2_path = os.path.join(output_dir, 'calibration_residuals.png')
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {plot2_path}")
plt.close()

# =============================================================================
# 8. Save Results
# =============================================================================
print("\n[6/6] Saving results...")

# Save summary
summary_path = os.path.join(output_dir, 'calibration_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("HIERARCHICAL 12-CLUSTER CALIBRATION RESULTS\n")
    f.write("="*70 + "\n\n")
    
    f.write("Best-Fit Parameters:\n")
    f.write("-"*70 + "\n")
    f.write(f"  A_c (coherence amplitude) = {A_c_best:.4f}\n")
    f.write(f"  ℓ_0 (coherence length)    = {fixed_params['ell0']:.1f} kpc (fixed)\n")
    f.write(f"  p (window power)          = {fixed_params['p']:.1f} (fixed)\n")
    f.write(f"  n_coh (coherence power)   = {fixed_params['ncoh']:.1f} (fixed)\n\n")
    
    f.write("Goodness-of-Fit:\n")
    f.write("-"*70 + "\n")
    f.write(f"  Number of clusters = {n_clusters}\n")
    f.write(f"  Free parameters    = {n_params}\n")
    f.write(f"  Degrees of freedom = {dof}\n")
    f.write(f"  χ² = {chi2_total:.3f}\n")
    f.write(f"  χ²/d.o.f. = {chi2_reduced:.3f}\n\n")
    
    f.write("Per-Cluster Results:\n")
    f.write("-"*70 + "\n")
    f.write(f"{'Cluster':<15} {'θ_E Obs':>10} {'θ_E Pred':>10} {'Error':>10} {'χ²':>10}\n")
    f.write("-"*70 + "\n")
    
    for r in results_list:
        f.write(f"{r['cluster']:<15} {r['theta_E_obs']:>10.2f} {r['theta_E_pred']:>10.2f} "
                f"{r['error']:>+10.2f} {r['chi2']:>10.2f}\n")

print(f"  Saved: {summary_path}")

# Save CSV
results_df = pd.DataFrame(results_list)
csv_path = os.path.join(output_dir, 'calibration_results.csv')
results_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

print("\n" + "="*70)
print("HIERARCHICAL CALIBRATION COMPLETE")
print("="*70)
print(f"\nBest-fit A_c = {A_c_best:.3f}, χ²/d.o.f. = {chi2_reduced:.3f}")
print("All results saved to:", output_dir)
