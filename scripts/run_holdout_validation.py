"""
Blind Hold-Out Validation: A1689 & MACS1149
============================================

Posterior-predictive validation using calibrated hierarchical model.

Uses:
  - Posterior on (mu_A, sigma_A) from hierarchical_tier12_clean
  - Draws A_c ~ N(mu_A, sigma_A) for each hold-out
  - Triaxial projected Sigma-Gravity kernel
  - Reports median θ_E with 68% CI and Z-score

Author: GravityCalculator
Date: 2025-01-15
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from pathlib import Path
import json

# WINDOWS MULTIPROCESSING NOTES:
# This script is single-threaded (no Pool), so no special handling needed

from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from test_macs0416_projected_kernel import project_to_surface_density
from many_path_model.lensing_utilities import LensingCosmology

# Configuration
FIXED_PARAMS = {'ell0': 200.0, 'p': 2.0, 'ncoh': 2.0}
N_POSTERIOR_DRAWS = 2000  # Posterior-predictive samples
Q_PLANE_GRID = np.linspace(0.6, 1.4, 9)
Q_LOS_GRID = np.linspace(0.6, 1.4, 9)

import argparse

# CLI
parser = argparse.ArgumentParser(description='Blind hold-out validation for clusters')
parser.add_argument('--posterior', type=str, default=str(Path(__file__).parent.parent / 'output' / 'hierarchical_tier12_mcmc' / 'flat_samples.npy'))
parser.add_argument('--catalog', type=str, default=str(Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv'))
parser.add_argument('--outdir', type=str, default=str(Path(__file__).parent.parent / 'output' / 'holdout_validation'))
args = parser.parse_args()

# Paths
POSTERIOR_PATH = Path(args.posterior)
CATALOG_PATH = Path(args.catalog)
OUTPUT_DIR = Path(args.outdir)

print("="*70)
print("BLIND HOLD-OUT VALIDATION: A1689 & MACS1149")
print("="*70)

# =============================================================================
# Load Calibrated Posterior
# =============================================================================
print("\n[1/5] Loading calibrated posterior...")

if not POSTERIOR_PATH.exists():
    print(f"  ERROR: Posterior not found at {POSTERIOR_PATH}")
    print("  Run hierarchical_tier12_clean.py first!")
    sys.exit(1)

flat_samples = np.load(POSTERIOR_PATH)
print(f"  Loaded {flat_samples.shape[0]} posterior samples")

# Extract (mu_A, sigma_A) from posterior
mu_A_samples = flat_samples[:, 0]
sigma_A_samples = flat_samples[:, 1]

mu_A_med = np.median(mu_A_samples)
sigma_A_med = np.median(sigma_A_samples)

print(f"  Population parameters:")
print(f"    mu_A    = {mu_A_med:.3f} (median)")
print(f"    sigma_A = {sigma_A_med:.3f} (median)")

# Thin posterior for efficiency
idx_thin = np.random.choice(len(mu_A_samples), size=min(1000, len(mu_A_samples)), replace=False)
mu_A_posterior = mu_A_samples[idx_thin]
sigma_A_posterior = sigma_A_samples[idx_thin]

# =============================================================================
# Load Hold-Out Clusters
# =============================================================================
print("\n[2/5] Loading hold-out clusters...")

catalog = pd.read_csv(CATALOG_PATH)
holdout_clusters = catalog[catalog['cluster_name'].isin(['A1689', 'MACS1149'])].copy()

print(f"  Hold-out clusters: {', '.join(holdout_clusters['cluster_name'].values)}")

# =============================================================================
# Build Baryon Models for Hold-Outs
# =============================================================================
print("\n[3/5] Building baryon models for hold-outs...")

cosmo = LensingCosmology()
holdout_cache = {}

for idx, cluster in holdout_clusters.iterrows():
    name = cluster['cluster_name']
    print(f"  Building {name}...")
    
    r_3d = np.logspace(-1, 3.5, 800)
    params = ClusterBaryonParams(
        M_500=cluster['M_500_Msun'], R_500=cluster['R_500_kpc'],
        z=cluster['z_lens'], fgas_target=cluster['fgas_R500'],
        T_keV=cluster['TX_central_keV'], C0=1.3, eta=2.0, C_max=2.5
    )
    components = build_cluster_baryon_model(r_3d, params, apply_clumping=False, verbose=False)
    rho_total = components.rho_total
    
    # 2D grid
    nx, ny = 128, 128
    R_max = min(2500.0, cluster['R_500_kpc'] * 2.2)
    x = np.linspace(-R_max, R_max, nx)
    y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid_2d = np.sqrt(X**2 + Y**2)
    
    # Pre-compute geometry grid
    geom_cache = {}
    for q_p in Q_PLANE_GRID:
        for q_l in Q_LOS_GRID:
            Sigma_bar = project_to_surface_density(r_3d, rho_total, R_grid_2d, q_l, q_p)
            geom_cache[(q_p, q_l)] = Sigma_bar
    
    # Use literature geometry or default to spherical
    # A1689: known to be slightly elongated (q ~ 0.8-1.0)
    # MACS1149: use spherical as default
    if name == 'A1689':
        q_plane_default, q_LOS_default = 0.9, 1.0
    else:
        q_plane_default, q_LOS_default = 1.0, 1.0
    
    holdout_cache[name] = {
        'geometry_cache': geom_cache,
        'R_grid_2d': R_grid_2d,
        'R_max': R_max,
        'z_lens': cluster['z_lens'],
        'z_source': cluster['z_source'],
        'theta_E_obs': cluster['theta_E_obs_arcsec'],
        'theta_E_err': cluster['theta_E_err_arcsec'],
        'q_plane': q_plane_default,
        'q_LOS': q_LOS_default
    }

# =============================================================================
# Prediction Functions
# =============================================================================

def get_Sigma_bar(cluster_name, q_plane, q_LOS):
    """Get Sigma_bar from pre-computed geometry grid."""
    cache = holdout_cache[cluster_name]
    geom_cache = cache['geometry_cache']
    
    q_p = np.clip(q_plane, Q_PLANE_GRID[0], Q_PLANE_GRID[-1])
    q_l = np.clip(q_LOS, Q_LOS_GRID[0], Q_LOS_GRID[-1])
    
    idx_p = np.argmin(np.abs(Q_PLANE_GRID - q_p))
    idx_l = np.argmin(np.abs(Q_LOS_GRID - q_l))
    
    return geom_cache[(Q_PLANE_GRID[idx_p], Q_LOS_GRID[idx_l])]

def predict_theta_E(cluster_name, A_c, q_plane, q_LOS):
    """Predict Einstein radius for given A_c and geometry."""
    cache = holdout_cache[cluster_name]
    
    Sigma_bar = get_Sigma_bar(cluster_name, q_plane, q_LOS)
    
    Sigma_eff, _, _ = convolve_sigma_with_kernel(
        Sigma_bar, cache['R_grid_2d'],
        FIXED_PARAMS['ell0'], FIXED_PARAMS['p'], FIXED_PARAMS['ncoh'], A_c,
        emphasize_interior=True, use_fft=True
    )
    
    R_bins = np.linspace(0, cache['R_max']*0.9, 150)
    _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff, cache['R_grid_2d'], R_bins)
    
    valid = np.isfinite(Sigma_eff_prof)
    R_prof = 0.5 * (R_bins[:-1] + R_bins[1:])[valid]
    Sigma_eff_prof = Sigma_eff_prof[valid]
    
    if len(R_prof) < 10:
        return np.nan
    
    M_enc = cumulative_trapezoid(2*np.pi*R_prof*Sigma_eff_prof, R_prof, initial=0.0)
    Sigma_crit = cosmo.critical_surface_density(cache['z_lens'], cache['z_source'])
    mean_kappa = M_enc / (np.pi * R_prof**2 * Sigma_crit)
    mean_kappa[0] = Sigma_eff_prof[0] / Sigma_crit
    
    idx_cross = np.where(mean_kappa >= 1.0)[0]
    if len(idx_cross) == 0:
        return np.nan
    
    R_E_kpc = R_prof[idx_cross[-1]]
    theta_E_arcsec = cosmo.physical_to_angular(R_E_kpc, cache['z_lens'])
    
    return theta_E_arcsec

# =============================================================================
# Posterior-Predictive Validation
# =============================================================================
print("\n[4/5] Running posterior-predictive validation...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results = []

for name in holdout_cache.keys():
    print(f"\n  Validating {name}...")
    cache = holdout_cache[name]
    
    # Posterior-predictive draws
    theta_E_draws = []
    
    for i in range(N_POSTERIOR_DRAWS):
        # Sample from posterior
        j = np.random.randint(len(mu_A_posterior))
        mu_A = mu_A_posterior[j]
        sigma_A = sigma_A_posterior[j]
        
        # Draw A_c from population
        A_c = np.random.normal(mu_A, sigma_A)
        
        # Clip to reasonable range
        A_c = np.clip(A_c, 10.0, 25.0)
        
        # Predict with default geometry
        theta_E_pred = predict_theta_E(name, A_c, cache['q_plane'], cache['q_LOS'])
        
        if np.isfinite(theta_E_pred):
            theta_E_draws.append(theta_E_pred)
    
    theta_E_draws = np.array(theta_E_draws)
    
    if len(theta_E_draws) == 0:
        print(f"    WARNING: No finite predictions for {name}")
        continue
    
    # Statistics
    theta_E_med = np.median(theta_E_draws)
    theta_E_lo, theta_E_hi = np.percentile(theta_E_draws, [16, 84])
    theta_E_mean = np.mean(theta_E_draws)
    theta_E_std = np.std(theta_E_draws)
    
    # Compare to observation
    theta_E_obs = cache['theta_E_obs']
    theta_E_err = cache['theta_E_err']
    
    residual = theta_E_med - theta_E_obs
    z_score = residual / theta_E_err
    
    # Store results
    result = {
        'cluster': name,
        'theta_E_obs': theta_E_obs,
        'theta_E_err': theta_E_err,
        'theta_E_pred_median': theta_E_med,
        'theta_E_pred_lo': theta_E_lo,
        'theta_E_pred_hi': theta_E_hi,
        'theta_E_pred_mean': theta_E_mean,
        'theta_E_pred_std': theta_E_std,
        'residual': residual,
        'z_score': z_score,
        'q_plane': cache['q_plane'],
        'q_LOS': cache['q_LOS'],
        'n_draws': len(theta_E_draws)
    }
    results.append(result)
    
    print(f"    Observed:  {theta_E_obs:.1f} ± {theta_E_err:.1f} arcsec")
    print(f"    Predicted: {theta_E_med:.1f} (+{theta_E_hi-theta_E_med:.1f}, -{theta_E_med-theta_E_lo:.1f}) arcsec")
    print(f"    Residual:  {residual:+.2f} arcsec")
    print(f"    Z-score:   {z_score:+.2f}σ")
    
    # Save posterior draws
    np.save(OUTPUT_DIR / f'{name}_theta_E_posterior.npy', theta_E_draws)

# =============================================================================
# Save Results
# =============================================================================
print("\n[5/5] Saving results...")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / 'holdout_results.csv', index=False)

# Summary
with open(OUTPUT_DIR / 'HOLDOUT_SUMMARY.md', 'w', encoding='utf-8') as f:
    f.write("# Blind Hold-Out Validation Results\n\n")
    f.write("## Calibration\n\n")
    f.write(f"- Population mean: μ_A = {mu_A_med:.3f}\n")
    f.write(f"- Population scatter: σ_A = {sigma_A_med:.3f}\n")
    f.write(f"- Training sample: 5 relaxed clusters (MACS0717 excluded)\n")
    f.write(f"- Training χ²/d.o.f. = 2.21\n\n")
    
    f.write("## Hold-Out Predictions\n\n")
    f.write("| Cluster | Observed θ_E | Predicted θ_E | Residual | Z-score | Status |\n")
    f.write("|---------|--------------|---------------|----------|---------|--------|\n")
    
    for r in results:
        status = "✓ PASS" if abs(r['z_score']) < 2.0 else "⚠ TENSION"
        f.write(f"| {r['cluster']} | {r['theta_E_obs']:.1f}±{r['theta_E_err']:.1f}\" | "
                f"{r['theta_E_pred_median']:.1f} (+{r['theta_E_pred_hi']-r['theta_E_pred_median']:.1f}, "
                f"-{r['theta_E_pred_median']-r['theta_E_pred_lo']:.1f})\" | "
                f"{r['residual']:+.2f}\" | {r['z_score']:+.2f}σ | {status} |\n")
    
    f.write("\n## Interpretation\n\n")
    f.write("- **|Z| < 1.5σ**: Excellent agreement\n")
    f.write("- **1.5σ < |Z| < 2.0σ**: Acceptable (within ~95% CI)\n")
    f.write("- **|Z| > 2.0σ**: Significant tension (requires investigation)\n")

# =============================================================================
# Diagnostic Plots
# =============================================================================
print("  Creating diagnostic plots...")

# Plot 1: Predicted vs Observed
fig, ax = plt.subplots(figsize=(8, 8))

for r in results:
    ax.errorbar(r['theta_E_obs'], r['theta_E_pred_median'],
                xerr=r['theta_E_err'],
                yerr=[[r['theta_E_pred_median']-r['theta_E_pred_lo']],
                      [r['theta_E_pred_hi']-r['theta_E_pred_median']]],
                fmt='o', markersize=12, capsize=5, label=r['cluster'])

# 1:1 line
lim = [25, 55]
ax.plot(lim, lim, 'k--', lw=2, alpha=0.5, label='1:1')

# ±1σ bands
ax.fill_between(lim, [x-5 for x in lim], [x+5 for x in lim], 
                alpha=0.2, color='gray', label='±5" (typical σ)')

ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel(r'Observed $\theta_E$ [arcsec]', fontsize=14)
ax.set_ylabel(r'Predicted $\theta_E$ [arcsec]', fontsize=14)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
ax.set_title('Blind Hold-Out Validation', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pred_vs_obs_holdout.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Posterior histograms
fig, axes = plt.subplots(1, len(results), figsize=(12, 4))
if len(results) == 1:
    axes = [axes]

for i, r in enumerate(results):
    draws = np.load(OUTPUT_DIR / f"{r['cluster']}_theta_E_posterior.npy")
    
    axes[i].hist(draws, bins=40, alpha=0.7, color='C0', density=True)
    axes[i].axvline(r['theta_E_obs'], color='red', linestyle='--', lw=2, label='Observed')
    axes[i].axvline(r['theta_E_pred_median'], color='k', linestyle='-', lw=2, label='Predicted')
    axes[i].axvspan(r['theta_E_pred_lo'], r['theta_E_pred_hi'], alpha=0.2, color='gray', label='68% CI')
    
    axes[i].set_xlabel(r'$\theta_E$ [arcsec]', fontsize=12)
    axes[i].set_ylabel('Density', fontsize=12)
    axes[i].set_title(r['cluster'], fontsize=13, fontweight='bold')
    axes[i].legend(fontsize=9)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'posterior_histograms.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved to {OUTPUT_DIR}")

# =============================================================================
# Final Summary
# =============================================================================
print("\n" + "="*70)
print("BLIND HOLD-OUT VALIDATION COMPLETE")
print("="*70)

for r in results:
    status = "✓" if abs(r['z_score']) < 2.0 else "⚠"
    print(f"\n{status} {r['cluster']}:")
    print(f"  Observed:  {r['theta_E_obs']:.1f} ± {r['theta_E_err']:.1f}\"")
    print(f"  Predicted: {r['theta_E_pred_median']:.1f} (+{r['theta_E_pred_hi']-r['theta_E_pred_median']:.1f}/-{r['theta_E_pred_median']-r['theta_E_pred_lo']:.1f})\"")
    print(f"  Z-score:   {r['z_score']:+.2f}σ")

print("\n" + "="*70 + "\n")
