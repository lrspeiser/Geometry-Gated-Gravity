"""
Tier-1+2 Hierarchical MCMC Calibration (Simplified)
===================================================

Calibration using 8 Tier-1+2 clusters with spherical geometry.
Fits only global A_c parameter (no per-cluster geometry yet).

Global Parameter:
----------------
  - A_c: Coherence amplitude [flat prior 10-25]

Fixed Parameters:
----------------
  - ell0 = 200 kpc
  - p = 2.0
  - n_coh = 2.0
  - Spherical geometry (q_plane = q_LOS = 1.0)

Training Set (6 clusters):
- MACS0416, A2744, A370 (Tier-1)
- MACS0717, RXJ1347, CL0024 (Tier-2)

Hold-Out Set (2 clusters, BLIND):
- A1689, MACS1149

MCMC Setup:
-----------
- 64 walkers
- 800 burn-in steps
- 2500 production steps

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
from scipy.stats import norm
import emcee
import corner
from pathlib import Path
import time

# Import baryon model
from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams

# Import Sigma-Gravity kernel
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average

# Import test helpers
from test_macs0416_projected_kernel import project_to_surface_density

# Import cosmology
from many_path_model.lensing_utilities import LensingCosmology

print("="*70)
print("TIER-1+2 HIERARCHICAL MCMC (SIMPLIFIED - SPHERICAL)")
print("="*70)

# =============================================================================
# Configuration
# =============================================================================

# Fixed parameters
FIXED_PARAMS = {
    'ell0': 200.0,  # kpc
    'p': 2.0,
    'ncoh': 2.0
}

# MCMC settings
N_WALKERS = 64
N_STEPS = 2500
N_BURN = 800

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'tier12_mcmc_simple'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Fixed: ℓ_0 = {FIXED_PARAMS['ell0']:.0f} kpc, p = {FIXED_PARAMS['p']:.1f}, n_coh = {FIXED_PARAMS['ncoh']:.1f}")
print(f"  Free: A_c (global only)")
print(f"  Geometry: Spherical (q_plane = q_LOS = 1.0)")
print(f"  MCMC: {N_WALKERS} walkers × {N_STEPS} steps ({N_BURN} burn-in)")

# =============================================================================
# Load Catalog and Create Train/Hold-Out Split
# =============================================================================
print("\n[1/8] Loading Tier-1+2 clusters and creating train/hold-out split...")

catalog_path = Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv'
catalog = pd.read_csv(catalog_path)

# Use Tier-1 + Tier-2 (8 clusters total)
tier12_clusters = catalog[catalog['tier'] <= 2].copy()

print(f"  Tier-1+2 clusters available: {len(tier12_clusters)}")

# Define train and hold-out sets explicitly
train_names = ['MACS0416', 'A2744', 'A370', 'MACS0717', 'RXJ1347', 'CL0024']
holdout_names = ['A1689', 'MACS1149']

train_clusters = tier12_clusters[tier12_clusters['cluster_name'].isin(train_names)].copy()
holdout_clusters = tier12_clusters[tier12_clusters['cluster_name'].isin(holdout_names)].copy()

print(f"\n  Train set: {len(train_clusters)} clusters")
for idx, row in train_clusters.iterrows():
    print(f"    - {row['cluster_name']}: z={row['z_lens']:.3f}, θ_E={row['theta_E_obs_arcsec']:.1f}\", Tier-{row['tier']}")

print(f"\n  Hold-out set: {len(holdout_clusters)} clusters (BLIND)")
for idx, row in holdout_clusters.iterrows():
    print(f"    - {row['cluster_name']}: z={row['z_lens']:.3f}, θ_E={row['theta_E_obs_arcsec']:.1f}\", Tier-{row['tier']}")

# =============================================================================
# Baryon Profile Cache
# =============================================================================
print("\n[2/8] Building and caching baryon profiles...")

baryon_cache = {}
cosmo = LensingCosmology()

for idx, cluster in train_clusters.iterrows():
    name = cluster['cluster_name']
    print(f"  Building {name}...")
    
    # Build 3D profile
    r_3d = np.logspace(-1, 3.5, 1200)
    
    params = ClusterBaryonParams(
        M_500=cluster['M_500_Msun'],
        R_500=cluster['R_500_kpc'],
        z=cluster['z_lens'],
        fgas_target=cluster['fgas_R500'],
        T_keV=cluster['TX_central_keV'],
        C0=1.3,
        eta=2.0,
        C_max=2.5
    )
    
    components = build_cluster_baryon_model(r_3d, params, apply_clumping=False, verbose=False)
    rho_total = components.rho_total
    
    # Project to 2D (moderate resolution for speed)
    nx, ny = 256, 256
    R_max = min(2500.0, cluster['R_500_kpc'] * 2.2)
    x = np.linspace(-R_max, R_max, nx)
    y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid_2d = np.sqrt(X**2 + Y**2)
    
    # Spherical projection
    Sigma_bar = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)
    
    # Store in cache
    baryon_cache[name] = {
        'Sigma_bar': Sigma_bar,
        'R_grid_2d': R_grid_2d,
        'R_max': R_max,
        'nx': nx,
        'z_lens': cluster['z_lens'],
        'z_source': cluster['z_source'],
        'theta_E_obs': cluster['theta_E_obs_arcsec'],
        'theta_E_err': cluster['theta_E_err_arcsec']
    }

print(f"  Cached {len(baryon_cache)} baryon profiles.")

# =============================================================================
# Prediction Function
# =============================================================================

def predict_einstein_radius(cluster_name, A_c):
    """
    Predict Einstein radius with spherical geometry.
    
    Parameters
    ----------
    cluster_name : str
        Cluster name
    A_c : float
        Coherence amplitude
        
    Returns
    -------
    theta_E_pred : float
        Predicted Einstein radius [arcsec]
    """
    cache = baryon_cache[cluster_name]
    
    # Apply Sigma-Gravity kernel
    Sigma_eff, _, _ = convolve_sigma_with_kernel(
        cache['Sigma_bar'], cache['R_grid_2d'], 
        FIXED_PARAMS['ell0'], FIXED_PARAMS['p'], FIXED_PARAMS['ncoh'], A_c,
        emphasize_interior=True, use_fft=True
    )
    
    # Find Einstein radius (where mean kappa = 1)
    R_bins = np.linspace(0, cache['R_max']*0.9, 300)
    _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff, cache['R_grid_2d'], R_bins)
    
    # Remove NaNs
    valid = np.isfinite(Sigma_eff_prof)
    R_prof = 0.5 * (R_bins[:-1] + R_bins[1:])[valid]
    Sigma_eff_prof = Sigma_eff_prof[valid]
    
    if len(R_prof) < 10:
        return np.nan
    
    # Cumulative mass and mean convergence
    M_enc = cumulative_trapezoid(2*np.pi*R_prof*Sigma_eff_prof, R_prof, initial=0.0)
    Sigma_crit = cosmo.critical_surface_density(cache['z_lens'], cache['z_source'])
    mean_kappa = M_enc / (np.pi * R_prof**2 * Sigma_crit)
    mean_kappa[0] = Sigma_eff_prof[0] / Sigma_crit
    
    # Find where mean_kappa crosses 1.0
    idx_cross = np.where(mean_kappa >= 1.0)[0]
    if len(idx_cross) == 0:
        return np.nan
    
    R_E_kpc = R_prof[idx_cross[-1]]
    theta_E_arcsec = cosmo.physical_to_angular(R_E_kpc, cache['z_lens'])
    
    return theta_E_arcsec

# =============================================================================
# MCMC Setup
# =============================================================================

cluster_names_train = train_clusters['cluster_name'].values

print(f"\n[3/8] MCMC parameter structure:")
print(f"  Total parameters: 1 (A_c only)")

# =============================================================================
# Priors and Likelihood
# =============================================================================

def log_prior(A_c):
    """Log-prior for A_c: flat [10, 25]."""
    if A_c < 10.0 or A_c > 25.0:
        return -np.inf
    return 0.0

def log_likelihood(A_c):
    """Log-likelihood: chi-squared on Einstein radii."""
    chi2 = 0.0
    
    for cluster_name in cluster_names_train:
        try:
            theta_E_pred = predict_einstein_radius(cluster_name, A_c)
        except Exception as e:
            return -np.inf
        
        if not np.isfinite(theta_E_pred):
            return -np.inf
        
        cache = baryon_cache[cluster_name]
        theta_E_obs = cache['theta_E_obs']
        theta_E_err = cache['theta_E_err']
        
        chi2 += ((theta_E_obs - theta_E_pred) / theta_E_err)**2
    
    return -0.5 * chi2

def log_probability(A_c):
    """Combined log-posterior."""
    lp = log_prior(A_c)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(A_c)
    return lp + ll

# =============================================================================
# Initialize Walkers
# =============================================================================
print("\n[4/8] Initializing walkers...")

# Starting point
A_c_init = 16.5
pos = A_c_init + 0.5 * np.random.randn(N_WALKERS)
pos = np.clip(pos, 12.0, 20.0)

print(f"  Initialized {N_WALKERS} walkers around A_c = {A_c_init:.2f}")

# =============================================================================
# Run MCMC
# =============================================================================
print(f"\n[5/8] Running MCMC ({N_STEPS} steps)...")
print("  This may take 20-30 minutes...\n")

sampler = emcee.EnsembleSampler(N_WALKERS, 1, log_probability)

start_time = time.time()
sampler.run_mcmc(pos.reshape(-1, 1), N_STEPS, progress=True)
end_time = time.time()

print(f"\n  MCMC completed in {(end_time - start_time)/60:.1f} minutes")

# =============================================================================
# Analyze Chains
# =============================================================================
print("\n[6/8] Analyzing chains...")

acc_frac = np.mean(sampler.acceptance_fraction)
print(f"  Mean acceptance fraction: {acc_frac:.3f}")

if acc_frac < 0.2 or acc_frac > 0.6:
    print("  ⚠️  WARNING: Acceptance fraction outside healthy range [0.2, 0.6]")
else:
    print("  ✅ Acceptance fraction healthy")

# Discard burn-in
flat_samples = sampler.get_chain(discard=N_BURN, flat=True)

print(f"  Samples after burn-in: {flat_samples.shape[0]}")

# Posteriors
percentiles = np.percentile(flat_samples[:, 0], [16, 50, 84])
A_c_med = percentiles[1]
A_c_err_low = percentiles[1] - percentiles[0]
A_c_err_high = percentiles[2] - percentiles[1]

print(f"\n  Posterior:")
print(f"    A_c = {A_c_med:.3f} (+{A_c_err_high:.3f}, -{A_c_err_low:.3f})")

# =============================================================================
# Evaluate on Train and Hold-Out Sets
# =============================================================================
print("\n[7/8] Evaluating model on train and hold-out sets...")

# Train set
print("\n  TRAIN SET:")
train_results = []

for cluster_name in cluster_names_train:
    cache = baryon_cache[cluster_name]
    theta_E_pred = predict_einstein_radius(cluster_name, A_c_med)
    theta_E_obs = cache['theta_E_obs']
    theta_E_err = cache['theta_E_err']
    
    error = theta_E_pred - theta_E_obs
    chi2 = (error / theta_E_err)**2
    
    train_results.append({
        'cluster': cluster_name,
        'theta_E_obs': theta_E_obs,
        'theta_E_pred': theta_E_pred,
        'error': error,
        'chi2': chi2
    })
    
    print(f"    {cluster_name:12s}: obs={theta_E_obs:.1f}\", pred={theta_E_pred:.1f}\", err={error:+.2f}\", χ²={chi2:.2f}")

train_chi2_total = sum(r['chi2'] for r in train_results)
train_dof = len(train_results) - 1
print(f"\n  Train χ² = {train_chi2_total:.2f}, d.o.f. = {train_dof}, χ²/d.o.f. = {train_chi2_total/train_dof:.2f}")

# Hold-out set
print("\n  HOLD-OUT SET (BLIND):")
holdout_results = []

for idx, cluster in holdout_clusters.iterrows():
    name = cluster['cluster_name']
    
    # Build baryon profile
    print(f"    Building {name}...")
    r_3d = np.logspace(-1, 3.5, 1200)
    
    params = ClusterBaryonParams(
        M_500=cluster['M_500_Msun'],
        R_500=cluster['R_500_kpc'],
        z=cluster['z_lens'],
        fgas_target=cluster['fgas_R500'],
        T_keV=cluster['TX_central_keV'],
        C0=1.3,
        eta=2.0,
        C_max=2.5
    )
    
    components = build_cluster_baryon_model(r_3d, params, apply_clumping=False, verbose=False)
    rho_total = components.rho_total
    
    nx, ny = 256, 256
    R_max = min(2500.0, cluster['R_500_kpc'] * 2.2)
    x = np.linspace(-R_max, R_max, nx)
    y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid_2d = np.sqrt(X**2 + Y**2)
    
    Sigma_bar = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)
    
    # Temporarily add to cache
    baryon_cache[name] = {
        'Sigma_bar': Sigma_bar,
        'R_grid_2d': R_grid_2d,
        'R_max': R_max,
        'nx': nx,
        'z_lens': cluster['z_lens'],
        'z_source': cluster['z_source'],
        'theta_E_obs': cluster['theta_E_obs_arcsec'],
        'theta_E_err': cluster['theta_E_err_arcsec']
    }
    
    # Predict
    theta_E_pred = predict_einstein_radius(name, A_c_med)
    
    error = theta_E_pred - cluster['theta_E_obs_arcsec']
    chi2 = (error / cluster['theta_E_err_arcsec'])**2
    
    holdout_results.append({
        'cluster': name,
        'theta_E_obs': cluster['theta_E_obs_arcsec'],
        'theta_E_pred': theta_E_pred,
        'error': error,
        'chi2': chi2
    })
    
    print(f"      {name:12s}: obs={cluster['theta_E_obs_arcsec']:.1f}\", pred={theta_E_pred:.1f}\", err={error:+.2f}\", χ²={chi2:.2f}")

holdout_chi2_total = sum(r['chi2'] for r in holdout_results)
holdout_dof = len(holdout_results)
print(f"\n  Hold-out χ² = {holdout_chi2_total:.2f}, d.o.f. = {holdout_dof}, χ²/d.o.f. = {holdout_chi2_total/holdout_dof:.2f}")

# =============================================================================
# Save Results
# =============================================================================
print("\n[8/8] Saving results and generating plots...")

np.save(OUTPUT_DIR / 'flat_samples.npy', flat_samples)

with open(OUTPUT_DIR / 'posterior_summary.txt', 'w') as f:
    f.write("Tier-1+2 Hierarchical MCMC (Simplified) - Posterior Summary\n")
    f.write("="*70 + "\n\n")
    f.write(f"A_c = {A_c_med:.3f} (+{A_c_err_high:.3f}, -{A_c_err_low:.3f})\n\n")
    f.write(f"Train χ²/d.o.f. = {train_chi2_total/train_dof:.2f}\n")
    f.write(f"Hold-out χ²/d.o.f. = {holdout_chi2_total/holdout_dof:.2f}\n")

train_df = pd.DataFrame(train_results)
holdout_df = pd.DataFrame(holdout_results)
train_df.to_csv(OUTPUT_DIR / 'train_results.csv', index=False)
holdout_df.to_csv(OUTPUT_DIR / 'holdout_results.csv', index=False)

# Histogram of A_c
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(flat_samples[:, 0], bins=50, alpha=0.7, color='C0')
ax.axvline(A_c_med, color='k', linestyle='--', linewidth=2, label=f'Median = {A_c_med:.2f}')
ax.axvline(percentiles[0], color='gray', linestyle=':', linewidth=1.5)
ax.axvline(percentiles[2], color='gray', linestyle=':', linewidth=1.5)
ax.set_xlabel('$A_c$', fontsize=14)
ax.set_ylabel('Samples', fontsize=14)
ax.set_title('Posterior Distribution', fontsize=15)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'posterior_Ac.png', dpi=150, bbox_inches='tight')
plt.close()

# Predicted vs Observed
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

obs_train = [r['theta_E_obs'] for r in train_results]
pred_train = [r['theta_E_pred'] for r in train_results]
err_train = [baryon_cache[r['cluster']]['theta_E_err'] for r in train_results]

ax1.errorbar(obs_train, pred_train, yerr=err_train, fmt='o', capsize=5, label='Train', color='C0', markersize=8)
lim1 = [min(obs_train + pred_train) * 0.9, max(obs_train + pred_train) * 1.1]
ax1.plot(lim1, lim1, 'k--', alpha=0.5, label='1:1')
ax1.set_xlabel('Observed $\\theta_E$ [arcsec]', fontsize=12)
ax1.set_ylabel('Predicted $\\theta_E$ [arcsec]', fontsize=12)
ax1.set_title(f'Train Set (χ²/d.o.f. = {train_chi2_total/train_dof:.2f})', fontsize=13)
ax1.legend()
ax1.grid(alpha=0.3)

obs_holdout = [r['theta_E_obs'] for r in holdout_results]
pred_holdout = [r['theta_E_pred'] for r in holdout_results]
err_holdout = [holdout_clusters[holdout_clusters['cluster_name']==r['cluster']]['theta_E_err_arcsec'].values[0] for r in holdout_results]

ax2.errorbar(obs_holdout, pred_holdout, yerr=err_holdout, fmt='s', capsize=5, label='Hold-out', color='C1', markersize=8)
lim2 = [min(obs_holdout + pred_holdout) * 0.9, max(obs_holdout + pred_holdout) * 1.1]
ax2.plot(lim2, lim2, 'k--', alpha=0.5, label='1:1')
ax2.set_xlabel('Observed $\\theta_E$ [arcsec]', fontsize=12)
ax2.set_ylabel('Predicted $\\theta_E$ [arcsec]', fontsize=12)
ax2.set_title(f'Hold-Out Set (χ²/d.o.f. = {holdout_chi2_total/holdout_dof:.2f})', fontsize=13)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'predicted_vs_observed.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Results saved to {OUTPUT_DIR}")

print("\n" + "="*70)
print("TIER-1+2 CALIBRATION COMPLETE")
print("="*70)
print(f"\nFinal Results:")
print(f"  A_c = {A_c_med:.3f} (+{A_c_err_high:.3f}, -{A_c_err_low:.3f})")
print(f"  Train χ²/d.o.f. = {train_chi2_total/train_dof:.2f}")
print(f"  Hold-out χ²/d.o.f. = {holdout_chi2_total/holdout_dof:.2f}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("="*70 + "\n")
