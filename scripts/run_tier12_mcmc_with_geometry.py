"""
Tier-1+2 Hierarchical MCMC Calibration with Geometry
====================================================

Enhanced calibration using 8 Tier-1+2 clusters with per-cluster geometry
parameters (q_plane, q_LOS) as nuisance parameters.

Global Parameters:
-----------------
  - A_c: Coherence amplitude [flat prior 10-25]
  - ell0: Coherence length (optional, fixed at 200 kpc initially)

Per-Cluster Nuisance Parameters:
--------------------------------
  - q_plane: In-plane axis ratio [Normal(1.0, 0.15), truncated 0.6-1.4]
  - q_LOS: Line-of-sight axis ratio [derived from cos(θ) uniform, 0.6-1.6]
  - kappa_ext: External convergence [Normal(0.0, 0.03)]

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
- Parallel tempering disabled (for speed)

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
print("TIER-1+2 HIERARCHICAL MCMC WITH GEOMETRY")
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
OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'tier12_mcmc_geometry'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Fixed: ℓ_0 = {FIXED_PARAMS['ell0']:.0f} kpc, p = {FIXED_PARAMS['p']:.1f}, n_coh = {FIXED_PARAMS['ncoh']:.1f}")
print(f"  Free: A_c (global), per-cluster (q_plane, q_LOS, κ_ext)")
print(f"  MCMC: {N_WALKERS} walkers × {N_STEPS} steps ({N_BURN} burn-in)")

# =============================================================================
# Load Catalog and Create Train/Hold-Out Split
# =============================================================================
print("\n[1/9] Loading Tier-1+2 clusters and creating train/hold-out split...")

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
print("\n[2/9] Building and caching baryon profiles...")

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
    
    # Store in cache (spherical projection, will apply geometry in kernel)
    baryon_cache[name] = {
        'r_3d': r_3d,
        'rho_total': rho_total,
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
# Prediction Function with Geometry
# =============================================================================

def predict_einstein_radius_with_geometry(cluster_name, A_c, q_plane, q_LOS, kappa_ext):
    """
    Predict Einstein radius with triaxial geometry.
    
    Parameters
    ----------
    cluster_name : str
        Cluster name
    A_c : float
        Coherence amplitude
    q_plane : float
        In-plane axis ratio (0.6-1.4)
    q_LOS : float
        Line-of-sight axis ratio (0.6-1.6)
    kappa_ext : float
        External convergence sheet
        
    Returns
    -------
    theta_E_pred : float
        Predicted Einstein radius [arcsec]
    """
    cache = baryon_cache[cluster_name]
    
    # Apply triaxial projection
    Sigma_baryon = project_to_surface_density(
        cache['r_3d'], cache['rho_total'], cache['R_grid_2d'], 
        q_plane, q_LOS
    )
    
    # Apply Sigma-Gravity kernel
    Sigma_eff, _, _ = convolve_sigma_with_kernel(
        Sigma_baryon, cache['R_grid_2d'], 
        FIXED_PARAMS['ell0'], FIXED_PARAMS['p'], FIXED_PARAMS['ncoh'], A_c,
        emphasize_interior=True, use_fft=True
    )
    
    # Add external sheet
    if abs(kappa_ext) > 1e-6:
        Sigma_crit = cosmo.critical_surface_density(cache['z_lens'], cache['z_source'])
        Sigma_eff += kappa_ext * Sigma_crit
    
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
# MCMC Parameter Structure
# =============================================================================
# Params = [A_c, q_plane_1, q_LOS_1, kappa_ext_1, q_plane_2, ...]
#
# Indices:
#   0: A_c (global)
#   1+3*i: q_plane for cluster i
#   2+3*i: q_LOS for cluster i
#   3+3*i: kappa_ext for cluster i

n_clusters_train = len(train_clusters)
n_params_total = 1 + 3 * n_clusters_train

cluster_names_train = train_clusters['cluster_name'].values

print(f"\n[3/9] Parameter structure:")
print(f"  Total parameters: {n_params_total}")
print(f"  - 1 global (A_c)")
print(f"  - {3*n_clusters_train} nuisance ({n_clusters_train} clusters × 3 params each)")

# =============================================================================
# Priors
# =============================================================================

def log_prior(theta):
    """
    Log-prior for all parameters.
    
    Parameters
    ----------
    theta : array
        [A_c, q_plane_1, q_LOS_1, kappa_ext_1, q_plane_2, ...]
    """
    A_c = theta[0]
    
    # A_c: flat prior [10, 25]
    if A_c < 10.0 or A_c > 25.0:
        return -np.inf
    
    log_p = 0.0
    
    # Per-cluster nuisance parameters
    for i in range(n_clusters_train):
        q_plane = theta[1 + 3*i]
        q_LOS = theta[2 + 3*i]
        kappa_ext = theta[3 + 3*i]
        
        # q_plane: Normal(1.0, 0.15), truncated [0.6, 1.4]
        if q_plane < 0.6 or q_plane > 1.4:
            return -np.inf
        log_p += norm.logpdf(q_plane, 1.0, 0.15)
        
        # q_LOS: uniform-derived, effective range [0.6, 1.6]
        if q_LOS < 0.6 or q_LOS > 1.6:
            return -np.inf
        # Flat in this range (uniform cos(θ) translates to approx uniform q)
        
        # kappa_ext: Normal(0.0, 0.03), soft bounds [-0.15, 0.15]
        if abs(kappa_ext) > 0.15:
            return -np.inf
        log_p += norm.logpdf(kappa_ext, 0.0, 0.03)
    
    return log_p

# =============================================================================
# Likelihood
# =============================================================================

def log_likelihood(theta):
    """
    Log-likelihood: chi-squared on Einstein radii.
    
    L = -0.5 * Σ [(θ_E_obs - θ_E_model) / σ_θ]²
    """
    A_c = theta[0]
    
    chi2 = 0.0
    
    for i, cluster_name in enumerate(cluster_names_train):
        q_plane = theta[1 + 3*i]
        q_LOS = theta[2 + 3*i]
        kappa_ext = theta[3 + 3*i]
        
        # Predict
        try:
            theta_E_pred = predict_einstein_radius_with_geometry(
                cluster_name, A_c, q_plane, q_LOS, kappa_ext
            )
        except Exception as e:
            print(f"    Error predicting {cluster_name}: {e}")
            return -np.inf
        
        if not np.isfinite(theta_E_pred):
            return -np.inf
        
        # Observed
        cache = baryon_cache[cluster_name]
        theta_E_obs = cache['theta_E_obs']
        theta_E_err = cache['theta_E_err']
        
        # Chi-squared contribution
        chi2 += ((theta_E_obs - theta_E_pred) / theta_E_err)**2
    
    return -0.5 * chi2

def log_probability(theta):
    """Combined log-posterior."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# =============================================================================
# Initialize Walkers
# =============================================================================
print("\n[4/9] Initializing walkers...")

# Starting point near expected values
theta_init = np.zeros(n_params_total)
theta_init[0] = 16.5  # A_c

for i in range(n_clusters_train):
    theta_init[1 + 3*i] = 1.0   # q_plane
    theta_init[2 + 3*i] = 1.0   # q_LOS
    theta_init[3 + 3*i] = 0.0   # kappa_ext

# Perturb walkers
pos = theta_init + 1e-3 * np.random.randn(N_WALKERS, n_params_total)

# Ensure priors are satisfied
for j in range(N_WALKERS):
    pos[j, 0] = np.clip(pos[j, 0], 12.0, 20.0)  # A_c
    for i in range(n_clusters_train):
        pos[j, 1+3*i] = np.clip(pos[j, 1+3*i], 0.7, 1.3)  # q_plane
        pos[j, 2+3*i] = np.clip(pos[j, 2+3*i], 0.7, 1.3)  # q_LOS
        pos[j, 3+3*i] = np.clip(pos[j, 3+3*i], -0.05, 0.05)  # kappa_ext

print(f"  Initialized {N_WALKERS} walkers around:")
print(f"    A_c = {theta_init[0]:.2f}")
print(f"    q_plane = {theta_init[1]:.2f}, q_LOS = {theta_init[2]:.2f}, κ_ext = {theta_init[3]:.3f}")

# =============================================================================
# Run MCMC
# =============================================================================
print(f"\n[5/9] Running MCMC ({N_STEPS} steps)...")
print("  This may take 30-60 minutes...\n")

sampler = emcee.EnsembleSampler(N_WALKERS, n_params_total, log_probability)

start_time = time.time()
sampler.run_mcmc(pos, N_STEPS, progress=True)
end_time = time.time()

print(f"\n  MCMC completed in {(end_time - start_time)/60:.1f} minutes")

# =============================================================================
# Analyze Chains
# =============================================================================
print("\n[6/9] Analyzing chains...")

# Acceptance fraction
acc_frac = np.mean(sampler.acceptance_fraction)
print(f"  Mean acceptance fraction: {acc_frac:.3f}")

if acc_frac < 0.2 or acc_frac > 0.6:
    print("  ⚠️  WARNING: Acceptance fraction outside healthy range [0.2, 0.6]")
else:
    print("  ✅ Acceptance fraction healthy")

# Discard burn-in
samples = sampler.get_chain(discard=N_BURN, flat=False)  # shape: (steps, walkers, params)
flat_samples = sampler.get_chain(discard=N_BURN, flat=True)  # shape: (steps*walkers, params)

print(f"  Samples after burn-in: {flat_samples.shape[0]} (from {N_WALKERS} walkers × {N_STEPS - N_BURN} steps)")

# Parameter posteriors (16th, 50th, 84th percentiles)
percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)

print("\n  Posterior summaries:")
print(f"    A_c = {percentiles[1,0]:.3f} (+{percentiles[2,0]-percentiles[1,0]:.3f}, -{percentiles[1,0]-percentiles[0,0]:.3f})")

for i, name in enumerate(cluster_names_train):
    q_p_med = percentiles[1, 1+3*i]
    q_l_med = percentiles[1, 2+3*i]
    k_med = percentiles[1, 3+3*i]
    print(f"    {name}: q_plane={q_p_med:.3f}, q_LOS={q_l_med:.3f}, κ_ext={k_med:+.3f}")

# =============================================================================
# Evaluate on Train and Hold-Out Sets
# =============================================================================
print("\n[7/9] Evaluating model on train and hold-out sets...")

# Use median parameters
theta_best = percentiles[1, :]
A_c_best = theta_best[0]

# Train set
print("\n  TRAIN SET:")
train_results = []

for i, cluster_name in enumerate(cluster_names_train):
    cache = baryon_cache[cluster_name]
    q_plane = theta_best[1 + 3*i]
    q_LOS = theta_best[2 + 3*i]
    kappa_ext = theta_best[3 + 3*i]
    
    theta_E_pred = predict_einstein_radius_with_geometry(cluster_name, A_c_best, q_plane, q_LOS, kappa_ext)
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
train_dof = len(train_results) - 1  # Minus 1 global parameter
print(f"\n  Train χ² = {train_chi2_total:.2f}, d.o.f. = {train_dof}, χ²/d.o.f. = {train_chi2_total/train_dof:.2f}")

# Hold-out set (need to build baryon profiles)
print("\n  HOLD-OUT SET (BLIND):")
holdout_results = []

for idx, cluster in holdout_clusters.iterrows():
    name = cluster['cluster_name']
    
    # Build baryon profile (not cached)
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
    
    # Temporarily add to cache
    baryon_cache[name] = {
        'r_3d': r_3d,
        'rho_total': rho_total,
        'R_grid_2d': R_grid_2d,
        'R_max': R_max,
        'nx': nx,
        'z_lens': cluster['z_lens'],
        'z_source': cluster['z_source'],
        'theta_E_obs': cluster['theta_E_obs_arcsec'],
        'theta_E_err': cluster['theta_E_err_arcsec']
    }
    
    # Predict with spherical geometry (no geometry fitted for hold-out)
    theta_E_pred = predict_einstein_radius_with_geometry(name, A_c_best, 1.0, 1.0, 0.0)
    
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
print("\n[8/9] Saving results...")

# Save chains
np.save(OUTPUT_DIR / 'chains.npy', samples)
np.save(OUTPUT_DIR / 'flat_samples.npy', flat_samples)

# Save posterior summaries
with open(OUTPUT_DIR / 'posterior_summary.txt', 'w') as f:
    f.write("Tier-1+2 Hierarchical MCMC with Geometry - Posterior Summary\n")
    f.write("="*70 + "\n\n")
    f.write(f"A_c = {percentiles[1,0]:.3f} (+{percentiles[2,0]-percentiles[1,0]:.3f}, -{percentiles[1,0]-percentiles[0,0]:.3f})\n\n")
    f.write("Per-cluster parameters:\n")
    for i, name in enumerate(cluster_names_train):
        f.write(f"  {name}:\n")
        f.write(f"    q_plane = {percentiles[1,1+3*i]:.3f} (+{percentiles[2,1+3*i]-percentiles[1,1+3*i]:.3f}, -{percentiles[1,1+3*i]-percentiles[0,1+3*i]:.3f})\n")
        f.write(f"    q_LOS   = {percentiles[1,2+3*i]:.3f} (+{percentiles[2,2+3*i]-percentiles[1,2+3*i]:.3f}, -{percentiles[1,2+3*i]-percentiles[0,2+3*i]:.3f})\n")
        f.write(f"    κ_ext   = {percentiles[1,3+3*i]:+.3f} (+{percentiles[2,3+3*i]-percentiles[1,3+3*i]:.3f}, -{percentiles[1,3+3*i]-percentiles[0,3+3*i]:.3f})\n")

# Save train/holdout results
train_df = pd.DataFrame(train_results)
holdout_df = pd.DataFrame(holdout_results)
train_df.to_csv(OUTPUT_DIR / 'train_results.csv', index=False)
holdout_df.to_csv(OUTPUT_DIR / 'holdout_results.csv', index=False)

print(f"  Saved to {OUTPUT_DIR}")

# =============================================================================
# Generate Plots
# =============================================================================
print("\n[9/9] Generating plots...")

# Corner plot (A_c and first cluster's parameters as example)
fig = corner.corner(
    flat_samples[:, :4],
    labels=['$A_c$', '$q_{\\rm plane}$ (MACS0416)', '$q_{\\rm LOS}$ (MACS0416)', '$\\kappa_{\\rm ext}$ (MACS0416)'],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.3f'
)
fig.savefig(OUTPUT_DIR / 'corner_plot_sample.png', dpi=150, bbox_inches='tight')
plt.close()

# Predicted vs Observed
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Train set
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

# Hold-out set
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

print(f"  Plots saved to {OUTPUT_DIR}")

print("\n" + "="*70)
print("TIER-1+2 CALIBRATION COMPLETE")
print("="*70)
print(f"\nFinal Results:")
print(f"  A_c = {percentiles[1,0]:.3f} (+{percentiles[2,0]-percentiles[1,0]:.3f}, -{percentiles[1,0]-percentiles[0,0]:.3f})")
print(f"  Train χ²/d.o.f. = {train_chi2_total/train_dof:.2f}")
print(f"  Hold-out χ²/d.o.f. = {holdout_chi2_total/holdout_dof:.2f}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("="*70 + "\n")
