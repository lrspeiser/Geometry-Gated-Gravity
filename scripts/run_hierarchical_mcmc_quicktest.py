"""
Quick Test Version of Hierarchical MCMC Calibration
====================================================

Reduced settings for fast pipeline verification (~10-15 minutes):
- Fewer walkers (16 instead of 32)
- Fewer steps (200 production + 100 burn-in)
- Lower resolution grids

Use this to verify the pipeline works before committing to the full run.

For production, use: run_hierarchical_mcmc_calibration.py
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

# Import baryon model
from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams

# Import Sigma-Gravity kernel
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average

# Import test helpers
from test_macs0416_projected_kernel import project_to_surface_density

# Import cosmology
from many_path_model.lensing_utilities import LensingCosmology

print("="*70)
print("HIERARCHICAL MCMC CALIBRATION - QUICK TEST")
print("="*70)

# =============================================================================
# Configuration - REDUCED FOR SPEED
# =============================================================================

FIXED_PARAMS = {
    'ell0': 200.0,
    'p': 2.0,
    'ncoh': 2.0
}

FIT_ELL0 = False

# REDUCED SETTINGS FOR QUICK TEST
N_WALKERS = 16  # Reduced from 32
N_STEPS = 200   # Reduced from 2000
N_BURN = 100    # Reduced from 500

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'hierarchical_mcmc_quicktest')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n⚠️  QUICK TEST MODE - Reduced settings for fast verification")
print(f"   For production run, use: run_hierarchical_mcmc_calibration.py")
print(f"\nConfiguration:")
print(f"  Fixed: ℓ_0 = {FIXED_PARAMS['ell0']:.0f} kpc, p = {FIXED_PARAMS['p']:.1f}, n_coh = {FIXED_PARAMS['ncoh']:.1f}")
print(f"  Free parameters: A_c" + (", ℓ_0" if FIT_ELL0 else ""))
print(f"  MCMC: {N_WALKERS} walkers × {N_STEPS} steps ({N_BURN} burn-in)")
print(f"  Expected runtime: ~10-15 minutes")

# =============================================================================
# Load Catalog and Create Train/Hold-out Split
# =============================================================================
print("\n[1/8] Loading cluster catalog and creating train/hold-out split...")

catalog_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'clusters', 'master_catalog.csv')
catalog = pd.read_csv(catalog_path)

tier1_clusters = catalog[catalog['tier'] == 1].copy()

tier1_sorted = tier1_clusters.sort_values('z_lens')
holdout_indices = [0, -1]
holdout_clusters = tier1_sorted.iloc[holdout_indices].copy()
train_clusters = tier1_sorted.drop(tier1_sorted.iloc[holdout_indices].index).copy()

print(f"\n  Train set: {len(train_clusters)} clusters")
for idx, row in train_clusters.iterrows():
    print(f"    - {row['cluster_name']}: z={row['z_lens']:.3f}, θ_E={row['theta_E_obs_arcsec']:.1f}\"")

print(f"\n  Hold-out set: {len(holdout_clusters)} clusters (BLIND)")
for idx, row in holdout_clusters.iterrows():
    print(f"    - {row['cluster_name']}: z={row['z_lens']:.3f}, θ_E={row['theta_E_obs_arcsec']:.1f}\"")

# =============================================================================
# Prediction Function - OPTIMIZED FOR SPEED
# =============================================================================

def predict_einstein_radius_fast(cluster_params, kernel_params, cache=None):
    """Fast Einstein radius prediction with caching"""
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
    kappa_ext = cluster_params.get('kappa_ext', 0.0)
    
    cache_key = f"{cluster_params['name']}"
    
    if cache is not None and cache_key in cache:
        Sigma_baryon = cache[cache_key]['Sigma_baryon']
        R_grid_2d = cache[cache_key]['R_grid_2d']
        R_max = cache[cache_key]['R_max']
    else:
        # REDUCED RESOLUTION FOR SPEED
        r_3d = np.logspace(-1, 3.5, 800)  # Reduced from 1200
        
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
        
        # REDUCED GRID SIZE FOR SPEED
        nx, ny = 128, 128  # Reduced from 256
        R_max = min(2500.0, R_500 * 2.2)
        x = np.linspace(-R_max, R_max, nx)
        y = np.linspace(-R_max, R_max, ny)
        X, Y = np.meshgrid(x, y)
        R_grid_2d = np.sqrt(X**2 + Y**2)
        
        Sigma_baryon = project_to_surface_density(r_3d, rho_total, R_grid_2d, 1.0, 1.0)
        
        if cache is not None:
            cache[cache_key] = {
                'Sigma_baryon': Sigma_baryon,
                'R_grid_2d': R_grid_2d,
                'R_max': R_max
            }
    
    Sigma_eff_2d, _, _ = convolve_sigma_with_kernel(
        Sigma_baryon, R_grid_2d, ell0, p, ncoh, A_c,
        emphasize_interior=True, use_fft=True
    )
    
    R_bins = np.linspace(0, R_max, 151)  # Reduced from 201
    R_prof, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff_2d, R_grid_2d, R_bins)
    
    valid_mask = ~np.isnan(Sigma_eff_prof)
    R_prof = R_prof[valid_mask]
    Sigma_eff_prof = Sigma_eff_prof[valid_mask]
    
    if abs(kappa_ext) > 1e-6:
        Sigma_eff_prof = Sigma_eff_prof / (1.0 - kappa_ext)
    
    cosmo = LensingCosmology()
    Sigma_crit = cosmo.critical_surface_density(z_lens, z_src)
    kappa_eff = Sigma_eff_prof / Sigma_crit
    
    M_eff_cum = cumulative_trapezoid(2.0 * np.pi * R_prof * Sigma_eff_prof, R_prof, initial=0.0)
    mean_kappa_eff = M_eff_cum / (np.pi * R_prof**2 * Sigma_crit)
    mean_kappa_eff[0] = kappa_eff[0]
    
    idx_E = np.where(mean_kappa_eff >= 1.0)[0]
    
    if len(idx_E) > 0:
        R_E_kpc = R_prof[idx_E[-1]]
        theta_E_pred = cosmo.physical_to_angular(R_E_kpc, z_lens)
    else:
        theta_E_pred = 0.0
    
    return theta_E_pred

BARYON_CACHE = {}

# =============================================================================
# Prior and Likelihood Functions
# =============================================================================

def log_prior(params):
    if FIT_ELL0:
        A_c, ell0 = params
        if not (5.0 < A_c < 30.0):
            return -np.inf
        if not (50.0 < ell0 < 600.0):
            return -np.inf
        log_p = -np.log(ell0)
    else:
        A_c = params[0]
        if not (5.0 < A_c < 30.0):
            return -np.inf
        log_p = 0.0
    
    return log_p

def log_likelihood(params, cluster_data):
    if FIT_ELL0:
        A_c, ell0 = params
    else:
        A_c = params[0]
        ell0 = FIXED_PARAMS['ell0']
    
    kernel_params = {
        'A_c': A_c,
        'ell0': ell0,
        'p': FIXED_PARAMS['p'],
        'ncoh': FIXED_PARAMS['ncoh']
    }
    
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
            'T_keV': row['TX_central_keV'],
            'kappa_ext': 0.0
        }
        
        try:
            theta_E_pred = predict_einstein_radius_fast(cluster_params, kernel_params, cache=BARYON_CACHE)
            
            if theta_E_pred > 0:
                theta_E_obs = row['theta_E_obs_arcsec']
                sigma_obs = row['theta_E_err_arcsec']
                
                residual = (theta_E_pred - theta_E_obs) / sigma_obs
                chi2 += residual**2
                n_valid += 1
        except Exception as e:
            chi2 += 1e6
            continue
    
    log_L = -0.5 * chi2
    return log_L

def log_probability(params, cluster_data):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, cluster_data)
    return lp + ll

# =============================================================================
# Initialize Walkers
# =============================================================================
print("\n[2/8] Initializing MCMC walkers...")

if FIT_ELL0:
    ndim = 2
    p0_center = np.array([16.4, 200.0])
    p0_scatter = np.array([0.5, 20.0])
else:
    ndim = 1
    p0_center = np.array([16.4])
    p0_scatter = np.array([0.5])

p0 = p0_center + p0_scatter * np.random.randn(N_WALKERS, ndim)

if FIT_ELL0:
    p0[:, 0] = np.clip(p0[:, 0], 8.0, 25.0)
    p0[:, 1] = np.clip(p0[:, 1], 80.0, 400.0)
else:
    p0[:, 0] = np.clip(p0[:, 0], 8.0, 25.0)

print(f"  Number of dimensions: {ndim}")
print(f"  Number of walkers: {N_WALKERS}")
print(f"  Initial center: {p0_center}")

# =============================================================================
# Pre-compute Baryon Profiles
# =============================================================================
print("\n[3/8] Pre-computing baryon profiles (cache warmup)...")

for idx, row in train_clusters.iterrows():
    cluster_params = {
        'name': row['cluster_name'],
        'M_500': row['M_500_Msun'],
        'R_500': row['R_500_kpc'],
        'z': row['z_lens'],
        'z_src': row['z_source'],
        'fgas': row['fgas_R500'],
        'T_keV': row['TX_central_keV']
    }
    
    print(f"  Caching {row['cluster_name']}...", end="")
    
    test_kernel = {'A_c': 16.0, 'ell0': 200.0, 'p': 2.0, 'ncoh': 2.0}
    _ = predict_einstein_radius_fast(cluster_params, test_kernel, cache=BARYON_CACHE)
    
    print(" done")

print(f"  Cache populated: {len(BARYON_CACHE)} clusters")

# =============================================================================
# Run MCMC
# =============================================================================
print("\n[4/8] Running MCMC sampling...")
print(f"  Estimated runtime: ~10-15 minutes")

sampler = emcee.EnsembleSampler(
    N_WALKERS, ndim, log_probability,
    args=(train_clusters,)
)

print("\n  Starting burn-in...")
state = sampler.run_mcmc(p0, N_BURN, progress=True)
sampler.reset()

print(f"\n  Burn-in complete. Starting production run...")
sampler.run_mcmc(state, N_STEPS, progress=True)

print("\n  MCMC complete!")
print(f"  Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

# =============================================================================
# Analyze MCMC Results
# =============================================================================
print("\n[5/8] Analyzing MCMC chains...")

samples = sampler.get_chain(flat=True)

if FIT_ELL0:
    param_names = ['A_c', 'ell0']
else:
    param_names = ['A_c']

percentiles = np.percentile(samples, [16, 50, 84], axis=0)
medians = percentiles[1]
lower_err = medians - percentiles[0]
upper_err = percentiles[2] - medians

print("\n  Posterior summary:")
print("  " + "-"*60)
for i, name in enumerate(param_names):
    print(f"  {name}: {medians[i]:.3f} + {upper_err[i]:.3f} - {lower_err[i]:.3f}")
print("  " + "-"*60)

if FIT_ELL0:
    A_c_best = medians[0]
    ell0_best = medians[1]
else:
    A_c_best = medians[0]
    ell0_best = FIXED_PARAMS['ell0']

best_kernel_params = {
    'A_c': A_c_best,
    'ell0': ell0_best,
    'p': FIXED_PARAMS['p'],
    'ncoh': FIXED_PARAMS['ncoh']
}

# =============================================================================
# Evaluate Train and Hold-out Sets
# =============================================================================
print("\n[6/8] Evaluating train and hold-out sets with best-fit parameters...")

def evaluate_clusters(cluster_data, kernel_params, dataset_name):
    results_list = []
    chi2_total = 0.0
    
    print(f"\n  {dataset_name}:")
    print(f"  {'Cluster':<15} {'θ_E Obs':<10} {'θ_E Pred':<10} {'Error':<10} {'χ²':<10}")
    print("  " + "-"*60)
    
    for idx, row in cluster_data.iterrows():
        cluster_params = {
            'name': row['cluster_name'],
            'M_500': row['M_500_Msun'],
            'R_500': row['R_500_kpc'],
            'z': row['z_lens'],
            'z_src': row['z_source'],
            'fgas': row['fgas_R500'],
            'T_keV': row['TX_central_keV'],
            'kappa_ext': 0.0
        }
        
        try:
            theta_E_pred = predict_einstein_radius_fast(cluster_params, kernel_params, cache=BARYON_CACHE)
            
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
                'chi2': chi2
            })
            
            print(f"  {row['cluster_name']:<15} {theta_E_obs:>5.1f}±{sigma_obs:.1f}\"  "
                  f"{theta_E_pred:>6.2f}\"    {error:>+6.2f}\"   {chi2:>6.2f}")
            
        except Exception as e:
            print(f"  {row['cluster_name']:<15} {'FAILED':>10} {str(e)[:20]}")
    
    print("  " + "-"*60)
    
    n_clusters = len(results_list)
    n_params = ndim
    dof = max(n_clusters - n_params, 1)
    chi2_reduced = chi2_total / dof
    
    print(f"\n  {dataset_name} Goodness-of-fit:")
    print(f"    χ² = {chi2_total:.2f}")
    print(f"    d.o.f. = {dof}")
    print(f"    χ²/d.o.f. = {chi2_reduced:.3f}")
    
    return results_list, chi2_reduced

train_results, train_chi2_reduced = evaluate_clusters(train_clusters, best_kernel_params, "TRAIN SET")
holdout_results, holdout_chi2_reduced = evaluate_clusters(holdout_clusters, best_kernel_params, "HOLD-OUT SET (BLIND)")

# =============================================================================
# Generate Plots
# =============================================================================
print("\n[7/8] Generating diagnostic plots...")

# Corner plot
fig = corner.corner(
    samples,
    labels=param_names,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt='.3f',
    truths=medians
)
corner_path = os.path.join(OUTPUT_DIR, 'corner_plot.png')
plt.savefig(corner_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {corner_path}")
plt.close()

# MCMC chains
fig, axes = plt.subplots(ndim, figsize=(10, 3*ndim), sharex=True)
if ndim == 1:
    axes = [axes]

chain = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(chain[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(chain))
    ax.set_ylabel(param_names[i])
    ax.axhline(medians[i], color='r', lw=2)

axes[-1].set_xlabel("Step number")
chains_path = os.path.join(OUTPUT_DIR, 'mcmc_chains.png')
plt.tight_layout()
plt.savefig(chains_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {chains_path}")
plt.close()

# Predicted vs Observed
fig, ax = plt.subplots(1, 1, figsize=(9, 9))

train_obs = [r['theta_E_obs'] for r in train_results]
train_pred = [r['theta_E_pred'] for r in train_results]
ax.errorbar(train_obs, train_pred, 
            yerr=[train_clusters[train_clusters['cluster_name']==r['cluster']]['theta_E_err_arcsec'].values[0] 
                  for r in train_results],
            fmt='o', markersize=10, capsize=5, color='blue', alpha=0.7, label='Train')

holdout_obs = [r['theta_E_obs'] for r in holdout_results]
holdout_pred = [r['theta_E_pred'] for r in holdout_results]
ax.errorbar(holdout_obs, holdout_pred,
            yerr=[holdout_clusters[holdout_clusters['cluster_name']==r['cluster']]['theta_E_err_arcsec'].values[0] 
                  for r in holdout_results],
            fmt='s', markersize=10, capsize=5, color='red', alpha=0.7, label='Hold-out (blind)')

theta_range = [0, max(max(train_obs + holdout_obs), max(train_pred + holdout_pred)) * 1.1]
ax.plot(theta_range, theta_range, 'k--', linewidth=2, alpha=0.5, label='1:1')
ax.fill_between(theta_range, [0.9*x for x in theta_range], [1.1*x for x in theta_range],
                alpha=0.2, color='gray', label='±10%')

for r in train_results:
    ax.text(r['theta_E_obs'] + 0.8, r['theta_E_pred'], r['cluster'], fontsize=8, alpha=0.7)
for r in holdout_results:
    ax.text(r['theta_E_obs'] + 0.8, r['theta_E_pred'], r['cluster'], fontsize=8, alpha=0.7, color='red')

ax.set_xlabel('Observed θ_E (arcsec)', fontsize=13)
ax.set_ylabel('Predicted θ_E (arcsec)', fontsize=13)
ax.set_title(f'Quick Test: MCMC Calibration\n'
             f'A_c = {A_c_best:.3f} ± {(upper_err[0]+lower_err[0])/2:.3f}, '
             f'Train: χ²/d.o.f. = {train_chi2_reduced:.2f}, '
             f'Hold-out: χ²/d.o.f. = {holdout_chi2_reduced:.2f}',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
pred_obs_path = os.path.join(OUTPUT_DIR, 'predicted_vs_observed.png')
plt.savefig(pred_obs_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {pred_obs_path}")
plt.close()

# =============================================================================
# Save Results
# =============================================================================
print("\n[8/8] Saving results...")

samples_path = os.path.join(OUTPUT_DIR, 'posterior_samples.npy')
np.save(samples_path, samples)
print(f"  Saved: {samples_path}")

summary_path = os.path.join(OUTPUT_DIR, 'calibration_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("HIERARCHICAL MCMC CALIBRATION - QUICK TEST RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write("⚠️  WARNING: This is a QUICK TEST with reduced settings\n")
    f.write("   For production results, use: run_hierarchical_mcmc_calibration.py\n\n")
    
    f.write("Posterior Summary:\n")
    f.write("-"*70 + "\n")
    for i, name in enumerate(param_names):
        f.write(f"  {name}: {medians[i]:.4f} + {upper_err[i]:.4f} - {lower_err[i]:.4f}\n")
    f.write(f"\n  Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}\n\n")
    
    f.write("Train Set Performance:\n")
    f.write("-"*70 + "\n")
    f.write(f"  N clusters = {len(train_results)}\n")
    f.write(f"  χ²/d.o.f. = {train_chi2_reduced:.3f}\n\n")
    
    for r in train_results:
        f.write(f"  {r['cluster']:<15} {r['theta_E_obs']:>7.2f}  {r['theta_E_pred']:>7.2f}  "
                f"{r['error']:>+7.2f}  {r['chi2']:>7.2f}\n")
    
    f.write("\nHold-out Set Performance (BLIND):\n")
    f.write("-"*70 + "\n")
    f.write(f"  N clusters = {len(holdout_results)}\n")
    f.write(f"  χ²/d.o.f. = {holdout_chi2_reduced:.3f}\n\n")
    
    for r in holdout_results:
        f.write(f"  {r['cluster']:<15} {r['theta_E_obs']:>7.2f}  {r['theta_E_pred']:>7.2f}  "
                f"{r['error']:>+7.2f}  {r['chi2']:>7.2f}\n")

print(f"  Saved: {summary_path}")

print("\n" + "="*70)
print("QUICK TEST COMPLETE")
print("="*70)
print(f"\nBest-fit: A_c = {A_c_best:.3f} ± {(upper_err[0]+lower_err[0])/2:.3f}")
print(f"Train χ²/d.o.f. = {train_chi2_reduced:.3f}")
print(f"Hold-out χ²/d.o.f. = {holdout_chi2_reduced:.3f}")
print("\n⚠️  REMINDER: This was a quick test with reduced settings")
print("   For publication-quality results, run:")
print("   python scripts/run_hierarchical_mcmc_calibration.py")
print("\nAll results saved to:", OUTPUT_DIR)
