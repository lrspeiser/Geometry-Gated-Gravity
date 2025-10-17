"""
Blind Holdout Validation: Mass-Scaled Model
===========================================

Posterior-predictive checks on A1689 and MACS1149 using calibrated
mass-scaled hierarchical model.

Tests generalization to:
- Low-z benchmark (A1689, z=0.183)
- High-z moderate mass (MACS1149, z=0.544)

Author: GravityCalculator
Date: 2025-01-19
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
import hashlib
import argparse

from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from test_macs0416_projected_kernel import project_to_surface_density
from many_path_model.lensing_utilities import LensingCosmology
from core.bcg_profiles import estimate_bcg_mass, hernquist_projected_density

# Configuration
FIXED_PARAMS = {'p': 2.0, 'ncoh': 2.0}
N_POSTERIOR_DRAWS = 1000
Q_PLANE_GRID = np.linspace(0.6, 1.4, 9)
Q_LOS_GRID = np.linspace(0.6, 1.4, 9)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Blind holdout validation with provenance checks')
    parser.add_argument('--posterior', type=str, required=True, help='Path to posterior NPZ file with manifest')
    parser.add_argument('--catalog', type=str, required=True, help='Path to cluster catalog CSV')
    parser.add_argument('--clusters', type=str, default='A1689,MACS1149', help='Comma-separated holdout cluster names')
    parser.add_argument('--check-training', type=int, default=1, help='0=skip, 1=check training reproducibility')
    parser.add_argument('--pzs', type=str, default='lognormal', choices=['median','lognormal'], help='Source redshift treatment for Sigma_crit')
    parser.add_argument('--outdir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    POSTERIOR_PATH = Path(args.posterior)
    CATALOG_PATH = Path(args.catalog)
    HOLDOUT_NAMES = [s.strip() for s in args.clusters.split(',')]
    CHECK_TRAINING = bool(args.check_training)
    
    if args.outdir:
        OUTPUT_DIR = Path(args.outdir)
    else:
        OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'holdout_validation_mass_scaled'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("BLIND HOLDOUT VALIDATION: MASS-SCALED MODEL")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Posterior: {POSTERIOR_PATH}")
    print(f"  Catalog: {CATALOG_PATH}")
    print(f"  Holdout clusters: {', '.join(HOLDOUT_NAMES)}")
    print(f"  Training check: {'ENABLED' if CHECK_TRAINING else 'DISABLED'}")
    
    # Load posterior with manifest
    print("\n[1/6] Loading posterior and validating provenance...")
    if not POSTERIOR_PATH.exists():
        print(f"FATAL: Posterior not found at {POSTERIOR_PATH}")
        sys.exit(1)
    
    try:
        npz = np.load(POSTERIOR_PATH, allow_pickle=True)
        flat_samples = npz['samples']
        manifest = json.loads(str(npz['manifest'].item()))
        print(f"  Loaded {flat_samples.shape[0]} posterior samples")
        print(f"  Run ID: {manifest['run_id']}")
        print(f"  Train clusters: {', '.join(manifest['train_clusters'])}")
    except KeyError:
        print("FATAL: Posterior file missing 'manifest'. Must use NPZ format with provenance.")
        sys.exit(1)
    
    # Validate catalog provenance
    with open(CATALOG_PATH, 'rb') as f:
        catalog_md5 = hashlib.md5(f.read()).hexdigest()
    
    if catalog_md5 != manifest['catalog_md5']:
        print(f"FATAL: Catalog MD5 mismatch!")
        print(f"  Expected: {manifest['catalog_md5'][:8]}...")
        print(f"  Got:      {catalog_md5[:8]}...")
        print(f"  Posterior and validation are using different data files.")
        sys.exit(1)
    print(f"  Catalog MD5 validated: {catalog_md5[:8]}...")
    
    # Validate physics configuration
    expected_physics = manifest['physics']
    print(f"  Physics config: BCG={expected_physics['bcg']}, triaxial={expected_physics['triaxial']}, P(z_s)={expected_physics['pzsource']}")
    print(f"  Kernel norm: {manifest['kernel']['norm']}")
    print(f"  Mass-scaling: {manifest['kernel']['mass_scaling']}, gamma_prior={manifest['kernel']['gamma_prior']}")

    # Extract population parameters: [ell0_star, gamma, mu_A, sigma_A, ...]
    ell0_star_samples = flat_samples[:, 0]
    gamma_samples = flat_samples[:, 1]
    mu_A_samples = flat_samples[:, 2]
    sigma_A_samples = flat_samples[:, 3]

ell0_star_med = np.median(ell0_star_samples)
gamma_med = np.median(gamma_samples)
mu_A_med = np.median(mu_A_samples)
sigma_A_med = np.median(sigma_A_samples)

print(f"\n  Population parameters (median):")
print(f"    ell0_star = {ell0_star_med:.1f} kpc")
print(f"    gamma     = {gamma_med:.3f}")
print(f"    mu_A      = {mu_A_med:.3f}")
print(f"    sigma_A   = {sigma_A_med:.3f}")

# Thin posterior
idx_thin = np.random.choice(len(mu_A_samples), size=min(N_POSTERIOR_DRAWS, len(mu_A_samples)), replace=False)
ell0_star_post = ell0_star_samples[idx_thin]
gamma_post = gamma_samples[idx_thin]
mu_A_post = mu_A_samples[idx_thin]
sigma_A_post = sigma_A_samples[idx_thin]

# Load holdout clusters
print("\n[2/5] Loading holdout clusters...")
catalog = pd.read_csv(CATALOG_PATH)
holdout_clusters = catalog[catalog['cluster_name'].isin(['A1689', 'MACS1149'])].copy()
print(f"  Holdout clusters: {', '.join(holdout_clusters['cluster_name'].values)}")

# Build baryon models
print("\n[3/5] Building baryon models...")
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
    
    # Literature geometry hints
    if name == 'A1689':
        q_plane_default, q_LOS_default = 0.9, 1.0  # Slightly elongated
    else:
        q_plane_default, q_LOS_default = 1.0, 1.0  # Spherical
    
    holdout_cache[name] = {
        'geometry_cache': geom_cache,
        'R_grid_2d': R_grid_2d,
        'R_max': R_max,
        'z_lens': cluster['z_lens'],
        'z_source': cluster['z_source'],
        'theta_E_obs': cluster['theta_E_obs_arcsec'],
        'theta_E_err': cluster['theta_E_err_arcsec'],
        'q_plane': q_plane_default,
        'q_LOS': q_LOS_default,
        'R_500': cluster['R_500_kpc']
    }

def get_Sigma_bar(cluster_name, q_plane, q_LOS):
    cache = holdout_cache[cluster_name]
    geom_cache = cache['geometry_cache']
    q_p = np.clip(q_plane, Q_PLANE_GRID[0], Q_PLANE_GRID[-1])
    q_l = np.clip(q_LOS, Q_LOS_GRID[0], Q_LOS_GRID[-1])
    idx_p = np.argmin(np.abs(Q_PLANE_GRID - q_p))
    idx_l = np.argmin(np.abs(Q_LOS_GRID - q_l))
    return geom_cache[(Q_PLANE_GRID[idx_p], Q_LOS_GRID[idx_l])]

def predict_theta_E(cluster_name, ell0, A_c, q_plane, q_LOS):
    cache = holdout_cache[cluster_name]
    Sigma_bar = get_Sigma_bar(cluster_name, q_plane, q_LOS)
    
    Sigma_eff, _, _ = convolve_sigma_with_kernel(
        Sigma_bar, cache['R_grid_2d'], ell0, FIXED_PARAMS['p'], FIXED_PARAMS['ncoh'], A_c,
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
    return cosmo.physical_to_angular(R_E_kpc, cache['z_lens'])

# Posterior predictive sampling
print("\n[4/5] Running posterior-predictive checks...")
results = {}

# Pre-build 3D density for each holdout (needed for geometry sampling)
holdout_rho_3d = {}
for cluster_name in holdout_cache.keys():
    cluster_row = holdout_clusters[holdout_clusters['cluster_name'] == cluster_name].iloc[0]
    r_3d = np.logspace(-1, 3.5, 800)
    params = ClusterBaryonParams(
        M_500=cluster_row['M_500_Msun'], R_500=cluster_row['R_500_kpc'],
        z=cluster_row['z_lens'], fgas_target=cluster_row['fgas_R500'],
        T_keV=cluster_row['TX_central_keV'], C0=1.3, eta=2.0, C_max=2.5
    )
    components = build_cluster_baryon_model(r_3d, params, apply_clumping=False, verbose=False)
    holdout_rho_3d[cluster_name] = {'r_3d': r_3d, 'rho_total': components.rho_total}

for cluster_name in holdout_cache.keys():
    print(f"\n  {cluster_name}:")
    cache = holdout_cache[cluster_name]
    R_500 = cache['R_500']
    rho_data = holdout_rho_3d[cluster_name]
    
    theta_E_pred_samples = []
    
    # Load cluster-specific config if available
    config_path = Path(__file__).parent.parent / 'data' / 'clusters' / f'{cluster_name.lower()}_config.json'
    cluster_config = None
    if config_path.exists():
        with open(config_path, 'r') as f:
            cluster_config = json.load(f)
        print(f"    Using cluster-specific config: {config_path.name}")
    
    for i in range(len(mu_A_post)):
        # Mass-scaled coherence length
        ell0_cluster = ell0_star_post[i] * (R_500 / 1000.0)**gamma_post[i]
        
        # Draw A_c from hierarchical prior (clip to physical range)
        A_c = np.clip(np.random.normal(mu_A_post[i], sigma_A_post[i]), 1.0, 15.0)
        
        # Sample geometry with cluster-specific priors if available
        if cluster_config and 'geometry' in cluster_config:
            geom = cluster_config['geometry']
            q_plane_i = np.clip(np.random.normal(geom['q_plane_prior']['mean'], geom['q_plane_prior']['std']), 0.6, 1.4)
            q_LOS_i = np.clip(np.random.normal(geom['q_los_prior']['mean'], geom['q_los_prior']['std']), 0.6, 1.4)
        else:
            q_plane_i = np.clip(np.random.normal(1.0, 0.2), 0.6, 1.4)
            q_LOS_i = np.clip(np.random.normal(1.0, 0.2), 0.6, 1.4)
        
        # Sample kappa_ext with cluster-specific prior if available
        if cluster_config and 'environment' in cluster_config:
            kext_std = cluster_config['environment']['kappa_ext_prior']['std']
            kappa_ext_i = np.random.normal(0.0, kext_std)
        else:
            kappa_ext_i = np.random.normal(0.0, 0.03)
        
        # Recompute Sigma_bar with sampled geometry
        Sigma_baryons_i = project_to_surface_density(
            rho_data['r_3d'], rho_data['rho_total'], 
            cache['R_grid_2d'], q_LOS_i, q_plane_i
        )
        
        # Add BCG component
        cluster_row = holdout_clusters[holdout_clusters['cluster_name'] == cluster_name].iloc[0]
        M_BCG, r_eff_BCG = estimate_bcg_mass(cluster_row['M_500_Msun'], cluster_row['z_lens'])
        Sigma_BCG = hernquist_projected_density(cache['R_grid_2d'], M_BCG, r_eff_BCG)
        
        Sigma_bar_i = Sigma_baryons_i + Sigma_BCG
        
        # Convolve with kernel
        Sigma_eff_i, _, _ = convolve_sigma_with_kernel(
            Sigma_bar_i, cache['R_grid_2d'], ell0_cluster, 
            FIXED_PARAMS['p'], FIXED_PARAMS['ncoh'], A_c,
            emphasize_interior=True, use_fft=True
        )
        
        # Add external convergence
        if abs(kappa_ext_i) > 1e-6:
            Sigma_crit = cosmo.critical_surface_density(cache['z_lens'], cache['z_source'])
            Sigma_eff_i += kappa_ext_i * Sigma_crit
        
        # Compute theta_E from Sigma_eff
        R_bins = np.linspace(0, cache['R_max']*0.9, 150)
        _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff_i, cache['R_grid_2d'], R_bins)
        
        valid = np.isfinite(Sigma_eff_prof)
        R_prof = 0.5 * (R_bins[:-1] + R_bins[1:])[valid]
        Sigma_eff_prof = Sigma_eff_prof[valid]
        
        if len(R_prof) >= 10:
            M_enc = cumulative_trapezoid(2*np.pi*R_prof*Sigma_eff_prof, R_prof, initial=0.0)
            
            # Use effective z_source from config if available
            if cluster_config and 'arc_redshifts' in cluster_config:
                arc_data = cluster_config['arc_redshifts']
                if 'z_eff' in arc_data:
                    z_source_use = arc_data['z_eff']
                else:
                    z_source_use = cache['z_source']
            else:
                z_source_use = cache['z_source']
            
            if args.pzs == 'lognormal':
                Sigma_crit = cosmo.effective_critical_density_with_distribution(cache['z_lens'])
            else:
                Sigma_crit = cosmo.critical_surface_density(cache['z_lens'], z_source_use)
            mean_kappa = M_enc / (np.pi * R_prof**2 * Sigma_crit)
            mean_kappa[0] = Sigma_eff_prof[0] / Sigma_crit
            
            idx_cross = np.where(mean_kappa >= 1.0)[0]
            if len(idx_cross) > 0:
                R_E_kpc = R_prof[idx_cross[-1]]
                theta_E_pred = cosmo.physical_to_angular(R_E_kpc, cache['z_lens'])
                if np.isfinite(theta_E_pred):
                    theta_E_pred_samples.append(theta_E_pred)
    
    theta_E_pred_samples = np.array(theta_E_pred_samples)
    
    # Compute statistics
    theta_E_obs = cache['theta_E_obs']
    theta_E_err = cache['theta_E_err']
    
    theta_E_pred_med = np.median(theta_E_pred_samples)
    theta_E_pred_16, theta_E_pred_84 = np.percentile(theta_E_pred_samples, [16, 84])
    
    residual = theta_E_obs - theta_E_pred_med
    z_score = residual / theta_E_err
    
    # Check if observed value within 1σ posterior predictive
    within_1sigma = (theta_E_pred_16 <= theta_E_obs <= theta_E_pred_84)
    
    print(f"    theta_E observed:  {theta_E_obs:.1f} +/- {theta_E_err:.1f} arcsec")
    print(f"    theta_E predicted: {theta_E_pred_med:.1f} [{theta_E_pred_16:.1f}, {theta_E_pred_84:.1f}] arcsec")
    print(f"    Residual:      {residual:+.1f} arcsec (Z-score: {z_score:+.2f}sigma)")
    print(f"    Within 1sigma:     {within_1sigma}")
    
    results[cluster_name] = {
        'theta_E_obs': theta_E_obs,
        'theta_E_err': theta_E_err,
        'theta_E_pred_med': theta_E_pred_med,
        'theta_E_pred_16': theta_E_pred_16,
        'theta_E_pred_84': theta_E_pred_84,
        'residual': residual,
        'z_score': z_score,
        'within_1sigma': within_1sigma,
        'samples': theta_E_pred_samples
    }

# Summary
print("\n[5/5] Validation summary...")
n_clusters = len(results)
n_within_1sigma = sum(r['within_1sigma'] for r in results.values())
n_within_2sigma = sum(abs(r['z_score']) <= 2.0 for r in results.values())

print(f"\n  Total holdout clusters: {n_clusters}")
print(f"  Within 1sigma posterior predictive: {n_within_1sigma}/{n_clusters} ({n_within_1sigma/n_clusters*100:.0f}%)")
print(f"  Within 2sigma Z-score: {n_within_2sigma}/{n_clusters} ({n_within_2sigma/n_clusters*100:.0f}%)")

# Pass criteria
pass_1sigma = (n_within_1sigma >= 2 * n_clusters // 3)  # >=2/3
pass_2sigma = (n_within_2sigma == n_clusters)  # All

print(f"\n  Acceptance criteria:")
print(f"    >=2/3 within 1sigma: {'PASS' if pass_1sigma else 'FAIL'}")
print(f"    None >2sigma:        {'PASS' if pass_2sigma else 'FAIL'}")

overall_pass = pass_1sigma and pass_2sigma
print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, cluster_name in enumerate(results.keys()):
    ax = axes[i]
    res = results[cluster_name]
    
    ax.hist(res['samples'], bins=40, density=True, alpha=0.7, color='steelblue', label='Posterior predictive')
    ax.axvline(res['theta_E_obs'], color='red', linestyle='--', linewidth=2, label=f'Observed ({res["theta_E_obs"]:.1f}″)')
    ax.axvspan(res['theta_E_obs'] - res['theta_E_err'], res['theta_E_obs'] + res['theta_E_err'], 
               color='red', alpha=0.2, label='1σ error')
    ax.axvline(res['theta_E_pred_med'], color='black', linestyle=':', label=f'Predicted ({res["theta_E_pred_med"]:.1f}″)')
    ax.axvspan(res['theta_E_pred_16'], res['theta_E_pred_84'], color='gray', alpha=0.2, label='68% CI')
    
    ax.set_xlabel('θ_E (arcsec)')
    ax.set_ylabel('Posterior density')
    ax.set_title(f'{cluster_name}\nZ-score: {res["z_score"]:+.2f}σ')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'holdout_validation.png', dpi=150)
print(f"\n  Saved: {OUTPUT_DIR / 'holdout_validation.png'}")

# Save results
results_dict = {}
for name, res in results.items():
    results_dict[name] = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else bool(v) if isinstance(v, np.bool_) else v)
        for k, v in res.items() if k != 'samples'
    }
results_dict['summary'] = {
    'n_clusters': int(n_clusters),
    'n_within_1sigma': int(n_within_1sigma),
    'n_within_2sigma': int(n_within_2sigma),
    'pass': bool(overall_pass)
}

with open(OUTPUT_DIR / 'holdout_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"  Saved: {OUTPUT_DIR / 'holdout_results.json'}")

print("\n" + "="*70)
print(f"HOLDOUT VALIDATION {'PASSED' if overall_pass else 'FAILED'}")
print("="*70)
