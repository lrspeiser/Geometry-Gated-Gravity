"""
Tier-1+2 MCMC with Geometry - GPU Accelerated
==============================================

Fast version using pre-computed baryon profiles on fixed geometry grid.

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
from scipy.stats import norm
import emcee
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count

# WINDOWS MULTIPROCESSING NOTES:
# 1. Must use if __name__ == '__main__': guard (spawn method)
# 2. Need _init_worker() to populate globals in each worker process
# 3. Ensure N_PROCESSES >= 1 to avoid pool creation errors
# 4. Use encoding='utf-8' when writing files (summary.txt)
# 5. Use raw strings r'' for matplotlib LaTeX labels

# Import baryon model
from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from test_macs0416_projected_kernel import project_to_surface_density
from many_path_model.lensing_utilities import LensingCosmology

# Configuration
FIXED_PARAMS = {'ell0': 200.0, 'p': 2.0, 'ncoh': 2.0}
N_WALKERS = 32
N_STEPS = 1000
N_BURN = 300
N_PROCESSES = max(1, min(8, cpu_count() - 2))
OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'tier12_mcmc_fast'
Q_PLANE_GRID = np.linspace(0.7, 1.3, 7)
Q_LOS_GRID = np.linspace(0.7, 1.3, 7)

# Global state (set in main and in worker initializer)
_baryon_cache = None
_cosmo = None
_cluster_names = None

def _init_worker(baryon_cache, cluster_names):
    """Initializer to populate globals in worker processes (Windows spawn-safe)."""
    global _baryon_cache, _cosmo, _cluster_names
    _baryon_cache = baryon_cache
    _cluster_names = cluster_names
    # Recreate cosmology object in each worker to avoid pickling issues.
    _cosmo = LensingCosmology()

def load_catalog():
    catalog_path = Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv'
    catalog = pd.read_csv(catalog_path)
    tier12 = catalog[catalog['tier'] <= 2].copy()
    train = tier12[tier12['cluster_name'].isin(['MACS0416', 'A2744', 'A370', 'MACS0717', 'RXJ1347', 'CL0024'])].copy()
    holdout = tier12[tier12['cluster_name'].isin(['A1689', 'MACS1149'])].copy()
    return train, holdout

def build_cache(train_clusters):
    print("\\n[2/8] Pre-computing geometry grid...")
    cosmo_local = LensingCosmology()
    cache = {}
    
    for idx, cluster in train_clusters.iterrows():
        name = cluster['cluster_name']
        print(f"  Pre-computing {name} on {len(Q_PLANE_GRID)}×{len(Q_LOS_GRID)} grid...")
        
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
        
        geom_cache = {}
        for q_p in Q_PLANE_GRID:
            for q_l in Q_LOS_GRID:
                Sigma_bar = project_to_surface_density(r_3d, rho_total, R_grid_2d, q_l, q_p)
                geom_cache[(q_p, q_l)] = Sigma_bar
        
        cache[name] = {
            'geometry_cache': geom_cache, 'R_grid_2d': R_grid_2d, 'R_max': R_max, 'nx': nx,
            'z_lens': cluster['z_lens'], 'z_source': cluster['z_source'],
            'theta_E_obs': cluster['theta_E_obs_arcsec'], 'theta_E_err': cluster['theta_E_err_arcsec']
        }
    
    print(f"  Cached {len(cache)} clusters")
    return cache, cosmo_local

def get_Sigma_interp(cluster_name, q_plane, q_LOS):
    cache = _baryon_cache[cluster_name]
    geom_cache = cache['geometry_cache']
    q_p = np.clip(q_plane, Q_PLANE_GRID[0], Q_PLANE_GRID[-1])
    q_l = np.clip(q_LOS, Q_LOS_GRID[0], Q_LOS_GRID[-1])
    idx_p = np.argmin(np.abs(Q_PLANE_GRID - q_p))
    idx_l = np.argmin(np.abs(Q_LOS_GRID - q_l))
    return geom_cache[(Q_PLANE_GRID[idx_p], Q_LOS_GRID[idx_l])]

def predict_theta_E(cluster_name, A_c, q_plane, q_LOS, kappa_ext):
    cache = _baryon_cache[cluster_name]
    Sigma_bar = get_Sigma_interp(cluster_name, q_plane, q_LOS)
    Sigma_eff, _, _ = convolve_sigma_with_kernel(
        Sigma_bar, cache['R_grid_2d'], FIXED_PARAMS['ell0'], 
        FIXED_PARAMS['p'], FIXED_PARAMS['ncoh'], A_c, emphasize_interior=True, use_fft=True
    )
    
    if abs(kappa_ext) > 1e-6:
        Sigma_crit = _cosmo.critical_surface_density(cache['z_lens'], cache['z_source'])
        Sigma_eff += kappa_ext * Sigma_crit
    
    R_bins = np.linspace(0, cache['R_max']*0.9, 150)
    _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff, cache['R_grid_2d'], R_bins)
    
    valid = np.isfinite(Sigma_eff_prof)
    R_prof = 0.5 * (R_bins[:-1] + R_bins[1:])[valid]
    Sigma_eff_prof = Sigma_eff_prof[valid]
    
    if len(R_prof) < 10:
        return np.nan
    
    M_enc = cumulative_trapezoid(2*np.pi*R_prof*Sigma_eff_prof, R_prof, initial=0.0)
    Sigma_crit = _cosmo.critical_surface_density(cache['z_lens'], cache['z_source'])
    mean_kappa = M_enc / (np.pi * R_prof**2 * Sigma_crit)
    mean_kappa[0] = Sigma_eff_prof[0] / Sigma_crit
    
    idx_cross = np.where(mean_kappa >= 1.0)[0]
    if len(idx_cross) == 0:
        return np.nan
    
    R_E_kpc = R_prof[idx_cross[-1]]
    return _cosmo.physical_to_angular(R_E_kpc, cache['z_lens'])

def log_prior(theta, cluster_names):
    A_c = theta[0]
    if A_c < 10.0 or A_c > 25.0:
        return -np.inf
    
    log_p = 0.0
    n_clusters = len(cluster_names)
    for i in range(n_clusters):
        q_plane = theta[1 + 3*i]
        q_LOS = theta[2 + 3*i]
        kappa_ext = theta[3 + 3*i]
        
        if not (0.6 <= q_plane <= 1.4) or not (0.6 <= q_LOS <= 1.6) or abs(kappa_ext) > 0.15:
            return -np.inf
        log_p += norm.logpdf(q_plane, 1.0, 0.15) + norm.logpdf(kappa_ext, 0.0, 0.03)
    
    return log_p

def log_likelihood(theta):
    A_c = theta[0]
    chi2 = 0.0
    
    for i, cluster_name in enumerate(_cluster_names):
        q_plane, q_LOS, kappa_ext = theta[1+3*i], theta[2+3*i], theta[3+3*i]
        
        try:
            theta_E_pred = predict_theta_E(cluster_name, A_c, q_plane, q_LOS, kappa_ext)
        except Exception:
            return -np.inf
        
        if not np.isfinite(theta_E_pred):
            return -np.inf
        
        cache = _baryon_cache[cluster_name]
        chi2 += ((cache['theta_E_obs'] - theta_E_pred) / cache['theta_E_err'])**2
    
    return -0.5 * chi2

def log_probability(theta):
    lp = log_prior(theta, _cluster_names)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

if __name__ == '__main__':
    print("="*70)
    print("TIER-1+2 MCMC WITH GEOMETRY - GPU ACCELERATED")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\\nConfiguration: {N_WALKERS} walkers × {N_STEPS} steps ({N_BURN} burn-in), {N_PROCESSES} processes")
    
    print("\\n[1/8] Loading clusters...")
    train_clusters, holdout_clusters = load_catalog()
    print(f"  Train: {len(train_clusters)} clusters")
    
    _baryon_cache, _cosmo = build_cache(train_clusters)
    _cluster_names = train_clusters['cluster_name'].values
    n_params = 1 + 3 * len(_cluster_names)
    
    print(f"\\n[3/8] MCMC: {n_params} parameters")
    print("\\n[4/8] Initializing walkers...")
    
    theta_init = np.zeros(n_params)
    theta_init[0] = 16.5
    for i in range(len(_cluster_names)):
        theta_init[1+3*i:4+3*i] = [1.0, 1.0, 0.0]
    
    pos = theta_init + 1e-3 * np.random.randn(N_WALKERS, n_params)
    for j in range(N_WALKERS):
        pos[j, 0] = np.clip(pos[j, 0], 12.0, 20.0)
        for i in range(len(_cluster_names)):
            pos[j, 1+3*i] = np.clip(pos[j, 1+3*i], 0.7, 1.3)
            pos[j, 2+3*i] = np.clip(pos[j, 2+3*i], 0.7, 1.3)
            pos[j, 3+3*i] = np.clip(pos[j, 3+3*i], -0.05, 0.05)
    
    # Test
    print("  Testing initial likelihood...")
    test_lnp = log_probability(theta_init)
    print(f"  Initial log-probability: {test_lnp:.3f}")
    if test_lnp == -np.inf:
        print("  ERROR: Initial position has -inf likelihood!")
        sys.exit(1)
    
    print(f"\n[5/8] Running MCMC ({N_PROCESSES} processes, ~30-60 min)...\n")
    
    # On Windows, multiprocessing uses 'spawn' start method, so child processes won't inherit
    # globals set under __main__. We pass required state via an initializer.
    with Pool(N_PROCESSES, initializer=_init_worker, initargs=(_baryon_cache, _cluster_names)) as pool:
        sampler = emcee.EnsembleSampler(N_WALKERS, n_params, log_probability, pool=pool)
        start_time = time.time()
        sampler.run_mcmc(pos, N_STEPS, progress=True)
        end_time = time.time()
    
    print(f"\\n  Completed in {(end_time - start_time)/60:.1f} minutes")
    
    print("\\n[6/8] Analyzing...")
    acc_frac = np.mean(sampler.acceptance_fraction)
    print(f"  Acceptance fraction: {acc_frac:.3f}")
    
    flat_samples = sampler.get_chain(discard=N_BURN, flat=True)
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    A_c_med = percentiles[1, 0]
    print(f"  A_c = {A_c_med:.3f} (+{percentiles[2,0]-A_c_med:.3f}, -{A_c_med-percentiles[0,0]:.3f})")
    
    print("\\n[7/8] Evaluating...")
    theta_best = percentiles[1, :]
    train_results = []
    
    for i, cluster_name in enumerate(_cluster_names):
        cache = _baryon_cache[cluster_name]
        q_plane, q_LOS, kappa_ext = theta_best[1+3*i], theta_best[2+3*i], theta_best[3+3*i]
        theta_E_pred = predict_theta_E(cluster_name, theta_best[0], q_plane, q_LOS, kappa_ext)
        error = theta_E_pred - cache['theta_E_obs']
        chi2 = (error / cache['theta_E_err'])**2
        
        train_results.append({
            'cluster': cluster_name, 'theta_E_obs': cache['theta_E_obs'],
            'theta_E_pred': theta_E_pred, 'error': error, 'chi2': chi2,
            'q_plane': q_plane, 'q_LOS': q_LOS, 'kappa_ext': kappa_ext
        })
        
        print(f"  {cluster_name:12s}: obs={cache['theta_E_obs']:.1f}\", pred={theta_E_pred:.1f}\", err={error:+.2f}\", χ²={chi2:.2f}")
    
    train_chi2 = sum(r['chi2'] for r in train_results)
    train_dof = len(train_results) - 1
    print(f"\\n  Train χ²/d.o.f. = {train_chi2/train_dof:.2f}")
    
    print("\\n[8/8] Saving...")
    np.save(OUTPUT_DIR / 'flat_samples.npy', flat_samples)
    pd.DataFrame(train_results).to_csv(OUTPUT_DIR / 'train_results.csv', index=False)
    
    with open(OUTPUT_DIR / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"A_c = {A_c_med:.3f}\nTrain χ²/d.o.f. = {train_chi2/train_dof:.2f}\n")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(flat_samples[:, 0], bins=50, alpha=0.7)
    ax.axvline(A_c_med, color='k', linestyle='--', lw=2)
    ax.set_xlabel('$A_c$', fontsize=14)
    ax.set_ylabel('Samples', fontsize=14)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'posterior_Ac.png', dpi=150)
    plt.close()
    
    print(f"  Saved to {OUTPUT_DIR}")
    print("\\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"A_c = {A_c_med:.3f}, χ²/d.o.f. = {train_chi2/train_dof:.2f}, Runtime: {(end_time-start_time)/60:.1f} min")
    print("="*70 + "\\n")
