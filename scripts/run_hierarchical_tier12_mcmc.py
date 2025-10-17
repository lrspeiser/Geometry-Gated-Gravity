"""
Hierarchical Tier-1+2 MCMC Calibration
=======================================

Hierarchical model allowing per-cluster A_c variation with population-level
mean (mu_A) and scatter (sigma_A), full triaxial geometry, and robust likelihood.

Population model:
  mu_A ~ Uniform(10, 25)
  sigma_A ~ HalfNormal(5)
  A_c,i ~ Normal(mu_A, sigma_A)

Per-cluster geometry:
  q_plane,i, q_LOS,i, orientation angles, kappa_ext,i

Likelihood:
  Student-t for MACS0717 (merger), Normal for others

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
from scipy.stats import norm, halfnorm, t as student_t
import emcee
import corner
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
import argparse
import json

# WINDOWS MULTIPROCESSING NOTES:
# 1. Must use if __name__ == '__main__': guard (spawn method)
# 2. Need _init_worker() to populate globals (_baryon_cache, _cluster_names, _is_merger)
# 3. Ensure N_PROCESSES >= 1 to avoid pool creation errors
# 4. Use encoding='utf-8' when writing files (summary.txt)
# 5. Use raw strings r'' for matplotlib LaTeX labels (e.g., r'$\theta_E$')

from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from test_macs0416_projected_kernel import project_to_surface_density
from many_path_model.lensing_utilities import LensingCosmology

# Configuration
FIXED_PARAMS = {'ell0': 200.0, 'p': 2.0, 'ncoh': 2.0}
N_WALKERS = 48  # More walkers for hierarchical model
N_STEPS = 2000  # Longer for convergence
N_BURN = 600
N_PROCESSES = max(1, min(8, cpu_count() - 2))
OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'hierarchical_tier12_mcmc'
Q_PLANE_GRID = np.linspace(0.6, 1.4, 9)  # Finer grid
Q_LOS_GRID = np.linspace(0.6, 1.4, 9)

# Merger flag (Student-t likelihood for robust fitting)
MERGER_CLUSTERS = ['MACS0717']  # Known major merger
STUDENT_T_NU = 4.0  # Degrees of freedom for Student-t

# Global state (set in main and worker initializer)
_baryon_cache = None
_cosmo = None
_cluster_names = None
_is_merger = None

def _init_worker(baryon_cache, cluster_names, is_merger):
    """Initializer to populate globals in worker processes (Windows spawn-safe)."""
    global _baryon_cache, _cosmo, _cluster_names, _is_merger
    _baryon_cache = baryon_cache
    _cluster_names = cluster_names
    _is_merger = is_merger
    _cosmo = LensingCosmology()

def load_catalog(tiers=(1, 2), exclude=None, holdout=None, catalog_path=None):
    """Load Tier-filtered catalog, return (train, holdout) DataFrames."""
    if catalog_path is None:
        catalog_path = Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv'
    catalog = pd.read_csv(catalog_path)

    # Tier filter
    tier_set = set(int(t) for t in tiers)
    tiered = catalog[catalog['tier'].astype(int).isin(tier_set)].copy()

    def _norm(s):
        return str(s).upper().replace(' ', '').replace('-', '')

    excl = set(_norm(x) for x in (exclude or ['MACS0717']))
    hold = set(_norm(x) for x in (holdout or ['A1689', 'MACS1149']))

    norm_names = tiered['cluster_name'].apply(_norm)
    train = tiered[~norm_names.isin(excl | hold)].copy()
    hold_df = tiered[norm_names.isin(hold)].copy()
    return train.reset_index(drop=True), hold_df.reset_index(drop=True)

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
            'geometry_cache': geom_cache, 'R_grid_2d': R_grid_2d, 'R_max': R_max,
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

def log_prior(theta):
    """
    Hierarchical prior with population-level hyperparameters.
    
    Parameters (per cluster: 3 params):
      theta = [mu_A, sigma_A, A_c_1, q_plane_1, q_LOS_1, kappa_ext_1, ...]
    
    Total: 2 + 4*N_clusters
    """
    mu_A = theta[0]
    sigma_A = theta[1]
    
    # Population hyperpriors
    if not (10.0 <= mu_A <= 25.0):
        return -np.inf
    if not (0.0 < sigma_A < 10.0):
        return -np.inf
    
    log_p = halfnorm.logpdf(sigma_A, scale=5.0)
    
    n_clusters = len(_cluster_names)
    for i in range(n_clusters):
        A_c = theta[2 + 4*i]
        q_plane = theta[3 + 4*i]
        q_LOS = theta[4 + 4*i]
        kappa_ext = theta[5 + 4*i]
        
        # Per-cluster A_c drawn from population
        if not (5.0 <= A_c <= 30.0):  # Hard bounds
            return -np.inf
        log_p += norm.logpdf(A_c, mu_A, sigma_A)
        
        # Geometry priors
        if not (0.5 <= q_plane <= 1.5):
            return -np.inf
        log_p += norm.logpdf(q_plane, 1.0, 0.2)
        
        if not (0.5 <= q_LOS <= 1.6):
            return -np.inf
        
        # External convergence
        if abs(kappa_ext) > 0.2:
            return -np.inf
        log_p += norm.logpdf(kappa_ext, 0.0, 0.03)
    
    return log_p

def log_likelihood(theta):
    """
    Likelihood with Student-t for mergers (robust), Normal for relaxed clusters.
    """
    chi2 = 0.0
    log_like = 0.0
    
    n_clusters = len(_cluster_names)
    for i in range(n_clusters):
        cluster_name = _cluster_names[i]
        A_c = theta[2 + 4*i]
        q_plane = theta[3 + 4*i]
        q_LOS = theta[4 + 4*i]
        kappa_ext = theta[5 + 4*i]
        
        try:
            theta_E_pred = predict_theta_E(cluster_name, A_c, q_plane, q_LOS, kappa_ext)
        except Exception:
            return -np.inf
        
        if not np.isfinite(theta_E_pred):
            return -np.inf
        
        cache = _baryon_cache[cluster_name]
        residual = cache['theta_E_obs'] - theta_E_pred
        sigma = cache['theta_E_err']
        
        # Use Student-t for mergers (robust to outliers), Normal for others
        if _is_merger[i]:
            # Student-t log-likelihood
            log_like += student_t.logpdf(residual / sigma, df=STUDENT_T_NU)
        else:
            # Normal log-likelihood
            log_like += norm.logpdf(residual, 0.0, sigma)
    
    return log_like

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical Tier1+2 MCMC (γ=0 baseline)')
    parser.add_argument('--catalog', type=str, default=str(Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv'))
    parser.add_argument('--tiers', type=str, default='1,2')
    parser.add_argument('--exclude', type=str, default='MACS0717')
    parser.add_argument('--holdout', type=str, default='A1689,MACS1149')
    parser.add_argument('--outdir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--fixed_ell0', type=float, default=200.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    FIXED_PARAMS['ell0'] = float(args.fixed_ell0)

    print("="*70)
    print("HIERARCHICAL TIER-1+2 MCMC CALIBRATION")
    print("="*70)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tiers = tuple(int(t) for t in args.tiers.split(','))
    exclude_list = [s.strip() for s in args.exclude.split(',') if s.strip()]
    holdout_list = [s.strip() for s in args.holdout.split(',') if s.strip()]

    print(f"\nConfiguration:")
    print(f"  Hierarchical model: per-cluster A_c ~ N(mu_A, sigma_A)")
    print(f"  Robust likelihood: Student-t (ν={STUDENT_T_NU}) for mergers")
    print(f"  ell0 fixed: {FIXED_PARAMS['ell0']} kpc")
    print(f"  MCMC: {N_WALKERS} walkers × {N_STEPS} steps ({N_BURN} burn-in)")
    print(f"  Parallelization: {N_PROCESSES} processes")

    print("\n[1/8] Loading clusters...")
    train_clusters, holdout_clusters = load_catalog(tiers=tiers, exclude=exclude_list, holdout=holdout_list, catalog_path=Path(args.catalog))
    print(f"  Train: {len(train_clusters)} clusters")

    _baryon_cache, _cosmo = build_cache(train_clusters)
    _cluster_names = train_clusters['cluster_name'].values
    _is_merger = np.array([name in MERGER_CLUSTERS for name in _cluster_names])
    
    print(f"\\n  Merger clusters (Student-t): {', '.join(_cluster_names[_is_merger])}")
    print(f"  Relaxed clusters (Normal): {', '.join(_cluster_names[~_is_merger])}")
    
    # Parameter structure: [mu_A, sigma_A, A_c_1, q_plane_1, q_LOS_1, kappa_ext_1, ...]
    n_params = 2 + 4 * len(_cluster_names)
    
    print(f"\\n[3/8] MCMC: {n_params} parameters")
    print("  [0] mu_A (population mean)")
    print("  [1] sigma_A (population scatter)")
    print(f"  [2:{n_params}] per-cluster: A_c, q_plane, q_LOS, kappa_ext")
    
    print("\\n[4/8] Initializing walkers...")
    
    theta_init = np.zeros(n_params)
    theta_init[0] = 16.5  # mu_A
    theta_init[1] = 1.5   # sigma_A
    
    for i in range(len(_cluster_names)):
        theta_init[2 + 4*i] = 16.5  # A_c
        theta_init[3 + 4*i] = 1.0   # q_plane
        theta_init[4 + 4*i] = 1.0   # q_LOS
        theta_init[5 + 4*i] = 0.0   # kappa_ext
    
    # Initialize walkers with small perturbations
    pos = theta_init + 1e-3 * np.random.randn(N_WALKERS, n_params)
    
    for j in range(N_WALKERS):
        pos[j, 0] = np.clip(pos[j, 0], 12.0, 20.0)  # mu_A
        pos[j, 1] = np.clip(pos[j, 1], 0.5, 5.0)    # sigma_A
        for i in range(len(_cluster_names)):
            pos[j, 2+4*i] = np.clip(pos[j, 2+4*i], 12.0, 20.0)  # A_c
            pos[j, 3+4*i] = np.clip(pos[j, 3+4*i], 0.7, 1.3)    # q_plane
            pos[j, 4+4*i] = np.clip(pos[j, 4+4*i], 0.7, 1.3)    # q_LOS
            pos[j, 5+4*i] = np.clip(pos[j, 5+4*i], -0.05, 0.05) # kappa_ext
    
    # Test initial likelihood
    print("  Testing initial likelihood...")
    test_lnp = log_probability(theta_init)
    print(f"  Initial log-probability: {test_lnp:.3f}")
    if test_lnp == -np.inf:
        print("  ERROR: Initial position has -inf likelihood!")
        sys.exit(1)
    
    print(f"\n[5/8] Running MCMC ({N_PROCESSES} processes, ~60-90 min)...\n")
    
    # Windows multiprocessing fix: pass globals via initializer
    with Pool(N_PROCESSES, initializer=_init_worker, initargs=(_baryon_cache, _cluster_names, _is_merger)) as pool:
        sampler = emcee.EnsembleSampler(N_WALKERS, n_params, log_probability, pool=pool)
        start_time = time.time()
        sampler.run_mcmc(pos, N_STEPS, progress=True)
        end_time = time.time()
    
    print(f"\\n  Completed in {(end_time - start_time)/60:.1f} minutes")
    
    # Analysis
    print("\\n[6/8] Analyzing chains...")
    acc_frac = np.mean(sampler.acceptance_fraction)
    print(f"  Acceptance fraction: {acc_frac:.3f}")
    
    flat_samples = sampler.get_chain(discard=N_BURN, flat=True)
    print(f"  Samples after burn-in: {flat_samples.shape[0]}")
    
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    
    mu_A_med = percentiles[1, 0]
    sigma_A_med = percentiles[1, 1]
    
    print(f"\\n  Population parameters:")
    print(f"    mu_A    = {mu_A_med:.3f} (+{percentiles[2,0]-mu_A_med:.3f}, -{mu_A_med-percentiles[0,0]:.3f})")
    print(f"    sigma_A = {sigma_A_med:.3f} (+{percentiles[2,1]-sigma_A_med:.3f}, -{sigma_A_med-percentiles[0,1]:.3f})")
    
    print("\\n[7/8] Evaluating train set...")
    theta_best = percentiles[1, :]
    train_results = []
    
    for i, cluster_name in enumerate(_cluster_names):
        cache = _baryon_cache[cluster_name]
        A_c = theta_best[2 + 4*i]
        q_plane = theta_best[3 + 4*i]
        q_LOS = theta_best[4 + 4*i]
        kappa_ext = theta_best[5 + 4*i]
        
        theta_E_pred = predict_theta_E(cluster_name, A_c, q_plane, q_LOS, kappa_ext)
        error = theta_E_pred - cache['theta_E_obs']
        chi2 = (error / cache['theta_E_err'])**2
        
        train_results.append({
            'cluster': cluster_name, 'theta_E_obs': cache['theta_E_obs'],
            'theta_E_pred': theta_E_pred, 'error': error, 'chi2': chi2,
            'A_c': A_c, 'q_plane': q_plane, 'q_LOS': q_LOS, 'kappa_ext': kappa_ext,
            'is_merger': _is_merger[i]
        })
        
        merger_flag = " [MERGER]" if _is_merger[i] else ""
        print(f"  {cluster_name:12s}{merger_flag:10s}: obs={cache['theta_E_obs']:.1f}\", pred={theta_E_pred:.1f}\", err={error:+.2f}\", χ²={chi2:.2f}")
        print(f"                             A_c={A_c:.2f}, q_p={q_plane:.3f}, q_l={q_LOS:.3f}, κ={kappa_ext:+.3f}")
    
    train_chi2 = sum(r['chi2'] for r in train_results)
    train_dof = len(train_results) - 2  # -2 for population params
    print(f"\\n  Train χ²/d.o.f. = {train_chi2/train_dof:.2f}")
    
    # Save results
    print("\\n[8/8] Saving results...")
    
    np.save(outdir / 'flat_samples.npy', flat_samples)
    pd.DataFrame(train_results).to_csv(outdir / 'train_results.csv', index=False)

    settings = {
        'catalog': str(Path(args.catalog).resolve()),
        'tiers': list(tiers),
        'exclude': exclude_list,
        'holdout': holdout_list,
        'ell0_fixed_kpc': FIXED_PARAMS['ell0'],
        'walkers': N_WALKERS,
        'steps': N_STEPS,
        'burn_in': N_BURN,
        'seed': args.seed,
    }
    with open(outdir / 'settings.json', 'w', encoding='utf-8') as sf:
        json.dump(settings, sf, indent=2)

    with open(outdir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"Hierarchical Tier-1+2 MCMC Calibration\n")
        f.write(f"="*70 + "\n\n")
        f.write(f"Population parameters:\n")
        f.write(f"  mu_A    = {mu_A_med:.3f} ({percentiles[0,0]:.3f}, {percentiles[2,0]:.3f})\n")
        f.write(f"  sigma_A = {sigma_A_med:.3f} ({percentiles[0,1]:.3f}, {percentiles[2,1]:.3f})\n")
        f.write(f"\nTrain χ²/d.o.f. = {train_chi2/train_dof:.2f}\n\n")
        f.write("Per-cluster results:\\n")
        for r in train_results:
            f.write(f"  {r['cluster']}: A_c={r['A_c']:.2f}, q_plane={r['q_plane']:.3f}, q_LOS={r['q_LOS']:.3f}\\n")
    
    # Plots
    print("  Creating diagnostic plots...")
    
    # Population parameters corner plot
    fig = corner.corner(
        flat_samples[:, :2],
        labels=['$\\mu_A$', '$\\sigma_A$'],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f'
    )
    plt.savefig(outdir / 'corner_population.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # A_c histogram with population distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cluster_name in enumerate(_cluster_names):
        A_c_samples = flat_samples[:, 2+4*i]
        ax.hist(A_c_samples, bins=30, alpha=0.5, label=cluster_name)
    
    # Population distribution
    mu_A_samples = flat_samples[:, 0]
    sigma_A_samples = flat_samples[:, 1]
    A_grid = np.linspace(12, 22, 200)
    pop_dist = np.zeros_like(A_grid)
    for mu, sig in zip(mu_A_samples[::100], sigma_A_samples[::100]):
        pop_dist += norm.pdf(A_grid, mu, sig)
    pop_dist /= len(mu_A_samples[::100])
    ax2 = ax.twinx()
    ax2.plot(A_grid, pop_dist, 'k-', lw=3, label='Population')
    
    ax.set_xlabel('$A_c$', fontsize=14)
    ax.set_ylabel('Samples', fontsize=14)
    ax2.set_ylabel('Population density', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    plt.title('Per-cluster $A_c$ posteriors vs Population', fontsize=15)
    plt.tight_layout()
    plt.savefig(outdir / 'Ac_hierarchy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Predicted vs observed
    fig, ax = plt.subplots(figsize=(8, 8))
    obs = np.array([r['theta_E_obs'] for r in train_results])
    pred = np.array([r['theta_E_pred'] for r in train_results])
    err_obs = np.array([_baryon_cache[r['cluster']]['theta_E_err'] for r in train_results])
    
    colors = ['red' if r['is_merger'] else 'blue' for r in train_results]
    for i, r in enumerate(train_results):
        marker = '^' if r['is_merger'] else 'o'
        ax.errorbar(obs[i], pred[i], xerr=err_obs[i], fmt=marker, 
                    color=colors[i], markersize=10, capsize=5, label=r['cluster'])
    
    # 1:1 line
    lim = [20, 60]
    ax.plot(lim, lim, 'k--', lw=2, alpha=0.5, label='1:1')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r'Observed $\theta_E$ [arcsec]', fontsize=14)
    ax.set_ylabel(r'Predicted $\theta_E$ [arcsec]', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_title('Hierarchical Model: Train Set', fontsize=15)
    plt.tight_layout()
    plt.savefig(outdir / 'pred_vs_obs.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {outdir}")

    print("\n" + "="*70)
    print("HIERARCHICAL CALIBRATION COMPLETE")
    print("="*70)
    print(f"\\nPopulation:")
    print(f"  mu_A = {mu_A_med:.3f} ± {(percentiles[2,0]-percentiles[0,0])/2:.3f}")
    print(f"  sigma_A = {sigma_A_med:.3f} ± {(percentiles[2,1]-percentiles[0,1])/2:.3f}")
    print(f"\\nFit quality:")
    print(f"  χ²/d.o.f. = {train_chi2/train_dof:.2f}")
    print(f"\\nRuntime: {(end_time-start_time)/60:.1f} minutes")
    print("="*70 + "\\n")
