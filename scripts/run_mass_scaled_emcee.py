"""
Mass-Scaled Hierarchical Inference using emcee
===============================================

Simplified version using emcee instead of PyMC for mass-scaling analysis.
Uses the same approach as run_hierarchical_tier12_clean.py but adds mass-scaling
relationship: ℓ₀(M) = ℓ₀,⋆ × (R₅₀₀/1Mpc)^γ

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
from scipy.stats import norm, halfnorm
import emcee
from emcee.moves import StretchMove, DEMove, DESnookerMove
import corner
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
import argparse
import json

from core.build_cluster_baryons import build_cluster_baryon_model, ClusterBaryonParams
from core.kernel2d_sigma import convolve_sigma_with_kernel, azimuthal_average
from test_macs0416_projected_kernel import project_to_surface_density
from many_path_model.lensing_utilities import LensingCosmology
from core.bcg_profiles import estimate_bcg_mass, hernquist_projected_density

# Configuration
FIXED_PARAMS = {'p': 2.0, 'ncoh': 2.0}
N_WALKERS = 64  # Increased for better mixing
N_STEPS = 3000  # Longer run for N=10
N_BURN = 1000   # More aggressive burn-in
N_PROCESSES = max(1, min(8, cpu_count() - 2))
OUTPUT_DIR = Path(__file__).parent.parent / 'output' / 'mass_scaled_emcee'
Q_PLANE_GRID = np.linspace(0.6, 1.4, 13)  # Finer grid for better interpolation
Q_LOS_GRID = np.linspace(0.6, 1.4, 13)

# Default selection controls (overridden by CLI)
DEFAULT_EXCLUDE = ['MACS0717']
DEFAULT_HOLDOUT = ['A1689', 'MACS1149']

# Global state
_baryon_cache = None
_cosmo = None
_cluster_names = None
_cluster_R500 = None
_pzs_mode = 'median'

def _init_worker(baryon_cache, cluster_names, cluster_R500, pzs_mode='median'):
    """Initializer to populate globals in worker processes (Windows spawn-safe)."""
    global _baryon_cache, _cosmo, _cluster_names, _cluster_R500, _pzs_mode
    _baryon_cache = baryon_cache
    _cluster_names = cluster_names
    _cluster_R500 = cluster_R500
    _pzs_mode = pzs_mode
    _cosmo = LensingCosmology()

def load_catalog(tiers=(1, 2), exclude=None, holdout=None, catalog_path=None):
    """Load Tier-filtered catalog, return (train, holdout) DataFrames."""
    if catalog_path is None:
        catalog_path = Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv'
    catalog = pd.read_csv(catalog_path)

    # Tier filter
    tier_set = set(int(t) for t in tiers)
    tiered = catalog[catalog['tier'].astype(int).isin(tier_set)].copy()

    # Normalize names for matching
    def _norm(s):
        return str(s).upper().replace(' ', '').replace('-', '')

    excl = set(_norm(x) for x in (exclude or DEFAULT_EXCLUDE))
    hold = set(_norm(x) for x in (holdout or DEFAULT_HOLDOUT))

    norm_names = tiered['cluster_name'].apply(_norm)
    mask_excl = ~norm_names.isin(excl)
    mask_hold = norm_names.isin(hold)

    train = tiered[mask_excl & ~mask_hold].copy()
    hold_df = tiered[mask_hold].copy()
    return train.reset_index(drop=True), hold_df.reset_index(drop=True)

def build_cache(train_clusters):
    print("\n[2/8] Pre-computing geometry grid...")
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
        
        # Add BCG/ICL component (spherical, independent of triaxial geometry)
        M_BCG, r_eff_BCG = estimate_bcg_mass(cluster['M_500_Msun'], cluster['z_lens'])
        Sigma_BCG_spherical = hernquist_projected_density(R_grid_2d, M_BCG, r_eff_BCG)
        
        geom_cache = {}
        for q_p in Q_PLANE_GRID:
            for q_l in Q_LOS_GRID:
                Sigma_baryons_triaxial = project_to_surface_density(r_3d, rho_total, R_grid_2d, q_l, q_p)
                # Add spherical BCG to triaxial dark matter + gas
                Sigma_bar = Sigma_baryons_triaxial + Sigma_BCG_spherical
                geom_cache[(q_p, q_l)] = Sigma_bar
        
        cache[name] = {
            'geometry_cache': geom_cache, 'R_grid_2d': R_grid_2d, 'R_max': R_max,
            'z_lens': cluster['z_lens'], 'z_source': cluster['z_source'],
            'theta_E_obs': cluster['theta_E_obs_arcsec'], 'theta_E_err': cluster['theta_E_err_arcsec'],
            'R_500': cluster['R_500_kpc']
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

def predict_theta_E(cluster_name, ell0_cluster, A_c, q_plane, q_LOS, kappa_ext):
    """Predict theta_E with mass-scaled ℓ₀."""
    cache = _baryon_cache[cluster_name]
    Sigma_bar = get_Sigma_interp(cluster_name, q_plane, q_LOS)
    Sigma_eff, _, _ = convolve_sigma_with_kernel(
        Sigma_bar, cache['R_grid_2d'], ell0_cluster, 
        FIXED_PARAMS['p'], FIXED_PARAMS['ncoh'], A_c, emphasize_interior=True, use_fft=True
    )
    
    # Sigma_crit per chosen source-redshift treatment
    if _pzs_mode == 'lognormal':
        Sigma_crit = _cosmo.effective_critical_density_with_distribution(cache['z_lens'])
    else:
        Sigma_crit = _cosmo.critical_surface_density(cache['z_lens'], cache['z_source'])
    
    if abs(kappa_ext) > 1e-6:
        Sigma_eff += kappa_ext * Sigma_crit
    
    R_bins = np.linspace(0, cache['R_max']*0.9, 150)
    _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff, cache['R_grid_2d'], R_bins)
    
    valid = np.isfinite(Sigma_eff_prof)
    R_prof = 0.5 * (R_bins[:-1] + R_bins[1:])[valid]
    Sigma_eff_prof = Sigma_eff_prof[valid]
    
    if len(R_prof) < 10:
        return np.nan
    
    M_enc = cumulative_trapezoid(2*np.pi*R_prof*Sigma_eff_prof, R_prof, initial=0.0)
    mean_kappa = M_enc / (np.pi * R_prof**2 * Sigma_crit)
    mean_kappa[0] = Sigma_eff_prof[0] / Sigma_crit
    
    idx_cross = np.where(mean_kappa >= 1.0)[0]
    if len(idx_cross) == 0:
        return np.nan
    
    R_E_kpc = R_prof[idx_cross[-1]]
    return _cosmo.physical_to_angular(R_E_kpc, cache['z_lens'])

def log_prior(theta):
    """
    theta = [ell0_star, gamma, mu_A, sigma_A, A_c_1, q_plane_1, q_LOS_1, kappa_ext_1, ...]
    
    Mass-scaling: ℓ₀(R₅₀₀) = ℓ₀,⋆ × (R₅₀₀/1Mpc)^γ
    """
    ell0_star = theta[0]  # Reference ℓ₀ at 1 Mpc
    gamma = theta[1]      # Mass-scaling exponent
    mu_A = theta[2]       # Population mean A_c
    sigma_A = theta[3]    # Population scatter A_c
    
    # Mass-scaling priors
    if not (50.0 <= ell0_star <= 500.0):
        return -np.inf
    if not (0.0 <= gamma <= 1.5):
        return -np.inf
    
    # Population priors (allow lower A_c to match data)
    if not (2.0 <= mu_A <= 12.0):
        return -np.inf
    if not (0.0 < sigma_A < 3.5):
        return -np.inf
    
    log_p = norm.logpdf(ell0_star, 200, 100)
    log_p += norm.logpdf(gamma, 0.5, 0.3)
    log_p += halfnorm.logpdf(sigma_A, scale=5.0)
    
    n_clusters = len(_cluster_names)
    for i in range(n_clusters):
        A_c = theta[4 + 4*i]
        q_plane = theta[5 + 4*i]
        q_LOS = theta[6 + 4*i]
        kappa_ext = theta[7 + 4*i]
        
        if not (2.0 <= A_c <= 12.0):
            return -np.inf
        log_p += norm.logpdf(A_c, mu_A, sigma_A)
        
        if not (0.5 <= q_plane <= 1.5):
            return -np.inf
        log_p += norm.logpdf(q_plane, 1.0, 0.2)
        
        if not (0.5 <= q_LOS <= 1.6):
            return -np.inf
        
        if abs(kappa_ext) > 0.2:
            return -np.inf
        log_p += norm.logpdf(kappa_ext, 0.0, 0.03)
    
    return log_p

def log_likelihood(theta):
    ell0_star = theta[0]
    gamma = theta[1]
    
    log_like = 0.0
    n_clusters = len(_cluster_names)
    
    for i in range(n_clusters):
        cluster_name = _cluster_names[i]
        R_500 = _cluster_R500[i]
        
        # Mass-scaled ℓ₀ for this cluster
        ell0_cluster = ell0_star * (R_500 / 1000.0)**gamma
        
        A_c = theta[4 + 4*i]
        q_plane = theta[5 + 4*i]
        q_LOS = theta[6 + 4*i]
        kappa_ext = theta[7 + 4*i]
        
        try:
            theta_E_pred = predict_theta_E(cluster_name, ell0_cluster, A_c, q_plane, q_LOS, kappa_ext)
        except Exception:
            return -np.inf
        
        if not np.isfinite(theta_E_pred):
            return -np.inf
        
        cache = _baryon_cache[cluster_name]
        residual = cache['theta_E_obs'] - theta_E_pred
        sigma = cache['theta_E_err']
        
        log_like += norm.logpdf(residual, 0.0, sigma)
    
    return log_like

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

if __name__ == '__main__':
    import hashlib, json
    parser = argparse.ArgumentParser(description='Mass-scaled hierarchical inference (emcee)')
    parser.add_argument('--catalog', type=str, default=str(Path(__file__).parent.parent / 'data' / 'clusters' / 'master_catalog.csv'))
    parser.add_argument('--tiers', type=str, default='1,2,3', help='Comma-separated tiers to include (default: 1,2,3 for N=10)')
    parser.add_argument('--exclude', type=str, default='MACS0717', help='Comma-separated cluster names to exclude')
    parser.add_argument('--holdout', type=str, default='A1689,MACS1149', help='Comma-separated hold-out clusters')
    parser.add_argument('--pzs', type=str, default='lognormal', choices=['median','lognormal'], help='Source redshift treatment: median or lognormal (default: lognormal)')
    parser.add_argument('--outdir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("="*70)
    print("MASS-SCALED HIERARCHICAL INFERENCE (emcee)")
    print("="*70)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tiers = tuple(int(t) for t in args.tiers.split(','))
    exclude_list = [s.strip() for s in args.exclude.split(',') if s.strip()]
    holdout_list = [s.strip() for s in args.holdout.split(',') if s.strip()]

    print(f"\nConfiguration:")
    print(f"  Tiers: {tiers}; Exclude: {exclude_list}; Hold-out: {holdout_list}")
    print(f"  Model: ell0(M) = ell0_star x (R_500/1Mpc)^gamma, A_c ~ N(mu_A, sigma_A)")
    print(f"  MCMC: {N_WALKERS} walkers x {N_STEPS} steps ({N_BURN} burn-in)")

    print(f"\n[1/8] Loading clusters...")
    train_clusters, holdout_clusters = load_catalog(tiers=tiers, exclude=exclude_list, holdout=holdout_list, catalog_path=Path(args.catalog))
    print(f"  Train: {len(train_clusters)} clusters")
    print(f"  Train set: {', '.join(train_clusters['cluster_name'].values)}")

    _baryon_cache, _cosmo = build_cache(train_clusters)
    _cluster_names = train_clusters['cluster_name'].values
    _cluster_R500 = train_clusters['R_500_kpc'].values
    
    # Parameters: [ell0_star, gamma, mu_A, sigma_A] + 4*N_clusters
    n_params = 4 + 4 * len(_cluster_names)
    
    print(f"\n[3/8] MCMC: {n_params} parameters")
    print("  [0] ell0_star (reference at R500=1Mpc)")
    print("  [1] gamma (mass-scaling exponent)")
    print("  [2] mu_A (population mean)")
    print("  [3] sigma_A (population scatter)")
    print(f"  [4:{n_params}] per-cluster: A_c, q_plane, q_LOS, kappa_ext")
    
    print("\n[4/8] Initializing walkers...")
    
    theta_init = np.zeros(n_params)
    theta_init[0] = 200.0  # ell0_star
    theta_init[1] = 0.5    # gamma
    theta_init[2] = 5.0    # mu_A (start lower to allow data to pull it down)
    theta_init[3] = 2.0    # sigma_A (HalfNormal scale ~3)
    
    for i in range(len(_cluster_names)):
        theta_init[4 + 4*i] = 5.0  # A_c (start at value that gives theta_E ≈ 38")
        theta_init[5 + 4*i] = 1.0
        theta_init[6 + 4*i] = 1.0
        theta_init[7 + 4*i] = 0.0
    
    pos = theta_init + 1e-3 * np.random.randn(N_WALKERS, n_params)
    
    for j in range(N_WALKERS):
        pos[j, 0] = np.clip(pos[j, 0], 100.0, 300.0)
        pos[j, 1] = np.clip(pos[j, 1], 0.0, 1.5)
        pos[j, 2] = np.clip(pos[j, 2], 4.0, 6.0)    # mu_A centered near 5
        pos[j, 3] = np.clip(pos[j, 3], 0.5, 3.0)
        for i in range(len(_cluster_names)):
            pos[j, 4+4*i] = np.clip(5.0 + 1.0*np.random.randn(), 2.0, 12.0)  # A_c around 5 ±1
            pos[j, 5+4*i] = np.clip(1.0 + 0.1*np.random.randn(), 0.7, 1.3)
            pos[j, 6+4*i] = np.clip(1.0 + 0.1*np.random.randn(), 0.7, 1.3)
            pos[j, 7+4*i] = np.clip(0.0 + 0.02*np.random.randn(), -0.1, 0.1)
    
    print("  Testing initial likelihood...")
    test_lnp = log_probability(theta_init)
    print(f"  Initial log-probability: {test_lnp:.3f}")
    
    # [5/8] Pre-run amplitude bracketing check
    print("\n[5/8] Pre-run amplitude bracketing check...")
    def bracket_thetaE(cluster_name):
        cache = _baryon_cache[cluster_name]
        R_500 = cache['R_500']
        obs = cache['theta_E_obs']
        A_vals = np.linspace(2.0, 12.0, 21)
        preds = []
        for A in A_vals:
            try:
                ell0_tmp = 200.0 * (R_500/1000.0)**0.1  # rough
                pred = predict_theta_E(cluster_name, ell0_tmp, A, 1.0, 1.0, 0.0)
            except Exception:
                pred = np.nan
            preds.append(pred)
        preds = np.array(preds)
        ok = np.nanmin(preds) < obs < np.nanmax(preds)
        print(f"  {cluster_name:12s}: bracket={'OK' if ok else 'FAIL'}  range=[{np.nanmin(preds):.1f},{np.nanmax(preds):.1f}]  obs={obs:.1f}")
        return ok
    _ = [bracket_thetaE(name) for name in _cluster_names]
    
    print(f"\n[6/8] Running MCMC ({N_PROCESSES} processes)...\n")
    moves = [(StretchMove(a=2.0), 0.5), (DEMove(), 0.3), (DESnookerMove(), 0.2)]
    with Pool(N_PROCESSES, initializer=_init_worker, initargs=(_baryon_cache, _cluster_names, _cluster_R500, args.pzs)) as pool:
        sampler = emcee.EnsembleSampler(N_WALKERS, n_params, log_probability, pool=pool, moves=moves)
        start_time = time.time()
        sampler.run_mcmc(pos, N_STEPS, progress=True)
        end_time = time.time()

    print(f"\n  Completed in {(end_time - start_time)/60:.1f} minutes")

    print("\n[7/8] Analyzing...")
    acc_frac = np.mean(sampler.acceptance_fraction)
    print(f"  Acceptance: {acc_frac:.3f}")

    flat_samples = sampler.get_chain(discard=N_BURN, flat=True)

    # Persist settings/metadata
    settings = {
        'catalog': str(Path(args.catalog).resolve()),
        'tiers': list(tiers),
        'exclude': exclude_list,
        'holdout': holdout_list,
        'walkers': N_WALKERS,
        'steps': N_STEPS,
        'burn_in': N_BURN,
        'acceptance_fraction_mean': float(acc_frac),
        'seed': args.seed,
    }
    with open(outdir / 'settings.json', 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2)
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    
    ell0_star_med = percentiles[1, 0]
    gamma_med = percentiles[1, 1]
    mu_A_med = percentiles[1, 2]
    sigma_A_med = percentiles[1, 3]
    
    print(f"\n  Mass-scaling parameters:")
    print(f"    ell0_star = {ell0_star_med:.1f} kpc [{percentiles[0,0]:.1f}, {percentiles[2,0]:.1f}]")
    print(f"    gamma     = {gamma_med:.3f} [{percentiles[0,1]:.3f}, {percentiles[2,1]:.3f}]")
    print(f"\n  Population parameters:")
    print(f"    mu_A  = {mu_A_med:.3f} [{percentiles[0,2]:.3f}, {percentiles[2,2]:.3f}]")
    print(f"    sigma_A  = {sigma_A_med:.3f} [{percentiles[0,3]:.3f}, {percentiles[2,3]:.3f}]")
    
    print("\n[8/8] Evaluating...")
    theta_best = percentiles[1, :]
    train_results = []
    
    for i, cluster_name in enumerate(_cluster_names):
        cache = _baryon_cache[cluster_name]
        R_500 = _cluster_R500[i]
        ell0_cluster = theta_best[0] * (R_500 / 1000.0)**theta_best[1]
        A_c = theta_best[4 + 4*i]
        q_plane = theta_best[5 + 4*i]
        q_LOS = theta_best[6 + 4*i]
        kappa_ext = theta_best[7 + 4*i]
        
        theta_E_pred = predict_theta_E(cluster_name, ell0_cluster, A_c, q_plane, q_LOS, kappa_ext)
        error = theta_E_pred - cache['theta_E_obs']
        chi2 = (error / cache['theta_E_err'])**2
        
        train_results.append({
            'cluster': cluster_name, 'theta_E_obs': cache['theta_E_obs'],
            'theta_E_pred': theta_E_pred, 'error': error, 'chi2': chi2,
            'R_500': R_500, 'ell0': ell0_cluster, 'A_c': A_c
        })
        
        print(f"  {cluster_name:12s}: obs={cache['theta_E_obs']:.1f}\", pred={theta_E_pred:.1f}\", err={error:+.2f}\", χ²={chi2:.2f}")
        print(f"                   R₅₀₀={R_500:.0f}kpc, ℓ₀={ell0_cluster:.0f}kpc, A_c={A_c:.2f}")
    
    train_chi2 = sum(r['chi2'] for r in train_results)
    train_dof = len(train_results) - 4  # 4 population params
    print(f"\n  χ²/d.o.f. = {train_chi2/train_dof:.2f}")
    
    # Build provenance manifest
    with open(args.catalog, 'rb') as f:
        catalog_md5 = hashlib.md5(f.read()).hexdigest()
    
    manifest = {
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "train_clusters": sorted(train_clusters['cluster_name'].tolist()),
        "tiers": list(tiers),
        "cosmology": {"H0": 70.0, "Om0": 0.3},
        "physics": {
            "bcg": True,
            "triaxial": True,
            "pzsource": args.pzs
        },
        "kernel": {
            "norm": "local_coherence",
            "ell0_star_init_kpc": float(theta_init[0]),
            "mass_scaling": True,
            "gamma_prior": "Uniform(0,1.5)"
        },
        "catalog_md5": catalog_md5,
        "catalog_path": str(Path(args.catalog).resolve())
    }
    
    # Save with provenance manifest
    npz_path = outdir / 'flat_samples.npz'
    np.savez(npz_path, samples=flat_samples, manifest=json.dumps(manifest))
    print(f"\n  Saved posterior with manifest: {npz_path}")
    print(f"  Run ID: {manifest['run_id']}")
    print(f"  Catalog MD5: {catalog_md5[:8]}...")
    
    # Also save legacy .npy for backward compatibility
    np.save(outdir / 'flat_samples.npy', flat_samples)
    pd.DataFrame(train_results).to_csv(outdir / 'train_results.csv', index=False)

    with open(outdir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"Mass-Scaled Hierarchical Calibration (5 relaxed clusters)\n")
        f.write(f"="*70 + "\n\n")
        f.write(f"ℓ₀,⋆ = {ell0_star_med:.1f} kpc [{percentiles[0,0]:.1f}, {percentiles[2,0]:.1f}]\n")
        f.write(f"γ = {gamma_med:.3f} [{percentiles[0,1]:.3f}, {percentiles[2,1]:.3f}]\n")
        f.write(f"μ_A = {mu_A_med:.3f} [{percentiles[0,2]:.3f}, {percentiles[2,2]:.3f}]\n")
        f.write(f"σ_A = {sigma_A_med:.3f} [{percentiles[0,3]:.3f}, {percentiles[2,3]:.3f}]\n")
        f.write(f"χ²/d.o.f. = {train_chi2/train_dof:.2f}\n")
    
    # Plots
    fig = corner.corner(flat_samples[:, :4], 
                       labels=[r'$\ell_{0,\star}$', r'$\gamma$', r'$\mu_A$', r'$\sigma_A$'],
                       quantiles=[0.16, 0.5, 0.84], show_titles=True)
    plt.savefig(outdir / 'corner_mass_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("MASS-SCALED CALIBRATION COMPLETE")
    print("="*70)
    print(f"Mass-scaling: ℓ₀,⋆ = {ell0_star_med:.1f} kpc, γ = {gamma_med:.3f}")
    print(f"Population: μ_A = {mu_A_med:.3f}, σ_A = {sigma_A_med:.3f}")
    print(f"χ²/d.o.f. = {train_chi2/train_dof:.2f}")
    print(f"Outputs: {outdir}")
    print("="*70 + "\n")
