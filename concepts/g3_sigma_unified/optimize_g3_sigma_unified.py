#!/usr/bin/env python3
"""
Optimize Unified G³-Σ Parameters
=================================

Grid search to find the single best parameter tuple that works
across galaxies, MW, and clusters.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import logging
from typing import Dict, Tuple, List
from dataclasses import dataclass, asdict
from scipy.optimize import differential_evolution
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
G_NEWTON = 4.302e-6  # kpc (km/s)^2 / M_sun

# ==============================================================================
# SIMPLIFIED G³-Σ FORMULAS FOR FAST OPTIMIZATION
# ==============================================================================

def g3_sigma_1d_galaxy(r_kpc, v_bar_kms, M_bary, r_half, sigma_mean,
                       S0, rc, gamma, beta, g_sat, sigma_star, alpha_sigma):
    """
    Simplified 1D G³-Σ for galaxy rotation curves (fast approximation)
    """
    r = np.asarray(r_kpc)
    v_b = np.asarray(v_bar_kms)
    
    # Effective parameters with geometry scaling
    rc_eff = rc * (r_half / 30.0) ** gamma
    S0_eff = S0 * (150.0 / max(sigma_mean, 1.0)) ** beta
    
    # Estimate local surface density
    sigma_loc = M_bary / (np.pi * r_half**2) * np.exp(-r / r_half) / 1e6  # Msun/pc^2
    
    # Sigma screen
    screen = 1.0 / (1.0 + (sigma_loc / sigma_star) ** alpha_sigma)
    
    # Field gradient estimate
    grad_phi = v_b**2 / (r + 1e-10)
    
    # Saturating mobility
    x = grad_phi / g_sat
    mobility = 1.0 / (1.0 + x**2)
    
    # Effective coupling
    coupling = S0_eff * screen * mobility
    
    # Tail velocity (simplified field solution)
    v_tail = 140.0 * np.sqrt(coupling) * np.sqrt(r / (r + rc_eff))
    
    # Total velocity
    v_total = np.sqrt(v_b**2 + v_tail**2)
    
    return v_total

def g3_sigma_cluster_temp(r_kpc, rho_gas, r_core, 
                         S0, rc, gamma, beta, g_sat, sigma_star, alpha_sigma):
    """
    Simplified G³-Σ for cluster temperature profiles (fast approximation)
    """
    r = np.asarray(r_kpc)
    
    # Effective parameters
    rc_eff = rc * (r_core / 30.0) ** gamma
    S0_eff = S0 * (150.0 / 1000.0) ** beta  # Clusters have high surface density
    
    # Simplified potential gradient
    dphi_dr = S0_eff * G_NEWTON * rho_gas * rc_eff / (1 + (r / rc_eff)**2)
    
    # Temperature from HSE
    mu_mp_kB = 7.0e-11  # keV s^2/kpc^2
    T_keV = mu_mp_kB * dphi_dr * r
    
    # Apply saturation at high gradients
    T_max = 15.0  # Maximum reasonable temperature
    T_keV = np.minimum(T_keV, T_max)
    
    return T_keV

# ==============================================================================
# OBJECTIVE FUNCTION
# ==============================================================================

def compute_objective(params: np.ndarray, data: Dict, weights: Dict = None) -> float:
    """
    Compute combined objective across all regimes
    
    Parameters:
    -----------
    params : array
        [S0, rc, gamma, beta, g_sat, sigma_star, alpha_sigma]
    data : dict
        Test data for galaxies, MW, and clusters
    weights : dict
        Relative weights for each regime
        
    Returns:
    --------
    objective : float
        Combined error metric (lower is better)
    """
    
    S0, rc, gamma, beta, g_sat, sigma_star, alpha_sigma = params
    
    if weights is None:
        weights = {'galaxies': 0.4, 'mw': 0.3, 'clusters': 0.3}
    
    errors = []
    
    # Test on galaxies
    if 'galaxies' in data:
        for gal in data['galaxies']:
            v_model = g3_sigma_1d_galaxy(
                gal['r'], gal['v_bar'], gal['M_bary'], gal['r_half'], gal['sigma_mean'],
                S0, rc, gamma, beta, g_sat, sigma_star, alpha_sigma
            )
            
            # Outer median error
            outer_mask = gal['r'] >= np.median(gal['r'])
            if np.any(outer_mask):
                frac_diff = np.abs(v_model[outer_mask] - gal['v_obs'][outer_mask]) / (gal['v_obs'][outer_mask] + 1e-10)
                error = np.median(frac_diff)
                errors.append(error * weights['galaxies'])
    
    # Test on MW
    if 'mw' in data:
        mw = data['mw']
        v_model = g3_sigma_1d_galaxy(
            mw['r'], mw['v_bar'], mw['M_bary'], mw['r_half'], mw['sigma_mean'],
            S0, rc, gamma, beta, g_sat, sigma_star, alpha_sigma
        )
        
        frac_diff = np.abs(v_model - mw['v_obs']) / (mw['v_obs'] + 1e-10)
        error = np.median(frac_diff)
        errors.append(error * weights['mw'])
    
    # Test on clusters
    if 'clusters' in data:
        for cluster in data['clusters']:
            T_model = g3_sigma_cluster_temp(
                cluster['r'], cluster['rho_gas'], cluster['r_core'],
                S0, rc, gamma, beta, g_sat, sigma_star, alpha_sigma
            )
            
            # Temperature residuals
            valid = (T_model > 0) & (cluster['T_obs'] > 0)
            if np.any(valid):
                residuals = np.abs(T_model[valid] - cluster['T_obs'][valid]) / cluster['T_obs'][valid]
                error = np.median(residuals)
                errors.append(error * weights['clusters'])
    
    return np.mean(errors) if errors else np.inf

# ==============================================================================
# PARAMETER OPTIMIZATION
# ==============================================================================

def optimize_unified_parameters():
    """
    Find the best single parameter tuple for all regimes
    """
    
    logger.info("="*70)
    logger.info("OPTIMIZING UNIFIED G³-Σ PARAMETERS")
    logger.info("="*70)
    
    # Load test data
    logger.info("\nLoading test data...")
    
    data = {'galaxies': [], 'mw': None, 'clusters': []}
    
    # Load SPARC galaxies
    try:
        sparc_df = pd.read_parquet('data/sparc_rotmod_ltg.parquet')
        master_df = pd.read_parquet('data/sparc_master_clean.parquet')
        
        # Select diverse test galaxies
        test_galaxies = ['NGC2403', 'DDO154', 'NGC3198', 'NGC6946', 'NGC2841',
                        'NGC3521', 'UGC2259', 'NGC7814']
        
        for galaxy_name in test_galaxies[:5]:  # Use 5 for optimization
            if galaxy_name not in sparc_df['galaxy'].values:
                continue
                
            gal_rc = sparc_df[sparc_df['galaxy'] == galaxy_name]
            gal_props = master_df[master_df['galaxy'] == galaxy_name].iloc[0] if galaxy_name in master_df['galaxy'].values else {}
            
            # Extract data
            r = gal_rc['R_kpc'].values
            v_obs = gal_rc['Vobs_kms'].values
            v_gas = gal_rc['Vgas_kms'].values if 'Vgas_kms' in gal_rc else np.zeros(len(r))
            v_disk = gal_rc['Vdisk_kms'].values if 'Vdisk_kms' in gal_rc else np.zeros(len(r))
            v_bul = gal_rc.get('Vbul_kms', pd.Series(np.zeros(len(r)))).values
            v_bar = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
            
            M_bary = gal_props.get('M_bary', 1e10)
            r_half = gal_props.get('Rdisk', 3.0) * 1.68  # Convert to half-mass
            sigma_mean = M_bary / (np.pi * r_half**2) / 1e6  # Msun/pc^2
            
            data['galaxies'].append({
                'name': galaxy_name,
                'r': r,
                'v_obs': v_obs,
                'v_bar': v_bar,
                'M_bary': M_bary,
                'r_half': r_half,
                'sigma_mean': sigma_mean
            })
            
        logger.info(f"  Loaded {len(data['galaxies'])} SPARC galaxies")
        
    except Exception as e:
        logger.error(f"  Error loading SPARC data: {e}")
    
    # MW data
    r_mw = np.array([2, 4, 6, 8, 10, 12, 15, 20, 25])  # kpc
    v_obs_mw = np.array([200, 220, 235, 230, 225, 220, 215, 210, 205])  # km/s
    # Simplified MW baryon curve
    v_bar_mw = np.array([180, 190, 195, 190, 185, 180, 175, 170, 165])  # km/s
    
    data['mw'] = {
        'r': r_mw,
        'v_obs': v_obs_mw,
        'v_bar': v_bar_mw,
        'M_bary': 7.6e10,
        'r_half': 3.5,
        'sigma_mean': 7.6e10 / (np.pi * 3.5**2) / 1e6
    }
    logger.info("  Loaded Milky Way data")
    
    # Perseus cluster
    r_perseus = np.logspace(1, 3, 10)  # 10 to 1000 kpc
    T_obs_perseus = np.array([6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0])  # keV
    rho_gas_perseus = 1e-26 * (1 + (r_perseus/100)**2)**(-1.5)  # Simplified beta model
    
    data['clusters'].append({
        'name': 'Perseus',
        'r': r_perseus,
        'T_obs': T_obs_perseus,
        'rho_gas': rho_gas_perseus,
        'r_core': 100.0
    })
    logger.info("  Loaded Perseus cluster data")
    
    # ===========================================================================
    # OPTIMIZATION
    # ===========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("RUNNING OPTIMIZATION")
    logger.info("="*60)
    
    # Method 1: Grid search for coarse optimization
    logger.info("\nPhase 1: Coarse grid search...")
    
    param_grid = {
        'S0': [0.8e-4, 1.0e-4, 1.2e-4, 1.4e-4, 1.6e-4],
        'rc': [18, 22, 26, 30],
        'gamma': [0.4, 0.5, 0.6],
        'beta': [0.08, 0.10, 0.12],
        'g_sat': [2000, 2500, 3000],
        'sigma_star': [120, 150, 200],
        'alpha_sigma': [0.8, 1.0, 1.2]
    }
    
    # Test a subset of combinations
    best_params = None
    best_error = np.inf
    
    # Sample parameter combinations
    n_samples = 200  # Test 200 random combinations
    param_names = list(param_grid.keys())
    
    for i in range(n_samples):
        # Random sample from grid
        params = []
        for name in param_names:
            params.append(np.random.choice(param_grid[name]))
        
        error = compute_objective(params, data)
        
        if error < best_error:
            best_error = error
            best_params = params
            logger.info(f"  New best: error = {error:.4f}, params = {params}")
    
    logger.info(f"\nBest coarse params: {best_params}")
    logger.info(f"Best coarse error: {best_error:.4f}")
    
    # Method 2: Fine-tuning with differential evolution
    logger.info("\nPhase 2: Fine-tuning with differential evolution...")
    
    # Define bounds around best coarse params
    if best_params is not None:
        bounds = [
            (best_params[0] * 0.8, best_params[0] * 1.2),  # S0
            (best_params[1] * 0.85, best_params[1] * 1.15),  # rc
            (best_params[2] * 0.9, best_params[2] * 1.1),  # gamma
            (best_params[3] * 0.9, best_params[3] * 1.1),  # beta
            (best_params[4] * 0.9, best_params[4] * 1.1),  # g_sat
            (best_params[5] * 0.85, best_params[5] * 1.15),  # sigma_star
            (best_params[6] * 0.9, best_params[6] * 1.1),  # alpha_sigma
        ]
    else:
        # Default bounds
        bounds = [
            (5e-5, 2e-4),     # S0
            (15, 35),         # rc
            (0.3, 0.7),       # gamma
            (0.05, 0.15),     # beta
            (1500, 3500),     # g_sat
            (100, 250),       # sigma_star
            (0.5, 1.5),       # alpha_sigma
        ]
    
    result = differential_evolution(
        compute_objective,
        bounds,
        args=(data,),
        maxiter=50,
        popsize=10,
        tol=0.001,
        seed=42,
        disp=True
    )
    
    optimal_params = result.x
    optimal_error = result.fun
    
    # ===========================================================================
    # VALIDATION
    # ===========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION WITH OPTIMAL PARAMETERS")
    logger.info("="*60)
    
    S0, rc, gamma, beta, g_sat, sigma_star, alpha_sigma = optimal_params
    
    logger.info(f"\nOptimal parameters:")
    logger.info(f"  S0 = {S0:.3e}")
    logger.info(f"  rc = {rc:.1f} kpc")
    logger.info(f"  gamma = {gamma:.3f}")
    logger.info(f"  beta = {beta:.3f}")
    logger.info(f"  g_sat = {g_sat:.0f} (km/s)^2/kpc")
    logger.info(f"  sigma_star = {sigma_star:.0f} Msun/pc^2")
    logger.info(f"  alpha_sigma = {alpha_sigma:.2f}")
    
    # Test on individual systems
    logger.info("\nPerformance on test systems:")
    
    # Galaxies
    galaxy_accuracies = []
    for gal in data['galaxies']:
        v_model = g3_sigma_1d_galaxy(
            gal['r'], gal['v_bar'], gal['M_bary'], gal['r_half'], gal['sigma_mean'],
            S0, rc, gamma, beta, g_sat, sigma_star, alpha_sigma
        )
        
        outer_mask = gal['r'] >= np.median(gal['r'])
        if np.any(outer_mask):
            frac_diff = np.abs(v_model[outer_mask] - gal['v_obs'][outer_mask]) / gal['v_obs'][outer_mask]
            accuracy = 100 * (1 - np.median(frac_diff))
            galaxy_accuracies.append(accuracy)
            logger.info(f"  {gal['name']}: {accuracy:.1f}%")
    
    # MW
    mw = data['mw']
    v_model_mw = g3_sigma_1d_galaxy(
        mw['r'], mw['v_bar'], mw['M_bary'], mw['r_half'], mw['sigma_mean'],
        S0, rc, gamma, beta, g_sat, sigma_star, alpha_sigma
    )
    frac_diff = np.abs(v_model_mw - mw['v_obs']) / mw['v_obs']
    mw_accuracy = 100 * (1 - np.median(frac_diff))
    logger.info(f"  Milky Way: {mw_accuracy:.1f}%")
    
    # Perseus
    cluster = data['clusters'][0]
    T_model = g3_sigma_cluster_temp(
        cluster['r'], cluster['rho_gas'], cluster['r_core'],
        S0, rc, gamma, beta, g_sat, sigma_star, alpha_sigma
    )
    valid = (T_model > 0) & (cluster['T_obs'] > 0)
    if np.any(valid):
        residuals = np.abs(T_model[valid] - cluster['T_obs'][valid]) / cluster['T_obs'][valid]
        perseus_accuracy = 100 * (1 - np.median(residuals))
        logger.info(f"  Perseus: {perseus_accuracy:.1f}%")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    logger.info(f"\nGalaxies:")
    logger.info(f"  Mean accuracy: {np.mean(galaxy_accuracies):.1f}%")
    logger.info(f"  Median accuracy: {np.median(galaxy_accuracies):.1f}%")
    
    logger.info(f"\nMilky Way accuracy: {mw_accuracy:.1f}%")
    logger.info(f"\nPerseus accuracy: {perseus_accuracy:.1f}%")
    
    # Save results
    results = {
        'optimal_parameters': {
            'S0': float(S0),
            'rc_kpc': float(rc),
            'gamma': float(gamma),
            'beta': float(beta),
            'g_sat_kms2_per_kpc': float(g_sat),
            'sigma_star_Msun_pc2': float(sigma_star),
            'alpha_sigma': float(alpha_sigma)
        },
        'performance': {
            'galaxies_mean': float(np.mean(galaxy_accuracies)),
            'galaxies_median': float(np.median(galaxy_accuracies)),
            'mw': float(mw_accuracy),
            'perseus': float(perseus_accuracy),
            'combined_error': float(optimal_error)
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    output_file = Path('g3_sigma_optimal_params.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    results = optimize_unified_parameters()