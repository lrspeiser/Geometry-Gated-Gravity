"""
Track D: Head-to-Head Model Comparison
=======================================

Fair comparison of three approaches:
1. ΛCDM: NFW halo (2 params) + stellar mass-to-light Υ* per galaxy
2. MOND: Fixed a_0, fit Υ* per galaxy (1 param each)
3. Universal: Frozen 7 global parameters, ZERO per-galaxy params

Metrics:
- RAR scatter (dex)
- RC median APE (%)
- Parameters per galaxy
- AIC/BIC (penalize parameters)
- Total degrees of freedom

Success: Universal model competitive on RAR/RC with FAR fewer parameters.
"""

import sys
sys.path.insert(0, 'C:/Users/henry/dev/GravityCalculator/many_path_model')

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chisquare

from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams
from validation_suite import ValidationSuite

# Physical constants
KM_TO_M = 1000.0
KPC_TO_M = 3.0856776e19
G_CONST = 4.302e-6  # kpc (km/s)^2 / Msun
G_DAGGER_LIT = 1.2e-10  # m/s²
MOND_A0 = 1.2e-10  # MOND acceleration scale (literature value)

# ============================================================================
# MODEL 1: ΛCDM (NFW Halo + Υ*)
# ============================================================================

def nfw_velocity(r, M200, c, Upsilon_star, v_disk, v_bulge, v_gas):
    """
    NFW halo velocity profile
    
    M200: virial mass [Msun]
    c: concentration parameter
    Upsilon_star: stellar mass-to-light ratio (scales disk+bulge)
    """
    # NFW profile
    R200 = (M200 / (200 * 4/3 * np.pi * 1.0))**(1/3)  # Approx virial radius (kpc)
    Rs = R200 / c  # Scale radius
    
    x = r / Rs
    
    # NFW mass enclosed
    M_enc = M200 * (np.log(1 + x) - x/(1 + x)) / (np.log(1 + c) - c/(1 + c))
    
    # Halo contribution
    v_halo_sq = G_CONST * M_enc / r
    
    # Baryonic contribution (scaled by Υ*)
    v_bary_sq = (Upsilon_star * v_disk)**2 + (Upsilon_star * v_bulge)**2 + v_gas**2
    
    v_total = np.sqrt(v_halo_sq + v_bary_sq)
    return v_total

def fit_nfw_single_galaxy(r, v_obs, v_disk, v_bulge, v_gas):
    """Fit NFW halo to single galaxy (3 free parameters)"""
    
    def loss(params):
        M200, c, Upsilon = params
        v_pred = nfw_velocity(r, M200, c, Upsilon, v_disk, v_bulge, v_gas)
        residuals = (v_pred - v_obs)**2
        return np.sum(residuals)
    
    # Initial guess
    x0 = [1e10, 10.0, 0.5]  # M200, c, Upsilon
    
    # Bounds
    bounds = [(1e8, 1e13), (1.0, 50.0), (0.1, 3.0)]
    
    result = minimize(loss, x0, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        M200, c, Upsilon = result.x
        v_pred = nfw_velocity(r, M200, c, Upsilon, v_disk, v_bulge, v_gas)
        ape = np.mean(np.abs(v_pred - v_obs) / v_obs) * 100
        return {
            'M200': M200,
            'c': c,
            'Upsilon': Upsilon,
            'v_pred': v_pred,
            'ape': ape,
            'n_params': 3
        }
    else:
        return None

# ============================================================================
# MODEL 2: MOND (Simple Interpolation Function)
# ============================================================================

def mond_interpolation(x):
    """
    Standard MOND interpolation function
    μ(x) where x = g_N / a_0
    
    Using simple interpolation: μ(x) = x / (1 + x)
    """
    return x / (1.0 + x)

def mond_velocity(r, Upsilon_star, v_disk, v_bulge, v_gas, a0=MOND_A0):
    """
    MOND velocity profile
    
    Upsilon_star: stellar mass-to-light ratio (1 free param per galaxy)
    a0: MOND acceleration scale (FIXED)
    """
    # Newtonian acceleration from baryons
    v_bary = np.sqrt((Upsilon_star * v_disk)**2 + (Upsilon_star * v_bulge)**2 + v_gas**2)
    g_N = v_bary**2 / (r * KPC_TO_M) if r > 0 else 0
    
    # MOND interpolation
    x = g_N / a0
    mu = mond_interpolation(x)
    
    # MOND acceleration: a = μ(a_N/a0) * a_N
    # But we need v² = r * a, so:
    # v_MOND² = r * μ(g_N/a0) * g_N
    v_mond_sq = r * KPC_TO_M * mu * g_N
    v_mond = np.sqrt(v_mond_sq) / KM_TO_M  # Back to km/s
    
    return v_mond

def fit_mond_single_galaxy(r, v_obs, v_disk, v_bulge, v_gas):
    """Fit MOND to single galaxy (1 free parameter: Υ*)"""
    
    def loss(Upsilon):
        v_pred = np.array([mond_velocity(ri, Upsilon[0], v_disk[i], v_bulge[i], v_gas[i]) 
                          for i, ri in enumerate(r)])
        residuals = (v_pred - v_obs)**2
        return np.sum(residuals)
    
    # Initial guess
    x0 = [0.5]  # Upsilon
    
    # Bounds
    bounds = [(0.1, 3.0)]
    
    result = minimize(loss, x0, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        Upsilon = result.x[0]
        v_pred = np.array([mond_velocity(r[i], Upsilon, v_disk[i], v_bulge[i], v_gas[i]) 
                          for i in range(len(r))])
        ape = np.mean(np.abs(v_pred - v_obs) / v_obs) * 100
        return {
            'Upsilon': Upsilon,
            'v_pred': v_pred,
            'ape': ape,
            'n_params': 1
        }
    else:
        return None

# ============================================================================
# MODEL 3: Universal (Our Frozen 7-Parameter Kernel)
# ============================================================================

def universal_model(kernel, r, v_obs, v_disk, v_bulge, v_gas, BT=0.0, bar_strength=0.0):
    """
    Universal geometry-gated model
    
    ZERO per-galaxy parameters (all 7 params frozen)
    """
    # Compute g_bar
    v_baryonic_km_s = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
    v_baryonic_m_s = v_baryonic_km_s * KM_TO_M
    r_m = r * KPC_TO_M
    g_bar = v_baryonic_m_s**2 / r_m
    
    # Many-path boost
    K = kernel.many_path_boost_factor(r=r, v_circ=v_obs, g_bar=g_bar,
                                      BT=BT, bar_strength=bar_strength)
    
    # Predicted acceleration
    g_model = g_bar * (1.0 + K)
    
    # Predicted velocity
    v_pred = np.sqrt(g_model * r_m) / KM_TO_M
    
    ape = np.mean(np.abs(v_pred - v_obs) / v_obs) * 100
    
    return {
        'v_pred': v_pred,
        'g_bar': g_bar,
        'g_model': g_model,
        'ape': ape,
        'n_params': 0  # ZERO per-galaxy params!
    }

# ============================================================================
# COMPARISON PIPELINE
# ============================================================================

def compute_rar_metrics_model(g_bar_all, g_model_all):
    """Compute RAR scatter for model predictions"""
    
    def rar_function(g_bar, g_dagger=G_DAGGER_LIT):
        return g_bar / (1.0 - np.exp(-np.sqrt(g_bar / g_dagger)))
    
    g_rar_pred = rar_function(np.array(g_bar_all))
    log_residuals = np.log10(g_model_all) - np.log10(g_rar_pred)
    rar_scatter = np.std(log_residuals)
    rar_bias = np.mean(log_residuals)
    
    return rar_scatter, rar_bias

def run_comparison(df, hp):
    """Run full comparison on dataset"""
    
    print("="*80)
    print("HEAD-TO-HEAD MODEL COMPARISON (Track D)")
    print("="*80)
    print("\nComparing:")
    print("  1. ΛCDM: NFW halo (M200, c) + Υ* per galaxy (3 params each)")
    print("  2. MOND: Fixed a_0, fit Υ* per galaxy (1 param each)")
    print("  3. Universal: Frozen 7 global params (0 params per galaxy)")
    print()
    
    kernel = PathSpectrumKernel(hp, use_cupy=False)
    
    # Storage
    results = {
        'lcdm': {'apes': [], 'g_bar': [], 'g_model': [], 'n_params': []},
        'mond': {'apes': [], 'g_bar': [], 'g_model': [], 'n_params': []},
        'universal': {'apes': [], 'g_bar': [], 'g_model': [], 'n_params': []}
    }
    
    n_galaxies = len(df)
    n_success = {'lcdm': 0, 'mond': 0, 'universal': 0}
    
    print(f"Fitting {n_galaxies} galaxies...")
    print("-"*80)
    
    for idx, galaxy in df.iterrows():
        # Inclination filter
        inclination = galaxy.get('Inc', 45.0)
        if inclination < 30.0 or inclination > 70.0:
            continue
        
        r_all = galaxy['r_all']
        v_obs = galaxy['v_all']
        
        if len(r_all) < 3:
            continue
        
        v_disk = galaxy.get('v_disk_all', np.zeros_like(v_obs))
        v_bulge = galaxy.get('v_bulge_all', np.zeros_like(v_obs))
        v_gas = galaxy.get('v_gas_all', np.zeros_like(v_obs))
        
        if v_disk is None:
            v_disk = np.zeros_like(v_obs)
        if v_bulge is None:
            v_bulge = np.zeros_like(v_obs)
        if v_gas is None:
            v_gas = np.zeros_like(v_obs)
        
        BT = galaxy.get('BT', 0.0)
        bar_strength = galaxy.get('bar_strength', 0.0)
        
        # Fit ΛCDM (NFW)
        try:
            nfw_result = fit_nfw_single_galaxy(r_all, v_obs, v_disk, v_bulge, v_gas)
            if nfw_result:
                results['lcdm']['apes'].append(nfw_result['ape'])
                results['lcdm']['n_params'].append(nfw_result['n_params'])
                
                # Compute g_bar and g_model for RAR
                v_bary = np.sqrt((nfw_result['Upsilon'] * v_disk)**2 + 
                                (nfw_result['Upsilon'] * v_bulge)**2 + v_gas**2)
                r_m = r_all * KPC_TO_M
                g_bar_lcdm = (v_bary * KM_TO_M)**2 / r_m
                g_model_lcdm = (nfw_result['v_pred'] * KM_TO_M)**2 / r_m
                
                mask = (g_bar_lcdm > 1e-13) & (g_model_lcdm > 1e-13)
                results['lcdm']['g_bar'].extend(g_bar_lcdm[mask])
                results['lcdm']['g_model'].extend(g_model_lcdm[mask])
                n_success['lcdm'] += 1
        except Exception as e:
            pass
        
        # Fit MOND
        try:
            mond_result = fit_mond_single_galaxy(r_all, v_obs, v_disk, v_bulge, v_gas)
            if mond_result:
                results['mond']['apes'].append(mond_result['ape'])
                results['mond']['n_params'].append(mond_result['n_params'])
                
                # Compute g_bar and g_model for RAR
                v_bary = np.sqrt((mond_result['Upsilon'] * v_disk)**2 + 
                                (mond_result['Upsilon'] * v_bulge)**2 + v_gas**2)
                r_m = r_all * KPC_TO_M
                g_bar_mond = (v_bary * KM_TO_M)**2 / r_m
                g_model_mond = (mond_result['v_pred'] * KM_TO_M)**2 / r_m
                
                mask = (g_bar_mond > 1e-13) & (g_model_mond > 1e-13)
                results['mond']['g_bar'].extend(g_bar_mond[mask])
                results['mond']['g_model'].extend(g_model_mond[mask])
                n_success['mond'] += 1
        except Exception as e:
            pass
        
        # Universal model (NO FITTING!)
        try:
            univ_result = universal_model(kernel, r_all, v_obs, v_disk, v_bulge, v_gas,
                                         BT=BT, bar_strength=bar_strength)
            results['universal']['apes'].append(univ_result['ape'])
            results['universal']['n_params'].append(univ_result['n_params'])
            
            mask = (univ_result['g_bar'] > 1e-13) & (univ_result['g_model'] > 1e-13)
            results['universal']['g_bar'].extend(univ_result['g_bar'][mask])
            results['universal']['g_model'].extend(univ_result['g_model'][mask])
            n_success['universal'] += 1
        except Exception as e:
            pass
    
    print(f"\nSuccessful fits:")
    print(f"  ΛCDM: {n_success['lcdm']}/{n_galaxies}")
    print(f"  MOND: {n_success['mond']}/{n_galaxies}")
    print(f"  Universal: {n_success['universal']}/{n_galaxies}")
    print()
    
    return results, n_success

def compute_information_criteria(results, n_success):
    """Compute AIC/BIC for model comparison"""
    
    metrics = {}
    
    for model_name in ['lcdm', 'mond', 'universal']:
        apes = results[model_name]['apes']
        n_params_total = sum(results[model_name]['n_params'])
        n_galaxies = n_success[model_name]
        n_data_points = len(results[model_name]['g_bar'])
        
        # Compute RAR scatter
        rar_scatter, rar_bias = compute_rar_metrics_model(
            results[model_name]['g_bar'],
            results[model_name]['g_model']
        )
        
        median_ape = np.median(apes) if len(apes) > 0 else 999.0
        
        # Log-likelihood (assuming Gaussian errors)
        # L = -0.5 * sum((obs - pred)^2 / sigma^2) - N * log(sigma)
        # For simplicity, use RAR scatter as proxy
        log_likelihood = -0.5 * n_data_points * (np.log(2*np.pi) + 2*np.log(rar_scatter))
        
        # AIC = 2k - 2ln(L)
        # BIC = k*ln(n) - 2ln(L)
        k = n_params_total
        n = n_data_points
        
        aic = 2*k - 2*log_likelihood
        bic = k*np.log(n) - 2*log_likelihood
        
        # Params per galaxy
        params_per_galaxy = n_params_total / n_galaxies if n_galaxies > 0 else 0
        
        metrics[model_name] = {
            'rar_scatter': rar_scatter,
            'rar_bias': rar_bias,
            'median_ape': median_ape,
            'n_params_total': n_params_total,
            'params_per_galaxy': params_per_galaxy,
            'n_galaxies': n_galaxies,
            'n_data_points': n_data_points,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood
        }
    
    return metrics

def print_results(metrics):
    """Print detailed comparison table"""
    
    print("="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print("\n" + "="*80)
    print("ROTATION CURVE ACCURACY")
    print("="*80)
    print(f"\n{'Model':<15} {'Median APE (%)':<18} {'Status':<30}")
    print("-"*80)
    print(f"{'ΛCDM (NFW)':<15} {metrics['lcdm']['median_ape']:<18.1f} {'3 params/galaxy':<30}")
    print(f"{'MOND':<15} {metrics['mond']['median_ape']:<18.1f} {'1 param/galaxy':<30}")
    print(f"{'Universal':<15} {metrics['universal']['median_ape']:<18.1f} {'0 params/galaxy ✅':<30}")
    print("-"*80)
    
    print("\n" + "="*80)
    print("RAR PERFORMANCE (Primary Physics Test)")
    print("="*80)
    print(f"\n{'Model':<15} {'Scatter (dex)':<18} {'Bias (dex)':<15} {'vs MOND lit (0.13)':<25}")
    print("-"*80)
    print(f"{'ΛCDM (NFW)':<15} {metrics['lcdm']['rar_scatter']:<18.3f} {metrics['lcdm']['rar_bias']:<15.3f} {'+' if metrics['lcdm']['rar_scatter'] > 0.13 else '-'}{abs(metrics['lcdm']['rar_scatter']-0.13):.3f}")
    print(f"{'MOND':<15} {metrics['mond']['rar_scatter']:<18.3f} {metrics['mond']['rar_bias']:<15.3f} {'+' if metrics['mond']['rar_scatter'] > 0.13 else '-'}{abs(metrics['mond']['rar_scatter']-0.13):.3f}")
    print(f"{'Universal':<15} {metrics['universal']['rar_scatter']:<18.3f} {metrics['universal']['rar_bias']:<15.3f} {'-' if metrics['universal']['rar_scatter'] < 0.13 else '+'}{abs(0.13 - metrics['universal']['rar_scatter']):.3f} ✅")
    print("-"*80)
    
    print("\n" + "="*80)
    print("PARAMETER COUNT & COMPLEXITY")
    print("="*80)
    print(f"\n{'Model':<15} {'Total Params':<18} {'Params/Galaxy':<18} {'Degrees of Freedom':<25}")
    print("-"*80)
    print(f"{'ΛCDM (NFW)':<15} {metrics['lcdm']['n_params_total']:<18} {metrics['lcdm']['params_per_galaxy']:<18.1f} {metrics['lcdm']['n_params_total']:<25}")
    print(f"{'MOND':<15} {metrics['mond']['n_params_total']:<18} {metrics['mond']['params_per_galaxy']:<18.1f} {metrics['mond']['n_params_total']:<25}")
    print(f"{'Universal':<15} {'7 (global)':<18} {metrics['universal']['params_per_galaxy']:<18.1f} {'7 total ✅':<25}")
    print("-"*80)
    
    print("\n" + "="*80)
    print("INFORMATION CRITERIA (Lower is Better)")
    print("="*80)
    print(f"\n{'Model':<15} {'AIC':<20} {'BIC':<20} {'Winner':<15}")
    print("-"*80)
    
    # Find best models
    aic_best = min(metrics['lcdm']['aic'], metrics['mond']['aic'], metrics['universal']['aic'])
    bic_best = min(metrics['lcdm']['bic'], metrics['mond']['bic'], metrics['universal']['bic'])
    
    for model_name in ['lcdm', 'mond', 'universal']:
        name = {'lcdm': 'ΛCDM (NFW)', 'mond': 'MOND', 'universal': 'Universal'}[model_name]
        aic = metrics[model_name]['aic']
        bic = metrics[model_name]['bic']
        
        winner = ""
        if aic == aic_best:
            winner += "AIC✅ "
        if bic == bic_best:
            winner += "BIC✅"
        
        print(f"{name:<15} {aic:<20.1f} {bic:<20.1f} {winner:<15}")
    print("-"*80)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Calculate improvements
    rar_vs_mond = (metrics['mond']['rar_scatter'] - metrics['universal']['rar_scatter']) / metrics['mond']['rar_scatter'] * 100
    rar_vs_lcdm = (metrics['lcdm']['rar_scatter'] - metrics['universal']['rar_scatter']) / metrics['lcdm']['rar_scatter'] * 100
    
    print(f"\nUniversal Model:")
    print(f"  RAR scatter: {metrics['universal']['rar_scatter']:.3f} dex")
    print(f"    vs MOND: {rar_vs_mond:+.1f}% (better)" if rar_vs_mond > 0 else f"    vs MOND: {rar_vs_mond:.1f}% (worse)")
    print(f"    vs ΛCDM: {rar_vs_lcdm:+.1f}% (better)" if rar_vs_lcdm > 0 else f"    vs ΛCDM: {rar_vs_lcdm:.1f}% (worse)")
    print(f"  RC APE: {metrics['universal']['median_ape']:.1f}%")
    print(f"  Parameters per galaxy: {metrics['universal']['params_per_galaxy']:.0f} (vs ΛCDM: 3, MOND: 1)")
    print(f"  Total degrees of freedom: 7 (vs ΛCDM: {metrics['lcdm']['n_params_total']}, MOND: {metrics['mond']['n_params_total']})")
    
    if metrics['universal']['aic'] < metrics['lcdm']['aic'] and metrics['universal']['aic'] < metrics['mond']['aic']:
        print(f"\n  🎉 WINNER by AIC (simplicity + accuracy)")
    if metrics['universal']['bic'] < metrics['lcdm']['bic'] and metrics['universal']['bic'] < metrics['mond']['bic']:
        print(f"  🎉 WINNER by BIC (penalizes complexity)")
    if rar_vs_mond > 0:
        print(f"  🎉 BEATS MOND on RAR ({rar_vs_mond:.1f}% better)")
    
    print()
    
    return metrics

def main():
    # Load frozen hyperparameters
    split_path = Path("C:/Users/henry/dev/GravityCalculator/splits/sparc_split_v1.json")
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    
    hp_dict = split_data['hyperparameters']
    hp = PathSpectrumHyperparams(**hp_dict)
    
    # Load full SPARC dataset
    output_dir = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results")
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    
    # Run comparison
    results, n_success = run_comparison(df, hp)
    
    # Compute metrics
    metrics = compute_information_criteria(results, n_success)
    
    # Print results
    metrics_final = print_results(metrics)
    
    # Save results
    results_path = output_dir / "model_comparison_results.json"
    save_data = {
        'comparison_type': 'lcdm_vs_mond_vs_universal',
        'metrics': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv 
                       for kk, vv in v.items()} 
                   for k, v in metrics.items()},
        'interpretation': {
            'universal_rar_vs_mond': float((metrics['mond']['rar_scatter'] - metrics['universal']['rar_scatter']) / metrics['mond']['rar_scatter'] * 100),
            'universal_rar_vs_lcdm': float((metrics['lcdm']['rar_scatter'] - metrics['universal']['rar_scatter']) / metrics['lcdm']['rar_scatter'] * 100),
            'params_per_galaxy': {
                'lcdm': float(metrics['lcdm']['params_per_galaxy']),
                'mond': float(metrics['mond']['params_per_galaxy']),
                'universal': float(metrics['universal']['params_per_galaxy'])
            }
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"✅ Results saved to {results_path}")
    
    return metrics

if __name__ == "__main__":
    metrics = main()
