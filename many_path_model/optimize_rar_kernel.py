"""
RAR-Driven Kernel Optimization
===============================

Optimizes path-spectrum kernel hyperparameters to minimize RAR scatter
with fixed g† = 1.2e-10 m/s² (literature value).

Key innovation: p exponent controls RAR slope at low acceleration.

Target: RAR scatter ≤ 0.15 dex on 20% holdout while maintaining:
- Newtonian limit (K < 0.01 at r < 0.1 kpc)
- Median APE ≤ 10% on rotation curves
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import time

from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams
from validation_suite import ValidationSuite

# Physical constants
KM_TO_M = 1000.0
KPC_TO_M = 3.0856776e19
G_DAGGER_LIT = 1.2e-10  # Fixed literature value [m/s²]

def compute_rar_metrics(df: pd.DataFrame, hp: PathSpectrumHyperparams, 
                       inclination_filter: bool = True) -> dict:
    """Compute RAR scatter and rotation curve APE for given hyperparameters"""
    
    kernel = PathSpectrumKernel(hp, use_cupy=False)
    
    # Collect RAR points and APE
    g_bar_all = []
    g_model_all = []
    ape_values = []
    
    for idx, galaxy in df.iterrows():
        # Inclination filter
        if inclination_filter:
            inclination = galaxy.get('Inc', galaxy.get('inclination', 45.0))
            if inclination < 30.0 or inclination > 70.0:
                continue
        
        r_all = galaxy['r_all']
        v_obs = galaxy['v_all']
        
        if len(r_all) < 3:
            continue
        
        # Compute g_bar (baryonic)
        v_disk = galaxy.get('v_disk_all', np.zeros_like(v_obs))
        v_bulge = galaxy.get('v_bulge_all', np.zeros_like(v_obs))
        v_gas = galaxy.get('v_gas_all', np.zeros_like(v_obs))
        
        if v_disk is None:
            v_disk = np.zeros_like(v_obs)
        if v_bulge is None:
            v_bulge = np.zeros_like(v_obs)
        if v_gas is None:
            v_gas = np.zeros_like(v_obs)
        
        # Quadrature method (correct)
        v_baryonic_km_s = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
        v_baryonic_m_s = v_baryonic_km_s * KM_TO_M
        r_m = r_all * KPC_TO_M
        g_bar = v_baryonic_m_s**2 / r_m
        
        # Compute many-path boost with actual g_bar
        BT = galaxy.get('BT', 0.0)
        bar_strength = galaxy.get('bar_strength', 0.0)
        K = kernel.many_path_boost_factor(r=r_all, v_circ=v_obs, g_bar=g_bar,
                                          BT=BT, bar_strength=bar_strength)
        
        # Model prediction
        g_model = g_bar * (1.0 + K)
        
        # Rotation curve APE
        v_model = np.sqrt(g_model * r_m) / KM_TO_M  # Back to km/s
        ape = np.mean(np.abs(v_model - v_obs) / v_obs) * 100
        ape_values.append(ape)
        
        # Filter valid RAR points
        mask = (g_bar > 1e-13) & (g_model > 1e-13) & np.isfinite(g_bar) & np.isfinite(g_model)
        
        if np.sum(mask) > 0:
            g_bar_all.extend(g_bar[mask])
            g_model_all.extend(g_model[mask])
    
    g_bar_arr = np.array(g_bar_all)
    g_model_arr = np.array(g_model_all)
    
    # RAR with FIXED g†
    def rar_function(g_bar, g_dagger=G_DAGGER_LIT):
        return g_bar / (1.0 - np.exp(-np.sqrt(g_bar / g_dagger)))
    
    g_rar_pred = rar_function(g_bar_arr)
    log_residuals = np.log10(g_model_arr) - np.log10(g_rar_pred)
    rar_scatter = np.std(log_residuals)
    rar_bias = np.mean(log_residuals)
    
    median_ape = np.median(ape_values) if len(ape_values) > 0 else 999.0
    
    return {
        'rar_scatter': rar_scatter,
        'rar_bias': rar_bias,
        'median_ape': median_ape,
        'n_rar_points': len(g_bar_arr),
        'n_galaxies': len(ape_values)
    }

def objective_function(x, train_df):
    """Optimization objective: minimize RAR scatter + APE penalty"""
    
    # Unpack parameters
    p, L_0, beta_bulge, alpha_shear, gamma_bar, A_0, n_coh = x
    
    # Create hyperparameters
    hp = PathSpectrumHyperparams(
        p=p,
        L_0=L_0,
        beta_bulge=beta_bulge,
        alpha_shear=alpha_shear,
        gamma_bar=gamma_bar,
        A_0=A_0,
        n_coh=n_coh,
        g_dagger=G_DAGGER_LIT  # Fixed
    )
    
    # Compute metrics
    metrics = compute_rar_metrics(train_df, hp)
    
    # Loss = RAR scatter + small APE penalty
    # We prioritize RAR scatter but keep rotation curves reasonable
    loss = metrics['rar_scatter'] + 0.02 * max(0, metrics['median_ape'] - 15.0)
    
    return loss

def run_optimization(train_df: pd.DataFrame, n_iter: int = 60):
    """Run differential evolution optimization"""
    
    print("="*80)
    print("RAR-DRIVEN KERNEL OPTIMIZATION")
    print("="*80)
    print(f"\nTraining galaxies: {len(train_df)}")
    print(f"Fixed g† = {G_DAGGER_LIT:.2e} m/s²")
    print(f"Target: RAR scatter ≤ 0.15 dex\n")
    
    # Parameter bounds: (p, L_0, beta_bulge, alpha_shear, gamma_bar, A_0, n_coh)
    bounds = [
        (0.3, 1.2),      # p: RAR slope exponent
        (1.0, 5.0),      # L_0: baseline coherence length [kpc]
        (0.5, 2.0),      # beta_bulge: bulge suppression
        (0.01, 0.15),    # alpha_shear: shear suppression rate
        (0.5, 3.5),      # gamma_bar: bar suppression
        (0.5, 3.0),      # A_0: global amplitude
        (0.5, 2.0)       # n_coh: coherence damping exponent
    ]
    
    print("Parameter bounds:")
    param_names = ['p', 'L_0', 'beta_bulge', 'alpha_shear', 'gamma_bar', 'A_0', 'n_coh']
    for name, (low, high) in zip(param_names, bounds):
        print(f"  {name:<15} [{low:.3f}, {high:.3f}]")
    print()
    
    # Run optimization
    print(f"Running differential evolution ({n_iter} iterations)...")
    print("-" * 80)
    
    start_time = time.time()
    
    result = differential_evolution(
        objective_function,
        bounds,
        args=(train_df,),
        maxiter=n_iter,
        popsize=15,
        tol=1e-4,
        atol=1e-4,
        seed=42,
        disp=True,
        workers=1
    )
    
    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.1f} seconds")
    print(f"Best loss: {result.fun:.4f}")
    print()
    
    # Extract best parameters
    p, L_0, beta_bulge, alpha_shear, gamma_bar, A_0, n_coh = result.x
    
    best_hp = PathSpectrumHyperparams(
        p=p,
        L_0=L_0,
        beta_bulge=beta_bulge,
        alpha_shear=alpha_shear,
        gamma_bar=gamma_bar,
        A_0=A_0,
        n_coh=n_coh,
        g_dagger=G_DAGGER_LIT
    )
    
    return best_hp, result

def validate_on_holdout(best_hp: PathSpectrumHyperparams, train_df: pd.DataFrame, 
                        test_df: pd.DataFrame):
    """Validate optimized kernel on holdout set"""
    
    print("="*80)
    print("VALIDATION ON HOLDOUT SET")
    print("="*80)
    
    # Train metrics
    print("\nTrain set:")
    train_metrics = compute_rar_metrics(train_df, best_hp)
    print(f"  RAR scatter: {train_metrics['rar_scatter']:.3f} dex")
    print(f"  RAR bias: {train_metrics['rar_bias']:.3f} dex")
    print(f"  Median APE: {train_metrics['median_ape']:.1f}%")
    print(f"  Galaxies: {train_metrics['n_galaxies']}")
    
    # Test metrics
    print("\nTest set (HOLDOUT):")
    test_metrics = compute_rar_metrics(test_df, best_hp)
    print(f"  RAR scatter: {test_metrics['rar_scatter']:.3f} dex")
    print(f"  RAR bias: {test_metrics['rar_bias']:.3f} dex")
    print(f"  Median APE: {test_metrics['median_ape']:.1f}%")
    print(f"  Galaxies: {test_metrics['n_galaxies']}")
    
    # Pass/fail
    print("\n" + "-"*80)
    if test_metrics['rar_scatter'] <= 0.15:
        print("✅ RAR SCATTER TARGET MET (≤ 0.15 dex)")
    else:
        print(f"⚠️  RAR scatter {test_metrics['rar_scatter']:.3f} dex (target ≤ 0.15)")
    
    if test_metrics['median_ape'] <= 10.0:
        print("✅ ROTATION CURVE APE TARGET MET (≤ 10%)")
    else:
        print(f"⚠️  Median APE {test_metrics['median_ape']:.1f}% (target ≤ 10%)")
    
    return train_metrics, test_metrics

def main():
    # Load SPARC data
    output_dir = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results")
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    
    # 80/20 stratified split
    train_df, test_df = suite.perform_train_test_split()
    
    # Run optimization on training set
    best_hp, result = run_optimization(train_df, n_iter=60)
    
    # Print optimal hyperparameters
    print("="*80)
    print("OPTIMAL HYPERPARAMETERS")
    print("="*80)
    for key, value in best_hp.to_dict().items():
        print(f"  {key:<15} = {value:.6f}")
    print("="*80)
    
    # Validate on holdout
    train_metrics, test_metrics = validate_on_holdout(best_hp, train_df, test_df)
    
    # Test Newtonian limit
    print("\n" + "="*80)
    print("NEWTONIAN LIMIT CHECK")
    print("="*80)
    kernel = PathSpectrumKernel(best_hp, use_cupy=False)
    r_small = np.array([0.001, 0.01, 0.1])
    v_small = np.array([50, 100, 150])
    K_small = kernel.many_path_boost_factor(r_small, v_small)
    
    for i in range(len(r_small)):
        print(f"  r = {r_small[i]:6.3f} kpc: K = {K_small[i]:.6f} (boost = {K_small[i]*100:.3f}%)")
    
    if np.all(K_small < 0.01):
        print("\n✅ Newtonian limit preserved (K < 1% at small r)")
    else:
        print("\n⚠️  WARNING: Newtonian limit may be violated!")
    
    # Save results
    results_path = output_dir / "rar_optimization_results.json"
    import json
    results_dict = {
        'hyperparameters': best_hp.to_dict(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'optimization_info': {
            'n_iter': 60,
            'success': result.success,
            'message': result.message
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Results saved to {results_path}")

if __name__ == "__main__":
    main()
