"""
Test A_0 calibration to match RAR g† = 1.2e-10 m/s²

This script tests different A_0 values to find the optimal amplitude
that brings our model's g† in line with McGaugh et al. 2016.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams
from validation_suite import ValidationSuite

# Physical constants
KM_TO_M = 1000.0  # km/s to m/s
KPC_TO_M = 3.0856776e19  # kpc to meters

def compute_rar_with_A0(df: pd.DataFrame, A_0: float) -> tuple:
    """Compute RAR g† with specified A_0 value"""
    
    # Initialize kernel with current hyperparameters + A_0
    hp = PathSpectrumHyperparams(
        L_0=1.82, 
        beta_bulge=1.09, 
        alpha_shear=0.056, 
        gamma_bar=1.06,
        A_0=A_0  # <<< Test parameter
    )
    kernel = PathSpectrumKernel(hp, use_cupy=False)
    
    # Collect all RAR points
    g_bar_all = []
    g_model_all = []
    
    for idx, galaxy in df.iterrows():
        # Inclination filter: 30° < i < 70°
        inclination = galaxy.get('Inc', galaxy.get('inclination', 45.0))
        if inclination < 30.0 or inclination > 70.0:
            continue
        
        r_all = galaxy['r_all']
        v_all = galaxy['v_all']
        
        if len(r_all) < 3:
            continue
        
        # Compute g_bar (baryonic)
        v_disk = galaxy.get('v_disk_all', np.zeros_like(v_all))
        v_bulge = galaxy.get('v_bulge_all', np.zeros_like(v_all))
        v_gas = galaxy.get('v_gas_all', np.zeros_like(v_all))
        
        if v_disk is None:
            v_disk = np.zeros_like(v_all)
        if v_bulge is None:
            v_bulge = np.zeros_like(v_all)
        if v_gas is None:
            v_gas = np.zeros_like(v_all)
        
        # Quadrature method (correct)
        v_baryonic_km_s = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
        v_baryonic_m_s = v_baryonic_km_s * KM_TO_M
        r_m = r_all * KPC_TO_M
        g_bar = v_baryonic_m_s**2 / r_m
        
        # Compute many-path boost with A_0
        BT = galaxy.get('BT', 0.0)
        bar_strength = galaxy.get('bar_strength', 0.0)
        K = kernel.many_path_boost_factor(r=r_all, v_circ=v_all, BT=BT, bar_strength=bar_strength)
        
        # Model prediction
        g_model = g_bar * (1.0 + K)
        
        # Filter valid points
        mask = (g_bar > 1e-13) & (g_model > 1e-13) & np.isfinite(g_bar) & np.isfinite(g_model)
        
        if np.sum(mask) > 0:
            g_bar_all.extend(g_bar[mask])
            g_model_all.extend(g_model[mask])
    
    g_bar_arr = np.array(g_bar_all)
    g_model_arr = np.array(g_model_all)
    
    # Fit RAR: g_model = g_bar / (1 - exp(-sqrt(g_bar/g†)))
    def rar_function(g_bar, g_dagger):
        return g_bar / (1.0 - np.exp(-np.sqrt(g_bar / g_dagger)))
    
    def rar_residuals(g_dagger):
        g_pred = rar_function(g_bar_arr, g_dagger)
        residuals = np.log10(g_model_arr) - np.log10(g_pred)
        return np.std(residuals)
    
    # Optimize g†
    result = minimize_scalar(rar_residuals, bounds=(1e-12, 1e-9), method='bounded')
    g_dagger_fit = result.x
    
    # Compute scatter
    g_model_pred = rar_function(g_bar_arr, g_dagger_fit)
    log_residuals = np.log10(g_model_arr) - np.log10(g_model_pred)
    scatter = np.std(log_residuals)
    
    return g_dagger_fit, scatter, len(g_bar_arr)

def main():
    print("="*80)
    print("A_0 CALIBRATION TEST FOR RAR MATCHING")
    print("="*80)
    print("\nGoal: Find A_0 such that fitted g† ≈ 1.2e-10 m/s²")
    print("Literature value from McGaugh et al. 2016\n")
    
    # Load SPARC data using ValidationSuite loader
    output_dir = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results")
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    print(f"\nLoaded {len(df)} SPARC galaxies for A_0 calibration\n")
    
    # Test A_0 values
    A_0_values = [0.25, 0.30, 0.31, 0.33, 0.35, 0.40, 0.50, 1.0]
    results = []
    
    print("Testing A_0 values:")
    print("-" * 80)
    print(f"{'A_0':<8} {'g† (m/s²)':<15} {'Ratio vs Lit':<15} {'Scatter (dex)':<15} {'N points'}")
    print("-" * 80)
    
    for A_0 in A_0_values:
        g_dagger, scatter, n_points = compute_rar_with_A0(df, A_0)
        ratio = g_dagger / 1.2e-10
        results.append({
            'A_0': A_0,
            'g_dagger': g_dagger,
            'ratio': ratio,
            'scatter': scatter,
            'n_points': n_points
        })
        print(f"{A_0:<8.2f} {g_dagger:<15.2e} {ratio:<15.2f} {scatter:<15.3f} {n_points}")
    
    print("-" * 80)
    
    # Find best A_0 (closest ratio to 1.0)
    results_df = pd.DataFrame(results)
    best_idx = np.argmin(np.abs(results_df['ratio'] - 1.0))
    best_result = results_df.iloc[best_idx]
    
    print(f"\n✅ OPTIMAL A_0 = {best_result['A_0']:.2f}")
    print(f"   g† = {best_result['g_dagger']:.2e} m/s²")
    print(f"   Ratio vs literature: {best_result['ratio']:.2f}×")
    print(f"   RAR scatter: {best_result['scatter']:.3f} dex")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: g† vs A_0
    ax1.plot(results_df['A_0'], results_df['g_dagger'], 'o-', linewidth=2, markersize=8)
    ax1.axhline(1.2e-10, color='red', linestyle='--', linewidth=2, label='Literature g†')
    ax1.set_xlabel('A_0', fontsize=14)
    ax1.set_ylabel('Fitted g† (m/s²)', fontsize=14)
    ax1.set_title('RAR Acceleration Scale vs Amplitude', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Scatter vs A_0
    ax2.plot(results_df['A_0'], results_df['scatter'], 'o-', linewidth=2, markersize=8, color='orange')
    ax2.axhline(0.15, color='red', linestyle='--', linewidth=2, label='Target scatter (0.15 dex)')
    ax2.set_xlabel('A_0', fontsize=14)
    ax2.set_ylabel('RAR Scatter (dex)', fontsize=14)
    ax2.set_title('RAR Scatter vs Amplitude', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results/A0_calibration_test.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved calibration plot to {output_path}")

if __name__ == "__main__":
    main()
