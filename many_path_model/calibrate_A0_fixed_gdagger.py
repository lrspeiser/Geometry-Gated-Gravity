"""
A_0 Calibration with FIXED g† = 1.2e-10 m/s²

The correct approach: don't fit g† as a free parameter.
Instead, use the literature value and find A_0 that minimizes scatter.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams
from validation_suite import ValidationSuite

# Physical constants
KM_TO_M = 1000.0
KPC_TO_M = 3.0856776e19

# FIXED g† from McGaugh et al. 2016
G_DAGGER_LIT = 1.2e-10  # m/s²

def compute_rar_scatter_fixed_gdagger(df: pd.DataFrame, A_0: float) -> tuple:
    """Compute RAR scatter with FIXED g† = 1.2e-10 m/s²"""
    
    hp = PathSpectrumHyperparams(
        L_0=1.82, 
        beta_bulge=1.09, 
        alpha_shear=0.056, 
        gamma_bar=1.06,
        A_0=A_0
    )
    kernel = PathSpectrumKernel(hp, use_cupy=False)
    
    # Collect all RAR points
    g_bar_all = []
    g_model_all = []
    
    for idx, galaxy in df.iterrows():
        # Inclination filter
        inclination = galaxy.get('Inc', galaxy.get('inclination', 45.0))
        if inclination < 30.0 or inclination > 70.0:
            continue
        
        r_all = galaxy['r_all']
        v_all = galaxy['v_all']
        
        if len(r_all) < 3:
            continue
        
        # Compute g_bar
        v_disk = galaxy.get('v_disk_all', np.zeros_like(v_all))
        v_bulge = galaxy.get('v_bulge_all', np.zeros_like(v_all))
        v_gas = galaxy.get('v_gas_all', np.zeros_like(v_all))
        
        if v_disk is None:
            v_disk = np.zeros_like(v_all)
        if v_bulge is None:
            v_bulge = np.zeros_like(v_all)
        if v_gas is None:
            v_gas = np.zeros_like(v_all)
        
        v_baryonic_km_s = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
        v_baryonic_m_s = v_baryonic_km_s * KM_TO_M
        r_m = r_all * KPC_TO_M
        g_bar = v_baryonic_m_s**2 / r_m
        
        # Compute many-path boost
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
    
    # Compute RAR prediction with FIXED g†
    def rar_function(g_bar, g_dagger=G_DAGGER_LIT):
        return g_bar / (1.0 - np.exp(-np.sqrt(g_bar / g_dagger)))
    
    # Predict g_model using RAR with fixed g†
    g_rar_pred = rar_function(g_bar_arr)
    
    # Compute scatter in log-space (dex)
    log_residuals = np.log10(g_model_arr) - np.log10(g_rar_pred)
    scatter = np.std(log_residuals)
    mean_residual = np.mean(log_residuals)
    
    return scatter, mean_residual, len(g_bar_arr)

def main():
    print("="*80)
    print("A_0 CALIBRATION WITH FIXED g† = 1.2e-10 m/s²")
    print("="*80)
    print("\nProper approach: Use literature g† and minimize scatter\n")
    
    # Load SPARC data
    output_dir = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results")
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    print(f"\nLoaded {len(df)} SPARC galaxies\n")
    
    # Test A_0 values (finer grid)
    A_0_values = np.linspace(0.2, 3.0, 15)
    results = []
    
    print("Testing A_0 values with FIXED g† = 1.2e-10:")
    print("-" * 90)
    print(f"{'A_0':<8} {'Scatter (dex)':<18} {'Mean bias (dex)':<18} {'N points':<12} {'Status'}")
    print("-" * 90)
    
    for A_0 in A_0_values:
        scatter, mean_bias, n_points = compute_rar_scatter_fixed_gdagger(df, A_0)
        results.append({
            'A_0': A_0,
            'scatter': scatter,
            'mean_bias': mean_bias,
            'n_points': n_points
        })
        
        # Status indicator
        status = ""
        if scatter < 0.15:
            status = "✅ EXCELLENT"
        elif scatter < 0.20:
            status = "✓  GOOD"
        elif scatter < 0.25:
            status = "○  FAIR"
        else:
            status = "✗  POOR"
        
        print(f"{A_0:<8.2f} {scatter:<18.3f} {mean_bias:< 18.3f} {n_points:<12d} {status}")
    
    print("-" * 90)
    
    # Find optimal A_0
    results_df = pd.DataFrame(results)
    best_idx = np.argmin(results_df['scatter'])
    best_result = results_df.iloc[best_idx]
    
    print(f"\n{'='*80}")
    print(f"✅ OPTIMAL A_0 = {best_result['A_0']:.3f}")
    print(f"   RAR scatter: {best_result['scatter']:.3f} dex")
    print(f"   Mean bias: {best_result['mean_bias']:.3f} dex")
    print(f"   Target scatter: < 0.15 dex (literature)")
    print(f"   Status: {'PASS ✅' if best_result['scatter'] < 0.15 else 'NEEDS IMPROVEMENT ⚠️'}")
    print(f"{'='*80}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Scatter vs A_0
    ax1.plot(results_df['A_0'], results_df['scatter'], 'o-', linewidth=2, markersize=8, color='blue')
    ax1.axhline(0.15, color='red', linestyle='--', linewidth=2, label='Target (0.15 dex)')
    ax1.axhline(0.13, color='green', linestyle='--', linewidth=2, alpha=0.5, label='MOND literature (0.13 dex)')
    ax1.axvline(best_result['A_0'], color='orange', linestyle=':', linewidth=2, label=f'Optimal A_0 = {best_result["A_0"]:.2f}')
    ax1.set_xlabel('A_0 (Amplitude Parameter)', fontsize=14)
    ax1.set_ylabel('RAR Scatter (dex)', fontsize=14)
    ax1.set_title('RAR Scatter vs A_0 (Fixed g† = 1.2e-10)', fontsize=16)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, max(results_df['scatter']) * 1.1])
    
    # Plot 2: Mean bias vs A_0
    ax2.plot(results_df['A_0'], results_df['mean_bias'], 'o-', linewidth=2, markersize=8, color='orange')
    ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero bias')
    ax2.axvline(best_result['A_0'], color='orange', linestyle=':', linewidth=2, label=f'Optimal A_0 = {best_result["A_0"]:.2f}')
    ax2.set_xlabel('A_0 (Amplitude Parameter)', fontsize=14)
    ax2.set_ylabel('Mean RAR Bias (dex)', fontsize=14)
    ax2.set_title('Mean log(g_model/g_RAR) vs A_0', fontsize=16)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results/A0_calibration_fixed_gdagger.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved calibration plot to {output_path}")
    
    # Save optimal hyperparameters
    optimal_hp = PathSpectrumHyperparams(
        L_0=1.82,
        beta_bulge=1.09,
        alpha_shear=0.056,
        gamma_bar=1.06,
        A_0=best_result['A_0']
    )
    
    print(f"\n{'='*80}")
    print("OPTIMAL HYPERPARAMETERS:")
    print(f"{'='*80}")
    for key, value in optimal_hp.to_dict().items():
        print(f"  {key:<15} = {value:.4f}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
