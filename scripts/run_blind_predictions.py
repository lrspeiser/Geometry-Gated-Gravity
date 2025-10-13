"""
Blind Rotation Curve Predictions (Track A.1)

Load frozen split and hyperparameters from v-pathspec-0.9-rar0p087.
Compute predictions on TEST SET ONLY with ZERO retraining.

Target: Median APE ≤ 16% on holdout (currently 19.1%)

This is a decisive test: if we can predict unseen rotation curves
with frozen parameters, it proves the model generalizes.
"""

import sys
sys.path.insert(0, 'C:/Users/henry/dev/GravityCalculator/many_path_model')

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from validation_suite import ValidationSuite
from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams

# Physical constants
KM_TO_M = 1000.0
KPC_TO_M = 3.0856776e19

def load_frozen_split(split_path):
    """Load frozen split and hyperparameters"""
    with open(split_path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct hyperparameters
    hp_dict = data['hyperparameters']
    hp = PathSpectrumHyperparams(**hp_dict)
    
    return data, hp

def compute_blind_predictions(df, split_data, hp, test_only=True):
    """Compute predictions on test set with frozen hyperparameters"""
    
    kernel = PathSpectrumKernel(hp, use_cupy=False)
    
    # Extract test set indices
    test_indices = [g['index'] for g in split_data['test_set']]
    test_df = df.loc[test_indices].copy()
    
    print(f"\nComputing blind predictions for {len(test_df)} test galaxies...")
    print("(NO retraining - frozen hyperparameters only)\n")
    
    results = []
    
    for idx, galaxy in test_df.iterrows():
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
        
        # Quadrature method
        v_baryonic_km_s = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
        v_baryonic_m_s = v_baryonic_km_s * KM_TO_M
        r_m = r_all * KPC_TO_M
        g_bar = v_baryonic_m_s**2 / r_m
        
        # Compute many-path boost
        BT = galaxy.get('BT', 0.0)
        bar_strength = galaxy.get('bar_strength', 0.0)
        K = kernel.many_path_boost_factor(r=r_all, v_circ=v_obs, g_bar=g_bar,
                                          BT=BT, bar_strength=bar_strength)
        
        # Model prediction: g_model = g_bar * (1 + K)
        g_model = g_bar * (1.0 + K)
        
        # Convert back to velocity
        v_model = np.sqrt(g_model * r_m) / KM_TO_M
        
        # Compute APE
        ape = np.mean(np.abs(v_model - v_obs) / v_obs) * 100
        
        results.append({
            'galaxy': galaxy['Galaxy'],
            'type': galaxy['type'],
            'ape': ape,
            'r_all': r_all,
            'v_obs': v_obs,
            'v_model': v_model,
            'K': K
        })
    
    return pd.DataFrame(results)

def plot_blind_predictions(results_df, output_dir):
    """Create publication-quality blind prediction plots"""
    
    # Figure 1: Parity plot (v_model vs v_obs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect all velocity pairs
    v_obs_all = []
    v_model_all = []
    for idx, row in results_df.iterrows():
        v_obs_all.extend(row['v_obs'])
        v_model_all.extend(row['v_model'])
    
    v_obs_all = np.array(v_obs_all)
    v_model_all = np.array(v_model_all)
    
    # Parity plot
    ax = axes[0]
    ax.hexbin(v_obs_all, v_model_all, gridsize=30, cmap='Blues', mincnt=1)
    ax.plot([0, 300], [0, 300], 'r--', linewidth=2, label='1:1')
    ax.plot([0, 300], [0, 330], 'r:', linewidth=1, alpha=0.5, label='±10%')
    ax.plot([0, 300], [0, 270], 'r:', linewidth=1, alpha=0.5)
    ax.set_xlabel('V_obs (km/s)', fontsize=14)
    ax.set_ylabel('V_model (km/s)', fontsize=14)
    ax.set_title('Blind Prediction Parity Plot', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 300])
    ax.set_ylim([0, 300])
    ax.set_aspect('equal')
    
    # APE distribution
    ax = axes[1]
    ax.hist(results_df['ape'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(results_df['ape'].median(), color='red', linestyle='--', linewidth=2, 
               label=f'Median APE = {results_df["ape"].median():.1f}%')
    ax.axvline(16, color='green', linestyle=':', linewidth=2, label='Target (16%)')
    ax.set_xlabel('APE (%)', fontsize=14)
    ax.set_ylabel('Number of Galaxies', fontsize=14)
    ax.set_title('Test Set APE Distribution', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    parity_path = output_dir / 'blind_prediction_parity.png'
    plt.savefig(parity_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved parity plot to {parity_path}")
    
    # Figure 2: Per-galaxy rotation curves (top 6 best, top 6 worst)
    results_sorted = results_df.sort_values('ape')
    best_6 = results_sorted.head(6)
    worst_6 = results_sorted.tail(6)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (idx, row) in enumerate(best_6.iterrows()):
        ax = axes[i]
        ax.plot(row['r_all'], row['v_obs'], 'ko', markersize=5, label='Observed')
        ax.plot(row['r_all'], row['v_model'], 'r-', linewidth=2, label='Model')
        ax.set_title(f"{row['galaxy']} (APE={row['ape']:.1f}%)", fontsize=11)
        ax.set_xlabel('R (kpc)', fontsize=10)
        ax.set_ylabel('V (km/s)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    for i, (idx, row) in enumerate(worst_6.iterrows()):
        ax = axes[i+6]
        ax.plot(row['r_all'], row['v_obs'], 'ko', markersize=5, label='Observed')
        ax.plot(row['r_all'], row['v_model'], 'r-', linewidth=2, label='Model')
        ax.set_title(f"{row['galaxy']} (APE={row['ape']:.1f}%)", fontsize=11)
        ax.set_xlabel('R (kpc)', fontsize=10)
        ax.set_ylabel('V (km/s)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    curves_path = output_dir / 'blind_prediction_curves.png'
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved rotation curves to {curves_path}")

def main():
    print("="*80)
    print("BLIND ROTATION CURVE PREDICTIONS (Track A.1)")
    print("="*80)
    print("\nFrozen kernel: v-pathspec-0.9-rar0p087")
    print("Target: Median APE ≤ 16% on test set")
    print("Current baseline: 19.1% (from optimization)\n")
    
    # Load frozen split
    split_path = Path("C:/Users/henry/dev/GravityCalculator/splits/sparc_split_v1.json")
    split_data, hp = load_frozen_split(split_path)
    
    print("✅ Loaded frozen split and hyperparameters")
    print(f"   Test set: {split_data['metadata']['test_galaxies']} galaxies")
    print(f"   Tag: {split_data['metadata']['tag']}\n")
    
    # Load SPARC data
    output_dir = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results")
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    
    # Compute blind predictions
    results_df = compute_blind_predictions(df, split_data, hp)
    
    # Statistics
    print("="*80)
    print("BLIND PREDICTION RESULTS")
    print("="*80)
    
    median_ape = results_df['ape'].median()
    mean_ape = results_df['ape'].mean()
    std_ape = results_df['ape'].std()
    
    print(f"\nTest Set Performance (BLIND, frozen hyperparameters):")
    print(f"  Median APE: {median_ape:.1f}%")
    print(f"  Mean APE:   {mean_ape:.1f}%")
    print(f"  Std APE:    {std_ape:.1f}%")
    print(f"  Min APE:    {results_df['ape'].min():.1f}%")
    print(f"  Max APE:    {results_df['ape'].max():.1f}%")
    
    # Target assessment
    print(f"\n" + "-"*80)
    if median_ape <= 16:
        print(f"✅ TARGET MET! Median APE {median_ape:.1f}% ≤ 16%")
        print(f"   This demonstrates excellent generalization!")
    elif median_ape <= 19:
        print(f"⚠️  CLOSE: Median APE {median_ape:.1f}% (target 16%)")
        print(f"   Within 3% of target - minor improvements possible")
    else:
        print(f"⚠️  Median APE {median_ape:.1f}% > 19%")
        print(f"   Consider galaxy-specific corrections or selective gates")
    
    # Stratification by morphology
    print(f"\n" + "-"*80)
    print("Performance by Morphology Type:")
    for gtype in sorted(results_df['type'].unique()):
        subset = results_df[results_df['type'] == gtype]
        print(f"  {gtype:<5}: median={subset['ape'].median():.1f}%, n={len(subset)}")
    
    # Generate plots
    print(f"\n" + "-"*80)
    plot_blind_predictions(results_df, output_dir)
    
    # Save results
    results_path = output_dir / "blind_predictions_results.json"
    results_export = {
        'test_set_performance': {
            'median_ape': float(median_ape),
            'mean_ape': float(mean_ape),
            'std_ape': float(std_ape),
            'n_galaxies': len(results_df),
            'target_met': bool(median_ape <= 16)
        },
        'per_galaxy_results': results_df[['galaxy', 'type', 'ape']].to_dict('records')
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_export, f, indent=2)
    
    print(f"✅ Results saved to {results_path}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if median_ape <= 16:
        print("\n🎉 Blind prediction test PASSED!")
        print("   Ready for Track A.2 (outer annulus) and Track A.3 (bar stratification)")
    else:
        print("\n📊 Blind prediction test shows room for improvement")
        print("   Consider Track C engineering before moving to lensing")

if __name__ == "__main__":
    main()
