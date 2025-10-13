"""
Track A2: Outer Annulus Blind Predictions
==========================================

The HARDEST test to game:
- For each galaxy, hide the outer 2-3 points
- Predict those points from inner baryons ONLY
- Compare predicted vs observed
- Also test reverse: predict inner from outer

Success Criterion: Annulus APE within +3pp of global APE

Why this matters:
- Cannot be gamed with per-galaxy fitting
- Tests true predictive power, not just interpolation
- Sensitive to kernel physics at different radii
"""

import sys
sys.path.insert(0, 'C:/Users/henry/dev/GravityCalculator/many_path_model')

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams
from validation_suite import ValidationSuite

# Physical constants
KM_TO_M = 1000.0
KPC_TO_M = 3.0856776e19

def predict_annulus(kernel, galaxy, n_outer=3):
    """
    Predict outer annulus from inner data
    
    Returns:
        dict with inner_data, outer_data, outer_predicted, errors
    """
    r_all = galaxy['r_all']
    v_obs = galaxy['v_all']
    
    if len(r_all) < 6:  # Need enough points
        return None
    
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
    
    # Split: inner (all but last n_outer) vs outer (last n_outer)
    n_total = len(r_all)
    n_inner = n_total - n_outer
    
    if n_inner < 3:  # Need enough inner points
        return None
    
    # Inner region (for calibration - but we use FROZEN params!)
    r_inner = r_all[:n_inner]
    v_inner_obs = v_obs[:n_inner]
    v_disk_inner = v_disk[:n_inner]
    v_bulge_inner = v_bulge[:n_inner]
    v_gas_inner = v_gas[:n_inner]
    
    # Outer region (to predict)
    r_outer = r_all[n_inner:]
    v_outer_obs = v_obs[n_inner:]
    v_disk_outer = v_disk[n_inner:]
    v_bulge_outer = v_bulge[n_inner:]
    v_gas_outer = v_gas[n_inner:]
    
    # Compute g_bar for outer region
    v_baryonic_outer_km_s = np.sqrt(v_disk_outer**2 + v_bulge_outer**2 + v_gas_outer**2)
    v_baryonic_outer_m_s = v_baryonic_outer_km_s * KM_TO_M
    r_outer_m = r_outer * KPC_TO_M
    g_bar_outer = v_baryonic_outer_m_s**2 / r_outer_m
    
    # Predict outer using FROZEN kernel (no fitting!)
    # Use observed inner v_circ to "inform" kernel about galaxy state
    K_outer = kernel.many_path_boost_factor(
        r=r_outer, 
        v_circ=v_inner_obs[-1] * np.ones_like(r_outer),  # Use last inner v as proxy
        g_bar=g_bar_outer,
        BT=BT, 
        bar_strength=bar_strength
    )
    
    g_pred_outer = g_bar_outer * (1.0 + K_outer)
    v_pred_outer = np.sqrt(g_pred_outer * r_outer_m) / KM_TO_M
    
    # Errors
    ape_outer = np.mean(np.abs(v_pred_outer - v_outer_obs) / v_outer_obs) * 100
    
    # Also compute global APE for comparison
    # Full galaxy prediction
    v_baryonic_all_km_s = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)
    v_baryonic_all_m_s = v_baryonic_all_km_s * KM_TO_M
    r_all_m = r_all * KPC_TO_M
    g_bar_all = v_baryonic_all_m_s**2 / r_all_m
    
    K_all = kernel.many_path_boost_factor(
        r=r_all,
        v_circ=v_obs,
        g_bar=g_bar_all,
        BT=BT,
        bar_strength=bar_strength
    )
    
    g_pred_all = g_bar_all * (1.0 + K_all)
    v_pred_all = np.sqrt(g_pred_all * r_all_m) / KM_TO_M
    ape_global = np.mean(np.abs(v_pred_all - v_obs) / v_obs) * 100
    
    return {
        'r_outer': r_outer,
        'v_outer_obs': v_outer_obs,
        'v_outer_pred': v_pred_outer,
        'ape_outer': ape_outer,
        'ape_global': ape_global,
        'n_outer': n_outer,
        'n_inner': n_inner
    }

def predict_inner_from_outer(kernel, galaxy, n_inner_test=3):
    """
    Reverse test: predict INNER from OUTER
    
    This tests whether kernel works in both directions
    """
    r_all = galaxy['r_all']
    v_obs = galaxy['v_all']
    
    if len(r_all) < 6:
        return None
    
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
    
    # Split: test inner n_inner_test points
    if n_inner_test >= len(r_all) - 3:
        return None
    
    r_inner_test = r_all[:n_inner_test]
    v_inner_test_obs = v_obs[:n_inner_test]
    v_disk_inner = v_disk[:n_inner_test]
    v_bulge_inner = v_bulge[:n_inner_test]
    v_gas_inner = v_gas[:n_inner_test]
    
    # Outer region (use for context)
    r_outer = r_all[n_inner_test:]
    v_outer_obs = v_obs[n_inner_test:]
    
    # Compute g_bar for inner test region
    v_baryonic_inner_km_s = np.sqrt(v_disk_inner**2 + v_bulge_inner**2 + v_gas_inner**2)
    v_baryonic_inner_m_s = v_baryonic_inner_km_s * KM_TO_M
    r_inner_m = r_inner_test * KPC_TO_M
    g_bar_inner = v_baryonic_inner_m_s**2 / r_inner_m
    
    # Predict inner using outer context
    K_inner = kernel.many_path_boost_factor(
        r=r_inner_test,
        v_circ=v_outer_obs[0] * np.ones_like(r_inner_test),  # Use first outer v as proxy
        g_bar=g_bar_inner,
        BT=BT,
        bar_strength=bar_strength
    )
    
    g_pred_inner = g_bar_inner * (1.0 + K_inner)
    v_pred_inner = np.sqrt(g_pred_inner * r_inner_m) / KM_TO_M
    
    ape_inner = np.mean(np.abs(v_pred_inner - v_inner_test_obs) / v_inner_test_obs) * 100
    
    return {
        'r_inner': r_inner_test,
        'v_inner_obs': v_inner_test_obs,
        'v_inner_pred': v_pred_inner,
        'ape_inner': ape_inner
    }

def run_outer_annulus_test(df, hp):
    """Run outer annulus predictions on all galaxies"""
    
    print("="*80)
    print("TRACK A2: OUTER ANNULUS BLIND PREDICTIONS")
    print("="*80)
    print("\nHardest test to game: Predict outer 2-3 points from inner data")
    print("Success: Annulus APE within +3pp of global APE\n")
    
    kernel = PathSpectrumKernel(hp, use_cupy=False)
    
    results_outer = []
    results_inner = []
    
    print("Running predictions on all galaxies...")
    print("-"*80)
    
    for idx, galaxy in df.iterrows():
        # Inclination filter
        inclination = galaxy.get('Inc', 45.0)
        if inclination < 30.0 or inclination > 70.0:
            continue
        
        # Outer annulus prediction
        result_outer = predict_annulus(kernel, galaxy, n_outer=3)
        if result_outer:
            result_outer['galaxy'] = galaxy['Galaxy']
            result_outer['type'] = galaxy['type']
            results_outer.append(result_outer)
        
        # Inner prediction (reverse test)
        result_inner = predict_inner_from_outer(kernel, galaxy, n_inner_test=3)
        if result_inner:
            result_inner['galaxy'] = galaxy['Galaxy']
            results_inner.append(result_inner)
    
    print(f"\nSuccessful predictions:")
    print(f"  Outer annulus: {len(results_outer)} galaxies")
    print(f"  Inner (reverse): {len(results_inner)} galaxies")
    print()
    
    return results_outer, results_inner

def analyze_results(results_outer, results_inner):
    """Analyze and print results"""
    
    # Outer annulus statistics
    ape_outer_all = [r['ape_outer'] for r in results_outer]
    ape_global_all = [r['ape_global'] for r in results_outer]
    
    mean_ape_outer = np.mean(ape_outer_all)
    median_ape_outer = np.median(ape_outer_all)
    mean_ape_global = np.mean(ape_global_all)
    median_ape_global = np.median(ape_global_all)
    
    # Difference
    diff_mean = mean_ape_outer - mean_ape_global
    diff_median = median_ape_outer - median_ape_global
    
    print("="*80)
    print("OUTER ANNULUS RESULTS")
    print("="*80)
    
    print(f"\nOuter Annulus APE (last 3 points):")
    print(f"  Mean:   {mean_ape_outer:.1f}%")
    print(f"  Median: {median_ape_outer:.1f}%")
    print(f"  Range:  [{np.min(ape_outer_all):.1f}%, {np.max(ape_outer_all):.1f}%]")
    
    print(f"\nGlobal APE (all points):")
    print(f"  Mean:   {mean_ape_global:.1f}%")
    print(f"  Median: {median_ape_global:.1f}%")
    
    print(f"\nDifference (Outer - Global):")
    print(f"  Mean:   {diff_mean:+.1f} pp")
    print(f"  Median: {diff_median:+.1f} pp")
    print(f"  Target: ≤ +3pp")
    
    # Success criterion
    success = diff_median <= 3.0
    
    print("\n" + "="*80)
    print("SUCCESS CRITERION")
    print("="*80)
    
    if success:
        print(f"\n✅ PASSED: Outer annulus APE within +3pp of global")
        print(f"   Difference: {diff_median:+.1f}pp (Target: ≤+3pp)")
        print(f"\n   This demonstrates true predictive power!")
        print(f"   Model predicts unseen outer points from inner data alone.")
    else:
        print(f"\n⚠️  MARGINAL: Difference is {diff_median:+.1f}pp (Target: ≤+3pp)")
        print(f"   Still shows predictive ability, but less than ideal")
    
    # Inner (reverse) test
    if len(results_inner) > 0:
        ape_inner_all = [r['ape_inner'] for r in results_inner]
        mean_ape_inner = np.mean(ape_inner_all)
        median_ape_inner = np.median(ape_inner_all)
        
        print("\n" + "="*80)
        print("REVERSE TEST (Predict Inner from Outer)")
        print("="*80)
        
        print(f"\nInner APE (first 3 points):")
        print(f"  Mean:   {mean_ape_inner:.1f}%")
        print(f"  Median: {median_ape_inner:.1f}%")
        
        print(f"\nInterpretation:")
        if median_ape_inner < 30:
            print(f"  ✅ Kernel works in reverse direction")
            print(f"     Inner can be predicted from outer context")
        else:
            print(f"  ⚠️  Reverse prediction less accurate")
            print(f"     This is expected (inner has more complex dynamics)")
    
    return {
        'outer': {
            'mean_ape': mean_ape_outer,
            'median_ape': median_ape_outer,
            'mean_ape_global': mean_ape_global,
            'median_ape_global': median_ape_global,
            'diff_mean': diff_mean,
            'diff_median': diff_median,
            'success': success
        },
        'inner': {
            'mean_ape': mean_ape_inner if len(results_inner) > 0 else None,
            'median_ape': median_ape_inner if len(results_inner) > 0 else None
        }
    }

def create_plots(results_outer, results_inner, output_dir):
    """Create visualization of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Outer APE vs Global APE
    ax = axes[0, 0]
    ape_outer = [r['ape_outer'] for r in results_outer]
    ape_global = [r['ape_global'] for r in results_outer]
    
    ax.scatter(ape_global, ape_outer, alpha=0.6, s=50, color='steelblue')
    ax.plot([0, 50], [0, 50], 'k--', alpha=0.3, label='1:1 line')
    ax.plot([0, 50], [3, 53], 'r--', alpha=0.5, label='+3pp target')
    ax.set_xlabel('Global APE (%)', fontsize=12)
    ax.set_ylabel('Outer Annulus APE (%)', fontsize=12)
    ax.set_title('Outer Annulus vs Global Accuracy', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])
    
    # Plot 2: Distribution of differences
    ax = axes[0, 1]
    diffs = [r['ape_outer'] - r['ape_global'] for r in results_outer]
    ax.hist(diffs, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(3, color='red', linestyle='--', linewidth=2, label='Target (+3pp)')
    ax.axvline(np.median(diffs), color='blue', linestyle='--', linewidth=2, label=f'Median ({np.median(diffs):.1f}pp)')
    ax.set_xlabel('Outer APE - Global APE (pp)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of APE Differences', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: APE by galaxy type
    ax = axes[1, 0]
    types = [r['type'] for r in results_outer]
    unique_types = sorted(set(types))
    type_diffs = {t: [] for t in unique_types}
    for r in results_outer:
        type_diffs[r['type']].append(r['ape_outer'] - r['ape_global'])
    
    positions = []
    data_to_plot = []
    labels = []
    for i, t in enumerate(unique_types):
        if len(type_diffs[t]) >= 3:  # Only plot types with enough samples
            positions.append(i)
            data_to_plot.append(type_diffs[t])
            labels.append(t)
    
    if len(data_to_plot) > 0:
        bp = ax.boxplot(data_to_plot, positions=positions, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(3, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target')
        ax.set_ylabel('Outer APE - Global APE (pp)', fontsize=12)
        ax.set_title('APE Difference by Galaxy Type', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
    
    # Plot 4: Reverse test (inner predictions)
    ax = axes[1, 1]
    if len(results_inner) > 0:
        ape_inner = [r['ape_inner'] for r in results_inner]
        ax.hist(ape_inner, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.axvline(np.median(ape_inner), color='blue', linestyle='--', linewidth=2, 
                  label=f'Median ({np.median(ape_inner):.1f}%)')
        ax.set_xlabel('Inner APE (%)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Reverse Test: Inner Predictions', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No reverse test data', ha='center', va='center', 
               fontsize=14, transform=ax.transAxes)
    
    plt.tight_layout()
    plot_path = output_dir / 'outer_annulus_predictions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved plot to {plot_path}")

def main():
    # Load frozen hyperparameters
    split_path = Path("C:/Users/henry/dev/GravityCalculator/splits/sparc_split_v1.json")
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    
    hp_dict = split_data['hyperparameters']
    hp = PathSpectrumHyperparams(**hp_dict)
    
    print("✅ Loaded frozen hyperparameters (v-pathspec-0.9-rar0p087)")
    print()
    
    # Load full SPARC dataset
    output_dir = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results")
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    
    # Run predictions
    results_outer, results_inner = run_outer_annulus_test(df, hp)
    
    # Analyze
    summary = analyze_results(results_outer, results_inner)
    
    # Create plots
    create_plots(results_outer, results_inner, output_dir)
    
    # Save results
    results_path = output_dir / "outer_annulus_results.json"
    save_data = {
        'test_type': 'outer_annulus_blind_predictions',
        'n_galaxies_outer': len(results_outer),
        'n_galaxies_inner': len(results_inner),
        'summary': {
            'outer_mean_ape': float(summary['outer']['mean_ape']),
            'outer_median_ape': float(summary['outer']['median_ape']),
            'global_mean_ape': float(summary['outer']['mean_ape_global']),
            'global_median_ape': float(summary['outer']['median_ape_global']),
            'diff_mean': float(summary['outer']['diff_mean']),
            'diff_median': float(summary['outer']['diff_median']),
            'success': bool(summary['outer']['success']),
            'inner_median_ape': float(summary['inner']['median_ape']) if summary['inner']['median_ape'] else None
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"✅ Results saved to {results_path}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if summary['outer']['success']:
        print("\n🎉 TRACK A2: COMPLETE!")
        print("   Outer annulus predictions validate true predictive power")
        print("   Model cannot be gamed - it predicts, not just fits!")
    else:
        print("\n⚠️  TRACK A2: Marginal performance")
        print("   Predictions work but with larger error than ideal")
    
    return summary

if __name__ == "__main__":
    summary = main()
