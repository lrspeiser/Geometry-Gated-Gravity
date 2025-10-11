#!/usr/bin/env python3
"""
Ablation Studies - Which Ingredients Are Essential?

Test impact of removing each model component to determine:
1. Which terms are needed for flat rotation curve?
2. Which terms are needed for correct vertical lag?
3. Which terms prevent outer overshoot?

This guards against "too many parameters" critique by demonstrating
that each ingredient serves a specific, measurable purpose.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from gaia_comparison import load_gaia_data, compute_observed_rotation_curve
from toy_many_path_gravity import (
    sample_exponential_disk, sample_hernquist_bulge,
    rotation_curve, to_cpu, xp_array, xp_zeros
)
from parameter_optimizer import loss_manypath

try:
    import cupy as cp
    _USING_CUPY = True
except Exception:
    import numpy as cp
    _USING_CUPY = False


def baseline_params():
    """Final optimized parameters (baseline)."""
    return {
        'eta': 0.39,
        'R_gate': 0.5,
        'p_gate': 4.0,
        'R0': 5.0,
        'p': 2.0,
        'R1': 70.0,
        'q': 3.5,
        'k_an': 1.4,
        'ring_amp': 0.07,
        'lambda_ring': 42.0,
        'M_max': 3.3,
        'Z0_in': 1.02,
        'Z0_out': 1.72,
        'R_lag': 8.0,
        'w_lag': 1.9,
        'k_boost': 0.75,
    }


def ablation_configs():
    """
    Define ablation configurations.
    
    Each removes or modifies one key ingredient to test its impact.
    """
    configs = {}
    
    # Baseline (no ablation)
    configs['baseline'] = {
        'name': 'Baseline (Full Model)',
        'params': baseline_params(),
        'description': 'Final optimized model with all ingredients'
    }
    
    # Ablation 1: No radial modulation
    params_no_mod = baseline_params()
    params_no_mod['Z0_in'] = 1.5  # Use average
    params_no_mod['Z0_out'] = 1.5
    params_no_mod['k_boost'] = 0.0  # No bump
    configs['no_modulation'] = {
        'name': 'No Radial Modulation',
        'params': params_no_mod,
        'description': 'Uniform anisotropy, no R-dependent enhancement'
    }
    
    # Ablation 2: No ring winding
    params_no_ring = baseline_params()
    params_no_ring['ring_amp'] = 0.0
    configs['no_ring'] = {
        'name': 'No Ring Winding',
        'params': params_no_ring,
        'description': 'No azimuthal wraparound term'
    }
    
    # Ablation 3: Looser saturation
    params_loose_sat = baseline_params()
    params_loose_sat['q'] = 2.0  # Was 3.5 (harder)
    configs['loose_saturation'] = {
        'name': 'Looser Saturation',
        'params': params_loose_sat,
        'description': 'Softer distance roll-off (q=2.0 vs 3.5)'
    }
    
    # Ablation 4: No distance gating
    params_no_gate = baseline_params()
    params_no_gate['R_gate'] = 0.01  # Effectively removes gate
    configs['no_gate'] = {
        'name': 'No Distance Gate',
        'params': params_no_gate,
        'description': 'Solar system not protected'
    }
    
    # Ablation 5: Weaker anisotropy
    params_weak_anis = baseline_params()
    params_weak_anis['k_an'] = 0.7  # Was 1.4
    params_weak_anis['k_boost'] = 0.35  # Was 0.75
    configs['weak_anisotropy'] = {
        'name': 'Weaker Anisotropy',
        'params': params_weak_anis,
        'description': 'Half the planar preference strength'
    }
    
    return configs


def run_ablation_study(src_pos, src_mass, obs_curve, R_grid, configs, 
                       eps=0.05, batch_size=50000):
    """
    Run all ablation configurations and compare results.
    
    Returns:
        results: dict with all metrics for each configuration
    """
    results = {}
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY - Testing Model Components")
    print(f"{'='*70}\n")
    
    for key, config in configs.items():
        print(f"\nTesting: {config['name']}")
        print(f"  {config['description']}")
        
        params = config['params']
        
        # Compute loss
        loss, components = loss_manypath(
            params, src_pos, src_mass, obs_curve, R_grid,
            eps=eps, batch_size=batch_size, verbose=False
        )
        
        # Compute rotation curve for visualization
        vM, _ = rotation_curve(src_pos, src_mass, xp_array(R_grid), z=0.0,
                              eps=eps, params=params, use_multiplier=True,
                              batch_size=batch_size)
        vM_np = to_cpu(vM)
        
        results[key] = {
            'name': config['name'],
            'description': config['description'],
            'params': params,
            'total_loss': loss,
            'chi2_rot': components['chi2_rot'],
            'chi2_lag': components['chi2_lag'],
            'chi2_slope': components['chi2_slope'],
            'lag_mean': components['lag_mean'],
            'lag_std': components['lag_std'],
            'v_curve': vM_np,
        }
        
        print(f"  Loss: {loss:.2f} (χ²={components['chi2_rot']:.0f}, "
              f"lag={components['lag_mean']:.1f} km/s, slope={components['chi2_slope']:.0f})")
    
    return results


def analyze_ablations(results, output_dir: Path):
    """Analyze and visualize ablation study results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get baseline
    baseline = results['baseline']
    
    # Compute deltas
    print(f"\n{'='*70}")
    print("ABLATION IMPACT ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"{'Configuration':30s} {'Δχ²':>10} {'Δlag':>8} {'Δslope':>10} {'Δtotal':>10}")
    print("-" * 70)
    
    for key in ['no_modulation', 'no_ring', 'loose_saturation', 'no_gate', 'weak_anisotropy']:
        if key not in results:
            continue
        
        r = results[key]
        delta_chi2 = r['chi2_rot'] - baseline['chi2_rot']
        delta_lag = r['lag_mean'] - baseline['lag_mean']
        delta_slope = r['chi2_slope'] - baseline['chi2_slope']
        delta_total = r['total_loss'] - baseline['total_loss']
        
        print(f"{r['name']:30s} {delta_chi2:10.0f} {delta_lag:8.1f} {delta_slope:10.0f} {delta_total:10.0f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: χ² comparison
    ax = axes[0, 0]
    names = [r['name'] for r in results.values()]
    chi2s = [r['chi2_rot'] for r in results.values()]
    colors = ['green' if k == 'baseline' else 'orange' for k in results.keys()]
    
    bars = ax.barh(names, chi2s, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Rotation χ²', fontsize=12)
    ax.set_title('Impact on Rotation Curve Fit', fontsize=14, fontweight='bold')
    ax.axvline(baseline['chi2_rot'], color='green', linestyle='--', 
              alpha=0.5, linewidth=2, label='Baseline')
    ax.legend()
    
    # Plot 2: Vertical lag
    ax = axes[0, 1]
    lags = [r['lag_mean'] for r in results.values()]
    lag_stds = [r['lag_std'] for r in results.values()]
    
    ax.barh(names, lags, xerr=lag_stds, color=colors, alpha=0.7, 
           edgecolor='black', capsize=3)
    ax.axvline(15, color='green', linestyle='--', alpha=0.5, 
              linewidth=2, label='Target: 15 km/s')
    ax.axvspan(10, 20, alpha=0.1, color='green', label='Acceptable')
    ax.set_xlabel('Vertical Lag (km/s)', fontsize=12)
    ax.set_title('Impact on Vertical Structure', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Plot 3: Outer slope
    ax = axes[1, 0]
    slopes = [r['chi2_slope'] for r in results.values()]
    
    ax.barh(names, slopes, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(400, color='red', linestyle='--', alpha=0.5, 
              linewidth=2, label='Threshold: 400')
    ax.set_xlabel('Outer Slope Penalty', fontsize=12)
    ax.set_title('Impact on Outer Curve Flatness', fontsize=14, fontweight='bold')
    ax.legend()
    
    # Plot 4: Total loss
    ax = axes[1, 1]
    totals = [r['total_loss'] for r in results.values()]
    
    ax.barh(names, totals, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(baseline['total_loss'], color='green', linestyle='--', 
              alpha=0.5, linewidth=2, label='Baseline')
    ax.set_xlabel('Total Loss', fontsize=12)
    ax.set_title('Overall Multi-Objective Loss', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    
    output_file = output_dir / "ablation_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved ablation plot: {output_file}")
    plt.close()
    
    # Save summary table
    summary_file = output_dir / "ablation_summary.csv"
    df = pd.DataFrame([{
        'configuration': r['name'],
        'chi2_rotation': r['chi2_rot'],
        'vertical_lag_km_s': r['lag_mean'],
        'outer_slope': r['chi2_slope'],
        'total_loss': r['total_loss'],
        'delta_chi2': r['chi2_rot'] - baseline['chi2_rot'],
        'delta_lag': r['lag_mean'] - baseline['lag_mean'],
        'delta_total': r['total_loss'] - baseline['total_loss'],
    } for r in results.values()])
    
    df.to_csv(summary_file, index=False)
    print(f"✓ Saved summary table: {summary_file}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation studies for many-path model")
    parser.add_argument("--n_sources", type=int, default=100000)
    parser.add_argument("--n_bulge", type=int, default=20000)
    parser.add_argument("--use_bulge", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/ablations")
    args = parser.parse_args()
    
    # Load Gaia data
    df_gaia = load_gaia_data()
    obs_curve = compute_observed_rotation_curve(df_gaia)
    
    # Sample source distribution
    print(f"\nSampling {args.n_sources} disk particles...")
    disk_pos, m_disk = sample_exponential_disk(
        args.n_sources, M_disk=5e10, R_d=2.6, z_d=0.3, R_max=30.0, seed=42
    )
    
    if args.use_bulge:
        print(f"Sampling {args.n_bulge} bulge particles...")
        bulge_pos, m_bulge = sample_hernquist_bulge(
            args.n_bulge, M_bulge=1e10, a=0.7, seed=123
        )
        src_pos = cp.concatenate([disk_pos, bulge_pos], axis=0)
        src_mass = cp.concatenate([
            xp_zeros(disk_pos.shape[0]) + m_disk,
            xp_zeros(bulge_pos.shape[0]) + m_bulge
        ])
    else:
        src_pos = disk_pos
        src_mass = xp_zeros(disk_pos.shape[0]) + m_disk
    
    print(f"✓ Total source particles: {src_pos.shape[0]}\n")
    
    # Target grid
    R_grid = np.linspace(5, 15, 60)
    
    # Get ablation configurations
    configs = ablation_configs()
    
    # Run ablation study
    results = run_ablation_study(src_pos, src_mass, obs_curve, R_grid, configs,
                                 eps=0.05, batch_size=args.batch_size)
    
    # Analyze results
    output_dir = Path(args.output_dir)
    analyze_ablations(results, output_dir)
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}\n")
    
    print("Key Findings:")
    baseline = results['baseline']
    
    if 'no_modulation' in results:
        delta = results['no_modulation']['chi2_rot'] - baseline['chi2_rot']
        if delta > 100:
            print(f"  ✓ Radial modulation is ESSENTIAL (Δχ² = +{delta:.0f})")
        else:
            print(f"  ⚠ Radial modulation has modest impact (Δχ² = +{delta:.0f})")
    
    if 'loose_saturation' in results:
        delta = results['loose_saturation']['chi2_rot'] - baseline['chi2_rot']
        if delta > 100:
            print(f"  ✓ Hard saturation is ESSENTIAL (Δχ² = +{delta:.0f})")
        else:
            print(f"  ⚠ Saturation hardness has modest impact (Δχ² = +{delta:.0f})")
    
    if 'no_ring' in results:
        delta = results['no_ring']['chi2_rot'] - baseline['chi2_rot']
        if abs(delta) < 50:
            print(f"  → Ring term is optional (Δχ² = {delta:+.0f})")
        else:
            print(f"  ✓ Ring term contributes (Δχ² = {delta:+.0f})")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
