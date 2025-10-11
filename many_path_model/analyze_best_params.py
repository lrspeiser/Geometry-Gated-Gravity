#!/usr/bin/env python3
"""
Analyze Best Parameters from Hierarchical Search

Loads the best parameters and evaluates them in detail:
- Per-galaxy APE breakdown
- Performance by morphological type
- Comparison to BTFR predictions
- Visualization of best/worst fits

Usage:
    python analyze_best_params.py results/hierarchical/final_results.json
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from sparc_stratified_test import load_sparc_master_table, load_sparc_galaxy


def evaluate_galaxy(gal, params):
    """Evaluate one galaxy with given parameters"""
    # Compute v_bar
    v_bar_sq = gal.v_gas**2 + gal.v_disk**2 + gal.v_bulge**2
    v_bar = np.sqrt(np.maximum(v_bar_sq, 1e-10))
    
    # Extract params
    eta = params['eta']
    ring_amp = params['ring_amp']
    M_max = params['M_max']
    bulge_gate_power = params['bulge_gate_power']
    
    r = gal.r_kpc
    
    # Many-path multiplier
    R0, R1, p, q = 5.0, 70.0, 2.0, 3.5
    R_gate, p_gate = 0.5, 4.0
    lambda_ring = 42.0
    
    gate = 1.0 - np.exp(-(r / R_gate)**p_gate)
    f_d = (r / R0)**p / (1.0 + (r / R1)**q)
    
    x = (2.0 * np.pi * r) / lambda_ring
    ex = np.exp(-x)
    ring_term_base = ring_amp * (ex / np.maximum(1e-20, 1.0 - ex))
    
    bulge_gate = (1.0 - np.minimum(gal.bulge_frac, 1.0))**bulge_gate_power
    ring_term = ring_term_base * bulge_gate
    
    M = eta * gate * f_d * (1.0 + ring_term)
    M = np.minimum(M, M_max)
    
    # Predicted velocity
    v_pred = np.sqrt(v_bar**2 * (1.0 + M))
    
    # Compute metrics
    mask = gal.v_obs > 0
    ape = np.abs(v_pred - gal.v_obs) / gal.v_obs * 100.0
    ape_masked = ape[mask]
    
    return {
        'ape_mean': float(np.mean(ape_masked)),
        'ape_median': float(np.median(ape_masked)),
        'ape_max': float(np.max(ape_masked)),
        'n_points': int(len(ape_masked))
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze best parameters")
    parser.add_argument('results_file', type=str, 
                       help='Path to results JSON (e.g., final_results.json)')
    parser.add_argument('--sparc_dir', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/external_data/Rotmod_LTG')
    parser.add_argument('--master_file', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/data/SPARC_Lelli2016c.mrt')
    parser.add_argument('--output_dir', type=str, default='results/analysis')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    params = results['best_params']
    
    print("="*80)
    print("BEST PARAMETERS ANALYSIS")
    print("="*80)
    print(f"Overall median APE: {results['best_score']:.2f}%")
    print(f"Total evaluations: {results['total_evaluations']:,}")
    print(f"\nParameters:")
    for name, val in params.items():
        print(f"  {name:20s} = {val:.6f}")
    
    # Load SPARC galaxies
    print(f"\nLoading SPARC galaxies...")
    master_info = load_sparc_master_table(Path(args.master_file))
    sparc_dir = Path(args.sparc_dir)
    galaxy_files = list(sparc_dir.glob('*_rotmod.dat'))
    
    galaxies = []
    for gfile in galaxy_files:
        try:
            gal = load_sparc_galaxy(gfile, master_info)
            galaxies.append(gal)
        except Exception:
            continue
    
    print(f"✓ Loaded {len(galaxies)} galaxies")
    
    # Evaluate all galaxies
    print(f"\nEvaluating all galaxies...")
    galaxy_results = []
    for gal in galaxies:
        result = evaluate_galaxy(gal, params)
        result['name'] = gal.name
        result['type'] = gal.type_group
        galaxy_results.append(result)
    
    # Sort by APE
    galaxy_results.sort(key=lambda x: x['ape_median'])
    
    # Statistics by type
    print("\n" + "="*80)
    print("PERFORMANCE BY MORPHOLOGICAL TYPE")
    print("="*80)
    
    type_stats = {}
    for res in galaxy_results:
        t = res['type']
        if t not in type_stats:
            type_stats[t] = []
        type_stats[t].append(res['ape_median'])
    
    print(f"{'Type':10s} {'Count':>6s} {'Median APE':>12s} {'Mean APE':>12s} {'Std APE':>12s}")
    print("-"*80)
    for t in sorted(type_stats.keys()):
        apes = type_stats[t]
        print(f"{t:10s} {len(apes):6d} {np.median(apes):12.2f}% {np.mean(apes):12.2f}% {np.std(apes):12.2f}%")
    
    # Best and worst fits
    print("\n" + "="*80)
    print("BEST FITS (Top 10)")
    print("="*80)
    for i, res in enumerate(galaxy_results[:10]):
        print(f"{i+1:2d}. {res['name']:15s} (Type {res['type']:3s}): {res['ape_median']:6.2f}% APE")
    
    print("\n" + "="*80)
    print("WORST FITS (Bottom 10)")
    print("="*80)
    for i, res in enumerate(galaxy_results[-10:]):
        idx = len(galaxy_results) - 10 + i
        print(f"{idx+1:2d}. {res['name']:15s} (Type {res['type']:3s}): {res['ape_median']:6.2f}% APE")
    
    # Save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detailed_file = output_dir / 'detailed_galaxy_results.json'
    with open(detailed_file, 'w') as f:
        json.dump(galaxy_results, f, indent=2)
    print(f"\n✓ Detailed results saved to: {detailed_file}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        
        # APE distribution histogram
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        apes = [res['ape_median'] for res in galaxy_results]
        axes[0].hist(apes, bins=30, alpha=0.7, edgecolor='black')
        axes[0].axvline(np.median(apes), color='red', linestyle='--', 
                       label=f'Median: {np.median(apes):.2f}%')
        axes[0].set_xlabel('Median APE (%)')
        axes[0].set_ylabel('Number of Galaxies')
        axes[0].set_title('Distribution of Galaxy Fit Quality')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # APE by type boxplot
        type_labels = sorted(type_stats.keys())
        type_data = [type_stats[t] for t in type_labels]
        axes[1].boxplot(type_data, labels=type_labels)
        axes[1].set_xlabel('Morphological Type')
        axes[1].set_ylabel('Median APE (%)')
        axes[1].set_title('Performance by Galaxy Type')
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = output_dir / 'performance_summary.png'
        plt.savefig(plot_file, dpi=150)
        print(f"✓ Plot saved to: {plot_file}")


if __name__ == '__main__':
    main()
