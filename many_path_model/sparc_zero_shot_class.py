#!/usr/bin/env python3
"""
SPARC Zero-Shot Class-Wise Testing
===================================

Tests universality of the model by applying class-wise median parameters
(early/intermediate/late types) to all 175 SPARC galaxies without per-galaxy optimization.

This is the KEY test for universality:
- Converts 175 tailored fits → 3 parameter sets
- If performance stays competitive, proves the model generalizes
- Dramatically reduces parameter budget (5×3 vs 5×175)

Usage:
    # Test with class-wise parameters (3 sets)
    python many_path_model/sparc_zero_shot_class.py --params results/mega_parallel/class_params_for_zero_shot.json --mode class --output_dir results/zero_shot_class

    # Test with single global parameter set (strictest test)
    python many_path_model/sparc_zero_shot_class.py --params results/mega_parallel/class_params_for_zero_shot.json --mode global --output_dir results/zero_shot_global

    # Test both modes
    python many_path_model/sparc_zero_shot_class.py --params results/mega_parallel/class_params_for_zero_shot.json --mode both --output_dir results/zero_shot
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List

# GPU check
try:
    import cupy as cp
    _USING_CUPY = True
except ImportError:
    import numpy as cp
    _USING_CUPY = False

# Import galaxy loading
sys.path.insert(0, str(Path(__file__).parent))
from sparc_stratified_test import load_sparc_galaxy, load_sparc_master_table, predict_rotation_curve_fast, compute_metrics


def test_zero_shot(galaxies: List, params: Dict, mode: str = 'class') -> List[Dict]:
    """
    Test galaxies with zero-shot parameters (no per-galaxy optimization).
    
    Args:
        galaxies: List of SPARCGalaxy objects
        params: Dictionary with 'class_params' and 'global_median_params'
        mode: 'class' (use type-specific params), 'global' (single set), or 'both'
    
    Returns:
        List of results dictionaries
    """
    results = []
    
    for i, galaxy in enumerate(galaxies, 1):
        print(f"[{i:3d}/{len(galaxies)}] Testing {galaxy.name:12s} ({galaxy.type_group:12s})...", end=' ')
        
        try:
            # Select parameters based on mode
            if mode == 'class':
                test_params = params['class_params'][galaxy.type_group]
                param_source = f"class_{galaxy.type_group}"
            elif mode == 'global':
                test_params = params['global_median_params']
                param_source = "global_median"
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Predict rotation curve
            start = time.time()
            v_pred = predict_rotation_curve_fast(
                galaxy, 
                test_params, 
                use_bulge_gate=True,
                n_particles=100000
            )
            elapsed = time.time() - start
            
            # Compute metrics
            metrics = compute_metrics(galaxy.v_obs, v_pred, galaxy.v_err)
            
            result = {
                'name': galaxy.name,
                'hubble_type': galaxy.hubble_name,
                'type_group': galaxy.type_group,
                'param_source': param_source,
                'ape': metrics['ape'],
                'rms': metrics['rms'],
                'chi2_reduced': metrics['chi2_reduced'],
                'n_points': metrics['n_points'],
                'elapsed': elapsed,
                'params_used': test_params
            }
            
            print(f"APE: {metrics['ape']:6.2f}%  ({elapsed:.1f}s)")
            results.append(result)
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'name': galaxy.name,
                'hubble_type': galaxy.hubble_name,
                'type_group': galaxy.type_group,
                'param_source': param_source if 'param_source' in locals() else 'unknown',
                'ape': np.inf,
                'error': str(e)
            })
    
    return results


def analyze_results(results: List[Dict], mode: str) -> Dict:
    """Compute summary statistics."""
    successful = [r for r in results if 'ape' in r and np.isfinite(r['ape'])]
    
    if not successful:
        return {'error': 'No successful predictions'}
    
    apes = [r['ape'] for r in successful]
    
    # Overall stats
    stats = {
        'mode': mode,
        'n_total': len(results),
        'n_successful': len(successful),
        'n_failed': len(results) - len(successful),
        'mean_ape': float(np.mean(apes)),
        'median_ape': float(np.median(apes)),
        'std_ape': float(np.std(apes)),
        'min_ape': float(np.min(apes)),
        'max_ape': float(np.max(apes)),
        'q25_ape': float(np.percentile(apes, 25)),
        'q75_ape': float(np.percentile(apes, 75)),
    }
    
    # Per-type stats
    stats['by_type'] = {}
    for type_group in ['early', 'intermediate', 'late']:
        type_results = [r for r in successful if r['type_group'] == type_group]
        if type_results:
            type_apes = [r['ape'] for r in type_results]
            stats['by_type'][type_group] = {
                'n': len(type_results),
                'mean_ape': float(np.mean(type_apes)),
                'median_ape': float(np.median(type_apes)),
                'std_ape': float(np.std(type_apes)),
                'min_ape': float(np.min(type_apes)),
                'max_ape': float(np.max(type_apes)),
            }
    
    return stats


def print_summary(stats: Dict):
    """Pretty print summary statistics."""
    print("\n" + "=" * 80)
    print(f"ZERO-SHOT RESULTS ({stats['mode'].upper()} MODE)")
    print("=" * 80)
    print(f"Successful: {stats['n_successful']}/{stats['n_total']}")
    print(f"Failed:     {stats['n_failed']}")
    print()
    print("Overall APE Statistics:")
    print(f"  Mean:      {stats['mean_ape']:6.2f}%")
    print(f"  Median:    {stats['median_ape']:6.2f}%")
    print(f"  Std Dev:   {stats['std_ape']:6.2f}%")
    print(f"  Min:       {stats['min_ape']:6.2f}%")
    print(f"  Max:       {stats['max_ape']:6.2f}%")
    print(f"  Q25-Q75:   {stats['q25_ape']:6.2f}% - {stats['q75_ape']:6.2f}%")
    
    if 'by_type' in stats:
        print()
        print("By Morphological Type:")
        for type_name in ['early', 'intermediate', 'late']:
            if type_name in stats['by_type']:
                t = stats['by_type'][type_name]
                print(f"  {type_name.capitalize():12s} (n={t['n']:3d}): "
                      f"median={t['median_ape']:6.2f}%, "
                      f"mean={t['mean_ape']:6.2f}%, "
                      f"range=[{t['min_ape']:5.2f}-{t['max_ape']:6.2f}]%")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='SPARC Zero-Shot Class-Wise Testing')
    parser.add_argument('--params', required=True, help='Path to class parameters JSON')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--mode', default='class', choices=['class', 'global', 'both'],
                       help='Test mode: class-wise, global, or both')
    parser.add_argument('--data_dir', default='data/Rotmod_LTG', help='SPARC data directory')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    master_file = data_dir / "MasterSheet_SPARC.mrt"
    
    # Load parameters
    with open(args.params, 'r') as f:
        params = json.load(f)
    
    # Load all galaxies
    print("Loading SPARC galaxies...")
    master_info = load_sparc_master_table(master_file)
    
    rotmod_files = sorted(data_dir.glob('*_rotmod.dat'))
    galaxies = []
    for rotmod_file in rotmod_files:
        try:
            galaxy = load_sparc_galaxy(rotmod_file, master_info)
            galaxies.append(galaxy)
        except Exception as e:
            print(f"Warning: Failed to load {rotmod_file.name}: {e}")
    
    print(f"Loaded {len(galaxies)} galaxies")
    print(f"GPU acceleration: {'ENABLED' if _USING_CUPY else 'DISABLED'}")
    
    # Determine which modes to run
    modes_to_run = []
    if args.mode == 'both':
        modes_to_run = ['class', 'global']
    else:
        modes_to_run = [args.mode]
    
    # Run tests
    all_results = {}
    all_stats = {}
    
    for mode in modes_to_run:
        print("\n" + "=" * 80)
        print(f"RUNNING ZERO-SHOT TEST: {mode.upper()} MODE")
        print("=" * 80)
        
        if mode == 'class':
            print("\nParameters by type:")
            for type_name, type_params in params['class_params'].items():
                print(f"  {type_name.capitalize():12s}: eta={type_params['eta']:.3f}, "
                      f"ring_amp={type_params['ring_amp']:.3f}, "
                      f"M_max={type_params['M_max']:.3f}")
        else:
            gp = params['global_median_params']
            print("\nGlobal parameters:")
            print(f"  eta={gp['eta']:.3f}, ring_amp={gp['ring_amp']:.3f}, "
                  f"M_max={gp['M_max']:.3f}, "
                  f"bulge_gate_power={gp['bulge_gate_power']:.2f}, "
                  f"lambda_hat={gp['lambda_hat']:.2f}")
        
        print()
        
        # Run zero-shot test
        results = test_zero_shot(galaxies, params, mode=mode)
        
        # Analyze
        stats = analyze_results(results, mode)
        
        # Print summary
        print_summary(stats)
        
        # Save results
        df = pd.DataFrame(results)
        csv_file = output_dir / f'zero_shot_{mode}_results.csv'
        df.to_csv(csv_file, index=False)
        print(f"\nResults saved to: {csv_file}")
        
        # Save full results with stats
        output_data = {
            'mode': mode,
            'parameters': params['class_params'] if mode == 'class' else {'global': params['global_median_params']},
            'statistics': stats,
            'results': results
        }
        json_file = output_dir / f'zero_shot_{mode}_complete.json'
        with open(json_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Complete results saved to: {json_file}")
        
        all_results[mode] = results
        all_stats[mode] = stats
    
    # If both modes were run, compare them
    if len(modes_to_run) == 2:
        print("\n" + "=" * 80)
        print("COMPARISON: CLASS vs GLOBAL")
        print("=" * 80)
        print(f"Class-wise median APE:  {all_stats['class']['median_ape']:6.2f}%")
        print(f"Global median APE:      {all_stats['global']['median_ape']:6.2f}%")
        print(f"Improvement from class: {all_stats['global']['median_ape'] - all_stats['class']['median_ape']:+6.2f}%")
        print()
        print("This shows how much **class conditioning** improves universality")
        print("=" * 80)
        
        # Save comparison
        comparison = {
            'class_stats': all_stats['class'],
            'global_stats': all_stats['global'],
            'delta_median_ape': all_stats['global']['median_ape'] - all_stats['class']['median_ape'],
            'improvement_pct': 100 * (1 - all_stats['class']['median_ape'] / all_stats['global']['median_ape'])
        }
        comp_file = output_dir / 'comparison.json'
        with open(comp_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {comp_file}")


if __name__ == '__main__':
    main()
