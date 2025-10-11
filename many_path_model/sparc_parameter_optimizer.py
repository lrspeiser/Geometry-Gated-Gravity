#!/usr/bin/env python3
"""
SPARC Parameter Optimizer - Find optimal many-path gravity parameters

This optimizer systematically searches parameter space to minimize APE across
galaxy types. It uses a coarse-to-fine grid search strategy with early stopping.

Key strategy:
1. Start with MW-frozen structure (gates, anisotropy, saturation)
2. Optimize AMPLITUDE parameters (eta, ring_amp) which are likely off-scale
3. Use median APE across diverse types as fitness (robust to outliers)
4. Track improvements and stop if no progress

Usage:
    # Stage A: Optimize disk physics (late-type galaxies)
    python sparc_parameter_optimizer.py --stage disk \
        --filter_types Sm,Scd --n_galaxies 20 \
        --output results/stage_a_disk_params.json
    
    # Stage B: Optimize bulge physics (early-type galaxies)  
    python sparc_parameter_optimizer.py --stage bulge \
        --filter_types Sbc,Sb --n_galaxies 20 \
        --load_base results/stage_a_disk_params.json \
        --output results/stage_b_bulge_params.json
"""

import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from itertools import product

# Import SPARC testing infrastructure
try:
    from sparc_stratified_test import (
        load_sparc_master_table, load_sparc_galaxy, 
        test_galaxy, load_mw_frozen_params, SPARCGalaxy
    )
except ImportError:
    print("ERROR: Could not import sparc_stratified_test")
    sys.exit(1)


class ParameterOptimizer:
    """
    Intelligent parameter optimizer for SPARC galaxies.
    
    Strategy:
    - Keep MW-frozen structure (gates, scales, anisotropy)
    - Optimize amplitudes (eta, ring_amp) which scale the effect
    - Use coarse-to-fine grid search
    - Early stopping if no improvement for N iterations
    """
    
    def __init__(self, galaxies: List[SPARCGalaxy], base_params: Dict,
                 n_particles: int = 100000):
        self.galaxies = galaxies
        self.base_params = base_params.copy()
        self.n_particles = n_particles
        self.best_params = base_params.copy()
        self.best_score = np.inf
        self.history = []
        
    def evaluate_params(self, params: Dict, use_bulge_gate: bool = False) -> Dict:
        """
        Evaluate parameters on all galaxies.
        
        Returns dict with median_ape, success_rate, and per-type metrics.
        """
        results = []
        
        for gal in self.galaxies:
            try:
                result = test_galaxy(gal, params, use_bulge_gate, self.n_particles)
                if 'ape' in result and np.isfinite(result['ape']):
                    results.append(result)
            except Exception as e:
                print(f"    Error on {gal.name}: {e}")
                continue
        
        if not results:
            return {'median_ape': np.inf, 'mean_ape': np.inf, 'success_rate': 0.0}
        
        apes = [r['ape'] for r in results]
        successes = [r['success'] for r in results]
        
        # Compute by-type metrics
        by_type = {}
        for group in ['late', 'intermediate', 'early']:
            group_results = [r for r in results if r.get('type_group') == group]
            if group_results:
                group_apes = [r['ape'] for r in group_results]
                by_type[group] = {
                    'n': len(group_results),
                    'median_ape': float(np.median(group_apes)),
                    'mean_ape': float(np.mean(group_apes))
                }
        
        return {
            'median_ape': float(np.median(apes)),
            'mean_ape': float(np.mean(apes)),
            'success_rate': float(np.mean(successes)),
            'n_evaluated': len(results),
            'by_type': by_type
        }
    
    def grid_search(self, param_grid: Dict[str, List], use_bulge_gate: bool = False,
                   max_no_improvement: int = 10) -> Dict:
        """
        Coarse-to-fine grid search with early stopping.
        
        param_grid: Dict mapping parameter names to lists of values to try
        """
        print(f"\n{'='*80}")
        print("GRID SEARCH")
        print('='*80)
        print(f"Parameters to optimize: {list(param_grid.keys())}")
        print(f"Grid sizes: {[len(v) for v in param_grid.values()]}")
        print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
        print(f"Bulge gating: {'ENABLED' if use_bulge_gate else 'DISABLED'}")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"\nEvaluating {len(combinations)} parameter combinations...")
        
        no_improvement_count = 0
        
        for i, combo in enumerate(combinations, 1):
            # Create test parameters
            test_params = self.base_params.copy()
            for name, value in zip(param_names, combo):
                test_params[name] = value
            
            # Evaluate
            print(f"\n[{i}/{len(combinations)}] Testing:", end=" ")
            for name, value in zip(param_names, combo):
                print(f"{name}={value:.3f}", end=" ")
            print()
            
            start = time.time()
            metrics = self.evaluate_params(test_params, use_bulge_gate)
            elapsed = time.time() - start
            
            median_ape = metrics['median_ape']
            
            print(f"  Median APE: {median_ape:.2f}%")
            print(f"  Success rate: {metrics['success_rate']*100:.1f}%")
            print(f"  Evaluated in {elapsed:.1f}s")
            
            # Print by-type if available
            if 'by_type' in metrics:
                for group, gmetrics in metrics['by_type'].items():
                    print(f"    {group:12s}: {gmetrics['median_ape']:.1f}% (n={gmetrics['n']})")
            
            # Track history
            self.history.append({
                'iteration': i,
                'params': dict(zip(param_names, combo)),
                'metrics': metrics
            })
            
            # Check for improvement
            if median_ape < self.best_score:
                improvement = self.best_score - median_ape
                self.best_score = median_ape
                self.best_params = test_params.copy()
                no_improvement_count = 0
                print(f"  ✓ NEW BEST! (improved by {improvement:.2f}%)")
            else:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    print(f"\n  Early stopping: No improvement for {max_no_improvement} iterations")
                    break
        
        print(f"\n{'='*80}")
        print("GRID SEARCH COMPLETE")
        print('='*80)
        print(f"Best median APE: {self.best_score:.2f}%")
        print(f"Best parameters:")
        for name in param_names:
            print(f"  {name:20s}: {self.best_params[name]:.4f}")
        
        return self.best_params
    
    def save_results(self, output_path: Path):
        """Save optimization results to JSON."""
        results = {
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'n_galaxies': len(self.galaxies),
            'galaxy_types': [g.type_group for g in self.galaxies],
            'history': [
                {
                    'iteration': h['iteration'],
                    'params': h['params'],
                    'median_ape': h['metrics']['median_ape'],
                    'mean_ape': h['metrics']['mean_ape'],
                    'success_rate': h['metrics']['success_rate']
                }
                for h in self.history
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SPARC parameter optimizer")
    parser.add_argument('--sparc_dir', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/external_data/Rotmod_LTG')
    parser.add_argument('--master_file', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/data/SPARC_Lelli2016c.mrt')
    
    # Galaxy selection
    parser.add_argument('--stage', type=str, choices=['disk', 'bulge', 'universal'],
                       default='disk', help='Optimization stage')
    parser.add_argument('--filter_types', type=str, default='',
                       help='Comma-separated Hubble types to include')
    parser.add_argument('--n_galaxies', type=int, default=20,
                       help='Number of galaxies (random sample from filtered)')
    
    # Optimization strategy
    parser.add_argument('--load_base', type=str, default='',
                       help='Load base parameters from previous stage JSON')
    parser.add_argument('--use_bulge_gate', type=int, default=0,
                       help='Enable bulge gating')
    
    # Parameter ranges (coarse grid)
    parser.add_argument('--eta_range', type=str, default='0.05,0.1,0.15,0.2,0.25,0.3,0.35',
                       help='Comma-separated eta values')
    parser.add_argument('--ring_amp_range', type=str, default='0.0,0.03,0.05,0.07,0.1',
                       help='Comma-separated ring_amp values')
    parser.add_argument('--M_max_range', type=str, default='2.5,3.0,3.5,4.0',
                       help='Comma-separated M_max values')
    
    # Performance
    parser.add_argument('--n_particles', type=int, default=100000)
    parser.add_argument('--max_no_improvement', type=int, default=10,
                       help='Early stopping threshold')
    
    # Output
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("="*80)
    print("SPARC PARAMETER OPTIMIZER")
    print("="*80)
    print(f"Stage: {args.stage}")
    print(f"Target galaxies: {args.n_galaxies}")
    
    # Load base parameters
    if args.load_base:
        print(f"Loading base parameters from: {args.load_base}")
        with open(args.load_base, 'r') as f:
            loaded = json.load(f)
            base_params = loaded['best_params']
    else:
        print("Loading MW-frozen baseline parameters")
        base_params = load_mw_frozen_params()
    
    # Load SPARC data
    print(f"\nLoading SPARC data from {args.sparc_dir}...")
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
    
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Filter by type
    if args.filter_types:
        filter_types = [t.strip() for t in args.filter_types.split(',')]
        galaxies = [g for g in galaxies if g.hubble_name in filter_types]
        print(f"Filtered to types {filter_types}: {len(galaxies)} galaxies")
    
    # Random sample
    if len(galaxies) > args.n_galaxies:
        galaxies = list(np.random.choice(galaxies, args.n_galaxies, replace=False))
        print(f"Random sample: {len(galaxies)} galaxies")
    
    if not galaxies:
        print("ERROR: No galaxies selected!")
        sys.exit(1)
    
    # Show galaxy distribution
    type_counts = {}
    for g in galaxies:
        type_counts[g.type_group] = type_counts.get(g.type_group, 0) + 1
    print(f"\nGalaxy types: {type_counts}")
    
    # Define parameter grid based on stage
    param_grid = {}
    
    if args.stage == 'disk':
        # Stage A: Optimize disk amplitudes (late-type galaxies)
        param_grid['eta'] = [float(x) for x in args.eta_range.split(',')]
        param_grid['ring_amp'] = [float(x) for x in args.ring_amp_range.split(',')]
        
    elif args.stage == 'bulge':
        # Stage B: Optimize bulge gating (early-type galaxies)
        param_grid['bulge_gate_power'] = [1.0, 1.5, 2.0, 2.5, 3.0]
        # Fine-tune eta around best disk value
        best_eta = base_params.get('eta', 0.2)
        param_grid['eta'] = [best_eta * 0.8, best_eta, best_eta * 1.2]
        
    else:  # universal
        # Full optimization
        param_grid['eta'] = [float(x) for x in args.eta_range.split(',')]
        param_grid['ring_amp'] = [float(x) for x in args.ring_amp_range.split(',')]
        param_grid['M_max'] = [float(x) for x in args.M_max_range.split(',')]
    
    # Create optimizer
    optimizer = ParameterOptimizer(galaxies, base_params, args.n_particles)
    
    # Run optimization
    best_params = optimizer.grid_search(
        param_grid, 
        use_bulge_gate=bool(args.use_bulge_gate),
        max_no_improvement=args.max_no_improvement
    )
    
    # Save results
    optimizer.save_results(Path(args.output))
    
    # Print final summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print('='*80)
    print(f"Evaluated {len(optimizer.history)} parameter combinations")
    print(f"Best median APE: {optimizer.best_score:.2f}%")
    print(f"\nNext steps:")
    if args.stage == 'disk':
        print(f"  python sparc_parameter_optimizer.py --stage bulge \\")
        print(f"    --filter_types Sbc,Sb,Sab --load_base {args.output} \\")
        print(f"    --output results/stage_b_bulge_params.json")
    elif args.stage == 'bulge':
        print(f"  # Test full sample with optimized parameters")
        print(f"  # Then run parameter_optimizer with --stage universal")


if __name__ == '__main__':
    main()
