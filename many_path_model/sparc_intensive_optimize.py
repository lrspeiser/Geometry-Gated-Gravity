#!/usr/bin/env python3
"""
Intensive Per-Galaxy Optimization with Parallel Processing

Runs aggressive optimization for each galaxy independently:
- Extended CMA-ES iterations (200+ instead of 30)
- Multiple random restarts per galaxy
- Parallel processing across GPUs if available
- Saves best result for each galaxy

Usage:
    python sparc_intensive_optimize.py \
        --global_params_file results/multistart/multistart_results.json \
        --output_dir results/intensive_per_galaxy \
        --max_iter 200 \
        --n_restarts 5 \
        --n_workers 8
"""

import argparse
import sys
import json
import time
import pickle
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import cupy as cp
    HAS_CUPY = True
    print("✓ CuPy available - GPU acceleration ENABLED")
except ImportError:
    cp = np
    HAS_CUPY = False
    print("✗ CuPy not available - CPU mode")

from sparc_stratified_test import load_sparc_master_table, load_sparc_galaxy, SPARCGalaxy
from sparc_hierarchical_search_v2 import HierarchicalSearchV2


def optimize_single_galaxy_intensive(args_tuple):
    """
    Intensive optimization for a single galaxy with multiple restarts.
    
    This function will be called in parallel for each galaxy.
    """
    gal_data, global_params, param_bounds, max_iter, n_restarts, gal_idx, total_galaxies = args_tuple
    
    try:
        # Re-import in subprocess (needed for multiprocessing)
        import cupy as cp
        import cma
        from sparc_stratified_test import SPARCGalaxy
        from sparc_hierarchical_search_v2 import HierarchicalSearchV2
        
        # Reconstruct galaxy object from data
        gal = SPARCGalaxy(
            name=gal_data['name'],
            r_kpc=np.array(gal_data['r_kpc']),
            v_obs=np.array(gal_data['v_obs']),
            v_err=np.array(gal_data['v_err']),
            v_gas=np.array(gal_data['v_gas']),
            v_disk=np.array(gal_data['v_disk']),
            v_bulge=np.array(gal_data['v_bulge']),
            bulge_frac=np.array(gal_data['bulge_frac']),
            type_group=gal_data['type_group']
        )
        gal.R_d_kpc = gal_data['R_d_kpc']
        
        print(f"\n[{gal_idx+1}/{total_galaxies}] Intensive optimization: {gal.name} ({gal.type_group})")
        print(f"  Starting {n_restarts} restarts with {max_iter} iterations each...")
        
        # Create single-galaxy searcher
        searcher = HierarchicalSearchV2([gal], param_bounds)
        
        # Track best across all restarts
        overall_best_score = np.inf
        overall_best_params = None
        overall_best_x = None
        
        # Multiple restarts with different initializations
        for restart_idx in range(n_restarts):
            # Starting point for this restart
            if restart_idx == 0:
                # First restart: start from global params
                x0 = [global_params[name] for name in param_bounds.keys()]
                sigma0 = 0.3  # 30% variation
            else:
                # Subsequent restarts: random initialization within bounds
                x0 = []
                for name in param_bounds.keys():
                    low, high = param_bounds[name]
                    x0.append(np.random.uniform(low, high))
                sigma0 = 0.5  # Wider exploration
            
            def objective(x):
                params_gpu = cp.array([x], dtype=cp.float32)
                score = float(searcher.evaluate_params_vectorized(params_gpu)[0])
                return score
            
            # Bounds for CMA-ES
            bounds = [[param_bounds[name][0] for name in param_bounds.keys()],
                      [param_bounds[name][1] for name in param_bounds.keys()]]
            
            # CMA-ES with extended iterations
            es = cma.CMAEvolutionStrategy(x0, sigma0, {
                'bounds': bounds,
                'popsize': 30,  # Larger population
                'maxiter': max_iter,
                'verb_disp': 0,
                'verb_log': 0,
                'tolfun': 1e-4,  # Tighter convergence
                'tolx': 1e-6
            })
            
            # Run optimization
            es.optimize(objective)
            
            # Check if this restart found a better solution
            if es.result.fbest < overall_best_score:
                overall_best_score = es.result.fbest
                overall_best_x = es.result.xbest
                overall_best_params = {name: float(overall_best_x[i]) 
                                      for i, name in enumerate(param_bounds.keys())}
                print(f"    Restart {restart_idx+1}/{n_restarts}: {overall_best_score:.3f}% APE ✓ NEW BEST")
            else:
                print(f"    Restart {restart_idx+1}/{n_restarts}: {es.result.fbest:.3f}% APE")
        
        print(f"  ✓ Best result: {overall_best_score:.3f}% APE")
        
        result = {
            'name': gal.name,
            'params': overall_best_params,
            'score': float(overall_best_score),
            'type_group': gal.type_group,
            'n_restarts': n_restarts,
            'max_iter': max_iter
        }
        
        return result
        
    except Exception as e:
        print(f"  ✗ Failed {gal_data['name']}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'name': gal_data['name'],
            'error': str(e),
            'type_group': gal_data.get('type_group', 'unknown')
        }


def main():
    parser = argparse.ArgumentParser(description="Intensive Per-Galaxy Optimization")
    parser.add_argument('--sparc_dir', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/external_data/Rotmod_LTG')
    parser.add_argument('--master_file', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/data/SPARC_Lelli2016c.mrt')
    parser.add_argument('--global_params_file', type=str, required=True,
                       help='Path to global parameters JSON')
    parser.add_argument('--output_dir', type=str, default='results/intensive_per_galaxy')
    parser.add_argument('--max_iter', type=int, default=200,
                       help='Max CMA-ES iterations per restart')
    parser.add_argument('--n_restarts', type=int, default=5,
                       help='Number of random restarts per galaxy')
    parser.add_argument('--n_workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Load SPARC data
    print("="*80)
    print("INTENSIVE PER-GALAXY OPTIMIZATION")
    print("="*80)
    print(f"Settings:")
    print(f"  Max iterations per restart: {args.max_iter}")
    print(f"  Random restarts per galaxy: {args.n_restarts}")
    print(f"  Parallel workers: {args.n_workers}")
    print()
    
    master_info = load_sparc_master_table(Path(args.master_file))
    sparc_dir = Path(args.sparc_dir)
    galaxy_files = list(sparc_dir.glob('*_rotmod.dat'))
    
    galaxies = []
    for gfile in galaxy_files:
        try:
            gal = load_sparc_galaxy(gfile, master_info)
            gal.R_d_kpc = np.max(gal.r_kpc) / 3.0  # Approximation
            galaxies.append(gal)
        except Exception:
            continue
    
    print(f"✓ Loaded {len(galaxies)} galaxies")
    
    # Load global parameters
    with open(args.global_params_file, 'r') as f:
        global_params_data = json.load(f)
        if 'best_params' in global_params_data:
            global_params = global_params_data['best_params']
        else:
            global_params = global_params_data
    
    # Add default λ_hat if not present
    if 'lambda_hat' not in global_params:
        global_params['lambda_hat'] = 20.0
    
    print(f"\nStarting from global parameters:")
    for name, val in global_params.items():
        print(f"  {name:20s} = {val:.6f}")
    
    # Parameter bounds (wider for intensive search)
    param_bounds = {
        'eta': (0.01, 2.0),
        'ring_amp': (0.0, 15.0),
        'M_max': (0.5, 5.0),
        'bulge_gate_power': (0.5, 60.0),
        'lambda_hat': (5.0, 50.0)
    }
    
    # Prepare galaxy data for multiprocessing (serialize for pickling)
    galaxy_data_list = []
    for i, gal in enumerate(galaxies):
        gal_data = {
            'name': gal.name,
            'r_kpc': gal.r_kpc.tolist(),
            'v_obs': gal.v_obs.tolist(),
            'v_err': gal.v_err.tolist(),
            'v_gas': gal.v_gas.tolist(),
            'v_disk': gal.v_disk.tolist(),
            'v_bulge': gal.v_bulge.tolist(),
            'bulge_frac': gal.bulge_frac.tolist(),
            'type_group': gal.type_group,
            'R_d_kpc': gal.R_d_kpc
        }
        args_tuple = (gal_data, global_params, param_bounds, 
                     args.max_iter, args.n_restarts, i, len(galaxies))
        galaxy_data_list.append(args_tuple)
    
    # Run parallel optimization
    print(f"\n{'='*80}")
    print(f"PARALLEL OPTIMIZATION ({args.n_workers} workers)")
    print(f"{'='*80}\n")
    
    results = {}
    start_time = time.time()
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        # Submit all tasks
        future_to_galaxy = {
            executor.submit(optimize_single_galaxy_intensive, gal_args): gal_args[0]['name']
            for gal_args in galaxy_data_list
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_galaxy):
            galaxy_name = future_to_galaxy[future]
            try:
                result = future.result()
                if 'error' not in result:
                    results[result['name']] = result
                    completed += 1
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(galaxies) - completed) / rate if rate > 0 else 0
                    
                    print(f"\nCompleted {completed}/{len(galaxies)} galaxies "
                          f"({elapsed/60:.1f}m elapsed, ETA: {eta/60:.1f}m)")
            except Exception as e:
                print(f"\nError processing {galaxy_name}: {e}")
    
    total_time = time.time() - start_time
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'intensive_per_galaxy_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successfully optimized: {len(results)}/{len(galaxies)} galaxies")
    print(f"Average time per galaxy: {total_time/len(results):.1f}s")
    print(f"\nResults saved to: {output_file}")
    
    # Score statistics
    scores = [r['score'] for r in results.values() if 'score' in r]
    if scores:
        print(f"\nScore statistics (APE %):")
        print(f"  Median: {np.median(scores):.2f}%")
        print(f"  Mean:   {np.mean(scores):.2f}%")
        print(f"  Min:    {np.min(scores):.2f}%")
        print(f"  Max:    {np.max(scores):.2f}%")
        print(f"  Std:    {np.std(scores):.2f}%")
        
        # Count by quality
        excellent = sum(1 for s in scores if s < 5)
        good = sum(1 for s in scores if 5 <= s < 10)
        fair = sum(1 for s in scores if 10 <= s < 20)
        poor = sum(1 for s in scores if s >= 20)
        
        print(f"\nFit quality distribution:")
        print(f"  Excellent (< 5%):    {excellent:3d} ({100*excellent/len(scores):.1f}%)")
        print(f"  Good (5-10%):        {good:3d} ({100*good/len(scores):.1f}%)")
        print(f"  Fair (10-20%):       {fair:3d} ({100*fair/len(scores):.1f}%)")
        print(f"  Poor (≥ 20%):        {poor:3d} ({100*poor/len(scores):.1f}%)")


if __name__ == '__main__':
    main()
