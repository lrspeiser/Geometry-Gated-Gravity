#!/usr/bin/env python3
"""
SPARC Mega Parallel Optimizer
==============================

Runs intensive per-galaxy optimization across all 175 SPARC galaxies in parallel.
Each galaxy is optimized hundreds of thousands of times using multiple restarts.

Features:
- Full multiprocessing across all CPU cores
- Each worker loads complete galaxy data (no serialization issues)
- Multiple CMA-ES restarts per galaxy (default: 10)
- Extended iterations per restart (default: 500)
- GPU acceleration within each worker
- Real-time progress tracking
- Comprehensive results logging

Usage:
    # Mega optimization with 10 restarts, 500 iterations each, 8 workers
    python many_path_model/sparc_mega_parallel.py --global_params results/multistart/multistart_results.json --output_dir results/mega_parallel --iterations 500 --restarts 10 --workers 8

    # Quick test with fewer iterations
    python many_path_model/sparc_mega_parallel.py --global_params results/multistart/multistart_results.json --output_dir results/mega_test --iterations 100 --restarts 3 --workers 4
"""

import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np

# GPU check
try:
    import cupy as cp
    _USING_CUPY = True
except ImportError:
    import numpy as cp
    _USING_CUPY = False

# Import galaxy loading functions
sys.path.insert(0, str(Path(__file__).parent))
from sparc_stratified_test import load_sparc_galaxy, load_sparc_master_table

# Import optimization
from sparc_hierarchical_search_v2 import HierarchicalSearchV2


def optimize_single_galaxy(args):
    """
    Optimize a single galaxy with multiple restarts.
    This runs in a separate process.
    
    Args:
        args: Tuple of (galaxy_name, global_params, param_bounds, data_dir, iterations, restarts)
    
    Returns:
        dict: Results for this galaxy
    """
    galaxy_name, global_params, param_bounds, data_dir, iterations, restarts = args
    
    try:
        # Load complete galaxy data from original files
        master_file = data_dir / "MasterSheet_SPARC.mrt"
        rotmod_file = data_dir / f"{galaxy_name}_rotmod.dat"
        
        if not rotmod_file.exists():
            return {
                'name': galaxy_name,
                'success': False,
                'error': f'File not found: {rotmod_file}'
            }
        
        # Load master table
        master_info = load_sparc_master_table(master_file)
        
        # Load galaxy with all required fields
        galaxy = load_sparc_galaxy(rotmod_file, master_info)
        
        # Create hierarchical search instance (requires parameter bounds)
        search = HierarchicalSearchV2(galaxies=[galaxy], param_bounds=param_bounds)
        
        # Run multiple optimization restarts using CMA-ES
        best_error = float('inf')
        best_params = None
        all_attempts = []
        
        start_time = time.time()
        
        # Import locally to ensure availability in subprocess
        import cma
        
        for restart_idx in range(restarts):
            # Starting point for this restart
            if restart_idx == 0:
                # First restart from provided global params
                x0 = [global_params[name] for name in param_bounds.keys()]
                sigma0 = 0.3
            else:
                # Subsequent restarts: random init within bounds
                x0 = [
                    np.random.uniform(param_bounds[name][0], param_bounds[name][1])
                    for name in param_bounds.keys()
                ]
                sigma0 = 0.5
            
            def objective(x):
                params_gpu = cp.array([x], dtype=cp.float32)
                return float(search.evaluate_params_vectorized(params_gpu)[0])
            
            bounds = [[param_bounds[name][0] for name in param_bounds.keys()],
                      [param_bounds[name][1] for name in param_bounds.keys()]]
            
            es = cma.CMAEvolutionStrategy(x0, sigma0, {
                'bounds': bounds,
                'popsize': 20,
                'maxiter': iterations,
                'verb_disp': 0,
                'verb_log': 0
            })
            
            es.optimize(objective)
            err = float(es.result.fbest)
            xbest = es.result.xbest
            params_dict = {name: float(xbest[i]) for i, name in enumerate(param_bounds.keys())}
            
            all_attempts.append({
                'restart': restart_idx,
                'error': err,
                'params': params_dict
            })
            
            if err < best_error:
                best_error = err
                best_params = params_dict
        
        elapsed = time.time() - start_time
        
        return {
            'name': galaxy_name,
            'success': True,
            'hubble_type': galaxy.hubble_name,
            'type_group': galaxy.type_group,
            'best_error': best_error,
            'best_params': best_params,
            'restarts': restarts,
            'iterations_per_restart': iterations,
            'total_evaluations': restarts * iterations * 20,  # population_size=20
            'elapsed_time': elapsed,
            'all_attempts': all_attempts
        }
        
    except Exception as e:
        return {
            'name': galaxy_name,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='SPARC Mega Parallel Optimizer')
    parser.add_argument('--global_params', required=True, help='Path to global parameters JSON file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--iterations', type=int, default=500, help='CMA-ES iterations per restart')
    parser.add_argument('--restarts', type=int, default=10, help='Number of random restarts per galaxy')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--data_dir', default='data/Rotmod_LTG', help='Path to SPARC data directory')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Load global parameters
    with open(args.global_params, 'r') as f:
        data = json.load(f)
        global_params = data['best_params'] if 'best_params' in data else data
    
    # Ensure V2 includes dimensionless lambda_hat
    if 'lambda_hat' not in global_params:
        global_params['lambda_hat'] = 20.0
    
    # Parameter bounds for V2 (5 parameters)
    param_bounds = {
        'eta': (0.01, 2.0),
        'ring_amp': (0.0, 15.0),
        'M_max': (0.5, 5.0),
        'bulge_gate_power': (0.5, 60.0),
        'lambda_hat': (5.0, 50.0)
    }
    
    # Get all galaxy names from data directory
    rotmod_files = sorted(data_dir.glob('*_rotmod.dat'))
    galaxy_names = [f.stem.replace('_rotmod', '') for f in rotmod_files]
    
    print("=" * 80)
    print("SPARC MEGA PARALLEL OPTIMIZER")
    print("=" * 80)
    print(f"Galaxies:              {len(galaxy_names)}")
    print(f"Restarts per galaxy:   {args.restarts}")
    print(f"Iterations per restart: {args.iterations}")
    print(f"Evaluations per galaxy: {args.restarts * args.iterations * 20:,}")
    print(f"Total evaluations:      {len(galaxy_names) * args.restarts * args.iterations * 20:,}")
    
    # Determine worker count
    n_workers = args.workers if args.workers else cpu_count()
    print(f"Parallel workers:       {n_workers}")
    print(f"GPU acceleration:       {'ENABLED (CuPy)' if _USING_CUPY else 'DISABLED'}")
    print()
    print("Global parameters:")
    for key, val in global_params.items():
        print(f"  {key:20s} = {f'{val:.6f}' if isinstance(val, (int, float)) else str(val) if isinstance(val, dict) else f'{val}'}")
    print("=" * 80)
    print()
    
    # Prepare work items
    work_items = [
        (name, global_params, param_bounds, data_dir, args.iterations, args.restarts)
        for name in galaxy_names
    ]
    
    # Run parallel optimization
    print(f"Starting optimization with {n_workers} workers...\n")
    start_time = time.time()
    
    results = []
    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(optimize_single_galaxy, work_items), 1):
            results.append(result)
            
            # Progress update
            if result['success']:
                print(f"[{i:3d}/{len(galaxy_names)}] {result['name']:12s} "
                      f"({result['type_group']:12s}) - "
                      f"Best APE: {result['best_error']:6.2f}% "
                      f"({result['elapsed_time']:5.1f}s, "
                      f"{result['total_evaluations']:,} evals)")
            else:
                print(f"[{i:3d}/{len(galaxy_names)}] {result['name']:12s} - FAILED: {result.get('error', 'Unknown')}")
    
    elapsed = time.time() - start_time
    
    # Compute statistics
    errors = [] # Initialize errors list
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print()
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Total time:       {elapsed/60:.1f} minutes")
    print(f"Successful:       {len(successful)}/{len(galaxy_names)}")
    print(f"Failed:           {len(failed)}")
    
    if successful:
        errors = [r['best_error'] for r in successful]
        print()
        print("Error Statistics:")
        print(f"  Mean APE:       {np.mean(errors):.2f}%")
        print(f"  Median APE:     {np.median(errors):.2f}%")
        print(f"  Min APE:        {np.min(errors):.2f}%")
        print(f"  Max APE:        {np.max(errors):.2f}%")
        print(f"  Std Dev:        {np.std(errors):.2f}%")
        
        # Best and worst
        best = min(successful, key=lambda x: x['best_error'])
        worst = max(successful, key=lambda x: x['best_error'])
        print()
        print(f"Best:  {best['name']} ({best['type_group']}) - {best['best_error']:.2f}% APE")
        print(f"Worst: {worst['name']} ({worst['type_group']}) - {worst['best_error']:.2f}% APE")
    
    # Save results
    output_file = output_dir / 'mega_parallel_results.json'
    output_data = {
        'config': {
            'iterations': args.iterations,
            'restarts': args.restarts,
            'workers': n_workers,
            'data_dir': str(data_dir),
            'global_params': global_params,
            'total_galaxies': len(galaxy_names),
            'total_evaluations': len(galaxy_names) * args.restarts * args.iterations * 20,
            'total_time': elapsed
        },
        'results': results,
        'statistics': {
            'successful': len(successful),
            'failed': len(failed),
            'mean_ape': float(np.mean(errors)) if errors else None,
            'median_ape': float(np.median(errors)) if errors else None,
            'min_ape': float(np.min(errors)) if errors else None,
            'max_ape': float(np.max(errors)) if errors else None,
            'std_ape': float(np.std(errors)) if errors else None
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print()
    print(f"Results saved to: {output_file}")
    
    # Also save simplified CSV for quick analysis
    csv_file = output_dir / 'mega_parallel_summary.csv'
    with open(csv_file, 'w') as f:
        f.write('name,hubble_type,type_group,best_ape,restarts,total_evals,time_seconds\n')
        for r in successful:
            f.write(f"{r['name']},{r['hubble_type']},{r['type_group']},"
                   f"{r['best_error']:.2f},{r['restarts']},"
                   f"{r['total_evaluations']},{r['elapsed_time']:.1f}\n")
    
    print(f"Summary CSV saved to: {csv_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
