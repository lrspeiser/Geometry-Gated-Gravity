#!/usr/bin/env python3
"""
Multi-Start Refinement for Hierarchical Search

Runs CMA-ES from multiple promising regions to avoid local minima.

Usage:
    # Refine top 5 regions from Stage 2
    python sparc_multistart_refine.py results/hierarchical_10x/stage2_regions.pkl \
        --n_starts 5 --max_iter 50
"""

import argparse
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import time

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from sparc_hierarchical_search import HierarchicalSearch, SearchRegion
from sparc_stratified_test import load_sparc_master_table, load_sparc_galaxy


def multi_start_refinement(regions_file: Path, n_starts: int = 5, 
                           max_iter: int = 50, output_dir: Path = None):
    """
    Run CMA-ES from multiple starting regions to find global optimum.
    
    Args:
        regions_file: Path to stage2_regions.pkl
        n_starts: Number of regions to try
        max_iter: CMA-ES iterations per region
        output_dir: Where to save results
    """
    
    # Load regions from Stage 2
    with open(regions_file, 'rb') as f:
        regions = pickle.load(f)
    
    print("="*80)
    print("MULTI-START REFINEMENT")
    print("="*80)
    print(f"Loaded {len(regions)} regions from Stage 2")
    print(f"Will refine top {n_starts} regions")
    print(f"CMA-ES iterations per region: {max_iter}")
    
    # Load SPARC data
    master_file = Path('C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/data/SPARC_Lelli2016c.mrt')
    sparc_dir = Path('C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/external_data/Rotmod_LTG')
    
    master_info = load_sparc_master_table(master_file)
    galaxy_files = list(sparc_dir.glob('*_rotmod.dat'))
    
    galaxies = []
    for gfile in galaxy_files:
        try:
            gal = load_sparc_galaxy(gfile, master_info)
            galaxies.append(gal)
        except Exception:
            continue
    
    print(f"✓ Loaded {len(galaxies)} galaxies\n")
    
    # Parameter bounds (use wide bounds from search)
    param_bounds = {
        'eta': (0.01, 2.0),
        'ring_amp': (0.0, 3.0),
        'M_max': (0.5, 5.0),
        'bulge_gate_power': (0.5, 20.0)
    }
    
    # Create searcher
    searcher = HierarchicalSearch(galaxies, param_bounds)
    
    # Results tracking
    all_results = []
    best_overall_score = np.inf
    best_overall_params = None
    
    # Try each region
    for i, region in enumerate(regions[:n_starts]):
        print(f"\n{'='*80}")
        print(f"START {i+1}/{n_starts}: Refining region with score={region.score:.2f}%")
        print(f"{'='*80}")
        print(f"Center: {region.center}")
        
        start_time = time.time()
        
        try:
            # Run CMA-ES from this region
            final_params = searcher.stage3_refinement(region, n_iterations=max_iter)
            final_score = searcher.best_score
            
            elapsed = time.time() - start_time
            
            result = {
                'start_number': i + 1,
                'initial_score': region.score,
                'final_score': final_score,
                'improvement': region.score - final_score,
                'final_params': final_params,
                'elapsed_time': elapsed
            }
            all_results.append(result)
            
            # Track global best
            if final_score < best_overall_score:
                best_overall_score = final_score
                best_overall_params = final_params
                print(f"\n🌟 NEW GLOBAL BEST: {best_overall_score:.2f}%")
            
            print(f"\n✓ Start {i+1} complete:")
            print(f"  Initial: {region.score:.2f}% → Final: {final_score:.2f}%")
            print(f"  Improvement: {result['improvement']:.2f}%")
            print(f"  Time: {elapsed:.1f}s")
        
        except Exception as e:
            print(f"\n⚠️  Start {i+1} failed: {e}")
            continue
    
    # Summary
    print("\n" + "="*80)
    print("MULTI-START REFINEMENT COMPLETE")
    print("="*80)
    
    print(f"\nResults from {len(all_results)} successful starts:")
    all_results.sort(key=lambda x: x['final_score'])
    
    for i, res in enumerate(all_results):
        print(f"\n{i+1}. Start #{res['start_number']}: {res['final_score']:.2f}% "
              f"(improved by {res['improvement']:.2f}%)")
    
    print(f"\n{'='*80}")
    print(f"🏆 GLOBAL BEST: {best_overall_score:.2f}%")
    print(f"{'='*80}")
    print("\nBest parameters:")
    for name, val in best_overall_params.items():
        print(f"  {name:20s} = {val:.6f}")
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / 'multistart_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'best_score': float(best_overall_score),
                'best_params': best_overall_params,
                'all_results': all_results,
                'n_starts': n_starts
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {results_file}")
    
    return best_overall_params, best_overall_score


def main():
    parser = argparse.ArgumentParser(description="Multi-start refinement")
    parser.add_argument('regions_file', type=str,
                       help='Path to stage2_regions.pkl')
    parser.add_argument('--n_starts', type=int, default=5,
                       help='Number of regions to refine')
    parser.add_argument('--max_iter', type=int, default=50,
                       help='CMA-ES iterations per region')
    parser.add_argument('--output_dir', type=str, default='results/multistart')
    
    args = parser.parse_args()
    
    regions_file = Path(args.regions_file)
    if not regions_file.exists():
        print(f"ERROR: {regions_file} not found")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    multi_start_refinement(regions_file, args.n_starts, args.max_iter, output_dir)


if __name__ == '__main__':
    main()
