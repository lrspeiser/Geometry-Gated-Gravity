#!/usr/bin/env python3
"""
Test the optimized G³ parameters
"""

import numpy as np
from g3_alternatives_test import G3ContinuumModel, G3ContinuumParams, test_on_sparc, test_on_milky_way
import json
from pathlib import Path

# Best parameters from optimization (5.7% SPARC error!)
OPTIMIZED_PARAMS = {
    'v0': 398.55,
    'rc0': 7.37,
    'gamma': 0.261,
    'beta': 0.937,
    'Sigma_star': 177.96,
    'alpha': 0.510,
    'kappa': 0.504,
    'p_in': 2.050,
    'p_out': 1.915,
    'lambda_p': 1.190,
    'g_star': 0.288,
    'alpha_g': 0.654,
    'kappa_g': 1.700,
    'g_sat': 3313.95,
    'n_sat': 3.275
}

def main():
    print("="*80)
    print("TESTING OPTIMIZED G³ PARAMETERS")
    print("="*80)
    print("\nOptimized parameters (5.7% SPARC training error):")
    for k, v in OPTIMIZED_PARAMS.items():
        print(f"  {k:12s} = {v:8.3f}")
    
    # Create model with optimized parameters
    params = G3ContinuumParams(**OPTIMIZED_PARAMS)
    model = G3ContinuumModel(params)
    
    # Test on SPARC
    print("\n" + "-"*40)
    print("Testing on SPARC (full dataset)...")
    sparc_results = test_on_sparc(model)
    if sparc_results:
        print(f"  Median error: {sparc_results['median_error']*100:.1f}%")
        print(f"  N galaxies: {sparc_results['n_galaxies']}")
        print(f"  Best: {sparc_results['best_galaxy']['name']} ({sparc_results['best_galaxy']['median_error']*100:.1f}%)")
        print(f"  Worst: {sparc_results['worst_galaxy']['name']} ({sparc_results['worst_galaxy']['median_error']*100:.1f}%)")
    
    # Test on MW
    print("\n" + "-"*40)
    print("Testing on Milky Way (144k stars)...")
    mw_results = test_on_milky_way(model)
    if mw_results:
        print(f"  Median error: {mw_results['median_error']*100:.1f}%")
        print(f"  Points <10% error: {mw_results['under_10pct']*100:.1f}%")
    
    # Test lensing predictions
    print("\n" + "-"*40)
    print("Lensing predictions:")
    
    # Simple lensing test at key scales
    scales = [
        ('Galaxy (100 kpc)', 100, 100),
        ('Group (500 kpc)', 500, 300),
        ('Cluster (1000 kpc)', 1000, 500)
    ]
    
    for name, r_kpc, sigma_0 in scales:
        sigma = sigma_0 * np.exp(-r_kpc / (r_kpc/3))
        # For single point test, use numpy
        r_arr = np.array([r_kpc])
        sigma_arr = np.array([sigma])
        
        try:
            import cupy as cp
            # Convert to GPU
            r_gpu = cp.asarray(r_arr)
            sigma_gpu = cp.asarray(sigma_arr)
            g_tail, _ = model.compute_tail_acceleration(r_gpu, 0, sigma_gpu)
            g_tail = cp.asnumpy(g_tail)[0]
        except:
            # CPU fallback
            g_tail, _ = model.compute_tail_acceleration(r_arr, 0, sigma_arr)
            g_tail = g_tail[0]
        
        G = 4.300917270e-6
        M_eff = g_tail * r_kpc**2 / G
        print(f"  {name}: M_eff = {M_eff:.2e} Msun")
    
    # Save results
    output_dir = Path('out/g3_alternatives')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        'optimized_params': OPTIMIZED_PARAMS,
        'sparc_results': sparc_results,
        'mw_results': mw_results,
        'note': 'Parameters optimized with lensing constraints'
    }
    
    with open(output_dir / 'optimized_test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✅ Optimization successful: 5.7% SPARC error during training")
    print(f"✅ Full SPARC test: {sparc_results['median_error']*100:.1f}% median error")
    if mw_results:
        print(f"✅ Milky Way test: {mw_results['median_error']*100:.1f}% median error")
    print("📊 Results saved to out/g3_alternatives/optimized_test_results.json")
    print("="*80)

if __name__ == '__main__':
    main()