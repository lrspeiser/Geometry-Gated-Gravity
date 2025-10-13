"""
Full 200-Iteration RAR Optimization with Power Law Coherence

After quick test success (RAR 0.088 dex in 20 iter), run full optimization
to achieve best possible performance.

Target: RAR scatter < 0.08 dex (better than MOND literature 0.13 dex)
"""

import sys
sys.path.insert(0, 'C:/Users/henry/dev/GravityCalculator/many_path_model')

from optimize_rar_kernel import *
import matplotlib.pyplot as plt

def main():
    print("="*80)
    print("FULL 200-ITERATION RAR OPTIMIZATION (Power Law Coherence)")
    print("="*80)
    print("\nQuick test result: RAR scatter 0.088 dex (✅ PASSED 0.15 target!)")
    print("Goal: Fine-tune to < 0.08 dex (beat MOND's 0.13 dex)\n")
    
    # Load data
    output_dir = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results")
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    
    # Train/test split
    train_df, test_df = suite.perform_train_test_split()
    
    # Full optimization (200 iterations)
    print("Starting 200-iteration optimization...")
    print("Estimated time: ~3-4 minutes\n")
    
    best_hp, result = run_optimization(train_df, n_iter=200)
    
    print("\n" + "="*80)
    print("FINAL OPTIMIZED HYPERPARAMETERS")
    print("="*80)
    for key, value in best_hp.to_dict().items():
        print(f"  {key:<15} = {value:.6f}")
    print("="*80)
    
    # Validation
    train_metrics, test_metrics = validate_on_holdout(best_hp, train_df, test_df)
    
    # Newtonian limit check
    print("\n" + "="*80)
    print("PHYSICS VALIDATION")
    print("="*80)
    kernel = PathSpectrumKernel(best_hp, use_cupy=False)
    r_small = np.array([0.001, 0.01, 0.1])
    v_small = np.array([50, 100, 150])
    K_small = kernel.many_path_boost_factor(r_small, v_small)
    
    print("\nNewtonian limit:")
    for i in range(len(r_small)):
        print(f"  r = {r_small[i]:6.3f} kpc: K = {K_small[i]:.6f} (boost = {K_small[i]*100:.3f}%)")
    
    if np.all(K_small < 0.01):
        print("✅ Newtonian limit preserved (K < 1% at small r)")
    else:
        print("⚠️  WARNING: Check Newtonian limit")
    
    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print("\nRAR Performance:")
    print(f"  Train scatter: {train_metrics['rar_scatter']:.3f} dex")
    print(f"  Test scatter:  {test_metrics['rar_scatter']:.3f} dex")
    print(f"  Test bias:     {test_metrics['rar_bias']:.3f} dex")
    
    if test_metrics['rar_scatter'] <= 0.08:
        print(f"\n  🎉 EXCELLENT! Better than MOND (0.13 dex)")
    elif test_metrics['rar_scatter'] <= 0.13:
        print(f"\n  ✅ GOOD! Competitive with MOND")
    elif test_metrics['rar_scatter'] <= 0.15:
        print(f"\n  ✅ TARGET MET! Within literature standards")
    
    print("\nRotation Curve Performance:")
    print(f"  Train APE:     {train_metrics['median_ape']:.1f}%")
    print(f"  Test APE:      {test_metrics['median_ape']:.1f}%")
    
    if test_metrics['median_ape'] <= 10:
        print(f"  ✅ EXCELLENT!")
    elif test_metrics['median_ape'] <= 20:
        print(f"  ⚠️  GOOD but room for improvement")
    else:
        print(f"  ⚠️  Needs work (universal law vs per-galaxy fit tradeoff)")
    
    # Comparison table
    print("\n" + "="*80)
    print("PROGRESS COMPARISON")
    print("="*80)
    print("\n| Stage | RAR Scatter | RAR Bias | Median APE | Status |")
    print("|-------|-------------|----------|------------|--------|")
    print("| Initial (Old Kernel) | 0.256 dex | -0.33 dex | ~23% | Baseline |")
    print("| + p Exponent (Exp) | 0.221 dex | -0.22 dex | 23.3% | +14% |")
    print("| + Power Law (20 iter) | 0.088 dex | -0.08 dex | 19.0% | +60% ✅ |")
    print(f"| + Full Optim (200 iter) | {test_metrics['rar_scatter']:.3f} dex | {test_metrics['rar_bias']:.3f} dex | {test_metrics['median_ape']:.1f}% | FINAL |")
    
    # Save results
    results_path = output_dir / "final_optimization_200iter_results.json"
    import json
    results_dict = {
        'optimization': '200_iterations_power_law_coherence',
        'hyperparameters': best_hp.to_dict(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'baseline_comparison': {
            'initial_rar': 0.256,
            'p_exponent_rar': 0.221,
            'quick_test_rar': 0.088,
            'final_rar': test_metrics['rar_scatter']
        },
        'optimization_info': {
            'n_iter': 200,
            'success': result.success,
            'message': result.message,
            'final_loss': float(result.fun)
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Final results saved to {results_path}")
    
    # Next steps recommendation
    print("\n" + "="*80)
    print("RECOMMENDED NEXT STEPS")
    print("="*80)
    
    if test_metrics['rar_scatter'] < 0.10 and test_metrics['median_ape'] < 20:
        print("\n1. ✅ RAR performance excellent - move to external validation")
        print("2. 📊 Test on non-SPARC galaxies")
        print("3. 🔭 Cluster lensing mass maps")
        print("4. 🌌 Milky Way vertical structure (Gaia)")
        print("5. 📝 Prepare publication draft")
    else:
        print("\n1. Investigate per-galaxy RAR scatter contributors")
        print("2. Consider galaxy-specific corrections (selective gates)")
        print("3. Balance RAR vs rotation curve performance")
        print("4. Explore alternative loss functions")

if __name__ == "__main__":
    main()
