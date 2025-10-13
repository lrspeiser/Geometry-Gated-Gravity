"""
Quick Test: Power Law Coherence (20 iterations)

Tests whether the power law coherence damping improves RAR scatter
compared to exponential. If successful, run full 200-iteration optimization.
"""

import sys
sys.path.insert(0, 'C:/Users/henry/dev/GravityCalculator/many_path_model')

from optimize_rar_kernel import *

def main():
    print("="*80)
    print("QUICK TEST: POWER LAW COHERENCE (20 iterations)")
    print("="*80)
    print("\nGoal: Verify improvement over exponential (RAR 0.221 dex)")
    print("Expected: RAR scatter ~0.16-0.19 dex\n")
    
    # Load data
    output_dir = Path("C:/Users/henry/dev/GravityCalculator/many_path_model/results")
    suite = ValidationSuite(output_dir, load_sparc=True)
    df = suite.sparc_data
    
    # Train/test split
    train_df, test_df = suite.perform_train_test_split()
    
    # Quick optimization (20 iterations)
    best_hp, result = run_optimization(train_df, n_iter=20)
    
    print("\n" + "="*80)
    print("QUICK TEST RESULTS")
    print("="*80)
    
    # Hyperparameters
    print("\nOptimal hyperparameters (20 iter):")
    for key, value in best_hp.to_dict().items():
        print(f"  {key:<15} = {value:.6f}")
    
    # Validation
    train_metrics, test_metrics = validate_on_holdout(best_hp, train_df, test_df)
    
    # Decision
    print("\n" + "="*80)
    print("DECISION")
    print("="*80)
    
    improvement = 0.221 - test_metrics['rar_scatter']
    improvement_pct = (improvement / 0.221) * 100
    
    if test_metrics['rar_scatter'] < 0.19:
        print(f"✅ SUCCESS! RAR scatter = {test_metrics['rar_scatter']:.3f} dex")
        print(f"   Improvement: {improvement:.3f} dex ({improvement_pct:.1f}%)")
        print(f"   Recommendation: Proceed with 200-iteration full optimization")
    elif test_metrics['rar_scatter'] < 0.21:
        print(f"⚠️  MARGINAL: RAR scatter = {test_metrics['rar_scatter']:.3f} dex")
        print(f"   Improvement: {improvement:.3f} dex ({improvement_pct:.1f}%)")
        print(f"   Recommendation: Try 100 iterations to explore further")
    else:
        print(f"❌ NO IMPROVEMENT: RAR scatter = {test_metrics['rar_scatter']:.3f} dex")
        print(f"   Change: {improvement:.3f} dex ({improvement_pct:.1f}%)")
        print(f"   Recommendation: Revisit Priority 2 (K_max parameter)")
    
    # Save quick test results
    results_path = output_dir / "quick_test_power_law_results.json"
    import json
    results_dict = {
        'test_type': '20_iteration_quick_test',
        'hyperparameters': best_hp.to_dict(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'baseline_rar_scatter': 0.221,
        'improvement_dex': float(improvement),
        'improvement_percent': float(improvement_pct)
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Quick test results saved to {results_path}")

if __name__ == "__main__":
    main()
