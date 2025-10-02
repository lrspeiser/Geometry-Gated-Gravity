"""
quick_diagnostic.py

Quick diagnostic test: Can velocity dispersion gating theoretically close the cluster gap?

Tests different (e, α) parameter combinations to see if we can get enough amplification
without breaking the model (negative denominator).

Author: Henry Speiser
Date: October 2, 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from velocity_dispersion_model import fX_ratio_curv_sigma


def diagnostic_test():
    """
    Test if velocity dispersion gating can provide enough amplification.
    """
    print("=" * 80)
    print("DIAGNOSTIC TEST: Velocity Dispersion Gating Amplification")
    print("=" * 80)
    
    # O2 baseline parameters (fixed)
    a_fixed = 0.6687
    b_fixed = 0.1401
    d_fixed = 0.0871
    
    # Representative conditions
    # Cluster outskirts (where we need the most boost)
    x_cluster = 10.0  # Far from center
    Sigma_hat_cluster = -1.5  # Low surface density (outskirts)
    grad_ln_Sigma_cluster = 0.3  # Moderate gradient
    
    # Galaxy typical
    x_galaxy = 5.0
    Sigma_hat_galaxy = -0.5
    grad_ln_Sigma_galaxy = 0.5
    
    # Velocity dispersions from data
    sigma_cluster = 1132.0  # km/s (median of lensing clusters)
    sigma_galaxy = 120.0    # km/s (typical SPARC)
    
    print("\nTest Conditions:")
    print(f"  Cluster: x={x_cluster}, Σ̂={Sigma_hat_cluster}, |∇ln Σ|={grad_ln_Sigma_cluster}")
    print(f"           σ = {sigma_cluster:.0f} km/s")
    print(f"  Galaxy:  x={x_galaxy}, Σ̂={Sigma_hat_galaxy}, |∇ln Σ|={grad_ln_Sigma_galaxy}")
    print(f"           σ = {sigma_galaxy:.0f} km/s")
    
    # Baseline (no velocity dispersion gating)
    params_baseline = (a_fixed, b_fixed, d_fixed, 0.0, 0.0)
    fX_baseline_cluster = fX_ratio_curv_sigma(
        params_baseline, x_cluster, Sigma_hat_cluster, 
        grad_ln_Sigma_cluster, sigma_cluster
    )
    fX_baseline_galaxy = fX_ratio_curv_sigma(
        params_baseline, x_galaxy, Sigma_hat_galaxy,
        grad_ln_Sigma_galaxy, sigma_galaxy
    )
    
    print(f"\nBaseline O2 (no σ-gating):")
    print(f"  fX_cluster = {fX_baseline_cluster:.3f}")
    print(f"  fX_galaxy  = {fX_baseline_galaxy:.3f}")
    
    # Test grid of (e, α) parameters
    print("\n" + "=" * 80)
    print("Testing (e, α) Parameter Space")
    print("=" * 80)
    
    e_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
    alpha_values = [1.0, 1.2, 1.5, 1.8, 2.0]
    
    results = []
    
    for e in e_values:
        for alpha in alpha_values:
            params_sigma = (a_fixed, b_fixed, d_fixed, e, alpha)
            
            # Compute fX for cluster
            fX_cluster = fX_ratio_curv_sigma(
                params_sigma, x_cluster, Sigma_hat_cluster,
                grad_ln_Sigma_cluster, sigma_cluster
            )
            
            # Compute fX for galaxy
            fX_galaxy = fX_ratio_curv_sigma(
                params_sigma, x_galaxy, Sigma_hat_galaxy,
                grad_ln_Sigma_galaxy, sigma_galaxy
            )
            
            # Amplification factors
            cluster_amplification = fX_cluster / fX_baseline_cluster if fX_baseline_cluster > 0 else 0
            galaxy_amplification = fX_galaxy / fX_baseline_galaxy if fX_baseline_galaxy > 0 else 0
            
            # Check if denominator is positive (stability)
            # Cluster case (most critical)
            sigma_term_cluster = e * (sigma_cluster / 100.0) ** alpha
            denom_cluster = a_fixed - b_fixed * Sigma_hat_cluster - d_fixed * abs(grad_ln_Sigma_cluster) - sigma_term_cluster
            stable_cluster = denom_cluster > 0.01  # Keep some margin
            
            # Galaxy case
            sigma_term_galaxy = e * (sigma_galaxy / 100.0) ** alpha
            denom_galaxy = a_fixed - b_fixed * Sigma_hat_galaxy - d_fixed * abs(grad_ln_Sigma_galaxy) - sigma_term_galaxy
            stable_galaxy = denom_galaxy > 0.01
            
            stable = stable_cluster and stable_galaxy
            
            # Estimate galaxy impact (rough proxy for APE change)
            galaxy_impact_pct = (galaxy_amplification - 1.0) * 100
            
            result = {
                'e': e,
                'alpha': alpha,
                'fX_cluster': fX_cluster,
                'fX_galaxy': fX_galaxy,
                'cluster_boost': cluster_amplification,
                'galaxy_boost': galaxy_amplification,
                'galaxy_impact_pct': galaxy_impact_pct,
                'stable': stable,
                'denom_cluster': denom_cluster,
                'denom_galaxy': denom_galaxy
            }
            
            results.append(result)
    
    df = pd.DataFrame(results)
    
    # Find best candidates
    # Criteria:
    # 1. Stable (denominator > 0)
    # 2. Maximize cluster boost
    # 3. Minimize galaxy impact (< 30% change ideally)
    
    df_stable = df[df['stable'] == True]
    
    if len(df_stable) == 0:
        print("\n❌ NO STABLE SOLUTIONS FOUND!")
        print("   All (e, α) combinations make denominator negative.")
        print("\n🚨 CONSTRAINT IDENTIFIED:")
        print("   Current limits: Need denominator > 0")
        print("   This constraint may be too restrictive!")
        return
    
    # Sort by cluster boost (descending)
    df_sorted = df_stable.sort_values('cluster_boost', ascending=False)
    
    print("\n" + "-" * 80)
    print("Top 10 Stable Configurations (by cluster boost):")
    print("-" * 80)
    print(f"{'e':>6} {'α':>6} {'Cluster':>10} {'Galaxy':>10} {'Gal Impact':>12} {'Stable':>8}")
    print(f"{'':>6} {'':>6} {'Boost':>10} {'Boost':>10} {'(%)':>12} {'?':>8}")
    print("-" * 80)
    
    for idx, row in df_sorted.head(10).iterrows():
        print(f"{row['e']:6.2f} {row['alpha']:6.2f} {row['cluster_boost']:10.2f}× "
              f"{row['galaxy_boost']:10.3f}× {row['galaxy_impact_pct']:11.1f}% "
              f"{'✅' if row['stable'] else '❌':>8}")
    
    # Find best compromise
    # Accept galaxy impact up to 30% (rough proxy for APE degradation)
    df_acceptable = df_sorted[df_sorted['galaxy_impact_pct'] < 30]
    
    if len(df_acceptable) == 0:
        print("\n⚠️ NO SOLUTIONS WITH GALAXY IMPACT < 30%")
        print("   All configurations significantly affect galaxies")
        best = df_sorted.iloc[0]
    else:
        best = df_acceptable.iloc[0]
    
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION:")
    print("=" * 80)
    print(f"  e = {best['e']:.3f}")
    print(f"  α = {best['alpha']:.2f}")
    print(f"\n  Cluster amplification: {best['cluster_boost']:.1f}×")
    print(f"  Galaxy amplification:  {best['galaxy_boost']:.3f}× ({best['galaxy_impact_pct']:.1f}% change)")
    print(f"\n  Stability:")
    print(f"    Cluster denominator: {best['denom_cluster']:.4f} {'✅' if best['denom_cluster'] > 0 else '❌'}")
    print(f"    Galaxy denominator:  {best['denom_galaxy']:.4f} {'✅' if best['denom_galaxy'] > 0 else '❌'}")
    
    # Estimate if this is enough
    print("\n" + "=" * 80)
    print("ASSESSMENT:")
    print("=" * 80)
    
    # Current O2 underpredicts cluster lensing by 40-140×
    # Need at least 40× boost to close gap for best case
    needed_min = 40.0
    needed_max = 140.0
    
    achieved = best['cluster_boost']
    
    print(f"\n  Current O2 underprediction: 40-140× too small")
    print(f"  This model achieves:        {achieved:.1f}× amplification")
    print(f"\n  Gap closure:")
    print(f"    Best case (40× needed):  {achieved/needed_min*100:.0f}%")
    print(f"    Worst case (140× needed): {achieved/needed_max*100:.0f}%")
    
    if achieved >= needed_min:
        print("\n  ✅ SUCCESS! Can potentially close the gap!")
        print("     Continue with full fitting and validation.")
    elif achieved >= needed_min * 0.5:
        print("\n  ⚠️ PARTIAL! Gets within factor of 2-3.")
        print("     This is significant progress but may not fully close gap.")
        print("     Worth testing, but may need two-regime model.")
    else:
        print("\n  ❌ INSUFFICIENT! Falls short by large factor.")
        print("     Velocity dispersion gating alone cannot solve cluster problem.")
        print("     Recommendation: Move to Test 2 (Hot Gas Fraction)")
    
    # Check for artificial constraints
    print("\n" + "=" * 80)
    print("CONSTRAINT ANALYSIS:")
    print("=" * 80)
    
    # Test what happens if we violate stability constraint
    print("\nTesting WITHOUT stability constraint (denominator can go negative):")
    
    e_aggressive = 0.20
    alpha_aggressive = 2.0
    params_aggressive = (a_fixed, b_fixed, d_fixed, e_aggressive, alpha_aggressive)
    
    # Compute amplification (even if unstable)
    sigma_term_cluster_agg = e_aggressive * (sigma_cluster / 100.0) ** alpha_aggressive
    denom_cluster_agg = a_fixed - b_fixed * Sigma_hat_cluster - d_fixed * abs(grad_ln_Sigma_cluster) - sigma_term_cluster_agg
    
    if denom_cluster_agg < 0:
        # Model breaks down, but what would amplification be if we allowed it?
        # This tells us if stability constraint is the bottleneck
        print(f"\n  With e={e_aggressive}, α={alpha_aggressive}:")
        print(f"    Cluster denominator: {denom_cluster_agg:.4f} ❌ (NEGATIVE)")
        print(f"    Model becomes unstable/unphysical")
        print(f"\n  🚨 BOTTLENECK IDENTIFIED:")
        print(f"     Stability constraint (denom > 0) is the LIMITING FACTOR")
        print(f"     To get more amplification, we'd need:")
        print(f"     1. Different functional form (not 1/denom)")
        print(f"     2. Modify baseline O2 parameters (a, b, d)")
        print(f"     3. Add multiplicative term instead of subtractive")
    else:
        fX_cluster_agg = fX_ratio_curv_sigma(
            params_aggressive, x_cluster, Sigma_hat_cluster,
            grad_ln_Sigma_cluster, sigma_cluster
        )
        amplification_agg = fX_cluster_agg / fX_baseline_cluster
        print(f"\n  With e={e_aggressive}, α={alpha_aggressive}:")
        print(f"    Cluster amplification: {amplification_agg:.1f}×")
        print(f"    Still stable! Can push further.")
    
    print("\n" + "=" * 80)
    
    # Save results
    output_file = Path(__file__).parent / "diagnostic_results.csv"
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n✅ Full results saved to: diagnostic_results.csv")
    
    return best


if __name__ == "__main__":
    best_config = diagnostic_test()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\nBased on diagnostic results:")
    print("  1. If SUCCESS (≥40× boost): Proceed with full fitting")
    print("  2. If PARTIAL (20-40× boost): Proceed but expect two-regime model")
    print("  3. If FAIL (<20× boost): Document and move to Test 2")
    print("\n" + "=" * 80)
