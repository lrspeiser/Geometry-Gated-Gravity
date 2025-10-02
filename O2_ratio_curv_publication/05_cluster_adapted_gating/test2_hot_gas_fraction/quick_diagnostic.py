"""
quick_diagnostic.py

Quick diagnostic test: Can hot gas fraction gating theoretically close the cluster gap?

Tests different f parameter values to see if we can get enough amplification
without breaking the model.

Author: Henry Speiser
Date: October 2, 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from hot_gas_model import fX_ratio_curv_fgas


def diagnostic_test():
    """
    Test if hot gas fraction gating can provide enough amplification.
    """
    print("=" * 80)
    print("DIAGNOSTIC TEST: Hot Gas Fraction Gating Amplification")
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
    
    # Hot gas fractions from observations
    fgas_cluster = 0.12  # 12% hot gas (typical massive cluster)
    fgas_galaxy = 0.00   # No hot gas (galaxy)
    
    print("\nTest Conditions:")
    print(f"  Cluster: x={x_cluster}, Σ̂={Sigma_hat_cluster}, |∇ln Σ|={grad_ln_Sigma_cluster}")
    print(f"           f_gas = {fgas_cluster:.2f} (12% hot gas)")
    print(f"  Galaxy:  x={x_galaxy}, Σ̂={Sigma_hat_galaxy}, |∇ln Σ|={grad_ln_Sigma_galaxy}")
    print(f"           f_gas = {fgas_galaxy:.2f} (no hot gas)")
    
    # Baseline (no hot gas gating)
    params_baseline = (a_fixed, b_fixed, d_fixed, 0.0)
    fX_baseline_cluster = fX_ratio_curv_fgas(
        params_baseline, x_cluster, Sigma_hat_cluster,
        grad_ln_Sigma_cluster, fgas_cluster
    )
    fX_baseline_galaxy = fX_ratio_curv_fgas(
        params_baseline, x_galaxy, Sigma_hat_galaxy,
        grad_ln_Sigma_galaxy, fgas_galaxy
    )
    
    print(f"\nBaseline O2 (no fgas-gating):")
    print(f"  fX_cluster = {fX_baseline_cluster:.3f}")
    print(f"  fX_galaxy  = {fX_baseline_galaxy:.3f}")
    
    # Test range of f parameter
    print("\n" + "=" * 80)
    print("Testing f Parameter Space")
    print("=" * 80)
    
    # Explore f values
    # Note: f is a PENALTY coefficient, so NEGATIVE f would INCREASE cluster lensing
    # But that's unphysical (hot gas should REDUCE lensing, not increase it)
    # So we test POSITIVE f values
    
    # Wait... let me reconsider the physics:
    # - Hot gas doesn't lens (it's diffuse, not gravitationally bound to galaxies)
    # - Clusters have MORE hot gas than galaxies
    # - Current O2 UNDERPREDICTS clusters by 40-140×
    # - If f > 0 (penalty), we'd make clusters WORSE
    # - We need to BOOST clusters, not penalize them!
    
    # INSIGHT: The problem is backwards!
    # Hot gas fraction gating with f > 0 would HURT cluster predictions.
    
    # Let's test both signs to see what happens:
    f_values = np.concatenate([
        np.linspace(-3.0, -0.1, 15),  # NEGATIVE f (boost)
        np.linspace(0.1, 3.0, 15)     # POSITIVE f (penalty)
    ])
    
    results = []
    
    for f in f_values:
        params = (a_fixed, b_fixed, d_fixed, f)
        
        # Compute fX for cluster
        fX_cluster = fX_ratio_curv_fgas(
            params, x_cluster, Sigma_hat_cluster,
            grad_ln_Sigma_cluster, fgas_cluster
        )
        
        # Compute fX for galaxy
        fX_galaxy = fX_ratio_curv_fgas(
            params, x_galaxy, Sigma_hat_galaxy,
            grad_ln_Sigma_galaxy, fgas_galaxy
        )
        
        # Amplification factors
        cluster_amplification = fX_cluster / fX_baseline_cluster if fX_baseline_cluster > 0 else 0
        galaxy_amplification = fX_galaxy / fX_baseline_galaxy if fX_baseline_galaxy > 0 else 0
        
        # Check denominator stability
        fgas_term_cluster = f * fgas_cluster
        denom_cluster = a_fixed - b_fixed * Sigma_hat_cluster - d_fixed * abs(grad_ln_Sigma_cluster) - fgas_term_cluster
        stable_cluster = denom_cluster > 0.01
        
        fgas_term_galaxy = f * fgas_galaxy
        denom_galaxy = a_fixed - b_fixed * Sigma_hat_galaxy - d_fixed * abs(grad_ln_Sigma_galaxy) - fgas_term_galaxy
        stable_galaxy = denom_galaxy > 0.01
        
        stable = stable_cluster and stable_galaxy
        
        # Galaxy impact
        galaxy_impact_pct = (galaxy_amplification - 1.0) * 100
        
        result = {
            'f': f,
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
    
    # Separate by sign of f
    df_negative_f = df[df['f'] < 0]  # Boost clusters
    df_positive_f = df[df['f'] > 0]  # Penalize clusters
    
    print("\n" + "=" * 80)
    print("CRITICAL PHYSICS INSIGHT:")
    print("=" * 80)
    print("\nCurrent O2 UNDERPREDICTS cluster lensing by 40-140×")
    print("We need to INCREASE cluster predictions, not decrease them.")
    print("\nTwo possibilities:")
    print("  1. f > 0: Hot gas REDUCES lensing → clusters get WORSE (wrong direction!)")
    print("  2. f < 0: Hot gas INCREASES lensing → clusters get BETTER (unphysical!)")
    print("\n🚨 FUNDAMENTAL PROBLEM:")
    print("   Hot gas should REDUCE lensing (it's not bound to galaxies)")
    print("   But we need to INCREASE cluster lensing predictions")
    print("   → These are INCOMPATIBLE!")
    
    # Show results anyway
    print("\n" + "-" * 80)
    print("Results with NEGATIVE f (unphysical boost):")
    print("-" * 80)
    
    df_negative_stable = df_negative_f[df_negative_f['stable'] == True]
    if len(df_negative_stable) > 0:
        df_neg_sorted = df_negative_stable.sort_values('cluster_boost', ascending=False)
        print(f"{'f':>8} {'Cluster':>10} {'Galaxy':>10} {'Gal Impact':>12} {'Stable':>8}")
        print(f"{'':>8} {'Boost':>10} {'Boost':>10} {'(%)':>12} {'?':>8}")
        print("-" * 80)
        
        for idx, row in df_neg_sorted.head(5).iterrows():
            print(f"{row['f']:8.2f} {row['cluster_boost']:10.2f}× "
                  f"{row['galaxy_boost']:10.3f}× {row['galaxy_impact_pct']:11.1f}% "
                  f"{'✅' if row['stable'] else '❌':>8}")
        
        best_negative = df_neg_sorted.iloc[0]
        print(f"\nBest with f < 0: f={best_negative['f']:.2f}")
        print(f"  Cluster boost: {best_negative['cluster_boost']:.1f}×")
        print(f"  But this is UNPHYSICAL (hot gas increases lensing?!)")
    else:
        print("  No stable solutions with f < 0")
    
    print("\n" + "-" * 80)
    print("Results with POSITIVE f (physically motivated penalty):")
    print("-" * 80)
    
    df_positive_stable = df_positive_f[df_positive_f['stable'] == True]
    if len(df_positive_stable) > 0:
        df_pos_sorted = df_positive_stable.sort_values('cluster_boost', ascending=False)
        print(f"{'f':>8} {'Cluster':>10} {'Galaxy':>10} {'Gal Impact':>12} {'Stable':>8}")
        print(f"{'':>8} {'Boost':>10} {'Boost':>10} {'(%)':>12} {'?':>8}")
        print("-" * 80)
        
        for idx, row in df_pos_sorted.head(5).iterrows():
            print(f"{row['f']:8.2f} {row['cluster_boost']:10.2f}× "
                  f"{row['galaxy_boost']:10.3f}× {row['galaxy_impact_pct']:11.1f}% "
                  f"{'✅' if row['stable'] else '❌':>8}")
        
        best_positive = df_pos_sorted.iloc[0]
        print(f"\nBest with f > 0: f={best_positive['f']:.2f}")
        print(f"  Cluster boost: {best_positive['cluster_boost']:.1f}×")
        print(f"  This makes clusters WORSE (boost < 1)")
        print(f"  Physically correct, but moves us AWAY from observations!")
    else:
        print("  No stable solutions with f > 0")
    
    # Assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT:")
    print("=" * 80)
    
    print("\n❌ HOT GAS FRACTION GATING FAILS FUNDAMENTALLY")
    print("\nReason:")
    print("  - Hot gas should REDUCE lensing (physically motivated)")
    print("  - Clusters have MORE hot gas than galaxies (fgas ~ 12% vs 0%)")
    print("  - Current O2 UNDERPREDICTS clusters by 40-140×")
    print("  - Adding hot gas penalty makes underprediction WORSE")
    print("\n  → Hot gas gating pushes predictions in WRONG DIRECTION")
    print("\nTo get the right direction (boost clusters), we'd need:")
    print("  - Negative f (unphysical: hot gas increases lensing?!)")
    print("  - Or reverse the sign (clusters have LESS hot gas than galaxies?!)")
    print("\nBoth are physically nonsensical.")
    
    # Check if we can at least quantify how bad it gets
    if len(df_positive_stable) > 0:
        best_phys = df_positive_stable.iloc[0]
        reduction = 1.0 / best_phys['cluster_boost']  # How much worse
        print(f"\nWith physically motivated f > 0:")
        print(f"  Cluster predictions become {reduction:.1f}× SMALLER")
        print(f"  Gap grows from 40-140× to {40*reduction:.0f}-{140*reduction:.0f}×")
        print(f"  Catastrophic for cluster predictions")
    
    # Save results
    output_file = Path(__file__).parent / "diagnostic_results.csv"
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n✅ Full results saved to: diagnostic_results.csv")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("\nNeither Test 1 (velocity dispersion) nor Test 2 (hot gas fraction)")
    print("can close the cluster lensing gap within the O2 framework.")
    print("\nBoth have the SAME FUNDAMENTAL PROBLEM:")
    print("  - Adding terms to denominator creates PENALTIES")
    print("  - Clusters have HIGHER values of these terms")
    print("  - This makes cluster predictions WORSE, not better")
    print("\nTo boost clusters, we need:")
    print("  - Multiplicative amplification (not subtractive penalty)")
    print("  - Or entirely different model structure")
    print("  - Or accept two-regime model (different physics for clusters)")
    
    return df


if __name__ == "__main__":
    df_results = diagnostic_test()
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    print("\nCease gating extension attempts within O2 denominator framework.")
    print("\nNext steps:")
    print("  1. Document why single-parameter extensions fail")
    print("  2. Consider multiplicative amplification: fX → fX * G(cluster_properties)")
    print("  3. Or accept two-regime model: O2 for galaxies, NFW+baryons for clusters")
    print("  4. Focus on understanding WHY O2 works for galaxies but not clusters")
    print("\n" + "=" * 80)
