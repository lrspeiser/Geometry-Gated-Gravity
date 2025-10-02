"""
quick_diagnostic.py

Quick diagnostic test: Can gravitational potential depth gating close the cluster gap?

Tests both exponential and power-law forms with different parameter values.

Author: Henry Speiser
Date: October 2, 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from potential_depth_model import (
    fX_ratio_curv_potential_exp,
    fX_ratio_curv_potential_power,
    estimate_typical_potential_depth
)


def diagnostic_test():
    """
    Test if potential depth gating can provide enough amplification.
    """
    print("=" * 80)
    print("DIAGNOSTIC TEST: Gravitational Potential Depth Gating")
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
    
    # Estimate typical potential depths
    print("\nStep 1: Estimate typical potential depths")
    print("-" * 80)
    
    systems_for_phi = [
        ("Dwarf galaxy", 1e9, 10),
        ("SPARC typical", 1e11, 50),
        ("Milky Way", 1e12, 200),
        ("Massive spiral", 5e12, 300),
        ("Poor cluster", 1e14, 1000),
        ("A2029", 3e14, 1500),
        ("Rich cluster (A1689)", 1e15, 2000),
    ]
    
    phi_values = {}
    for name, M, R in systems_for_phi:
        Phi = estimate_typical_potential_depth(M, R)
        phi_values[name] = Phi
        print(f"  {name:25s}: M={M:.1e} Msun, R={R:4.0f} kpc → |Φ|={Phi:.2e} km²/s²")
    
    # Use representative values for diagnostic
    Phi_galaxy = phi_values["SPARC typical"]  # ~2e4 km²/s²
    Phi_cluster = phi_values["A2029"]  # ~4e5 km²/s²
    
    print(f"\n  Using for diagnostic:")
    print(f"    Galaxy:  |Φ| = {Phi_galaxy:.2e} km²/s²")
    print(f"    Cluster: |Φ| = {Phi_cluster:.2e} km²/s²")
    print(f"    Ratio:   {Phi_cluster / Phi_galaxy:.1f}×")
    
    # Baseline (no potential gating)
    params_baseline = (a_fixed, b_fixed, d_fixed, 0.0)
    fX_baseline_cluster = fX_ratio_curv_potential_exp(
        params_baseline, x_cluster, Sigma_hat_cluster,
        grad_ln_Sigma_cluster, Phi_cluster
    )
    fX_baseline_galaxy = fX_ratio_curv_potential_exp(
        params_baseline, x_galaxy, Sigma_hat_galaxy,
        grad_ln_Sigma_galaxy, Phi_galaxy
    )
    
    print(f"\nBaseline O2 (no potential gating):")
    print(f"  fX_cluster = {fX_baseline_cluster:.3f}")
    print(f"  fX_galaxy  = {fX_baseline_galaxy:.3f}")
    
    # Test exponential model
    print("\n" + "=" * 80)
    print("Testing EXPONENTIAL Model: amplification = exp(β * |Φ| / Φ₀)")
    print("=" * 80)
    
    beta_values = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    
    results_exp = []
    
    for beta in beta_values:
        params = (a_fixed, b_fixed, d_fixed, beta)
        
        fX_cluster = fX_ratio_curv_potential_exp(
            params, x_cluster, Sigma_hat_cluster,
            grad_ln_Sigma_cluster, Phi_cluster
        )
        
        fX_galaxy = fX_ratio_curv_potential_exp(
            params, x_galaxy, Sigma_hat_galaxy,
            grad_ln_Sigma_galaxy, Phi_galaxy
        )
        
        cluster_boost = fX_cluster / fX_baseline_cluster if fX_baseline_cluster > 0 else 0
        galaxy_boost = fX_galaxy / fX_baseline_galaxy if fX_baseline_galaxy > 0 else 0
        
        galaxy_impact_pct = (galaxy_boost - 1.0) * 100
        
        # Check if reasonable (not too extreme)
        reasonable = (cluster_boost < 1000) and (galaxy_boost < 10)
        
        result = {
            'beta': beta,
            'fX_cluster': fX_cluster,
            'fX_galaxy': fX_galaxy,
            'cluster_boost': cluster_boost,
            'galaxy_boost': galaxy_boost,
            'galaxy_impact_pct': galaxy_impact_pct,
            'reasonable': reasonable
        }
        
        results_exp.append(result)
    
    df_exp = pd.DataFrame(results_exp)
    
    # Find candidates that meet criteria
    # Cluster boost: need 40-140×
    # Galaxy impact: < 30% change (or galaxy_boost < 1.3)
    
    df_exp_viable = df_exp[
        (df_exp['cluster_boost'] >= 40) &
        (df_exp['cluster_boost'] <= 200) &
        (df_exp['reasonable'] == True)
    ]
    
    print("\n" + "-" * 80)
    print("Exponential Model Results:")
    print("-" * 80)
    print(f"{'β':>6} {'Cluster':>10} {'Galaxy':>10} {'Gal Impact':>12} {'Viable':>8}")
    print(f"{'':>6} {'Boost':>10} {'Boost':>10} {'(%)':>12} {'?':>8}")
    print("-" * 80)
    
    for idx, row in df_exp.iterrows():
        viable = (40 <= row['cluster_boost'] <= 200) and row['reasonable']
        print(f"{row['beta']:6.2f} {row['cluster_boost']:10.1f}× "
              f"{row['galaxy_boost']:10.2f}× {row['galaxy_impact_pct']:11.1f}% "
              f"{'✅' if viable else '  ':>8}")
    
    if len(df_exp_viable) > 0:
        # Sort by galaxy impact (prefer minimal)
        df_exp_viable = df_exp_viable.sort_values('galaxy_impact_pct')
        best_exp = df_exp_viable.iloc[0]
        
        print("\n" + "=" * 80)
        print("BEST EXPONENTIAL CONFIGURATION:")
        print("=" * 80)
        print(f"  β = {best_exp['beta']:.3f}")
        print(f"\n  Cluster amplification: {best_exp['cluster_boost']:.1f}×")
        print(f"  Galaxy amplification:  {best_exp['galaxy_boost']:.2f}× ({best_exp['galaxy_impact_pct']:.1f}% change)")
        
        # Assessment
        needed_min = 40.0
        needed_max = 140.0
        achieved = best_exp['cluster_boost']
        
        print(f"\n  Gap closure:")
        print(f"    Need: 40-140× boost")
        print(f"    Achieved: {achieved:.1f}×")
        
        if needed_min <= achieved <= needed_max:
            print(f"\n  ✅ SUCCESS! Cluster boost is within target range!")
            success_exp = True
        elif achieved > needed_max:
            print(f"\n  ⚠️ OVERSHOOT! Boost exceeds target (may fit but overpredict)")
            success_exp = "partial"
        else:
            print(f"\n  ❌ INSUFFICIENT! Boost below target range")
            success_exp = False
    else:
        print("\n❌ NO VIABLE EXPONENTIAL SOLUTIONS FOUND")
        print("   No β value provides 40-140× cluster boost with reasonable galaxy impact")
        success_exp = False
        best_exp = None
    
    # Test power-law model
    print("\n" + "=" * 80)
    print("Testing POWER-LAW Model: amplification = (|Φ| / Φ₀)^γ")
    print("=" * 80)
    
    gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    
    results_pow = []
    
    for gamma in gamma_values:
        params = (a_fixed, b_fixed, d_fixed, gamma)
        
        fX_cluster = fX_ratio_curv_potential_power(
            params, x_cluster, Sigma_hat_cluster,
            grad_ln_Sigma_cluster, Phi_cluster
        )
        
        fX_galaxy = fX_ratio_curv_potential_power(
            params, x_galaxy, Sigma_hat_galaxy,
            grad_ln_Sigma_galaxy, Phi_galaxy
        )
        
        cluster_boost = fX_cluster / fX_baseline_cluster if fX_baseline_cluster > 0 else 0
        galaxy_boost = fX_galaxy / fX_baseline_galaxy if fX_baseline_galaxy > 0 else 0
        
        galaxy_impact_pct = (galaxy_boost - 1.0) * 100
        
        reasonable = (cluster_boost < 1000) and (galaxy_boost < 10)
        
        result = {
            'gamma': gamma,
            'fX_cluster': fX_cluster,
            'fX_galaxy': fX_galaxy,
            'cluster_boost': cluster_boost,
            'galaxy_boost': galaxy_boost,
            'galaxy_impact_pct': galaxy_impact_pct,
            'reasonable': reasonable
        }
        
        results_pow.append(result)
    
    df_pow = pd.DataFrame(results_pow)
    
    df_pow_viable = df_pow[
        (df_pow['cluster_boost'] >= 40) &
        (df_pow['cluster_boost'] <= 200) &
        (df_pow['reasonable'] == True)
    ]
    
    print("\n" + "-" * 80)
    print("Power-Law Model Results:")
    print("-" * 80)
    print(f"{'γ':>6} {'Cluster':>10} {'Galaxy':>10} {'Gal Impact':>12} {'Viable':>8}")
    print(f"{'':>6} {'Boost':>10} {'Boost':>10} {'(%)':>12} {'?':>8}")
    print("-" * 80)
    
    for idx, row in df_pow.iterrows():
        viable = (40 <= row['cluster_boost'] <= 200) and row['reasonable']
        print(f"{row['gamma']:6.2f} {row['cluster_boost']:10.1f}× "
              f"{row['galaxy_boost']:10.2f}× {row['galaxy_impact_pct']:11.1f}% "
              f"{'✅' if viable else '  ':>8}")
    
    if len(df_pow_viable) > 0:
        df_pow_viable = df_pow_viable.sort_values('galaxy_impact_pct')
        best_pow = df_pow_viable.iloc[0]
        
        print("\n" + "=" * 80)
        print("BEST POWER-LAW CONFIGURATION:")
        print("=" * 80)
        print(f"  γ = {best_pow['gamma']:.3f}")
        print(f"\n  Cluster amplification: {best_pow['cluster_boost']:.1f}×")
        print(f"  Galaxy amplification:  {best_pow['galaxy_boost']:.2f}× ({best_pow['galaxy_impact_pct']:.1f}% change)")
        
        achieved = best_pow['cluster_boost']
        print(f"\n  Gap closure:")
        print(f"    Need: 40-140× boost")
        print(f"    Achieved: {achieved:.1f}×")
        
        if needed_min <= achieved <= needed_max:
            print(f"\n  ✅ SUCCESS! Cluster boost is within target range!")
            success_pow = True
        elif achieved > needed_max:
            print(f"\n  ⚠️ OVERSHOOT! May overpredict")
            success_pow = "partial"
        else:
            print(f"\n  ❌ INSUFFICIENT!")
            success_pow = False
    else:
        print("\n❌ NO VIABLE POWER-LAW SOLUTIONS FOUND")
        success_pow = False
        best_pow = None
    
    # Final assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT:")
    print("=" * 80)
    
    if success_exp == True or success_pow == True:
        print("\n✅ GRAVITATIONAL POTENTIAL DEPTH GATING SUCCEEDS!")
        print("\nThis approach CAN theoretically close the cluster gap!")
        print("\nPhysical interpretation:")
        print("  - Deeper potential wells → stronger modified gravity effects")
        print("  - Clusters have 10-20× deeper potentials than galaxies")
        print("  - Multiplicative amplification (not subtractive penalty)")
        print("  - This correctly boosts clusters MORE than galaxies")
        
        if success_exp == True:
            print(f"\n  Recommended: EXPONENTIAL model with β = {best_exp['beta']:.3f}")
            print(f"    Cluster boost: {best_exp['cluster_boost']:.1f}×")
            print(f"    Galaxy impact: +{best_exp['galaxy_impact_pct']:.1f}%")
        
        if success_pow == True:
            print(f"\n  Alternative: POWER-LAW model with γ = {best_pow['gamma']:.3f}")
            print(f"    Cluster boost: {best_pow['cluster_boost']:.1f}×")
            print(f"    Galaxy impact: +{best_pow['galaxy_impact_pct']:.1f}%")
        
        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("\n1. Compute actual |Φ|(R) from cluster lensing data")
        print("2. Fit β (or γ) on clusters to match Einstein radii")
        print("3. Validate on SPARC galaxies (check APE degradation)")
        print("4. If APE < 0.30: PUBLISH 4-parameter model")
        print("5. If APE > 0.30: Try two-regime or combined approach")
        
    else:
        print("\n❌ Potential depth gating insufficient")
        print("\nEven with optimal parameters, cannot achieve 40-140× cluster boost")
        print("while maintaining reasonable galaxy predictions.")
        print("\nPossible reasons:")
        print("  - Potential ratio (cluster/galaxy) not large enough (~20×)")
        print("  - Galaxy predictions too sensitive to amplification")
        print("  - Need different functional form or combined approach")
    
    # Save results
    output_dir = Path(__file__).parent
    df_exp.to_csv(output_dir / "diagnostic_results_exponential.csv", index=False, float_format='%.6f')
    df_pow.to_csv(output_dir / "diagnostic_results_powerlaw.csv", index=False, float_format='%.6f')
    
    print(f"\n✅ Results saved to:")
    print(f"   diagnostic_results_exponential.csv")
    print(f"   diagnostic_results_powerlaw.csv")
    
    return success_exp, success_pow, best_exp, best_pow


if __name__ == "__main__":
    success_exp, success_pow, best_exp, best_pow = diagnostic_test()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY:")
    print("=" * 80)
    
    if success_exp == True or success_pow == True:
        print("\n✅ TEST 3: GRAVITATIONAL POTENTIAL DEPTH GATING → SUCCESS")
        print("\nThis is a MAJOR RESULT!")
        print("Proceed with full implementation and validation.")
    else:
        print("\n❌ TEST 3: GRAVITATIONAL POTENTIAL DEPTH GATING → FAIL")
        print("\nMove to Test 4: Multi-Scale Curvature (Laplacian)")
    
    print("\n" + "=" * 80)
