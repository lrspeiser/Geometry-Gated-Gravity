#!/usr/bin/env python3
"""
Validate cluster_lensing_catalog.csv for mass-scaled inference.

Checks:
1. Schema compliance
2. Physical plausibility (mass-radius, redshift-θ_E relations)
3. Sample composition (tier distribution)
4. Einstein radius vs mass scaling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
REPO = Path(__file__).parent.parent
CATALOG = REPO / 'data' / 'cluster_lensing_catalog.csv'
OUTDIR = REPO / 'output' / 'validation'
OUTDIR.mkdir(parents=True, exist_ok=True)

def validate_schema(df):
    """Check required columns and data types."""
    required = ['cluster_name', 'z_lens', 'z_source', 'theta_E_obs', 
                'sigma_theta_E', 'M500_1e14Msun', 'R500_Mpc', 'tier']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check ranges
    assert (df['z_lens'] > 0).all(), "Invalid z_lens"
    assert (df['z_source'] > df['z_lens']).all(), "z_source must be > z_lens"
    assert (df['theta_E_obs'] > 0).all(), "Invalid theta_E_obs"
    assert (df['sigma_theta_E'] > 0).all(), "Invalid sigma_theta_E"
    assert (df['M500_1e14Msun'] > 0).all(), "Invalid M500"
    assert (df['R500_Mpc'] > 0).all(), "Invalid R500"
    assert df['tier'].isin([1, 2, 3]).all(), "tier must be 1, 2, or 3"
    
    print("✓ Schema validation passed")
    return True

def validate_physics(df):
    """Check physical plausibility."""
    # M500-R500 relation: M = (4π/3) × 500 × ρ_crit × R^3
    # At z=0.2, ρ_crit ≈ 1.2e-29 g/cm³
    # Expected: M500 ~ 5e14 Msun × (R500/Mpc)³
    
    M_expected = 5.0 * (df['R500_Mpc'] ** 3)  # rough check
    M_ratio = df['M500_1e14Msun'] / M_expected
    
    outliers = (M_ratio < 0.5) | (M_ratio > 2.0)
    if outliers.any():
        print(f"⚠ Warning: {outliers.sum()} clusters have unusual M500-R500 relation:")
        print(df.loc[outliers, ['cluster_name', 'M500_1e14Msun', 'R500_Mpc']])
    
    # Einstein radius vs mass: θ_E ∝ √M at fixed z
    # Crude check: θ_E should be 20-60 arcsec for this mass range
    low_theta = df['theta_E_obs'] < 20
    high_theta = df['theta_E_obs'] > 60
    if low_theta.any() or high_theta.any():
        print(f"⚠ Warning: {(low_theta | high_theta).sum()} clusters with unusual θ_E")
    
    print("✓ Physics checks passed (with warnings above)")
    return True

def plot_sample_properties(df):
    """Generate diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Mass-radius relation
    ax = axes[0, 0]
    colors = {1: 'gold', 2: 'silver', 3: 'gray'}
    for tier in [1, 2, 3]:
        mask = df['tier'] == tier
        ax.scatter(df.loc[mask, 'R500_Mpc'], 
                   df.loc[mask, 'M500_1e14Msun'],
                   c=colors[tier], s=100, alpha=0.7, 
                   label=f'Tier {tier}', edgecolor='k', linewidth=1)
    
    # Expected M ∝ R³
    R_theory = np.linspace(1.2, 2.5, 50)
    M_theory = 5.0 * (R_theory ** 3)
    ax.plot(R_theory, M_theory, 'k--', alpha=0.5, label='M ∝ R³')
    
    ax.set_xlabel('R₅₀₀ [Mpc]', fontsize=12)
    ax.set_ylabel('M₅₀₀ [10¹⁴ M☉]', fontsize=12)
    ax.set_title('Mass-Radius Relation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Einstein radius vs mass
    ax = axes[0, 1]
    for tier in [1, 2, 3]:
        mask = df['tier'] == tier
        ax.errorbar(df.loc[mask, 'M500_1e14Msun'], 
                    df.loc[mask, 'theta_E_obs'],
                    yerr=df.loc[mask, 'sigma_theta_E'],
                    fmt='o', c=colors[tier], ms=8, alpha=0.7,
                    label=f'Tier {tier}', capsize=3)
    
    # Expected θ_E ∝ √M (crude)
    M_theory = np.linspace(4, 30, 50)
    theta_theory = 25 * np.sqrt(M_theory / 10)  # normalized at M=10
    ax.plot(M_theory, theta_theory, 'k--', alpha=0.5, label='θ_E ∝ √M')
    
    ax.set_xlabel('M₅₀₀ [10¹⁴ M☉]', fontsize=12)
    ax.set_ylabel('θ_E [arcsec]', fontsize=12)
    ax.set_title('Einstein Radius vs Mass', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Redshift distribution
    ax = axes[1, 0]
    ax.hist(df['z_lens'], bins=15, color='steelblue', alpha=0.7, edgecolor='k')
    ax.set_xlabel('z_lens', fontsize=12)
    ax.set_ylabel('N clusters', fontsize=12)
    ax.set_title('Redshift Distribution', fontsize=14, fontweight='bold')
    ax.axvline(df['z_lens'].median(), color='red', linestyle='--', 
               label=f'Median: {df["z_lens"].median():.3f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Tier composition
    ax = axes[1, 1]
    tier_counts = df['tier'].value_counts().sort_index()
    bars = ax.bar(tier_counts.index, tier_counts.values, 
                  color=[colors[t] for t in tier_counts.index],
                  edgecolor='k', linewidth=2, alpha=0.8)
    ax.set_xlabel('Tier', fontsize=12)
    ax.set_ylabel('N clusters', fontsize=12)
    ax.set_title('Sample Composition', fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['1 (Gold)', '2 (Silver)', '3 (Complex)'])
    
    # Add counts on bars
    for bar, count in zip(bars, tier_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    outfile = OUTDIR / 'catalog_validation.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"✓ Saved diagnostic plots to {outfile}")
    plt.close()

def print_summary(df):
    """Print catalog summary statistics."""
    print("\n" + "="*60)
    print("CLUSTER LENSING CATALOG SUMMARY")
    print("="*60)
    print(f"Total clusters: {len(df)}")
    print(f"\nTier composition:")
    for tier in [1, 2, 3]:
        count = (df['tier'] == tier).sum()
        print(f"  Tier {tier}: {count} clusters")
    
    # Analysis sample (tier 1+2, exclude MACS0717)
    analysis = df[(df['tier'].isin([1, 2])) & 
                  (df['cluster_name'] != 'MACSJ0717.5+3745')]
    print(f"\nAnalysis sample (tiers 1+2, exclude MACS0717): {len(analysis)} clusters")
    
    print(f"\nMass range:")
    print(f"  M₅₀₀: {df['M500_1e14Msun'].min():.1f} - {df['M500_1e14Msun'].max():.1f} × 10¹⁴ M☉")
    print(f"  R₅₀₀: {df['R500_Mpc'].min():.2f} - {df['R500_Mpc'].max():.2f} Mpc")
    
    print(f"\nRedshift range: {df['z_lens'].min():.3f} - {df['z_lens'].max():.3f}")
    print(f"  Median: {df['z_lens'].median():.3f}")
    
    print(f"\nEinstein radius:")
    print(f"  Range: {df['theta_E_obs'].min():.1f} - {df['theta_E_obs'].max():.1f} arcsec")
    print(f"  Median: {df['theta_E_obs'].median():.1f} arcsec")
    print(f"  Mean uncertainty: {df['sigma_theta_E'].mean():.1f} arcsec ({100*df['sigma_theta_E'].mean()/df['theta_E_obs'].mean():.1f}%)")
    
    print("="*60 + "\n")

def test_scaling_relation(df):
    """Test θ_E ∝ M^α scaling."""
    # Fit log(θ_E) vs log(M)
    log_M = np.log10(df['M500_1e14Msun'])
    log_theta = np.log10(df['theta_E_obs'])
    
    # Simple linear fit (ignoring z-dependence for now)
    coeffs = np.polyfit(log_M, log_theta, 1)
    alpha = coeffs[0]
    
    print(f"\nScaling relation check:")
    print(f"  θ_E ∝ M^{alpha:.2f}")
    print(f"  Expected: α ≈ 0.5 (standard lensing)")
    
    if 0.4 < alpha < 0.6:
        print("  ✓ Scaling consistent with expectations")
    else:
        print("  ⚠ Warning: Unusual scaling exponent")
    
    return alpha

def main():
    print("Validating cluster lensing catalog...")
    print(f"Catalog: {CATALOG}\n")
    
    # Load catalog
    df = pd.read_csv(CATALOG)
    
    # Run validation
    validate_schema(df)
    validate_physics(df)
    
    # Summary statistics
    print_summary(df)
    
    # Test scaling
    alpha = test_scaling_relation(df)
    
    # Generate plots
    plot_sample_properties(df)
    
    print("\n✓ Validation complete!")
    print(f"  - Catalog is ready for mass-scaled inference")
    print(f"  - N_analysis = {len(df[(df['tier'].isin([1,2])) & (df['cluster_name'] != 'MACSJ0717.5+3745')])} clusters")
    print(f"  - See plots: {OUTDIR / 'catalog_validation.png'}")

if __name__ == '__main__':
    main()
