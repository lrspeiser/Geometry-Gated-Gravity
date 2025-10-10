#!/usr/bin/env python3
"""
SIMPLE BASELINE: Stars at Same Radius, Different Mass
======================================================

Pure observational analysis - NO MODELS, NO FORMULAS

Question: Do stars of different masses at the same galactocentric radius
          travel at different speeds?

Output for each radius bin:
- Low-mass stars: median mass, median v_phi, count
- High-mass stars: median mass, median v_phi, count  
- Observed Δv = v_phi(low) - v_phi(high)
- % difference
- Statistical significance

This is BASELINE RESEARCH ONLY - we're just looking at the data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate_realistic_test_stars(n_stars=20000):
    """
    Generate test stars in Milky Way outer disk.
    
    This is SYNTHETIC for testing the pipeline.
    Replace with real Gaia data when available!
    """
    print("⚠️  Using SYNTHETIC data for testing!")
    print("   Replace with real Gaia 144k stars for actual science\n")
    
    np.random.seed(42)
    
    # Galactocentric radii in outer disk
    R = np.random.uniform(15.0, 25.0, n_stars)
    
    # Mass: bimodal (low-mass and solar-mass)
    mass_type = np.random.choice([0, 1], n_stars, p=[0.6, 0.4])
    mass = np.where(
        mass_type == 0,
        np.random.uniform(0.6, 0.9, n_stars),   # Low-mass
        np.random.uniform(0.9, 1.5, n_stars)    # Higher-mass
    )
    
    # Velocities: flat rotation + dispersion
    v_circ_expected = 220.0  # km/s
    sigma_v = 30.0  # km/s
    
    # **KEY**: Add a small mass-dependent offset for testing
    # In reality, this would come from the data itself
    # For now: low-mass stars ~2 km/s faster (to test if we detect it)
    mass_offset = -2.0 * (mass - 0.6) / 0.9  # -2 km/s at low mass, 0 at high mass
    
    v_phi = v_circ_expected + mass_offset + np.random.normal(0, sigma_v, n_stars)
    
    # Add azimuthal angle
    phi = np.random.uniform(0, 2*np.pi, n_stars)
    
    # Vertical position
    z = np.random.normal(0, 0.3, n_stars)
    
    # Filter for near-circular (quality cut)
    v_R = np.random.normal(0, sigma_v, n_stars)
    near_circular = np.abs(v_R) < 25.0
    
    df = pd.DataFrame({
        'R_kpc': R[near_circular],
        'phi_rad': phi[near_circular],
        'z_kpc': z[near_circular],
        'mass_Msun': mass[near_circular],
        'v_phi_km_s': v_phi[near_circular],
        'v_R_km_s': v_R[near_circular],
    })
    
    print(f"Generated {len(df)} stars passing quality cuts")
    print(f"  R range: [{df.R_kpc.min():.1f}, {df.R_kpc.max():.1f}] kpc")
    print(f"  Mass range: [{df.mass_Msun.min():.2f}, {df.mass_Msun.max():.2f}] M_sun")
    print(f"  v_phi range: [{df.v_phi_km_s.min():.1f}, {df.v_phi_km_s.max():.1f}] km/s\n")
    
    return df


def compare_velocities_at_same_radius(df: pd.DataFrame, R_bin_size=1.0) -> pd.DataFrame:
    """
    Compare velocities of low-mass vs high-mass stars at same radius.
    
    Pure observation - no model assumptions!
    """
    print(f"{'='*70}")
    print("COMPARING VELOCITIES AT SAME RADIUS")
    print(f"{'='*70}\n")
    
    # Bin by radius
    df['R_bin'] = (np.floor(df.R_kpc / R_bin_size) * R_bin_size).astype(float)
    
    results = []
    
    for R_bin in sorted(df.R_bin.unique()):
        in_bin = df[df.R_bin == R_bin]
        
        if len(in_bin) < 20:  # Need enough stars
            continue
        
        # Split by mass (median split within this radius bin)
        mass_median = in_bin.mass_Msun.median()
        low_mass = in_bin[in_bin.mass_Msun < mass_median]
        high_mass = in_bin[in_bin.mass_Msun >= mass_median]
        
        if len(low_mass) < 5 or len(high_mass) < 5:
            continue
        
        # Get velocities
        v_low = low_mass.v_phi_km_s.values
        v_high = high_mass.v_phi_km_s.values
        
        # Statistics
        v_low_median = np.median(v_low)
        v_high_median = np.median(v_high)
        delta_v = v_low_median - v_high_median
        pct_diff = 100 * delta_v / v_high_median
        
        # Statistical test (t-test equivalent via standard errors)
        se_low = np.std(v_low) / np.sqrt(len(v_low))
        se_high = np.std(v_high) / np.sqrt(len(high_mass))
        se_diff = np.sqrt(se_low**2 + se_high**2)
        snr = abs(delta_v) / se_diff if se_diff > 0 else 0
        
        results.append({
            'R_center_kpc': R_bin + R_bin_size/2,
            'N_low_mass': len(low_mass),
            'N_high_mass': len(high_mass),
            'mass_low_median': low_mass.mass_Msun.median(),
            'mass_high_median': high_mass.mass_Msun.median(),
            'v_low_median_km_s': v_low_median,
            'v_high_median_km_s': v_high_median,
            'delta_v_km_s': delta_v,
            'delta_v_stderr': se_diff,
            'pct_difference': pct_diff,
            'significance_sigma': snr,
        })
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print(f"Analyzed {len(results_df)} radius bins\n")
    print(f"{'R(kpc)':>8} {'N_low':>6} {'N_high':>7} {'M_low':>6} {'M_high':>7} "
          f"{'v_low':>7} {'v_high':>8} {'Δv':>7} {'%diff':>7} {'σ':>5}")
    print("-" * 85)
    
    for _, row in results_df.iterrows():
        print(f"{row.R_center_kpc:8.1f} {row.N_low_mass:6.0f} {row.N_high_mass:7.0f} "
              f"{row.mass_low_median:6.2f} {row.mass_high_median:7.2f} "
              f"{row.v_low_median_km_s:7.1f} {row.v_high_median_km_s:8.1f} "
              f"{row.delta_v_km_s:7.2f} {row.pct_difference:7.2f} {row.significance_sigma:5.1f}")
    
    # Overall statistics
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}\n")
    
    mean_delta_v = results_df.delta_v_km_s.mean()
    std_delta_v = results_df.delta_v_km_s.std()
    mean_pct = results_df.pct_difference.mean()
    
    # Combined significance (weighted average)
    weights = 1.0 / (results_df.delta_v_stderr**2)
    weighted_delta_v = np.average(results_df.delta_v_km_s, weights=weights)
    weighted_stderr = 1.0 / np.sqrt(weights.sum())
    combined_snr = abs(weighted_delta_v) / weighted_stderr
    
    print(f"Mean Δv:          {mean_delta_v:7.2f} ± {std_delta_v:.2f} km/s")
    print(f"Mean % difference: {mean_pct:7.2f}%")
    print(f"Weighted Δv:      {weighted_delta_v:7.2f} ± {weighted_stderr:.2f} km/s")
    print(f"Combined SNR:     {combined_snr:7.2f}σ\n")
    
    if combined_snr > 3:
        print("✅ SIGNIFICANT DETECTION: Low-mass stars have different velocities!")
        print(f"   → Δv = {weighted_delta_v:.2f} km/s ({combined_snr:.1f}σ)")
    elif combined_snr > 2:
        print("⚠️  MARGINAL DETECTION: Some evidence for velocity difference")
        print(f"   → Δv = {weighted_delta_v:.2f} km/s ({combined_snr:.1f}σ)")
    else:
        print("❌ NO SIGNIFICANT DETECTION: Velocities consistent within errors")
        print(f"   → Upper limit: |Δv| < {3*weighted_stderr:.2f} km/s (3σ)")
    
    return results_df


def plot_results(results_df: pd.DataFrame, output_dir: Path):
    """Create diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Velocities vs radius
    ax = axes[0, 0]
    ax.errorbar(results_df.R_center_kpc, results_df.v_low_median_km_s,
                yerr=results_df.delta_v_stderr, fmt='o-', label='Low-mass stars', 
                color='blue', alpha=0.7)
    ax.errorbar(results_df.R_center_kpc, results_df.v_high_median_km_s,
                yerr=results_df.delta_v_stderr, fmt='s-', label='High-mass stars',
                color='red', alpha=0.7)
    ax.set_xlabel('Galactocentric Radius (kpc)')
    ax.set_ylabel('Median v_φ (km/s)')
    ax.set_title('Rotation Velocities by Mass')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Velocity difference vs radius
    ax = axes[0, 1]
    ax.errorbar(results_df.R_center_kpc, results_df.delta_v_km_s,
                yerr=results_df.delta_v_stderr, fmt='o-', color='purple')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Galactocentric Radius (kpc)')
    ax.set_ylabel('Δv = v(low-mass) - v(high-mass) [km/s]')
    ax.set_title('Velocity Difference vs Radius')
    ax.grid(alpha=0.3)
    
    # Plot 3: Percentage difference
    ax = axes[1, 0]
    ax.plot(results_df.R_center_kpc, results_df.pct_difference, 'o-', color='green')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Galactocentric Radius (kpc)')
    ax.set_ylabel('Percentage Difference (%)')
    ax.set_title('Relative Velocity Difference')
    ax.grid(alpha=0.3)
    
    # Plot 4: Significance
    ax = axes[1, 1]
    colors = ['red' if s < 2 else 'orange' if s < 3 else 'green' 
              for s in results_df.significance_sigma]
    ax.bar(results_df.R_center_kpc, results_df.significance_sigma, 
           width=0.8, color=colors, alpha=0.7)
    ax.axhline(2, color='orange', linestyle='--', label='2σ', alpha=0.5)
    ax.axhline(3, color='green', linestyle='--', label='3σ', alpha=0.5)
    ax.set_xlabel('Galactocentric Radius (kpc)')
    ax.set_ylabel('Detection Significance (σ)')
    ax.set_title('Statistical Significance by Radius')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "baseline_mass_velocity_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_file}")
    plt.close()


def main():
    """Run simple baseline comparison."""
    print(f"\n{'='*70}")
    print(" BASELINE: Stars at Same Radius, Different Mass ")
    print(" NO MODELS - PURE OBSERVATION ")
    print(f"{'='*70}\n")
    
    # Generate test data (replace with real Gaia data!)
    df = generate_realistic_test_stars(n_stars=20000)
    
    # Compare velocities
    results = compare_velocities_at_same_radius(df, R_bin_size=1.0)
    
    # Save results
    output_dir = Path("gaia_test/results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / "mass_velocity_comparison.csv"
    results.to_csv(csv_file, index=False)
    print(f"\n✓ Saved results: {csv_file}")
    
    # Plot
    plot_results(results, output_dir)
    
    print(f"\n{'='*70}")
    print(" BASELINE ANALYSIS COMPLETE ")
    print(f"{'='*70}\n")
    
    print("What we measured:")
    print("- Stars at same radius with different masses")
    print("- Their observed rotation velocities")
    print("- The difference between low-mass and high-mass stars")
    print("- Statistical significance of any difference\n")
    
    print("⚠️  IMPORTANT: This used SYNTHETIC test data!")
    print("   For real science, provide Gaia 144k star dataset with:")
    print("   - Galactocentric coordinates (R, phi, z)")
    print("   - Velocities (v_phi, v_R, v_z)")
    print("   - Stellar masses (or mass proxies)")


if __name__ == "__main__":
    main()
