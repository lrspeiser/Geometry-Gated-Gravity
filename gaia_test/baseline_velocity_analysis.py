#!/usr/bin/env python3
"""
Baseline Velocity Analysis - NO MODEL ASSUMPTIONS

Goal: Check if there are OBSERVABLE velocity differences for tracers
at the same galactocentric radius, BEFORE invoking any gravity model.

This answers: "Do we even SEE a signal to explain?"

For stellar data: Compare low-mass vs high-mass stars at same R
For galaxy data: Compare different tracer types (HI vs stars) at same R

Output Format:
--------------
For each radius bin:
- Tracer A mass, velocity
- Tracer B mass, velocity  
- Observed Δv = v_A - v_B
- Expected Δv from GR (asymmetric drift, if applicable)
- Residual Δv after corrections
- % difference
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict

ROOT = Path(__file__).resolve().parents[1]


def load_sparc_galaxy_data(galaxy_name: str) -> pd.DataFrame:
    """Load SPARC rotation curve data for a single galaxy."""
    rotmod_file = ROOT / "data" / "Rotmod_LTG" / f"{galaxy_name}_rotmod.dat"
    
    if not rotmod_file.exists():
        raise FileNotFoundError(f"Galaxy {galaxy_name} not found: {rotmod_file}")
    
    # SPARC format: R(kpc), Vobs(km/s), errV(km/s), Vgas, Vdisk, Vbul, SBdisk, SBbul
    df = pd.read_csv(rotmod_file, delim_whitespace=True, comment='#',
                     names=['R_kpc', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
    
    return df


def analyze_sparc_baseline(galaxy_name: str) -> pd.DataFrame:
    """
    Baseline analysis for SPARC galaxy data.
    
    Compares gas velocity (HI tracer, low mass per particle)
    vs stellar velocity (disk stars, higher mass per particle)
    at the same radius.
    """
    print(f"\n{'='*70}")
    print(f" BASELINE ANALYSIS: {galaxy_name} ")
    print(f"{'='*70}\n")
    
    df = load_sparc_galaxy_data(galaxy_name)
    
    print(f"Loaded {len(df)} radius points")
    print(f"Radius range: {df.R_kpc.min():.1f} - {df.R_kpc.max():.1f} kpc")
    print(f"Velocity range: {df.Vobs.min():.1f} - {df.Vobs.max():.1f} km/s\n")
    
    # Calculate circular velocities from components
    # Vobs² = Vgas² + Vdisk² + Vbul² (approximately, in quadrature)
    df['V_gas_circ'] = df.Vgas.abs()
    df['V_stars_circ'] = np.sqrt(df.Vdisk**2 + df.Vbul**2)
    
    # Only consider radii where we have both gas and stellar data
    valid = (df.V_gas_circ > 0) & (df.V_stars_circ > 0) & (df.Vobs > 0)
    df_valid = df[valid].copy()
    
    if len(df_valid) == 0:
        print("⚠️  No radii with both gas and stellar velocities!")
        return pd.DataFrame()
    
    # Compute velocity differences
    df_valid['Delta_V'] = df_valid.V_gas_circ - df_valid.V_stars_circ
    df_valid['Delta_V_pct'] = 100 * df_valid.Delta_V / df_valid.Vobs
    
    # GR expectation: asymmetric drift
    # Stars have velocity dispersion → lower mean v_circ than gas
    # Expect: V_stars < V_gas by ~few km/s in outer disk
    # For now, use empirical: Delta_V_expected ~ -10 km/s (stars slower)
    df_valid['Delta_V_expected'] = -10.0  # km/s (asymmetric drift correction)
    df_valid['Delta_V_residual'] = df_valid.Delta_V - df_valid.Delta_V_expected
    
    # Statistics
    print(f"{'='*70}")
    print("VELOCITY DIFFERENCES (Gas - Stars)")
    print(f"{'='*70}\n")
    
    print(f"Mean Δv:     {df_valid.Delta_V.mean():6.2f} ± {df_valid.Delta_V.std():.2f} km/s")
    print(f"Median Δv:   {df_valid.Delta_V.median():6.2f} km/s")
    print(f"Mean % diff: {df_valid.Delta_V_pct.mean():6.2f} ± {df_valid.Delta_V_pct.std():.2f} %")
    
    print(f"\nAfter asymmetric drift correction (-10 km/s):")
    print(f"Residual Δv: {df_valid.Delta_V_residual.mean():6.2f} ± {df_valid.Delta_V_residual.std():.2f} km/s")
    
    # Check if residual is significant
    stderr = df_valid.Delta_V_residual.std() / np.sqrt(len(df_valid))
    snr = abs(df_valid.Delta_V_residual.mean()) / stderr if stderr > 0 else 0
    
    print(f"\nStatistical significance: {snr:.2f}σ")
    if snr > 3:
        print("✅ SIGNIFICANT residual detected after corrections!")
    elif snr > 2:
        print("⚠️  Marginal residual (~2-3σ)")
    else:
        print("❌ No significant residual (<2σ)")
    
    # Format output
    output = df_valid[[
        'R_kpc', 'Vobs', 'V_gas_circ', 'V_stars_circ', 
        'Delta_V', 'Delta_V_expected', 'Delta_V_residual', 'Delta_V_pct'
    ]].copy()
    
    output.columns = [
        'R_kpc', 'V_total', 'V_gas', 'V_stars',
        'Delta_V_obs', 'Delta_V_GR_expected', 'Delta_V_residual', 'Pct_diff'
    ]
    
    return output


def plot_baseline_results(results: pd.DataFrame, galaxy_name: str, output_dir: Path):
    """Create diagnostic plots for baseline analysis."""
    if len(results) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Rotation curves
    ax = axes[0, 0]
    ax.plot(results.R_kpc, results.V_total, 'k-', label='Total (observed)', linewidth=2)
    ax.plot(results.R_kpc, results.V_gas, 'b--', label='Gas (HI)', alpha=0.7)
    ax.plot(results.R_kpc, results.V_stars, 'r--', label='Stars (disk+bulge)', alpha=0.7)
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Circular Velocity (km/s)')
    ax.set_title(f'{galaxy_name}: Rotation Curve by Tracer Type')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Velocity differences
    ax = axes[0, 1]
    ax.plot(results.R_kpc, results.Delta_V_obs, 'o-', label='Observed Δv (gas - stars)')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(results.Delta_V_GR_expected.iloc[0], color='r', linestyle=':', 
               label='GR expected (asymmetric drift)', alpha=0.7)
    ax.fill_between(results.R_kpc, 
                     results.Delta_V_obs - results.Delta_V_obs.std(),
                     results.Delta_V_obs + results.Delta_V_obs.std(),
                     alpha=0.2)
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Δv = V_gas - V_stars (km/s)')
    ax.set_title('Velocity Difference vs Radius')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Residuals after GR correction
    ax = axes[1, 0]
    ax.plot(results.R_kpc, results.Delta_V_residual, 'o-', color='purple')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.fill_between(results.R_kpc,
                     results.Delta_V_residual - results.Delta_V_residual.std(),
                     results.Delta_V_residual + results.Delta_V_residual.std(),
                     alpha=0.2, color='purple')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Residual Δv after GR correction (km/s)')
    ax.set_title('Residual Velocity Difference (GR removed)')
    ax.grid(alpha=0.3)
    
    # Plot 4: Percentage differences
    ax = axes[1, 1]
    ax.plot(results.R_kpc, results.Pct_diff, 'o-', color='green')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Percentage Difference (%)')
    ax.set_title('Δv / V_total (%)')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / f"{galaxy_name}_baseline_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_file}")
    plt.close()


def main():
    """Run baseline analysis on available galaxy data."""
    print(f"\n{'='*70}")
    print(" BASELINE VELOCITY ANALYSIS (NO MODEL ASSUMPTIONS) ")
    print(f"{'='*70}\n")
    
    print("Goal: Check if velocity differences exist BEFORE invoking models")
    print("Method: Compare different tracer types at same radius\n")
    
    # Test galaxies with good outer disk data
    test_galaxies = [
        "NGC6503",    # Well-studied dwarf spiral
        "NGC3198",    # Classic rotation curve
        "DDO154",     # Low surface brightness
        "IC2574",     # Gas-rich dwarf
        "NGC2403",    # Nearby spiral
    ]
    
    output_dir = Path("gaia_test/results/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for galaxy in test_galaxies:
        try:
            results = analyze_sparc_baseline(galaxy)
            
            if len(results) > 0:
                # Save CSV
                csv_file = output_dir / f"{galaxy}_baseline.csv"
                results.to_csv(csv_file, index=False)
                print(f"✓ Saved: {csv_file}")
                
                # Plot
                plot_baseline_results(results, galaxy, output_dir)
                
                all_results[galaxy] = results
                
        except FileNotFoundError as e:
            print(f"⚠️  Skipping {galaxy}: {e}")
            continue
        except Exception as e:
            print(f"❌ Error processing {galaxy}: {e}")
            continue
    
    # Summary across all galaxies
    if all_results:
        print(f"\n{'='*70}")
        print("SUMMARY ACROSS ALL GALAXIES")
        print(f"{'='*70}\n")
        
        all_delta_v = []
        all_residuals = []
        
        for galaxy, df in all_results.items():
            all_delta_v.extend(df.Delta_V_obs.values)
            all_residuals.extend(df.Delta_V_residual.values)
        
        all_delta_v = np.array(all_delta_v)
        all_residuals = np.array(all_residuals)
        
        print(f"Total data points: {len(all_delta_v)}")
        print(f"\nObserved Δv (gas - stars):")
        print(f"  Mean:   {np.mean(all_delta_v):6.2f} ± {np.std(all_delta_v):.2f} km/s")
        print(f"  Median: {np.median(all_delta_v):6.2f} km/s")
        
        print(f"\nResidual Δv (after GR correction):")
        print(f"  Mean:   {np.mean(all_residuals):6.2f} ± {np.std(all_residuals):.2f} km/s")
        print(f"  Median: {np.median(all_residuals):6.2f} km/s")
        
        stderr = np.std(all_residuals) / np.sqrt(len(all_residuals))
        snr = abs(np.mean(all_residuals)) / stderr if stderr > 0 else 0
        
        print(f"\nCombined significance: {snr:.2f}σ")
        
        if snr > 3:
            print("\n✅ STRONG SIGNAL: Systematic velocity difference detected!")
            print("   → This is what your cooperative response model could explain")
        elif snr > 2:
            print("\n⚠️  MARGINAL SIGNAL: Some evidence for velocity differences")
            print("   → Worth investigating with cooperative response model")
        else:
            print("\n❌ NO CLEAR SIGNAL: Velocity differences consistent with GR")
            print("   → Either no anomaly exists, or systematic errors dominate")
    
    print(f"\n{'='*70}")
    print("BASELINE ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    print("Next steps:")
    print("1. If signal detected → Test cooperative response model")
    print("2. If no signal → Place upper limits on mass-dependent effects")
    print("3. Get Gaia stellar data → Test at individual star level")


if __name__ == "__main__":
    main()
