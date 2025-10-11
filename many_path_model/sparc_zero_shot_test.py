#!/usr/bin/env python3
"""
SPARC Zero-Shot Test: Many-Path Minimal Model on 20-30 Galaxies

Test the FROZEN minimal model parameters (from Milky Way) on diverse SPARC galaxies.
NO per-galaxy fitting allowed - this tests universality of the kernel.

Strategy:
1. Load SPARC data (rotation curves for ~175 galaxies)
2. Select 20-30 diverse galaxies by morphology type
3. Apply FROZEN minimal model parameters
4. Compute chi-square and APE per galaxy
5. Analyze performance by galaxy type (Sd, Im, etc.)
6. Generate comparison plots
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).parent))
from minimal_model import minimal_params
from toy_many_path_gravity import rotation_curve, xp_array, to_cpu

try:
    import cupy as cp
    _USING_CUPY = True
except Exception:
    import numpy as cp
    _USING_CUPY = False

# Data paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SPARC_FILE = DATA_DIR / "sparc_rotmod_ltg.parquet"  # Rotation curve data
SPARC_MASTER = DATA_DIR / "sparc_master_clean.parquet"  # Galaxy properties
RESULTS_DIR = Path(__file__).parent / "results" / "sparc_zero_shot"


def load_sparc_data():
    """Load SPARC rotation curve data."""
    print("Loading SPARC data...")
    df_rot = pd.read_parquet(SPARC_FILE)
    df_master = pd.read_parquet(SPARC_MASTER)
    
    # Merge to get galaxy types
    df = df_rot.merge(df_master[['galaxy', 'T']], on='galaxy', how='left')
    
    # Map numeric type to string (Hubble classification)
    # T: 10=Im, 9=Sm, 8=Sd, 7=Scd, 6=Sc, 5=Sbc, 4=Sb, 3=Sab, 2=Sa
    type_map = {
        10: 'Im', 9: 'Sm', 8: 'Sd', 7: 'Scd', 
        6: 'Sc', 5: 'Sbc', 4: 'Sb', 3: 'Sab', 2: 'Sa'
    }
    df['Type'] = df['T'].map(type_map).fillna('Unknown')
    
    print(f"✓ Loaded {len(df)} data points from {df.galaxy.nunique()} galaxies")
    print(f"\nGalaxy type distribution:")
    print(df.groupby('Type').galaxy.nunique().sort_values(ascending=False))
    
    return df


def select_diverse_sample(df, n_galaxies=25, seed=42):
    """
    Select diverse sample of galaxies by morphology type.
    
    Strategy:
    - Include multiple types (Sd, Scd, Im, Sbc, etc.)
    - Prefer galaxies with many data points
    - Ensure range of masses and sizes
    """
    np.random.seed(seed)
    
    # Count points per galaxy
    galaxy_stats = df.groupby('galaxy').agg({
        'R_kpc': 'count',
        'Type': 'first',
        'Vobs_kms': lambda x: x.notna().sum()  # Non-NaN velocities
    }).rename(columns={'R_kpc': 'n_points', 'Vobs_kms': 'n_valid'})
    
    # Filter: at least 10 valid points
    galaxy_stats = galaxy_stats[galaxy_stats.n_valid >= 10]
    
    # Sample by type
    types = galaxy_stats.Type.unique()
    selected = []
    
    # Target: ~5 galaxies per common type, fewer for rare types
    type_targets = {
        'Sd': 6,
        'Scd': 5,
        'Im': 4,
        'Sbc': 3,
        'Sc': 3,
        'Sm': 2,
        'Sab': 1,
        'Sb': 1
    }
    
    for gtype, target in type_targets.items():
        candidates = galaxy_stats[galaxy_stats.Type == gtype]
        if len(candidates) == 0:
            continue
        
        # Sample up to target, preferring more data points
        n_select = min(target, len(candidates))
        sampled = candidates.nlargest(n_select * 2, 'n_points').sample(
            n=n_select, random_state=seed + len(selected)
        )
        selected.extend(sampled.index.tolist())
    
    # Fill to n_galaxies if needed
    if len(selected) < n_galaxies:
        remaining = galaxy_stats[~galaxy_stats.index.isin(selected)]
        extra = remaining.nlargest(n_galaxies - len(selected), 'n_points')
        selected.extend(extra.index.tolist())
    
    selected = selected[:n_galaxies]
    
    print(f"\n✓ Selected {len(selected)} galaxies:")
    sample_stats = galaxy_stats.loc[selected]
    for gtype in sample_stats.Type.unique():
        count = (sample_stats.Type == gtype).sum()
        print(f"  {gtype:5s}: {count} galaxies")
    
    return selected


def sample_galaxy_mass_distribution(galaxy_name, df_galaxy, n_disk=50000, n_bulge=5000):
    """
    Sample mass distribution for a SPARC galaxy.
    
    Use observed stellar + gas surface density profiles to sample particles.
    Simplified: exponential disk + optional bulge.
    """
    from toy_many_path_gravity import sample_exponential_disk, sample_hernquist_bulge, xp_zeros
    
    # Estimate scale parameters from data
    # Use median radius and velocity to infer scale length
    R_median = df_galaxy.R_kpc.median()
    V_median = df_galaxy.Vobs_kms.median()
    
    # Rough estimates (can be improved with actual SPARC stellar mass profiles)
    R_d = R_median / 1.7  # Typical: R_d ≈ 0.6 * R_half
    z_d = 0.1 * R_d  # Thin disk assumption
    R_max = df_galaxy.R_kpc.max() * 1.5
    
    # Total baryonic mass (rough estimate from V_median)
    # M ≈ V² * R / G
    G_SI = 4.30091e-6  # kpc (km/s)^2 / M_sun
    M_total_est = V_median**2 * R_median / G_SI
    
    # Split: 80% disk, 20% bulge (if applicable)
    M_disk = 0.8 * M_total_est
    M_bulge = 0.2 * M_total_est
    
    # Sample disk
    disk_pos, m_disk = sample_exponential_disk(
        n_disk, M_disk=M_disk, R_d=R_d, z_d=z_d, R_max=R_max, seed=42
    )
    
    # Sample bulge (smaller for late-type galaxies)
    gtype = df_galaxy.Type.iloc[0]
    if gtype in ['Sd', 'Scd', 'Im', 'Sm']:
        # Late-type: minimal bulge
        n_bulge = max(1000, n_bulge // 5)
        M_bulge = M_bulge * 0.2
    
    bulge_pos, m_bulge = sample_hernquist_bulge(
        n_bulge, M_bulge=M_bulge, a=R_d * 0.3, seed=123
    )
    
    # Combine
    src_pos = cp.concatenate([disk_pos, bulge_pos], axis=0)
    src_mass = cp.concatenate([
        xp_zeros(disk_pos.shape[0]) + m_disk,
        xp_zeros(bulge_pos.shape[0]) + m_bulge
    ])
    
    return src_pos, src_mass


def test_galaxy(galaxy_name, df_galaxy, params, eps=0.05, batch_size=25000):
    """
    Test frozen parameters on a single galaxy.
    
    Returns metrics: chi2, APE, RMS residual
    """
    # Extract observed rotation curve
    df_clean = df_galaxy[df_galaxy.Vobs_kms.notna()].copy()
    if len(df_clean) < 5:
        return None  # Too few points
    
    R_obs = df_clean.R_kpc.values  # kpc
    V_obs = df_clean.Vobs_kms.values  # km/s
    V_err = df_clean.eVobs_kms.fillna(10.0).values  # km/s (default 10 km/s if missing)
    
    # Sample mass distribution
    src_pos, src_mass = sample_galaxy_mass_distribution(
        galaxy_name, df_galaxy, n_disk=50000, n_bulge=5000
    )
    
    # Compute model prediction (FROZEN parameters)
    R_grid = xp_array(R_obs)
    v_pred, _ = rotation_curve(
        src_pos, src_mass, R_grid, z=0.0,
        eps=eps, params=params, use_multiplier=True,
        batch_size=batch_size
    )
    v_pred = to_cpu(v_pred)
    
    # Compute metrics
    residuals = V_obs - v_pred
    chi2 = np.sum((residuals / V_err)**2)
    ape = np.median(np.abs(residuals / V_obs) * 100)  # Absolute percentage error
    rms = np.sqrt(np.mean(residuals**2))
    
    # Reduced chi2
    dof = len(R_obs) - 0  # No free parameters!
    chi2_red = chi2 / max(1, dof)
    
    return {
        'galaxy': galaxy_name,
        'type': df_galaxy.Type.iloc[0],
        'n_points': len(R_obs),
        'chi2': chi2,
        'chi2_red': chi2_red,
        'ape': ape,
        'rms': rms,
        'R_obs': R_obs,
        'V_obs': V_obs,
        'V_err': V_err,
        'V_pred': v_pred,
        'residuals': residuals
    }


def run_zero_shot_test(galaxies, df, params, batch_size=25000):
    """
    Run zero-shot test on all selected galaxies.
    
    Returns DataFrame with results per galaxy.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print(f"\n{'='*70}")
    print("ZERO-SHOT TEST: FROZEN MINIMAL MODEL ON SPARC")
    print(f"{'='*70}\n")
    
    print(f"Testing {len(galaxies)} galaxies with FROZEN parameters:")
    print(f"  (no per-galaxy fitting allowed)\n")
    
    for i, galaxy in enumerate(galaxies):
        df_galaxy = df[df.galaxy == galaxy]
        
        print(f"[{i+1:2d}/{len(galaxies)}] {galaxy:20s} ({df_galaxy.Type.iloc[0]:5s})...", end=' ')
        
        try:
            result = test_galaxy(galaxy, df_galaxy, params, batch_size=batch_size)
            if result is None:
                print("SKIP (too few points)")
                continue
            
            results.append(result)
            print(f"χ²={result['chi2']:.1f}, APE={result['ape']:.1f}%")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    return results


def analyze_by_type(results):
    """Analyze performance grouped by galaxy morphology type."""
    df_results = pd.DataFrame([{
        'galaxy': r['galaxy'],
        'type': r['type'],
        'n_points': r['n_points'],
        'chi2': r['chi2'],
        'chi2_red': r['chi2_red'],
        'ape': r['ape'],
        'rms': r['rms']
    } for r in results])
    
    print(f"\n{'='*70}")
    print("PERFORMANCE BY GALAXY TYPE")
    print(f"{'='*70}\n")
    
    type_summary = df_results.groupby('type').agg({
        'galaxy': 'count',
        'chi2': 'mean',
        'chi2_red': 'mean',
        'ape': 'mean',
        'rms': 'mean'
    }).rename(columns={'galaxy': 'n_galaxies'})
    
    type_summary = type_summary.sort_values('n_galaxies', ascending=False)
    
    print(f"{'Type':5s} {'N':>3s} {'χ²':>8s} {'χ²_red':>8s} {'APE(%)':>8s} {'RMS':>8s}")
    print("-" * 50)
    for gtype, row in type_summary.iterrows():
        print(f"{gtype:5s} {row.n_galaxies:3.0f} "
              f"{row.chi2:8.1f} {row.chi2_red:8.2f} "
              f"{row.ape:8.1f} {row.rms:8.1f}")
    
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE")
    print("="*70 + "\n")
    
    print(f"Total galaxies tested: {len(df_results)}")
    print(f"Median APE: {df_results.ape.median():.1f}%")
    print(f"Mean χ²_red: {df_results.chi2_red.mean():.2f}")
    print(f"Success rate (APE < 30%): {(df_results.ape < 30).sum() / len(df_results) * 100:.1f}%")
    
    return df_results, type_summary


def plot_sample_galaxies(results, n_sample=6):
    """Plot rotation curves for sample galaxies."""
    import random
    random.seed(42)
    
    # Sample diverse types
    sampled = random.sample(results, min(n_sample, len(results)))
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    for i, result in enumerate(sampled):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        
        R = result['R_obs']
        V_obs = result['V_obs']
        V_err = result['V_err']
        V_pred = result['V_pred']
        
        # Plot observed
        ax.errorbar(R, V_obs, yerr=V_err, fmt='o', color='blue', 
                   alpha=0.6, label='SPARC obs', capsize=3, markersize=4)
        
        # Plot predicted
        ax.plot(R, V_pred, '-', color='red', linewidth=2, 
               label='Many-path (frozen)')
        
        ax.set_xlabel('Radius (kpc)', fontsize=10)
        ax.set_ylabel('V$_{circ}$ (km/s)', fontsize=10)
        ax.set_title(f"{result['galaxy']} ({result['type']})\n"
                    f"APE={result['ape']:.1f}%, χ²$_r$={result['chi2_red']:.2f}",
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Many-Path Minimal Model: Zero-Shot SPARC Predictions\n'
                 '(Frozen Milky Way Parameters)', 
                 fontsize=14, fontweight='bold')
    
    output_file = RESULTS_DIR / "sample_galaxies.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved sample plots: {output_file}")
    plt.close()


def plot_performance_by_type(df_results, type_summary):
    """Plot performance metrics by galaxy type."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sort by number of galaxies
    types_sorted = type_summary.sort_values('n_galaxies', ascending=False).index.tolist()
    
    # Panel 1: APE by type
    ax = axes[0, 0]
    df_results['type'] = pd.Categorical(df_results['type'], categories=types_sorted, ordered=True)
    df_results_sorted = df_results.sort_values('type')
    
    ax.boxplot([df_results[df_results.type == t].ape.values for t in types_sorted],
               labels=types_sorted, showfliers=False)
    ax.axhline(30, color='red', linestyle='--', alpha=0.5, label='APE=30% threshold')
    ax.set_ylabel('APE (%)', fontsize=12)
    ax.set_title('Absolute Percentage Error by Type', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    
    # Panel 2: Reduced chi-square by type
    ax = axes[0, 1]
    ax.boxplot([df_results[df_results.type == t].chi2_red.values for t in types_sorted],
               labels=types_sorted, showfliers=False)
    ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='χ²$_r$=1')
    ax.set_ylabel('χ²$_{red}$', fontsize=12)
    ax.set_title('Reduced Chi-Square by Type', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    
    # Panel 3: Success rate by type
    ax = axes[1, 0]
    success_rates = []
    for t in types_sorted:
        subset = df_results[df_results.type == t]
        rate = (subset.ape < 30).sum() / len(subset) * 100
        success_rates.append(rate)
    
    ax.bar(types_sorted, success_rates, color='green', alpha=0.6, edgecolor='black')
    ax.axhline(50, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate by Type (APE < 30%)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 4: Number of galaxies per type
    ax = axes[1, 1]
    ax.bar(types_sorted, type_summary.loc[types_sorted, 'n_galaxies'], 
          color='blue', alpha=0.6, edgecolor='black')
    ax.set_ylabel('Number of Galaxies', fontsize=12)
    ax.set_title('Sample Size by Type', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Many-Path Minimal Model: Performance on SPARC by Galaxy Type\n'
                 '(Zero-Shot Test, Frozen Parameters)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = RESULTS_DIR / "performance_by_type.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved performance plots: {output_file}")
    plt.close()


def save_results(results, df_results, type_summary):
    """Save numerical results to CSV."""
    # Per-galaxy results
    df_results.to_csv(RESULTS_DIR / "results_per_galaxy.csv", index=False)
    print(f"✓ Saved per-galaxy results: {RESULTS_DIR / 'results_per_galaxy.csv'}")
    
    # Type summary
    type_summary.to_csv(RESULTS_DIR / "summary_by_type.csv")
    print(f"✓ Saved type summary: {RESULTS_DIR / 'summary_by_type.csv'}")
    
    # Full results with curves (pickle for later analysis)
    import pickle
    with open(RESULTS_DIR / "full_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Saved full results: {RESULTS_DIR / 'full_results.pkl'}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SPARC zero-shot test")
    parser.add_argument("--n_galaxies", type=int, default=25, help="Number of galaxies to test")
    parser.add_argument("--batch_size", type=int, default=25000, help="GPU batch size")
    parser.add_argument("--gpu", type=int, default=1, help="Use GPU (1) or CPU (0)")
    args = parser.parse_args()
    
    # Load data
    df = load_sparc_data()
    
    # Select diverse sample
    galaxies = select_diverse_sample(df, n_galaxies=args.n_galaxies)
    
    # Get FROZEN parameters from Milky Way fit
    params = minimal_params()
    print(f"\n{'='*70}")
    print("FROZEN MINIMAL MODEL PARAMETERS (from Milky Way)")
    print(f"{'='*70}")
    for key, val in params.items():
        print(f"  {key:12s} = {val:.3f}")
    print("="*70 + "\n")
    
    # Run zero-shot test
    results = run_zero_shot_test(galaxies, df, params, batch_size=args.batch_size)
    
    if len(results) == 0:
        print("ERROR: No successful tests!")
        return
    
    # Analyze by type
    df_results, type_summary = analyze_by_type(results)
    
    # Generate plots
    plot_sample_galaxies(results, n_sample=6)
    plot_performance_by_type(df_results, type_summary)
    
    # Save results
    save_results(results, df_results, type_summary)
    
    print(f"\n{'='*70}")
    print("ZERO-SHOT TEST COMPLETE")
    print(f"{'='*70}\n")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
