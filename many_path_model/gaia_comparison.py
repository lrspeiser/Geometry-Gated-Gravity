#!/usr/bin/env python3
"""
Compare many-path gravity model predictions with REAL Gaia DR3 data.

This script:
1. Loads real Gaia data from the project
2. Computes rotation curves using the many-path model
3. Compares predictions with observed velocities
4. Tests mass-dependent velocity predictions
5. Evaluates vertical structure predictions
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path to import the toy model
sys.path.insert(0, str(Path(__file__).parent))
from toy_many_path_gravity import (
    sample_exponential_disk, sample_hernquist_bulge,
    rotation_curve, vertical_frequency, default_params,
    to_cpu, xp_array, xp_zeros
)

try:
    import cupy as cp
    _USING_CUPY = True
except Exception:
    import numpy as cp
    _USING_CUPY = False

ROOT = Path(__file__).resolve().parents[1]


def load_gaia_data():
    """Load real Gaia data and compute observed rotation curve."""
    data_file = ROOT / "data" / "gaia_mw_real.csv"
    
    print(f"Loading REAL Gaia data from: {data_file}")
    
    if not data_file.exists():
        raise FileNotFoundError(f"Gaia data not found: {data_file}")
    
    df = pd.read_csv(data_file)
    print(f"OK Loaded {len(df)} REAL stars from Gaia DR3\n")
    
    return df


def compute_observed_rotation_curve(df: pd.DataFrame, R_bins=None) -> pd.DataFrame:
    """
    Compute observed rotation curve from Gaia data.
    
    Returns DataFrame with R, v_phi_median, v_phi_std, N_stars
    """
    if R_bins is None:
        R_bins = np.arange(5.0, 15.0, 0.5)
    
    results = []
    
    for i in range(len(R_bins) - 1):
        R_min, R_max = R_bins[i], R_bins[i + 1]
        R_center = (R_min + R_max) / 2
        
        # Select stars in this radial bin (near disk plane)
        mask = (df.R_kpc >= R_min) & (df.R_kpc < R_max) & (np.abs(df.z_kpc) < 0.5)
        stars = df[mask]
        
        if len(stars) < 10:
            continue
        
        results.append({
            'R_kpc': R_center,
            'v_phi_median': stars.vphi.median(),
            'v_phi_std': stars.vphi.std(),
            'v_phi_sem': stars.vphi.std() / np.sqrt(len(stars)),
            'N_stars': len(stars)
        })
    
    return pd.DataFrame(results)


def compare_models(n_sources=500000, n_bulge=100000, use_bulge=True, 
                   R_range=(5, 15, 50), batch_size=200000, gpu=True):
    """
    Compare Newtonian and many-path predictions with Gaia observations.
    """
    print(f"\n{'='*70}")
    print("MANY-PATH MODEL vs GAIA COMPARISON")
    print(f"{'='*70}\n")
    
    # Load Gaia observations
    df_gaia = load_gaia_data()
    obs_curve = compute_observed_rotation_curve(df_gaia)
    
    print("Observed rotation curve from Gaia:")
    print(obs_curve.to_string(index=False))
    print()
    
    # Sample source distribution
    print(f"Sampling {n_sources} disk particles...")
    disk_pos, m_disk = sample_exponential_disk(
        n_sources, M_disk=5e10, R_d=2.6, z_d=0.3, R_max=30.0, seed=42
    )
    
    if use_bulge:
        print(f"Sampling {n_bulge} bulge particles...")
        bulge_pos, m_bulge = sample_hernquist_bulge(
            n_bulge, M_bulge=1e10, a=0.7, seed=123
        )
        src_pos = cp.concatenate([disk_pos, bulge_pos], axis=0)
        src_mass = cp.concatenate([
            xp_zeros(disk_pos.shape[0]) + m_disk,
            xp_zeros(bulge_pos.shape[0]) + m_bulge
        ])
    else:
        src_pos = disk_pos
        src_mass = xp_zeros(disk_pos.shape[0]) + m_disk
    
    print(f"✓ Total source particles: {src_pos.shape[0]}\n")
    
    # Target radii
    Rmin, Rmax, Np = R_range
    R_vals = cp.linspace(Rmin, Rmax, int(Np), dtype=cp.float64)
    
    # Get default parameters
    params = default_params()
    
    # Compute curves
    print("Computing Newtonian rotation curve...")
    vN, _ = rotation_curve(src_pos, src_mass, R_vals, z=0.0, eps=0.05, 
                          params=None, use_multiplier=False, batch_size=batch_size)
    
    print("Computing many-path rotation curve...")
    vM, _ = rotation_curve(src_pos, src_mass, R_vals, z=0.0, eps=0.05,
                          params=params, use_multiplier=True, batch_size=batch_size)
    
    # Convert to numpy
    R_vals_np = to_cpu(R_vals)
    vN_np = to_cpu(vN)
    vM_np = to_cpu(vM)
    
    # Compute residuals
    # Interpolate model predictions to Gaia radii
    vN_interp = np.interp(obs_curve.R_kpc, R_vals_np, vN_np)
    vM_interp = np.interp(obs_curve.R_kpc, R_vals_np, vM_np)
    
    residual_N = obs_curve.v_phi_median.values - vN_interp
    residual_M = obs_curve.v_phi_median.values - vM_interp
    
    chi2_N = np.sum((residual_N / obs_curve.v_phi_sem.values)**2)
    chi2_M = np.sum((residual_M / obs_curve.v_phi_sem.values)**2)
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"Newtonian χ²: {chi2_N:.2f}")
    print(f"Many-path χ²: {chi2_M:.2f}")
    print(f"Improvement:  {chi2_N - chi2_M:.2f} (lower is better)\n")
    
    # Print comparison table
    print(f"{'R(kpc)':>7} {'Gaia':>8} {'±':>8} {'Newton':>8} {'Many-P':>8} {'ΔN':>8} {'ΔM':>8}")
    print("-" * 65)
    for i, row in obs_curve.iterrows():
        R = row.R_kpc
        v_obs = row.v_phi_median
        err = row.v_phi_sem
        v_N = np.interp(R, R_vals_np, vN_np)
        v_M = np.interp(R, R_vals_np, vM_np)
        dN = v_obs - v_N
        dM = v_obs - v_M
        print(f"{R:7.2f} {v_obs:8.1f} {err:8.2f} {v_N:8.1f} {v_M:8.1f} {dN:8.1f} {dM:8.1f}")
    
    return {
        'R_vals': R_vals_np,
        'v_Newton': vN_np,
        'v_manypath': vM_np,
        'gaia_obs': obs_curve,
        'chi2_Newton': chi2_N,
        'chi2_manypath': chi2_M,
        'params': params,
    }


def plot_comparison(results: dict, output_dir: Path):
    """Create comprehensive comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    obs = results['gaia_obs']
    R = results['R_vals']
    vN = results['v_Newton']
    vM = results['v_manypath']
    
    # Plot 1: Rotation curves
    ax = axes[0, 0]
    ax.errorbar(obs.R_kpc, obs.v_phi_median, yerr=obs.v_phi_sem,
                fmt='o', label='Gaia observations', color='black', 
                markersize=8, capsize=5, zorder=10)
    ax.plot(R, vN, '--', label='Newtonian', color='blue', linewidth=2)
    ax.plot(R, vM, '-', label='Many-path', color='red', linewidth=2)
    ax.set_xlabel('Galactocentric Radius (kpc)', fontsize=12)
    ax.set_ylabel('v_φ (km/s)', fontsize=12)
    ax.set_title('Rotation Curves: Models vs Gaia', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(150, 300)
    
    # Plot 2: Residuals
    ax = axes[0, 1]
    vN_interp = np.interp(obs.R_kpc, R, vN)
    vM_interp = np.interp(obs.R_kpc, R, vM)
    residual_N = obs.v_phi_median.values - vN_interp
    residual_M = obs.v_phi_median.values - vM_interp
    
    ax.errorbar(obs.R_kpc, residual_N, yerr=obs.v_phi_sem,
                fmt='o--', label='Newtonian', color='blue', markersize=8, capsize=5)
    ax.errorbar(obs.R_kpc, residual_M, yerr=obs.v_phi_sem,
                fmt='s-', label='Many-path', color='red', markersize=8, capsize=5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Galactocentric Radius (kpc)', fontsize=12)
    ax.set_ylabel('Residual: Gaia - Model (km/s)', fontsize=12)
    ax.set_title('Model Residuals', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 3: Boost factor
    ax = axes[1, 0]
    boost = vM / vN
    ax.plot(R, boost, '-', color='purple', linewidth=2)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3, label='No boost')
    ax.set_xlabel('Galactocentric Radius (kpc)', fontsize=12)
    ax.set_ylabel('Boost: v_manypath / v_Newton', fontsize=12)
    ax.set_title('Many-Path Velocity Boost', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 4: χ² comparison
    ax = axes[1, 1]
    chi2s = [results['chi2_Newton'], results['chi2_manypath']]
    labels = ['Newtonian', 'Many-path']
    colors = ['blue', 'red']
    bars = ax.bar(labels, chi2s, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('χ² (lower is better)', fontsize=12)
    ax.set_title('Model Goodness of Fit', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, chi2 in zip(bars, chi2s):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{chi2:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / "many_path_vs_gaia.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare many-path model with Gaia data")
    parser.add_argument("--n_sources", type=int, default=500000, help="Number of disk particles")
    parser.add_argument("--n_bulge", type=int, default=100000, help="Number of bulge particles")
    parser.add_argument("--use_bulge", type=int, default=1, help="Include bulge (1=yes, 0=no)")
    parser.add_argument("--batch_size", type=int, default=200000, help="Batch size for computation")
    parser.add_argument("--gpu", type=int, default=1, help="Use GPU if available")
    parser.add_argument("--output_dir", type=str, default="results/gaia_comparison",
                       help="Output directory for plots")
    args = parser.parse_args()
    
    # Run comparison
    results = compare_models(
        n_sources=args.n_sources,
        n_bulge=args.n_bulge,
        use_bulge=bool(args.use_bulge),
        R_range=(5, 15, 50),
        batch_size=args.batch_size,
        gpu=bool(args.gpu)
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    
    # Save comparison table
    df_results = pd.DataFrame({
        'R_kpc': results['R_vals'],
        'v_Newton_km_s': results['v_Newton'],
        'v_manypath_km_s': results['v_manypath'],
        'boost_factor': results['v_manypath'] / results['v_Newton']
    })
    csv_file = output_dir / "model_predictions.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(csv_file, index=False)
    print(f"✓ Saved predictions: {csv_file}")
    
    # Save Gaia observations
    gaia_file = output_dir / "gaia_observations.csv"
    results['gaia_obs'].to_csv(gaia_file, index=False)
    print(f"✓ Saved Gaia data: {gaia_file}")
    
    # Plot
    plot_comparison(results, output_dir)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    
    # Summary
    chi2_improvement = results['chi2_Newton'] - results['chi2_manypath']
    if chi2_improvement > 10:
        print("✅ Many-path model shows SIGNIFICANT improvement over Newtonian")
    elif chi2_improvement > 0:
        print("⚠️  Many-path model shows modest improvement")
    else:
        print("❌ Many-path model does NOT improve fit")
    
    print(f"\nΔχ² = {chi2_improvement:.2f}")
    print(f"Output saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
