#!/usr/bin/env python3
"""
Parameter optimizer for many-path gravity model.

Multi-objective loss function includes:
1. Rotation curve χ² vs Gaia
2. Vertical lag penalty (keep disk thin, not razor-thin)
3. Outer slope penalty (keep curve flat at R > 12 kpc)

This ensures the model stays physically reasonable while fitting observations.
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
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


def pack_params(eta, R0, R1, k_an, ring_amp, lambda_ring,
                Z0_in=1.1, Z0_out=1.6, R_lag=8.0, w_lag=2.0, k_boost=0.6,
                R_gate=0.5, p_gate=4.0, p=2.0, q=3.5, M_max=3.5):
    """Pack optimization parameters into full parameter dict."""
    return dict(
        eta=eta, R0=R0, R1=R1, k_an=k_an, 
        ring_amp=ring_amp, lambda_ring=lambda_ring,
        # Radially-modulated anisotropy
        Z0_in=Z0_in, Z0_out=Z0_out, R_lag=R_lag, w_lag=w_lag, k_boost=k_boost,
        # Fixed/legacy
        Z0=1.0,  # legacy, not used with modulated version
        R_gate=R_gate, p_gate=p_gate, p=p, q=q, M_max=M_max
    )


def loss_manypath(params, src_pos, src_mass, obs_curve, R_grid, 
                  eps=0.05, batch_size=200_000, verbose=False):
    """
    Multi-objective loss function for many-path model.
    
    Components:
    1. χ² vs Gaia rotation curve
    2. Vertical lag penalty (should be ~15 ± 5 km/s at z=0.5 kpc)
    3. Outer slope penalty (keep curve flat for R > 12 kpc)
    
    Returns:
        total_loss: scalar
        components: dict with individual loss terms
    """
    # 1. Rotation curve in-plane
    vM, _ = rotation_curve(src_pos, src_mass, xp_array(R_grid), z=0.0, eps=eps,
                          params=params, use_multiplier=True, batch_size=batch_size)
    vM = to_cpu(vM)
    
    # χ² vs Gaia medians/SEM at observed radii
    R_obs = obs_curve.R_kpc.values
    v_obs = obs_curve.v_phi_median.values
    err = np.maximum(1.0, obs_curve.v_phi_sem.values)  # floor to avoid tiny SEM explosions
    vM_interp = np.interp(R_obs, R_grid, vM)
    chi2_rot = np.sum(((v_obs - vM_interp) / err)**2)
    
    # 2. Vertical lag at a few radii: target ≈ 15 ± 5 km/s slower at z=0.5 kpc
    # Test at ~8 points from 6-14 kpc
    test_Rs = np.clip(R_obs[::max(1, len(R_obs)//8)], 6, 14)
    v0, _ = rotation_curve(src_pos, src_mass, xp_array(test_Rs), z=0.0, eps=eps,
                          params=params, use_multiplier=True, batch_size=batch_size)
    vZ, _ = rotation_curve(src_pos, src_mass, xp_array(test_Rs), z=0.5, eps=eps,
                          params=params, use_multiplier=True, batch_size=batch_size)
    v0, vZ = to_cpu(v0), to_cpu(vZ)
    lag = v0 - vZ  # should be ~ +10..20 km/s
    chi2_lag = np.sum(((lag - 15.0) / 5.0)**2)
    
    # 3. Outer slope penalty to keep curve roughly flat for R > 12 kpc
    mask = (R_grid >= 12.0)
    if mask.sum() > 1:
        slope = np.gradient(vM[mask], R_grid[mask])
        chi2_slope = np.sum((slope / 2.0)**2)  # penalize |dv/dR| > ~2 km/s/kpc
    else:
        chi2_slope = 0.0
    
    # Total with tunable weights
    w_rot = 1.0
    w_lag = 0.8  # Increased from 0.5 to make 15±5 km/s a sharper target
    w_slope = 2.0
    
    total_loss = w_rot * chi2_rot + w_lag * chi2_lag + w_slope * chi2_slope
    
    components = {
        'total': total_loss,
        'chi2_rot': chi2_rot,
        'chi2_lag': chi2_lag,
        'chi2_slope': chi2_slope,
        'lag_mean': np.mean(lag),
        'lag_std': np.std(lag),
    }
    
    if verbose:
        print(f"  Loss components:")
        print(f"    Rotation χ²:  {chi2_rot:.2f}")
        print(f"    Vertical lag: {chi2_lag:.2f} (mean lag: {np.mean(lag):.1f} ± {np.std(lag):.1f} km/s)")
        print(f"    Outer slope:  {chi2_slope:.2f}")
        print(f"    TOTAL:        {total_loss:.2f}")
    
    return total_loss, components


def random_narrow_search(src_pos, src_mass, obs_curve, R_grid, 
                        iters=4, samples=200, seed=0, 
                        eps=0.05, batch_size=200_000):
    """
    Random search with iterative box narrowing.
    
    Starts with broad parameter ranges, evaluates random samples,
    keeps top performers, narrows box around them, repeat.
    """
    rng = np.random.default_rng(seed)
    
    # Initial box - Family A (moderate anisotropy, safer outer curve)
    # Tightened to hit 10-20 km/s vertical lag without overshoot
    box = {
        'eta':          (0.32, 0.44),
        'R0':           (4.0, 6.0),
        'R1':           (60.0, 90.0),
        'k_an':         (1.0, 1.8),
        'ring_amp':     (0.03, 0.12),
        'lambda_ring':  (35.0, 55.0),
        # Radially-modulated anisotropy parameters
        'Z0_in':        (1.0, 1.3),
        'Z0_out':       (1.5, 2.0),
        'R_lag':        (7.0, 9.0),
        'w_lag':        (1.5, 2.5),
        'k_boost':      (0.4, 0.8),
    }
    
    print(f"\n{'='*70}")
    print("PARAMETER OPTIMIZATION - Random Narrow Search")
    print(f"{'='*70}\n")
    print(f"Iterations: {iters}")
    print(f"Samples per iteration: {samples}")
    print(f"Initial parameter ranges:")
    for name, (lo, hi) in box.items():
        print(f"  {name:15s}: [{lo:.3f}, {hi:.3f}]")
    print()
    
    best = None
    history = []
    
    for k in range(iters):
        print(f"\n--- Iteration {k+1}/{iters} ---")
        print(f"Current box:")
        for name, (lo, hi) in box.items():
            print(f"  {name:15s}: [{lo:.3f}, {hi:.3f}]")
        print()
        
        cand = []
        for i in range(samples):
            # Sample parameters
            p = {name: rng.uniform(*box[name]) for name in box}
            params = pack_params(**p)
            
            # Evaluate loss
            try:
                L, components = loss_manypath(params, src_pos, src_mass, obs_curve, 
                                             R_grid, eps=eps, batch_size=batch_size, 
                                             verbose=False)
                cand.append((L, p, components))
            except Exception as e:
                print(f"  Sample {i}: Failed with {e}")
                continue
            
            if (i + 1) % max(1, samples // 10) == 0:
                print(f"  Evaluated {i+1}/{samples} samples...")
        
        if not cand:
            print("  ERROR: No valid candidates found!")
            break
        
        # Sort by loss
        cand.sort(key=lambda x: x[0])
        
        # Keep top 10%
        n_elite = max(10, samples // 10)
        elite = cand[:n_elite]
        
        # Best of this iteration
        best_iter = elite[0]
        history.append(best_iter)
        
        print(f"\n  Best in iteration {k+1}:")
        print(f"    Loss: {best_iter[0]:.2f}")
        print(f"    Rotation χ²: {best_iter[2]['chi2_rot']:.2f}")
        print(f"    Vertical lag: {best_iter[2]['lag_mean']:.1f} ± {best_iter[2]['lag_std']:.1f} km/s")
        print(f"    Params: {best_iter[1]}")
        
        # Shrink box around elite
        for name in box:
            vals = [p[name] for _, p, _ in elite]
            lo = min(vals)
            hi = max(vals)
            # Pad 20% around the elite range
            span = hi - lo
            pad = 0.2 * span if span > 0 else 0.1 * (box[name][1] - box[name][0])
            box[name] = (max(box[name][0], lo - pad), 
                        min(box[name][1], hi + pad))
        
        best = best_iter
    
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}\n")
    print(f"Final best loss: {best[0]:.2f}")
    print(f"Final best parameters:")
    for name, val in best[1].items():
        print(f"  {name:15s}: {val:.4f}")
    print()
    
    return best, history


def quick_param_tweak():
    """
    Quick parameter tweaks - Family A center values.
    
    Targets:
    - Rotation curve χ² < 1000
    - Vertical lag ~15 km/s at z=0.5 kpc
    - Flat outer curve (R > 12 kpc)
    
    Changes from v1 tweaks:
    - Stronger planar preference (Z0_in lower, k_an higher)
    - Radially modulated to avoid outer overshoot
    - Lower eta to compensate for stronger anisotropy
    """
    tweaked = {
        'eta': 0.39,          # Keep balanced for rotation curve
        'R_gate': 0.5,        # Keep solar system safe
        'p_gate': 4.0,        # Keep
        'R0': 5.0,            # Keep
        'p': 2.0,             # Keep
        'R1': 70.0,           # Keep hard saturation
        'q': 3.5,             # Keep hard saturation
        'k_an': 1.4,          # Strong anisotropy
        'ring_amp': 0.07,     # Modest ring contribution
        'lambda_ring': 42.0,  # Keep
        'M_max': 3.3,         # Keep
        # Radially-modulated anisotropy - FINAL: targeting 13-15 km/s lag
        'Z0_in': 1.02,        # Slightly stronger (lower = stronger)
        'Z0_out': 1.72,       # Keep
        'R_lag': 8.0,         # Center at solar circle
        'w_lag': 1.9,         # Keep
        'k_boost': 0.75,      # Increased from 0.68, but conservative
    }
    return tweaked


def plot_optimization_results(history, output_dir: Path):
    """Plot optimization convergence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iters = np.arange(1, len(history) + 1)
    losses = [h[0] for h in history]
    chi2_rots = [h[2]['chi2_rot'] for h in history]
    chi2_lags = [h[2]['chi2_lag'] for h in history]
    chi2_slopes = [h[2]['chi2_slope'] for h in history]
    
    # Plot 1: Total loss
    ax = axes[0, 0]
    ax.plot(iters, losses, 'o-', linewidth=2, markersize=10)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Plot 2: Loss components
    ax = axes[0, 1]
    ax.plot(iters, chi2_rots, 'o-', label='Rotation χ²', linewidth=2)
    ax.plot(iters, chi2_lags, 's-', label='Vertical lag', linewidth=2)
    ax.plot(iters, chi2_slopes, '^-', label='Outer slope', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Component Loss', fontsize=12)
    ax.set_title('Loss Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Key parameters evolution
    ax = axes[1, 0]
    etas = [h[1]['eta'] for h in history]
    R0s = [h[1]['R0'] for h in history]
    Z0s = [h[1]['Z0'] for h in history]
    ax.plot(iters, etas, 'o-', label='eta', linewidth=2)
    ax.plot(iters, np.array(R0s) / 5, 's-', label='R0 / 5', linewidth=2)
    ax.plot(iters, Z0s, '^-', label='Z0', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Key Parameter Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Plot 4: Vertical lag evolution
    ax = axes[1, 1]
    lag_means = [h[2]['lag_mean'] for h in history]
    lag_stds = [h[2]['lag_std'] for h in history]
    ax.errorbar(iters, lag_means, yerr=lag_stds, fmt='o-', 
                linewidth=2, markersize=10, capsize=5)
    ax.axhline(15, color='green', linestyle='--', alpha=0.5, 
              linewidth=2, label='Target: 15 km/s')
    ax.fill_between(iters, 10, 20, alpha=0.2, color='green', 
                    label='Acceptable range')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Vertical Lag (km/s)', fontsize=12)
    ax.set_title('Disk Thickness Constraint', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "optimization_convergence.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved optimization plot: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Optimize many-path model parameters")
    parser.add_argument("--mode", type=str, default="tweak", 
                       choices=["tweak", "optimize"],
                       help="'tweak' = quick manual adjustments, 'optimize' = full search")
    parser.add_argument("--n_sources", type=int, default=100000, 
                       help="Number of disk particles")
    parser.add_argument("--n_bulge", type=int, default=20000, 
                       help="Number of bulge particles")
    parser.add_argument("--use_bulge", type=int, default=1, 
                       help="Include bulge (1=yes, 0=no)")
    parser.add_argument("--batch_size", type=int, default=50000, 
                       help="Batch size for computation")
    parser.add_argument("--gpu", type=int, default=0, 
                       help="Use GPU if available")
    parser.add_argument("--iters", type=int, default=4, 
                       help="Optimization iterations")
    parser.add_argument("--samples", type=int, default=200, 
                       help="Samples per iteration")
    parser.add_argument("--output_dir", type=str, default="results/optimization",
                       help="Output directory")
    args = parser.parse_args()
    
    # Load Gaia data
    from gaia_comparison import load_gaia_data, compute_observed_rotation_curve
    
    df_gaia = load_gaia_data()
    obs_curve = compute_observed_rotation_curve(df_gaia)
    
    # Sample source distribution
    print(f"\nSampling {args.n_sources} disk particles...")
    disk_pos, m_disk = sample_exponential_disk(
        args.n_sources, M_disk=5e10, R_d=2.6, z_d=0.3, R_max=30.0, seed=42
    )
    
    if args.use_bulge:
        print(f"Sampling {args.n_bulge} bulge particles...")
        bulge_pos, m_bulge = sample_hernquist_bulge(
            args.n_bulge, M_bulge=1e10, a=0.7, seed=123
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
    
    # Target grid
    R_grid = np.linspace(5, 15, 60)
    
    if args.mode == "tweak":
        print("\n=== QUICK PARAMETER TWEAK MODE ===\n")
        print("Testing manually adjusted parameters to fix overshoot...")
        
        # Get tweaked params
        params_tweaked = quick_param_tweak()
        
        print("Tweaked parameters:")
        for k, v in params_tweaked.items():
            print(f"  {k:15s}: {v}")
        print()
        
        # Evaluate
        loss, components = loss_manypath(
            params_tweaked, src_pos, src_mass, obs_curve, R_grid,
            eps=0.05, batch_size=args.batch_size, verbose=True
        )
        
        # Save
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "tweaked_params.txt", 'w', encoding='utf-8') as f:
            f.write("Tweaked Parameters\n")
            f.write("==================\n\n")
            for k, v in params_tweaked.items():
                f.write(f"{k:15s}: {v}\n")
            f.write(f"\nTotal Loss: {loss:.2f}\n")
            f.write(f"Rotation chi2: {components['chi2_rot']:.2f}\n")
            f.write(f"Vertical lag: {components['lag_mean']:.1f} +/- {components['lag_std']:.1f} km/s\n")
        
        print(f"\n✓ Saved tweaked parameters to {output_dir / 'tweaked_params.txt'}")
    
    elif args.mode == "optimize":
        print("\n=== FULL OPTIMIZATION MODE ===\n")
        
        # Run optimization
        best, history = random_narrow_search(
            src_pos, src_mass, obs_curve, R_grid,
            iters=args.iters, samples=args.samples, seed=42,
            eps=0.05, batch_size=args.batch_size
        )
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best params
        with open(output_dir / "optimized_params.txt", 'w', encoding='utf-8') as f:
            f.write("Optimized Parameters\n")
            f.write("====================\n\n")
            for k, v in best[1].items():
                f.write(f"{k:15s}: {v:.6f}\n")
            f.write(f"\nFinal Loss: {best[0]:.2f}\n")
            f.write(f"Rotation chi2: {best[2]['chi2_rot']:.2f}\n")
            f.write(f"Vertical lag: {best[2]['lag_mean']:.1f} +/- {best[2]['lag_std']:.1f} km/s\n")
        
        # Plot convergence
        plot_optimization_results(history, output_dir)
        
        print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    main()
