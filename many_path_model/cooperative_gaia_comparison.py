#!/usr/bin/env python3
"""
Cooperative Response Model - Gaia Comparison

Run the cooperative response (density-dependent G_eff) model on the SAME
Gaia benchmark as the many-path model for fair apples-to-apples comparison.

This provides the critical Step 3 comparison to guard against cherry-picking.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gaia_comparison import load_gaia_data, compute_observed_rotation_curve
from toy_many_path_gravity import (
    sample_exponential_disk, sample_hernquist_bulge,
    to_cpu, xp_array, xp_zeros, G
)

try:
    import cupy as cp
    _USING_CUPY = True
except Exception:
    import numpy as cp
    _USING_CUPY = False

ROOT = Path(__file__).resolve().parents[1]


def cooperative_response_params():
    """
    Default cooperative response parameters.
    
    Model: G_eff(ρ) = G × [1 + α × (ρ/ρ_solar)^β × tanh(ρ/ρ_threshold)]
    
    Based on prior cluster+galaxy fitting (need to verify these are current best-fit).
    """
    return {
        'alpha': 0.5,            # Amplitude of density boost
        'beta': 0.3,             # Power-law index
        'rho_solar': 0.1,        # Solar neighborhood density (Msun/pc³)
        'rho_threshold': 0.01,   # Density threshold for turn-on (Msun/pc³)
    }


def compute_density_field(src_pos, src_mass, tgt_pos, h_smooth=0.5):
    """
    Estimate local density at target positions using SPH-like kernel.
    
    Args:
        src_pos: [N_src, 3] source positions (kpc)
        src_mass: [N_src] source masses (Msun) or scalar
        tgt_pos: [N_tgt, 3] target positions (kpc)
        h_smooth: smoothing length (kpc)
    
    Returns:
        rho: [N_tgt] density estimates (Msun/kpc³)
    """
    N_src = src_pos.shape[0]
    N_tgt = tgt_pos.shape[0]
    
    if isinstance(src_mass, (float, int)):
        src_mass_arr = xp_zeros(N_src, dtype=cp.float64) + float(src_mass)
    else:
        src_mass_arr = src_mass.astype(cp.float64)
    
    rho = xp_zeros(N_tgt, dtype=cp.float64)
    
    # Cubic spline kernel normalization in 3D
    h3 = h_smooth**3
    norm = 1.0 / (np.pi * h3)
    
    # Batch process for memory efficiency
    batch_size = 10000
    for i0 in range(0, N_src, batch_size):
        i1 = min(i0 + batch_size, N_src)
        s = src_pos[i0:i1]
        m = src_mass_arr[i0:i1]
        
        # Distance from each target to these sources
        dvec = tgt_pos[:, None, :] - s[None, :, :]  # [N_tgt, batch, 3]
        r = cp.sqrt((dvec**2).sum(axis=2))          # [N_tgt, batch]
        q = r / h_smooth
        
        # Cubic spline kernel
        W = xp_zeros(q.shape, dtype=cp.float64)
        mask1 = q < 1.0
        mask2 = (q >= 1.0) & (q < 2.0)
        
        W[mask1] = 1.0 - 1.5*q[mask1]**2 + 0.75*q[mask1]**3
        W[mask2] = 0.25 * (2.0 - q[mask2])**3
        
        W *= norm
        
        # Accumulate density
        rho += (W * m[None, :]).sum(axis=1)
        
        if _USING_CUPY:
            del dvec, r, q, W
            cp._default_memory_pool.free_all_blocks()
    
    return rho


def G_effective_cooperative(rho, params):
    """
    Compute effective G from cooperative response model.
    
    G_eff(ρ) = G × [1 + α × (ρ/ρ_solar)^β × tanh(ρ/ρ_threshold)]
    
    Args:
        rho: density (Msun/kpc³)
        params: parameter dict with alpha, beta, rho_solar, rho_threshold
    
    Returns:
        G_eff: effective gravitational constant
    """
    alpha = params['alpha']
    beta = params['beta']
    rho_solar = params['rho_solar']
    rho_threshold = params['rho_threshold']
    
    # Convert Msun/kpc³ to Msun/pc³ for comparison with thresholds
    rho_pc3 = rho / 1e9
    
    # Density-dependent boost
    rho_ratio = rho_pc3 / rho_solar
    gate = cp.tanh(rho_pc3 / rho_threshold)
    
    boost = 1.0 + alpha * (rho_ratio**beta) * gate
    
    return G * boost


def rotation_curve_cooperative(src_pos, src_mass, R_vals, z=0.0, 
                               params=None, h_smooth=0.5, eps=0.05, 
                               batch_size=100_000):
    """
    Compute rotation curve with cooperative response (density-dependent G).
    
    Args:
        src_pos: [N_src, 3] source positions
        src_mass: source masses
        R_vals: radii to evaluate (can be numpy or cupy)
        z: height above plane
        params: cooperative response parameters
        h_smooth: density smoothing scale
        eps: force softening
        batch_size: batch size for force calculation
    
    Returns:
        v_c: circular velocities
        a_R: radial accelerations
    """
    # Ensure R_vals is cupy array
    if isinstance(R_vals, (list, tuple)):
        R_vals = xp_array(R_vals, dtype=cp.float64)
    elif isinstance(R_vals, np.ndarray):
        R_vals = xp_array(R_vals, dtype=cp.float64)
    
    N_tgt = len(R_vals)
    N_src = src_pos.shape[0]
    
    # Target positions
    tgt_pos = xp_zeros((N_tgt, 3), dtype=cp.float64)
    tgt_pos[:, 0] = R_vals
    tgt_pos[:, 1] = 0.0
    tgt_pos[:, 2] = z
    
    # Estimate density at target positions
    rho_tgt = compute_density_field(src_pos, src_mass, tgt_pos, h_smooth=h_smooth)
    
    # Effective G at each target
    if params is not None:
        G_eff_tgt = G_effective_cooperative(rho_tgt, params)
    else:
        G_eff_tgt = xp_zeros(N_tgt, dtype=cp.float64) + G
    
    # Compute accelerations
    acc = xp_zeros((N_tgt, 3), dtype=cp.float64)
    
    if isinstance(src_mass, (float, int)):
        src_mass_arr = xp_zeros(N_src, dtype=cp.float64) + float(src_mass)
    else:
        src_mass_arr = src_mass.astype(cp.float64)
    
    eps2 = eps * eps
    
    # Batch computation
    for i0 in range(0, N_src, batch_size):
        i1 = min(i0 + batch_size, N_src)
        s = src_pos[i0:i1]
        m = src_mass_arr[i0:i1]
        
        # Pairwise differences
        dvec = tgt_pos[None, :, :] - s[:, None, :]  # [batch, N_tgt, 3]
        r2 = (dvec**2).sum(axis=2) + eps2           # [batch, N_tgt]
        inv_r3 = r2**(-1.5)
        
        # Apply effective G at each target
        # contrib[i, j] = -G_eff[j] * m[i] / r³[i,j] * dvec[i,j]
        contrib = -G_eff_tgt[None, :] * m[:, None] * inv_r3
        contrib = contrib[:, :, None] * dvec  # [batch, N_tgt, 3]
        
        acc += contrib.sum(axis=0)
        
        if _USING_CUPY:
            del dvec, r2, inv_r3, contrib
            cp._default_memory_pool.free_all_blocks()
    
    # Extract radial component
    a_R = acc[:, 0]
    
    # Circular velocity
    v_c = cp.sqrt(cp.maximum(0.0, R_vals * (-a_R)))
    
    return v_c, a_R


def compute_vertical_lag_cooperative(src_pos, src_mass, R_vals, z=0.5,
                                     params=None, h_smooth=0.5, eps=0.05,
                                     batch_size=100_000):
    """
    Compute vertical lag: v(z=0) - v(z=0.5 kpc).
    """
    v0, _ = rotation_curve_cooperative(src_pos, src_mass, R_vals, z=0.0,
                                       params=params, h_smooth=h_smooth, 
                                       eps=eps, batch_size=batch_size)
    vz, _ = rotation_curve_cooperative(src_pos, src_mass, R_vals, z=z,
                                       params=params, h_smooth=h_smooth,
                                       eps=eps, batch_size=batch_size)
    
    lag = v0 - vz
    return to_cpu(lag)


def compare_cooperative_vs_gaia(n_sources=100000, n_bulge=20000, use_bulge=True,
                               R_range=(5, 15, 50), batch_size=50000, gpu=False):
    """
    Compare cooperative response model with Gaia observations.
    
    Returns comparison metrics matching many-path model format.
    """
    print(f"\n{'='*70}")
    print("COOPERATIVE RESPONSE MODEL vs GAIA COMPARISON")
    print(f"{'='*70}\n")
    
    # Load Gaia observations (SAME as many-path)
    df_gaia = load_gaia_data()
    obs_curve = compute_observed_rotation_curve(df_gaia)
    
    print("Observed rotation curve from Gaia:")
    print(obs_curve.to_string(index=False))
    print()
    
    # Sample source distribution (SAME as many-path)
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
    
    # Get cooperative response parameters
    params = cooperative_response_params()
    
    print("Cooperative Response Parameters:")
    for k, v in params.items():
        print(f"  {k:15s}: {v}")
    print()
    
    # Compute curves
    print("Computing Newtonian rotation curve...")
    vN, _ = rotation_curve_cooperative(src_pos, src_mass, R_vals, z=0.0,
                                      params=None, h_smooth=0.5, eps=0.05,
                                      batch_size=batch_size)
    
    print("Computing cooperative response rotation curve...")
    vC, _ = rotation_curve_cooperative(src_pos, src_mass, R_vals, z=0.0,
                                      params=params, h_smooth=0.5, eps=0.05,
                                      batch_size=batch_size)
    
    # Compute vertical lag
    print("Computing vertical lag...")
    test_R = to_cpu(R_vals[::max(1, len(R_vals)//8)])
    lag_coop = compute_vertical_lag_cooperative(src_pos, src_mass, test_R, z=0.5,
                                                params=params, batch_size=batch_size)
    
    # Convert to numpy
    R_vals_np = to_cpu(R_vals)
    vN_np = to_cpu(vN)
    vC_np = to_cpu(vC)
    
    # Compute residuals and chi2
    vN_interp = np.interp(obs_curve.R_kpc, R_vals_np, vN_np)
    vC_interp = np.interp(obs_curve.R_kpc, R_vals_np, vC_np)
    
    residual_N = obs_curve.v_phi_median.values - vN_interp
    residual_C = obs_curve.v_phi_median.values - vC_interp
    
    err = np.maximum(1.0, obs_curve.v_phi_sem.values)
    chi2_N = np.sum((residual_N / err)**2)
    chi2_C = np.sum((residual_C / err)**2)
    
    # Outer slope penalty
    mask_outer = R_vals_np >= 12.0
    if mask_outer.sum() > 1:
        slope_C = np.gradient(vC_np[mask_outer], R_vals_np[mask_outer])
        chi2_slope_C = np.sum((slope_C / 2.0)**2)
    else:
        chi2_slope_C = 0.0
    
    # Vertical lag penalty
    chi2_lag_C = np.sum(((lag_coop - 15.0) / 5.0)**2)
    
    # Total loss (matching many-path weights)
    total_loss_C = 1.0 * chi2_C + 0.8 * chi2_lag_C + 2.0 * chi2_slope_C
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"Newtonian χ²:     {chi2_N:.2f}")
    print(f"Cooperative χ²:   {chi2_C:.2f}")
    print(f"Improvement:      {chi2_N - chi2_C:.2f} (lower is better)\n")
    
    print(f"Vertical lag:     {np.mean(lag_coop):.1f} ± {np.std(lag_coop):.1f} km/s")
    print(f"Outer slope:      {chi2_slope_C:.2f}")
    print(f"Total loss:       {total_loss_C:.2f}\n")
    
    # Print comparison table
    print(f"{'R(kpc)':>7} {'Gaia':>8} {'±':>8} {'Newton':>8} {'Coop':>8} {'ΔN':>8} {'ΔC':>8}")
    print("-" * 65)
    for i, row in obs_curve.iterrows():
        R = row.R_kpc
        v_obs = row.v_phi_median
        err_obs = row.v_phi_sem
        v_N = np.interp(R, R_vals_np, vN_np)
        v_C = np.interp(R, R_vals_np, vC_np)
        dN = v_obs - v_N
        dC = v_obs - v_C
        print(f"{R:7.2f} {v_obs:8.1f} {err_obs:8.2f} {v_N:8.1f} {v_C:8.1f} {dN:8.1f} {dC:8.1f}")
    
    return {
        'R_vals': R_vals_np,
        'v_Newton': vN_np,
        'v_cooperative': vC_np,
        'gaia_obs': obs_curve,
        'chi2_Newton': chi2_N,
        'chi2_cooperative': chi2_C,
        'chi2_slope': chi2_slope_C,
        'chi2_lag': chi2_lag_C,
        'total_loss': total_loss_C,
        'lag_mean': np.mean(lag_coop),
        'lag_std': np.std(lag_coop),
        'params': params,
        'n_params': 4,  # alpha, beta, rho_solar, rho_threshold
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cooperative response model Gaia comparison")
    parser.add_argument("--n_sources", type=int, default=100000)
    parser.add_argument("--n_bulge", type=int, default=20000)
    parser.add_argument("--use_bulge", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/cooperative_comparison")
    args = parser.parse_args()
    
    # Run comparison
    results = compare_cooperative_vs_gaia(
        n_sources=args.n_sources,
        n_bulge=args.n_bulge,
        use_bulge=bool(args.use_bulge),
        R_range=(5, 15, 50),
        batch_size=args.batch_size,
        gpu=bool(args.gpu)
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame({
        'R_kpc': results['R_vals'],
        'v_Newton_km_s': results['v_Newton'],
        'v_cooperative_km_s': results['v_cooperative'],
        'boost_factor': results['v_cooperative'] / results['v_Newton']
    })
    
    csv_file = output_dir / "cooperative_predictions.csv"
    df_results.to_csv(csv_file, index=False)
    print(f"\n✓ Saved predictions: {csv_file}")
    
    # Save summary
    summary_file = output_dir / "comparison_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Cooperative Response Model - Gaia Comparison\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Parameters: {results['n_params']}\n")
        for k, v in results['params'].items():
            f.write(f"  {k:15s}: {v}\n")
        f.write(f"\nRotation chi2:  {results['chi2_cooperative']:.2f}\n")
        f.write(f"Vertical lag:   {results['lag_mean']:.1f} +/- {results['lag_std']:.1f} km/s\n")
        f.write(f"Outer slope:    {results['chi2_slope']:.2f}\n")
        f.write(f"Total loss:     {results['total_loss']:.2f}\n")
    
    print(f"✓ Saved summary: {summary_file}\n")


if __name__ == "__main__":
    main()
