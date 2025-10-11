#!/usr/bin/env python3
"""
Conservative potential-based evaluation of many-path gravity.

Instead of multiplying accelerations, we define a scalar potential:
    Φ(r) = -G ∫ ρ(r') × (1 + M(d, geometry)) / d  d³r'

Then compute accelerations as a = -∇Φ via finite differences.

This guarantees curl-free field (∇ × a = 0) and proper energy conservation.
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from toy_many_path_gravity import (
    sample_exponential_disk, sample_hernquist_bulge,
    many_path_multiplier, default_params, to_cpu, xp_array, xp_zeros, G
)

try:
    import cupy as cp
    _USING_CUPY = True
except Exception:
    import numpy as cp
    _USING_CUPY = False


def compute_potential_grid(src_pos, src_mass, R_grid, z_grid, 
                           params=None, eps=0.05, batch_size=100_000):
    """
    Compute gravitational potential Φ(R, z) on a 2D grid.
    
    Args:
        src_pos: [N_src, 3] source positions
        src_mass: [N_src] source masses (or scalar)
        R_grid: [N_R] radial grid points (kpc)
        z_grid: [N_z] vertical grid points (kpc)
        params: parameter dict for many-path multiplier
        eps: softening length (kpc)
        batch_size: batch size for source summation
    
    Returns:
        Phi: [N_R, N_z] potential grid (km/s)²
    """
    N_R = len(R_grid)
    N_z = len(z_grid)
    N_src = src_pos.shape[0]
    
    # Initialize potential grid
    Phi = xp_zeros((N_R, N_z), dtype=cp.float64)
    
    # Source cylindrical coordinates
    Rs = cp.sqrt(src_pos[:, 0]**2 + src_pos[:, 1]**2)
    zs = src_pos[:, 2]
    
    # Convert masses to array
    if isinstance(src_mass, (float, int)):
        src_mass_arr = xp_zeros(N_src, dtype=cp.float64) + float(src_mass)
    else:
        src_mass_arr = src_mass.astype(cp.float64)
    
    eps2 = eps * eps
    
    # Loop over grid points
    for i, R in enumerate(R_grid):
        for j, z in enumerate(z_grid):
            # Target position (on x-axis, at height z)
            tgt = xp_array([[R, 0.0, z]], dtype=cp.float64)
            
            phi_point = 0.0
            
            # Sum over sources in batches
            for i0 in range(0, N_src, batch_size):
                i1 = min(i0 + batch_size, N_src)
                s = src_pos[i0:i1]          # [B, 3]
                m = src_mass_arr[i0:i1]     # [B]
                Rs_b = Rs[i0:i1]            # [B]
                zs_b = zs[i0:i1]            # [B]
                
                # Distance from target to sources
                dvec = tgt[None, :, :] - s[:, None, :]  # [B, 1, 3]
                r2 = (dvec**2).sum(axis=2) + eps2       # [B, 1]
                r = cp.sqrt(r2)                         # [B, 1]
                
                # Many-path multiplier
                if params is not None:
                    Rt = xp_array([R])
                    zt = xp_array([z])
                    M = many_path_multiplier(r[:, 0], Rs_b, zs_b, Rt, zt, params)  # [B, 1]
                    factor = 1.0 + M[:, 0]
                else:
                    factor = 1.0
                
                # Potential contribution: -G m (1+M) / r
                phi_point += cp.sum(-G * m * factor / r[:, 0])
                
                # Clean up GPU memory
                if _USING_CUPY:
                    del dvec, r2, r
                    cp._default_memory_pool.free_all_blocks()
            
            Phi[i, j] = phi_point
    
    return Phi


def gradient_2d(Phi, R_grid, z_grid):
    """
    Compute 2D gradient of potential: (∂Φ/∂R, ∂Φ/∂z).
    
    Uses second-order central differences in interior,
    forward/backward at boundaries.
    
    Returns:
        a_R: [N_R, N_z] radial acceleration
        a_z: [N_R, N_z] vertical acceleration
    """
    dR = R_grid[1] - R_grid[0]  # Assume uniform spacing
    dz = z_grid[1] - z_grid[0]
    
    # Radial gradient
    a_R = -np.gradient(Phi, dR, axis=0)
    
    # Vertical gradient
    a_z = -np.gradient(Phi, dz, axis=1)
    
    return a_R, a_z


def rotation_curve_from_potential(src_pos, src_mass, R_vals, z=0.0,
                                  params=None, eps=0.05, batch_size=100_000,
                                  dR=0.05):
    """
    Compute rotation curve from potential via finite differences.
    
    Args:
        R_vals: radii to evaluate (kpc)
        z: height above plane (kpc)
        dR: step size for finite difference (kpc)
    
    Returns:
        v_c: [N_R] circular velocities (km/s)
        a_R: [N_R] radial accelerations (km/s)²/kpc
    """
    # Compute potential at R and R+dR
    N_R = len(R_vals)
    R_grid_extended = np.concatenate([R_vals, R_vals + dR])
    z_grid = np.array([z])
    
    Phi_grid = compute_potential_grid(src_pos, src_mass, R_grid_extended, z_grid,
                                     params=params, eps=eps, batch_size=batch_size)
    Phi_grid = to_cpu(Phi_grid)
    
    # Finite difference: a_R = -∂Φ/∂R
    Phi_at_R = Phi_grid[:N_R, 0]
    Phi_at_R_plus = Phi_grid[N_R:, 0]
    a_R = -(Phi_at_R_plus - Phi_at_R) / dR
    
    # Circular velocity: v_c = sqrt(R × |a_R|)
    v_c = np.sqrt(np.maximum(0.0, R_vals * (-a_R)))
    
    return v_c, a_R


def check_curl_free(Phi, R_grid, z_grid):
    """
    Verify that field is curl-free: ∇ × a ≈ 0.
    
    For axisymmetric field in cylindrical coords (R, φ, z):
        (∇ × a)_φ = ∂a_z/∂R - ∂a_R/∂z
    
    Returns:
        curl_phi: [N_R, N_z] curl component (should be ~ 0)
        max_curl: maximum absolute curl value
    """
    # Convert to numpy for gradient
    Phi_np = to_cpu(Phi) if _USING_CUPY else Phi
    
    # Compute accelerations
    a_R, a_z = gradient_2d(Phi_np, R_grid, z_grid)
    
    # Curl (φ component in cylindrical coords)
    dR = R_grid[1] - R_grid[0]
    dz = z_grid[1] - z_grid[0]
    
    # ∂a_z/∂R
    da_z_dR = np.gradient(a_z, dR, axis=0)
    
    # ∂a_R/∂z
    da_R_dz = np.gradient(a_R, dz, axis=1)
    
    # Curl
    curl_phi = da_z_dR - da_R_dz
    
    max_curl = np.max(np.abs(curl_phi))
    mean_accel = np.mean(np.sqrt(a_R**2 + a_z**2))
    
    # Relative curl
    rel_curl = max_curl / mean_accel if mean_accel > 0 else 0.0
    
    return curl_phi, max_curl, rel_curl


def compare_acceleration_vs_potential(src_pos, src_mass, R_vals, z=0.0,
                                     params=None, eps=0.05, batch_size=100_000):
    """
    Compare acceleration-multiplier method vs potential-gradient method.
    
    Returns:
        results: dict with both methods' outputs and curl check
    """
    from toy_many_path_gravity import rotation_curve as accel_rotation_curve
    
    # Method 1: Acceleration multiplier (current)
    R_vals_gpu = xp_array(R_vals, dtype=cp.float64)
    v_accel, a_accel = accel_rotation_curve(src_pos, src_mass, R_vals_gpu, z=z,
                                           eps=eps, params=params, 
                                           use_multiplier=(params is not None),
                                           batch_size=batch_size)
    v_accel = to_cpu(v_accel)
    a_accel = to_cpu(a_accel)
    
    # Method 2: Potential gradient (conservative)
    v_pot, a_pot = rotation_curve_from_potential(src_pos, src_mass, R_vals, z=z,
                                                params=params, eps=eps, 
                                                batch_size=batch_size)
    
    # Check curl on a small grid
    R_grid_check = np.linspace(R_vals.min(), R_vals.max(), 20)
    z_grid_check = np.linspace(-0.5, 0.5, 11)
    Phi_check = compute_potential_grid(src_pos, src_mass, R_grid_check, z_grid_check,
                                      params=params, eps=eps, batch_size=batch_size)
    curl_field, max_curl, rel_curl = check_curl_free(Phi_check, R_grid_check, z_grid_check)
    
    # Compute differences
    delta_v = v_accel - v_pot
    rel_diff_v = np.abs(delta_v) / v_pot
    
    results = {
        'R': R_vals,
        'v_accel': v_accel,
        'v_potential': v_pot,
        'a_accel': a_accel,
        'a_potential': a_pot,
        'delta_v': delta_v,
        'rel_diff_v': rel_diff_v,
        'max_curl': max_curl,
        'rel_curl': rel_curl,
        'mean_v': np.mean([v_accel, v_pot], axis=0),
    }
    
    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" CONSERVATIVE POTENTIAL-BASED GRAVITY TEST ")
    print("="*70 + "\n")
    
    # Sample test distribution
    print("Sampling test distribution...")
    disk_pos, m_disk = sample_exponential_disk(50000, seed=42)
    bulge_pos, m_bulge = sample_hernquist_bulge(10000, seed=123)
    
    src_pos = cp.concatenate([disk_pos, bulge_pos], axis=0)
    src_mass = cp.concatenate([
        xp_zeros(disk_pos.shape[0]) + m_disk,
        xp_zeros(bulge_pos.shape[0]) + m_bulge
    ])
    
    print(f"✓ Total particles: {src_pos.shape[0]}\n")
    
    # Get parameters
    params = default_params()
    
    # Test radii
    R_vals = np.linspace(5, 15, 20)
    
    # Compare methods
    print("Comparing acceleration-multiplier vs potential-gradient...\n")
    results = compare_acceleration_vs_potential(src_pos, src_mass, R_vals, z=0.0,
                                               params=params, eps=0.05, batch_size=50000)
    
    # Print comparison
    print(f"{'R (kpc)':>8} {'v_accel':>10} {'v_pot':>10} {'Δv':>10} {'rel %':>8}")
    print("-" * 55)
    for i, R in enumerate(results['R']):
        v_a = results['v_accel'][i]
        v_p = results['v_potential'][i]
        dv = results['delta_v'][i]
        rel = 100 * results['rel_diff_v'][i]
        print(f"{R:8.2f} {v_a:10.2f} {v_p:10.2f} {dv:10.3f} {rel:8.3f}")
    
    print(f"\n{'='*70}")
    print(" CONSERVATIVITY CHECK ")
    print(f"{'='*70}\n")
    
    print(f"Maximum curl: {results['max_curl']:.3e}")
    print(f"Relative curl: {results['rel_curl']:.3e}")
    
    if results['rel_curl'] < 1e-3:
        print("\n✅ Field is conservative (curl ≈ 0)")
    elif results['rel_curl'] < 1e-2:
        print("\n⚠️  Field is approximately conservative")
    else:
        print("\n❌ Field shows significant curl")
    
    print(f"\nMean velocity difference: {np.mean(np.abs(results['delta_v'])):.3f} km/s")
    print(f"Mean relative difference: {100*np.mean(results['rel_diff_v']):.3f}%")
    
    if np.mean(results['rel_diff_v']) < 0.01:
        print("\n✅ Methods agree within 1%")
    elif np.mean(results['rel_diff_v']) < 0.05:
        print("\n⚠️  Methods agree within 5%")
    else:
        print("\n❌ Methods show significant disagreement")
