#!/usr/bin/env python3
"""
SPARC Zero-Shot Test for Many-Path Gravity Model

This script applies the many-path gravity model with FIXED global parameters
to SPARC galaxies and evaluates how well it predicts rotation curves without
per-galaxy fitting.

Key features:
- Loads SPARC rotation curve data (_rotmod.dat files)
- Computes bulge fractions for each galaxy
- Applies many-path kernel with fixed parameters
- Supports both standard and bulge-gated kernels
- Computes metrics: APE (Absolute Percentage Error), chi-squared, success rate
- Analyzes performance by galaxy type

Usage:
    python sparc_zero_shot_test.py --sparc_dir external_data/Rotmod_LTG \
        --n_galaxies 25 --use_bulge_gate 1 --output results/sparc_zero_shot.csv
"""

import argparse
import sys
import csv
import json
from pathlib import Path
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

# Try CuPy first for GPU acceleration
try:
    import cupy as cp
    _USING_CUPY = True
except Exception:
    import numpy as cp
    _USING_CUPY = False

# Import the many-path gravity functions
# Assuming they are in the same directory
try:
    from toy_many_path_gravity import (
        compute_accel_batched, rotation_curve, default_params,
        xp_array, xp_zeros, to_cpu, G
    )
except ImportError:
    print("ERROR: Could not import toy_many_path_gravity module")
    print("Make sure toy_many_path_gravity.py is in the same directory")
    sys.exit(1)


def load_sparc_galaxy(filepath: Path) -> Dict:
    """
    Load a SPARC galaxy rotation curve file (_rotmod.dat format).
    
    Returns dict with:
        - r_kpc: array of radii
        - v_obs: observed velocities
        - v_err: velocity errors
        - v_gas, v_disk, v_bulge: baryonic velocity components
        - distance_mpc: galaxy distance
        - name: galaxy name
    """
    # Parse header for distance
    distance_mpc = None
    with open(filepath, 'r') as f:
        for line in f:
            if 'Distance' in line:
                match = re.search(r'([\d.]+)\s*Mpc', line)
                if match:
                    distance_mpc = float(match.group(1))
                break
    
    # Load data
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    data.append([float(x) for x in parts[:8]])
                except ValueError:
                    continue
    
    if not data:
        raise ValueError(f"No valid data in {filepath}")
    
    data = np.array(data)
    name = filepath.stem.replace('_rotmod', '')
    
    return {
        'name': name,
        'distance_mpc': distance_mpc,
        'r_kpc': data[:, 0],
        'v_obs': data[:, 1],
        'v_err': data[:, 2],
        'v_gas': data[:, 3],
        'v_disk': data[:, 4],
        'v_bulge': data[:, 5],
        'sb_disk': data[:, 6],
        'sb_bulge': data[:, 7],
    }


def compute_bulge_fraction(v_gas: np.ndarray, v_disk: np.ndarray, 
                          v_bulge: np.ndarray) -> np.ndarray:
    """
    Compute bulge fraction at each radius.
    
    bulge_frac = V_bulge^2 / (V_gas^2 + V_disk^2 + V_bulge^2)
    
    This represents the fractional contribution of the bulge to the
    total baryonic circular velocity squared.
    """
    v_total_sq = np.maximum(v_gas**2 + v_disk**2 + v_bulge**2, 1e-10)
    bulge_frac = v_bulge**2 / v_total_sq
    return bulge_frac


def classify_galaxy_type(galaxy: Dict) -> str:
    """
    Simple classification based on bulge prominence.
    
    Returns: 'bulge_dominated', 'intermediate', or 'disk_dominated'
    """
    bulge_frac = compute_bulge_fraction(
        galaxy['v_gas'], galaxy['v_disk'], galaxy['v_bulge']
    )
    
    # Average bulge fraction weighted by velocity
    v_total = np.sqrt(galaxy['v_gas']**2 + galaxy['v_disk']**2 + galaxy['v_bulge']**2)
    avg_bulge_frac = np.average(bulge_frac, weights=v_total)
    
    if avg_bulge_frac > 0.3:
        return 'bulge_dominated'
    elif avg_bulge_frac > 0.1:
        return 'intermediate'
    else:
        return 'disk_dominated'


def create_particle_distribution(galaxy: Dict, n_particles: int = 50000,
                                 R_max: float = 40.0) -> Tuple:
    """
    Create a simple particle distribution from SPARC velocity profiles.
    
    This is a simplified approach: we sample particles with mass proportional
    to the surface brightness and velocity contributions.
    
    Returns: (positions[N,3], masses[N])
    """
    r_kpc = galaxy['r_kpc']
    sb_disk = galaxy['sb_disk']
    sb_bulge = galaxy['sb_bulge']
    
    # Total mass proxy from surface brightness
    sb_total = sb_disk + sb_bulge
    
    # Interpolate to create sampling distribution
    # Sample more particles where there's more light
    r_sample = []
    m_sample = []
    
    # Exponential disk-like sampling based on SB profile
    for i in range(n_particles):
        # Sample radius weighted by SB and area (2πr)
        weights = sb_total * r_kpc
        weights = weights / np.sum(weights)
        r = np.random.choice(r_kpc, p=weights)
        
        # Random azimuthal angle
        phi = np.random.uniform(0, 2*np.pi)
        
        # Vertical position (thin disk approximation)
        z = np.random.normal(0, 0.3)  # 0.3 kpc scale height
        
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        
        r_sample.append([x, y, z])
        
        # Mass proportional to local SB
        sb_interp = np.interp(r, r_kpc, sb_total, left=0, right=0)
        m_sample.append(sb_interp)
    
    positions = np.array(r_sample)
    masses = np.array(m_sample)
    
    # Normalize total mass to match velocity at some fiducial radius
    # Use V^2 * R ~ GM, so M ~ V^2 * R / G
    idx_mid = len(r_kpc) // 2
    r_fid = r_kpc[idx_mid]
    v_fid = np.sqrt(galaxy['v_gas'][idx_mid]**2 + 
                    galaxy['v_disk'][idx_mid]**2 + 
                    galaxy['v_bulge'][idx_mid]**2)
    M_total = v_fid**2 * r_fid / G
    
    masses = masses / np.sum(masses) * M_total
    
    return xp_array(positions, dtype=cp.float64), xp_array(masses, dtype=cp.float64)


def predict_rotation_curve(galaxy: Dict, params: Dict, 
                           use_bulge_gate: bool = False,
                           n_particles: int = 50000) -> np.ndarray:
    """
    Predict rotation curve using many-path gravity with fixed parameters.
    
    Returns: v_pred[n_radii] in km/s
    """
    # Create particle distribution
    positions, masses = create_particle_distribution(galaxy, n_particles)
    
    # Target radii (where we have observations)
    R_vals = xp_array(galaxy['r_kpc'], dtype=cp.float64)
    
    # Bulge fractions for gating
    bulge_frac = None
    if use_bulge_gate:
        bulge_frac = xp_array(
            compute_bulge_fraction(galaxy['v_gas'], galaxy['v_disk'], galaxy['v_bulge']),
            dtype=cp.float64
        )
    
    # Compute rotation curve with many-path gravity
    v_pred, _ = rotation_curve(
        positions, masses, R_vals, z=0.0, 
        eps=0.05, params=params,
        use_multiplier=True, batch_size=50000,
        bulge_frac=bulge_frac
    )
    
    return to_cpu(v_pred)


def compute_metrics(v_obs: np.ndarray, v_pred: np.ndarray, 
                   v_err: np.ndarray) -> Dict[str, float]:
    """
    Compute performance metrics for rotation curve fit.
    
    Returns:
        - ape: Absolute Percentage Error (mean)
        - rms: RMS error
        - chi2_reduced: Reduced chi-squared
        - success: 1 if APE < 15%, 0 otherwise
    """
    # Mask out zero/negative observations
    mask = (v_obs > 0) & (v_pred > 0)
    
    if np.sum(mask) == 0:
        return {'ape': np.inf, 'rms': np.inf, 'chi2_reduced': np.inf, 'success': 0}
    
    v_obs_m = v_obs[mask]
    v_pred_m = v_pred[mask]
    v_err_m = v_err[mask]
    
    # Absolute percentage error
    ape = 100.0 * np.abs(v_pred_m - v_obs_m) / v_obs_m
    mean_ape = np.mean(ape)
    
    # RMS error
    rms = np.sqrt(np.mean((v_pred_m - v_obs_m)**2))
    
    # Chi-squared
    chi2 = np.sum(((v_pred_m - v_obs_m) / np.maximum(v_err_m, 2.0))**2)
    chi2_reduced = chi2 / len(v_obs_m) if len(v_obs_m) > 0 else np.inf
    
    # Success: APE < 15% threshold (commonly used in literature)
    success = 1 if mean_ape < 15.0 else 0
    
    return {
        'ape': mean_ape,
        'rms': rms,
        'chi2_reduced': chi2_reduced,
        'success': success,
        'n_points': int(np.sum(mask))
    }


def test_galaxy(galaxy_file: Path, params: Dict, use_bulge_gate: bool = False) -> Dict:
    """
    Test many-path model on a single galaxy.
    
    Returns dict with galaxy info and metrics.
    """
    print(f"Testing {galaxy_file.name}...")
    
    try:
        # Load galaxy data
        galaxy = load_sparc_galaxy(galaxy_file)
        
        # Classify galaxy type
        gal_type = classify_galaxy_type(galaxy)
        
        # Predict rotation curve
        v_pred = predict_rotation_curve(galaxy, params, use_bulge_gate)
        
        # Compute metrics
        metrics = compute_metrics(galaxy['v_obs'], v_pred, galaxy['v_err'])
        
        # Compute baryonic velocity for reference
        v_bar = np.sqrt(galaxy['v_gas']**2 + galaxy['v_disk']**2 + galaxy['v_bulge']**2)
        
        result = {
            'name': galaxy['name'],
            'type': gal_type,
            'distance_mpc': galaxy['distance_mpc'],
            'n_points': len(galaxy['r_kpc']),
            **metrics,
            'mean_v_obs': float(np.mean(galaxy['v_obs'])),
            'mean_v_bar': float(np.mean(v_bar)),
        }
        
        print(f"  {galaxy['name']}: APE={metrics['ape']:.1f}%, Type={gal_type}, Success={metrics['success']}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'name': galaxy_file.stem.replace('_rotmod', ''),
            'type': 'error',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="SPARC zero-shot test for many-path gravity")
    parser.add_argument('--sparc_dir', type=str, required=True,
                       help='Directory containing SPARC _rotmod.dat files')
    parser.add_argument('--n_galaxies', type=int, default=25,
                       help='Number of galaxies to test (default: 25)')
    parser.add_argument('--use_bulge_gate', type=int, default=0,
                       help='Use bulge-gated kernel (1=yes, 0=no)')
    parser.add_argument('--output', type=str, default='sparc_zero_shot_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for galaxy selection')
    parser.add_argument('--n_particles', type=int, default=50000,
                       help='Number of particles for galaxy model')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    if _USING_CUPY:
        cp.random.seed(args.seed)
    
    # Load default parameters for many-path model
    params = default_params()
    print(f"\nUsing many-path parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"\nBulge gating: {'ENABLED' if args.use_bulge_gate else 'DISABLED'}")
    
    # Find SPARC galaxy files
    sparc_dir = Path(args.sparc_dir)
    if not sparc_dir.exists():
        print(f"ERROR: SPARC directory not found: {sparc_dir}")
        sys.exit(1)
    
    galaxy_files = list(sparc_dir.glob('*_rotmod.dat'))
    if not galaxy_files:
        print(f"ERROR: No *_rotmod.dat files found in {sparc_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(galaxy_files)} SPARC galaxies")
    
    # Select random subset
    if len(galaxy_files) > args.n_galaxies:
        galaxy_files = np.random.choice(galaxy_files, args.n_galaxies, replace=False).tolist()
    
    print(f"Testing {len(galaxy_files)} galaxies\n")
    
    # Test each galaxy
    results = []
    for gal_file in galaxy_files:
        result = test_galaxy(gal_file, params, bool(args.use_bulge_gate))
        results.append(result)
    
    # Write results to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n=== Results written to {output_path} ===\n")
        
        # Print summary statistics
        valid_results = [r for r in results if 'ape' in r and np.isfinite(r['ape'])]
        
        if valid_results:
            apes = [r['ape'] for r in valid_results]
            successes = [r['success'] for r in valid_results]
            
            print(f"=== Summary Statistics ===")
            print(f"Total galaxies tested: {len(valid_results)}")
            print(f"Mean APE: {np.mean(apes):.2f}%")
            print(f"Median APE: {np.median(apes):.2f}%")
            print(f"Success rate (APE < 15%): {np.mean(successes)*100:.1f}%")
            
            # By galaxy type
            print(f"\n=== By Galaxy Type ===")
            for gtype in ['disk_dominated', 'intermediate', 'bulge_dominated']:
                type_results = [r for r in valid_results if r['type'] == gtype]
                if type_results:
                    type_apes = [r['ape'] for r in type_results]
                    type_success = [r['success'] for r in type_results]
                    print(f"{gtype}:")
                    print(f"  N = {len(type_results)}")
                    print(f"  Mean APE = {np.mean(type_apes):.2f}%")
                    print(f"  Success rate = {np.mean(type_success)*100:.1f}%")


if __name__ == '__main__':
    main()
