#!/usr/bin/env python3
"""
SPARC Stratified Zero-Shot Test with MW Baseline

This script implements the systematic research plan for validating many-path
gravity across galaxy types without falling back on dark matter or MOND.

Key features:
- Stratified sampling by morphological type
- MW-frozen baseline parameter tracking
- Type-specific filtering (disk-dominated vs bulge-dominated)
- High-performance GPU acceleration (5090 optimized)
- Comprehensive metrics tracking (APE, chi², BTFR, RAR)

Usage:
    # Stratified zero-shot (50 galaxies, 10 per type)
    python sparc_stratified_test.py --strategy stratified \
        --n_per_type 10 --types Sm,Scd,Sc,Sbc,Sb \
        --use_bulge_gate 0 --output results/stratified_standard.csv
    
    # Bulge-gate power sweep (early types only)
    python sparc_stratified_test.py --strategy filter \
        --filter_types Sbc,Sb,Sab --bulge_gate_power 2.0 \
        --output results/bulge_power_2.0.csv
    
    # Full sample test (all 175 galaxies)
    python sparc_stratified_test.py --strategy all \
        --use_bulge_gate 1 --output results/full_sample.csv
"""

import argparse
import sys
import csv
import json
from pathlib import Path
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time

# Try CuPy first for GPU acceleration
try:
    import cupy as cp
    _USING_CUPY = True
    print("GPU acceleration ENABLED (CuPy found)")
except Exception:
    import numpy as cp
    _USING_CUPY = False
    print("GPU acceleration DISABLED (CuPy not found, using NumPy)")

# Import many-path gravity functions
try:
    from toy_many_path_gravity import (
        compute_accel_batched, rotation_curve, default_params,
        xp_array, xp_zeros, to_cpu, G
    )
except ImportError:
    print("ERROR: Could not import toy_many_path_gravity module")
    sys.exit(1)


# Morphological type mappings
HUBBLE_TYPES = {
    0: 'S0', 1: 'Sa', 2: 'Sab', 3: 'Sb', 4: 'Sbc', 5: 'Sc',
    6: 'Scd', 7: 'Sd', 8: 'Sdm', 9: 'Sm', 10: 'Im', 11: 'BCD'
}

# Type groupings for analysis
TYPE_GROUPS = {
    'early': ['S0', 'Sa', 'Sab', 'Sb'],  # Bulge-dominated
    'intermediate': ['Sbc', 'Sc'],        # Mixed
    'late': ['Scd', 'Sd', 'Sdm', 'Sm', 'Im', 'BCD']  # Disk-dominated
}


@dataclass
class SPARCGalaxy:
    """SPARC galaxy data structure with morphology info."""
    name: str
    hubble_type: int
    hubble_name: str
    type_group: str  # early/intermediate/late
    distance_mpc: float
    r_kpc: np.ndarray
    v_obs: np.ndarray
    v_err: np.ndarray
    v_gas: np.ndarray
    v_disk: np.ndarray
    v_bulge: np.ndarray
    sb_disk: np.ndarray
    sb_bulge: np.ndarray
    bulge_frac: np.ndarray  # Computed B/T at each radius
    avg_bulge_frac: float   # Average B/T
    
    def to_dict(self):
        """Convert to dictionary for CSV export (excluding arrays)."""
        return {
            'name': self.name,
            'hubble_type': self.hubble_type,
            'hubble_name': self.hubble_name,
            'type_group': self.type_group,
            'distance_mpc': self.distance_mpc,
            'avg_bulge_frac': self.avg_bulge_frac,
            'n_points': len(self.r_kpc)
        }


def load_sparc_master_table(master_file: Path) -> Dict[str, Dict]:
    """
    Load SPARC master table with morphological types.
    
    Returns dict mapping galaxy names to properties.
    """
    galaxies = {}
    
    with open(master_file, 'r') as f:
        lines = f.readlines()
    
    # Find data section (after column descriptions)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('-------'):
            data_start = i + 1
            break
    
    # Parse data lines  
    for line in lines[data_start:]:
        if not line.strip() or line.startswith('#'):
            continue
        
        # Skip lines that are too short
        if len(line) < 20:
            continue
            
        # Parse fixed-width format (from MRT description)
        # Columns: Galaxy (1-11), T (12-13), D (14-19), e_D (20-24), f_D (25-26),
        #          Inc (27-30), e_Inc (31-34), L[3.6] (35-41), e_L[3.6] (42-48),
        #          Reff (49-53), SBeff (54-61), Rdisk (62-66), SBdisk (67-74),
        #          MHI (75-81), RHI (82-86), Vflat (87-91), e_Vflat (92-96)
        try:
            # Galaxy name: columns 1-11
            name = line[0:11].strip()
            if not name:  # Skip empty names
                continue
                
            # Hubble type: columns 12-13
            hubble_type = int(line[12:14].strip()) if line[12:14].strip() else -1
            
            # Distance: columns 14-19
            distance_str = line[14:20].strip() if len(line) > 19 else ''
            distance = float(distance_str) if distance_str else None
            
            # Rdisk (disk scale length): columns 62-66
            Rdisk_str = line[62:67].strip() if len(line) > 66 else ''
            R_d_kpc = float(Rdisk_str) if Rdisk_str else None
            
            # SBdisk (disk central surface brightness): columns 67-74
            SBdisk_str = line[67:75].strip() if len(line) > 74 else ''
            SBdisk = float(SBdisk_str) if SBdisk_str else None
            
            galaxies[name] = {
                'hubble_type': hubble_type,
                'hubble_name': HUBBLE_TYPES.get(hubble_type, 'Unknown'),
                'distance_mpc': distance,
                'R_d_kpc': R_d_kpc,
                'SBdisk_solLum_pc2': SBdisk,
                'mu0_mag': None  # Will compute from SBdisk if needed
            }
        except (ValueError, IndexError) as e:
            continue
    
    return galaxies


def determine_type_group(hubble_name: str) -> str:
    """Determine if galaxy is early/intermediate/late type."""
    for group, types in TYPE_GROUPS.items():
        if hubble_name in types:
            return group
    return 'unknown'


def load_sparc_galaxy(filepath: Path, master_info: Optional[Dict] = None) -> SPARCGalaxy:
    """
    Load SPARC galaxy with morphology info.
    """
    # Parse rotation curve file
    distance_mpc = None
    with open(filepath, 'r') as f:
        for line in f:
            if 'Distance' in line:
                match = re.search(r'([\d.]+)\s*Mpc', line)
                if match:
                    distance_mpc = float(match.group(1))
                break
    
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
    
    # Get morphology from master table
    hubble_type = -1
    hubble_name = 'Unknown'
    if master_info and name in master_info:
        hubble_type = master_info[name]['hubble_type']
        hubble_name = master_info[name]['hubble_name']
        if distance_mpc is None:
            distance_mpc = master_info[name]['distance_mpc']
    
    type_group = determine_type_group(hubble_name)
    
    # Compute bulge fractions
    v_gas = data[:, 3]
    v_disk = data[:, 4]
    v_bulge = data[:, 5]
    v_total_sq = np.maximum(v_gas**2 + v_disk**2 + v_bulge**2, 1e-10)
    bulge_frac = v_bulge**2 / v_total_sq
    
    # Average B/T weighted by total velocity
    v_total = np.sqrt(v_total_sq)
    avg_bulge_frac = np.average(bulge_frac, weights=v_total)
    
    return SPARCGalaxy(
        name=name,
        hubble_type=hubble_type,
        hubble_name=hubble_name,
        type_group=type_group,
        distance_mpc=distance_mpc,
        r_kpc=data[:, 0],
        v_obs=data[:, 1],
        v_err=data[:, 2],
        v_gas=v_gas,
        v_disk=v_disk,
        v_bulge=v_bulge,
        sb_disk=data[:, 6],
        sb_bulge=data[:, 7],
        bulge_frac=bulge_frac,
        avg_bulge_frac=avg_bulge_frac
    )


def create_particle_distribution_fast(galaxy: SPARCGalaxy, n_particles: int = 100000) -> Tuple:
    """
    Fast particle distribution optimized for 5090 GPU.
    
    Uses vectorized sampling for maximum throughput.
    """
    r_kpc = galaxy.r_kpc
    sb_total = galaxy.sb_disk + galaxy.sb_bulge
    
    # Vectorized sampling
    weights = sb_total * r_kpc
    weights = weights / np.sum(weights)
    
    # Sample all radii at once
    r_samples = np.random.choice(r_kpc, size=n_particles, p=weights)
    
    # Vectorized azimuthal angles
    phi = np.random.uniform(0, 2*np.pi, n_particles)
    
    # Vectorized vertical positions (thin disk)
    z = np.random.normal(0, 0.3, n_particles)
    
    # Construct positions
    x = r_samples * np.cos(phi)
    y = r_samples * np.sin(phi)
    positions = np.stack([x, y, z], axis=1)
    
    # Masses proportional to SB
    masses = np.interp(r_samples, r_kpc, sb_total, left=0, right=0)
    
    # Normalize to match velocity at fiducial radius
    idx_mid = len(r_kpc) // 2
    r_fid = r_kpc[idx_mid]
    v_fid = np.sqrt(galaxy.v_gas[idx_mid]**2 + 
                    galaxy.v_disk[idx_mid]**2 + 
                    galaxy.v_bulge[idx_mid]**2)
    M_total = v_fid**2 * r_fid / G
    masses = masses / np.sum(masses) * M_total
    
    return xp_array(positions, dtype=cp.float64), xp_array(masses, dtype=cp.float64)


def predict_rotation_curve_fast(galaxy: SPARCGalaxy, params: Dict,
                                use_bulge_gate: bool = False,
                                n_particles: int = 100000) -> np.ndarray:
    """
    GPU-optimized rotation curve prediction.
    """
    start = time.time()
    
    # Create particle distribution
    positions, masses = create_particle_distribution_fast(galaxy, n_particles)
    
    # Target radii
    R_vals = xp_array(galaxy.r_kpc, dtype=cp.float64)
    
    # Bulge fractions
    bulge_frac = None
    if use_bulge_gate:
        bulge_frac = xp_array(galaxy.bulge_frac, dtype=cp.float64)
    
    # Compute rotation curve
    v_pred, _ = rotation_curve(
        positions, masses, R_vals, z=0.0,
        eps=0.05, params=params,
        use_multiplier=True, batch_size=100000,
        bulge_frac=bulge_frac
    )
    
    elapsed = time.time() - start
    print(f"    Computed in {elapsed:.2f}s ({n_particles/elapsed/1e6:.2f}M particles/s)")
    
    return to_cpu(v_pred)


def compute_metrics(v_obs: np.ndarray, v_pred: np.ndarray, v_err: np.ndarray) -> Dict:
    """Compute comprehensive performance metrics."""
    mask = (v_obs > 0) & (v_pred > 0)
    
    if np.sum(mask) == 0:
        return {'ape': np.inf, 'rms': np.inf, 'chi2': np.inf, 'chi2_reduced': np.inf, 'success': 0}
    
    v_obs_m = v_obs[mask]
    v_pred_m = v_pred[mask]
    v_err_m = np.maximum(v_err[mask], 2.0)  # Floor of 2 km/s
    
    # APE
    ape = 100.0 * np.abs(v_pred_m - v_obs_m) / v_obs_m
    mean_ape = np.mean(ape)
    
    # RMS
    rms = np.sqrt(np.mean((v_pred_m - v_obs_m)**2))
    
    # Chi-squared
    chi2 = np.sum(((v_pred_m - v_obs_m) / v_err_m)**2)
    chi2_reduced = chi2 / max(1, len(v_obs_m) - 1)
    
    # Success (more lenient 30% threshold for zero-shot)
    success = 1 if mean_ape < 30.0 else 0
    
    return {
        'ape': mean_ape,
        'rms': rms,
        'chi2': chi2,
        'chi2_reduced': chi2_reduced,
        'success': success,
        'n_points': int(np.sum(mask))
    }


def test_galaxy(galaxy: SPARCGalaxy, params: Dict, use_bulge_gate: bool = False,
               n_particles: int = 100000) -> Dict:
    """Test many-path model on single galaxy."""
    print(f"\nTesting {galaxy.name} ({galaxy.hubble_name}, {galaxy.type_group}, B/T={galaxy.avg_bulge_frac:.2f})...")
    
    try:
        # Predict rotation curve
        v_pred = predict_rotation_curve_fast(galaxy, params, use_bulge_gate, n_particles)
        
        # Compute metrics
        metrics = compute_metrics(galaxy.v_obs, v_pred, galaxy.v_err)
        
        # Combine galaxy info and metrics
        result = {
            **galaxy.to_dict(),
            **metrics,
            'mean_v_obs': float(np.mean(galaxy.v_obs)),
            'v_flat': float(galaxy.v_obs[len(galaxy.v_obs)//2:].mean())  # Outer velocity
        }
        
        print(f"  APE={metrics['ape']:.1f}%, chi2_red={metrics['chi2_reduced']:.2f}, Success={metrics['success']}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            **galaxy.to_dict(),
            'ape': np.inf,
            'error': str(e)
        }


def load_mw_frozen_params() -> Dict:
    """Load MW-frozen baseline parameters."""
    mw_file = Path(__file__).parent / 'MW_FROZEN_V1.txt'
    
    if not mw_file.exists():
        print("WARNING: MW_FROZEN_V1.txt not found, using default_params()")
        return default_params()
    
    params = {}
    with open(mw_file, 'r') as f:
        for line in f:
            if ':' in line and not line.startswith('#') and not line.startswith('='):
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value_str = parts[1].split('#')[0].strip()  # Remove comments
                    try:
                        value = float(value_str)
                        params[key] = value
                    except ValueError:
                        pass
    
    if not params:
        print("WARNING: Could not parse MW_FROZEN_V1.txt, using default_params()")
        return default_params()
    
    print(f"Loaded MW-frozen baseline: {len(params)} parameters")
    return params


def stratified_sampling(galaxies: List[SPARCGalaxy], types: List[str], 
                       n_per_type: int, seed: int = 42) -> List[SPARCGalaxy]:
    """Stratified sampling by Hubble type."""
    np.random.seed(seed)
    selected = []
    
    for htype in types:
        candidates = [g for g in galaxies if g.hubble_name == htype]
        if len(candidates) >= n_per_type:
            selected.extend(np.random.choice(candidates, n_per_type, replace=False))
        else:
            print(f"WARNING: Only {len(candidates)} galaxies of type {htype} available")
            selected.extend(candidates)
    
    return selected


def main():
    parser = argparse.ArgumentParser(description="SPARC stratified zero-shot test")
    parser.add_argument('--sparc_dir', type=str, 
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/external_data/Rotmod_LTG',
                       help='Directory containing SPARC _rotmod.dat files')
    parser.add_argument('--master_file', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/data/SPARC_Lelli2016c.mrt',
                       help='SPARC master table with morphology')
    
    # Sampling strategy
    parser.add_argument('--strategy', type=str, default='stratified',
                       choices=['stratified', 'filter', 'all', 'random'],
                       help='Sampling strategy')
    parser.add_argument('--types', type=str, default='Sm,Scd,Sc,Sbc,Sb',
                       help='Comma-separated list of Hubble types for stratified sampling')
    parser.add_argument('--n_per_type', type=int, default=10,
                       help='Number of galaxies per type (stratified strategy)')
    parser.add_argument('--filter_types', type=str, default='',
                       help='Filter to specific types (filter strategy)')
    parser.add_argument('--n_galaxies', type=int, default=50,
                       help='Total galaxies (random strategy)')
    
    # Model parameters
    parser.add_argument('--use_bulge_gate', type=int, default=0,
                       help='Use bulge-gated kernel (1=yes, 0=no)')
    parser.add_argument('--bulge_gate_power', type=float, default=2.0,
                       help='Bulge gate power parameter')
    parser.add_argument('--n_particles', type=int, default=100000,
                       help='Number of particles (more for 5090 GPU)')
    
    # Output
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    if _USING_CUPY:
        cp.random.seed(args.seed)
    
    print("="*80)
    print("SPARC STRATIFIED ZERO-SHOT TEST")
    print("="*80)
    print(f"Strategy: {args.strategy}")
    print(f"Bulge gating: {'ENABLED' if args.use_bulge_gate else 'DISABLED'}")
    print(f"GPU: {'YES (CuPy)' if _USING_CUPY else 'NO (NumPy)'}")
    print(f"Particles per galaxy: {args.n_particles:,}")
    
    # Load MW-frozen baseline
    params = load_mw_frozen_params()
    if args.use_bulge_gate:
        params['bulge_gate_power'] = args.bulge_gate_power
    
    print("\nParameters:")
    for k, v in sorted(params.items()):
        print(f"  {k:20s}: {v}")
    
    # Load SPARC master table
    print(f"\nLoading SPARC master table from {args.master_file}...")
    master_info = load_sparc_master_table(Path(args.master_file))
    print(f"Loaded morphology info for {len(master_info)} galaxies")
    
    # Load all galaxies
    sparc_dir = Path(args.sparc_dir)
    galaxy_files = list(sparc_dir.glob('*_rotmod.dat'))
    print(f"\nFound {len(galaxy_files)} SPARC rotation curve files")
    
    galaxies = []
    for gfile in galaxy_files:
        try:
            gal = load_sparc_galaxy(gfile, master_info)
            galaxies.append(gal)
        except Exception as e:
            print(f"  Skipping {gfile.name}: {e}")
    
    print(f"Successfully loaded {len(galaxies)} galaxies")
    
    # Apply sampling strategy
    if args.strategy == 'stratified':
        types = [t.strip() for t in args.types.split(',')]
        test_galaxies = stratified_sampling(galaxies, types, args.n_per_type, args.seed)
        print(f"\nStratified sampling: {len(test_galaxies)} galaxies")
        
    elif args.strategy == 'filter':
        filter_types = [t.strip() for t in args.filter_types.split(',')]
        test_galaxies = [g for g in galaxies if g.hubble_name in filter_types]
        print(f"\nFiltered to types {filter_types}: {len(test_galaxies)} galaxies")
        
    elif args.strategy == 'all':
        test_galaxies = galaxies
        print(f"\nTesting ALL galaxies: {len(test_galaxies)}")
        
    else:  # random
        test_galaxies = list(np.random.choice(galaxies, min(args.n_galaxies, len(galaxies)), replace=False))
        print(f"\nRandom sampling: {len(test_galaxies)} galaxies")
    
    # Test each galaxy
    print(f"\n{'='*80}")
    print("TESTING GALAXIES")
    print('='*80)
    
    results = []
    for i, gal in enumerate(test_galaxies, 1):
        print(f"\n[{i}/{len(test_galaxies)}]", end=' ')
        result = test_galaxy(gal, params, bool(args.use_bulge_gate), args.n_particles)
        results.append(result)
    
    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*80}")
        print(f"Results written to: {output_path}")
        print('='*80)
        
        # Summary statistics
        valid = [r for r in results if 'ape' in r and np.isfinite(r['ape'])]
        if valid:
            apes = [r['ape'] for r in valid]
            successes = [r['success'] for r in valid]
            
            print(f"\nSUMMARY")
            print(f"  Total tested: {len(valid)}")
            print(f"  Mean APE: {np.mean(apes):.2f}%")
            print(f"  Median APE: {np.median(apes):.2f}%")
            print(f"  Success rate (APE < 30%): {np.mean(successes)*100:.1f}%")
            
            # By type group
            for group in ['late', 'intermediate', 'early']:
                group_results = [r for r in valid if r['type_group'] == group]
                if group_results:
                    group_apes = [r['ape'] for r in group_results]
                    group_success = [r['success'] for r in group_results]
                    print(f"\n  {group.upper()} TYPE:")
                    print(f"    N = {len(group_results)}")
                    print(f"    Mean APE = {np.mean(group_apes):.2f}%")
                    print(f"    Success rate = {np.mean(group_success)*100:.1f}%")


if __name__ == '__main__':
    main()
