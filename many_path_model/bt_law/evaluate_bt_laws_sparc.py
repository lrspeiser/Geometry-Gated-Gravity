#!/usr/bin/env python3
"""
Evaluate B/T Law Predictions on Full SPARC Sample
==================================================

Applies the learned continuous B/T laws to all 175 SPARC galaxies
and compares performance against per-galaxy best fits.

Usage:
    # Full evaluation
    python many_path_model/bt_law/evaluate_bt_laws_sparc.py
    
    # With custom paths
    python many_path_model/bt_law/evaluate_bt_laws_sparc.py \
        --bt_params many_path_model/bt_law/bt_law_params.json \
        --per_galaxy_results results/mega_test/mega_parallel_results.json \
        --output_dir results/bt_law_evaluation
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cupy as cp
    HAS_CUPY = True
    print("[OK] CuPy available - GPU acceleration ENABLED")
except ImportError:
    import numpy as cp
    HAS_CUPY = False
    print("[X] CuPy not available - CPU mode")

from sparc_stratified_test import load_sparc_master_table, load_sparc_galaxy, SPARCGalaxy

# Import B/T law utilities
sys.path.insert(0, str(Path(__file__).parent))
from bt_laws import load_theta, eval_all_laws, morph_to_bt


def compute_galaxy_ape(galaxy: SPARCGalaxy, params: dict) -> float:
    """
    Compute APE for a single galaxy using the many-path model.
    
    Args:
        galaxy: SPARC galaxy data
        params: Parameter dict with eta, ring_amp, M_max, lambda_ring, etc.
    
    Returns:
        APE percentage
    """
    # Upload to GPU
    r = cp.array(galaxy.r_kpc, dtype=cp.float32)
    v_obs = cp.array(galaxy.v_obs, dtype=cp.float32)
    v_gas = cp.array(galaxy.v_gas, dtype=cp.float32)
    v_disk = cp.array(galaxy.v_disk, dtype=cp.float32)
    v_bulge = cp.array(galaxy.v_bulge, dtype=cp.float32)
    bulge_frac = cp.array(galaxy.bulge_frac, dtype=cp.float32)
    
    # Baryonic velocity
    v_bar_sq = v_gas**2 + v_disk**2 + v_bulge**2
    v_bar = cp.sqrt(cp.maximum(v_bar_sq, 1e-10))
    
    # Many-path multiplier M(r)
    eta = params['eta']
    ring_amp = params['ring_amp']
    M_max = params['M_max']
    lambda_ring = params['lambda_ring']
    
    # Fixed parameters
    R0 = params.get('R0', 5.0)
    R1 = params.get('R1', 70.0)
    p = params.get('p', 2.0)
    q = params.get('q', 3.5)
    
    # Gate (turn on at galactic scales)
    R_gate = 0.5
    p_gate = 4.0
    gate = 1.0 - cp.exp(-(r / R_gate)**p_gate)
    
    # Growth with saturation
    f_d = (r / R0)**p / (1.0 + (r / R1)**q)
    
    # Ring winding term
    x = (2.0 * cp.pi * r) / lambda_ring
    ex = cp.exp(-x)
    ring_term_base = ring_amp * (ex / cp.maximum(1e-20, 1.0 - ex))
    
    # Bulge gating
    bulge_gate_power = params.get('bulge_gate_power', 2.0)
    bulge_gate = (1.0 - cp.minimum(bulge_frac, 1.0))**bulge_gate_power
    ring_term = ring_term_base * bulge_gate
    
    # Final multiplier
    M = eta * gate * f_d * (1.0 + ring_term)
    M = cp.minimum(M, M_max)
    
    # Predicted velocity
    v_pred_sq = v_bar**2 * (1.0 + M)
    v_pred = cp.sqrt(cp.maximum(v_pred_sq, 0.0))
    
    # APE
    mask = v_obs > 0
    ape = cp.abs(v_pred - v_obs) / cp.maximum(v_obs, 1.0) * 100.0
    ape = cp.where(mask, ape, 0.0)
    
    return float(cp.mean(ape))


def main():
    parser = argparse.ArgumentParser(description='Evaluate B/T Laws on SPARC')
    parser.add_argument('--bt_params', type=Path, 
                       default=Path('many_path_model/bt_law/bt_law_params_extended.json'),
                       help='Path to fitted B/T law parameters')
    parser.add_argument('--disk_params', type=Path,
                       default=Path('many_path_model/bt_law/sparc_disk_params.json'),
                       help='Path to disk parameters (R_d, Sigma0)')
    parser.add_argument('--per_galaxy_results', type=Path,
                       default=Path('results/mega_test/mega_parallel_results.json'),
                       help='Path to per-galaxy best fit results for comparison')
    parser.add_argument('--sparc_dir', type=Path,
                       default=Path('data/Rotmod_LTG'),
                       help='Path to SPARC data directory')
    parser.add_argument('--master_file', type=Path,
                       default=Path('data/SPARC_Lelli2016c.mrt'),
                       help='Path to SPARC master table')
    parser.add_argument('--output_dir', type=Path,
                       default=Path('results/bt_law_evaluation_extended'),
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load B/T law parameters
    print("="*80)
    print("EXTENDED B/T LAW EVALUATION ON FULL SPARC SAMPLE")
    print("="*80)
    print(f"\nLoading B/T law parameters from: {args.bt_params}")
    theta = load_theta(args.bt_params)
    
    # Load disk parameters
    print(f"Loading disk parameters from: {args.disk_params}")
    with open(args.disk_params, 'r') as f:
        disk_params = json.load(f)
    print(f"  Loaded disk params for {len(disk_params)} galaxies")
    
    print("\nLaw parameters:")
    for param_name, param_val in theta.items():
        if isinstance(param_val, dict) and 'lo' in param_val:
            print(f"  {param_name:15s}: lo={param_val['lo']:.4f}, hi={param_val['hi']:.4f}, gamma={param_val['gamma']:.4f}")
        else:
            print(f"  {param_name:15s}: {param_val}")
    
    # Load per-galaxy best fits for comparison
    print(f"\nLoading per-galaxy best fits from: {args.per_galaxy_results}")
    with open(args.per_galaxy_results, 'r') as f:
        per_galaxy_data = json.load(f)
    
    per_galaxy_ape = {}
    per_galaxy_params = {}
    for result in per_galaxy_data['results']:
        if result.get('success', False):
            name = result['name']
            per_galaxy_ape[name] = result['best_error']
            per_galaxy_params[name] = result['best_params']
    
    print(f"  Loaded {len(per_galaxy_ape)} successful per-galaxy fits")
    
    # Load SPARC data
    print(f"\nLoading SPARC galaxies from: {args.sparc_dir}")
    master_info = load_sparc_master_table(args.master_file)
    rotmod_files = sorted(args.sparc_dir.glob('*_rotmod.dat'))
    
    print(f"  Found {len(rotmod_files)} galaxies")
    
    # Evaluate each galaxy
    results = []
    start_time = time.time()
    
    print("\n" + "="*80)
    print("EVALUATING GALAXIES")
    print("="*80)
    
    for i, rotmod_file in enumerate(rotmod_files, 1):
        galaxy_name = rotmod_file.stem.replace('_rotmod', '')
        
        try:
            # Load galaxy
            galaxy = load_sparc_galaxy(rotmod_file, master_info)
            
            # Get morphology and disk parameters from disk_params (more reliable)
            gal_disk = disk_params.get(galaxy_name, {})
            hubble_type = gal_disk.get('hubble_type', 'Unknown')
            type_group = gal_disk.get('type_group', 'unknown')
            R_d = gal_disk.get('R_d_kpc')
            Sigma0 = gal_disk.get('Sigma0')
            
            # Get B/T from morphology
            B_T = morph_to_bt(hubble_type, type_group)
            
            # Predict parameters using extended B/T laws with R_d and Sigma0
            bt_law_params = eval_all_laws(B_T, theta, Sigma0=Sigma0, R_d=R_d)
            
            # Compute APE with B/T law parameters
            bt_law_ape = compute_galaxy_ape(galaxy, bt_law_params)
            
            # Get per-galaxy best APE (if available)
            per_gal_ape = per_galaxy_ape.get(galaxy_name, None)
            
            result = {
                'name': galaxy_name,
                'hubble_type': hubble_type,
                'type_group': type_group,
                'B_T': B_T,
                'R_d_kpc': R_d,
                'Sigma0': Sigma0,
                'bt_law_ape': bt_law_ape,
                'per_galaxy_ape': per_gal_ape,
                'delta_ape': bt_law_ape - per_gal_ape if per_gal_ape else None,
                'bt_law_params': bt_law_params,
                'per_galaxy_params': per_galaxy_params.get(galaxy_name, None)
            }
            results.append(result)
            
            # Progress update
            status = f"[{i:3d}/{len(rotmod_files)}] {galaxy_name:12s} ({galaxy.type_group:12s})"
            status += f" B/T={B_T:.2f} | BT-Law: {bt_law_ape:6.2f}%"
            if per_gal_ape:
                status += f" | Per-Gal: {per_gal_ape:6.2f}% | Δ: {bt_law_ape - per_gal_ape:+6.2f}%"
            print(status)
            
        except Exception as e:
            print(f"[{i:3d}/{len(rotmod_files)}] {galaxy_name:12s} - FAILED: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    # Compute statistics
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Successful: {len(results)}/{len(rotmod_files)}")
    
    # B/T Law statistics
    bt_law_apes = [r['bt_law_ape'] for r in results]
    print("\n" + "-"*80)
    print("B/T LAW PERFORMANCE (Universal Parameters)")
    print("-"*80)
    print(f"  Mean APE:       {np.mean(bt_law_apes):.2f}%")
    print(f"  Median APE:     {np.median(bt_law_apes):.2f}%")
    print(f"  Std Dev:        {np.std(bt_law_apes):.2f}%")
    print(f"  Min APE:        {np.min(bt_law_apes):.2f}%")
    print(f"  Max APE:        {np.max(bt_law_apes):.2f}%")
    
    # Quality distribution
    excellent = sum(1 for ape in bt_law_apes if ape < 10)
    good = sum(1 for ape in bt_law_apes if 10 <= ape < 20)
    fair = sum(1 for ape in bt_law_apes if 20 <= ape < 30)
    poor = sum(1 for ape in bt_law_apes if ape >= 30)
    
    print(f"\nQuality Distribution:")
    print(f"  Excellent (< 10%):   {excellent:3d} ({100*excellent/len(bt_law_apes):.1f}%)")
    print(f"  Good (10-20%):       {good:3d} ({100*good/len(bt_law_apes):.1f}%)")
    print(f"  Fair (20-30%):       {fair:3d} ({100*fair/len(bt_law_apes):.1f}%)")
    print(f"  Poor (≥ 30%):        {poor:3d} ({100*poor/len(bt_law_apes):.1f}%)")
    
    # Comparison to per-galaxy best
    valid_comparisons = [r for r in results if r['per_galaxy_ape'] is not None]
    if valid_comparisons:
        deltas = [r['delta_ape'] for r in valid_comparisons]
        per_gal_apes = [r['per_galaxy_ape'] for r in valid_comparisons]
        
        print("\n" + "-"*80)
        print("COMPARISON TO PER-GALAXY BEST FITS")
        print("-"*80)
        print(f"  Galaxies compared:     {len(valid_comparisons)}")
        print(f"  Per-galaxy mean APE:   {np.mean(per_gal_apes):.2f}%")
        print(f"  Per-galaxy median APE: {np.median(per_gal_apes):.2f}%")
        print(f"\n  Mean Δ (BT - PerGal):  {np.mean(deltas):+.2f}%")
        print(f"  Median Δ:              {np.median(deltas):+.2f}%")
        print(f"  Std Δ:                 {np.std(deltas):.2f}%")
        
        # How many are within X% of per-galaxy best
        within_5 = sum(1 for d in deltas if abs(d) < 5)
        within_10 = sum(1 for d in deltas if abs(d) < 10)
        within_20 = sum(1 for d in deltas if abs(d) < 20)
        
        print(f"\n  Within ±5% of best:    {within_5:3d} ({100*within_5/len(deltas):.1f}%)")
        print(f"  Within ±10% of best:   {within_10:3d} ({100*within_10/len(deltas):.1f}%)")
        print(f"  Within ±20% of best:   {within_20:3d} ({100*within_20/len(deltas):.1f}%)")
    
    # Statistics by morphology
    print("\n" + "-"*80)
    print("PERFORMANCE BY MORPHOLOGY (B/T Law)")
    print("-"*80)
    for type_group in ['late', 'intermediate', 'early']:
        group_results = [r for r in results if r['type_group'] == type_group]
        if group_results:
            group_apes = [r['bt_law_ape'] for r in group_results]
            print(f"\n  {type_group.upper():12s} (n={len(group_results):3d}):")
            print(f"    Mean:   {np.mean(group_apes):6.2f}%")
            print(f"    Median: {np.median(group_apes):6.2f}%")
            print(f"    Std:    {np.std(group_apes):6.2f}%")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full results JSON
    output_file = args.output_dir / 'bt_law_evaluation_results.json'
    output_data = {
        'bt_law_params_file': str(args.bt_params),
        'evaluation_time': elapsed,
        'total_galaxies': len(rotmod_files),
        'successful': len(results),
        'bt_law_statistics': {
            'mean_ape': float(np.mean(bt_law_apes)),
            'median_ape': float(np.median(bt_law_apes)),
            'std_ape': float(np.std(bt_law_apes)),
            'min_ape': float(np.min(bt_law_apes)),
            'max_ape': float(np.max(bt_law_apes))
        },
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[OK] Full results saved to: {output_file}")
    
    # Summary CSV
    csv_file = args.output_dir / 'bt_law_evaluation_summary.csv'
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"[OK] Summary CSV saved to: {csv_file}")
    
    # Parameter comparison CSV (if per-galaxy available)
    if valid_comparisons:
        param_comp = []
        for r in valid_comparisons:
            row = {
                'name': r['name'],
                'type_group': r['type_group'],
                'B_T': r['B_T'],
                'bt_law_ape': r['bt_law_ape'],
                'per_galaxy_ape': r['per_galaxy_ape'],
                'delta_ape': r['delta_ape']
            }
            # Add parameter comparisons
            for param in ['eta', 'ring_amp', 'M_max', 'lambda_ring']:
                bt_val = r['bt_law_params'].get(param)
                pg_val = r['per_galaxy_params'].get(param) if r['per_galaxy_params'] else None
                row[f'bt_law_{param}'] = bt_val
                row[f'per_gal_{param}'] = pg_val
                if bt_val and pg_val:
                    row[f'delta_{param}'] = bt_val - pg_val
            param_comp.append(row)
        
        param_csv = args.output_dir / 'parameter_comparison.csv'
        pd.DataFrame(param_comp).to_csv(param_csv, index=False)
        print(f"[OK] Parameter comparison saved to: {param_csv}")
    
    print("="*80)


if __name__ == '__main__':
    main()
