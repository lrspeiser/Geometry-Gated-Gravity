#!/usr/bin/env python3
"""
Evaluate V2 B/T Laws on SPARC Sample
====================================

Applies v2 extended B/T laws with multi-predictor gating (B/T, Sigma0, shear)
and compares performance against per-galaxy best fits and v1 laws.

Usage:
    python many_path_model/bt_law/evaluate_bt_laws_v2.py
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from bt_laws import morph_to_bt
from bt_laws_v2 import load_theta, eval_all_laws_v2


def compute_galaxy_ape_v2(galaxy: SPARCGalaxy, params: dict) -> float:
    """
    Compute APE using V2 many-path model with kappa coherence factor.
    
    Args:
        galaxy: SPARC galaxy data
        params: Parameter dict with eta, ring_amp, M_max, lambda_ring, kappa, etc.
    
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
    
    # Many-path multiplier M(r) with V2 modifications
    eta = params['eta']
    ring_amp = params['ring_amp']
    M_max = params['M_max']
    lambda_ring = params['lambda_ring']
    kappa = params.get('kappa', 0.7)  # Coherence factor
    
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
    
    # V2: Ring winding term with coherence factor kappa
    # Modified denominator: 1 - kappa * exp(-x) instead of 1 - exp(-x)
    x = (2.0 * cp.pi * r) / lambda_ring
    ex = cp.exp(-x)
    ring_term_base = ring_amp * (ex / cp.maximum(1e-20, 1.0 - kappa * ex))
    
    # Bulge gating (suppress ring in bulge-dominated regions)
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
    parser = argparse.ArgumentParser(description='Evaluate V2 B/T Laws on SPARC')
    parser.add_argument('--bt_params', type=Path,
                       default=Path('many_path_model/bt_law/bt_law_params_v2.json'),
                       help='Path to v2 B/T law parameters')
    parser.add_argument('--disk_params', type=Path,
                       default=Path('many_path_model/bt_law/sparc_disk_params.json'),
                       help='Path to disk parameters (R_d, Sigma0)')
    parser.add_argument('--shear_preds', type=Path,
                       default=Path('many_path_model/bt_law/sparc_shear_predictors.json'),
                       help='Path to shear predictors')
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
                       default=Path('results/bt_law_evaluation_v2'),
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load B/T law parameters
    print("="*80)
    print("V2 B/T LAW EVALUATION ON FULL SPARC SAMPLE")
    print("="*80)
    print(f"\nLoading v2 B/T law parameters from: {args.bt_params}")
    theta = load_theta(args.bt_params)
    
    # Load disk parameters
    print(f"Loading disk parameters from: {args.disk_params}")
    with open(args.disk_params, 'r') as f:
        disk_params = json.load(f)
    print(f"  Loaded disk params for {len(disk_params)} galaxies")
    
    # Load shear predictors
    print(f"Loading shear predictors from: {args.shear_preds}")
    with open(args.shear_preds, 'r') as f:
        shear_preds = json.load(f)
    print(f"  Loaded shear predictors for {len(shear_preds)} galaxies")
    
    print("\nV2 Law parameters:")
    for param_name, param_val in theta.items():
        if isinstance(param_val, dict) and 'lo' in param_val:
            print(f"  {param_name:20s}: lo={param_val['lo']:.4f}, hi={param_val['hi']:.4f}, gamma={param_val['gamma']:.4f}")
        else:
            print(f"  {param_name:20s}: {param_val}")
    
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
    print("EVALUATING GALAXIES WITH V2 LAWS")
    print("="*80)
    
    for i, rotmod_file in enumerate(rotmod_files, 1):
        galaxy_name = rotmod_file.stem.replace('_rotmod', '')
        
        try:
            # Load galaxy
            galaxy = load_sparc_galaxy(rotmod_file, master_info)
            
            # Get morphology and disk parameters
            gal_disk = disk_params.get(galaxy_name, {})
            hubble_type = gal_disk.get('hubble_type', 'Unknown')
            type_group = gal_disk.get('type_group', 'unknown')
            R_d = gal_disk.get('R_d_kpc')
            Sigma0 = gal_disk.get('Sigma0')
            
            # Get shear
            gal_shear = shear_preds.get(galaxy_name, {})
            shear = gal_shear.get('shear_2p2Rd')
            compactness = gal_shear.get('compactness')
            
            # Get B/T from morphology
            B_T = morph_to_bt(hubble_type, type_group)
            
            # Predict parameters using v2 laws with all predictors
            v2_params = eval_all_laws_v2(B_T, theta,
                                         Sigma0=Sigma0,
                                         R_d=R_d,
                                         shear=shear,
                                         compactness=compactness)
            
            # Compute APE with v2 parameters
            v2_ape = compute_galaxy_ape_v2(galaxy, v2_params)
            
            # Get per-galaxy best APE (if available)
            per_gal_ape = per_galaxy_ape.get(galaxy_name, None)
            
            result = {
                'name': galaxy_name,
                'hubble_type': hubble_type,
                'type_group': type_group,
                'B_T': B_T,
                'R_d_kpc': R_d,
                'Sigma0': Sigma0,
                'shear': shear,
                'compactness': compactness,
                'v2_ape': v2_ape,
                'per_galaxy_ape': per_gal_ape,
                'delta_ape': v2_ape - per_gal_ape if per_gal_ape else None,
                'v2_params': v2_params,
                'per_galaxy_params': per_galaxy_params.get(galaxy_name, None)
            }
            results.append(result)
            
            # Progress update
            status = f"[{i:3d}/{len(rotmod_files)}] {galaxy_name:12s} ({type_group:12s})"
            status += f" B/T={B_T:.2f} Σ={Sigma0:.1f if Sigma0 else 0:>6} S={shear:.2f if shear else 0:>5}"
            status += f" | V2: {v2_ape:6.2f}%"
            if per_gal_ape:
                status += f" | Per-Gal: {per_gal_ape:6.2f}% | Δ: {v2_ape - per_gal_ape:+6.2f}%"
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
    
    # V2 Law statistics
    v2_apes = [r['v2_ape'] for r in results]
    print("\n" + "-"*80)
    print("V2 B/T LAW PERFORMANCE (Multi-Predictor Gating)")
    print("-"*80)
    print(f"  Mean APE:       {np.mean(v2_apes):.2f}%")
    print(f"  Median APE:     {np.median(v2_apes):.2f}%")
    print(f"  Std Dev:        {np.std(v2_apes):.2f}%")
    print(f"  Min APE:        {np.min(v2_apes):.2f}%")
    print(f"  Max APE:        {np.max(v2_apes):.2f}%")
    
    # Quality distribution
    excellent = sum(1 for ape in v2_apes if ape < 10)
    good = sum(1 for ape in v2_apes if 10 <= ape < 20)
    fair = sum(1 for ape in v2_apes if 20 <= ape < 30)
    poor = sum(1 for ape in v2_apes if ape >= 30)
    
    print(f"\nQuality Distribution:")
    print(f"  Excellent (< 10%):   {excellent:3d} ({100*excellent/len(v2_apes):.1f}%)")
    print(f"  Good (10-20%):       {good:3d} ({100*good/len(v2_apes):.1f}%)")
    print(f"  Fair (20-30%):       {fair:3d} ({100*fair/len(v2_apes):.1f}%)")
    print(f"  Poor (≥ 30%):        {poor:3d} ({100*poor/len(v2_apes):.1f}%)")
    
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
        print(f"\n  Mean Δ (V2 - PerGal):  {np.mean(deltas):+.2f}%")
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
    print("PERFORMANCE BY MORPHOLOGY (V2 Law)")
    print("-"*80)
    for type_group in ['late', 'intermediate', 'early']:
        group_results = [r for r in results if r['type_group'] == type_group]
        if group_results:
            group_apes = [r['v2_ape'] for r in group_results]
            print(f"\n  {type_group.upper():12s} (n={len(group_results):3d}):")
            print(f"    Mean:   {np.mean(group_apes):6.2f}%")
            print(f"    Median: {np.median(group_apes):6.2f}%")
            print(f"    Std:    {np.std(group_apes):6.2f}%")
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full results JSON
    output_file = args.output_dir / 'v2_evaluation_results.json'
    output_data = {
        'bt_law_params_file': str(args.bt_params),
        'evaluation_time': elapsed,
        'total_galaxies': len(rotmod_files),
        'successful': len(results),
        'v2_statistics': {
            'mean_ape': float(np.mean(v2_apes)),
            'median_ape': float(np.median(v2_apes)),
            'std_ape': float(np.std(v2_apes)),
            'min_ape': float(np.min(v2_apes)),
            'max_ape': float(np.max(v2_apes))
        },
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[OK] Full results saved to: {output_file}")
    
    # Summary CSV
    csv_file = args.output_dir / 'v2_evaluation_summary.csv'
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f"[OK] Summary CSV saved to: {csv_file}")
    
    # Generate comparison plots
    print("\nGenerating diagnostic plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. V2 APE vs Per-galaxy APE
    ax = axes[0, 0]
    if valid_comparisons:
        v2_vals = [r['v2_ape'] for r in valid_comparisons]
        pg_vals = [r['per_galaxy_ape'] for r in valid_comparisons]
        ax.scatter(pg_vals, v2_vals, alpha=0.5, s=20)
        lims = [0, max(max(v2_vals), max(pg_vals))]
        ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_xlabel('Per-Galaxy APE (%)')
        ax.set_ylabel('V2 APE (%)')
        ax.set_title('V2 vs Per-Galaxy Best Fit')
        ax.grid(alpha=0.3)
    
    # 2. APE vs B/T
    ax = axes[0, 1]
    bts = [r['B_T'] for r in results]
    ax.scatter(bts, v2_apes, alpha=0.5, s=20)
    ax.set_xlabel('B/T')
    ax.set_ylabel('V2 APE (%)')
    ax.set_title('V2 Performance vs Bulge Fraction')
    ax.grid(alpha=0.3)
    
    # 3. APE vs Sigma0
    ax = axes[0, 2]
    sigmas = [r['Sigma0'] for r in results if r['Sigma0'] is not None]
    apes_sigma = [r['v2_ape'] for r in results if r['Sigma0'] is not None]
    ax.scatter(sigmas, apes_sigma, alpha=0.5, s=20)
    ax.set_xscale('log')
    ax.set_xlabel('Sigma0 (M_sun/pc^2)')
    ax.set_ylabel('V2 APE (%)')
    ax.set_title('V2 Performance vs Compactness')
    ax.grid(alpha=0.3, which='both')
    
    # 4. APE vs Shear
    ax = axes[1, 0]
    shears = [r['shear'] for r in results if r['shear'] is not None]
    apes_shear = [r['v2_ape'] for r in results if r['shear'] is not None]
    ax.scatter(shears, apes_shear, alpha=0.5, s=20)
    ax.set_xlabel('Shear S')
    ax.set_ylabel('V2 APE (%)')
    ax.set_title('V2 Performance vs Shear')
    ax.axvline(theta['S0'], color='red', linestyle='--', alpha=0.7)
    ax.grid(alpha=0.3)
    
    # 5. Histogram of kappa values
    ax = axes[1, 1]
    kappas = [r['v2_params']['kappa'] for r in results]
    ax.hist(kappas, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Coherence Factor κ')
    ax.set_ylabel('Count')
    ax.set_title(f'Predicted Coherence Distribution')
    ax.axvline(theta['kappa_min'], color='red', linestyle='--', alpha=0.7)
    ax.axvline(theta['kappa_max'], color='red', linestyle='--', alpha=0.7)
    ax.grid(alpha=0.3)
    
    # 6. Morphology performance
    ax = axes[1, 2]
    type_groups = ['late', 'intermediate', 'early']
    means = []
    stds = []
    for tg in type_groups:
        group_apes = [r['v2_ape'] for r in results if r['type_group'] == tg]
        if group_apes:
            means.append(np.mean(group_apes))
            stds.append(np.std(group_apes))
        else:
            means.append(0)
            stds.append(0)
    
    x_pos = np.arange(len(type_groups))
    ax.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([tg.capitalize() for tg in type_groups])
    ax.set_ylabel('Mean APE (%)')
    ax.set_title('V2 Performance by Morphology')
    ax.grid(alpha=0.3, axis='y')
    
    fig.suptitle('V2 Extended B/T Laws: Multi-Predictor Evaluation', fontsize=16, fontweight='bold')
    fig.tight_layout()
    
    plot_file = args.output_dir / 'v2_evaluation_diagnostics.png'
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Diagnostic plots saved to: {plot_file}")
    
    print("="*80)


if __name__ == '__main__':
    main()
