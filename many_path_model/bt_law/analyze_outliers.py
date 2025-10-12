#!/usr/bin/env python3
"""
Analyze outlier galaxies with largest ΔAPE between B/T law and per-galaxy best.

Generates diagnostic plots showing:
- Rotation curves (observed, per-galaxy best, B/T law)
- Parameter comparisons (bar plots)
- Summary statistics
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    HAS_CUPY = False

from sparc_stratified_test import load_sparc_master_table, load_sparc_galaxy


def compute_rotation_curve(galaxy, params):
    """Compute rotation curve using many-path model."""
    r = cp.array(galaxy.r_kpc, dtype=cp.float32)
    v_gas = cp.array(galaxy.v_gas, dtype=cp.float32)
    v_disk = cp.array(galaxy.v_disk, dtype=cp.float32)
    v_bulge = cp.array(galaxy.v_bulge, dtype=cp.float32)
    bulge_frac = cp.array(galaxy.bulge_frac, dtype=cp.float32)
    
    v_bar_sq = v_gas**2 + v_disk**2 + v_bulge**2
    v_bar = cp.sqrt(cp.maximum(v_bar_sq, 1e-10))
    
    eta = params['eta']
    ring_amp = params['ring_amp']
    M_max = params['M_max']
    # Handle both lambda_ring and lambda_hat names
    lambda_ring = params.get('lambda_ring', params.get('lambda_hat', 25.0))
    R0 = params.get('R0', 5.0)
    R1 = params.get('R1', 70.0)
    p = params.get('p', 2.0)
    q = params.get('q', 3.5)
    
    R_gate = 0.5
    p_gate = 4.0
    gate = 1.0 - cp.exp(-(r / R_gate)**p_gate)
    f_d = (r / R0)**p / (1.0 + (r / R1)**q)
    
    x = (2.0 * cp.pi * r) / lambda_ring
    ex = cp.exp(-x)
    ring_term_base = ring_amp * (ex / cp.maximum(1e-20, 1.0 - ex))
    bulge_gate = (1.0 - cp.minimum(bulge_frac, 1.0))**2.0
    ring_term = ring_term_base * bulge_gate
    
    M = eta * gate * f_d * (1.0 + ring_term)
    M = cp.minimum(M, M_max)
    
    v_pred_sq = v_bar**2 * (1.0 + M)
    v_pred = cp.sqrt(cp.maximum(v_pred_sq, 0.0))
    
    return cp.asnumpy(v_pred)


def plot_outlier_diagnostics(galaxy_name, galaxy, result, output_dir):
    """
    Create diagnostic plot for a single outlier galaxy.
    
    Shows:
    - Top: Rotation curve comparison
    - Bottom left: Parameter comparison
    - Bottom right: Statistics text
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
    
    # Top: Rotation curve
    ax_rc = fig.add_subplot(gs[0, :])
    
    # Observed
    r = galaxy.r_kpc
    v_obs = galaxy.v_obs
    v_err = galaxy.v_err
    ax_rc.errorbar(r, v_obs, yerr=v_err, fmt='o', color='black', 
                   alpha=0.6, label='Observed', markersize=5, capsize=3)
    
    # Per-galaxy best
    if result['per_galaxy_params']:
        v_pergal = compute_rotation_curve(galaxy, result['per_galaxy_params'])
        ax_rc.plot(r, v_pergal, '-', color='green', linewidth=2.5, 
                  label=f'Per-galaxy best ({result["per_galaxy_ape"]:.1f}% APE)', alpha=0.8)
    
    # B/T law
    v_btlaw = compute_rotation_curve(galaxy, result['bt_law_params'])
    ax_rc.plot(r, v_btlaw, '-', color='red', linewidth=2.5, 
              label=f'B/T law ({result["bt_law_ape"]:.1f}% APE)', alpha=0.8)
    
    ax_rc.set_xlabel('Radius (kpc)', fontsize=12)
    ax_rc.set_ylabel('Rotation Velocity (km/s)', fontsize=12)
    ax_rc.legend(fontsize=11, loc='best')
    ax_rc.grid(alpha=0.3, linestyle=':')
    
    title = f"{galaxy_name} ({result['hubble_type']}, {result['type_group']}, B/T={result['B_T']:.2f})"
    ax_rc.set_title(title, fontsize=14, fontweight='bold')
    
    # Bottom left: Parameter comparison
    ax_params = fig.add_subplot(gs[1, 0])
    
    params = ['eta', 'ring_amp', 'M_max', 'lambda_ring']
    bt_vals = [result['bt_law_params'][p] for p in params]
    
    if result['per_galaxy_params']:
        # Handle lambda_ring vs lambda_hat naming
        pg_vals = []
        for p in params:
            if p == 'lambda_ring':
                val = result['per_galaxy_params'].get('lambda_ring', 
                      result['per_galaxy_params'].get('lambda_hat', 25.0))
            else:
                val = result['per_galaxy_params'].get(p, 0.0)
            pg_vals.append(val)
        
        x = np.arange(len(params))
        width = 0.35
        
        ax_params.bar(x - width/2, pg_vals, width, label='Per-galaxy', color='green', alpha=0.7)
        ax_params.bar(x + width/2, bt_vals, width, label='B/T law', color='red', alpha=0.7)
        
        ax_params.set_xticks(x)
        ax_params.set_xticklabels(params, fontsize=10)
        ax_params.set_ylabel('Parameter Value', fontsize=11)
        ax_params.legend(fontsize=10)
        ax_params.grid(alpha=0.3, axis='y', linestyle=':')
        ax_params.set_title('Parameter Comparison', fontsize=12, fontweight='bold')
    
    # Bottom right: Statistics
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis('off')
    
    stats_text = f"""
GALAXY STATISTICS
{'='*40}
Name:          {galaxy_name}
Hubble Type:   {result['hubble_type']}
Type Group:    {result['type_group']}
B/T:           {result['B_T']:.3f}

APE COMPARISON
{'='*40}
B/T Law APE:       {result['bt_law_ape']:.2f}%
Per-galaxy APE:    {result['per_galaxy_ape']:.2f}%
Delta (Δ):         {result['delta_ape']:+.2f}%

PARAMETER DELTAS
{'='*40}
"""
    
    if result['per_galaxy_params']:
        for param in params:
            bt = result['bt_law_params'][param]
            # Handle lambda_ring vs lambda_hat
            if param == 'lambda_ring':
                pg = result['per_galaxy_params'].get('lambda_ring',
                     result['per_galaxy_params'].get('lambda_hat', 25.0))
            else:
                pg = result['per_galaxy_params'].get(param, 0.0)
            delta = bt - pg
            pct_change = 100 * delta / pg if pg != 0 else 0
            stats_text += f"Δ{param:12s}: {delta:+7.3f} ({pct_change:+6.1f}%)\n"
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.tight_layout()
    
    # Save
    safe_name = galaxy_name.replace('/', '_')
    out_file = output_dir / f"outlier_{safe_name}.png"
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return out_file


def main():
    parser = argparse.ArgumentParser(description='Analyze B/T law outliers')
    parser.add_argument('--evaluation_results', type=Path,
                       default=Path('results/bt_law_evaluation/bt_law_evaluation_results.json'),
                       help='B/T law evaluation results JSON')
    parser.add_argument('--sparc_dir', type=Path,
                       default=Path('data/Rotmod_LTG'),
                       help='SPARC data directory')
    parser.add_argument('--master_file', type=Path,
                       default=Path('data/SPARC_Lelli2016c.mrt'),
                       help='SPARC master table')
    parser.add_argument('--output_dir', type=Path,
                       default=Path('results/bt_law_outliers'),
                       help='Output directory for diagnostic plots')
    parser.add_argument('--top_n', type=int, default=15,
                       help='Number of top outliers to analyze')
    
    args = parser.parse_args()
    
    # Load results
    print("="*80)
    print("B/T LAW OUTLIER ANALYSIS")
    print("="*80)
    print(f"\nLoading evaluation results from: {args.evaluation_results}")
    
    with open(args.evaluation_results, 'r') as f:
        eval_data = json.load(f)
    
    results = eval_data['results']
    
    # Filter to galaxies with valid comparisons
    valid_results = [r for r in results if r.get('delta_ape') is not None]
    print(f"Found {len(valid_results)} galaxies with valid comparisons")
    
    # Sort by absolute delta APE (worst performers)
    valid_results.sort(key=lambda x: abs(x['delta_ape']), reverse=True)
    
    # Take top N
    outliers = valid_results[:args.top_n]
    
    print(f"\nTop {args.top_n} outliers by |ΔAPE|:")
    print("-"*80)
    print(f"{'Rank':4s} {'Galaxy':12s} {'Type':8s} {'B/T':6s} {'BT-Law APE':12s} {'PerGal APE':12s} {'ΔAPE':8s}")
    print("-"*80)
    
    for i, r in enumerate(outliers, 1):
        print(f"{i:4d} {r['name']:12s} {r['type_group']:8s} {r['B_T']:6.2f} "
              f"{r['bt_law_ape']:12.2f}% {r['per_galaxy_ape']:12.2f}% {r['delta_ape']:+8.2f}%")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SPARC data
    print(f"\nLoading SPARC data from: {args.sparc_dir}")
    master_info = load_sparc_master_table(args.master_file)
    
    # Generate diagnostic plots
    print(f"\nGenerating diagnostic plots...")
    print("-"*80)
    
    for i, result in enumerate(outliers, 1):
        galaxy_name = result['name']
        
        try:
            # Load galaxy
            rotmod_file = args.sparc_dir / f"{galaxy_name}_rotmod.dat"
            galaxy = load_sparc_galaxy(rotmod_file, master_info)
            
            # Generate plot
            out_file = plot_outlier_diagnostics(galaxy_name, galaxy, result, args.output_dir)
            print(f"[{i:2d}/{len(outliers)}] {galaxy_name:12s} → {out_file.name}")
            
        except Exception as e:
            print(f"[{i:2d}/{len(outliers)}] {galaxy_name:12s} - FAILED: {e}")
    
    # Summary CSV
    summary_file = args.output_dir / 'outlier_summary.csv'
    df = pd.DataFrame(outliers)
    df.to_csv(summary_file, index=False)
    print(f"\n✓ Outlier summary saved to: {summary_file}")
    
    # Statistics
    print("\n" + "="*80)
    print("OUTLIER STATISTICS")
    print("="*80)
    
    deltas = [r['delta_ape'] for r in outliers]
    print(f"\nΔAPE statistics for top {args.top_n}:")
    print(f"  Mean:   {np.mean(deltas):+.2f}%")
    print(f"  Median: {np.median(deltas):+.2f}%")
    print(f"  Range:  {np.min(deltas):+.2f}% to {np.max(deltas):+.2f}%")
    
    # By type group
    print("\nBy morphology:")
    for type_group in ['late', 'intermediate', 'early']:
        group_outliers = [r for r in outliers if r['type_group'] == type_group]
        if group_outliers:
            group_deltas = [r['delta_ape'] for r in group_outliers]
            print(f"  {type_group:12s} (n={len(group_outliers):2d}): mean Δ = {np.mean(group_deltas):+.2f}%")
    
    print(f"\n✓ Diagnostic plots saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
