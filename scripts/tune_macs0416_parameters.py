#!/usr/bin/env python3
"""
MACS0416 Parameter Tuning - A_c Scan
====================================

Systematic scan of coherence amplitude A_c to find the value that brings
the predicted Einstein radius within ±15% of the observed 30 arcsec.

This establishes the baseline kernel parameters for the hierarchical
multi-cluster calibration.

Strategy:
---------
1. Coarse scan: A_c ∈ [1, 2, 5, 10, 20, 50] with fixed ell0=200 kpc
2. Find bracket around θ_E ≈ 30″
3. Fine scan within bracket
4. Report optimal A_c and diagnostics

Author: Sigma-Gravity Parameter Tuning
Date: 2025-01-14
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from datetime import datetime

# Import the MACS0416 test function
from test_macs0416_projected_kernel import test_macs0416_projected_kernel


def scan_A_c_coarse(verbose=True):
    """
    Coarse scan of coherence amplitude A_c.
    
    Returns
    -------
    results : list of dict
        Each dict contains A_c and test results
    """
    # Coarse grid
    A_c_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    if verbose:
        print("=" * 70)
        print("COARSE A_c SCAN")
        print("=" * 70)
        print()
        print(f"Testing {len(A_c_values)} values: {A_c_values}")
        print(f"Fixed parameters: ell0=200 kpc, p=2.0, ncoh=2.0")
        print()
    
    results = []
    
    for i, A_c in enumerate(A_c_values):
        if verbose:
            print(f"\n{'-'*70}")
            print(f"[{i+1}/{len(A_c_values)}] Testing A_c = {A_c:.1f}")
            print(f"{'-'*70}")
        
        try:
            res = test_macs0416_projected_kernel(
                q_los=1.0,
                q_plane=1.0,
                A_c=A_c,
                ell0=200.0,
                p=2.0,
                ncoh=2.0,
                emphasize_interior=True,
                verbose=False
            )
            
            result_dict = {
                'A_c': A_c,
                'theta_E_pred': res['theta_E_pred'],
                'theta_E_obs': res['theta_E_obs'],
                'error_arcsec': res['error'],
                'frac_error': res['frac_error'],
                'R_E_kpc': res['R_E_kpc'],
                'K_sigma_mean': res['kernel_diag']['K_sigma_mean'],
                'K_sigma_std': res['kernel_diag']['K_sigma_std'],
                'K_sigma_max': res['kernel_diag']['K_sigma_max'],
                'boost_mean': res['kernel_diag']['boost_factor_mean'],
                'success': True
            }
            
            results.append(result_dict)
            
            if verbose:
                print(f"  θ_E predicted: {res['theta_E_pred']:.2f}\"")
                print(f"  θ_E observed:  {res['theta_E_obs']:.2f}\"")
                print(f"  Error: {res['error']:.2f}\" ({res['frac_error']*100:.1f}%)")
                print(f"  <K_σ>: {res['kernel_diag']['K_sigma_mean']:.4f}")
                print(f"  <Boost>: {res['kernel_diag']['boost_factor_mean']:.4f}")
                
                if res['frac_error'] < 0.15:
                    print(f"  ✅ EXCELLENT: Within ±15% target!")
                elif res['frac_error'] < 0.25:
                    print(f"  ✓ GOOD: Within ±25%")
        
        except Exception as e:
            if verbose:
                print(f"  ❌ FAILED: {str(e)}")
            results.append({
                'A_c': A_c,
                'success': False,
                'error_msg': str(e)
            })
    
    return results


def scan_A_c_fine(A_c_low, A_c_high, n_points=10, verbose=True):
    """
    Fine scan within a bracket around the target θ_E.
    
    Parameters
    ----------
    A_c_low, A_c_high : float
        Bracket for A_c scan
    n_points : int
        Number of points in fine grid
    verbose : bool
    
    Returns
    -------
    results : list of dict
    """
    # Fine grid (linear spacing)
    A_c_values = np.linspace(A_c_low, A_c_high, n_points)
    
    if verbose:
        print("\n" + "=" * 70)
        print("FINE A_c SCAN")
        print("=" * 70)
        print()
        print(f"Bracket: A_c ∈ [{A_c_low:.2f}, {A_c_high:.2f}]")
        print(f"Points: {n_points}")
        print()
    
    results = []
    
    for i, A_c in enumerate(A_c_values):
        if verbose:
            print(f"[{i+1}/{n_points}] A_c = {A_c:.3f}", end="  ")
        
        try:
            res = test_macs0416_projected_kernel(
                q_los=1.0,
                q_plane=1.0,
                A_c=A_c,
                ell0=200.0,
                p=2.0,
                ncoh=2.0,
                emphasize_interior=True,
                verbose=False
            )
            
            result_dict = {
                'A_c': A_c,
                'theta_E_pred': res['theta_E_pred'],
                'theta_E_obs': res['theta_E_obs'],
                'error_arcsec': res['error'],
                'frac_error': res['frac_error'],
                'R_E_kpc': res['R_E_kpc'],
                'K_sigma_mean': res['kernel_diag']['K_sigma_mean'],
                'K_sigma_std': res['kernel_diag']['K_sigma_std'],
                'K_sigma_max': res['kernel_diag']['K_sigma_max'],
                'boost_mean': res['kernel_diag']['boost_factor_mean'],
                'success': True
            }
            
            results.append(result_dict)
            
            if verbose:
                print(f"θ_E = {res['theta_E_pred']:.2f}\" (err: {res['frac_error']*100:.1f}%)")
        
        except Exception as e:
            if verbose:
                print(f"FAILED: {str(e)}")
            results.append({
                'A_c': A_c,
                'success': False,
                'error_msg': str(e)
            })
    
    return results


def find_optimal_A_c(results):
    """
    Find A_c that minimizes |θ_E_pred - θ_E_obs|.
    
    Parameters
    ----------
    results : list of dict
        Scan results
        
    Returns
    -------
    optimal : dict
        Best-fit result
    """
    # Filter successful results
    valid = [r for r in results if r.get('success', False)]
    
    if not valid:
        return None
    
    # Find minimum absolute error
    errors = [r['error_arcsec'] for r in valid]
    idx_best = np.argmin(errors)
    
    return valid[idx_best]


def plot_parameter_scan(results_coarse, results_fine=None, output_dir='../results'):
    """
    Generate comprehensive diagnostic plots for parameter scan.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter successful results
    coarse_valid = [r for r in results_coarse if r.get('success', False)]
    
    if results_fine is not None:
        fine_valid = [r for r in results_fine if r.get('success', False)]
    else:
        fine_valid = []
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract data
    A_c_coarse = [r['A_c'] for r in coarse_valid]
    theta_E_coarse = [r['theta_E_pred'] for r in coarse_valid]
    K_sigma_coarse = [r['K_sigma_mean'] for r in coarse_valid]
    boost_coarse = [r['boost_mean'] for r in coarse_valid]
    err_frac_coarse = [r['frac_error'] for r in coarse_valid]
    
    if fine_valid:
        A_c_fine = [r['A_c'] for r in fine_valid]
        theta_E_fine = [r['theta_E_pred'] for r in fine_valid]
        K_sigma_fine = [r['K_sigma_mean'] for r in fine_valid]
        boost_fine = [r['boost_mean'] for r in fine_valid]
        err_frac_fine = [r['frac_error'] for r in fine_valid]
    
    theta_E_obs = 30.0  # arcsec
    
    # (0, 0): θ_E vs A_c
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogx(A_c_coarse, theta_E_coarse, 'o-', color='blue', 
                markersize=8, linewidth=2, label='Coarse scan')
    if fine_valid:
        ax.semilogx(A_c_fine, theta_E_fine, 's-', color='red', 
                    markersize=6, linewidth=1.5, alpha=0.7, label='Fine scan')
    ax.axhline(theta_E_obs, color='green', linestyle='--', linewidth=2, 
               label=f'Observed ({theta_E_obs}\")', zorder=10)
    ax.axhspan(theta_E_obs*0.85, theta_E_obs*1.15, alpha=0.2, color='green',
               label='±15% target')
    ax.set_xlabel('Coherence Amplitude A_c', fontsize=12, fontweight='bold')
    ax.set_ylabel('Einstein Radius θ_E [arcsec]', fontsize=12, fontweight='bold')
    ax.set_title('Einstein Radius vs A_c', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # (0, 1): Fractional error vs A_c
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogx(A_c_coarse, [e*100 for e in err_frac_coarse], 'o-', 
                color='blue', markersize=8, linewidth=2, label='Coarse scan')
    if fine_valid:
        ax.semilogx(A_c_fine, [e*100 for e in err_frac_fine], 's-', 
                    color='red', markersize=6, linewidth=1.5, alpha=0.7, label='Fine scan')
    ax.axhline(15, color='green', linestyle='--', linewidth=2, label='±15% target')
    ax.axhline(25, color='orange', linestyle=':', linewidth=1.5, label='±25% acceptable')
    ax.set_xlabel('Coherence Amplitude A_c', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fractional Error [%]', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Error vs A_c', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # (0, 2): <K_σ> vs A_c
    ax = fig.add_subplot(gs[0, 2])
    ax.loglog(A_c_coarse, K_sigma_coarse, 'o-', color='blue', 
              markersize=8, linewidth=2, label='Coarse scan')
    if fine_valid:
        ax.loglog(A_c_fine, K_sigma_fine, 's-', color='red', 
                  markersize=6, linewidth=1.5, alpha=0.7, label='Fine scan')
    ax.set_xlabel('Coherence Amplitude A_c', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Boost Kernel <K_σ>', fontsize=12, fontweight='bold')
    ax.set_title('Kernel Boost vs A_c', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # (1, 0): Mean boost factor vs A_c
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogx(A_c_coarse, boost_coarse, 'o-', color='blue', 
                markersize=8, linewidth=2, label='Coarse scan')
    if fine_valid:
        ax.semilogx(A_c_fine, boost_fine, 's-', color='red', 
                    markersize=6, linewidth=1.5, alpha=0.7, label='Fine scan')
    ax.axhline(1.0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Coherence Amplitude A_c', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Boost Factor (1 + <K_σ>)', fontsize=12, fontweight='bold')
    ax.set_title('Boost Factor vs A_c', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # (1, 1): θ_E vs <K_σ>
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(K_sigma_coarse, theta_E_coarse, 'o', color='blue', 
            markersize=8, label='Coarse scan')
    if fine_valid:
        ax.plot(K_sigma_fine, theta_E_fine, 's', color='red', 
                markersize=6, alpha=0.7, label='Fine scan')
    ax.axhline(theta_E_obs, color='green', linestyle='--', linewidth=2)
    ax.set_xlabel('Mean Boost Kernel <K_σ>', fontsize=12, fontweight='bold')
    ax.set_ylabel('Einstein Radius θ_E [arcsec]', fontsize=12, fontweight='bold')
    ax.set_title('θ_E vs Kernel Boost', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # (1, 2): Parameter space (A_c vs ell0) - placeholder for future 2D scan
    ax = fig.add_subplot(gs[1, 2])
    ax.text(0.5, 0.5, 'Future 2D Scan:\nA_c vs ell0\n\n(Currently: ell0=200 kpc fixed)', 
            transform=ax.transAxes, ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.set_title('2D Parameter Space (Future)', fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # (2, :): Summary table
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    
    # Find optimal
    all_results = coarse_valid + fine_valid
    optimal = find_optimal_A_c(all_results)
    
    if optimal:
        summary_text = f"""
        MACS0416 Parameter Tuning Results
        {'='*70}
        
        Target: θ_E = {theta_E_obs:.1f}\" (observed)
        Goal: |Δθ_E| < {theta_E_obs*0.15:.2f}\" (±15%)
        
        Optimal Parameters (Spherical, q_los=1.0):
        {'─'*70}
        A_c (coherence amplitude):     {optimal['A_c']:.3f}
        ell0 (coherence length):       200.0 kpc (fixed)
        p (window power):              2.0 (fixed)
        n_coh (coherence decay):       2.0 (fixed)
        Interior emphasis:             ON
        
        Results:
        {'─'*70}
        θ_E predicted:                 {optimal['theta_E_pred']:.2f}\"
        θ_E observed:                  {optimal['theta_E_obs']:.2f}\"
        Absolute error:                {optimal['error_arcsec']:.2f}\"
        Fractional error:              {optimal['frac_error']*100:.1f}%
        R_E (physical):                {optimal['R_E_kpc']:.1f} kpc
        
        Kernel Diagnostics:
        {'─'*70}
        <K_σ> (mean boost kernel):     {optimal['K_sigma_mean']:.4f}
        std(K_σ):                      {optimal['K_sigma_std']:.4f}
        K_σ (max):                     {optimal['K_sigma_max']:.4f}
        <Boost factor> (1+<K_σ>):     {optimal['boost_mean']:.4f}
        
        Assessment:
        {'─'*70}
        """
        
        if optimal['frac_error'] < 0.15:
            assessment = "✅ EXCELLENT - Within ±15% target for hierarchical fit!"
        elif optimal['frac_error'] < 0.25:
            assessment = "✓ GOOD - Within ±25%, acceptable for initial calibration"
        elif optimal['frac_error'] < 0.50:
            assessment = "⚠ ACCEPTABLE - Within ±50%, needs refinement"
        else:
            assessment = "❌ NEEDS WORK - More than 50% off, check physics/numerics"
        
        summary_text += f"        {assessment}\n        "
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('MACS0416 A_c Parameter Scan - Option A Projected Kernel',
                 fontsize=15, fontweight='bold', y=0.995)
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'macs0416_A_c_scan_{timestamp}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved: {output_path}")
    
    return fig, optimal


def save_results_json(results_coarse, results_fine, optimal, output_dir='../results'):
    """Save scan results to JSON for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    
    output = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'cluster': 'MACS0416',
            'target_theta_E': 30.0,
            'fixed_params': {
                'ell0': 200.0,
                'p': 2.0,
                'n_coh': 2.0,
                'q_los': 1.0,
                'q_plane': 1.0,
                'emphasize_interior': True
            }
        },
        'coarse_scan': results_coarse,
        'fine_scan': results_fine if results_fine else [],
        'optimal': optimal
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'macs0416_A_c_scan_{timestamp}.json')
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved: {output_path}")
    
    return output_path


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("MACS0416 PARAMETER TUNING: A_c SCAN")
    print("Option A: 2D Projected Sigma-Gravity Kernel")
    print("=" * 70)
    print()
    
    # Step 1: Coarse scan
    print("STEP 1: Coarse scan to bracket optimal A_c")
    print("-" * 70)
    results_coarse = scan_A_c_coarse(verbose=True)
    
    # Find which bracket contains θ_E ≈ 30″
    valid_coarse = [r for r in results_coarse if r.get('success', False)]
    
    if not valid_coarse:
        print("\n❌ ERROR: No successful coarse scan results!")
        sys.exit(1)
    
    # Check if we bracketed the target
    theta_E_vals = [r['theta_E_pred'] for r in valid_coarse]
    theta_E_target = 30.0
    
    # Find if we have values above and below target
    below = [r for r in valid_coarse if r['theta_E_pred'] < theta_E_target]
    above = [r for r in valid_coarse if r['theta_E_pred'] > theta_E_target]
    
    print("\n" + "=" * 70)
    print("COARSE SCAN SUMMARY")
    print("=" * 70)
    print()
    print(f"{'A_c':<10} {'θ_E [arcsec]':<15} {'Error [%]':<12} {'<K_σ>':<12} {'Status':<20}")
    print("-" * 70)
    for r in valid_coarse:
        status = ""
        if r['frac_error'] < 0.15:
            status = "✅ Excellent"
        elif r['frac_error'] < 0.25:
            status = "✓ Good"
        elif r['frac_error'] < 0.50:
            status = "○ Acceptable"
        else:
            status = "..."
        
        print(f"{r['A_c']:<10.1f} {r['theta_E_pred']:<15.2f} "
              f"{r['frac_error']*100:<12.1f} {r['K_sigma_mean']:<12.4f} {status:<20}")
    print()
    
    # Step 2: Fine scan if we bracketed
    results_fine = None
    
    if below and above:
        # Find tightest bracket
        A_c_below = max([r['A_c'] for r in below])
        A_c_above = min([r['A_c'] for r in above])
        
        print(f"✓ Target bracketed: A_c ∈ [{A_c_below:.1f}, {A_c_above:.1f}]")
        print()
        print("STEP 2: Fine scan within bracket")
        print("-" * 70)
        
        results_fine = scan_A_c_fine(A_c_below, A_c_above, n_points=15, verbose=True)
    else:
        print("⚠ Target not bracketed in coarse scan")
        print("  Proceeding with coarse results only")
    
    # Step 3: Find optimal and generate plots
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL PARAMETERS")
    print("=" * 70)
    
    all_results = valid_coarse + (results_fine if results_fine else [])
    optimal = find_optimal_A_c(all_results)
    
    if optimal:
        print()
        print("OPTIMAL PARAMETERS:")
        print("-" * 70)
        print(f"A_c = {optimal['A_c']:.3f}")
        print(f"θ_E predicted = {optimal['theta_E_pred']:.2f}\" (observed: {optimal['theta_E_obs']:.2f}\")")
        print(f"Error = {optimal['error_arcsec']:.2f}\" ({optimal['frac_error']*100:.1f}%)")
        print(f"<K_σ> = {optimal['K_sigma_mean']:.4f}")
        print(f"<Boost> = {optimal['boost_mean']:.4f}")
        print()
        
        if optimal['frac_error'] < 0.15:
            print("✅ EXCELLENT: Ready for hierarchical multi-cluster fit!")
        elif optimal['frac_error'] < 0.25:
            print("✓ GOOD: Acceptable starting point for calibration")
    
    # Step 4: Generate diagnostic plots and save results
    print("\n" + "=" * 70)
    print("GENERATING DIAGNOSTICS")
    print("=" * 70)
    
    fig, opt = plot_parameter_scan(results_coarse, results_fine)
    json_path = save_results_json(results_coarse, results_fine, optimal)
    
    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Review diagnostic plots and optimal parameters")
    print("  2. Run ablation studies (interior emphasis, window type)")
    print("  3. Hook up triaxial projection for geometry sensitivity test")
    print("  4. Proceed to hierarchical 12-cluster calibration")
    print()
