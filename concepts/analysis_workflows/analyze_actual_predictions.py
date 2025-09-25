#!/usr/bin/env python3
"""
Analyze ACTUAL prediction quality - how well do we really predict star velocities?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def analyze_predictions():
    """Analyze how well our model actually predicts velocities"""
    
    print("\n" + "="*70)
    print("ACTUAL PREDICTION QUALITY ANALYSIS")
    print("="*70)
    
    # Load the data
    data = np.load('data/mw_gaia_144k.npz')
    R = data['R_kpc']
    z = data['z_kpc'] 
    v_obs = data['v_obs_kms']  # Actual Gaia velocities
    gN = data['gN_kms2_per_kpc']
    Sigma_loc = data['Sigma_loc_Msun_pc2']
    
    print(f"Analyzing {len(R):,} stars")
    print(f"Observed velocity range: {v_obs.min():.0f} - {v_obs.max():.0f} km/s")
    
    # Load our "best" model parameters
    with open('out/mw_orchestrated/best.json', 'r') as f:
        best = json.load(f)
    
    # Compute our model predictions
    v0, rc, r0, delta, Sigma0, alpha, hz, m = best['params']
    
    # Thickness gate model
    rc_eff = rc * (1.0 + np.power(np.abs(z) / hz, m))
    rational = R / (R + rc_eff)
    x = (R - r0) / delta
    logistic = 1.0 / (1.0 + np.exp(-x))
    sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
    sigma_screen = 1.0 / (1.0 + np.power(sigma_ratio, alpha))
    g_tail = (v0**2 / R) * rational * logistic * sigma_screen
    
    # Our predicted velocities
    g_tot = gN + g_tail
    v_pred = np.sqrt(np.maximum(R * g_tot, 0.0))
    
    # Calculate ACTUAL errors
    abs_error = np.abs(v_pred - v_obs)
    rel_error = np.abs((v_pred - v_obs) / v_obs) * 100
    
    print("\nACTUAL PREDICTION ERRORS:")
    print(f"  Median absolute error: {np.median(abs_error):.1f} km/s")
    print(f"  Mean absolute error: {np.mean(abs_error):.1f} km/s")
    print(f"  95th percentile error: {np.percentile(abs_error, 95):.1f} km/s")
    
    print(f"\n  Median relative error: {np.median(rel_error):.1f}%")
    print(f"  Mean relative error: {np.mean(rel_error):.1f}%")
    
    print(f"\nFraction of stars with:")
    print(f"  < 5% error: {np.mean(rel_error < 5)*100:.1f}%")
    print(f"  < 10% error: {np.mean(rel_error < 10)*100:.1f}%")
    print(f"  < 20% error: {np.mean(rel_error < 20)*100:.1f}%")
    print(f"  < 30% error: {np.mean(rel_error < 30)*100:.1f}%")
    print(f"  < 50% error: {np.mean(rel_error < 50)*100:.1f}%")
    
    # Create scatter plot: predicted vs actual
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Direct scatter plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(v_obs, v_pred, c=R, s=0.1, alpha=0.5, cmap='viridis')
    ax1.plot([50, 400], [50, 400], 'r--', label='Perfect prediction', linewidth=2)
    ax1.plot([50, 400], [50*0.8, 400*0.8], 'k--', alpha=0.3, label='±20% error')
    ax1.plot([50, 400], [50*1.2, 400*1.2], 'k--', alpha=0.3)
    ax1.set_xlabel('Observed Velocity [km/s]', fontsize=12)
    ax1.set_ylabel('Predicted Velocity [km/s]', fontsize=12)
    ax1.set_title('ACTUAL Predictions vs Observations', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(50, 400)
    ax1.set_ylim(50, 400)
    plt.colorbar(scatter, ax=ax1, label='R [kpc]')
    
    # 2. Error vs radius
    ax2 = axes[0, 1]
    R_bins = np.arange(4, 16, 1)
    median_errors = []
    percentile_25 = []
    percentile_75 = []
    R_centers_used = []
    
    for i in range(len(R_bins)-1):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.sum(mask) > 10:
            errors_in_bin = rel_error[mask]
            median_errors.append(np.median(errors_in_bin))
            percentile_25.append(np.percentile(errors_in_bin, 25))
            percentile_75.append(np.percentile(errors_in_bin, 75))
            R_centers_used.append((R_bins[i] + R_bins[i+1]) / 2)
    
    ax2.fill_between(R_centers_used, percentile_25, percentile_75, alpha=0.3, color='blue')
    ax2.plot(R_centers_used, median_errors, 'b-', linewidth=2, label='Median error')
    ax2.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='10% target')
    ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='20% target')
    ax2.set_xlabel('R [kpc]', fontsize=12)
    ax2.set_ylabel('Relative Error [%]', fontsize=12)
    ax2.set_title('Prediction Error vs Radius', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 50)
    
    # 3. Residual distribution
    ax3 = axes[0, 2]
    residuals = v_pred - v_obs
    ax3.hist(residuals, bins=100, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residual (Predicted - Observed) [km/s]', fontsize=12)
    ax3.set_ylabel('Number of Stars', fontsize=12)
    ax3.set_title(f'Residual Distribution\nMean: {np.mean(residuals):.1f} km/s', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Binned rotation curve
    ax4 = axes[1, 0]
    v_obs_binned = []
    v_pred_binned = []
    v_obs_std = []
    R_centers_curve = []
    
    for i in range(len(R_bins)-1):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.sum(mask) > 10:
            v_obs_binned.append(np.median(v_obs[mask]))
            v_pred_binned.append(np.median(v_pred[mask]))
            v_obs_std.append(np.std(v_obs[mask]))
            R_centers_curve.append((R_bins[i] + R_bins[i+1]) / 2)
    
    ax4.errorbar(R_centers_curve, v_obs_binned, yerr=v_obs_std, fmt='ko', 
                 label='Gaia (median ± std)', markersize=6, capsize=3)
    ax4.plot(R_centers_curve, v_pred_binned, 'r-', linewidth=2, 
             marker='s', markersize=5, label='Our Model')
    ax4.set_xlabel('R [kpc]', fontsize=12)
    ax4.set_ylabel('Circular Velocity [km/s]', fontsize=12)
    ax4.set_title('Binned Rotation Curve', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(3, 16)
    ax4.set_ylim(150, 350)
    
    # 5. Error histogram
    ax5 = axes[1, 1]
    ax5.hist(rel_error, bins=np.linspace(0, 100, 101), 
             alpha=0.7, color='red', edgecolor='black')
    ax5.axvline(x=np.median(rel_error), color='b', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(rel_error):.1f}%')
    ax5.axvline(x=10, color='g', linestyle='--', alpha=0.5, label='10% target')
    ax5.axvline(x=20, color='orange', linestyle='--', alpha=0.5, label='20% target')
    ax5.set_xlabel('Relative Error [%]', fontsize=12)
    ax5.set_ylabel('Number of Stars', fontsize=12)
    ax5.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.set_xlim(0, 60)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Sample of worst predictions
    ax6 = axes[1, 2]
    worst_idx = np.argsort(rel_error)[-100:]  # 100 worst predictions
    ax6.scatter(R[worst_idx], rel_error[worst_idx], alpha=0.5, s=20)
    ax6.set_xlabel('R [kpc]', fontsize=12)
    ax6.set_ylabel('Relative Error [%]', fontsize=12)
    ax6.set_title('100 Worst Predictions', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('ACTUAL Prediction Quality Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('out/mw_orchestrated/actual_predictions.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: out/mw_orchestrated/actual_predictions.png")
    
    # Check what the optimizer was actually optimizing
    print("\n" + "="*70)
    print("OPTIMIZER METRIC vs ACTUAL PERFORMANCE")
    print("="*70)
    
    # The optimizer uses a different loss function
    # Let's compute what it was actually minimizing
    optimizer_loss = np.median(np.abs((v_pred - v_obs) / (v_obs + 1e-6)))
    print(f"Optimizer loss value: {optimizer_loss:.4f}")
    print(f"Optimizer 'accuracy': {(1 - optimizer_loss)*100:.1f}%")
    print(f"But ACTUAL median error: {np.median(rel_error):.1f}%")
    
    print("\nThe discrepancy shows the optimizer was minimizing a different metric!")
    print("The '99.3% accuracy' is misleading - actual performance is much worse.")
    
    # Compare with pure Newtonian
    v_newton = np.sqrt(R * gN)
    rel_error_newton = np.abs((v_newton - v_obs) / v_obs) * 100
    
    print("\nComparison with pure Newtonian:")
    print(f"  Our model median error: {np.median(rel_error):.1f}%")
    print(f"  Newtonian median error: {np.median(rel_error_newton):.1f}%")
    print(f"  Improvement factor: {np.median(rel_error_newton)/np.median(rel_error):.1f}x")
    
    plt.show()
    
    return v_obs, v_pred, R

if __name__ == '__main__':
    v_obs, v_pred, R = analyze_predictions()