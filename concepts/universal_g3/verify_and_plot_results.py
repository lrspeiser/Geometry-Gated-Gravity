#!/usr/bin/env python3
"""
Verify star counts and create detailed analysis plots comparing:
1. Our best G³ model (99.3% accuracy)
2. Actual Gaia stellar velocities
3. General Relativity (Newtonian) predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy import stats

# Try to use GPU if available
try:
    import cupy as cp
    HAS_CUPY = True
    print("[✓] Using GPU (CuPy)")
except ImportError:
    HAS_CUPY = False
    print("[!] Using CPU (NumPy)")

def verify_star_counts():
    """Verify the number of stars in each dataset"""
    
    print("\n" + "="*70)
    print("VERIFICATION OF STAR COUNTS")
    print("="*70)
    
    # Check original Gaia download
    try:
        df_original = pd.read_csv('data/gaia_sky_slices/all_sky_gaia.csv')
        print(f"Original Gaia download: {len(df_original)} stars")
    except:
        print("Original Gaia file not found")
    
    # Check converted MW format
    df_mw = pd.read_csv('data/gaia_mw_real.csv')
    print(f"Converted MW format: {len(df_mw)} stars")
    
    # Check NPZ file
    data_npz = np.load('data/mw_gaia_144k.npz')
    print(f"NPZ file for optimizer: {len(data_npz['R_kpc'])} stars")
    
    # Check which optimizer was used
    print("\nOptimizer results:")
    
    # Check original run
    try:
        with open('out/mw_g3_gaia_real/best_theta.json', 'r') as f:
            orig_result = json.load(f)
        print(f"  Original optimizer: {orig_result['n_stars']} stars, accuracy: {(1-orig_result['median_relative_error'])*100:.1f}%")
    except:
        print("  Original optimizer results not found")
    
    # Check orchestrated run
    with open('out/mw_orchestrated/best.json', 'r') as f:
        orch_result = json.load(f)
    print(f"  Orchestrated optimizer: accuracy: {orch_result['accuracy']*100:.1f}%")
    
    print("="*70)
    
    return data_npz, df_mw

def load_best_model():
    """Load the best model parameters"""
    
    with open('out/mw_orchestrated/best.json', 'r') as f:
        best = json.load(f)
    
    print(f"\nBest Model: {best['variant']}")
    print(f"Accuracy: {best['accuracy']*100:.2f}%")
    print(f"Loss: {best['loss']:.4f}")
    
    return best

def compute_thickness_gate_model(R, z, gN, Sigma_loc, theta):
    """Compute the thickness_gate model predictions"""
    
    v0, rc, r0, delta, Sigma0, alpha, hz, m = theta
    
    # Modified rc with z-dependence
    rc_eff = rc * (1.0 + np.power(np.abs(z) / hz, m))
    
    # Rational gate
    rational = R / (R + rc_eff)
    
    # Logistic gate
    x = (R - r0) / delta
    logistic = 1.0 / (1.0 + np.exp(-x))
    
    # Sigma screen
    sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
    sigma_screen = 1.0 / (1.0 + np.power(sigma_ratio, alpha))
    
    # G³ tail acceleration
    g_tail = (v0**2 / R) * rational * logistic * sigma_screen
    
    # Total acceleration and velocity
    g_tot = gN + g_tail
    v_model = np.sqrt(np.maximum(R * g_tot, 0.0))
    
    return v_model, g_tail

def create_comparison_plots(data_npz, df_mw, best_params):
    """Create detailed comparison plots"""
    
    print("\nGenerating comparison plots...")
    
    # Load data
    R = data_npz['R_kpc']
    z = data_npz['z_kpc']
    v_obs = data_npz['v_obs_kms']
    gN = data_npz['gN_kms2_per_kpc']
    Sigma_loc = data_npz['Sigma_loc_Msun_pc2']
    
    # Compute model predictions
    v_g3, g_tail = compute_thickness_gate_model(R, z, gN, Sigma_loc, best_params)
    
    # Compute Newtonian (GR) predictions
    v_newton = np.sqrt(R * gN)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Rotation curves binned by radius
    ax1 = plt.subplot(2, 3, 1)
    
    # Bin data every 0.5 kpc from 3 to 16 kpc
    R_bins = np.arange(3.0, 16.5, 0.5)
    R_centers = (R_bins[:-1] + R_bins[1:]) / 2
    
    v_obs_binned = []
    v_g3_binned = []
    v_newton_binned = []
    v_obs_std = []
    
    for i in range(len(R_bins)-1):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.sum(mask) > 10:  # Need at least 10 stars per bin
            v_obs_binned.append(np.median(v_obs[mask]))
            v_g3_binned.append(np.median(v_g3[mask]))
            v_newton_binned.append(np.median(v_newton[mask]))
            v_obs_std.append(np.std(v_obs[mask]))
        else:
            v_obs_binned.append(np.nan)
            v_g3_binned.append(np.nan)
            v_newton_binned.append(np.nan)
            v_obs_std.append(np.nan)
    
    # Plot with error bars
    ax1.errorbar(R_centers, v_obs_binned, yerr=v_obs_std, 
                 fmt='ko', label='Gaia Data', markersize=4, alpha=0.7, capsize=3)
    ax1.plot(R_centers, v_g3_binned, 'r-', linewidth=2, label='G³ Model (99.3%)')
    ax1.plot(R_centers, v_newton_binned, 'b--', linewidth=2, label='GR (Newtonian)')
    ax1.set_xlabel('R [kpc]', fontsize=12)
    ax1.set_ylabel('v_circ [km/s]', fontsize=12)
    ax1.set_title('Rotation Curves (0.5 kpc bins)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(3, 16)
    ax1.set_ylim(150, 350)
    
    # 2. Residuals plot
    ax2 = plt.subplot(2, 3, 2)
    
    # Calculate residuals for each model
    residuals_g3 = [(v_g3_binned[i] - v_obs_binned[i]) for i in range(len(R_centers))]
    residuals_newton = [(v_newton_binned[i] - v_obs_binned[i]) for i in range(len(R_centers))]
    
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.plot(R_centers, residuals_g3, 'r-', marker='o', label='G³ Residuals')
    ax2.plot(R_centers, residuals_newton, 'b--', marker='s', label='GR Residuals')
    ax2.set_xlabel('R [kpc]', fontsize=12)
    ax2.set_ylabel('Model - Data [km/s]', fontsize=12)
    ax2.set_title('Model Residuals', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(3, 16)
    
    # 3. Relative error distribution
    ax3 = plt.subplot(2, 3, 3)
    
    rel_err_g3 = np.abs((v_g3 - v_obs) / v_obs) * 100
    rel_err_newton = np.abs((v_newton - v_obs) / v_obs) * 100
    
    bins = np.linspace(0, 50, 51)
    ax3.hist(rel_err_g3, bins=bins, alpha=0.5, color='red', label=f'G³ (median: {np.median(rel_err_g3):.1f}%)', density=True)
    ax3.hist(rel_err_newton, bins=bins, alpha=0.5, color='blue', label=f'GR (median: {np.median(rel_err_newton):.1f}%)', density=True)
    ax3.set_xlabel('Relative Error [%]', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.set_xlim(0, 30)
    
    # 4. Tail acceleration contribution
    ax4 = plt.subplot(2, 3, 4)
    
    # Bin the tail contribution
    g_tail_frac = g_tail / (gN + g_tail + 1e-10)
    g_tail_binned = []
    
    for i in range(len(R_bins)-1):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.sum(mask) > 10:
            g_tail_binned.append(np.median(g_tail_frac[mask]) * 100)
        else:
            g_tail_binned.append(np.nan)
    
    ax4.plot(R_centers, g_tail_binned, 'g-', marker='o', linewidth=2)
    ax4.set_xlabel('R [kpc]', fontsize=12)
    ax4.set_ylabel('G³ Tail Contribution [%]', fontsize=12)
    ax4.set_title('Modified Gravity Contribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(3, 16)
    ax4.set_ylim(0, 100)
    
    # 5. Accuracy vs radius
    ax5 = plt.subplot(2, 3, 5)
    
    accuracy_g3 = []
    accuracy_newton = []
    n_stars_bin = []
    
    for i in range(len(R_bins)-1):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.sum(mask) > 10:
            # Accuracy = fraction within 10% error
            acc_g3 = np.mean(np.abs((v_g3[mask] - v_obs[mask])/v_obs[mask]) < 0.1) * 100
            acc_newton = np.mean(np.abs((v_newton[mask] - v_obs[mask])/v_obs[mask]) < 0.1) * 100
            accuracy_g3.append(acc_g3)
            accuracy_newton.append(acc_newton)
            n_stars_bin.append(np.sum(mask))
        else:
            accuracy_g3.append(np.nan)
            accuracy_newton.append(np.nan)
            n_stars_bin.append(0)
    
    ax5.plot(R_centers, accuracy_g3, 'r-', marker='o', linewidth=2, label='G³ Model')
    ax5.plot(R_centers, accuracy_newton, 'b--', marker='s', linewidth=2, label='GR')
    ax5.set_xlabel('R [kpc]', fontsize=12)
    ax5.set_ylabel('Accuracy [%]', fontsize=12)
    ax5.set_title('Accuracy vs Radius (< 10% error)', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(3, 16)
    ax5.set_ylim(0, 105)
    
    # 6. Star density
    ax6 = plt.subplot(2, 3, 6)
    
    ax6.bar(R_centers, n_stars_bin, width=0.4, alpha=0.7, color='gray')
    ax6.set_xlabel('R [kpc]', fontsize=12)
    ax6.set_ylabel('Number of Stars', fontsize=12)
    ax6.set_title('Star Distribution', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xlim(3, 16)
    
    plt.suptitle(f'G³ Model Performance on {len(R):,} Gaia DR3 Stars', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('out/mw_orchestrated/detailed_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved plot to: out/mw_orchestrated/detailed_comparison.png")
    
    # Print detailed statistics
    print("\n" + "="*70)
    print("DETAILED STATISTICS")
    print("="*70)
    print(f"Total stars analyzed: {len(R):,}")
    print(f"R range: {R.min():.1f} - {R.max():.1f} kpc")
    print(f"Velocity range: {v_obs.min():.0f} - {v_obs.max():.0f} km/s")
    
    print(f"\nG³ Model Performance:")
    print(f"  Median relative error: {np.median(rel_err_g3):.2f}%")
    print(f"  Mean relative error: {np.mean(rel_err_g3):.2f}%")
    print(f"  Stars within 5% error: {np.mean(rel_err_g3 < 5)*100:.1f}%")
    print(f"  Stars within 10% error: {np.mean(rel_err_g3 < 10)*100:.1f}%")
    print(f"  Stars within 20% error: {np.mean(rel_err_g3 < 20)*100:.1f}%")
    
    print(f"\nNewtonian (GR) Performance:")
    print(f"  Median relative error: {np.median(rel_err_newton):.2f}%")
    print(f"  Mean relative error: {np.mean(rel_err_newton):.2f}%")
    print(f"  Stars within 5% error: {np.mean(rel_err_newton < 5)*100:.1f}%")
    print(f"  Stars within 10% error: {np.mean(rel_err_newton < 10)*100:.1f}%")
    print(f"  Stars within 20% error: {np.mean(rel_err_newton < 20)*100:.1f}%")
    
    print("="*70)
    
    return fig

def list_important_files():
    """List the most important code and data files"""
    
    print("\n" + "="*70)
    print("KEY FILES FOR VALIDATION")
    print("="*70)
    
    print("\nCore Code Files:")
    core_files = [
        "mw_gpu_orchestrator.py - Multi-solver orchestrator with model variants",
        "mw_g3_gpu_opt.py - Original GPU optimizer", 
        "convert_gaia_to_mw.py - Convert raw Gaia to MW coordinates",
        "prepare_gaia_npz.py - Prepare data for orchestrator"
    ]
    for f in core_files:
        print(f"  • {f}")
    
    print("\nData Files:")
    data_files = [
        "data/gaia_sky_slices/all_sky_gaia.csv - Original 144k Gaia download",
        "data/gaia_mw_real.csv - Converted MW coordinates (143,995 stars)",
        "data/mw_gaia_144k.npz - NPZ format for optimizer (143,995 stars)",
        "data/mw_sigma_disk.csv - Surface density model"
    ]
    for f in data_files:
        print(f"  • {f}")
    
    print("\nResults Files:")
    result_files = [
        "out/mw_orchestrated/best.json - Best model (99.3% accuracy)",
        "out/mw_orchestrated/search_log.csv - Optimization history",
        "out/mw_g3_gaia_real/best_theta.json - Original run (80% accuracy)",
        "out/mw_orchestrated/detailed_comparison.png - Comparison plots"
    ]
    for f in result_files:
        print(f"  • {f}")
    
    print("="*70)

def main():
    # Verify counts
    data_npz, df_mw = verify_star_counts()
    
    # Load best model
    best = load_best_model()
    
    # Create plots
    fig = create_comparison_plots(data_npz, df_mw, best['params'])
    
    # List files
    list_important_files()
    
    plt.show()

if __name__ == '__main__':
    main()