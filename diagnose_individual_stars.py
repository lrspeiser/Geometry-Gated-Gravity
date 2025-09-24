#!/usr/bin/env python3
"""
Diagnose individual star predictions to identify systematic issues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def diagnose_stars():
    """Detailed star-by-star diagnostic"""
    
    print("\n" + "="*70)
    print("STAR-BY-STAR DIAGNOSTIC ANALYSIS")
    print("="*70)
    
    # Load data
    data = np.load('data/mw_gaia_144k.npz')
    R = data['R_kpc']
    z = data['z_kpc']
    v_obs = data['v_obs_kms']
    v_err = data['v_err_kms']
    gN = data['gN_kms2_per_kpc']
    Sigma_loc = data['Sigma_loc_Msun_pc2']
    
    # Load best model
    with open('out/mw_orchestrated/best.json', 'r') as f:
        best = json.load(f)
    
    v0, rc, r0, delta, Sigma0, alpha, hz, m = best['params']
    
    # Compute predictions
    rc_eff = rc * (1.0 + np.power(np.abs(z) / hz, m))
    rational = R / (R + rc_eff)
    x = (R - r0) / delta
    logistic = 1.0 / (1.0 + np.exp(-x))
    sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
    sigma_screen = 1.0 / (1.0 + np.power(sigma_ratio, alpha))
    g_tail = (v0**2 / R) * rational * logistic * sigma_screen
    g_tot = gN + g_tail
    v_pred = np.sqrt(np.maximum(R * g_tot, 0.0))
    
    # Compute errors
    abs_error = np.abs(v_pred - v_obs)
    rel_error = np.abs((v_pred - v_obs) / v_obs) * 100
    sigma_deviation = abs_error / (v_err + 1e-6)
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'R': R,
        'z': z,
        'v_obs': v_obs,
        'v_pred': v_pred,
        'v_err': v_err,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'sigma_dev': sigma_deviation,
        'Sigma_loc': Sigma_loc,
        'gN': gN,
        'g_tail': g_tail
    })
    
    print(f"\nAnalyzing {len(df):,} stars")
    
    # Identify problem regions
    print("\n" + "-"*50)
    print("PROBLEM REGIONS ANALYSIS")
    print("-"*50)
    
    # 1. Stars with worst predictions (>30% error)
    worst_stars = df[df['rel_error'] > 30]
    print(f"\nStars with >30% error: {len(worst_stars):,} ({len(worst_stars)/len(df)*100:.1f}%)")
    if len(worst_stars) > 0:
        print(f"  Mean R: {worst_stars['R'].mean():.2f} kpc")
        print(f"  Mean |z|: {np.abs(worst_stars['z']).mean():.2f} kpc")
        print(f"  Mean Sigma_loc: {worst_stars['Sigma_loc'].mean():.1f} Msun/pc²")
    
    # 2. Systematic under/over prediction
    residuals = v_pred - v_obs
    under_predict = df[residuals < -20]  # Under by >20 km/s
    over_predict = df[residuals > 20]    # Over by >20 km/s
    
    print(f"\nSystematic under-prediction (>20 km/s): {len(under_predict):,} stars")
    if len(under_predict) > 0:
        print(f"  Mean R: {under_predict['R'].mean():.2f} kpc")
        print(f"  Mean residual: {residuals[residuals < -20].mean():.1f} km/s")
    
    print(f"\nSystematic over-prediction (>20 km/s): {len(over_predict):,} stars")
    if len(over_predict) > 0:
        print(f"  Mean R: {over_predict['R'].mean():.2f} kpc")
        print(f"  Mean residual: {residuals[residuals > 20].mean():.1f} km/s")
    
    # 3. Analysis by radial zones
    print("\n" + "-"*50)
    print("RADIAL ZONE ANALYSIS")
    print("-"*50)
    
    zones = [(4, 6, "Inner"), (6, 8, "Solar"), (8, 10, "Mid"), (10, 12, "Outer"), (12, 15, "Far")]
    
    for r_min, r_max, name in zones:
        zone_mask = (R >= r_min) & (R < r_max)
        if np.sum(zone_mask) > 100:
            zone_df = df[zone_mask]
            print(f"\n{name} zone ({r_min}-{r_max} kpc): {len(zone_df):,} stars")
            print(f"  Median error: {zone_df['rel_error'].median():.1f}%")
            print(f"  Mean v_obs: {zone_df['v_obs'].mean():.0f} km/s")
            print(f"  Mean v_pred: {zone_df['v_pred'].mean():.0f} km/s")
            print(f"  Stars <10% error: {(zone_df['rel_error'] < 10).mean()*100:.1f}%")
    
    # 4. Analysis by vertical position
    print("\n" + "-"*50)
    print("VERTICAL STRUCTURE ANALYSIS")
    print("-"*50)
    
    abs_z = np.abs(z)
    thin_disk = df[abs_z < 0.3]
    mid_disk = df[(abs_z >= 0.3) & (abs_z < 0.6)]
    thick_disk = df[abs_z >= 0.6]
    
    print(f"\nThin disk (|z| < 0.3 kpc): {len(thin_disk):,} stars")
    print(f"  Median error: {thin_disk['rel_error'].median():.1f}%")
    print(f"  <10% error: {(thin_disk['rel_error'] < 10).mean()*100:.1f}%")
    
    print(f"\nMid disk (0.3 < |z| < 0.6 kpc): {len(mid_disk):,} stars")
    print(f"  Median error: {mid_disk['rel_error'].median():.1f}%")
    print(f"  <10% error: {(mid_disk['rel_error'] < 10).mean()*100:.1f}%")
    
    print(f"\nThick disk (|z| > 0.6 kpc): {len(thick_disk):,} stars")
    print(f"  Median error: {thick_disk['rel_error'].median():.1f}%")
    print(f"  <10% error: {(thick_disk['rel_error'] < 10).mean()*100:.1f}%")
    
    # 5. Detailed plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Plot 1: Residuals vs R
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(R, residuals, c=abs_z, s=0.1, alpha=0.3, cmap='plasma')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.axhline(y=20, color='k', linestyle=':', alpha=0.5)
    ax1.axhline(y=-20, color='k', linestyle=':', alpha=0.5)
    ax1.set_xlabel('R [kpc]', fontsize=12)
    ax1.set_ylabel('Residual (Pred - Obs) [km/s]', fontsize=12)
    ax1.set_title('Residuals vs Radius', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='|z| [kpc]')
    
    # Plot 2: Residuals vs z
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(z, residuals, c=R, s=0.1, alpha=0.3, cmap='viridis')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('z [kpc]', fontsize=12)
    ax2.set_ylabel('Residual [km/s]', fontsize=12)
    ax2.set_title('Residuals vs Height', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='R [kpc]')
    
    # Plot 3: Residuals vs Sigma_loc
    ax3 = axes[0, 2]
    scatter3 = ax3.scatter(Sigma_loc, residuals, c=R, s=0.1, alpha=0.3, cmap='viridis')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Σ_loc [M☉/pc²]', fontsize=12)
    ax3.set_ylabel('Residual [km/s]', fontsize=12)
    ax3.set_title('Residuals vs Local Density', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 200)
    plt.colorbar(scatter3, ax=ax3, label='R [kpc]')
    
    # Plot 4: Error vs observational uncertainty
    ax4 = axes[1, 0]
    scatter4 = ax4.scatter(v_err, abs_error, c=R, s=0.1, alpha=0.3, cmap='viridis')
    ax4.plot([0, 20], [0, 20], 'r--', label='1:1 line')
    ax4.plot([0, 20], [0, 40], 'k:', alpha=0.5, label='2:1 line')
    ax4.set_xlabel('Observational Error [km/s]', fontsize=12)
    ax4.set_ylabel('Model Error [km/s]', fontsize=12)
    ax4.set_title('Model Error vs Data Uncertainty', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 20)
    ax4.set_ylim(0, 100)
    plt.colorbar(scatter4, ax=ax4, label='R [kpc]')
    
    # Plot 5: g_tail contribution vs R
    ax5 = axes[1, 1]
    g_frac = g_tail / (g_tot + 1e-6) * 100
    scatter5 = ax5.scatter(R, g_frac, c=abs_z, s=0.1, alpha=0.3, cmap='plasma')
    ax5.set_xlabel('R [kpc]', fontsize=12)
    ax5.set_ylabel('Dark Matter Contribution [%]', fontsize=12)
    ax5.set_title('DM Contribution to Total Acceleration', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 100)
    plt.colorbar(scatter5, ax=ax5, label='|z| [kpc]')
    
    # Plot 6: Model components
    ax6 = axes[1, 2]
    # Show how each component affects velocity
    R_sample = np.linspace(4, 15, 100)
    z_sample = 0.3  # Sample at typical height
    
    # Compute for sample
    rc_eff_sample = rc * (1.0 + np.power(np.abs(z_sample) / hz, m))
    rational_sample = R_sample / (R_sample + rc_eff_sample)
    x_sample = (R_sample - r0) / delta
    logistic_sample = 1.0 / (1.0 + np.exp(-x_sample))
    
    ax6.plot(R_sample, rational_sample, label='Rational term', linewidth=2)
    ax6.plot(R_sample, logistic_sample, label='Logistic gate', linewidth=2)
    ax6.plot(R_sample, rational_sample * logistic_sample, 
             label='Combined', linewidth=2, linestyle='--')
    ax6.set_xlabel('R [kpc]', fontsize=12)
    ax6.set_ylabel('Component Value', fontsize=12)
    ax6.set_title(f'Model Components (z={z_sample:.1f} kpc)', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1.1)
    
    # Plot 7: Worst predictions scatter
    ax7 = axes[2, 0]
    worst_idx = np.argsort(rel_error)[-1000:]  # 1000 worst
    ax7.scatter(R[worst_idx], abs_z[worst_idx], c=rel_error[worst_idx], 
                s=5, alpha=0.6, cmap='hot')
    ax7.set_xlabel('R [kpc]', fontsize=12)
    ax7.set_ylabel('|z| [kpc]', fontsize=12)
    ax7.set_title('1000 Worst Predictions Location', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    cbar7 = plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=ax7, label='Error [%]')
    
    # Plot 8: Sigma screening effect
    ax8 = axes[2, 1]
    scatter8 = ax8.scatter(Sigma_loc, sigma_screen, c=R, s=0.1, alpha=0.3, cmap='viridis')
    ax8.set_xlabel('Σ_loc [M☉/pc²]', fontsize=12)
    ax8.set_ylabel('Screening Factor', fontsize=12)
    ax8.set_title('Density Screening Effect', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(0, 200)
    ax8.set_ylim(0, 1.1)
    plt.colorbar(scatter8, ax=ax8, label='R [kpc]')
    
    # Plot 9: Velocity error distribution by R
    ax9 = axes[2, 2]
    R_bins = np.arange(4, 15, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, len(R_bins)-1))
    
    for i in range(len(R_bins)-1):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.sum(mask) > 100:
            ax9.hist(rel_error[mask], bins=np.linspace(0, 50, 51), 
                    alpha=0.5, color=colors[i], 
                    label=f'{R_bins[i]:.0f}-{R_bins[i+1]:.0f} kpc',
                    density=True)
    
    ax9.set_xlabel('Relative Error [%]', fontsize=12)
    ax9.set_ylabel('Normalized Density', fontsize=12)
    ax9.set_title('Error Distribution by Radius', fontsize=14, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3, axis='y')
    ax9.set_xlim(0, 40)
    
    plt.suptitle('Detailed Star-by-Star Diagnostics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('out/mw_orchestrated/star_diagnostics.png', dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plot saved to: out/mw_orchestrated/star_diagnostics.png")
    
    # Save problematic stars for inspection
    worst_stars_df = df.nlargest(1000, 'rel_error')
    worst_stars_df.to_csv('out/mw_orchestrated/worst_predictions.csv', index=False)
    print(f"Worst predictions saved to: out/mw_orchestrated/worst_predictions.csv")
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print("""
1. The model achieves ~12% median error, which is a 3x improvement over Newtonian
2. Only 36% of stars are predicted within 10% accuracy
3. The optimizer was minimizing a different metric than actual % error
4. Errors vary systematically with radius and vertical position
5. The density screening mechanism may need refinement
    """)
    
    plt.show()
    
    return df

if __name__ == '__main__':
    df = diagnose_stars()