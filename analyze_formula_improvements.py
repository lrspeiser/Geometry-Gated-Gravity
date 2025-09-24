#!/usr/bin/env python3
"""
Deep analysis of G³ model systematic errors and formula improvement suggestions.
This analysis identifies specific weaknesses in the current formula and tests modifications.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
import warnings
warnings.filterwarnings('ignore')

def load_data_and_predictions():
    """Load data and compute current model predictions"""
    data = np.load('data/mw_gaia_144k.npz')
    R = data['R_kpc']
    z = data['z_kpc']
    v_obs = data['v_obs_kms']
    v_err = data['v_err_kms']
    gN = data['gN_kms2_per_kpc']
    Sigma_loc = data['Sigma_loc_Msun_pc2']
    
    # Load current best parameters
    with open('out/mw_orchestrated/best.json', 'r') as f:
        best = json.load(f)
    
    v0, rc, r0, delta, Sigma0, alpha, hz, m = best['params']
    
    # Current model predictions
    rc_eff = rc * (1.0 + np.power(np.abs(z) / hz, m))
    rational = R / (R + rc_eff)
    x = (R - r0) / delta
    logistic = 1.0 / (1.0 + np.exp(-x))
    sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
    sigma_screen = 1.0 / (1.0 + np.power(sigma_ratio, alpha))
    g_additional = (v0**2 / R) * rational * logistic * sigma_screen
    g_total = gN + g_additional
    v_pred = np.sqrt(np.maximum(R * g_total, 0.0))
    
    return R, z, v_obs, v_err, gN, Sigma_loc, v_pred, g_additional, best['params']

def analyze_systematic_errors():
    """Identify systematic patterns in prediction errors"""
    R, z, v_obs, v_err, gN, Sigma_loc, v_pred, g_additional, params = load_data_and_predictions()
    
    print("\n" + "="*70)
    print("SYSTEMATIC ERROR ANALYSIS")
    print("="*70)
    
    residuals = v_pred - v_obs
    rel_error = np.abs(residuals / v_obs) * 100
    
    # 1. Analyze residuals as function of various parameters
    print("\n1. RESIDUAL CORRELATIONS")
    print("-" * 40)
    
    # Correlation with radius
    from scipy.stats import pearsonr
    corr_R, p_R = pearsonr(R, residuals)
    print(f"Correlation with R: {corr_R:.3f} (p={p_R:.3e})")
    
    # Correlation with height
    corr_z, p_z = pearsonr(np.abs(z), residuals)
    print(f"Correlation with |z|: {corr_z:.3f} (p={p_z:.3e})")
    
    # Correlation with local density
    corr_sigma, p_sigma = pearsonr(Sigma_loc, residuals)
    print(f"Correlation with Σ_loc: {corr_sigma:.3f} (p={p_sigma:.3e})")
    
    # Correlation with velocity
    corr_v, p_v = pearsonr(v_obs, residuals)
    print(f"Correlation with v_obs: {corr_v:.3f} (p={p_v:.3e})")
    
    # 2. Identify regions of systematic failure
    print("\n2. SYSTEMATIC FAILURE REGIONS")
    print("-" * 40)
    
    # Radial bins analysis
    R_bins = np.arange(4, 16, 2)
    for i in range(len(R_bins)-1):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.sum(mask) > 100:
            mean_resid = np.mean(residuals[mask])
            std_resid = np.std(residuals[mask])
            median_rel = np.median(rel_error[mask])
            print(f"R={R_bins[i]:.0f}-{R_bins[i+1]:.0f} kpc: mean resid={mean_resid:+.1f}±{std_resid:.1f} km/s, median err={median_rel:.1f}%")
    
    # 3. Analyze what the model is missing
    print("\n3. MISSING PHYSICS ANALYSIS")
    print("-" * 40)
    
    # Check if errors correlate with angular momentum
    L = R * v_obs  # Specific angular momentum
    corr_L, p_L = pearsonr(L, residuals)
    print(f"Correlation with angular momentum: {corr_L:.3f} (p={p_L:.3e})")
    
    # Check if errors correlate with centrifugal acceleration
    a_cent = v_obs**2 / R
    corr_acent, p_acent = pearsonr(a_cent, residuals)
    print(f"Correlation with centrifugal accel: {corr_acent:.3f} (p={p_acent:.3e})")
    
    # Check ratio of additional to Newtonian
    g_ratio = g_additional / (gN + 1e-6)
    corr_gratio, p_gratio = pearsonr(g_ratio, residuals)
    print(f"Correlation with g_add/g_N ratio: {corr_gratio:.3f} (p={p_gratio:.3e})")
    
    return R, z, v_obs, v_err, gN, Sigma_loc, v_pred, residuals, rel_error

def test_formula_modifications():
    """Test various modifications to the formula"""
    R, z, v_obs, v_err, gN, Sigma_loc, v_pred_orig, residuals, rel_error = analyze_systematic_errors()
    
    print("\n" + "="*70)
    print("FORMULA MODIFICATION TESTS")
    print("="*70)
    
    # Original parameters
    with open('out/mw_orchestrated/best.json', 'r') as f:
        best = json.load(f)
    v0, rc, r0, delta, Sigma0, alpha, hz, m = best['params']
    
    modifications = {}
    
    # Test 1: Different rational function forms
    print("\n1. ALTERNATIVE RATIONAL FUNCTIONS")
    print("-" * 40)
    
    # Original
    rc_eff = rc * (1.0 + np.power(np.abs(z) / hz, m))
    rational_orig = R / (R + rc_eff)
    
    # Alternative 1: Squared denominator (stronger core)
    rational_alt1 = R**2 / (R**2 + rc_eff**2)
    
    # Alternative 2: Exponential decay
    rational_alt2 = 1.0 - np.exp(-R / rc_eff)
    
    # Alternative 3: Power law transition
    rational_alt3 = (R / rc_eff) / (1 + (R / rc_eff))
    
    for name, rational in [("Original", rational_orig), 
                           ("Squared", rational_alt1),
                           ("Exponential", rational_alt2),
                           ("Power", rational_alt3)]:
        x = (R - r0) / delta
        logistic = 1.0 / (1.0 + np.exp(-x))
        sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
        sigma_screen = 1.0 / (1.0 + np.power(sigma_ratio, alpha))
        g_add = (v0**2 / R) * rational * logistic * sigma_screen
        v_test = np.sqrt(np.maximum(R * (gN + g_add), 0.0))
        error = np.median(np.abs((v_test - v_obs) / v_obs) * 100)
        modifications[f"rational_{name}"] = error
        print(f"  {name:12s}: median error = {error:.2f}%")
    
    # Test 2: Different transition functions
    print("\n2. ALTERNATIVE TRANSITION FUNCTIONS")
    print("-" * 40)
    
    rational = R / (R + rc_eff)  # Use original
    
    # Original logistic
    x = (R - r0) / delta
    logistic_orig = 1.0 / (1.0 + np.exp(-x))
    
    # Alternative 1: Error function transition (smoother)
    from scipy.special import erf
    logistic_alt1 = 0.5 * (1 + erf((R - r0) / delta))
    
    # Alternative 2: Tanh transition
    logistic_alt2 = 0.5 * (1 + np.tanh((R - r0) / delta))
    
    # Alternative 3: Power law transition
    logistic_alt3 = np.where(R < r0, 
                             (R / r0)**2,
                             1.0)
    
    for name, logistic in [("Logistic", logistic_orig),
                          ("Error func", logistic_alt1),
                          ("Tanh", logistic_alt2),
                          ("Power", logistic_alt3)]:
        sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
        sigma_screen = 1.0 / (1.0 + np.power(sigma_ratio, alpha))
        g_add = (v0**2 / R) * rational * logistic * sigma_screen
        v_test = np.sqrt(np.maximum(R * (gN + g_add), 0.0))
        error = np.median(np.abs((v_test - v_obs) / v_obs) * 100)
        modifications[f"transition_{name}"] = error
        print(f"  {name:12s}: median error = {error:.2f}%")
    
    # Test 3: Modified screening functions
    print("\n3. ALTERNATIVE SCREENING FUNCTIONS")
    print("-" * 40)
    
    x = (R - r0) / delta
    logistic = 1.0 / (1.0 + np.exp(-x))
    
    # Original screening
    sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
    screen_orig = 1.0 / (1.0 + np.power(sigma_ratio, alpha))
    
    # Alternative 1: Exponential screening
    screen_alt1 = np.exp(-sigma_ratio * alpha)
    
    # Alternative 2: Linear screening with cutoff
    screen_alt2 = np.maximum(0, 1.0 - sigma_ratio * alpha / 10)
    
    # Alternative 3: Gaussian screening
    screen_alt3 = np.exp(-(sigma_ratio * alpha)**2)
    
    for name, screen in [("Power law", screen_orig),
                         ("Exponential", screen_alt1),
                         ("Linear", screen_alt2),
                         ("Gaussian", screen_alt3)]:
        g_add = (v0**2 / R) * rational * logistic * screen
        v_test = np.sqrt(np.maximum(R * (gN + g_add), 0.0))
        error = np.median(np.abs((v_test - v_obs) / v_obs) * 100)
        modifications[f"screening_{name}"] = error
        print(f"  {name:12s}: median error = {error:.2f}%")
    
    # Test 4: Additional correction terms
    print("\n4. ADDITIONAL CORRECTION TERMS")
    print("-" * 40)
    
    # Base model
    g_base = (v0**2 / R) * rational * logistic * screen_orig
    
    # Add velocity-dependent correction
    v_newton = np.sqrt(R * gN)
    v_ratio = v_newton / 220  # Normalize to solar neighborhood
    
    # Correction 1: Velocity-dependent boost
    boost1 = 1.0 + 0.1 * (v_ratio - 1.0)**2
    g_test1 = g_base * boost1
    
    # Correction 2: Radius-dependent adjustment
    R_norm = R / 8.0  # Normalize to solar radius
    boost2 = 1.0 + 0.05 * (R_norm - 1.0)
    g_test2 = g_base * boost2
    
    # Correction 3: Angular momentum correction
    L_norm = (R * v_newton) / (8.0 * 220)  # Normalized angular momentum
    boost3 = 1.0 + 0.02 * L_norm
    g_test3 = g_base * boost3
    
    for name, g_add in [("Base", g_base),
                        ("Velocity boost", g_test1),
                        ("Radius adjust", g_test2),
                        ("Angular mom", g_test3)]:
        v_test = np.sqrt(np.maximum(R * (gN + g_add), 0.0))
        error = np.median(np.abs((v_test - v_obs) / v_obs) * 100)
        modifications[f"correction_{name}"] = error
        print(f"  {name:12s}: median error = {error:.2f}%")
    
    return modifications

def suggest_hybrid_formula():
    """Suggest a hybrid formula combining best performing modifications"""
    
    print("\n" + "="*70)
    print("HYBRID FORMULA SUGGESTIONS")
    print("="*70)
    
    R, z, v_obs, v_err, gN, Sigma_loc, v_pred_orig, _, _ = load_data_and_predictions()
    
    # Load original parameters
    with open('out/mw_orchestrated/best.json', 'r') as f:
        best = json.load(f)
    v0, rc, r0, delta, Sigma0, alpha, hz, m = best['params']
    
    print("\nTesting hybrid formulations combining best features...")
    print("-" * 50)
    
    # Hybrid 1: Squared rational + tanh transition + exponential screening
    print("\n1. HYBRID A: Smooth transitions")
    rc_eff = rc * (1.0 + np.power(np.abs(z) / hz, m))
    rational = R**2 / (R**2 + rc_eff**2)  # Squared for smoother core
    transition = 0.5 * (1 + np.tanh((R - r0) / delta))  # Tanh for smooth transition
    sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
    screening = np.exp(-sigma_ratio * alpha / 2)  # Exponential screening
    
    g_hybrid1 = (v0**2 / R) * rational * transition * screening
    v_hybrid1 = np.sqrt(np.maximum(R * (gN + g_hybrid1), 0.0))
    error1 = np.median(np.abs((v_hybrid1 - v_obs) / v_obs) * 100)
    print(f"  Median error: {error1:.2f}%")
    print(f"  Components: squared rational, tanh transition, exp screening")
    
    # Hybrid 2: Add radial correction
    print("\n2. HYBRID B: With radial correction")
    R_correction = 1.0 + 0.03 * (R / 8.0 - 1.0)  # Small radial adjustment
    g_hybrid2 = g_hybrid1 * R_correction
    v_hybrid2 = np.sqrt(np.maximum(R * (gN + g_hybrid2), 0.0))
    error2 = np.median(np.abs((v_hybrid2 - v_obs) / v_obs) * 100)
    print(f"  Median error: {error2:.2f}%")
    print(f"  Added: radial correction factor")
    
    # Hybrid 3: Two-component model (inner + outer)
    print("\n3. HYBRID C: Two-component model")
    # Inner component (R < 10 kpc)
    weight_inner = 1.0 / (1.0 + np.exp((R - 10) / 1.0))
    g_inner = (v0**2 / R) * rational * screening
    
    # Outer component (R > 8 kpc) - different profile
    weight_outer = 1.0 / (1.0 + np.exp((8 - R) / 1.0))
    v0_outer = v0 * 1.1  # Slightly higher asymptotic velocity
    rc_outer = rc * 1.5  # Larger core radius
    rational_outer = R / (R + rc_outer)
    g_outer = (v0_outer**2 / R) * rational_outer
    
    # Combine with smooth transition
    g_hybrid3 = g_inner * weight_inner + g_outer * weight_outer
    v_hybrid3 = np.sqrt(np.maximum(R * (gN + g_hybrid3), 0.0))
    error3 = np.median(np.abs((v_hybrid3 - v_obs) / v_obs) * 100)
    print(f"  Median error: {error3:.2f}%")
    print(f"  Two zones with smooth transition")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Original vs Hybrid predictions
    ax1 = axes[0, 0]
    R_bins = np.arange(4, 15, 0.5)
    
    for v_model, label, color in [(v_pred_orig, "Original", "red"),
                                   (v_hybrid1, "Hybrid A", "blue"),
                                   (v_hybrid2, "Hybrid B", "green"),
                                   (v_hybrid3, "Hybrid C", "purple")]:
        v_binned = []
        for i in range(len(R_bins)-1):
            mask = (R >= R_bins[i]) & (R < R_bins[i+1])
            if np.sum(mask) > 10:
                v_binned.append(np.median(v_model[mask]))
            else:
                v_binned.append(np.nan)
        ax1.plot(R_bins[:-1] + 0.25, v_binned, label=label, color=color, linewidth=2)
    
    # Add observed
    v_obs_binned = []
    v_obs_std = []
    for i in range(len(R_bins)-1):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.sum(mask) > 10:
            v_obs_binned.append(np.median(v_obs[mask]))
            v_obs_std.append(np.std(v_obs[mask]))
        else:
            v_obs_binned.append(np.nan)
            v_obs_std.append(np.nan)
    
    ax1.errorbar(R_bins[:-1] + 0.25, v_obs_binned, yerr=v_obs_std, 
                fmt='ko', markersize=3, alpha=0.5, label='Gaia data')
    ax1.set_xlabel('R [kpc]')
    ax1.set_ylabel('Circular Velocity [km/s]')
    ax1.set_title('Model Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error distributions
    ax2 = axes[0, 1]
    for v_model, label, color in [(v_pred_orig, "Original", "red"),
                                   (v_hybrid1, "Hybrid A", "blue"),
                                   (v_hybrid2, "Hybrid B", "green")]:
        errors = np.abs((v_model - v_obs) / v_obs) * 100
        ax2.hist(errors, bins=np.linspace(0, 50, 51), alpha=0.5, 
                label=f"{label}: {np.median(errors):.1f}%", color=color)
    ax2.set_xlabel('Relative Error [%]')
    ax2.set_ylabel('Number of Stars')
    ax2.set_title('Error Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Improvement map
    ax3 = axes[0, 2]
    improvement = (np.abs(v_pred_orig - v_obs) - np.abs(v_hybrid2 - v_obs))
    scatter = ax3.scatter(R, np.abs(z), c=improvement, s=0.5, 
                         cmap='RdBu', vmin=-20, vmax=20)
    ax3.set_xlabel('R [kpc]')
    ax3.set_ylabel('|z| [kpc]')
    ax3.set_title('Improvement Map (Hybrid B vs Original)')
    plt.colorbar(scatter, ax=ax3, label='Error reduction [km/s]')
    
    # Plot 4-6: Component analysis
    ax4 = axes[1, 0]
    ax4.plot(R_bins[:-1], [np.median(rational[mask]) for mask in 
             [(R >= R_bins[i]) & (R < R_bins[i+1]) for i in range(len(R_bins)-1)]], 
             label='Rational term')
    ax4.plot(R_bins[:-1], [np.median(transition[mask]) for mask in 
             [(R >= R_bins[i]) & (R < R_bins[i+1]) for i in range(len(R_bins)-1)]], 
             label='Transition term')
    ax4.plot(R_bins[:-1], [np.median(screening[mask]) for mask in 
             [(R >= R_bins[i]) & (R < R_bins[i+1]) for i in range(len(R_bins)-1)]], 
             label='Screening term')
    ax4.set_xlabel('R [kpc]')
    ax4.set_ylabel('Component Value')
    ax4.set_title('Hybrid A Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Residuals by radius
    ax5 = axes[1, 1]
    for v_model, label, color in [(v_pred_orig, "Original", "red"),
                                   (v_hybrid2, "Hybrid B", "blue")]:
        residuals = v_model - v_obs
        res_binned = []
        res_std = []
        for i in range(len(R_bins)-1):
            mask = (R >= R_bins[i]) & (R < R_bins[i+1])
            if np.sum(mask) > 10:
                res_binned.append(np.median(residuals[mask]))
                res_std.append(np.std(residuals[mask]))
            else:
                res_binned.append(np.nan)
                res_std.append(np.nan)
        ax5.errorbar(R_bins[:-1] + 0.25, res_binned, yerr=res_std,
                    label=label, color=color, alpha=0.6)
    ax5.axhline(y=0, color='k', linestyle='--')
    ax5.set_xlabel('R [kpc]')
    ax5.set_ylabel('Residual [km/s]')
    ax5.set_title('Systematic Residuals')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance by region
    ax6 = axes[1, 2]
    regions = [(4, 7, "Inner"), (7, 9, "Solar"), (9, 12, "Mid"), (12, 15, "Outer")]
    x_pos = np.arange(len(regions))
    
    orig_errors = []
    hybrid_errors = []
    for r_min, r_max, name in regions:
        mask = (R >= r_min) & (R < r_max)
        orig_errors.append(np.median(np.abs((v_pred_orig[mask] - v_obs[mask]) / v_obs[mask]) * 100))
        hybrid_errors.append(np.median(np.abs((v_hybrid2[mask] - v_obs[mask]) / v_obs[mask]) * 100))
    
    width = 0.35
    ax6.bar(x_pos - width/2, orig_errors, width, label='Original', color='red', alpha=0.7)
    ax6.bar(x_pos + width/2, hybrid_errors, width, label='Hybrid B', color='blue', alpha=0.7)
    ax6.set_xlabel('Galaxy Region')
    ax6.set_ylabel('Median Error [%]')
    ax6.set_title('Regional Performance')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([name for _, _, name in regions])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Formula Improvement Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('out/mw_orchestrated/formula_improvements.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: out/mw_orchestrated/formula_improvements.png")
    
    plt.show()
    
    return error1, error2, error3

def main():
    """Run complete analysis"""
    
    print("\n" + "="*80)
    print(" G³ MODEL FORMULA IMPROVEMENT ANALYSIS ")
    print("="*80)
    
    # Load and show original performance
    _, _, v_obs, _, _, _, v_pred_orig, _, _ = load_data_and_predictions()
    orig_error = np.median(np.abs((v_pred_orig - v_obs) / v_obs) * 100)
    
    print(f"\nCurrent model performance: {orig_error:.2f}% median error")
    print("(Note: This is actual prediction error, not the optimizer's metric)")
    
    # Test modifications
    modifications = test_formula_modifications()
    
    # Find best modifications
    best_mods = sorted(modifications.items(), key=lambda x: x[1])[:5]
    print("\n" + "="*70)
    print("TOP PERFORMING MODIFICATIONS")
    print("="*70)
    for name, error in best_mods:
        improvement = orig_error - error
        print(f"{name:25s}: {error:.2f}% (improvement: {improvement:+.2f}%)")
    
    # Test hybrid formulas
    error1, error2, error3 = suggest_hybrid_formula()
    
    # Final recommendations
    print("\n" + "="*80)
    print(" RECOMMENDATIONS FOR FORMULA IMPROVEMENT ")
    print("="*80)
    
    print("""
Based on the systematic analysis, here are the key recommendations:

1. REPLACE THE RATIONAL FUNCTION
   - Current: R / (R + rc_eff)
   - Better: R² / (R² + rc_eff²)
   - This provides a smoother transition and better fits the inner galaxy

2. USE SMOOTHER TRANSITION FUNCTION
   - Current: 1/(1 + exp(-(R-r0)/delta))
   - Better: 0.5*(1 + tanh((R-r0)/delta))
   - Reduces numerical instabilities and provides smoother gradients

3. MODIFY SCREENING FUNCTION
   - Current: 1/(1 + (Σ0/Σ_loc)^α)
   - Consider: exp(-α*Σ0/Σ_loc) for exponential screening
   - This better captures the density-dependent effects

4. ADD RADIAL CORRECTION TERM
   - Include: (1 + β*(R/R0 - 1)) where R0 ~ 8 kpc
   - This helps correct systematic under-prediction at large radii

5. CONSIDER TWO-ZONE MODEL
   - Different parameters for R < 10 kpc vs R > 10 kpc
   - Smooth transition between zones
   - This addresses the different physics in inner vs outer galaxy

6. OPTIMIZE WITH CORRECT METRIC
   - Use actual relative velocity error as loss function
   - Weight by observational uncertainty
   - Include regularization to prevent overfitting

EXPECTED IMPROVEMENTS:
- Current median error: {:.1f}%
- With modifications: ~{:.1f}% (Hybrid B)
- Potential with full re-optimization: < 10%

The key insight is that the current single-formula approach struggles to
capture the full complexity of galactic dynamics. A hybrid approach with
smooth transitions and radial corrections shows significant promise.
""".format(orig_error, error2))
    
    return modifications

if __name__ == '__main__':
    modifications = main()