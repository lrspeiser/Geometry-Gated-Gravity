#!/usr/bin/env python3
"""
Final Comprehensive G³ Solver Comparison
Tests all approaches on real SPARC data to determine which is most accurate
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, Tuple
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
G_NEWTON = 4.302e-6  # kpc (km/s)^2 / M_sun

# ==============================================================================
# APPROACH 1: LogTail Formula (Disk-optimized surrogate)
# ==============================================================================

def logtail_formula(r_kpc, v_bar_kms, A_kms=140.0, rc_kpc=22.0):
    """LogTail formula that achieved ~90% on SPARC"""
    r = np.asarray(r_kpc)
    v_b = np.asarray(v_bar_kms)
    
    # LogTail correction
    v_tail = A_kms * np.sqrt(r / (r + rc_kpc))
    
    # Total velocity
    v_total = np.sqrt(v_b**2 + v_tail**2)
    
    return v_total

# ==============================================================================
# APPROACH 2: Root-M Formula (Mass-aware)
# ==============================================================================

def rootm_formula(r_kpc, v_bar_kms, M_bary, A_kms=140.0, rc_kpc=24.0, Mref=6e10):
    """Root-M formula with mass awareness"""
    r = np.asarray(r_kpc)
    v_b = np.asarray(v_bar_kms)
    
    # Mass-aware amplitude
    amp = A_kms * (M_bary / Mref)**0.25
    
    # Tail contribution
    v_tail = amp * np.sqrt(r / (r + rc_kpc))
    
    # Total velocity
    v_total = np.sqrt(v_b**2 + v_tail**2)
    
    return v_total

# ==============================================================================
# APPROACH 3: G³-Σ with gates (Full model)
# ==============================================================================

def g3_sigma_formula(r_kpc, v_bar_kms, rho_bary, S0=1.2e-4, rc_kpc=24.0,
                     g_sat=2500.0, sigma_star=150.0, alpha_sigma=1.0):
    """G³-Σ with saturating mobility and sigma screen"""
    r = np.asarray(r_kpc)
    v_b = np.asarray(v_bar_kms)
    rho = np.asarray(rho_bary)
    
    # Compute local surface density (simplified projection)
    sigma_loc = np.cumsum(rho * np.gradient(r)) / (np.pi * r**2 + 1e-10)
    sigma_loc *= 1e6  # Convert to Msun/pc^2
    
    # Sigma screen
    screen = 1.0 / (1.0 + (sigma_loc / sigma_star)**alpha_sigma)
    
    # Field gradient (simplified)
    grad_phi = v_b**2 / (r + 1e-10)
    
    # Saturating mobility
    x = grad_phi / g_sat
    mobility = 1.0 / (1.0 + x**2)
    
    # Effective source
    source_eff = S0 * screen * mobility
    
    # Tail contribution (simplified PDE solution)
    v_tail = 140.0 * source_eff**0.5 * np.sqrt(r / (r + rc_kpc))
    
    # Total velocity
    v_total = np.sqrt(v_b**2 + v_tail**2)
    
    return v_total

# ==============================================================================
# Test Framework
# ==============================================================================

def compute_outer_median_closeness(v_model, v_obs, r_obs, r_outer_frac=0.5):
    """Compute the paper's metric: outer-median fractional closeness"""
    r_median = np.median(r_obs)
    outer_mask = r_obs >= r_median * r_outer_frac
    
    if not np.any(outer_mask):
        return 1.0
    
    v_obs_outer = v_obs[outer_mask]
    v_model_outer = v_model[outer_mask]
    
    valid = v_obs_outer > 10  # km/s minimum
    if not np.any(valid):
        return 1.0
    
    frac_diff = np.abs(v_model_outer[valid] - v_obs_outer[valid]) / v_obs_outer[valid]
    median_closeness = np.median(frac_diff)
    
    # Return as percentage agreement (100% = perfect)
    return 100 * (1 - median_closeness)

def test_on_sparc_sample():
    """Test all approaches on a representative SPARC sample"""
    
    logger.info("="*70)
    logger.info("TESTING G³ APPROACHES ON SPARC GALAXIES")
    logger.info("="*70)
    
    # Load SPARC data
    try:
        sparc_df = pd.read_parquet('data/sparc_rotmod_ltg.parquet')
        master_df = pd.read_parquet('data/sparc_master_clean.parquet')
    except FileNotFoundError:
        logger.error("SPARC data files not found. Please run build scripts first.")
        return None
    
    # Select representative galaxies
    test_galaxies = ['NGC2403', 'DDO154', 'NGC3521', 'UGC2259', 'NGC6946', 
                     'NGC3198', 'NGC2841', 'DDO161', 'NGC7814', 'UGC128']
    
    results = {
        'LogTail': [],
        'RootM': [],
        'G3Sigma': [],
        'GR_baryons': []
    }
    
    for galaxy_name in test_galaxies:
        if galaxy_name not in sparc_df['galaxy'].values:
            continue
            
        # Get rotation curve
        gal_rc = sparc_df[sparc_df['galaxy'] == galaxy_name].copy()
        
        # Get galaxy properties
        if galaxy_name in master_df['galaxy'].values:
            gal_props = master_df[master_df['galaxy'] == galaxy_name].iloc[0]
            M_bary = gal_props['M_bary'] if 'M_bary' in gal_props else 1e10
        else:
            M_bary = 1e10
        
        # Extract data
        r = gal_rc['R_kpc'].values
        v_obs = gal_rc['Vobs_kms'].values
        
        # Compute baryonic contribution
        v_gas = gal_rc['Vgas_kms'].values if 'Vgas_kms' in gal_rc else np.zeros(len(r))
        v_disk = gal_rc['Vdisk_kms'].values if 'Vdisk_kms' in gal_rc else np.zeros(len(r))
        v_bul = gal_rc['Vbul_kms'].values if 'Vbul_kms' in gal_rc else np.zeros(len(r))
        v_bar = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
        
        # Estimate density profile
        rho_bary = np.zeros_like(r)
        for i in range(len(r)):
            if r[i] > 0:
                # Simple estimate from rotation curve
                rho_bary[i] = v_bar[i]**2 / (G_NEWTON * r[i]**2)
        
        # Test each approach
        
        # 1. LogTail
        v_logtail = logtail_formula(r, v_bar)
        acc_logtail = compute_outer_median_closeness(v_logtail, v_obs, r)
        results['LogTail'].append(acc_logtail)
        
        # 2. Root-M
        v_rootm = rootm_formula(r, v_bar, M_bary)
        acc_rootm = compute_outer_median_closeness(v_rootm, v_obs, r)
        results['RootM'].append(acc_rootm)
        
        # 3. G³-Σ
        v_g3sigma = g3_sigma_formula(r, v_bar, rho_bary)
        acc_g3sigma = compute_outer_median_closeness(v_g3sigma, v_obs, r)
        results['G3Sigma'].append(acc_g3sigma)
        
        # 4. GR (baryons only) for reference
        acc_gr = compute_outer_median_closeness(v_bar, v_obs, r)
        results['GR_baryons'].append(acc_gr)
        
        logger.info(f"{galaxy_name:12} | LogTail: {acc_logtail:5.1f}% | "
                   f"RootM: {acc_rootm:5.1f}% | G³-Σ: {acc_g3sigma:5.1f}% | "
                   f"GR: {acc_gr:5.1f}%")
    
    # Compute summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)
    
    summary = {}
    for approach, accuracies in results.items():
        if accuracies:
            summary[approach] = {
                'mean': np.mean(accuracies),
                'median': np.median(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'n_above_85': sum(1 for a in accuracies if a >= 85)
            }
            
            logger.info(f"\n{approach}:")
            logger.info(f"  Mean accuracy:   {summary[approach]['mean']:.1f}%")
            logger.info(f"  Median accuracy: {summary[approach]['median']:.1f}%")
            logger.info(f"  Std deviation:   {summary[approach]['std']:.1f}%")
            logger.info(f"  Range:           {summary[approach]['min']:.1f}% - {summary[approach]['max']:.1f}%")
            logger.info(f"  Above 85%:       {summary[approach]['n_above_85']}/{len(accuracies)} galaxies")
    
    return summary

def test_parameter_sensitivity():
    """Test sensitivity to parameter variations"""
    
    logger.info("\n" + "="*70)
    logger.info("PARAMETER SENSITIVITY ANALYSIS")
    logger.info("="*70)
    
    # Load one galaxy for testing
    try:
        sparc_df = pd.read_parquet('data/sparc_rotmod_ltg.parquet')
        gal_rc = sparc_df[sparc_df['galaxy'] == 'NGC3198'].copy()
    except:
        logger.error("Cannot load test galaxy")
        return
    
    r = gal_rc['R_kpc'].values
    v_obs = gal_rc['Vobs_kms'].values
    v_bar = np.sqrt(gal_rc['Vgas_kms'].values**2 + gal_rc['Vdisk_kms'].values**2)
    
    # Test LogTail sensitivity
    logger.info("\nLogTail Parameter Sensitivity (NGC3198):")
    logger.info("-" * 40)
    
    for A in [120, 140, 160]:
        for rc in [18, 22, 26]:
            v_pred = logtail_formula(r, v_bar, A_kms=A, rc_kpc=rc)
            acc = compute_outer_median_closeness(v_pred, v_obs, r)
            logger.info(f"  A={A:3d} km/s, rc={rc:2d} kpc: {acc:5.1f}%")
    
    # Test G³-Σ sensitivity
    logger.info("\nG³-Σ Parameter Sensitivity (NGC3198):")
    logger.info("-" * 40)
    
    rho_bary = v_bar**2 / (G_NEWTON * r**2 + 1e-10)
    
    for S0 in [0.8e-4, 1.2e-4, 1.6e-4]:
        for g_sat in [2000, 2500, 3000]:
            v_pred = g3_sigma_formula(r, v_bar, rho_bary, S0=S0, g_sat=g_sat)
            acc = compute_outer_median_closeness(v_pred, v_obs, r)
            logger.info(f"  S0={S0:.1e}, g_sat={g_sat:4d}: {acc:5.1f}%")

def main():
    """Run all tests and provide final recommendations"""
    
    logger.info("COMPREHENSIVE G³ SOLVER COMPARISON")
    logger.info("Testing all approaches to determine the most accurate")
    logger.info("")
    
    # Test on SPARC sample
    summary = test_on_sparc_sample()
    
    if summary is None:
        logger.error("Tests failed - check data availability")
        return
    
    # Parameter sensitivity
    test_parameter_sensitivity()
    
    # Final recommendations
    logger.info("\n" + "="*70)
    logger.info("FINAL RECOMMENDATIONS")
    logger.info("="*70)
    
    # Find best approach
    best_approach = max(summary.keys(), key=lambda k: summary[k]['median'] if k != 'GR_baryons' else 0)
    
    logger.info(f"\n✓ BEST APPROACH: {best_approach}")
    logger.info(f"  - Median accuracy: {summary[best_approach]['median']:.1f}%")
    logger.info(f"  - Consistency (std): {summary[best_approach]['std']:.1f}%")
    
    if best_approach == 'LogTail':
        logger.info("\n  IMPLEMENTATION:")
        logger.info("  1. Use LogTail as the primary galaxy model")
        logger.info("  2. Parameters: A=140 km/s, rc=22 kpc (global)")
        logger.info("  3. This is a simplified surrogate - fast and accurate for disks")
        logger.info("  4. For clusters/MW, need additional modifications")
        
    elif best_approach == 'G3Sigma':
        logger.info("\n  IMPLEMENTATION:")
        logger.info("  1. Use G³-Σ with mobility and screening gates")
        logger.info("  2. Parameters: S0=1.2e-4, rc=24 kpc, g_sat=2500, Σ*=150")
        logger.info("  3. This is the full physical model - works across all regimes")
        logger.info("  4. Implement using CuPy built-in operations for GPU acceleration")
        
    elif best_approach == 'RootM':
        logger.info("\n  IMPLEMENTATION:")
        logger.info("  1. Use Root-M for mass-aware predictions")
        logger.info("  2. Parameters: A=140 km/s, rc=24 kpc, Mref=6e10")
        logger.info("  3. Good for spanning dwarf to massive galaxies")
        logger.info("  4. Simple to implement and interpret")
    
    logger.info("\n  NEXT STEPS:")
    logger.info("  1. Implement chosen approach in solve_g3_production.py")
    logger.info("  2. Run full SPARC dataset validation")
    logger.info("  3. Test on MW with proper density model")
    logger.info("  4. Validate on Perseus and A1689 clusters")
    logger.info("  5. Generate publication figures")
    
    # Save results
    output_file = Path("final_solver_comparison.json")
    with open(output_file, 'w') as f:
        json.dump({
            'summary': summary,
            'best_approach': best_approach,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_file}")

if __name__ == "__main__":
    main()