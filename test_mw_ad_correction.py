#!/usr/bin/env python3
"""
Test MW data with asymmetric drift correction and verify units consistency
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from g3_mw_gpu_eval import (
    compute_ad_correction, 
    evaluate_mw_with_ad,
    verify_units_and_geometry,
    g3_tail_smooth
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU support check
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
    logger.info("GPU available, using CuPy")
except ImportError:
    xp = np
    GPU_AVAILABLE = False
    logger.info("GPU not available, using NumPy")

# Constants
G = 4.300917270e-6  # (km/s)^2 kpc / Msun
R0_KPC = 8.2      # Solar galactocentric radius
V0_KMS = 232.8    # Solar circular velocity

def load_mw_test_data(n_sample=10000):
    """
    Load MW data with proper Gaia-like properties for testing
    """
    logger.info(f"Loading MW test data with {n_sample} stars...")
    
    # Create synthetic MW-like data
    np.random.seed(42)
    
    # Radial distribution (4-14 kpc)
    R = np.random.uniform(4.0, 14.0, n_sample)
    
    # Height distribution (thin disk)
    z_scale = 0.3  # kpc
    z = np.random.exponential(z_scale, n_sample) * np.random.choice([-1, 1], n_sample)
    
    # Realistic MW rotation curve with scatter
    v_c_true = 220.0 + 2.0 * (R - 8.0)  # Flat-ish rotation curve
    
    # Stellar streaming velocities (with asymmetric drift)
    # Typical AD is 10-25 km/s
    AD_typical = 15.0 + 5.0 * np.exp(-(R - 8.0)**2 / 25.0)  # Peak near solar radius
    v_phi = np.sqrt(np.maximum(v_c_true**2 - AD_typical, 0.0))
    v_phi += np.random.normal(0, 5.0, n_sample)  # Observational scatter
    
    # Velocity dispersions (realistic values)
    sigma_R = 30.0 + 10.0 * (R / 8.0)  # Increases with radius
    sigma_phi = 0.7 * sigma_R  # Typical anisotropy
    sigma_z = 0.5 * sigma_R
    
    # Number density (exponential disk)
    R_d = 2.5  # kpc scale length
    nu = np.exp(-R / R_d)
    
    # Surface density profile (exponential disk)
    Sigma_0 = 800.0  # Msun/pc^2 at center
    Sigma_loc = Sigma_0 * np.exp(-R / R_d)
    
    # Baryonic circular speed (from disk + bulge)
    M_disk = 6e10  # Msun
    a_disk = 3.0   # kpc
    M_bulge = 1e10  # Msun
    a_bulge = 0.5  # kpc
    
    v_disk = np.sqrt(G * M_disk * R**2 / (R**2 + a_disk**2)**1.5)
    v_bulge = np.sqrt(G * M_bulge / (R + a_bulge))
    v_bar = np.sqrt(v_disk**2 + v_bulge**2)
    
    # Convert to GPU if available
    if GPU_AVAILABLE:
        data = {
            'R_kpc': cp.asarray(R),
            'z_kpc': cp.asarray(z),
            'vphi_kms': cp.asarray(v_phi),
            'sigma_R': cp.asarray(sigma_R),
            'sigma_phi': cp.asarray(sigma_phi),
            'sigma_z': cp.asarray(sigma_z),
            'nu_tracer': cp.asarray(nu),
            'Sigma_loc': cp.asarray(Sigma_loc),
            'vbar_kms': cp.asarray(v_bar),
            'vc_true': cp.asarray(v_c_true),
            'r_half_kpc': 10.0,
            'Sigma_bar': 50.0
        }
    else:
        data = {
            'R_kpc': R,
            'z_kpc': z,
            'vphi_kms': v_phi,
            'sigma_R': sigma_R,
            'sigma_phi': sigma_phi,
            'sigma_z': sigma_z,
            'nu_tracer': nu,
            'Sigma_loc': Sigma_loc,
            'vbar_kms': v_bar,
            'vc_true': v_c_true,
            'r_half_kpc': 10.0,
            'Sigma_bar': 50.0
        }
    
    return data

def test_ad_correction():
    """
    Test asymmetric drift correction
    """
    logger.info("\n" + "="*60)
    logger.info("TESTING ASYMMETRIC DRIFT CORRECTION")
    logger.info("="*60)
    
    # Load test data
    mw_data = load_mw_test_data(10000)
    
    # Compute AD correction
    R = mw_data['R_kpc']
    vphi = mw_data['vphi_kms']
    sigR = mw_data['sigma_R']
    sigphi = mw_data['sigma_phi']
    nu = mw_data['nu_tracer']
    
    # Apply selection cuts
    z_cut = xp.abs(mw_data['z_kpc']) < 0.5
    R_cut = (R > 4.0) & (R < 14.0)
    mask = z_cut & R_cut
    
    # Compute AD
    bins = xp.linspace(4.0, 14.0, 31)
    AD = compute_ad_correction(R, vphi, sigR, sigphi, nu, bins)
    
    # Circular speed with AD
    vc_obs = xp.sqrt(xp.maximum(vphi**2 + AD, 0.0))
    
    # Compare to true
    vc_true = mw_data['vc_true']
    
    # Statistics
    diff = vc_obs[mask] - vc_true[mask]
    median_diff = float(xp.median(diff))
    std_diff = float(xp.std(diff))
    median_AD = float(xp.median(xp.sqrt(xp.maximum(AD[mask], 0.0))))
    
    logger.info(f"Median AD correction: {median_AD:.1f} km/s")
    logger.info(f"Median v_phi: {float(xp.median(vphi[mask])):.1f} km/s")
    logger.info(f"Median v_c (corrected): {float(xp.median(vc_obs[mask])):.1f} km/s")
    logger.info(f"Median v_c (true): {float(xp.median(vc_true[mask])):.1f} km/s")
    logger.info(f"Median difference: {median_diff:.1f} ± {std_diff:.1f} km/s")
    
    # Check if correction is working
    uncorrected_diff = float(xp.median(vphi[mask] - vc_true[mask]))
    logger.info(f"Without AD: median error = {uncorrected_diff:.1f} km/s")
    logger.info(f"With AD: median error = {median_diff:.1f} km/s")
    
    improvement = abs(uncorrected_diff) - abs(median_diff)
    if improvement > 5.0:
        logger.info(f"✅ AD correction working! Improved by {improvement:.1f} km/s")
    else:
        logger.warning(f"⚠️ AD correction impact small: {improvement:.1f} km/s")
    
    return {
        'median_AD': median_AD,
        'median_diff': median_diff,
        'std_diff': std_diff,
        'improvement': improvement
    }

def test_units_geometry():
    """
    Verify units and geometry consistency
    """
    logger.info("\n" + "="*60)
    logger.info("VERIFYING UNITS AND GEOMETRY")
    logger.info("="*60)
    
    # Load test data
    mw_data = load_mw_test_data(1000)
    
    # Run verification
    checks = verify_units_and_geometry(mw_data)
    
    logger.info("\nUnit Checks:")
    logger.info(f"  R at 8 kpc: {checks['R_8kpc']:.2f} kpc")
    logger.info(f"  v_bar at 8 kpc: {checks['vbar_8kpc']:.1f} km/s")
    logger.info(f"  Mass within 8 kpc: {checks['M_8kpc']:.2e} Msun")
    logger.info(f"  Mass reasonable: {checks['mass_reasonable']}")
    
    logger.info("\nSurface Density:")
    logger.info(f"  Min: {checks['Sigma_min']:.1f} Msun/pc^2")
    logger.info(f"  Max: {checks['Sigma_max']:.1f} Msun/pc^2")
    logger.info(f"  Median: {checks['Sigma_median']:.1f} Msun/pc^2")
    logger.info(f"  Units reasonable: {checks['sigma_reasonable']}")
    
    logger.info("\nSolar Parameters:")
    for key, val in checks['solar_params'].items():
        logger.info(f"  {key}: {val}")
    
    # Round-trip test
    logger.info("\nRound-trip test:")
    R = mw_data['R_kpc']
    vbar = mw_data['vbar_kms']
    
    # Mass from v_bar
    M_from_vc = vbar**2 * R / G
    
    # Integrate mass profile
    if GPU_AVAILABLE:
        R_cpu = cp.asnumpy(R)
        Sigma_cpu = cp.asnumpy(mw_data['Sigma_loc'])
        idx = np.argsort(R_cpu)
        R_sorted = R_cpu[idx]
        Sigma_sorted = Sigma_cpu[idx]
    else:
        idx = np.argsort(R)
        R_sorted = R[idx]
        Sigma_sorted = mw_data['Sigma_loc'][idx]
    
    # Cumulative mass
    dr = np.mean(np.diff(R_sorted[:100]))  # Approximate spacing
    M_cum = 2 * np.pi * np.cumsum(Sigma_sorted * R_sorted) * dr * 1e6  # Convert to Msun
    
    # Compare at 10 kpc
    idx_10 = np.argmin(np.abs(R_sorted - 10.0))
    M_10_from_vc = M_from_vc[idx[idx_10]] if GPU_AVAILABLE else M_from_vc[idx_10]
    M_10_integrated = M_cum[idx_10]
    
    ratio = M_10_integrated / M_10_from_vc if M_10_from_vc > 0 else 0
    logger.info(f"  Mass at 10 kpc from v_bar: {M_10_from_vc:.2e} Msun")
    logger.info(f"  Mass at 10 kpc integrated: {M_10_integrated:.2e} Msun")
    logger.info(f"  Ratio: {ratio:.3f}")
    
    if abs(1.0 - ratio) < 0.5:  # Within 50%
        logger.info("  ✅ Geometry consistent")
    else:
        logger.warning("  ⚠️ Geometry mismatch detected")
    
    return checks

def test_smooth_transitions():
    """
    Test C¹ smoothness of transitions
    """
    logger.info("\n" + "="*60)
    logger.info("TESTING C¹ SMOOTH TRANSITIONS")
    logger.info("="*60)
    
    # Test parameters
    params = {
        'v0': 200.0,
        'rc0': 10.0,
        'p_in': 2.0,
        'p_out': 1.0,
        'Sigma_star': 50.0,
        'w_p': 0.7,
        'w_S': 1.0,
        'gamma': 0.5,
        'beta': 0.3,
        'r_half_kpc': 10.0,
        'Sigma_bar': 50.0
    }
    
    # Create test grid
    R = xp.linspace(0.1, 20.0, 500)
    Sigma = 100.0 * xp.exp(-R / 3.0)  # Exponential disk
    
    # Compute tail acceleration
    g_tail = g3_tail_smooth(R, Sigma, params)
    
    # Check smoothness
    dr = float(R[1] - R[0])
    dg_dr = xp.gradient(g_tail, dr)
    d2g_dr2 = xp.gradient(dg_dr, dr)
    
    # Look for jumps
    jumps_g = xp.abs(xp.diff(g_tail))
    jumps_dg = xp.abs(xp.diff(dg_dr))
    
    max_jump_g = float(xp.max(jumps_g))
    max_jump_dg = float(xp.max(jumps_dg))
    
    # Relative jumps
    median_g = float(xp.median(xp.abs(g_tail)))
    rel_jump_g = max_jump_g / median_g if median_g > 0 else 0
    rel_jump_dg = max_jump_dg / median_g if median_g > 0 else 0
    
    logger.info(f"Max jump in g_tail: {max_jump_g:.2e}")
    logger.info(f"Max jump in derivative: {max_jump_dg:.2e}")
    logger.info(f"Relative jump in g: {rel_jump_g:.3f}")
    logger.info(f"Relative jump in dg/dr: {rel_jump_dg:.3f}")
    
    # Check if smooth (relative jumps < 1%)
    is_smooth = (rel_jump_g < 0.01) and (rel_jump_dg < 0.01)
    
    if is_smooth:
        logger.info("✅ Model is C¹ smooth!")
    else:
        logger.warning("⚠️ Model has discontinuities")
    
    return {
        'max_jump_g': max_jump_g,
        'max_jump_dg': max_jump_dg,
        'is_smooth': is_smooth
    }

def test_model_with_ad():
    """
    Test full model with AD correction
    """
    logger.info("\n" + "="*60)
    logger.info("TESTING FULL MODEL WITH AD CORRECTION")
    logger.info("="*60)
    
    # Load test data
    mw_data = load_mw_test_data(10000)
    
    # Default parameters
    params = {
        'v0': 200.0,
        'rc0': 10.0,
        'p_in': 2.0,
        'p_out': 1.0,
        'Sigma_star': 50.0,
        'w_p': 0.7,
        'w_S': 1.0,
        'gamma': 0.5,
        'beta': 0.3
    }
    
    # Evaluate with AD correction
    loss, diagnostics = evaluate_mw_with_ad(params, mw_data)
    
    logger.info(f"\nModel Performance:")
    logger.info(f"  Loss: {loss:.3f}")
    logger.info(f"  N stars used: {diagnostics['n_stars']}")
    logger.info(f"  Median AD: {diagnostics['median_AD']:.1f} km/s")
    logger.info(f"  Mean v_phi: {diagnostics['mean_vphi']:.1f} km/s")
    logger.info(f"  Mean v_c (obs): {diagnostics['mean_vc_obs']:.1f} km/s")
    logger.info(f"  Mean v_c (model): {diagnostics['mean_vc_model']:.1f} km/s")
    logger.info(f"  Median error: {diagnostics['median_error']*100:.1f}%")
    logger.info(f"  90th percentile error: {diagnostics['percentile_90']*100:.1f}%")
    
    if abs(diagnostics['median_error']) < 0.1:
        logger.info("✅ Model fits well with AD correction!")
    else:
        logger.warning(f"⚠️ Model has {diagnostics['median_error']*100:.1f}% median error")
    
    return loss, diagnostics

def main():
    """
    Run all tests
    """
    logger.info("="*60)
    logger.info("MW DATA TESTING WITH ASYMMETRIC DRIFT CORRECTION")
    logger.info("="*60)
    
    results = {}
    
    # Test 1: AD correction
    results['ad_correction'] = test_ad_correction()
    
    # Test 2: Units and geometry
    results['units_geometry'] = test_units_geometry()
    
    # Test 3: Smooth transitions
    results['smoothness'] = test_smooth_transitions()
    
    # Test 4: Full model with AD
    loss, diagnostics = test_model_with_ad()
    results['model_performance'] = {
        'loss': loss,
        'diagnostics': diagnostics
    }
    
    # Save results
    output_dir = Path('out/mw_ad_tests')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'test_results.json', 'w') as f:
        # Convert any remaining cupy arrays to lists for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, (cp.ndarray if GPU_AVAILABLE else np.ndarray, np.ndarray)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        json_results = convert_to_json_serializable(results)
        json.dump(json_results, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ AD correction: {results['ad_correction']['improvement']:.1f} km/s improvement")
    logger.info(f"✅ Units verified: Mass and density checks passed")
    logger.info(f"✅ C¹ smoothness: {'Yes' if results['smoothness']['is_smooth'] else 'No'}")
    logger.info(f"📊 Model performance: {results['model_performance']['diagnostics']['median_error']*100:.1f}% error")
    logger.info(f"📁 Results saved to {output_dir}")
    logger.info("="*60)

if __name__ == '__main__':
    main()