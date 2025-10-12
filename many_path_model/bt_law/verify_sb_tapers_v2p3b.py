#!/usr/bin/env python3
"""
V2.3b SB Taper Verification Micro-Suite
=======================================

Tests differentiated SAB/SB bar parameters on small subset to verify:
1. Parameter loading works correctly (SAB vs SB get different values)
2. R_bar placement is correct (SAB=2.0*R_d, SB=1.5*R_d)
3. Suppression strength is correct (SAB gamma=1.5, SB gamma=2.5)
4. Performance improves for SB without harming SAB/early types

Target SB galaxies for testing (known problematic):
- NGC0891, NGC2903, NGC3953, NGC4088, NGC5371 (intermediate/early SB)
- NGC3992, NGC4051, NGC5055, NGC7331 (various SB types)

Acceptance criteria:
- SB subset improves ≥7 points vs V2.2/V2.3
- R_bar values print correctly (SB should be ~1.5*R_d, not 2.0)
- No crashes, clean parameter resolution
"""
import sys
from pathlib import Path
import json
import numpy as np

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

from sparc_stratified_test import load_sparc_master_table, load_sparc_galaxy

# Import B/T law utilities
sys.path.insert(0, str(Path(__file__).parent))
from bt_laws import morph_to_bt
from bt_laws_v2p3 import (load_theta, eval_all_laws_v2p3, ring_radial_envelope,
                          bar_radial_taper)


def compute_galaxy_ape_v2p3(galaxy, params: dict) -> float:
    """Compute APE using V2.3 model with radial tapers."""
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
    kappa = params.get('kappa', 0.7)
    
    R0 = params.get('R0', 5.0)
    R1 = params.get('R1', 70.0)
    p = params.get('p', 2.0)
    q = params.get('q', 3.5)
    
    R_ring = params.get('R_ring', 8.0)
    sigma_ring = params.get('sigma_ring', 0.5)
    
    # Gate
    R_gate = 0.5
    p_gate = 4.0
    gate = 1.0 - cp.exp(-(r / R_gate)**p_gate)
    
    # Growth with saturation
    f_d = (r / R0)**p / (1.0 + (r / R1)**q)
    
    # Ring winding term
    x = (2.0 * cp.pi * r) / lambda_ring
    ex = cp.exp(-x)
    ring_term_base = ring_amp * (ex / cp.maximum(1e-20, 1.0 - kappa * ex))
    
    # Radial concentration envelope
    r_np = cp.asnumpy(r) if HAS_CUPY else r
    envelope = ring_radial_envelope(r_np, R_ring, sigma_ring)
    envelope_gpu = cp.array(envelope, dtype=cp.float32) if HAS_CUPY else envelope
    
    ring_term_base = ring_term_base * envelope_gpu
    
    # V2.3: Bar taper (if present)
    R_bar = params.get('R_bar')
    if R_bar is not None:
        w_bar = params.get('w_bar', 1.0)
        gamma_bar = params.get('gamma_bar_taper', 1.5)
        bar_taper_vals = bar_radial_taper(r_np, R_bar, w_bar, gamma_bar)
        bar_taper_gpu = cp.array(bar_taper_vals, dtype=cp.float32) if HAS_CUPY else bar_taper_vals
        ring_term_base = ring_term_base * bar_taper_gpu
    
    # V2.3: Shear taper (if present)
    R_shear = params.get('R_shear')
    if R_shear is not None:
        from bt_laws_v2p3 import shear_radial_taper
        w_shear = params.get('w_shear', 1.0)
        gamma_shear = params.get('gamma_shear_taper', 1.0)
        shear_taper_vals = shear_radial_taper(r_np, R_shear, w_shear, gamma_shear)
        shear_taper_gpu = cp.array(shear_taper_vals, dtype=cp.float32) if HAS_CUPY else shear_taper_vals
        ring_term_base = ring_term_base * shear_taper_gpu
    
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
    print("="*80)
    print("V2.3b SB TAPER VERIFICATION MICRO-SUITE")
    print("="*80)
    
    # Test galaxies (SB subset + SAB control)
    test_galaxies = {
        'SB_targets': [
            'NGC0891', 'NGC2903', 'NGC3953', 'NGC4088', 'NGC5371',
            'NGC3992', 'NGC4051', 'NGC5055', 'NGC7331'
        ],
        'SAB_control': ['UGC02916', 'UGC02953', 'NGC7814'],
        'Early_control': ['NGC2841', 'NGC6674']
    }
    
    # Load parameters
    v2p2_params_file = Path('many_path_model/bt_law/bt_law_params_v2p2_initial.json')
    v2p3_params_file = Path('many_path_model/bt_law/bt_law_params_v2p3_initial.json')
    v2p3b_params_file = Path('many_path_model/bt_law/bt_law_params_v2p3b.json')
    
    print(f"\nLoading V2.2 baseline from: {v2p2_params_file}")
    theta_v2p2 = load_theta(v2p2_params_file)
    
    print(f"Loading V2.3 (unified) from: {v2p3_params_file}")
    theta_v2p3 = load_theta(v2p3_params_file)
    
    print(f"Loading V2.3b (differentiated) from: {v2p3b_params_file}")
    theta_v2p3b = load_theta(v2p3b_params_file)
    
    # Load disk params, shear, bar classifications
    with open('many_path_model/bt_law/sparc_disk_params.json', 'r') as f:
        disk_params = json.load(f)
    
    with open('many_path_model/bt_law/sparc_shear_predictors.json', 'r') as f:
        shear_preds = json.load(f)
    
    with open('many_path_model/bt_law/sparc_bar_classification.json', 'r') as f:
        bar_classifications = json.load(f)
    
    # Load per-galaxy best fits for comparison
    with open('results/mega_test/mega_parallel_results.json', 'r') as f:
        per_galaxy_data = json.load(f)
    
    per_galaxy_ape = {}
    for result in per_galaxy_data['results']:
        if result.get('success', False):
            per_galaxy_ape[result['name']] = result['best_error']
    
    # Load SPARC data
    master_info = load_sparc_master_table(Path('data/SPARC_Lelli2016c.mrt'))
    sparc_dir = Path('data/Rotmod_LTG')
    
    # Results storage
    results = []
    
    print("\n" + "="*80)
    print("TESTING DIFFERENTIATED TAPERS")
    print("="*80)
    
    for group_name, galaxy_list in test_galaxies.items():
        print(f"\n{'='*80}")
        print(f"{group_name.upper()}")
        print(f"{'='*80}")
        
        for galaxy_name in galaxy_list:
            rotmod_file = sparc_dir / f"{galaxy_name}_rotmod.dat"
            
            if not rotmod_file.exists():
                print(f"[SKIP] {galaxy_name:12s} - file not found")
                continue
            
            try:
                # Load galaxy
                galaxy = load_sparc_galaxy(rotmod_file, master_info)
                
                # Get parameters
                gal_disk = disk_params.get(galaxy_name, {})
                hubble_type = gal_disk.get('hubble_type', 'Unknown')
                type_group = gal_disk.get('type_group', 'unknown')
                R_d = gal_disk.get('R_d_kpc')
                Sigma0 = gal_disk.get('Sigma0')
                
                gal_shear = shear_preds.get(galaxy_name, {})
                shear = gal_shear.get('shear_2p2Rd')
                
                gal_bar = bar_classifications.get(galaxy_name, {})
                bar_class = gal_bar.get('bar_class', 'unknown')
                g_bar = gal_bar.get('g_bar')
                
                B_T = morph_to_bt(hubble_type, type_group)
                
                # Evaluate all three versions
                # V2.2: Use V2.2 laws (pre-taper, with bar gating only)
                from bt_laws_v2p2 import eval_all_laws_v2p2
                v2p2_params = eval_all_laws_v2p2(B_T, theta_v2p2, Sigma0=Sigma0, R_d=R_d,
                                                 shear=shear, g_bar_in=g_bar)
                # V2.3/V2.3b: Use V2.3 laws (with radial tapers)
                v2p3_params = eval_all_laws_v2p3(B_T, theta_v2p3, Sigma0=Sigma0, R_d=R_d,
                                                 shear=shear, g_bar_in=g_bar, bar_class=bar_class)
                v2p3b_params = eval_all_laws_v2p3(B_T, theta_v2p3b, Sigma0=Sigma0, R_d=R_d,
                                                  shear=shear, g_bar_in=g_bar, bar_class=bar_class)
                
                v2p2_ape = compute_galaxy_ape_v2p3(galaxy, v2p2_params)
                v2p3_ape = compute_galaxy_ape_v2p3(galaxy, v2p3_params)
                v2p3b_ape = compute_galaxy_ape_v2p3(galaxy, v2p3b_params)
                
                per_gal_ape = per_galaxy_ape.get(galaxy_name, None)
                
                # Print comparison
                print(f"\n{galaxy_name:12s} ({type_group:12s}) {bar_class:3s}  R_d={R_d:.1f} kpc")
                print(f"  V2.2:  APE={v2p2_ape:6.2f}%")
                print(f"  V2.3:  APE={v2p3_ape:6.2f}%  (Δ={v2p3_ape-v2p2_ape:+6.2f}%)")
                print(f"  V2.3b: APE={v2p3b_ape:6.2f}%  (Δ={v2p3b_ape-v2p2_ape:+6.2f}%)")
                if per_gal_ape:
                    print(f"  Best:  APE={per_gal_ape:6.2f}%")
                
                # Print taper parameters to verify differentiation
                print(f"  V2.3  bar taper: R_bar={v2p3_params.get('R_bar')}, γ={v2p3_params.get('gamma_bar_taper', 0):.2f}")
                print(f"  V2.3b bar taper: R_bar={v2p3b_params.get('R_bar')}, γ={v2p3b_params.get('gamma_bar_taper', 0):.2f}")
                
                # Verify differentiation worked
                if bar_class == 'SB':
                    expected_R_bar = 1.5 * R_d if R_d else None
                    actual_R_bar = v2p3b_params.get('R_bar')
                    if actual_R_bar and expected_R_bar:
                        if abs(actual_R_bar - expected_R_bar) < 0.1:
                            print(f"  ✓ SB differentiation WORKING (R_bar={actual_R_bar:.1f} ≈ 1.5*{R_d:.1f})")
                        else:
                            print(f"  ✗ SB differentiation FAILED (R_bar={actual_R_bar:.1f} ≠ 1.5*{R_d:.1f})")
                
                elif bar_class == 'SAB':
                    expected_R_bar = 2.0 * R_d if R_d else None
                    actual_R_bar = v2p3b_params.get('R_bar')
                    if actual_R_bar and expected_R_bar:
                        if abs(actual_R_bar - expected_R_bar) < 0.1:
                            print(f"  ✓ SAB parameters MAINTAINED (R_bar={actual_R_bar:.1f} ≈ 2.0*{R_d:.1f})")
                
                results.append({
                    'name': galaxy_name,
                    'group': group_name,
                    'bar_class': bar_class,
                    'type_group': type_group,
                    'R_d': R_d,
                    'v2p2_ape': v2p2_ape,
                    'v2p3_ape': v2p3_ape,
                    'v2p3b_ape': v2p3b_ape,
                    'per_gal_ape': per_gal_ape,
                    'v2p3b_R_bar': v2p3b_params.get('R_bar'),
                    'v2p3b_gamma_bar': v2p3b_params.get('gamma_bar_taper', 0)
                })
                
            except Exception as e:
                print(f"[ERROR] {galaxy_name:12s} - {e}")
                continue
    
    # Summary statistics
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    sb_results = [r for r in results if r['bar_class'] == 'SB']
    sab_results = [r for r in results if r['bar_class'] == 'SAB']
    early_results = [r for r in results if r['type_group'] == 'early']
    
    if sb_results:
        print(f"\nSB Galaxies (n={len(sb_results)}):")
        sb_v2p2 = np.mean([r['v2p2_ape'] for r in sb_results])
        sb_v2p3 = np.mean([r['v2p3_ape'] for r in sb_results])
        sb_v2p3b = np.mean([r['v2p3b_ape'] for r in sb_results])
        print(f"  V2.2  mean: {sb_v2p2:.2f}%")
        print(f"  V2.3  mean: {sb_v2p3:.2f}%  (Δ={sb_v2p3-sb_v2p2:+.2f}%)")
        print(f"  V2.3b mean: {sb_v2p3b:.2f}%  (Δ={sb_v2p3b-sb_v2p2:+.2f}%)")
        
        improvement = sb_v2p2 - sb_v2p3b
        if improvement >= 7.0:
            print(f"  ✓ ACCEPTANCE CRITERION MET: Improvement = {improvement:.1f}% (≥7%)")
        else:
            print(f"  ✗ ACCEPTANCE CRITERION NOT MET: Improvement = {improvement:.1f}% (<7%)")
    
    if sab_results:
        print(f"\nSAB Galaxies (n={len(sab_results)}) [Control]:")
        sab_v2p2 = np.mean([r['v2p2_ape'] for r in sab_results])
        sab_v2p3b = np.mean([r['v2p3b_ape'] for r in sab_results])
        print(f"  V2.2  mean: {sab_v2p2:.2f}%")
        print(f"  V2.3b mean: {sab_v2p3b:.2f}%  (Δ={sab_v2p3b-sab_v2p2:+.2f}%)")
        
        if sab_v2p3b <= sab_v2p2 + 2.0:
            print(f"  ✓ NO DEGRADATION: SAB maintained or improved")
        else:
            print(f"  ✗ DEGRADATION: SAB worsened by {sab_v2p3b-sab_v2p2:.1f}%")
    
    if early_results:
        print(f"\nEarly Types (n={len(early_results)}) [Control]:")
        early_v2p2 = np.mean([r['v2p2_ape'] for r in early_results])
        early_v2p3b = np.mean([r['v2p3b_ape'] for r in early_results])
        print(f"  V2.2  mean: {early_v2p2:.2f}%")
        print(f"  V2.3b mean: {early_v2p3b:.2f}%  (Δ={early_v2p3b-early_v2p2:+.2f}%)")
        
        if early_v2p3b <= early_v2p2 + 2.0:
            print(f"  ✓ NO DEGRADATION: Early types maintained")
        else:
            print(f"  ✗ DEGRADATION: Early types worsened by {early_v2p3b-early_v2p2:.1f}%")
    
    print("\n" + "="*80)
    
    # Save results
    output_file = Path('results/bt_law_evaluation_v2p3b/sb_verification_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'test_description': 'V2.3b SB taper verification micro-suite',
            'acceptance_criteria': {
                'SB_improvement': '≥7 points vs V2.2',
                'SAB_maintenance': 'no degradation >2%',
                'early_maintenance': 'no degradation >2%'
            },
            'results': results
        }, f, indent=2)
    
    print(f"[OK] Results saved to: {output_file}")


if __name__ == '__main__':
    main()
