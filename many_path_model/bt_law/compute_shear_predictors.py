#!/usr/bin/env python3
"""
Compute Shear and Additional Predictors from SPARC Rotation Curves
===================================================================

Extracts:
- Shear S = -d(ln Ω)/d(ln R) at R ~ 2.2 R_d
- Compactness C = V_max^2 R_d / (G M_bary)
- V_max and R at V_max

These predictors will gate ring coherence and amplitude.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from sparc_stratified_test import load_sparc_master_table, load_sparc_galaxy


def compute_shear_at_radius(r_kpc, v_kms, R_target_kpc):
    """
    Compute shear S = -d(ln Ω)/d(ln R) at target radius.
    
    Ω = V/R, so ln Ω = ln V - ln R
    d(ln Ω)/d(ln R) = d(ln V)/d(ln R) - 1
    
    Shear S = -d(ln Ω)/d(ln R) = 1 - d(ln V)/d(ln R)
    
    For a flat curve: d(ln V)/d(ln R) = 0, so S = 1
    For a rising curve: d(ln V)/d(ln R) > 0, so S < 1
    For a declining curve: d(ln V)/d(ln R) < 0, so S > 1
    """
    # Find index closest to target
    idx = np.argmin(np.abs(r_kpc - R_target_kpc))
    
    # Use 5-point stencil for smooth derivative
    window = 2
    i0 = max(0, idx - window)
    i1 = min(len(r_kpc), idx + window + 1)
    
    if i1 - i0 < 3:
        return np.nan
    
    r_win = r_kpc[i0:i1]
    v_win = v_kms[i0:i1]
    
    # Mask invalid values
    valid = (r_win > 0) & (v_win > 0)
    if valid.sum() < 3:
        return np.nan
    
    r_win = r_win[valid]
    v_win = v_win[valid]
    
    # Compute d(ln V)/d(ln R) via finite difference
    log_r = np.log(r_win)
    log_v = np.log(v_win)
    
    # Linear fit in log-log space around target
    if len(log_r) >= 3:
        p = np.polyfit(log_r, log_v, 1)
        dlnV_dlnR = p[0]
    else:
        # Simple two-point
        dlnV_dlnR = (log_v[-1] - log_v[0]) / (log_r[-1] - log_r[0])
    
    S = 1.0 - dlnV_dlnR
    
    return S


def compute_galaxy_predictors(galaxy, R_d_kpc):
    """
    Compute additional predictors for a galaxy.
    
    Returns:
        dict with shear, V_max, R_Vmax, compactness
    """
    r = galaxy.r_kpc
    v_obs = galaxy.v_obs
    
    # Baryonic velocity (for shear calculation)
    v_bar_sq = galaxy.v_gas**2 + galaxy.v_disk**2 + galaxy.v_bulge**2
    v_bar = np.sqrt(np.maximum(v_bar_sq, 1e-10))
    
    # V_max from observed curve
    valid = v_obs > 0
    if not valid.any():
        return {
            'V_max': np.nan,
            'R_Vmax': np.nan,
            'shear_2p2Rd': np.nan,
            'compactness': np.nan
        }
    
    V_max = np.max(v_obs[valid])
    R_Vmax = r[valid][np.argmax(v_obs[valid])]
    
    # Shear at 2.2 R_d
    if R_d_kpc and not np.isnan(R_d_kpc):
        R_target = 2.2 * R_d_kpc
        # Use baryonic curve for shear (smoother than observed)
        shear = compute_shear_at_radius(r[valid], v_bar[valid], R_target)
    else:
        shear = np.nan
    
    # Compactness: C = V_max^2 R_d / (G M_bary)
    # For simplicity, use M_bary ~ V_max^2 R_max / G as proxy
    # True compactness would need integrated mass
    # Use dimensionless proxy: C ~ (V_max / 100 km/s)^2 * (R_d / 3 kpc)
    if R_d_kpc and not np.isnan(R_d_kpc):
        compactness = (V_max / 100.0)**2 * (R_d_kpc / 3.0)
    else:
        compactness = np.nan
    
    return {
        'V_max': float(V_max),
        'R_Vmax': float(R_Vmax),
        'shear_2p2Rd': float(shear) if not np.isnan(shear) else None,
        'compactness': float(compactness) if not np.isnan(compactness) else None
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute shear and predictors')
    parser.add_argument('--sparc_dir', type=Path,
                       default=Path('data/Rotmod_LTG'))
    parser.add_argument('--master_file', type=Path,
                       default=Path('data/SPARC_Lelli2016c.mrt'))
    parser.add_argument('--disk_params', type=Path,
                       default=Path('many_path_model/bt_law/sparc_disk_params.json'))
    parser.add_argument('--output', type=Path,
                       default=Path('many_path_model/bt_law/sparc_shear_predictors.json'))
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPUTING SHEAR AND ADDITIONAL PREDICTORS")
    print("="*80)
    
    # Load disk params for R_d
    print(f"\nLoading disk parameters from: {args.disk_params}")
    with open(args.disk_params, 'r') as f:
        disk_params = json.load(f)
    print(f"  Loaded {len(disk_params)} galaxies")
    
    # Load SPARC data
    print(f"\nLoading SPARC rotation curves from: {args.sparc_dir}")
    master_info = load_sparc_master_table(args.master_file)
    rotmod_files = sorted(args.sparc_dir.glob('*_rotmod.dat'))
    print(f"  Found {len(rotmod_files)} galaxies")
    
    predictors = {}
    
    print("\nProcessing galaxies...")
    for i, rotmod_file in enumerate(rotmod_files, 1):
        galaxy_name = rotmod_file.stem.replace('_rotmod', '')
        
        try:
            galaxy = load_sparc_galaxy(rotmod_file, master_info)
            gal_disk = disk_params.get(galaxy_name, {})
            R_d = gal_disk.get('R_d_kpc')
            
            preds = compute_galaxy_predictors(galaxy, R_d)
            predictors[galaxy_name] = {
                'name': galaxy_name,
                **preds
            }
            
            if i % 25 == 0:
                print(f"  [{i:3d}/{len(rotmod_files)}] Processed {galaxy_name}")
                
        except Exception as e:
            print(f"  [{i:3d}/{len(rotmod_files)}] {galaxy_name} - FAILED: {e}")
            continue
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(predictors, f, indent=2)
    
    print(f"\n[OK] Saved predictors to: {args.output}")
    
    # Statistics
    valid_shear = [p['shear_2p2Rd'] for p in predictors.values() 
                   if p.get('shear_2p2Rd') is not None]
    valid_comp = [p['compactness'] for p in predictors.values()
                  if p.get('compactness') is not None]
    valid_vmax = [p['V_max'] for p in predictors.values()
                  if p.get('V_max') is not None and not np.isnan(p['V_max'])]
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"\nShear at 2.2 R_d (n={len(valid_shear)}):")
    if valid_shear:
        print(f"  Range: {np.min(valid_shear):.3f} - {np.max(valid_shear):.3f}")
        print(f"  Median: {np.median(valid_shear):.3f}")
        print(f"  Mean: {np.mean(valid_shear):.3f}")
    
    print(f"\nCompactness (n={len(valid_comp)}):")
    if valid_comp:
        print(f"  Range: {np.min(valid_comp):.3f} - {np.max(valid_comp):.3f}")
        print(f"  Median: {np.median(valid_comp):.3f}")
    
    print(f"\nV_max (n={len(valid_vmax)}):")
    if valid_vmax:
        print(f"  Range: {np.min(valid_vmax):.1f} - {np.max(valid_vmax):.1f} km/s")
        print(f"  Median: {np.median(valid_vmax):.1f} km/s")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
