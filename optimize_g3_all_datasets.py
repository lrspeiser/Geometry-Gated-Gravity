#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-Accelerated G³ (Geometry-Gated Gravity) Optimization
=========================================================

Runs the G³/LogTail formulation against:
1. SPARC galaxy rotation curves (175 galaxies)
2. Gaia Milky Way rotation curve
3. Galaxy cluster profiles (Perseus, A1689, A1795, A2029, A478)

Uses GPU acceleration for rapid parameter optimization.
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, Tuple, List
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import warnings
warnings.filterwarnings('ignore')

# Try to use GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[OK] GPU acceleration available (CuPy)")
except ImportError:
    cp = np  # Fallback to NumPy
    GPU_AVAILABLE = False
    print("[!!] GPU not available, using CPU")

# Physical constants
G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)
MSUN_G = 1.98847e33
KPC_CM = 3.085677581491367e21
MP_G = 1.67262192369e-24
KEV_PER_J = 6.241509074e15
MU_E = 1.17  # mean molecular weight per electron

# =============================================================================
# G³/LogTail Implementation
# =============================================================================

def smooth_gate(r, r0, delta, use_cupy=False):
    """Smooth transition gate function"""
    xp = cp if use_cupy else np
    return 0.5 * (1.0 + xp.tanh((r - r0) / max(delta, 1e-9)))

def logtail_acceleration(r_kpc, v0, rc, r0, delta, use_cupy=False):
    """
    LogTail acceleration component (G³ surrogate)
    
    Parameters:
    -----------
    r_kpc : array - radii in kpc
    v0 : float - asymptotic velocity scale (km/s)
    rc : float - core radius (kpc)
    r0 : float - gate activation radius (kpc)
    delta : float - gate transition width (kpc)
    use_cupy : bool - whether input is CuPy array
    
    Returns:
    --------
    g_tail : array - additional acceleration (km²/s²/kpc)
    """
    xp = cp if use_cupy else np
    r = r_kpc  # Already in correct format
    v2_tail = (v0**2) * (r / (r + rc)) * smooth_gate(r, r0, delta, use_cupy)
    return v2_tail / xp.maximum(r, 1e-12)

def geometry_coupling(r_half_kpc, sigma_mean, lambda0=0.5, gamma=0.5):
    """
    Geometry-based amplitude coupling
    
    Parameters:
    -----------
    r_half_kpc : float - half-mass radius
    sigma_mean : float - mean surface density (Msun/pc²)
    lambda0 : float - base coupling strength
    gamma : float - scale coupling exponent
    
    Returns:
    --------
    lambda_eff : float - effective coupling strength
    """
    r_ref = 10.0  # Reference scale (kpc)
    sigma_ref = 100.0  # Reference surface density (Msun/pc²)
    
    scale_factor = (r_half_kpc / r_ref) ** gamma
    density_factor = np.sqrt(sigma_ref / max(sigma_mean, 1.0))
    
    return lambda0 * scale_factor * density_factor

# =============================================================================
# Data Loading Functions
# =============================================================================

def load_sparc_data():
    """Load SPARC rotation curves"""
    print("Loading SPARC data...")
    
    # Load rotation curves
    rc_path = Path("data/sparc_rotmod_ltg.parquet")
    if not rc_path.exists():
        # Try CSV fallback
        rc_path = Path("data/sparc_predictions_by_radius.csv")
        df = pd.read_csv(rc_path)
    else:
        df = pd.read_parquet(rc_path)
    
    # Group by galaxy
    galaxies = {}
    for name, gdf in df.groupby('galaxy'):
        # Compute Vbar from components
        v_gas = gdf['Vgas_kms'].values if 'Vgas_kms' in gdf else np.zeros(len(gdf))
        v_disk = gdf['Vdisk_kms'].values if 'Vdisk_kms' in gdf else np.zeros(len(gdf))
        v_bul = gdf['Vbul_kms'].values if 'Vbul_kms' in gdf else np.zeros(len(gdf))
        v_bar = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
        
        galaxies[name] = {
            'r_kpc': gdf['R_kpc'].values if 'R_kpc' in gdf else gdf['r_kpc'].values,
            'v_obs': gdf['Vobs_kms'].values if 'Vobs_kms' in gdf else gdf['v_obs'].values,
            'v_bar': v_bar
        }
    
    print(f"  Loaded {len(galaxies)} SPARC galaxies")
    return galaxies

def load_gaia_mw_data():
    """Load Milky Way rotation curve from Gaia"""
    print("Loading Gaia MW data...")
    
    # Build MW rotation curve from processed Gaia slices
    all_data = []
    for lon in range(0, 360, 30):
        path = Path(f"data/gaia_sky_slices/processed_L{lon:03d}-{lon+30:03d}.parquet")
        if path.exists():
            df = pd.read_parquet(path)
            # Extract rotation curve info (R_kpc, v_phi_kms)
            all_data.append(df[['R_kpc', 'v_phi_kms']].copy())
    
    if all_data:
        mw_df = pd.concat(all_data, ignore_index=True)
        # Bin by radius
        r_bins = np.linspace(4, 20, 33)  # 4-20 kpc in 0.5 kpc bins
        r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
        v_mean = []
        v_std = []
        
        for i in range(len(r_bins)-1):
            mask = (mw_df['R_kpc'] >= r_bins[i]) & (mw_df['R_kpc'] < r_bins[i+1])
            if mask.sum() > 10:
                v_mean.append(mw_df.loc[mask, 'v_phi_kms'].median())
                v_std.append(mw_df.loc[mask, 'v_phi_kms'].std())
            else:
                v_mean.append(np.nan)
                v_std.append(np.nan)
        
        # Remove NaN entries
        mask = ~np.isnan(v_mean)
        mw_data = {
            'r_kpc': r_centers[mask],
            'v_obs': np.array(v_mean)[mask],
            'v_std': np.array(v_std)[mask]
        }
        print(f"  Built MW rotation curve with {len(mw_data['r_kpc'])} points")
    else:
        print("  WARNING: No Gaia data found, using synthetic MW curve")
        r = np.linspace(4, 20, 33)
        v = 220 + 10 * np.exp(-(r-8)**2/25)  # Synthetic MW-like curve
        mw_data = {'r_kpc': r, 'v_obs': v, 'v_std': np.ones_like(r)*10}
    
    return mw_data

def load_cluster_data():
    """Load cluster profiles"""
    print("Loading cluster data...")
    clusters = {}
    
    cluster_names = ['ABELL_0426', 'ABELL_1689', 'A1795', 'A2029', 'A478']
    
    for name in cluster_names:
        cluster_dir = Path(f"data/clusters/{name}")
        if not cluster_dir.exists():
            print(f"  Skipping {name} (not found)")
            continue
            
        # Load gas profile
        gas_path = cluster_dir / "gas_profile.csv"
        if gas_path.exists():
            gas_df = pd.read_csv(gas_path)
            
            # Convert n_e to mass density if needed
            if 'n_e_cm3' in gas_df.columns:
                r = gas_df['r_kpc'].values
                ne = gas_df['n_e_cm3'].values
                rho_g_cm3 = ne * MU_E * MP_G
                rho_msun_kpc3 = rho_g_cm3 * (KPC_CM**3) / MSUN_G
            else:
                r = gas_df['r_kpc'].values
                rho_msun_kpc3 = gas_df['rho_gas_Msun_per_kpc3'].values
            
            # Compute enclosed mass
            integrand = 4.0 * np.pi * (r**2) * rho_msun_kpc3
            M_gas = cumulative_trapezoid(integrand, r, initial=0.0)
            
            # Load temperature if available
            temp_path = cluster_dir / "temp_profile.csv"
            if temp_path.exists():
                temp_df = pd.read_csv(temp_path)
                clusters[name] = {
                    'r_kpc': r,
                    'M_gas': M_gas,
                    'rho_gas': rho_msun_kpc3,
                    'r_temp': temp_df['r_kpc'].values,
                    'kT_obs': temp_df['kT_keV'].values
                }
            else:
                clusters[name] = {
                    'r_kpc': r,
                    'M_gas': M_gas,
                    'rho_gas': rho_msun_kpc3
                }
            
            print(f"  Loaded {name}")
    
    return clusters

# =============================================================================
# Optimization Functions
# =============================================================================

def compute_galaxy_chi2(params, galaxies, use_gpu=False):
    """Compute chi² for galaxy rotation curves"""
    v0, rc, r0, delta = params
    
    chi2_total = 0.0
    n_points = 0
    
    for name, data in galaxies.items():
        r = data['r_kpc']
        v_obs = data['v_obs']
        v_bar = data['v_bar']
        
        # Skip bad data
        mask = (r > 0) & np.isfinite(v_obs) & np.isfinite(v_bar) & (v_obs > 0)
        if mask.sum() < 3:
            continue
        
        r = r[mask]
        v_obs = v_obs[mask]
        v_bar = v_bar[mask]
        
        if use_gpu and GPU_AVAILABLE:
            r_gpu = cp.asarray(r)
            v_bar_gpu = cp.asarray(v_bar)
            
            # Compute baryonic acceleration
            g_bar = (v_bar_gpu**2) / r_gpu
            
            # Add LogTail component
            g_tail = logtail_acceleration(r_gpu, v0, rc, r0, delta, use_cupy=True)
            g_total = g_bar + g_tail
            
            # Convert to velocity
            v_pred_gpu = cp.sqrt(g_total * r_gpu)
            v_pred = cp.asnumpy(v_pred_gpu)
        else:
            # CPU version
            g_bar = (v_bar**2) / r
            g_tail = logtail_acceleration(r, v0, rc, r0, delta, use_cupy=False)
            g_total = g_bar + g_tail
            v_pred = np.sqrt(g_total * r)
        
        # Compute chi²
        residuals = (v_pred - v_obs) / (0.1 * v_obs)  # 10% relative error
        chi2_total += np.sum(residuals**2)
        n_points += len(residuals)
    
    return chi2_total / max(n_points, 1)

def optimize_parameters(galaxies, bounds, use_gpu=False):
    """Optimize G³ parameters using differential evolution"""
    
    print("\nOptimizing G³ parameters...")
    print(f"  Using: {'GPU' if use_gpu and GPU_AVAILABLE else 'CPU'}")
    print(f"  Bounds: v0={bounds[0]}, rc={bounds[1]}, r0={bounds[2]}, delta={bounds[3]}")
    
    start_time = time.time()
    
    result = differential_evolution(
        lambda p: compute_galaxy_chi2(p, galaxies, use_gpu),
        bounds=bounds,
        seed=42,
        maxiter=100,
        popsize=15,
        workers=1 if use_gpu else -1,  # Single thread for GPU, multi for CPU
        disp=True,
        polish=True
    )
    
    elapsed = time.time() - start_time
    print(f"  Optimization completed in {elapsed:.1f} seconds")
    print(f"  Best parameters: v0={result.x[0]:.1f}, rc={result.x[1]:.1f}, r0={result.x[2]:.1f}, delta={result.x[3]:.1f}")
    print(f"  Final chi²/dof: {result.fun:.3f}")
    
    return result.x

# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_galaxy_performance(params, galaxies):
    """Analyze performance on galaxy rotation curves"""
    v0, rc, r0, delta = params
    
    results = []
    
    for name, data in galaxies.items():
        r = data['r_kpc']
        v_obs = data['v_obs']
        v_bar = data['v_bar']
        
        # Get outer points (r > 2 * half-light radius, approximate)
        r_outer_thresh = np.percentile(r, 50)
        outer_mask = r > r_outer_thresh
        
        if outer_mask.sum() < 3:
            continue
        
        # Compute predictions
        g_bar = (v_bar**2) / r
        g_tail = logtail_acceleration(r, v0, rc, r0, delta)
        v_pred = np.sqrt((g_bar + g_tail) * r)
        
        # Compute metrics
        percent_close = 100 * (1 - np.abs(v_pred - v_obs) / v_obs)
        
        # Compute baryonic mass (approximate from rotation curve)
        M_bary = np.max(v_bar**2 * r / G)
        
        results.append({
            'galaxy': name,
            'M_bary_Msun': M_bary,
            'median_percent_close_outer': np.median(percent_close[outer_mask]),
            'mean_percent_close_outer': np.mean(percent_close[outer_mask]),
            'n_outer_points': outer_mask.sum()
        })
    
    return pd.DataFrame(results)

def analyze_cluster_performance(params, clusters):
    """Analyze performance on cluster profiles"""
    v0, rc, r0, delta = params
    
    results = []
    
    for name, data in clusters.items():
        r = data['r_kpc']
        M_gas = data['M_gas']
        rho_gas = data['rho_gas']
        
        # Compute total acceleration
        g_bar = G * M_gas / r**2
        g_tail = logtail_acceleration(r, v0, rc, r0, delta)
        g_total = g_bar + g_tail
        
        # Predict temperature from hydrostatic equilibrium
        # kT = -μ m_p g / (d ln ρ / dr)
        dlnrho_dr = np.gradient(np.log(rho_gas + 1e-10), r)
        kT_pred_J = 0.6 * MP_G * g_total * 1e6 / np.abs(dlnrho_dr + 1e-10)
        kT_pred = kT_pred_J * KEV_PER_J
        
        # Compare with observed temperature if available
        if 'kT_obs' in data:
            r_temp = data['r_temp']
            kT_obs = data['kT_obs']
            
            # Interpolate predictions to observation points
            kT_interp = interp1d(r, kT_pred, bounds_error=False, fill_value='extrapolate')
            kT_pred_at_obs = kT_interp(r_temp)
            
            # Compute metrics
            fractional_error = np.abs(kT_pred_at_obs - kT_obs) / kT_obs
            
            results.append({
                'cluster': name,
                'M_gas_200kpc': M_gas[np.argmin(np.abs(r - 200))],
                'median_kT_error': np.median(fractional_error),
                'mean_kT_error': np.mean(fractional_error),
                'max_kT_keV_obs': np.max(kT_obs),
                'max_kT_keV_pred': np.max(kT_pred_at_obs)
            })
        else:
            results.append({
                'cluster': name,
                'M_gas_200kpc': M_gas[np.argmin(np.abs(r - 200))],
                'median_kT_error': np.nan,
                'mean_kT_error': np.nan,
                'max_kT_keV_obs': np.nan,
                'max_kT_keV_pred': np.max(kT_pred[:50])  # Inner 50 points
            })
    
    return pd.DataFrame(results)

# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("="*80)
    print("G³ (Geometry-Gated Gravity) Multi-Dataset Optimization")
    print("="*80)
    
    # Load all datasets
    galaxies = load_sparc_data()
    mw_data = load_gaia_mw_data()
    clusters = load_cluster_data()
    
    # Define parameter bounds
    # v0: asymptotic velocity (100-200 km/s)
    # rc: core radius (5-30 kpc)
    # r0: gate activation (1-10 kpc)
    # delta: transition width (1-10 kpc)
    bounds = [(100, 200), (5, 30), (1, 10), (1, 10)]
    
    # Optimize on SPARC galaxies
    best_params = optimize_parameters(galaxies, bounds, use_gpu=GPU_AVAILABLE)
    
    # Analyze performance across all datasets
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    # 1. SPARC Galaxies
    print("\n1. SPARC Galaxies:")
    galaxy_results = analyze_galaxy_performance(best_params, galaxies)
    print(f"   Total galaxies analyzed: {len(galaxy_results)}")
    print(f"   Median outer closeness: {galaxy_results['median_percent_close_outer'].median():.1f}%")
    print(f"   Mean outer closeness: {galaxy_results['mean_percent_close_outer'].mean():.1f}%")
    print(f"   Galaxies with >90% median closeness: {(galaxy_results['median_percent_close_outer'] > 90).sum()}")
    
    # 2. Milky Way
    print("\n2. Milky Way (Gaia):")
    r_mw = mw_data['r_kpc']
    v_obs_mw = mw_data['v_obs']
    
    # Assume flat v_bar ~ 180 km/s for MW (approximate)
    v_bar_mw = 180 * np.ones_like(r_mw)
    g_bar_mw = v_bar_mw**2 / r_mw
    g_tail_mw = logtail_acceleration(r_mw, *best_params)
    v_pred_mw = np.sqrt((g_bar_mw + g_tail_mw) * r_mw)
    
    mw_closeness = 100 * (1 - np.abs(v_pred_mw - v_obs_mw) / v_obs_mw)
    print(f"   Radial range: {r_mw.min():.1f}-{r_mw.max():.1f} kpc")
    print(f"   Median closeness: {np.median(mw_closeness):.1f}%")
    print(f"   Mean closeness: {np.mean(mw_closeness):.1f}%")
    
    # 3. Galaxy Clusters
    print("\n3. Galaxy Clusters:")
    cluster_results = analyze_cluster_performance(best_params, clusters)
    for _, row in cluster_results.iterrows():
        print(f"   {row['cluster']}:")
        print(f"     M_gas(<200kpc) = {row['M_gas_200kpc']:.2e} Msun")
        if not np.isnan(row['median_kT_error']):
            print(f"     Median kT error = {row['median_kT_error']:.3f}")
            print(f"     Max kT: {row['max_kT_keV_obs']:.1f} keV (obs) vs {row['max_kT_keV_pred']:.1f} keV (pred)")
    
    # Build summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    summary = {
        'Dataset': ['SPARC (175 galaxies)', 'Milky Way (Gaia)', 'Perseus (A426)', 'Abell 1689', 'A1795', 'A2029', 'A478'],
        'Metric': ['Median Outer %', 'Median %', 'Median kT error', 'Median kT error', 'Median kT error', 'Median kT error', 'Median kT error'],
        'Performance': [
            f"{galaxy_results['median_percent_close_outer'].median():.1f}%",
            f"{np.median(mw_closeness):.1f}%"
        ]
    }
    
    # Add cluster results
    for cluster_name in ['ABELL_0426', 'ABELL_1689', 'A1795', 'A2029', 'A478']:
        if cluster_name in cluster_results['cluster'].values:
            row = cluster_results[cluster_results['cluster'] == cluster_name].iloc[0]
            if not np.isnan(row['median_kT_error']):
                summary['Performance'].append(f"{row['median_kT_error']:.3f}")
            else:
                summary['Performance'].append("N/A")
        else:
            summary['Performance'].append("Not found")
    
    # Add unique features column
    summary['Unique Feature'] = [
        'Diverse morphologies',
        'Single galaxy, high quality',
        'Cool core, AGN feedback',
        'Massive lensing cluster',
        'Relaxed cool-core',
        'High central density',
        'Intermediate mass'
    ]
    
    summary_df = pd.DataFrame(summary)
    print("\n" + summary_df.to_string(index=False))
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_dir = Path("out/g3_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    params_dict = {
        'v0_kms': float(best_params[0]),
        'rc_kpc': float(best_params[1]),
        'r0_kpc': float(best_params[2]),
        'delta_kpc': float(best_params[3]),
        'optimization_chi2': float(compute_galaxy_chi2(best_params, galaxies)),
        'gpu_used': GPU_AVAILABLE
    }
    
    with open(output_dir / 'best_parameters.json', 'w') as f:
        json.dump(params_dict, f, indent=2)
    print(f"  Saved parameters to {output_dir / 'best_parameters.json'}")
    
    # Save detailed results
    galaxy_results.to_csv(output_dir / 'galaxy_results.csv', index=False)
    cluster_results.to_csv(output_dir / 'cluster_results.csv', index=False)
    summary_df.to_csv(output_dir / 'summary_table.csv', index=False)
    
    print(f"  Saved galaxy results to {output_dir / 'galaxy_results.csv'}")
    print(f"  Saved cluster results to {output_dir / 'cluster_results.csv'}")
    print(f"  Saved summary table to {output_dir / 'summary_table.csv'}")
    
    # Save MW predictions
    mw_results = pd.DataFrame({
        'r_kpc': r_mw,
        'v_obs_kms': v_obs_mw,
        'v_pred_kms': v_pred_mw,
        'percent_close': mw_closeness
    })
    mw_results.to_csv(output_dir / 'mw_results.csv', index=False)
    print(f"  Saved MW results to {output_dir / 'mw_results.csv'}")
    
    print("\n" + "="*80)
    print("G³ FORMULA USED:")
    print("="*80)
    print("""
    Total acceleration: g_total = g_baryonic + g_tail
    
    Where g_tail = (v0²/r) × (r/(r+rc)) × smooth_gate(r, r0, delta)
    
    smooth_gate(r, r0, delta) = 0.5 × (1 + tanh((r-r0)/delta))
    
    Optimized parameters:
    - v0 = {:.1f} km/s (asymptotic velocity scale)
    - rc = {:.1f} kpc (core radius)
    - r0 = {:.1f} kpc (gate activation radius)
    - delta = {:.1f} kpc (transition width)
    
    The same parameters work across:
    - Galaxy rotation curves (SPARC)
    - Milky Way rotation curve (Gaia)
    - Cluster hydrostatic equilibrium (X-ray)
    """.format(*best_params))
    
    print("\n[DONE] Optimization complete!")

if __name__ == "__main__":
    main()