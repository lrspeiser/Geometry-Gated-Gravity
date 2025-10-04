"""
GPU-Accelerated Cluster Fitting for Squared Gravity Model

This script uses the GeometricExponentGravity model with CuPy acceleration
to efficiently search parameter space for fits to CLASH Einstein radii.

The grid search explores:
- gamma1, gamma2: β(Σ,R) = 1 + γ₁|Σ̂| + γ₂log₁₀(R/R_scale)
- a, b, d: fX = (r/Rd)²/(a - bΣ̂ - d|∇ln Σ|)

Results are saved to data/clash/processed/squared_gravity/gpu_fits/
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from itertools import product
import os
import sys
import argparse

# Ensure local imports work when running from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from cupy_utils import get_array_module, to_array_module, trapz_cumsum
from geometric_exponent import GeometricExponentGravity


def load_clash_profiles(data_dir="data/clash/processed"):
    """Load CLASH cluster baryon profiles and Einstein radii."""
    profiles = {}
    data_path = Path(data_dir)
    
    for cluster_dir in data_path.glob("*"):
        if not cluster_dir.is_dir():
            continue
            
        cluster_name = cluster_dir.name
        profile_file = cluster_dir / "baryon_profile.csv"
        
        if profile_file.exists():
            df = pd.read_csv(profile_file)
            if all(col in df.columns for col in ['R_kpc', 'Sigma_bar_kpc2', 'Einstein_radius_kpc']):
                profiles[cluster_name] = {
                    'R_kpc': df['R_kpc'].values,
                    'Sigma_bar_kpc2': df['Sigma_bar_kpc2'].values,
                    'Rd_kpc': df.get('Rd_kpc', [50.0] * len(df)).iloc[0],  # Default disk scale
                    'Einstein_radius_obs': df['Einstein_radius_kpc'].iloc[0]
                }
    
    return profiles


def compute_einstein_radius_gpu(xp, R_kpc, Sigma_eff_kpc2, z_source=2.0):
    """Compute Einstein radius from effective surface density using convergence integration.

    Notes:
    - Uses a fixed Sigma_crit placeholder; we can wire in proper distance-based Sigma_crit once lens/source z are available.
    - Handles both NumPy and CuPy arrays and returns a Python float.
    """
    # Ensure backend arrays
    R_kpc = xp.asarray(R_kpc)
    Sigma_eff_kpc2 = xp.asarray(Sigma_eff_kpc2)

    # Critical surface density (approximate for z_source=2.0)
    Sigma_crit = 3.0e3  # M_sun/kpc^2 (placeholder)
    
    # Convergence κ = Σ_eff / Σ_crit
    kappa = Sigma_eff_kpc2 / Sigma_crit
    
    # Average convergence within radius R: κ̄(R) = ∫₀ᴿ κ(r) 2πr dr / (πR²)
    # Using cumulative integration: numerator integrates κ(r) 2r dr; denominator is R^2
    kappa_avg = trapz_cumsum(xp, kappa * R_kpc * 2.0, R_kpc) / (R_kpc**2)
    
    # Helper to convert backend scalar to Python float
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            try:
                return float(xp.asnumpy(v))
            except Exception:
                return float(np.asarray(v))

    # Einstein radius where κ̄ >= 1
    try:
        idx = xp.where(kappa_avg >= 1.0)[0]
        if getattr(idx, 'size', None) is None:
            # Ensure we can get size for both numpy/cupy
            idx_size = len(idx)
        else:
            idx_size = int(idx.size)

        if idx_size > 0:
            return _to_float(R_kpc[idx[0]])
        else:
            return _to_float(R_kpc[-1])  # No Einstein radius found within sampled range
    except Exception:
        return _to_float(R_kpc[-1])


def fit_cluster_gpu(xp, cluster_data, params):
    """Fit single cluster with given parameters using GPU backend."""
    gamma1, gamma2, a, b, d = params
    
    # Initialize model
    model = GeometricExponentGravity(
        gamma1=gamma1, gamma2=gamma2,
        a=a, b=b, d=d,
        R_scale_kpc=100.0,  # kpc
        beta_clip=(1.0, 5.0)
    )
    
    # Get profile data
    R_kpc = cluster_data['R_kpc']
    Sigma_bar = cluster_data['Sigma_bar_kpc2']
    Rd_kpc = cluster_data['Rd_kpc']
    R_E_obs = cluster_data['Einstein_radius_obs']
    
    # Compute effective surface density with GPU
    Sigma_eff, fX, beta = model.Sigma_effective_xp(xp, R_kpc, Sigma_bar, Rd_kpc)
    
    # Compute predicted Einstein radius
    R_E_pred = compute_einstein_radius_gpu(xp, 
                                         xp.asarray(R_kpc), 
                                         Sigma_eff)
    
    # Chi-squared metric
    chi2 = ((R_E_pred - R_E_obs) / (0.1 * R_E_obs))**2
    
    return float(chi2), float(R_E_pred)


def run_gpu_grid_search(profiles, param_grid, output_dir):
    """Run parameter grid search across all clusters using GPU acceleration."""
    
    xp = get_array_module()
    print(f"Using array backend: {'CuPy (GPU)' if xp.__name__ == 'cupy' else 'NumPy (CPU)'}")
    
    results = []
    total_combinations = len(list(product(*param_grid.values())))
    
    print(f"Running grid search with {total_combinations} parameter combinations")
    print(f"Fitting {len(profiles)} clusters")
    
    start_time = time.time()
    
    for i, params in enumerate(product(*param_grid.values())):
        param_dict = dict(zip(param_grid.keys(), params))
        
        total_chi2 = 0.0
        cluster_results = {}
        
        for cluster_name, cluster_data in profiles.items():
            try:
                chi2, R_E_pred = fit_cluster_gpu(xp, cluster_data, params)
                total_chi2 += chi2
                cluster_results[cluster_name] = {
                    'chi2': chi2,
                    'R_E_pred': R_E_pred,
                    'R_E_obs': cluster_data['Einstein_radius_obs']
                }
            except Exception as e:
                print(f"Error fitting {cluster_name}: {e}")
                cluster_results[cluster_name] = {
                    'chi2': 1e6,
                    'R_E_pred': -1,
                    'R_E_obs': cluster_data['Einstein_radius_obs']
                }
                total_chi2 += 1e6
        
        # Store results
        result = {
            'params': param_dict,
            'total_chi2': total_chi2,
            'clusters': cluster_results
        }
        results.append(result)
        
        # Progress report
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total_combinations - i - 1) / rate
            print(f"Progress: {i+1}/{total_combinations} ({100*(i+1)/total_combinations:.1f}%) "
                  f"Rate: {rate:.1f}/s ETA: {eta:.0f}s")
    
    # Sort by total chi2
    results.sort(key=lambda x: x['total_chi2'])
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    with open(output_path / 'gpu_grid_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save best fits summary
    best_fits = []
    for i, result in enumerate(results[:20]):  # Top 20
        fit_summary = {
            'rank': i + 1,
            'total_chi2': result['total_chi2'],
            **result['params']
        }
        # Add individual cluster chi2s
        for cluster_name, cluster_result in result['clusters'].items():
            fit_summary[f'{cluster_name}_chi2'] = cluster_result['chi2']
            fit_summary[f'{cluster_name}_R_E_ratio'] = cluster_result['R_E_pred'] / cluster_result['R_E_obs']
        best_fits.append(fit_summary)
    
    df_best = pd.DataFrame(best_fits)
    df_best.to_csv(output_path / 'best_fits_summary.csv', index=False)
    
    elapsed_total = time.time() - start_time
    print(f"\nGrid search completed in {elapsed_total:.1f}s")
    print(f"Best fit chi2: {results[0]['total_chi2']:.2f}")
    print(f"Best parameters: {results[0]['params']}")
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="GPU-accelerated grid search for clusters")
    parser.add_argument("--data-dir", default="data/clash/processed", help="Path to processed CLASH data")
    parser.add_argument("--output-dir", default="data/clash/processed/squared_gravity/gpu_fits", help="Output directory")
    # Parameter ranges: min max count
    parser.add_argument("--gamma1", nargs=3, type=float, default=[0.2, 1.0, 3], help="gamma1 min max count")
    parser.add_argument("--gamma2", nargs=3, type=float, default=[0.1, 0.5, 3], help="gamma2 min max count")
    parser.add_argument("--a", nargs=3, type=float, default=[1.0, 3.0, 3], help="a min max count")
    parser.add_argument("--b", nargs=3, type=float, default=[0.2, 0.8, 3], help="b min max count")
    parser.add_argument("--d", nargs=3, type=float, default=[0.2, 0.8, 3], help="d min max count")
    args = parser.parse_args()

    # Load CLASH data
    print("Loading CLASH cluster profiles...")
    profiles = load_clash_profiles(args.data_dir)
    print(f"Loaded {len(profiles)} clusters: {list(profiles.keys())}")
    
    if not profiles:
        print("No cluster profiles found! Check data directory structure.")
        return

    # Build parameter grid from args
    g1_min, g1_max, g1_n = args.gamma1
    g2_min, g2_max, g2_n = args.gamma2
    a_min, a_max, a_n = args.a
    b_min, b_max, b_n = args.b
    d_min, d_max, d_n = args.d

    param_grid = {
        'gamma1': np.linspace(g1_min, g1_max, int(g1_n)),
        'gamma2': np.linspace(g2_min, g2_max, int(g2_n)),
        'a': np.linspace(a_min, a_max, int(a_n)),
        'b': np.linspace(b_min, b_max, int(b_n)),
        'd': np.linspace(d_min, d_max, int(d_n))
    }
    
    print("Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {len(values)} values from {values[0]:.2f} to {values[-1]:.2f}")
    
    # Run grid search
    results = run_gpu_grid_search(profiles, param_grid, args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")
    print("Top 3 parameter combinations:")
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. chi2={result['total_chi2']:.2f}, params={result['params']}")


if __name__ == "__main__":
    main()