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

from cupy_utils import xp as get_xp, to_xp, xp_cumtrapz
from geometric_exponent import GeometricExponentGravity

# Use exact Sigma_crit from the real-Σ pipeline (no placeholders)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    sigma_crit_Msun_per_kpc2,
)


def load_clash_profiles(data_dir="data/clash/processed", include=None, exclude=None):
    """Load CLASH cluster baryon profiles and Einstein radii.

    Also wires in lens redshifts from data/clash/einstein_radii_observed.csv to support
    proper Sigma_crit(z_l, z_s) when computing θ_E.
    """
    profiles = {}
    data_path = Path(data_dir)

    # Lens redshifts
    obs_path = PROJECT_ROOT / 'data' / 'clash' / 'einstein_radii_observed.csv'
    z_map = {}
    if obs_path.exists():
        try:
            odf = pd.read_csv(obs_path)
            for _, row in odf.iterrows():
                z_map[str(row['cluster_id']).lower()] = float(row['z_lens'])
        except Exception:
            z_map = {}

    include_set = set([s.strip().lower() for s in include.split(',')]) if include else None
    exclude_set = set([s.strip().lower() for s in exclude.split(',')]) if exclude else set()

    for cluster_dir in data_path.glob("*"):
        if not cluster_dir.is_dir():
            continue

        cluster_name = cluster_dir.name.lower()
        if include_set is not None and cluster_name not in include_set:
            continue
        if cluster_name in exclude_set:
            continue

        profile_file = cluster_dir / "baryon_profile.csv"
        if profile_file.exists():
            df = pd.read_csv(profile_file)
            if all(col in df.columns for col in ['R_kpc', 'Sigma_bar_kpc2', 'Einstein_radius_kpc']):
                profiles[cluster_name] = {
                    'R_kpc': df['R_kpc'].values,
                    'Sigma_bar_kpc2': df['Sigma_bar_kpc2'].values,
                    'Rd_kpc': df.get('Rd_kpc', [50.0] * len(df)).iloc[0],  # Default disk scale
                    'Einstein_radius_obs': float(df['Einstein_radius_kpc'].iloc[0]),
                    'z_lens': float(z_map.get(cluster_name, np.nan)),
                }

    return profiles


def compute_einstein_radius_gpu(xp, R_kpc, Sigma_eff_kpc2, z_lens, z_source=2.0):
    """Compute Einstein radius from effective surface density using convergence integration.

    Notes:
    - Uses exact Sigma_crit(z_l, z_s) from cosmology (no placeholders).
    - Handles both NumPy and CuPy arrays and returns a Python float in kpc.
    """
    # Ensure backend arrays
    R_kpc = xp.asarray(R_kpc)
    Sigma_eff_kpc2 = xp.asarray(Sigma_eff_kpc2)

    # Critical surface density (Msun/kpc^2)
    Sigma_crit = sigma_crit_Msun_per_kpc2(float(z_lens), float(z_source))

    # Convergence κ = Σ_eff / Σ_crit
    kappa = Sigma_eff_kpc2 / Sigma_crit

    # Average convergence within radius R: κ̄(R) = ∫₀ᴿ κ(r) 2πr dr / (πR²)
    # Using cumulative integration: numerator integrates κ(r) 2r dr; denominator is R^2
    kappa_avg = xp_cumtrapz(kappa * R_kpc * 2.0, R_kpc) / (R_kpc**2)

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
        idx_size = int(getattr(idx, 'size', len(idx)))
        if idx_size > 0:
            return _to_float(R_kpc[idx[0]])
        else:
            return _to_float(R_kpc[-1])  # No Einstein radius found within sampled range
    except Exception:
        return _to_float(R_kpc[-1])


def fit_cluster_gpu(xp, cluster_data, params, z_source):
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
    z_lens = cluster_data.get('z_lens', np.nan)

    # Compute effective surface density with GPU
    Sigma_eff, fX, beta = model.Sigma_effective_xp(xp, R_kpc, Sigma_bar, Rd_kpc)

    # Compute predicted Einstein radius (kpc)
    R_E_pred = compute_einstein_radius_gpu(
        xp,
        xp.asarray(R_kpc),
        Sigma_eff,
        z_lens=z_lens,
        z_source=z_source,
    )

    # Chi-squared metric (10% of observed R_E as scale)
    chi2 = ((R_E_pred - R_E_obs) / (0.1 * max(R_E_obs, 1e-6)))**2

    return float(chi2), float(R_E_pred)


def run_gpu_grid_search(profiles, param_grid, output_dir, z_source):
    """Run parameter grid search across all clusters using GPU acceleration."""

    xp = get_xp()
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
                chi2, R_E_pred = fit_cluster_gpu(xp, cluster_data, params, z_source=z_source)
                total_chi2 += chi2
                cluster_results[cluster_name] = {
                    'chi2': chi2,
                    'R_E_pred_kpc': R_E_pred,
                    'R_E_obs_kpc': cluster_data['Einstein_radius_obs']
                }
            except Exception as e:
                print(f"Error fitting {cluster_name}: {e}")
                cluster_results[cluster_name] = {
                    'chi2': 1e6,
                    'R_E_pred_kpc': -1.0,
                    'R_E_obs_kpc': cluster_data['Einstein_radius_obs']
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
            print(
                f"Progress: {i+1}/{total_combinations} ({100*(i+1)/total_combinations:.1f}%) "
                f"Rate: {rate:.1f}/s ETA: {eta:.0f}s"
            )

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
            ratio = (
                cluster_result['R_E_pred_kpc'] / cluster_result['R_E_obs_kpc']
                if cluster_result['R_E_obs_kpc'] not in (None, 0) else np.nan
            )
            fit_summary[f'{cluster_name}_R_E_ratio'] = ratio
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
    parser.add_argument("--zs", type=float, default=2.0, help="Source redshift to use for Sigma_crit(zl,zs)")
    parser.add_argument("--include", type=str, default=None, help="Comma-separated cluster_ids to include (else all)")
    parser.add_argument("--exclude", type=str, default=None, help="Comma-separated cluster_ids to exclude")
    # Parameter ranges: min max count (expanded ranges by default)
    parser.add_argument("--gamma1", nargs=3, type=float, default=[0.1, 1.2, 5], help="gamma1 min max count")
    parser.add_argument("--gamma2", nargs=3, type=float, default=[0.0, 0.8, 5], help="gamma2 min max count")
    parser.add_argument("--a", nargs=3, type=float, default=[-1.0, 3.0, 7], help="a min max count (allows negatives)")
    parser.add_argument("--b", nargs=3, type=float, default=[0.0, 1.5, 7], help="b min max count (broader)")
    parser.add_argument("--d", nargs=3, type=float, default=[0.0, 2.0, 7], help="d min max count (broader)")
    args = parser.parse_args()

    # Load CLASH data
    print("Loading CLASH cluster profiles...")
    profiles = load_clash_profiles(args.data_dir, include=args.include, exclude=args.exclude)
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
    results = run_gpu_grid_search(profiles, param_grid, args.output_dir, z_source=args.zs)

    print(f"\nResults saved to {args.output_dir}")
    print("Top 3 parameter combinations:")
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. chi2={result['total_chi2']:.2f}, params={result['params']}")


if __name__ == "__main__":
    main()
