#!/usr/bin/env python3
"""
run_holdout_validation.py

Blind validation of mass-scaled hierarchical model on hold-out clusters.

Pre-registered pass criteria (publishable):
    - Median fractional error |Δθ_E|/θ_E,obs < 20%
    - Both hold-outs inside 68% posterior predictive intervals
    - No systematic sign bias (both low or both high)

Usage:
    python scripts/run_holdout_validation.py \\
        --posterior output/mass_scaled/trace.netcdf \\
        --holdout A1689,MACS1149 \\
        --use-mass-scaling 1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Local utilities
from scripts.cluster_overrides import load_cluster_override
from scripts import lensing_utils


def load_holdout_clusters(
    holdout_names: List[str],
    catalog_path: Optional[Path] = None
) -> pd.DataFrame:
    """Load hold-out cluster data"""
    if catalog_path is None:
        catalog_path = REPO_ROOT / "data" / "cluster_lensing_catalog.csv"
    
    df = pd.read_csv(catalog_path)
    
    # Normalize names
    holdout_upper = [h.upper().replace(' ', '').replace('-', '') for h in holdout_names]
    df['name_normalized'] = df['cluster_name'].str.upper().str.replace(' ', '').str.replace('-', '')
    
    holdouts = df[df['name_normalized'].isin(holdout_upper)].copy()
    
    if len(holdouts) == 0:
        raise ValueError(f"No hold-out clusters found: {holdout_names}")
    
    print(f"Loaded {len(holdouts)} hold-out clusters:")
    for _, row in holdouts.iterrows():
        print(f"  {row['cluster_name']}: θ_E = {row['theta_E_obs']:.2f} ± {row['sigma_theta_E']:.2f} arcsec")
    
    return holdouts


def predict_theta_E_from_posterior(
    cluster_row: pd.Series,
    trace: az.InferenceData,
    use_mass_scaling: bool = True
) -> Dict:
    """
    Generate posterior predictive distribution for θ_E.
    
    Returns dict with:
        - theta_E_pred_median
        - theta_E_pred_mean
        - theta_E_pred_16
        - theta_E_pred_84
        - samples (array)
    """
    # Extract posterior samples
    posterior = trace.posterior
    
    mu_A = posterior['mu_A'].values.flatten()
    sigma_A = posterior['sigma_A'].values.flatten()
    # Support both naming conventions
    ell_0_star = (posterior['ell_0_star_kpc'].values.flatten()
                  if 'ell_0_star_kpc' in posterior
                  else posterior['ell0_star_kpc'].values.flatten())
    sigma_int = posterior['sigma_int'].values.flatten() if 'sigma_int' in posterior else np.zeros_like(mu_A)
    
    if use_mass_scaling:
        gamma = posterior['gamma'].values.flatten()
    else:
        gamma = np.zeros_like(mu_A)
    
    # Geometry (use population means if hierarchical), allow override
    override = load_cluster_override(cluster_row['cluster_name'])
    if 'mu_q' in posterior:
        mu_q = posterior['mu_q'].values.flatten()
        sigma_q = posterior['sigma_q'].values.flatten()
    else:
        mu_q = np.ones_like(mu_A)
        sigma_q = np.zeros_like(mu_A)
    if override and 'geometry' in override:
        mu_q_override = float(override['geometry'].get('mu_q', mu_q.mean()))
        sigma_q_override = float(override['geometry'].get('sigma_q', sigma_q.mean() if sigma_q.size else 0.1))
        mu_q = np.full_like(mu_q, mu_q_override)
        sigma_q = np.full_like(sigma_q, sigma_q_override)
    
    n_samples = len(mu_A)
    theta_E_samples = np.zeros(n_samples)
    
    R500_Mpc = cluster_row['R500_Mpc']
    
    for i in range(n_samples):
        # Sample per-cluster parameters from population
        A_c_i = np.random.normal(mu_A[i], sigma_A[i])
        ell_0_i = ell_0_star[i] * (R500_Mpc / 1.0)**gamma[i]
        
        # Sample geometry
        q_LOS_i = np.clip(np.random.normal(mu_q[i], sigma_q[i]), 0.7, 1.4)
        q_plane_i = np.clip(np.random.normal(mu_q[i], sigma_q[i]), 0.7, 1.4)
        # Cluster-specific κ_ext prior width if provided
        override = load_cluster_override(cluster_row['cluster_name'])
        kappa_sigma = float(override.get('kappa_ext_sigma')) if (override and 'kappa_ext_sigma' in override) else 0.03
        kappa_ext_i = np.random.normal(0.0, kappa_sigma)
        
        # Compute θ_E (use same functions as inference script)
        # Import after path setup
        from run_mass_scaled_hierarchical_inference import (
            compute_theta_E_triaxial,
            compute_baryon_surface_density
        )
        
        R_kpc = np.linspace(1, 2000, 200)
        # reuse override from above
        Sigma_bar = compute_baryon_surface_density(cluster_row, R_kpc, override=override)
        
        D_lens, D_source, D_LS = lensing_utils.effective_distances(
            z_lens=cluster_row['z_lens'], z_source=cluster_row.get('z_source', None), override=override
        )
        
        theta_model = compute_theta_E_triaxial(
            Sigma_bar, R_kpc, A_c_i, ell_0_i,
            q_LOS_i, q_plane_i, kappa_ext_i,
            D_lens, D_source, D_LS
        )
        # Add intrinsic scatter draw if present
        theta_E_samples[i] = float(theta_model + np.random.normal(0.0, sigma_int[i]))
    
    return {
        'theta_E_pred_median': float(np.median(theta_E_samples)),
        'theta_E_pred_mean': float(np.mean(theta_E_samples)),
        'theta_E_pred_16': float(np.percentile(theta_E_samples, 16)),
        'theta_E_pred_84': float(np.percentile(theta_E_samples, 84)),
        'samples': theta_E_samples
    }


def validate_holdouts(
    holdouts: pd.DataFrame,
    trace: az.InferenceData,
    use_mass_scaling: bool,
    output_dir: Path
) -> Dict:
    """
    Run validation on hold-out clusters.
    
    Returns summary dict with pass/fail status.
    """
    print("="*60)
    print("HOLD-OUT VALIDATION")
    print("="*60)
    
    results = []
    
    for idx, row in holdouts.iterrows():
        print(f"\nPredicting {row['cluster_name']}...")
        
        pred = predict_theta_E_from_posterior(row, trace, use_mass_scaling)
        
        theta_obs = row['theta_E_obs']
        sigma_obs = row['sigma_theta_E']
        
        # Compute metrics
        frac_error = abs(pred['theta_E_pred_median'] - theta_obs) / theta_obs
        inside_68 = (pred['theta_E_pred_16'] <= theta_obs <= pred['theta_E_pred_84'])
        residual = pred['theta_E_pred_median'] - theta_obs
        
        results.append({
            'cluster': str(row['cluster_name']),
            'theta_E_obs': float(theta_obs),
            'sigma_obs': float(sigma_obs),
            'theta_E_pred_median': float(pred['theta_E_pred_median']),
            'theta_E_pred_16': float(pred['theta_E_pred_16']),
            'theta_E_pred_84': float(pred['theta_E_pred_84']),
            'frac_error': float(frac_error),
            'inside_68': bool(inside_68),
            'residual': float(residual)
        })
        
        print(f"  Observed: {theta_obs:.2f} ± {sigma_obs:.2f} arcsec")
        print(f"  Predicted: {pred['theta_E_pred_median']:.2f} [{pred['theta_E_pred_16']:.2f}, {pred['theta_E_pred_84']:.2f}]")
        print(f"  Frac error: {frac_error:.1%}")
        print(f"  Inside 68% CI: {inside_68}")
    
    # Aggregate metrics
    frac_errors = [r['frac_error'] for r in results]
    inside_68_count = sum(r['inside_68'] for r in results)
    residuals = [r['residual'] for r in results]
    
    median_frac_error = np.median(frac_errors)
    all_positive = all(r > 0 for r in residuals)
    all_negative = all(r < 0 for r in residuals)
    
    # Pass criteria
    pass_error = bool(median_frac_error < 0.20)
    pass_coverage = bool(inside_68_count >= len(holdouts) * 0.68)
    pass_bias = bool(not (all_positive or all_negative))
    
    pass_overall = bool(pass_error and pass_coverage and pass_bias)
    
    summary = {
        'n_holdouts': int(len(holdouts)),
        'median_frac_error': float(median_frac_error),
        'inside_68_frac': float(inside_68_count / len(holdouts)) if len(holdouts) > 0 else 0.0,
        'systematic_bias': 'positive' if all_positive else ('negative' if all_negative else 'none'),
        'pass_error_criterion': bool(pass_error),
        'pass_coverage_criterion': bool(pass_coverage),
        'pass_bias_criterion': bool(pass_bias),
        'pass_overall': bool(pass_overall),
        'per_cluster': results
    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'holdout_validation.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'holdout_predictions.csv', index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Hold-outs: {len(holdouts)}")
    print(f"Median frac error: {median_frac_error:.1%} {'✓' if pass_error else '✗'} (< 20%)")
    print(f"Inside 68% CI: {inside_68_count}/{len(holdouts)} {'✓' if pass_coverage else '✗'}")
    print(f"Systematic bias: {summary['systematic_bias']} {'✓' if pass_bias else '✗'}")
    print()
    print(f"OVERALL: {'PASS ✓' if pass_overall else 'FAIL ✗'}")
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Hold-out cluster validation")
    parser.add_argument('--posterior', required=True,
                        help='Path to trace.netcdf from inference')
    parser.add_argument('--holdout', required=True,
                        help='Comma-separated hold-out cluster names')
    parser.add_argument('--use-mass-scaling', type=int, default=1,
                        help='Use mass scaling (1=yes, 0=no)')
    parser.add_argument('--catalog', default=None,
                        help='Path to cluster catalog CSV')
    parser.add_argument('--out', default='output/holdout_validation',
                        help='Output directory')
    
    args = parser.parse_args()
    
    if not HAS_ARVIZ:
        print("ERROR: arviz not available. Install with: pip install arviz", file=sys.stderr)
        sys.exit(1)
    
    # Load posterior
    print(f"Loading posterior: {args.posterior}")
    trace = az.from_netcdf(args.posterior)
    
    # Load hold-outs
    holdout_names = args.holdout.split(',')
    catalog_path = Path(args.catalog) if args.catalog else None
    holdouts = load_holdout_clusters(holdout_names, catalog_path)
    
    # Run validation
    summary = validate_holdouts(
        holdouts,
        trace,
        bool(args.use_mass_scaling),
        Path(args.out)
    )
    
    # Exit code based on pass/fail
    sys.exit(0 if summary['pass_overall'] else 1)


if __name__ == '__main__':
    main()
