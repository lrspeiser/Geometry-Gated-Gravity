#!/usr/bin/env python3
"""
Train/Test Validation for Cooperative Response Model

Splits available clusters into:
- Training set (3-5 clusters): Optimize α_coeff
- Test set (remaining clusters): Validate generalization

This tests if we've found a universal scaling law or just overfitted.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.lensing_utils import CLASH, get_thetaE_observed
from scripts.run_real_cluster_tests import (
    load_baryon_profiles, extract_features
)
from scripts.cooperative_response_gpu import cooperative_response_wrapper
from scripts.optimize_response_params import (
    compute_deflection_from_sigma_eff, find_einstein_radius
)
from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    abel_project_sigma
)

# =============================================================================
# CLUSTER CATALOG WITH OBSERVED EINSTEIN RADII
# =============================================================================

def build_cluster_catalog() -> List[Dict]:
    """Build catalog of all available clusters with observed θ_E."""
    clusters = []
    
    # Check which clusters have data available
    data_dir = ROOT / "data" / "clusters"
    available_folders = {d.name for d in data_dir.iterdir() if d.is_dir()}
    
    for short_id, (local_name, z_lens) in CLASH.items():
        if local_name not in available_folders:
            continue
            
        # Try to get observed Einstein radius
        theta_E_obs = get_thetaE_observed(local_name)
        if theta_E_obs is None or not np.isfinite(theta_E_obs) or theta_E_obs <= 0:
            continue
        
        # Check if baryon data files exist
        gas_file = data_dir / local_name / "gas_profile.csv"
        stars_file = data_dir / local_name / "stars_profile.csv"
        if not (gas_file.exists() and stars_file.exists()):
            continue
        
        clusters.append({
            "short_id": short_id,
            "local_name": local_name,
            "z_lens": z_lens,
            "theta_E_obs": theta_E_obs,
        })
    
    # Sort by redshift for consistent ordering
    clusters.sort(key=lambda c: c["z_lens"])
    return clusters


# =============================================================================
# TRAIN/TEST SPLITS
# =============================================================================

def split_train_test(
    clusters: List[Dict],
    train_names: List[str] = None,
    train_fraction: float = 0.3,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split clusters into train and test sets.
    
    Parameters
    ----------
    clusters : List[Dict]
        Full cluster catalog
    train_names : List[str], optional
        Explicit list of cluster names for training (e.g., ["MACSJ0416", "MACSJ1149"])
        If None, use train_fraction to auto-select
    train_fraction : float
        Fraction to use for training if train_names not specified
    
    Returns
    -------
    train_set, test_set : Tuple[List[Dict], List[Dict]]
    """
    if train_names is not None:
        train_names_upper = [n.upper() for n in train_names]
        train = [c for c in clusters if c["local_name"].upper() in train_names_upper]
        test = [c for c in clusters if c["local_name"].upper() not in train_names_upper]
    else:
        n_train = max(3, int(len(clusters) * train_fraction))
        # Sample uniformly across redshift range
        indices = np.linspace(0, len(clusters)-1, n_train, dtype=int)
        train = [clusters[i] for i in indices]
        test_indices = set(range(len(clusters))) - set(indices)
        test = [clusters[i] for i in sorted(test_indices)]
    
    return train, test


# =============================================================================
# OPTIMIZATION ON TRAINING SET
# =============================================================================

def optimize_alpha_on_train(
    train_clusters: List[Dict],
    alpha_range: Tuple[float, float] = (0.5, 3.0),
    n_points: int = 50,
    z_source: float = 2.0,
    lambda_factor: float = 1.3,
    use_gpu: bool = True,
) -> Tuple[float, Dict]:
    """
    Find optimal α_coeff by minimizing mean error on training set.
    
    Returns
    -------
    alpha_opt : float
        Optimal alpha coefficient
    results : Dict
        Training results including per-cluster errors
    """
    print(f"\n{'='*60}")
    print(f"TRAINING on {len(train_clusters)} clusters")
    print(f"{'='*60}\n")
    
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], n_points)
    
    # Store errors for each cluster
    cluster_errors = {c["local_name"]: [] for c in train_clusters}
    mean_errors = []
    
    for alpha in alpha_values:
        errors = []
        for cluster in train_clusters:
            local_name = cluster["local_name"]
            z_l = cluster["z_lens"]
            theta_E_obs = cluster["theta_E_obs"]
            
            # Load baryon data
            try:
                r3d, rho3d = load_baryon_profiles(local_name, debug=False)
                R = np.logspace(np.log10(max(0.1, float(r3d.min()))), 
                               np.log10(float(r3d.max())), 700)
                Sigma = abel_project_sigma(r3d, rho3d, R)
            except Exception as e:
                errors.append(1000.0)  # Large penalty
                cluster_errors[local_name].append(1000.0)
                continue
            
            # Extract features
            feats = extract_features(R, Sigma)
            eps = max(feats.edge_sharp, 0.01)
            M_core = max(feats.M_core, 1e10)
            R_edge = feats.R_edge
            
            # Predict A_resp and λ
            A_resp = alpha * (eps ** 0.5) * ((M_core / 1e13) ** 0.3)
            lam = lambda_factor * max(R_edge, 1.0)
            
            # Compute effective surface density with cooperative response
            Sigma_eff, Sigma_resp = cooperative_response_wrapper(
                R, Sigma, A_resp, lam, nu=2.0, use_gpu=use_gpu,
                x0=0.3, w=0.3, conserve_mass=False, debug=False
            )
            
            # Compute deflection and find Einstein radius
            theta = np.linspace(5.0, 120.0, 220)
            alpha_model = compute_deflection_from_sigma_eff(R, Sigma_eff, z_l, z_source, theta)
            theta_E_model = find_einstein_radius(theta, alpha_model)
            
            # Error
            error = abs(theta_E_model - theta_E_obs)
            errors.append(error)
            cluster_errors[local_name].append(error)
        
        mean_errors.append(np.mean(errors))
    
    # Find optimal alpha
    opt_idx = int(np.argmin(mean_errors))
    alpha_opt = alpha_values[opt_idx]
    min_error = mean_errors[opt_idx]
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL α_coeff = {alpha_opt:.3f}")
    print(f"  Mean training error = {min_error:.2f} arcsec")
    print(f"{'='*60}\n")
    
    # Per-cluster training errors at optimum
    print("Training Set Performance:")
    for i, cluster in enumerate(train_clusters):
        name = cluster["local_name"]
        error = cluster_errors[name][opt_idx]
        print(f"  {name:15s}: θ_E,obs={cluster['theta_E_obs']:5.1f}\", error={error:5.2f}\"")
    
    results = {
        "alpha_opt": alpha_opt,
        "mean_train_error": min_error,
        "alpha_values": alpha_values.tolist(),
        "mean_errors": mean_errors,
        "cluster_errors": {k: v for k, v in cluster_errors.items()},
    }
    
    return alpha_opt, results


# =============================================================================
# EVALUATION ON TEST SET
# =============================================================================

def evaluate_on_test(
    test_clusters: List[Dict],
    alpha_coeff: float,
    z_source: float = 2.0,
    lambda_factor: float = 1.3,
    use_gpu: bool = True,
) -> Dict:
    """
    Evaluate fixed α_coeff on held-out test set.
    
    Returns
    -------
    results : Dict
        Test results including predictions and errors
    """
    print(f"\n{'='*60}")
    print(f"TESTING on {len(test_clusters)} held-out clusters")
    print(f"  Using fixed α_coeff = {alpha_coeff:.3f}")
    print(f"{'='*60}\n")
    
    predictions = []
    errors = []
    
    for cluster in test_clusters:
        local_name = cluster["local_name"]
        z_l = cluster["z_lens"]
        theta_E_obs = cluster["theta_E_obs"]
        
        # Load baryon data
        try:
            r3d, rho3d = load_baryon_profiles(local_name, debug=False)
            R = np.logspace(np.log10(max(0.1, float(r3d.min()))), 
                           np.log10(float(r3d.max())), 700)
            Sigma = abel_project_sigma(r3d, rho3d, R)
        except Exception as e:
            predictions.append({
                "local_name": local_name,
                "theta_E_obs": theta_E_obs,
                "theta_E_model": np.nan,
                "error": 1000.0,
                "status": "data_missing",
            })
            errors.append(1000.0)
            continue
        
        # Extract features
        feats = extract_features(R, Sigma)
        eps = max(feats.edge_sharp, 0.01)
        M_core = max(feats.M_core, 1e10)
        R_edge = feats.R_edge
        
        # Predict A_resp and λ
        A_resp = alpha_coeff * (eps ** 0.5) * ((M_core / 1e13) ** 0.3)
        lam = lambda_factor * max(R_edge, 1.0)
        
        # Compute effective surface density with cooperative response
        Sigma_eff, Sigma_resp = cooperative_response_wrapper(
            R, Sigma, A_resp, lam, nu=2.0, use_gpu=use_gpu,
            x0=0.3, w=0.3, conserve_mass=False, debug=False
        )
        
        # Compute deflection and find Einstein radius
        theta = np.linspace(5.0, 120.0, 220)
        alpha_model = compute_deflection_from_sigma_eff(R, Sigma_eff, z_l, z_source, theta)
        theta_E_model = find_einstein_radius(theta, alpha_model)
        
        # Error
        error = abs(theta_E_model - theta_E_obs)
        errors.append(error)
        
        predictions.append({
            "local_name": local_name,
            "z_lens": z_l,
            "theta_E_obs": theta_E_obs,
            "theta_E_model": theta_E_model,
            "error": error,
            "status": "ok",
        })
        
        print(f"  {local_name:15s}: θ_E,obs={theta_E_obs:5.1f}\", θ_E,model={theta_E_model:5.1f}\", error={error:5.2f}\"")
    
    mean_test_error = np.mean(errors)
    median_test_error = np.median(errors)
    
    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS:")
    print(f"  Mean error:   {mean_test_error:.2f} arcsec")
    print(f"  Median error: {median_test_error:.2f} arcsec")
    print(f"{'='*60}\n")
    
    return {
        "predictions": predictions,
        "mean_error": mean_test_error,
        "median_error": median_test_error,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train-clusters",
        type=str,
        default="MACSJ0416,MACSJ1149",
        help="Comma-separated list of training cluster names (default: MACSJ0416,MACSJ1149)",
    )
    ap.add_argument(
        "--auto-split",
        action="store_true",
        help="Auto-select training clusters (30%% of data) instead of using --train-clusters",
    )
    ap.add_argument(
        "--alpha-min",
        type=float,
        default=0.5,
        help="Minimum alpha coefficient to test (default: 0.5)",
    )
    ap.add_argument(
        "--alpha-max",
        type=float,
        default=3.0,
        help="Maximum alpha coefficient to test (default: 3.0)",
    )
    ap.add_argument(
        "--n-points",
        type=int,
        default=50,
        help="Number of alpha values to test (default: 50)",
    )
    ap.add_argument(
        "--zs",
        type=float,
        default=2.0,
        help="Source redshift (default: 2.0)",
    )
    ap.add_argument(
        "--lambda-factor",
        type=float,
        default=1.3,
        help="Factor for kernel scale: λ = factor * R_edge (default: 1.3)",
    )
    ap.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (use CPU only)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="out/train_test_validation.json",
        help="Output JSON file for results",
    )
    args = ap.parse_args()
    
    # Build catalog
    print("Building cluster catalog...")
    all_clusters = build_cluster_catalog()
    print(f"Found {len(all_clusters)} clusters with data and observed θ_E\n")
    
    if len(all_clusters) < 4:
        print("ERROR: Need at least 4 clusters for train/test split")
        return 1
    
    # Split into train/test
    if args.auto_split:
        train, test = split_train_test(all_clusters, train_names=None, train_fraction=0.3)
    else:
        train_names = [n.strip() for n in args.train_clusters.split(",") if n.strip()]
        train, test = split_train_test(all_clusters, train_names=train_names)
    
    print(f"Training set: {len(train)} clusters")
    for c in train:
        print(f"  - {c['local_name']} (z={c['z_lens']:.3f}, θ_E,obs={c['theta_E_obs']:.1f}\")")
    
    print(f"\nTest set: {len(test)} clusters")
    for c in test:
        print(f"  - {c['local_name']} (z={c['z_lens']:.3f}, θ_E,obs={c['theta_E_obs']:.1f}\")")
    
    # Optimize on training set
    use_gpu = not args.no_gpu
    alpha_opt, train_results = optimize_alpha_on_train(
        train,
        alpha_range=(args.alpha_min, args.alpha_max),
        n_points=args.n_points,
        z_source=args.zs,
        lambda_factor=args.lambda_factor,
        use_gpu=use_gpu,
    )
    
    # Evaluate on test set
    test_results = evaluate_on_test(
        test,
        alpha_coeff=alpha_opt,
        z_source=args.zs,
        lambda_factor=args.lambda_factor,
        use_gpu=use_gpu,
    )
    
    # Save results
    output = {
        "train_clusters": [c["local_name"] for c in train],
        "test_clusters": [c["local_name"] for c in test],
        "alpha_opt": alpha_opt,
        "train_results": train_results,
        "test_results": test_results,
        "parameters": {
            "alpha_range": [args.alpha_min, args.alpha_max],
            "n_points": args.n_points,
            "z_source": args.zs,
            "lambda_factor": args.lambda_factor,
        },
    }
    
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {out_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Optimal α_coeff:      {alpha_opt:.3f}")
    print(f"Mean train error:     {train_results['mean_train_error']:.2f}\"")
    print(f"Mean test error:      {test_results['mean_error']:.2f}\"")
    print(f"Median test error:    {test_results['median_error']:.2f}\"")
    print(f"Test/Train ratio:     {test_results['mean_error'] / train_results['mean_train_error']:.2f}x")
    print(f"{'='*60}\n")
    
    # Interpretation
    ratio = test_results['mean_error'] / train_results['mean_train_error']
    if ratio < 1.5:
        print("✅ Good generalization - model works on unseen clusters!")
    elif ratio < 2.5:
        print("⚠️  Moderate generalization - some overfitting to training set")
    else:
        print("❌ Poor generalization - model likely overfit to training clusters")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
