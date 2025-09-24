#!/usr/bin/env python3
"""
SPARC Formula Sweep - Find Universal Formulas Without Per-Galaxy Tuning
========================================================================

Tests hundreds of formula variants on ALL SPARC galaxies simultaneously.
Parameters are derived from observable galaxy properties, not tuned per-galaxy.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple

# Import our GPU solver suite
from g3_gpu_solver_suite import (
    xp, _GPU, G3Model, G3Params, TEMPLATE_BOUNDS, 
    SolverOrchestrator, _variants_default
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
G = 4.300917270e-6  # kpc km^2 s^-2 Msun^-1

class SPARCDataset:
    """Load and preprocess SPARC galaxy data."""
    
    def __init__(self, rotmod_path: str, meta_path: str = None):
        self.df = pd.read_parquet(rotmod_path)
        self.df_meta = pd.read_parquet(meta_path) if meta_path else None
        self.galaxies = {}
        self._load_galaxies()
        
    def _load_galaxies(self):
        """Load all SPARC galaxies with observable properties."""
        for name, gdf in self.df.groupby('galaxy'):
            # Basic data
            r = gdf['R_kpc'].values
            v_obs = gdf['Vobs_kms'].values
            v_err = gdf['eVobs_kms'].values
            v_gas = gdf['Vgas_kms'].values
            v_disk = gdf['Vdisk_kms'].values
            v_bulge = gdf['Vbul_kms'].values if 'Vbul_kms' in gdf else np.zeros_like(r)
            
            # Total baryonic velocity
            v_bar = np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)
            
            # Baryonic acceleration (Newtonian)
            g_N = v_bar**2 / (r + 1e-10)
            
            # Quality filter
            valid = (r > 0.1) & np.isfinite(v_obs) & np.isfinite(v_bar) & (v_obs > 10)
            if valid.sum() < 5:
                continue
                
            # Derive observable properties (no tuning!)
            r_valid = r[valid]
            v_bar_valid = v_bar[valid]
            
            # 1. Half-mass radius proxy (median radius weighted by v_bar^2)
            weights = v_bar_valid**2
            r_half = np.average(r_valid, weights=weights) if weights.sum() > 0 else np.median(r_valid)
            
            # 2. Mean surface density proxy (from rotation curve amplitude)
            # Sigma_bar ~ M_total / (pi * r_max^2) ~ v_max^2 / (G * r_max)
            r_max = np.max(r_valid)
            v_max = np.max(v_bar_valid)
            sigma_bar = (v_max**2 / (4 * np.pi * G * r_max)) / 1e6  # Convert to Msun/pc^2
            sigma_bar = np.clip(sigma_bar, 10, 1000)  # Reasonable bounds
            
            # 3. Local surface density (exponential disk approximation)
            # Sigma(r) = Sigma_0 * exp(-r/r_d) where r_d ~ r_half/1.68
            r_d = r_half / 1.68
            sigma_0 = sigma_bar * 2  # Central density ~ 2x mean for exponential disk
            sigma_loc = sigma_0 * np.exp(-r_valid / r_d)
            
            # Get galaxy type if available
            galaxy_type = 'Unknown'
            if self.df_meta is not None and name in self.df_meta['galaxy'].values:
                meta = self.df_meta[self.df_meta['galaxy'] == name].iloc[0]
                T_val = meta['T'] if 'T' in meta else 99
                if T_val <= 0:
                    galaxy_type = 'Early'
                elif 1 <= T_val <= 3:
                    galaxy_type = 'Sa-Sb'
                elif 4 <= T_val <= 5:
                    galaxy_type = 'Sbc-Sc'
                elif 6 <= T_val <= 7:
                    galaxy_type = 'Scd-Sd'
                elif T_val >= 8:
                    galaxy_type = 'Sdm-Irr'
                    
            self.galaxies[name] = {
                'r': r_valid,
                'v_obs': v_obs[valid],
                'v_bar': v_bar_valid,
                'g_N': g_N[valid],
                'sigma_loc': sigma_loc,
                'r_half': r_half,
                'sigma_bar': sigma_bar,
                'type': galaxy_type
            }
            
        logger.info(f"Loaded {len(self.galaxies)} SPARC galaxies")
        
        # Report by type
        type_counts = {}
        for name, data in self.galaxies.items():
            gtype = data['type']
            type_counts[gtype] = type_counts.get(gtype, 0) + 1
        for gtype, count in sorted(type_counts.items()):
            logger.info(f"  {gtype}: {count} galaxies")

def create_extended_variants():
    """Create extended set of formula variants to explore."""
    variants = []
    
    # Priority combinations based on physical intuition
    priority = [
        # Standard forms
        {"gating_type": "rational", "screen_type": "sigmoid", "exponent_type": "logistic_r"},
        {"gating_type": "rational", "screen_type": "exp", "exponent_type": "logistic_r"},
        {"gating_type": "rational", "screen_type": "powerlaw", "exponent_type": "logistic_r"},
        
        # Smooth transitions
        {"gating_type": "tanh", "screen_type": "sigmoid", "exponent_type": "logistic_r"},
        {"gating_type": "tanh", "screen_type": "exp", "exponent_type": "sigma_tuned"},
        
        # Higher-order rational
        {"gating_type": "rational2", "screen_type": "sigmoid", "exponent_type": "logistic_r"},
        {"gating_type": "rational2", "screen_type": "powerlaw", "exponent_type": "sigma_tuned"},
        
        # No screening (ablation)
        {"gating_type": "rational", "screen_type": "none", "exponent_type": "logistic_r"},
        {"gating_type": "tanh", "screen_type": "none", "exponent_type": "sigma_tuned"},
        
        # Softplus variants
        {"gating_type": "softplus", "screen_type": "sigmoid", "exponent_type": "logistic_r"},
        {"gating_type": "softplus", "screen_type": "exp", "exponent_type": "sigma_tuned"},
    ]
    
    # Add systematic grid search
    gatings = ["rational", "rational2", "tanh", "softplus"]
    screens = ["sigmoid", "exp", "powerlaw", "none"]
    exponents = ["logistic_r", "sigma_tuned"]
    
    # Add unique combinations not in priority list
    existing = set(str(v) for v in priority)
    for g in gatings:
        for s in screens:
            for e in exponents:
                variant = {"gating_type": g, "screen_type": s, "exponent_type": e}
                if str(variant) not in existing:
                    priority.append(variant)
                    existing.add(str(variant))
                    if len(priority) >= 30:  # Limit total variants
                        return priority
                        
    return priority

def objective_sparc_multi(params_vec, sparc_data: SPARCDataset, variant: dict, 
                         metric="median", galaxy_subset=None) -> float:
    """
    Evaluate parameters on multiple SPARC galaxies simultaneously.
    No per-galaxy tuning - same parameters for all.
    """
    # Clip parameters to bounds
    keys = list(TEMPLATE_BOUNDS.keys())
    lo = xp.asarray([TEMPLATE_BOUNDS[k][0] for k in keys])
    hi = xp.asarray([TEMPLATE_BOUNDS[k][1] for k in keys])
    x = xp.clip(params_vec, lo, hi)
    
    # Create model with parameters
    p = G3Params(
        v0_kms=float(x[0]), rc0_kpc=float(x[1]), gamma=float(x[2]), beta=float(x[3]),
        sigma_star=float(x[4]), alpha=float(x[5]), kappa=float(x[6]), eta=float(x[7]),
        delta_kpc=float(x[8]), p_in=float(x[9]), p_out=float(x[10]), g_sat=float(x[11]),
        gating_type=variant.get("gating_type", "rational"),
        screen_type=variant.get("screen_type", "sigmoid"),
        exponent_type=variant.get("exponent_type", "logistic_r"),
    )
    model = G3Model(p)
    
    # Evaluate on each galaxy
    all_errors = []
    galaxy_names = galaxy_subset if galaxy_subset else list(sparc_data.galaxies.keys())
    
    for name in galaxy_names:
        data = sparc_data.galaxies[name]
        
        # Convert to GPU arrays
        r = xp.asarray(data['r'])
        v_obs = xp.asarray(data['v_obs'])
        g_N = xp.asarray(data['g_N'])
        sigma_loc = xp.asarray(data['sigma_loc'])
        r_half = data['r_half']
        sigma_bar = data['sigma_bar']
        
        # Predict
        v_pred = model.predict_vel_kms(r, g_N, sigma_loc, r_half, sigma_bar)
        
        # Compute errors
        rel_errors = xp.abs(v_pred - v_obs) / (v_obs + 1e-10)
        rel_errors = rel_errors[xp.isfinite(rel_errors)]
        
        if len(rel_errors) > 0:
            # Weight by number of points (galaxies with more data count more)
            weight = len(rel_errors)
            if metric == "median":
                galaxy_error = float(xp.median(rel_errors))
            else:
                galaxy_error = float(xp.mean(rel_errors))
            all_errors.append((galaxy_error, weight))
    
    if len(all_errors) == 0:
        return 1.0
        
    # Aggregate across galaxies (weighted by number of points)
    errors, weights = zip(*all_errors)
    if metric == "median":
        # Weighted median
        sorted_pairs = sorted(zip(errors, weights))
        cum_weight = 0
        total_weight = sum(weights)
        for error, weight in sorted_pairs:
            cum_weight += weight
            if cum_weight >= total_weight / 2:
                return error
        return sorted_pairs[-1][0]
    else:
        # Weighted mean
        return np.average(errors, weights=weights)

def analyze_results_by_type(sparc_data: SPARCDataset, best_params, best_variant):
    """Analyze performance by galaxy type."""
    results_by_type = {}
    
    # Group galaxies by type
    galaxy_types = {}
    for name, data in sparc_data.galaxies.items():
        gtype = data['type']
        if gtype not in galaxy_types:
            galaxy_types[gtype] = []
        galaxy_types[gtype].append(name)
        
    # Evaluate each type
    for gtype, galaxy_names in galaxy_types.items():
        error = objective_sparc_multi(best_params, sparc_data, best_variant, 
                                     metric="median", galaxy_subset=galaxy_names)
        results_by_type[gtype] = {
            'median_error': error,
            'n_galaxies': len(galaxy_names)
        }
        
    return results_by_type

def main():
    parser = argparse.ArgumentParser(description="SPARC Formula Sweep - Find Universal Formulas")
    parser.add_argument("--rotmod_parquet", type=str, 
                       default="data/sparc_rotmod_ltg.parquet",
                       help="Path to SPARC rotation curves")
    parser.add_argument("--meta_parquet", type=str,
                       default="data/sparc_master_clean.parquet",
                       help="Path to SPARC metadata")
    parser.add_argument("--metric", type=str, default="median", 
                       choices=["median", "mean"])
    parser.add_argument("--branches", type=int, default=6,
                       help="Number of parallel search branches")
    parser.add_argument("--iters", type=int, default=500,
                       help="Iterations per branch")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--global_patience", type=int, default=150)
    parser.add_argument("--min_delta", type=float, default=0.001,
                       help="Minimum improvement to reset patience")
    parser.add_argument("--time_limit_s", type=float, default=1800,
                       help="Time limit in seconds (default 30 min)")
    parser.add_argument("--outdir", type=str, default="out/sparc_formula_sweep")
    parser.add_argument("--max_galaxies", type=int, default=None,
                       help="Limit number of galaxies for testing")
    
    args = parser.parse_args()
    
    # Load SPARC data
    logger.info("="*60)
    logger.info("SPARC FORMULA SWEEP - NO PER-GALAXY TUNING")
    logger.info("="*60)
    
    sparc_data = SPARCDataset(args.rotmod_parquet, args.meta_parquet)
    
    if args.max_galaxies:
        # Limit for testing
        galaxy_names = list(sparc_data.galaxies.keys())[:args.max_galaxies]
        filtered = {k: sparc_data.galaxies[k] for k in galaxy_names}
        sparc_data.galaxies = filtered
        logger.info(f"Limited to {args.max_galaxies} galaxies for testing")
    
    # Create evaluation function
    def eval_fn(params_vec, variant):
        return objective_sparc_multi(params_vec, sparc_data, variant, 
                                    metric=args.metric) * 100  # Convert to percent
    
    # Get variants
    variants = create_extended_variants()
    logger.info(f"Testing {len(variants)} formula variants")
    
    # Run orchestrator
    orchestrator = SolverOrchestrator(
        TEMPLATE_BOUNDS, eval_fn, variants,
        patience_iters=args.patience,
        min_delta=args.min_delta,
        global_patience=args.global_patience,
        time_limit_s=args.time_limit_s,
        seed=42
    )
    
    logger.info(f"Starting multi-solver search with {args.branches} branches...")
    start_time = time.time()
    
    best = orchestrator.run(
        branches=args.branches,
        iters_per_branch=args.iters,
        outdir=args.outdir
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Search completed in {elapsed:.1f} seconds")
    
    # Analyze by galaxy type
    logger.info("\nAnalyzing performance by galaxy type...")
    results_by_type = analyze_results_by_type(sparc_data, xp.asarray(best["x"]), best["variant"])
    
    # Save comprehensive results
    output_dir = Path(args.outdir)
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_used": _GPU,
        "n_galaxies": len(sparc_data.galaxies),
        "n_variants_tested": len(variants),
        "best_score_percent": best["score"],
        "best_params_vector": best["x"],
        "best_variant": best["variant"],
        "results_by_type": results_by_type,
        "param_names": list(TEMPLATE_BOUNDS.keys()),
        "bounds": TEMPLATE_BOUNDS,
        "search_time_s": elapsed,
        "config": {
            "metric": args.metric,
            "branches": args.branches,
            "iters": args.iters,
            "patience": args.patience
        }
    }
    
    with open(output_dir / "sparc_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    # Print summary
    print("\n" + "="*60)
    print("SPARC FORMULA SWEEP RESULTS")
    print("="*60)
    print(f"Best overall error: {best['score']:.2f}%")
    print(f"Best variant: {best['variant']}")
    print("\nPerformance by galaxy type:")
    for gtype, res in sorted(results_by_type.items(), key=lambda x: x[1]['median_error']):
        print(f"  {gtype:12s}: {res['median_error']*100:.2f}% ({res['n_galaxies']} galaxies)")
    
    print("\nConclusion:")
    if best['score'] < 10:
        print("✅ EXCELLENT! Found formula that works universally < 10% error")
    elif best['score'] < 15:
        print("✅ GOOD! Formula achieves < 15% error without per-galaxy tuning")
    elif best['score'] < 20:
        print("⚠️ MODERATE - Formula needs refinement for universal application")
    else:
        print("❌ More work needed - consider additional observable dependencies")
        
    print(f"\nResults saved to: {args.outdir}")
    print("="*60)

if __name__ == "__main__":
    main()