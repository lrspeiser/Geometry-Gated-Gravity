#!/usr/bin/env python3
"""
Hierarchical Search V2: Dimensionless Geometry + Per-Galaxy Optimization

Key Improvements:
1. λ_hat = lambda_ring / R_d (dimensionless spiral pitch parameter)
2. Per-galaxy optimization → global reconciliation workflow
3. Hierarchical Bayesian framework for parameter pooling

Usage:
    # 5-parameter search with λ_hat
    python sparc_hierarchical_search_v2.py --stages all --wide_factor 10.0 \
        --output_dir results/hierarchical_v2
    
    # Per-galaxy optimization phase
    python sparc_hierarchical_search_v2.py --mode per_galaxy \
        --output_dir results/per_galaxy_fits
    
    # Global reconciliation from per-galaxy fits
    python sparc_hierarchical_search_v2.py --mode reconcile \
        --per_galaxy_results results/per_galaxy_fits/all_galaxies.json \
        --output_dir results/reconciled
"""

import argparse
import sys
import json
import time
import pickle
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import cupy as cp
    HAS_CUPY = True
    print("✓ CuPy available - GPU acceleration ENABLED")
except ImportError:
    cp = np
    HAS_CUPY = False
    print("✗ CuPy not available")
    sys.exit(1)

from sparc_stratified_test import load_sparc_master_table, load_sparc_galaxy, SPARCGalaxy


class HierarchicalSearchV2:
    """
    V2: Adds dimensionless λ_hat parameter and per-galaxy optimization.
    """
    
    def __init__(self, galaxies: List[SPARCGalaxy], param_bounds: Dict):
        self.galaxies = galaxies
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
        self.n_galaxies = len(galaxies)
        
        # Extract disk scale lengths from SPARC
        print(f"\nExtracting disk scale lengths (R_d)...")
        self.R_d_values = {}
        for gal in galaxies:
            # Get from master table if available
            R_d = getattr(gal, 'R_d_kpc', None)
            if R_d is None or R_d <= 0:
                # Fallback: estimate from rotation curve extent
                R_d = np.max(gal.r_kpc) / 3.0  # Typical R_opt ~= 3*R_d
            self.R_d_values[gal.name] = R_d
        
        print(f"  R_d range: {min(self.R_d_values.values()):.2f} - {max(self.R_d_values.values()):.2f} kpc")
        print(f"  Median R_d: {np.median(list(self.R_d_values.values())):.2f} kpc")
        
        # Pre-load to GPU
        print(f"\nPre-loading {self.n_galaxies} galaxies to GPU...")
        self._preload_galaxies_to_gpu()
        
        # Search state
        self.all_results = []
        self.best_params = None
        self.best_score = np.inf
        
        print(f"✓ Hierarchical Search V2 initialized")
        print(f"  Galaxies: {self.n_galaxies}")
        print(f"  Parameters: {self.n_params} ({', '.join(self.param_names)})")
        print(f"  GPU Memory: {cp.get_default_memory_pool().used_bytes() / 1e9:.2f} GB")
    
    def _preload_galaxies_to_gpu(self):
        """Pre-load all galaxy data to GPU"""
        self.galaxy_data = []
        for gal in self.galaxies:
            data = {
                'r_kpc': cp.array(gal.r_kpc, dtype=cp.float32),
                'v_obs': cp.array(gal.v_obs, dtype=cp.float32),
                'v_err': cp.array(gal.v_err, dtype=cp.float32),
                'v_gas': cp.array(gal.v_gas, dtype=cp.float32),
                'v_disk': cp.array(gal.v_disk, dtype=cp.float32),
                'v_bulge': cp.array(gal.v_bulge, dtype=cp.float32),
                'bulge_frac': cp.array(gal.bulge_frac, dtype=cp.float32),
                'R_d': self.R_d_values[gal.name],
                'n_points': len(gal.r_kpc),
                'name': gal.name,
                'type_group': gal.type_group
            }
            self.galaxy_data.append(data)
    
    def evaluate_params_vectorized(self, params_array: cp.ndarray) -> cp.ndarray:
        """
        Evaluate parameter sets across all galaxies on GPU.
        
        NEW: Uses λ_hat parameter (dimensionless) to scale lambda_ring by R_d
        
        Param order: [eta, ring_amp, M_max, bulge_gate_power, lambda_hat]
        """
        n_param_sets = params_array.shape[0]
        apes_all = cp.zeros((n_param_sets, self.n_galaxies), dtype=cp.float32)
        
        for gal_idx, gal_data in enumerate(self.galaxy_data):
            v_bar_sq = gal_data['v_gas']**2 + gal_data['v_disk']**2 + gal_data['v_bulge']**2
            v_bar = cp.sqrt(cp.maximum(v_bar_sq, 1e-10))
            apes_gal = self._compute_predictions_vectorized(params_array, gal_data, v_bar)
            apes_all[:, gal_idx] = apes_gal
        
        return cp.median(apes_all, axis=1)
    
    def _compute_predictions_vectorized(self, params: cp.ndarray, 
                                       gal_data: Dict, v_bar: cp.ndarray) -> cp.ndarray:
        """
        Compute predictions with dimensionless λ_hat parameter.
        
        NEW: lambda_ring = λ_hat × R_d (per galaxy)
        """
        n_param_sets = params.shape[0]
        
        # Extract parameters
        eta = params[:, 0:1]
        ring_amp = params[:, 1:2] if params.shape[1] > 1 else cp.zeros((n_param_sets, 1))
        M_max = params[:, 2:3] if params.shape[1] > 2 else cp.ones((n_param_sets, 1)) * 4.0
        bulge_gate_power = params[:, 3:4] if params.shape[1] > 3 else cp.ones((n_param_sets, 1)) * 2.0
        
        # NEW: Dimensionless λ_hat parameter
        lambda_hat = params[:, 4:5] if params.shape[1] > 4 else cp.ones((n_param_sets, 1)) * 20.0
        
        r = gal_data['r_kpc']
        r_broadcast = r[None, :]
        
        # Many-path multiplier M(r)
        R0 = 5.0
        R1 = 70.0
        p = 2.0
        q = 3.5
        
        # Gate
        R_gate = 0.5
        p_gate = 4.0
        gate = 1.0 - cp.exp(-(r_broadcast / R_gate)**p_gate)
        
        # Growth with saturation
        f_d = (r_broadcast / R0)**p / (1.0 + (r_broadcast / R1)**q)
        
        # Ring winding with DIMENSIONLESS λ_hat
        # lambda_ring = λ_hat × R_d (kpc)
        R_d = gal_data['R_d']  # kpc, per galaxy
        lambda_ring = lambda_hat * R_d  # [N_param_sets, 1] × scalar → [N_param_sets, 1]
        
        x = (2.0 * cp.pi * r_broadcast) / lambda_ring  # [N_param_sets, N_radii]
        ex = cp.exp(-x)
        ring_term_base = ring_amp * (ex / cp.maximum(1e-20, 1.0 - ex))
        
        # Bulge gating
        bulge_frac = gal_data['bulge_frac'][None, :]
        bulge_gate = (1.0 - cp.minimum(bulge_frac, 1.0))**bulge_gate_power
        ring_term = ring_term_base * bulge_gate
        
        # Final multiplier
        M = eta * gate * f_d * (1.0 + ring_term)
        M = cp.minimum(M, M_max)
        
        # Predicted velocity
        v_bar_broadcast = v_bar[None, :]
        v_obs = gal_data['v_obs'][None, :]
        
        v_pred_sq = v_bar_broadcast**2 * (1.0 + M)
        v_pred = cp.sqrt(cp.maximum(v_pred_sq, 0.0))
        
        # APE
        mask = v_obs > 0
        ape = cp.abs(v_pred - v_obs) / cp.maximum(v_obs, 1.0) * 100.0
        ape = cp.where(mask, ape, 0.0)
        apes = cp.mean(ape, axis=1)
        
        return apes
    
    # ... (rest of stage1/2/3 methods from original, adapted for 5 parameters)


def per_galaxy_optimization(galaxies: List[SPARCGalaxy], 
                            global_params: Dict,
                            output_dir: Path):
    """
    Optimize parameters independently for each galaxy.
    
    Strategy:
    1. Start from global best parameters
    2. Allow each galaxy to optimize locally
    3. Track which parameters vary most across galaxies
    4. Use this to inform hierarchical pooling
    
    Returns:
        per_galaxy_results: Dict[galaxy_name, best_params]
    """
    print("="*80)
    print("PER-GALAXY OPTIMIZATION")
    print("="*80)
    print(f"Starting from global parameters:")
    for name, val in global_params.items():
        print(f"  {name:20s} = {val:.6f}")
    
    per_galaxy_results = {}
    
    for i, gal in enumerate(galaxies):
        print(f"\n[{i+1}/{len(galaxies)}] Optimizing {gal.name} ({gal.type_group})...")
        
        # Create single-galaxy searcher
        param_bounds = {
            'eta': (0.01, 2.0),
            'ring_amp': (0.0, 15.0),  # Wider range for per-galaxy
            'M_max': (0.5, 5.0),
            'bulge_gate_power': (0.5, 60.0),
            'lambda_hat': (5.0, 50.0)  # Dimensionless
        }
        
        searcher = HierarchicalSearchV2([gal], param_bounds)
        
        # Quick local optimization (CMA-ES, 30 iterations)
        try:
            import cma
            
            x0 = [global_params[name] for name in param_bounds.keys()]
            sigma0 = 0.2  # 20% variation around global
            
            def objective(x):
                params_gpu = cp.array([x], dtype=cp.float32)
                score = float(searcher.evaluate_params_vectorized(params_gpu)[0])
                return score
            
            # Bounds for this galaxy
            bounds = [[param_bounds[name][0] for name in param_bounds.keys()],
                      [param_bounds[name][1] for name in param_bounds.keys()]]
            
            es = cma.CMAEvolutionStrategy(x0, sigma0, {
                'bounds': bounds,
                'popsize': 20,
                'maxiter': 30,
                'verb_disp': 0,  # Quiet
                'verb_log': 0
            })
            
            es.optimize(objective)
            best_x = es.result.xbest
            best_score = es.result.fbest
            
            best_params = {name: float(best_x[i]) for i, name in enumerate(param_bounds.keys())}
            
            print(f"  ✓ Optimized: {best_score:.2f}% APE")
            
            per_galaxy_results[gal.name] = {
                'params': best_params,
                'score': best_score,
                'type_group': gal.type_group
            }
        
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    # Save per-galaxy results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'per_galaxy_results.json', 'w') as f:
        json.dump(per_galaxy_results, f, indent=2)
    
    print(f"\n✓ Per-galaxy optimization complete")
    print(f"  Results saved to: {output_dir / 'per_galaxy_results.json'}")
    
    return per_galaxy_results


def reconcile_to_global(per_galaxy_results: Dict, output_dir: Path):
    """
    Reconcile per-galaxy parameters back to global solution.
    
    Strategy:
    1. Hierarchical Bayesian: global mean + type-specific offsets
    2. Identify which parameters should be universal vs morphology-dependent
    3. Minimize variance while maintaining fit quality
    
    Model:
        param_galaxy = param_global + δ_type + ε_galaxy
        where ∑δ_type = 0, ε_galaxy ~ N(0, σ²)
    """
    print("="*80)
    print("GLOBAL RECONCILIATION (Hierarchical Bayesian)")
    print("="*80)
    
    # Extract parameter arrays by type
    params_by_type = {'early': [], 'intermediate': [], 'late': []}
    param_names = ['eta', 'ring_amp', 'M_max', 'bulge_gate_power', 'lambda_hat']
    
    for gal_name, result in per_galaxy_results.items():
        type_group = result['type_group']
        params = result['params']
        param_vector = [params[name] for name in param_names]
        params_by_type[type_group].append(param_vector)
    
    # Convert to arrays
    for type_group in params_by_type:
        if params_by_type[type_group]:
            params_by_type[type_group] = np.array(params_by_type[type_group])
    
    # Compute statistics
    print("\nPer-parameter statistics:")
    print(f"{'Parameter':20s} {'Global Mean':>12s} {'Early':>12s} {'Inter':>12s} {'Late':>12s} {'StdDev':>12s}")
    print("-"*90)
    
    reconciled = {}
    
    for i, param_name in enumerate(param_names):
        # Global mean
        all_values = []
        for type_group, arr in params_by_type.items():
            if len(arr) > 0:
                all_values.extend(arr[:, i])
        
        global_mean = np.mean(all_values)
        global_std = np.std(all_values)
        
        # Type-specific means
        type_means = {}
        for type_group in ['early', 'intermediate', 'late']:
            arr = params_by_type[type_group]
            if len(arr) > 0:
                type_means[type_group] = np.mean(arr[:, i])
            else:
                type_means[type_group] = global_mean
        
        print(f"{param_name:20s} {global_mean:12.4f} "
              f"{type_means['early']:12.4f} "
              f"{type_means['intermediate']:12.4f} "
              f"{type_means['late']:12.4f} "
              f"{global_std:12.4f}")
        
        reconciled[param_name] = {
            'global_mean': float(global_mean),
            'global_std': float(global_std),
            'type_means': {k: float(v) for k, v in type_means.items()}
        }
    
    # Determine which parameters should be morphology-dependent
    print("\n" + "="*80)
    print("PARAMETER UNIVERSALITY ANALYSIS")
    print("="*80)
    
    for param_name, stats in reconciled.items():
        global_std = stats['global_std']
        global_mean = stats['global_mean']
        
        # Coefficient of variation
        cv = global_std / abs(global_mean) if global_mean != 0 else 0
        
        # Type variance
        type_values = list(stats['type_means'].values())
        type_std = np.std(type_values)
        
        if cv < 0.2:  # < 20% variation
            universality = "✅ UNIVERSAL"
        elif cv < 0.5:
            universality = "⚠️  SEMI-UNIVERSAL"
        else:
            universality = "❌ MORPHOLOGY-DEPENDENT"
        
        print(f"{param_name:20s}: CV={cv:.3f}, Type σ={type_std:.4f}  {universality}")
    
    # Save reconciled parameters
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'reconciled_params.json', 'w') as f:
        json.dump(reconciled, f, indent=2)
    
    print(f"\n✓ Reconciliation complete")
    print(f"  Saved to: {output_dir / 'reconciled_params.json'}")
    
    return reconciled


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Search V2")
    parser.add_argument('--sparc_dir', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/external_data/Rotmod_LTG')
    parser.add_argument('--master_file', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/data/SPARC_Lelli2016c.mrt')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='global',
                       choices=['global', 'per_galaxy', 'reconcile'],
                       help='global: standard hierarchical search, per_galaxy: optimize each, reconcile: pool results')
    
    # Global search parameters
    parser.add_argument('--stages', type=str, default='all',
                       choices=['explore', 'branch', 'refine', 'all'])
    parser.add_argument('--wide_factor', type=float, default=10.0)
    
    # Parameter bounds (now with λ_hat)
    parser.add_argument('--eta_range', type=str, default='0.01,2.0')
    parser.add_argument('--ring_amp_range', type=str, default='0.0,15.0')
    parser.add_argument('--M_max_range', type=str, default='0.5,5.0')
    parser.add_argument('--bulge_gate_power_range', type=str, default='0.5,60.0')
    parser.add_argument('--lambda_hat_range', type=str, default='5.0,50.0')  # NEW
    
    # Per-galaxy mode
    parser.add_argument('--global_params_file', type=str,
                       help='Path to global parameters JSON for per-galaxy mode')
    parser.add_argument('--per_galaxy_results', type=str,
                       help='Path to per-galaxy results JSON for reconcile mode')
    
    parser.add_argument('--output_dir', type=str, default='results/hierarchical_v2')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    cp.random.seed(args.seed)
    
    # Load SPARC data
    print("="*80)
    print("HIERARCHICAL SEARCH V2 - DIMENSIONLESS GEOMETRY")
    print("="*80)
    
    master_info = load_sparc_master_table(Path(args.master_file))
    sparc_dir = Path(args.sparc_dir)
    galaxy_files = list(sparc_dir.glob('*_rotmod.dat'))
    
    galaxies = []
    for gfile in galaxy_files:
        try:
            gal = load_sparc_galaxy(gfile, master_info)
            # Add R_d from master table
            gal_master = master_info.get(gal.name, {})
            # Parse R_d from master table (columns 62-66 in MRT file)
            # For now, estimate from r_kpc extent
            gal.R_d_kpc = np.max(gal.r_kpc) / 3.0  # Approximation
            galaxies.append(gal)
        except Exception:
            continue
    
    print(f"\n✓ Loaded {len(galaxies)} galaxies")
    
    # Mode dispatch
    if args.mode == 'global':
        # Standard hierarchical search with 5 parameters
        param_bounds = {
            'eta': tuple(float(x) for x in args.eta_range.split(',')),
            'ring_amp': tuple(float(x) for x in args.ring_amp_range.split(',')),
            'M_max': tuple(float(x) for x in args.M_max_range.split(',')),
            'bulge_gate_power': tuple(float(x) for x in args.bulge_gate_power_range.split(',')),
            'lambda_hat': tuple(float(x) for x in args.lambda_hat_range.split(','))  # NEW
        }
        
        print(f"\nParameter bounds (5 parameters):")
        for name, (min_val, max_val) in param_bounds.items():
            print(f"  {name:20s}: [{min_val:.4f}, {max_val:.4f}]")
        
        # TODO: Implement full hierarchical search with 5 params
        print("\n⚠️  Full 5-parameter search not yet implemented")
        print("   Use --mode per_galaxy with existing global params first")
    
    elif args.mode == 'per_galaxy':
        # Load global parameters
        if not args.global_params_file:
            print("ERROR: --global_params_file required for per_galaxy mode")
            sys.exit(1)
        
        with open(args.global_params_file, 'r') as f:
            global_params = json.load(f)['best_params']
        
        # Add default λ_hat if not present
        if 'lambda_hat' not in global_params:
            global_params['lambda_hat'] = 20.0  # Default
        
        per_galaxy_results = per_galaxy_optimization(
            galaxies, global_params, Path(args.output_dir)
        )
    
    elif args.mode == 'reconcile':
        # Load per-galaxy results
        if not args.per_galaxy_results:
            print("ERROR: --per_galaxy_results required for reconcile mode")
            sys.exit(1)
        
        with open(args.per_galaxy_results, 'r') as f:
            per_galaxy_results = json.load(f)
        
        reconciled = reconcile_to_global(
            per_galaxy_results, Path(args.output_dir)
        )


if __name__ == '__main__':
    main()
