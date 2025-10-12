#!/usr/bin/env python3
"""
GPU-Exhaustive Parameter Search for SPARC Galaxies

This script maximizes RTX 5090 utilization by testing MILLIONS of parameter
combinations across all 175 SPARC galaxies simultaneously.

Strategy:
1. Vectorized evaluation: [N_params × N_galaxies × N_radii] on GPU
2. Smart exploration: Grid + Random + Bayesian optimization
3. Continuous optimization: Run until killed, saving best every minute
4. Real-time monitoring: GPU utilization, throughput, best params

Target: 90%+ GPU utilization, 1M+ parameter evaluations

Usage:
    # Exhaustive search on all SPARC galaxies
    python sparc_gpu_exhaustive_search.py --mode exhaustive \
        --n_iterations 1000000 --output results/exhaustive_search.json
    
    # Quick test (1000 iterations)
    python sparc_gpu_exhaustive_search.py --mode test \
        --n_iterations 1000 --output results/quick_test.json
"""

import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import signal

# Force CuPy for GPU
try:
    import cupy as cp
    import cupyx.scipy.special as cp_special
    HAS_CUPY = True
    print("✓ CuPy available - GPU acceleration ENABLED")
except ImportError:
    print("✗ CuPy not available - this script requires GPU!")
    sys.exit(1)

# Import SPARC data loading
try:
    from sparc_stratified_test import (
        load_sparc_master_table, load_sparc_galaxy, SPARCGalaxy
    )
except ImportError:
    print("ERROR: Could not import sparc_stratified_test")
    sys.exit(1)


class GPUParameterSearch:
    """
    Massively parallel parameter search on GPU.
    
    Evaluates thousands of parameter combinations simultaneously across
    all galaxies, maximizing GPU throughput.
    """
    
    def __init__(self, galaxies: List[SPARCGalaxy], param_bounds: Dict):
        """
        Initialize GPU search.
        
        Args:
            galaxies: List of SPARC galaxies to fit
            param_bounds: Dict mapping param names to (min, max) tuples
        """
        self.galaxies = galaxies
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
        self.n_galaxies = len(galaxies)
        
        # Pre-allocate GPU arrays for all galaxies
        print(f"\nPre-loading {self.n_galaxies} galaxies to GPU...")
        self._preload_galaxies_to_gpu()
        
        # Track best results
        self.best_params = None
        self.best_score = np.inf
        self.iteration = 0
        self.eval_history = []
        
        # Performance tracking
        self.start_time = time.time()
        self.last_save_time = time.time()
        
        print(f"✓ GPU search initialized:")
        print(f"  Galaxies: {self.n_galaxies}")
        print(f"  Parameters: {self.n_params} ({', '.join(self.param_names)})")
        print(f"  GPU Memory allocated: {cp.get_default_memory_pool().used_bytes() / 1e9:.2f} GB")
    
    def _preload_galaxies_to_gpu(self):
        """
        Pre-load all galaxy data to GPU for maximum performance.
        
        Store as ragged arrays since galaxies have different n_points.
        """
        # Prepare galaxy data structures
        self.galaxy_data = []
        
        for gal in self.galaxies:
            # Upload to GPU
            data = {
                'r_kpc': cp.array(gal.r_kpc, dtype=cp.float32),
                'v_obs': cp.array(gal.v_obs, dtype=cp.float32),
                'v_err': cp.array(gal.v_err, dtype=cp.float32),
                'v_gas': cp.array(gal.v_gas, dtype=cp.float32),
                'v_disk': cp.array(gal.v_disk, dtype=cp.float32),
                'v_bulge': cp.array(gal.v_bulge, dtype=cp.float32),
                'bulge_frac': cp.array(gal.bulge_frac, dtype=cp.float32),
                'n_points': len(gal.r_kpc),
                'name': gal.name,
                'type_group': gal.type_group
            }
            self.galaxy_data.append(data)
    
    def evaluate_params_vectorized(self, params_array: cp.ndarray) -> cp.ndarray:
        """
        Evaluate parameter sets across ALL galaxies on GPU.
        
        Args:
            params_array: [N_param_sets, N_params] parameter combinations
        
        Returns:
            scores: [N_param_sets] median APE across galaxies
        """
        n_param_sets = params_array.shape[0]
        
        # Allocate results [N_param_sets, N_galaxies]
        apes_all = cp.zeros((n_param_sets, self.n_galaxies), dtype=cp.float32)
        
        # Evaluate each galaxy (can't fully vectorize due to different n_points)
        for gal_idx, gal_data in enumerate(self.galaxy_data):
            # Compute v_bar
            v_bar_sq = gal_data['v_gas']**2 + gal_data['v_disk']**2 + gal_data['v_bulge']**2
            v_bar = cp.sqrt(cp.maximum(v_bar_sq, 1e-10))
            
            # For each parameter set, compute predictions
            # Shape: [N_param_sets, N_radii]
            apes_gal = self._compute_predictions_vectorized(
                params_array, gal_data, v_bar
            )
            
            # Store median APE for this galaxy across param sets
            apes_all[:, gal_idx] = apes_gal
        
        # Return median APE across ALL galaxies for each param set
        return cp.median(apes_all, axis=1)
    
    def _compute_predictions_vectorized(self, params: cp.ndarray, 
                                       gal_data: Dict, v_bar: cp.ndarray) -> cp.ndarray:
        """
        Compute predictions for all parameter sets for ONE galaxy.
        
        Uses analytical profile approach (no particles).
        
        Args:
            params: [N_param_sets, N_params] 
            gal_data: Galaxy data dict
            v_bar: [N_radii] baryonic velocity
        
        Returns:
            apes: [N_param_sets] APE for this galaxy
        """
        n_param_sets = params.shape[0]
        n_radii = gal_data['n_points']
        
        # Extract parameters (assuming order: eta, ring_amp, M_max, bulge_gate_power)
        # Broadcast to [N_param_sets, N_radii]
        eta = params[:, 0:1]  # [N_param_sets, 1]
        ring_amp = params[:, 1:2] if params.shape[1] > 1 else cp.zeros((n_param_sets, 1))
        M_max = params[:, 2:3] if params.shape[1] > 2 else cp.ones((n_param_sets, 1)) * 4.0
        bulge_gate_power = params[:, 3:4] if params.shape[1] > 3 else cp.ones((n_param_sets, 1)) * 2.0
        
        # Radii: [N_radii]
        r = gal_data['r_kpc']
        
        # Compute many-path multiplier M(r) for each param set
        # Using simplified radial-only formula for speed
        # M(r) = eta * f(r) * (1 + ring_term)
        
        # Distance growth with saturation
        R0 = 5.0  # kpc, fixed
        R1 = 70.0  # kpc, fixed
        p = 2.0
        q = 3.5
        
        # Broadcast r to [N_param_sets, N_radii]
        r_broadcast = r[None, :]  # [1, N_radii]
        
        # Gate (turn on at galactic scales)
        R_gate = 0.5
        p_gate = 4.0
        gate = 1.0 - cp.exp(-(r_broadcast / R_gate)**p_gate)
        
        # Growth with saturation
        f_d = (r_broadcast / R0)**p / (1.0 + (r_broadcast / R1)**q)
        
        # Ring winding term (simplified - no pairwise geometry)
        lambda_ring = 42.0
        x = (2.0 * cp.pi * r_broadcast) / lambda_ring
        ex = cp.exp(-x)
        ring_term_base = ring_amp * (ex / cp.maximum(1e-20, 1.0 - ex))
        
        # Bulge gating
        bulge_frac = gal_data['bulge_frac'][None, :]  # [1, N_radii]
        bulge_gate = (1.0 - cp.minimum(bulge_frac, 1.0))**bulge_gate_power
        ring_term = ring_term_base * bulge_gate
        
        # Final multiplier M
        M = eta * gate * f_d * (1.0 + ring_term)
        M = cp.minimum(M, M_max)
        
        # Predicted velocity: v_pred = sqrt(v_bar^2 * (1 + M))
        v_bar_broadcast = v_bar[None, :]  # [1, N_radii]
        v_obs = gal_data['v_obs'][None, :]  # [1, N_radii]
        
        v_pred_sq = v_bar_broadcast**2 * (1.0 + M)
        v_pred = cp.sqrt(cp.maximum(v_pred_sq, 0.0))
        
        # Compute APE for each param set
        # APE = mean(|v_pred - v_obs| / v_obs) * 100
        mask = v_obs > 0  # [1, N_radii]
        
        ape = cp.abs(v_pred - v_obs) / cp.maximum(v_obs, 1.0) * 100.0
        ape = cp.where(mask, ape, 0.0)
        
        # Mean APE across radii for each param set
        apes = cp.mean(ape, axis=1)  # [N_param_sets]
        
        return apes
    
    def sample_params(self, n_samples: int, strategy: str = 'random') -> cp.ndarray:
        """
        Sample parameter combinations.
        
        Args:
            n_samples: Number of parameter sets to generate
            strategy: 'random', 'grid', 'sobol', or 'adaptive'
        
        Returns:
            params: [N_samples, N_params] on GPU
        """
        if strategy == 'random':
            # Uniform random sampling
            params = cp.random.uniform(0, 1, (n_samples, self.n_params), dtype=cp.float32)
            
        elif strategy == 'grid':
            # Grid sampling (coarse)
            n_per_dim = int(np.ceil(n_samples ** (1.0 / self.n_params)))
            grids = [cp.linspace(0, 1, n_per_dim, dtype=cp.float32) for _ in range(self.n_params)]
            # Create meshgrid on GPU
            mesh = cp.meshgrid(*grids, indexing='ij')
            params = cp.stack([g.ravel() for g in mesh], axis=1)[:n_samples]
            
        elif strategy == 'sobol':
            # Sobol sequence (quasi-random, better coverage)
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.n_params, scramble=True)
            params_cpu = sampler.random(n_samples).astype(np.float32)
            params = cp.array(params_cpu)
            
        elif strategy == 'adaptive':
            # Sample around best parameters with some exploration
            if self.best_params is None:
                # Fall back to random
                params = cp.random.uniform(0, 1, (n_samples, self.n_params), dtype=cp.float32)
            else:
                # Sample from Gaussian around best
                best = cp.array([self.best_params[name] for name in self.param_names], dtype=cp.float32)
                # Normalize to [0, 1]
                best_norm = cp.zeros(self.n_params, dtype=cp.float32)
                for i, name in enumerate(self.param_names):
                    min_val, max_val = self.param_bounds[name]
                    best_norm[i] = (best[i] - min_val) / (max_val - min_val)
                
                # Sample with std=0.1 (10% of range) and clip to [0, 1]
                params = cp.random.normal(best_norm[None, :], 0.1, (n_samples, self.n_params)).astype(cp.float32)
                params = cp.clip(params, 0.0, 1.0)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Denormalize from [0, 1] to actual param ranges
        params_denorm = cp.zeros_like(params)
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[name]
            params_denorm[:, i] = params[:, i] * (max_val - min_val) + min_val
        
        return params_denorm
    
    def search_iteration(self, batch_size: int, strategy: str = 'random') -> Dict:
        """
        Single search iteration evaluating batch_size parameter sets.
        
        Returns:
            metrics: Dict with iteration stats
        """
        iter_start = time.time()
        
        # Sample parameters
        params = self.sample_params(batch_size, strategy)
        
        # Evaluate on GPU
        scores = self.evaluate_params_vectorized(params)
        
        # Find best in this batch
        best_idx = cp.argmin(scores)
        best_score_batch = float(scores[best_idx])
        best_params_batch = params[best_idx].get()
        
        # Update global best
        if best_score_batch < self.best_score:
            self.best_score = best_score_batch
            self.best_params = {name: float(best_params_batch[i]) 
                               for i, name in enumerate(self.param_names)}
            print(f"\n{'='*60}")
            print(f"🎯 NEW BEST at iteration {self.iteration}!")
            print(f"   Median APE: {self.best_score:.2f}%")
            for name, val in self.best_params.items():
                print(f"   {name:20s} = {val:.4f}")
            print(f"{'='*60}\n")
        
        iter_time = time.time() - iter_start
        throughput = batch_size / iter_time
        
        self.iteration += 1
        
        return {
            'iteration': self.iteration,
            'batch_size': batch_size,
            'best_score_batch': best_score_batch,
            'best_score_global': self.best_score,
            'throughput': throughput,
            'iter_time': iter_time,
            'strategy': strategy
        }
    
    def run_exhaustive_search(self, n_iterations: int, batch_size: int = 10000,
                             strategies: List[str] = None, save_every: int = 60):
        """
        Run exhaustive parameter search.
        
        Args:
            n_iterations: Number of iterations to run
            batch_size: Parameter sets per iteration
            strategies: List of sampling strategies to alternate
            save_every: Save checkpoint every N seconds
        """
        if strategies is None:
            strategies = ['sobol', 'random', 'adaptive', 'grid']
        
        print(f"\n{'='*80}")
        print("STARTING EXHAUSTIVE GPU PARAMETER SEARCH")
        print(f"{'='*80}")
        print(f"Target iterations: {n_iterations:,}")
        print(f"Batch size: {batch_size:,} param sets/iteration")
        print(f"Total evaluations: {n_iterations * batch_size:,}")
        print(f"Strategies: {', '.join(strategies)}")
        print(f"{'='*80}\n")
        
        try:
            for i in range(n_iterations):
                # Alternate strategies
                strategy = strategies[i % len(strategies)]
                
                # Run iteration
                metrics = self.search_iteration(batch_size, strategy)
                
                # Print progress
                if i % 10 == 0 or metrics['best_score_batch'] < self.best_score:
                    elapsed = time.time() - self.start_time
                    total_evals = self.iteration * batch_size
                    evals_per_sec = total_evals / elapsed
                    
                    print(f"[{i:5d}/{n_iterations}] "
                          f"Best={self.best_score:.2f}%, "
                          f"Batch={metrics['best_score_batch']:.2f}%, "
                          f"Strategy={strategy:8s}, "
                          f"Throughput={metrics['throughput']:,.0f} param/s, "
                          f"Total={evals_per_sec:,.0f} eval/s")
                
                # Save checkpoint periodically
                if time.time() - self.last_save_time > save_every:
                    self._save_checkpoint()
                    self.last_save_time = time.time()
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Search interrupted by user")
            self._save_checkpoint()
    
    def _save_checkpoint(self):
        """Save current best parameters to checkpoint file."""
        checkpoint = {
            'iteration': self.iteration,
            'best_score': float(self.best_score),
            'best_params': self.best_params,
            'elapsed_time': time.time() - self.start_time,
            'total_evaluations': self.iteration * 10000  # Assuming batch_size=10000
        }
        
        checkpoint_path = Path('results/search_checkpoint.json')
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"\n💾 Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="GPU exhaustive parameter search")
    parser.add_argument('--sparc_dir', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/external_data/Rotmod_LTG')
    parser.add_argument('--master_file', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/data/SPARC_Lelli2016c.mrt')
    
    # Search parameters
    parser.add_argument('--mode', type=str, choices=['test', 'exhaustive'], default='exhaustive',
                       help='Test mode (1K iters) or exhaustive (1M+ iters)')
    parser.add_argument('--n_iterations', type=int, default=None,
                       help='Number of iterations (overrides mode default)')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Parameter sets per iteration')
    
    # Parameter bounds
    parser.add_argument('--eta_range', type=str, default='0.05,0.5',
                       help='Min,max for eta')
    parser.add_argument('--ring_amp_range', type=str, default='0.0,0.15',
                       help='Min,max for ring_amp')
    parser.add_argument('--M_max_range', type=str, default='2.0,5.0',
                       help='Min,max for M_max')
    parser.add_argument('--bulge_gate_power_range', type=str, default='0.5,4.0',
                       help='Min,max for bulge_gate_power')
    
    # Output
    parser.add_argument('--output', type=str, default='results/exhaustive_search.json')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    cp.random.seed(args.seed)
    
    # Determine n_iterations
    if args.n_iterations is None:
        n_iterations = 1000 if args.mode == 'test' else 100000
    else:
        n_iterations = args.n_iterations
    
    print("="*80)
    print("GPU EXHAUSTIVE PARAMETER SEARCH")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Iterations: {n_iterations:,}")
    print(f"Batch size: {args.batch_size:,}")
    print(f"Total evaluations: {n_iterations * args.batch_size:,}")
    
    # Check GPU
    try:
        gpu_info = cp.cuda.Device(0)
        print(f"\n✓ GPU detected: {gpu_info}")
        print(f"  Compute capability: {gpu_info.compute_capability}")
        print(f"  Memory: {gpu_info.mem_info[1] / 1e9:.1f} GB total")
    except Exception as e:
        print(f"\n✗ GPU check failed: {e}")
        sys.exit(1)
    
    # Load SPARC data
    print(f"\nLoading SPARC data from {args.sparc_dir}...")
    master_info = load_sparc_master_table(Path(args.master_file))
    
    sparc_dir = Path(args.sparc_dir)
    galaxy_files = list(sparc_dir.glob('*_rotmod.dat'))
    
    galaxies = []
    for gfile in galaxy_files:
        try:
            gal = load_sparc_galaxy(gfile, master_info)
            galaxies.append(gal)
        except Exception:
            continue
    
    print(f"✓ Loaded {len(galaxies)} galaxies")
    
    # Type distribution
    type_counts = {}
    for g in galaxies:
        type_counts[g.type_group] = type_counts.get(g.type_group, 0) + 1
    print(f"  Type distribution: {type_counts}")
    
    # Define parameter bounds
    param_bounds = {
        'eta': tuple(float(x) for x in args.eta_range.split(',')),
        'ring_amp': tuple(float(x) for x in args.ring_amp_range.split(',')),
        'M_max': tuple(float(x) for x in args.M_max_range.split(',')),
        'bulge_gate_power': tuple(float(x) for x in args.bulge_gate_power_range.split(','))
    }
    
    print(f"\nParameter bounds:")
    for name, (min_val, max_val) in param_bounds.items():
        print(f"  {name:20s}: [{min_val:.3f}, {max_val:.3f}]")
    
    # Create searcher
    searcher = GPUParameterSearch(galaxies, param_bounds)
    
    # Run search
    searcher.run_exhaustive_search(
        n_iterations=n_iterations,
        batch_size=args.batch_size,
        strategies=['sobol', 'random', 'adaptive'],
        save_every=60
    )
    
    # Save final results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'best_params': searcher.best_params,
        'best_score': float(searcher.best_score),
        'total_iterations': searcher.iteration,
        'total_evaluations': searcher.iteration * args.batch_size,
        'elapsed_time': time.time() - searcher.start_time,
        'n_galaxies': len(galaxies)
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Best median APE: {searcher.best_score:.2f}%")
    print(f"Total evaluations: {results['total_evaluations']:,}")
    print(f"Results saved to: {output_path}")
    print(f"\nBest parameters:")
    for name, val in searcher.best_params.items():
        print(f"  {name:20s} = {val:.4f}")


if __name__ == '__main__':
    main()
