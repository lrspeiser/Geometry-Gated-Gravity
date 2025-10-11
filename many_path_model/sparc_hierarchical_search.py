#!/usr/bin/env python3
"""
Hierarchical Multi-Stage Parameter Search for SPARC Galaxies

Strategy: Wide exploration → Branch identification → Local refinement

Stage 1: EXPLORATION (Wide bounds, coarse sampling)
    - Test extremely wide parameter ranges
    - Identify top N "promising regions" 
    - Use diverse sampling (Sobol, Latin Hypercube)
    
Stage 2: BRANCHING (Multi-region parallel search)
    - Spawn independent searches around top N regions
    - Each branch explores ±50% of its region
    - Prune poor-performing branches
    
Stage 3: REFINEMENT (Narrow optimization)
    - Fine-grained search around best from Stage 2
    - Adaptive mesh refinement
    - Gradient-free optimization (CMA-ES)

Usage:
    # Full hierarchical search
    python sparc_hierarchical_search.py --stages all
    
    # Just exploration phase with custom bounds
    python sparc_hierarchical_search.py --stages explore \
        --eta_range 0.001,2.0 --ring_amp_range 0.0,0.5
"""

import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import pickle

# GPU
try:
    import cupy as cp
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


@dataclass
class SearchRegion:
    """Defines a parameter search region"""
    center: Dict[str, float]
    widths: Dict[str, float]  # ± width around center
    score: float = np.inf
    n_evals: int = 0
    
    def sample_uniform(self, n_samples: int, param_names: List[str]) -> np.ndarray:
        """Sample uniformly within this region"""
        samples = np.zeros((n_samples, len(param_names)))
        for i, name in enumerate(param_names):
            low = max(0.0, self.center[name] - self.widths[name])
            high = self.center[name] + self.widths[name]
            samples[:, i] = np.random.uniform(low, high, n_samples)
        return samples


class HierarchicalSearch:
    """
    Multi-stage hierarchical parameter search with adaptive branching.
    """
    
    def __init__(self, galaxies: List[SPARCGalaxy], param_bounds: Dict):
        self.galaxies = galaxies
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
        self.n_galaxies = len(galaxies)
        
        # Pre-load to GPU
        print(f"\nPre-loading {self.n_galaxies} galaxies to GPU...")
        self._preload_galaxies_to_gpu()
        
        # Search state
        self.all_results = []  # (params, score) history
        self.best_params = None
        self.best_score = np.inf
        self.stage = 0
        
        print(f"✓ Hierarchical search initialized")
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
                'n_points': len(gal.r_kpc),
                'name': gal.name,
                'type_group': gal.type_group
            }
            self.galaxy_data.append(data)
    
    def evaluate_params_vectorized(self, params_array: cp.ndarray) -> cp.ndarray:
        """Evaluate parameter sets across all galaxies on GPU"""
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
        """Compute predictions for all parameter sets for one galaxy"""
        n_param_sets = params.shape[0]
        
        # Extract parameters
        eta = params[:, 0:1]
        ring_amp = params[:, 1:2] if params.shape[1] > 1 else cp.zeros((n_param_sets, 1))
        M_max = params[:, 2:3] if params.shape[1] > 2 else cp.ones((n_param_sets, 1)) * 4.0
        bulge_gate_power = params[:, 3:4] if params.shape[1] > 3 else cp.ones((n_param_sets, 1)) * 2.0
        
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
        
        # Ring winding
        lambda_ring = 42.0
        x = (2.0 * cp.pi * r_broadcast) / lambda_ring
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
    
    def stage1_exploration(self, n_samples: int = 100000, 
                          wide_factor: float = 3.0) -> List[SearchRegion]:
        """
        Stage 1: Wide exploration to find promising regions
        
        Args:
            n_samples: Total parameter sets to evaluate
            wide_factor: How much to expand bounds (3.0 = 3x wider)
        
        Returns:
            List of top N promising regions
        """
        print("\n" + "="*80)
        print("STAGE 1: WIDE EXPLORATION")
        print("="*80)
        
        # Expand parameter bounds
        wide_bounds = {}
        for name, (min_val, max_val) in self.param_bounds.items():
            center = (min_val + max_val) / 2
            width = (max_val - min_val) / 2 * wide_factor
            wide_bounds[name] = (max(0.0, center - width), center + width)
        
        print("\nExpanded parameter ranges:")
        for name, (min_val, max_val) in wide_bounds.items():
            print(f"  {name:20s}: [{min_val:.4f}, {max_val:.4f}]")
        
        # Multiple sampling strategies for diverse exploration
        strategies = [
            ('sobol', n_samples // 3),
            ('latin_hypercube', n_samples // 3),
            ('random', n_samples // 3)
        ]
        
        all_params = []
        all_scores = []
        
        for strategy_name, n_strat in strategies:
            print(f"\n  Sampling {n_strat:,} points using {strategy_name}...")
            
            if strategy_name == 'sobol':
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=self.n_params, scramble=True)
                samples_norm = sampler.random(n_strat).astype(np.float32)
            elif strategy_name == 'latin_hypercube':
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=self.n_params)
                samples_norm = sampler.random(n_strat).astype(np.float32)
            else:  # random
                samples_norm = np.random.uniform(0, 1, (n_strat, self.n_params)).astype(np.float32)
            
            # Denormalize to wide bounds
            samples = np.zeros_like(samples_norm)
            for i, name in enumerate(self.param_names):
                min_val, max_val = wide_bounds[name]
                samples[:, i] = samples_norm[:, i] * (max_val - min_val) + min_val
            
            # Evaluate on GPU in batches
            batch_size = 10000
            scores_list = []
            
            for batch_start in range(0, len(samples), batch_size):
                batch_end = min(batch_start + batch_size, len(samples))
                batch = samples[batch_start:batch_end]
                
                params_gpu = cp.array(batch, dtype=cp.float32)
                scores_gpu = self.evaluate_params_vectorized(params_gpu)
                scores_batch = scores_gpu.get()
                scores_list.append(scores_batch)
                
                if (batch_end) % 50000 == 0:
                    print(f"    Evaluated {batch_end:,}/{n_strat:,}...")
            
            scores = np.concatenate(scores_list)
            all_params.append(samples)
            all_scores.append(scores)
        
        # Combine all results
        all_params = np.vstack(all_params)
        all_scores = np.concatenate(all_scores)
        
        # Store in history
        for params, score in zip(all_params, all_scores):
            self.all_results.append((params, float(score)))
            if score < self.best_score:
                self.best_score = float(score)
                self.best_params = {name: float(params[i]) 
                                   for i, name in enumerate(self.param_names)}
        
        print(f"\n✓ Exploration complete: {len(all_params):,} evaluations")
        print(f"  Best score found: {self.best_score:.2f}%")
        print(f"  Best params: {self.best_params}")
        
        # Identify top regions using clustering
        n_regions = 10  # Number of promising regions to identify
        top_indices = np.argsort(all_scores)[:n_samples // 10]  # Top 10%
        top_params = all_params[top_indices]
        top_scores = all_scores[top_indices]
        
        # K-means clustering to find diverse regions
        from sklearn.cluster import KMeans
        print(f"\n  Clustering top 10% into {n_regions} regions...")
        
        # Normalize params for clustering
        params_norm = np.zeros_like(top_params)
        for i, name in enumerate(self.param_names):
            min_val, max_val = wide_bounds[name]
            params_norm[:, i] = (top_params[:, i] - min_val) / (max_val - min_val)
        
        kmeans = KMeans(n_clusters=n_regions, random_state=42, n_init=10)
        labels = kmeans.fit_predict(params_norm)
        
        # Create SearchRegion for each cluster
        regions = []
        for cluster_id in range(n_regions):
            mask = labels == cluster_id
            cluster_params = top_params[mask]
            cluster_scores = top_scores[mask]
            
            # Region center = best in cluster
            best_idx = np.argmin(cluster_scores)
            center_params = cluster_params[best_idx]
            
            center = {name: float(center_params[i]) 
                     for i, name in enumerate(self.param_names)}
            
            # Region width = std of cluster
            widths = {}
            for i, name in enumerate(self.param_names):
                std = np.std(cluster_params[:, i])
                widths[name] = max(std, 0.01)  # Minimum width
            
            region = SearchRegion(
                center=center,
                widths=widths,
                score=float(cluster_scores[best_idx]),
                n_evals=len(cluster_params)
            )
            regions.append(region)
        
        # Sort by score
        regions.sort(key=lambda r: r.score)
        
        print(f"\n  Top {n_regions} promising regions identified:")
        for i, region in enumerate(regions[:5]):  # Show top 5
            print(f"    Region {i+1}: score={region.score:.2f}%, n_evals={region.n_evals}")
            for name in self.param_names:
                print(f"      {name}: {region.center[name]:.4f} ± {region.widths[name]:.4f}")
        
        return regions
    
    def stage2_branching(self, regions: List[SearchRegion], 
                        n_samples_per_region: int = 50000) -> List[SearchRegion]:
        """
        Stage 2: Parallel exploration of multiple promising regions
        
        Args:
            regions: List of regions from stage 1
            n_samples_per_region: Samples to evaluate per region
        
        Returns:
            Refined list of best regions
        """
        print("\n" + "="*80)
        print("STAGE 2: MULTI-REGION BRANCHING")
        print("="*80)
        
        refined_regions = []
        
        for i, region in enumerate(regions[:10]):  # Explore top 10 regions
            print(f"\n  Branch {i+1}/10: Exploring region (score={region.score:.2f}%)...")
            
            # Sample within this region
            samples = region.sample_uniform(n_samples_per_region, self.param_names)
            
            # Evaluate on GPU in batches
            batch_size = 10000
            scores_list = []
            
            for batch_start in range(0, len(samples), batch_size):
                batch_end = min(batch_start + batch_size, len(samples))
                batch = samples[batch_start:batch_end]
                
                params_gpu = cp.array(batch, dtype=cp.float32)
                scores_gpu = self.evaluate_params_vectorized(params_gpu)
                scores_batch = scores_gpu.get()
                scores_list.append(scores_batch)
            
            scores = np.concatenate(scores_list)
            
            # Update history
            for params, score in zip(samples, scores):
                self.all_results.append((params, float(score)))
                if score < self.best_score:
                    self.best_score = float(score)
                    self.best_params = {name: float(params[i]) 
                                       for i, name in enumerate(self.param_names)}
                    print(f"\n    🎯 NEW BEST: {self.best_score:.2f}%")
                    for name, val in self.best_params.items():
                        print(f"       {name}: {val:.4f}")
            
            # Find best in this branch
            best_idx = np.argmin(scores)
            best_params = samples[best_idx]
            
            # Create refined region around best
            new_center = {name: float(best_params[i]) 
                         for i, name in enumerate(self.param_names)}
            
            # Narrow the widths by 50%
            new_widths = {name: region.widths[name] * 0.5 
                         for name in self.param_names}
            
            refined_region = SearchRegion(
                center=new_center,
                widths=new_widths,
                score=float(scores[best_idx]),
                n_evals=n_samples_per_region
            )
            refined_regions.append(refined_region)
            
            print(f"    Best in branch: {refined_region.score:.2f}%")
        
        # Sort and return top regions
        refined_regions.sort(key=lambda r: r.score)
        
        print(f"\n✓ Branching complete")
        print(f"  Best overall: {self.best_score:.2f}%")
        
        return refined_regions
    
    def stage3_refinement(self, region: SearchRegion, 
                         n_iterations: int = 100) -> Dict:
        """
        Stage 3: Fine-grained local optimization using CMA-ES
        
        Args:
            region: Best region from stage 2
            n_iterations: Number of CMA-ES iterations
        
        Returns:
            Final optimized parameters
        """
        print("\n" + "="*80)
        print("STAGE 3: LOCAL REFINEMENT (CMA-ES)")
        print("="*80)
        
        try:
            import cma
        except ImportError:
            print("⚠️  CMA-ES not available (pip install cma). Using adaptive grid instead.")
            return self._fallback_refinement(region, n_samples=500000)
        
        # Objective function (CPU-based for CMA-ES)
        def objective(x):
            # CMA-ES handles bounds internally - don't clip!
            # Just evaluate the parameters as-is
            params_gpu = cp.array([x], dtype=cp.float32)
            score = float(self.evaluate_params_vectorized(params_gpu)[0])
            
            # Update best
            if score < self.best_score:
                self.best_score = score
                self.best_params = {name: float(x[i]) 
                                   for i, name in enumerate(self.param_names)}
                print(f"  🎯 NEW BEST: {self.best_score:.2f}%")
                for name, val in self.best_params.items():
                    print(f"     {name}: {val:.4f}")
            
            return score
        
        # Initial guess and sigma
        x0 = [region.center[name] for name in self.param_names]
        # Use smaller sigma to avoid bounds issues
        sigma0 = min(0.1, np.mean([region.widths[name] for name in self.param_names]) * 0.5)
        
        # Bounds: Use region-based bounds, NOT original narrow param_bounds
        # This allows refinement around parameters found in wide search
        bounds = [
            [max(0.0, region.center[name] - region.widths[name] * 2) 
             for name in self.param_names],
            [region.center[name] + region.widths[name] * 2 
             for name in self.param_names]
        ]
        
        print(f"\n  Starting CMA-ES from: {x0}")
        print(f"  Initial sigma: {sigma0:.4f}")
        print(f"  Search bounds:")
        for i, name in enumerate(self.param_names):
            print(f"    {name:20s}: [{bounds[0][i]:.4f}, {bounds[1][i]:.4f}]")
        
        # Run CMA-ES with conservative settings
        try:
            es = cma.CMAEvolutionStrategy(x0, sigma0, {
                'bounds': bounds,
                'popsize': 30,  # Smaller population
                'maxiter': n_iterations,
                'verb_disp': 20,  # Less verbose
                'tolx': 1e-6,  # Tighter convergence
                'tolfun': 1e-4
            })
            
            es.optimize(objective)
            
            print(f"\n✓ CMA-ES refinement complete")
            print(f"  Final best: {self.best_score:.2f}%")
        
        except Exception as e:
            print(f"\n⚠️  CMA-ES failed: {e}")
            print("   Falling back to adaptive grid search...")
            return self._fallback_refinement(region, n_samples=500000)
        
        return self.best_params
        
        print(f"\n✓ Refinement complete")
        print(f"  Final best: {self.best_score:.2f}%")
        
        return self.best_params
    
    def _fallback_refinement(self, region: SearchRegion, n_samples: int) -> Dict:
        """Fallback refinement using dense adaptive grid"""
        print(f"  Using dense adaptive grid ({n_samples:,} samples)...")
        
        # Initialize best from region if not already set
        if self.best_score == np.inf:
            self.best_score = region.score
            self.best_params = region.center.copy()
        
        samples = region.sample_uniform(n_samples, self.param_names)
        
        # Evaluate in batches
        batch_size = 10000
        scores_list = []
        n_improved = 0
        
        for batch_start in range(0, len(samples), batch_size):
            batch_end = min(batch_start + batch_size, len(samples))
            batch = samples[batch_start:batch_end]
            
            params_gpu = cp.array(batch, dtype=cp.float32)
            scores_gpu = self.evaluate_params_vectorized(params_gpu)
            scores_batch = scores_gpu.get()
            scores_list.append(scores_batch)
            
            # Check for improvements in this batch
            batch_best_idx = np.argmin(scores_batch)
            if scores_batch[batch_best_idx] < self.best_score:
                self.best_score = float(scores_batch[batch_best_idx])
                global_idx = batch_start + batch_best_idx
                self.best_params = {name: float(samples[global_idx, i]) 
                                   for i, name in enumerate(self.param_names)}
                n_improved += 1
                print(f"    🎯 NEW BEST: {self.best_score:.2f}%")
            
            if (batch_end) % 100000 == 0:
                print(f"    Evaluated {batch_end:,}/{n_samples:,}...")
        
        scores = np.concatenate(scores_list)
        print(f"\n  ✓ Fallback refinement complete: {n_improved} improvements found")
        
        return self.best_params
    
    def run_full_search(self, save_dir: Path):
        """Run all three stages sequentially"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # Stage 1
        regions = self.stage1_exploration(n_samples=100000, wide_factor=5.0)
        self._save_checkpoint(save_dir / 'stage1_regions.pkl', regions)
        
        # Stage 2
        refined_regions = self.stage2_branching(regions, n_samples_per_region=50000)
        self._save_checkpoint(save_dir / 'stage2_regions.pkl', refined_regions)
        
        # Stage 3
        best_region = refined_regions[0]
        final_params = self.stage3_refinement(best_region, n_iterations=100)
        
        elapsed = time.time() - start_time
        
        # Save final results
        results = {
            'best_params': final_params,
            'best_score': float(self.best_score),
            'elapsed_time': elapsed,
            'total_evaluations': len(self.all_results),
            'stages_completed': 3
        }
        
        with open(save_dir / 'final_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*80)
        print("HIERARCHICAL SEARCH COMPLETE")
        print("="*80)
        print(f"Best score: {self.best_score:.2f}%")
        print(f"Total evaluations: {len(self.all_results):,}")
        print(f"Elapsed time: {elapsed/60:.1f} minutes")
        print(f"\nBest parameters:")
        for name, val in final_params.items():
            print(f"  {name:20s} = {val:.6f}")
        
        return results
    
    def _save_checkpoint(self, path: Path, data):
        """Save checkpoint"""
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  💾 Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical parameter search")
    parser.add_argument('--sparc_dir', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/external_data/Rotmod_LTG')
    parser.add_argument('--master_file', type=str,
                       default='C:/Users/henry/Documents/GitHub/DensityDependentMetricModel/data/SPARC_Lelli2016c.mrt')
    
    # Search control
    parser.add_argument('--stages', type=str, default='all',
                       choices=['explore', 'branch', 'refine', 'all'],
                       help='Which stages to run')
    parser.add_argument('--wide_factor', type=float, default=5.0,
                       help='How much to expand bounds in exploration (5.0 = 5x wider)')
    
    # Parameter bounds (initial narrow bounds)
    parser.add_argument('--eta_range', type=str, default='0.01,1.0')
    parser.add_argument('--ring_amp_range', type=str, default='0.0,0.5')
    parser.add_argument('--M_max_range', type=str, default='1.0,10.0')
    parser.add_argument('--bulge_gate_power_range', type=str, default='0.5,6.0')
    
    parser.add_argument('--output_dir', type=str, default='results/hierarchical')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    cp.random.seed(args.seed)
    
    print("="*80)
    print("HIERARCHICAL MULTI-STAGE PARAMETER SEARCH")
    print("="*80)
    print(f"Wide expansion factor: {args.wide_factor}x")
    
    # Load SPARC
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
    
    print(f"\n✓ Loaded {len(galaxies)} galaxies")
    
    # Parameter bounds
    param_bounds = {
        'eta': tuple(float(x) for x in args.eta_range.split(',')),
        'ring_amp': tuple(float(x) for x in args.ring_amp_range.split(',')),
        'M_max': tuple(float(x) for x in args.M_max_range.split(',')),
        'bulge_gate_power': tuple(float(x) for x in args.bulge_gate_power_range.split(','))
    }
    
    print(f"\nInitial parameter bounds:")
    for name, (min_val, max_val) in param_bounds.items():
        print(f"  {name:20s}: [{min_val:.4f}, {max_val:.4f}]")
    
    # Create searcher
    searcher = HierarchicalSearch(galaxies, param_bounds)
    
    # Run search based on stage selection
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.stages == 'all':
        results = searcher.run_full_search(output_dir)
    
    elif args.stages == 'explore':
        print("\n⚡ Running EXPLORATION stage only")
        start_time = time.time()
        regions = searcher.stage1_exploration(n_samples=100000, wide_factor=args.wide_factor)
        searcher._save_checkpoint(output_dir / 'stage1_regions.pkl', regions)
        elapsed = time.time() - start_time
        
        results = {
            'best_params': searcher.best_params,
            'best_score': float(searcher.best_score),
            'elapsed_time': elapsed,
            'total_evaluations': len(searcher.all_results),
            'stages_completed': 1,
            'n_regions': len(regions)
        }
        with open(output_dir / 'stage1_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Stage 1 complete: {len(searcher.all_results):,} evaluations in {elapsed/60:.1f} min")
    
    elif args.stages == 'branch':
        print("\n⚡ Running BRANCHING stage only (requires stage1_regions.pkl)")
        # Load regions from Stage 1
        regions_file = output_dir / 'stage1_regions.pkl'
        if not regions_file.exists():
            print(f"\n✗ ERROR: {regions_file} not found. Run --stages explore first.")
            sys.exit(1)
        
        with open(regions_file, 'rb') as f:
            regions = pickle.load(f)
        print(f"  Loaded {len(regions)} regions from Stage 1")
        
        start_time = time.time()
        refined_regions = searcher.stage2_branching(regions, n_samples_per_region=50000)
        searcher._save_checkpoint(output_dir / 'stage2_regions.pkl', refined_regions)
        elapsed = time.time() - start_time
        
        results = {
            'best_params': searcher.best_params,
            'best_score': float(searcher.best_score),
            'elapsed_time': elapsed,
            'total_evaluations': len(searcher.all_results),
            'stages_completed': 2
        }
        with open(output_dir / 'stage2_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Stage 2 complete: {len(searcher.all_results):,} evaluations in {elapsed/60:.1f} min")
    
    elif args.stages == 'refine':
        print("\n⚡ Running REFINEMENT stage only (requires stage2_regions.pkl)")
        # Load regions from Stage 2
        regions_file = output_dir / 'stage2_regions.pkl'
        if not regions_file.exists():
            print(f"\n✗ ERROR: {regions_file} not found. Run --stages branch first.")
            sys.exit(1)
        
        with open(regions_file, 'rb') as f:
            refined_regions = pickle.load(f)
        print(f"  Loaded {len(refined_regions)} refined regions from Stage 2")
        print(f"  Best from Stage 2: {refined_regions[0].score:.2f}%")
        
        start_time = time.time()
        best_region = refined_regions[0]
        final_params = searcher.stage3_refinement(best_region, n_iterations=100)
        elapsed = time.time() - start_time
        
        results = {
            'best_params': final_params,
            'best_score': float(searcher.best_score),
            'elapsed_time': elapsed,
            'total_evaluations': len(searcher.all_results),
            'stages_completed': 3
        }
        with open(output_dir / 'stage3_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Stage 3 complete: {len(searcher.all_results):,} evaluations in {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
