#!/usr/bin/env python3
"""
GPU-Optimized SPARC-Specific LogTail/G³ Parameter Search
=========================================================

This optimizer finds the best LogTail parameters specifically for SPARC galaxies,
independent of MW constraints. This tests if different parameter sets are needed
for different galaxy populations.

Key differences from MW optimization:
- Optimizes on all 175 SPARC galaxies simultaneously
- Different parameter bounds based on SPARC characteristics
- Weighted by galaxy quality and number of points
- Can explore wider parameter space
"""

import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
import argparse
import logging
from datetime import datetime

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[✓] GPU acceleration enabled (CuPy)")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("[!] Running on CPU (CuPy not available)")

# Physical constants
G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class SPARCOptimizer:
    """GPU-accelerated optimizer for SPARC galaxy rotation curves."""
    
    def __init__(self, data_path="../data", use_gpu=True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.data_path = Path(data_path)
        self.galaxies = {}
        self.optimization_history = []
        
        if self.use_gpu:
            # Print GPU info
            mempool = cp.get_default_memory_pool()
            logger.info(f"GPU Memory: {mempool.used_bytes()/1e9:.2f} GB used")
    
    def load_sparc_data(self, max_galaxies=None):
        """Load SPARC rotation curves."""
        logger.info("Loading SPARC data...")
        
        # Try multiple locations
        sparc_files = [
            self.data_path / "sparc_rotmod_ltg.parquet",
            self.data_path / "sparc_predictions_by_radius.csv",
            self.data_path / "sparc" / "sparc_rotmod_ltg.parquet"
        ]
        
        df = None
        for file in sparc_files:
            if file.exists():
                if file.suffix == '.parquet':
                    df = pd.read_parquet(file)
                else:
                    df = pd.read_csv(file)
                logger.info(f"Loaded from {file}")
                break
        
        if df is None:
            raise FileNotFoundError("No SPARC data file found")
        
        # Process galaxies
        galaxy_groups = df.groupby('galaxy' if 'galaxy' in df else 'Galaxy')
        
        count = 0
        total_points = 0
        for name, gdf in galaxy_groups:
            if max_galaxies and count >= max_galaxies:
                break
            
            # Extract columns
            r = gdf['R_kpc'].values if 'R_kpc' in gdf else gdf['r_kpc'].values
            v_obs = gdf['Vobs_kms'].values if 'Vobs_kms' in gdf else gdf['v_obs'].values
            v_err = gdf['errV'].values if 'errV' in gdf else np.full_like(v_obs, 5.0)
            
            # Component velocities
            v_gas = gdf['Vgas_kms'].values if 'Vgas_kms' in gdf else np.zeros_like(r)
            v_disk = gdf['Vdisk_kms'].values if 'Vdisk_kms' in gdf else np.zeros_like(r)
            v_bulge = gdf['Vbul_kms'].values if 'Vbul_kms' in gdf else np.zeros_like(r)
            
            # Total baryonic
            v_bar = np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)
            
            # Quality filter
            valid = (r > 0) & np.isfinite(v_obs) & np.isfinite(v_bar) & (v_obs > 0)
            if valid.sum() < 3:
                continue
            
            # Transfer to GPU if available
            if self.use_gpu:
                galaxy_data = {
                    'r': cp.asarray(r[valid]),
                    'v_obs': cp.asarray(v_obs[valid]),
                    'v_err': cp.asarray(v_err[valid]),
                    'v_bar': cp.asarray(v_bar[valid]),
                    'weight': len(r[valid]) / (np.median(v_err[valid]) + 1)  # Weight by quality
                }
            else:
                galaxy_data = {
                    'r': r[valid],
                    'v_obs': v_obs[valid],
                    'v_err': v_err[valid],
                    'v_bar': v_bar[valid],
                    'weight': len(r[valid]) / (np.median(v_err[valid]) + 1)
                }
            
            self.galaxies[name] = galaxy_data
            count += 1
            total_points += valid.sum()
        
        logger.info(f"Loaded {len(self.galaxies)} galaxies with {total_points} total points")
        return len(self.galaxies)
    
    def logtail_model(self, r, v_bar, v0, rc, r0, delta, gamma, beta, sigma_ref=100.0):
        """
        Compute LogTail rotation curve.
        
        Extended parameter space for SPARC:
        - gamma: radial profile power (can be different from MW)
        - beta: surface density coupling (may be stronger for SPARC)
        """
        # Smooth gate
        gate = 0.5 * (1.0 + self.xp.tanh((r - r0) / (delta + 1e-9)))
        
        # Radial profile - more flexible for SPARC
        f_r = self.xp.power(r / (r + rc + 1e-9), gamma)
        
        # Surface density effect (estimated from v_bar)
        sigma_proxy = v_bar / self.xp.sqrt(r + 1e-9)  # Simple proxy
        sigma_factor = self.xp.power(self.xp.maximum(sigma_proxy / sigma_ref, 0.1), beta)
        
        # LogTail acceleration
        g_tail = (v0**2 / (r + 1e-9)) * f_r * gate * sigma_factor
        
        # Total velocity
        g_bar = v_bar**2 / (r + 1e-9)
        v_total = self.xp.sqrt((g_bar + g_tail) * r)
        
        return v_total
    
    def compute_loss(self, params):
        """
        Compute total loss across all SPARC galaxies.
        
        Uses robust weighted median absolute percentage error.
        """
        v0, rc, r0, delta, gamma, beta = params
        
        total_loss = 0.0
        total_weight = 0.0
        
        for name, data in self.galaxies.items():
            # Predict velocities
            v_pred = self.logtail_model(
                data['r'], data['v_bar'], v0, rc, r0, delta, gamma, beta
            )
            
            # Compute weighted chi2
            chi2 = self.xp.sum(((v_pred - data['v_obs']) / data['v_err'])**2)
            chi2_reduced = chi2 / len(data['r'])
            
            # Add to total with galaxy weight
            total_loss += chi2_reduced * data['weight']
            total_weight += data['weight']
        
        # Return weighted average
        return float(total_loss / total_weight) if self.use_gpu else (total_loss / total_weight)
    
    def optimize(self, n_iterations=1000, population_size=64, 
                 bounds=None, seed=42):
        """
        Run CMA-ES optimization on GPU.
        
        SPARC-specific bounds allow exploring different regime than MW.
        """
        if bounds is None:
            # SPARC-optimized bounds (can be different from MW)
            bounds = {
                'v0': (50, 300),      # Wider range for diverse galaxy types
                'rc': (1, 50),        # Can be smaller for dwarf galaxies
                'r0': (0.1, 15),      # Earlier activation for some galaxies
                'delta': (0.1, 10),   # Variable transition width
                'gamma': (0.1, 2.0),  # More flexible profile
                'beta': (0.0, 0.5)    # Stronger density coupling allowed
            }
        
        # Initial guess - different from MW optimization
        x0 = self.xp.array([
            150.0,  # v0 - slightly higher for SPARC
            15.0,   # rc - smaller core
            2.0,    # r0 - earlier activation
            2.0,    # delta
            0.8,    # gamma - different profile
            0.2     # beta - stronger coupling
        ], dtype=self.xp.float64)
        
        # CMA-ES parameters
        sigma = 0.3
        weights = self.xp.log(population_size/2 + 0.5) - \
                 self.xp.log(self.xp.arange(1, population_size//2 + 1))
        weights = weights / self.xp.sum(weights)
        mu_eff = 1.0 / self.xp.sum(weights**2)
        
        # Adaptation parameters
        cc = 4.0 / (len(x0) + 4.0)
        cs = (mu_eff + 2.0) / (len(x0) + mu_eff + 5.0)
        c1 = 2.0 / ((len(x0) + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((len(x0) + 2)**2 + mu_eff))
        
        # Initialize
        mean = x0.copy()
        ps = self.xp.zeros_like(mean)
        pc = self.xp.zeros_like(mean)
        C = self.xp.eye(len(mean), dtype=self.xp.float64)
        
        best_loss = float('inf')
        best_params = None
        
        logger.info("Starting SPARC-specific optimization...")
        logger.info(f"Population: {population_size}, Iterations: {n_iterations}")
        
        start_time = time.time()
        
        for iteration in range(n_iterations):
            # Sample population
            z = self.xp.random.randn(population_size, len(mean))
            y = z @ self.xp.linalg.cholesky(C).T
            x = mean + sigma * y
            
            # Apply bounds
            for i, (param_name, (low, high)) in enumerate(bounds.items()):
                x[:, i] = self.xp.clip(x[:, i], low, high)
            
            # Evaluate fitness
            fitness = self.xp.array([self.compute_loss(x[i]) for i in range(population_size)])
            
            # Sort by fitness
            idx = self.xp.argsort(fitness)
            fitness = fitness[idx]
            z = z[idx]
            
            # Update best
            if fitness[0] < best_loss:
                best_loss = float(fitness[0])
                best_params = x[idx[0]].copy()
                if self.use_gpu:
                    best_params = cp.asnumpy(best_params)
            
            # Update mean
            z_mean = self.xp.sum(z[:population_size//2] * weights[:, None], axis=0)
            mean = mean + sigma * C @ z_mean
            
            # Update evolution paths
            ps = (1 - cs) * ps + self.xp.sqrt(cs * (2 - cs) * mu_eff) * z_mean
            hsig = self.xp.linalg.norm(ps) / self.xp.sqrt(1 - (1-cs)**(2*(iteration+1))) < \
                   1.4 + 2/(len(mean)+1)
            pc = (1 - cc) * pc + hsig * self.xp.sqrt(cc * (2 - cc) * mu_eff) * C @ z_mean
            
            # Update covariance
            C = (1 - c1 - cmu) * C + \
                c1 * (pc[:, None] @ pc[None, :]) + \
                cmu * (z[:population_size//2].T @ (weights[:, None] * z[:population_size//2]))
            
            # Update step size
            sigma = sigma * self.xp.exp((cs / 2) * (self.xp.linalg.norm(ps) / 
                                        self.xp.sqrt(len(mean)) - 1))
            
            # Progress report
            if iteration % 50 == 0:
                elapsed = time.time() - start_time
                rate = (iteration + 1) / elapsed
                logger.info(f"[Iter {iteration}] Loss: {best_loss:.4f} | "
                          f"Sigma: {float(sigma):.4f} | Rate: {rate:.1f} iter/s")
                
                # Save checkpoint
                self.save_checkpoint(iteration, best_params, best_loss)
            
            # Early stopping if converged
            if sigma < 1e-10:
                logger.info(f"Converged at iteration {iteration}")
                break
        
        elapsed = time.time() - start_time
        logger.info(f"Optimization complete in {elapsed:.1f} seconds")
        
        return best_params, best_loss
    
    def save_checkpoint(self, iteration, params, loss):
        """Save optimization checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'loss': float(loss),
            'parameters': {
                'v0_kms': float(params[0]),
                'rc_kpc': float(params[1]),
                'r0_kpc': float(params[2]),
                'delta_kpc': float(params[3]),
                'gamma': float(params[4]),
                'beta': float(params[5])
            },
            'dataset': 'SPARC',
            'n_galaxies': len(self.galaxies),
            'gpu_used': self.use_gpu
        }
        
        # Save to file
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'sparc_optimization_checkpoint.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def save_final_results(self, params, loss):
        """Save final optimized parameters."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'final_loss': float(loss),
            'parameters': {
                'v0_kms': float(params[0]),
                'rc_kpc': float(params[1]),
                'r0_kpc': float(params[2]),
                'delta_kpc': float(params[3]),
                'gamma': float(params[4]),
                'beta': float(params[5])
            },
            'dataset_info': {
                'type': 'SPARC',
                'n_galaxies': len(self.galaxies),
                'total_points': sum(len(g['r']) for g in self.galaxies.values())
            },
            'optimization_info': {
                'gpu_used': self.use_gpu,
                'gpu_available': GPU_AVAILABLE
            }
        }
        
        # Save main results
        with open('sparc_optimized_parameters.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save in results directory
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'sparc_final_parameters.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to sparc_optimized_parameters.json")
        
        # Print summary
        print("\n" + "="*60)
        print("SPARC OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Final Loss (χ²/dof): {loss:.4f}")
        print("\nOptimized Parameters:")
        for param_name, value in results['parameters'].items():
            print(f"  {param_name}: {value:.3f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='SPARC-specific LogTail optimization')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of optimization iterations')
    parser.add_argument('--population', type=int, default=64,
                       help='CMA-ES population size')
    parser.add_argument('--max-galaxies', type=int, default=None,
                       help='Limit number of galaxies (None = all 175)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if GPU available')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = SPARCOptimizer(use_gpu=not args.cpu)
    
    # Load data
    optimizer.load_sparc_data(max_galaxies=args.max_galaxies)
    
    # Run optimization
    best_params, best_loss = optimizer.optimize(
        n_iterations=args.iterations,
        population_size=args.population
    )
    
    # Save results
    optimizer.save_final_results(best_params, best_loss)


if __name__ == "__main__":
    main()