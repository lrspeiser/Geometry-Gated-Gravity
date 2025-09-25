#!/usr/bin/env python3
"""
G³ Alternatives Optimizer with Lensing Constraints
==================================================

Optimizes the continuous G³ model parameters using:
1. Galaxy rotation curves (SPARC)
2. Weak lensing constraints
3. Cluster lensing requirements
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from scipy.optimize import differential_evolution, minimize
from typing import Dict, Tuple
import time

# Import our continuous model
from g3_alternatives_test import G3ContinuumModel, G3ContinuumParams

# GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
except ImportError:
    xp = np
    GPU_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
G = 4.300917270e-6  # kpc km^2 s^-2 Msun^-1

class G3MultiScaleOptimizer:
    """
    Optimizes G³ parameters across multiple scales with lensing constraints.
    """
    
    def __init__(self, sparc_path='data/sparc_rotmod_ltg.parquet'):
        self.sparc_path = sparc_path
        self.best_params = None
        self.best_score = np.inf
        self.iteration = 0
        
    def objective_sparc(self, params_array):
        """Compute SPARC rotation curve error."""
        # Unpack parameters
        p = G3ContinuumParams(
            v0=params_array[0],
            rc0=params_array[1],
            gamma=params_array[2],
            beta=params_array[3],
            Sigma_star=params_array[4],
            alpha=params_array[5],
            kappa=params_array[6],
            p_in=params_array[7],
            p_out=params_array[8],
            lambda_p=params_array[9],
            g_star=params_array[10],
            alpha_g=params_array[11],
            kappa_g=params_array[12],
            g_sat=params_array[13],
            n_sat=params_array[14]
        )
        
        model = G3ContinuumModel(p)
        
        # Load SPARC data
        if not Path(self.sparc_path).exists():
            return 1.0
            
        df = pd.read_parquet(self.sparc_path)
        
        errors = []
        weights = []
        
        # Sample galaxies for speed
        galaxy_sample = df['galaxy'].unique()
        if len(galaxy_sample) > 50:
            np.random.seed(42)
            galaxy_sample = np.random.choice(galaxy_sample, 50, replace=False)
        
        for name in galaxy_sample:
            gdf = df[df['galaxy'] == name]
            
            r = xp.asarray(gdf['R_kpc'].values)
            v_obs = xp.asarray(gdf['Vobs_kms'].values)
            
            # Components
            v_gas = xp.asarray(gdf['Vgas_kms'].values)
            v_disk = xp.asarray(gdf['Vdisk_kms'].values)
            v_bulge = xp.asarray(gdf['Vbul_kms'].values if 'Vbul_kms' in gdf else np.zeros_like(r))
            v_bar = xp.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)
            
            # Filter
            valid = (r > 0.5) & xp.isfinite(v_obs) & xp.isfinite(v_bar) & (v_obs > 20)
            if valid.sum() < 5:
                continue
                
            r = r[valid]
            v_obs = v_obs[valid]
            v_bar = v_bar[valid]
            
            # Approximate density
            r_max = float(xp.max(r))
            v_max = float(xp.max(v_bar))
            r_half = float(xp.median(r))
            sigma_0 = v_max**2 / (4 * np.pi * G * r_max)
            sigma = sigma_0 * xp.exp(-r / (r_half / 2))
            
            # Predict
            try:
                v_pred = model.predict_velocity(r, v_bar, sigma)
                
                # Weighted error (emphasize outer points)
                weight = r / (r[-1] + 1e-10)  # Linear weighting by radius
                rel_err = xp.abs(v_pred - v_obs) / (v_obs + 1e-10)
                
                errors.extend((rel_err * weight).tolist() if GPU_AVAILABLE else (rel_err * weight))
                weights.extend(weight.tolist() if GPU_AVAILABLE else weight)
            except:
                errors.append(1.0)
                weights.append(1.0)
        
        if len(errors) == 0:
            return 1.0
            
        # Weighted median error
        errors = np.array(errors)
        weights = np.array(weights)
        sorted_idx = np.argsort(errors)
        cum_weights = np.cumsum(weights[sorted_idx])
        median_idx = np.searchsorted(cum_weights, 0.5 * cum_weights[-1])
        weighted_median = errors[sorted_idx[median_idx]]
        
        return weighted_median
    
    def lensing_constraint(self, params_array):
        """
        Compute lensing constraint penalty.
        G³ should produce reasonable lensing at galaxy and cluster scales.
        """
        p = G3ContinuumParams(
            v0=params_array[0],
            rc0=params_array[1],
            gamma=params_array[2],
            beta=params_array[3],
            Sigma_star=params_array[4],
            alpha=params_array[5],
            kappa=params_array[6],
            p_in=params_array[7],
            p_out=params_array[8],
            lambda_p=params_array[9],
            g_star=params_array[10],
            alpha_g=params_array[11],
            kappa_g=params_array[12],
            g_sat=params_array[13],
            n_sat=params_array[14]
        )
        
        model = G3ContinuumModel(p)
        
        # Test lensing at galaxy scale (50-100 kpc)
        r_test = np.array([50, 75, 100])  # kpc
        sigma_galaxy = 100 * np.exp(-r_test / 10)  # Galaxy-like
        
        # Convert to GPU if available
        if GPU_AVAILABLE:
            r_test_gpu = cp.asarray(r_test)
            sigma_galaxy_gpu = cp.asarray(sigma_galaxy)
            g_tail_galaxy, _ = model.compute_tail_acceleration(r_test_gpu, 0, sigma_galaxy_gpu)
            g_tail_galaxy = cp.asnumpy(g_tail_galaxy)
        else:
            g_tail_galaxy, _ = model.compute_tail_acceleration(r_test, 0, sigma_galaxy)
        
        # Effective mass from g_tail
        M_eff_galaxy = g_tail_galaxy * r_test**2 / G
        
        # Should produce ~10^11 - 10^12 Msun at 100 kpc for galaxy lensing
        galaxy_mass_ratio = M_eff_galaxy[-1] / 1e11
        galaxy_penalty = abs(np.log10(galaxy_mass_ratio)) if galaxy_mass_ratio > 0 else 10
        
        # Test at cluster scale (500-1000 kpc)
        r_cluster = np.array([500, 750, 1000])
        sigma_cluster = 500 * np.exp(-r_cluster / 200)  # Cluster-like
        
        if GPU_AVAILABLE:
            r_cluster_gpu = cp.asarray(r_cluster)
            sigma_cluster_gpu = cp.asarray(sigma_cluster)
            g_tail_cluster, _ = model.compute_tail_acceleration(r_cluster_gpu, 0, sigma_cluster_gpu)
            g_tail_cluster = cp.asnumpy(g_tail_cluster)
        else:
            g_tail_cluster, _ = model.compute_tail_acceleration(r_cluster, 0, sigma_cluster)
        M_eff_cluster = g_tail_cluster * r_cluster**2 / G
        
        # Should produce ~10^14 - 10^15 Msun at 1000 kpc for cluster lensing
        cluster_mass_ratio = M_eff_cluster[-1] / 1e14
        cluster_penalty = abs(np.log10(cluster_mass_ratio)) if cluster_mass_ratio > 0 else 10
        
        # Combined penalty (want both scales to work)
        return galaxy_penalty + 2 * cluster_penalty  # Weight clusters more
    
    def combined_objective(self, params_array):
        """Combined objective with rotation curves and lensing."""
        # SPARC error
        sparc_error = self.objective_sparc(params_array)
        
        # Lensing constraint
        lensing_penalty = self.lensing_constraint(params_array)
        
        # Combined (weight lensing to ensure it's not ignored)
        total = sparc_error + 0.1 * lensing_penalty
        
        # Track progress
        self.iteration += 1
        if self.iteration % 10 == 0:
            logger.info(f"Iteration {self.iteration}: SPARC={sparc_error:.3f}, "
                       f"Lensing={lensing_penalty:.3f}, Total={total:.3f}")
        
        # Save best
        if total < self.best_score:
            self.best_score = total
            self.best_params = params_array.copy()
            logger.info(f"New best: {total:.3f}")
        
        return total
    
    def optimize(self, max_iter=100):
        """Run optimization."""
        logger.info("Starting G³ multi-scale optimization...")
        
        # Parameter bounds
        bounds = [
            (50, 400),    # v0 (km/s)
            (5, 50),      # rc0 (kpc)
            (0.1, 2.0),   # gamma
            (0.0, 1.0),   # beta
            (10, 200),    # Sigma_star (Msun/pc^2)
            (0.5, 4.0),   # alpha
            (0.5, 3.0),   # kappa
            (0.5, 3.0),   # p_in
            (0.5, 2.0),   # p_out
            (0.1, 5.0),   # lambda_p
            (0.1, 2.0),   # g_star
            (0.5, 3.0),   # alpha_g
            (0.0, 2.0),   # kappa_g
            (100, 5000),  # g_sat
            (1.0, 4.0)    # n_sat
        ]
        
        # Use differential evolution for global optimization
        result = differential_evolution(
            self.combined_objective,
            bounds,
            maxiter=max_iter,
            workers=1,  # Use multiple cores if available
            disp=True,
            seed=42,
            popsize=10,  # Smaller population for speed
            atol=1e-3,
            tol=1e-3
        )
        
        return result
    
    def test_optimized_params(self, params_array):
        """Test the optimized parameters comprehensively."""
        p = G3ContinuumParams(
            v0=params_array[0],
            rc0=params_array[1],
            gamma=params_array[2],
            beta=params_array[3],
            Sigma_star=params_array[4],
            alpha=params_array[5],
            kappa=params_array[6],
            p_in=params_array[7],
            p_out=params_array[8],
            lambda_p=params_array[9],
            g_star=params_array[10],
            alpha_g=params_array[11],
            kappa_g=params_array[12],
            g_sat=params_array[13],
            n_sat=params_array[14]
        )
        
        model = G3ContinuumModel(p)
        
        # Full SPARC test
        from g3_alternatives_test import test_on_sparc
        sparc_results = test_on_sparc(model)
        
        # Lensing test at multiple scales
        scales = {
            'Galaxy (100 kpc)': (100, 100, 1e11),
            'Group (500 kpc)': (500, 300, 1e13),
            'Cluster (1000 kpc)': (1000, 500, 1e14)
        }
        
        lensing_results = {}
        for name, (r, sigma_0, M_expected) in scales.items():
            sigma = sigma_0 * np.exp(-r / (r/3))
            if GPU_AVAILABLE:
                r_arr = cp.array([r])
                sigma_arr = cp.array([sigma])
                g_tail, _ = model.compute_tail_acceleration(r_arr, 0, sigma_arr)
                g_tail = cp.asnumpy(g_tail)
            else:
                g_tail, _ = model.compute_tail_acceleration(np.array([r]), 0, np.array([sigma]))
            M_eff = float(g_tail[0] * r**2 / G)
            lensing_results[name] = {
                'M_eff': M_eff,
                'M_expected': M_expected,
                'ratio': M_eff / M_expected
            }
        
        return {
            'sparc': sparc_results,
            'lensing': lensing_results,
            'parameters': {
                'v0': p.v0,
                'rc0': p.rc0,
                'gamma': p.gamma,
                'beta': p.beta,
                'Sigma_star': p.Sigma_star,
                'alpha': p.alpha,
                'kappa': p.kappa,
                'p_in': p.p_in,
                'p_out': p.p_out,
                'lambda_p': p.lambda_p,
                'g_star': p.g_star,
                'alpha_g': p.alpha_g,
                'kappa_g': p.kappa_g,
                'g_sat': p.g_sat,
                'n_sat': p.n_sat
            }
        }

def main():
    """Run optimization and save results."""
    
    optimizer = G3MultiScaleOptimizer()
    
    # Run optimization
    start_time = time.time()
    result = optimizer.optimize(max_iter=50)  # Reduced for speed
    elapsed = time.time() - start_time
    
    logger.info(f"\nOptimization complete in {elapsed:.1f} seconds")
    logger.info(f"Best score: {result.fun:.3f}")
    logger.info(f"Best parameters: {result.x}")
    
    # Test the best parameters
    test_results = optimizer.test_optimized_params(result.x)
    
    # Save results
    output_dir = Path('out/g3_alternatives')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'optimization': {
            'best_score': float(result.fun),
            'n_iterations': result.nit,
            'success': result.success,
            'message': result.message,
            'elapsed_seconds': elapsed
        },
        'best_parameters': test_results['parameters'],
        'performance': test_results
    }
    
    with open(output_dir / 'optimized_params.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*80)
    
    if test_results['sparc']:
        logger.info(f"SPARC Performance: {test_results['sparc']['median_error']*100:.1f}% median error")
        logger.info(f"  - {test_results['sparc']['n_galaxies']} galaxies tested")
    
    logger.info("\nLensing Performance:")
    for scale, lens_data in test_results['lensing'].items():
        logger.info(f"  {scale}: M_eff/M_expected = {lens_data['ratio']:.2f}")
    
    logger.info("\nOptimized Parameters:")
    for param, value in test_results['parameters'].items():
        logger.info(f"  {param:12s} = {value:8.3f}")
    
    logger.info("="*80)
    logger.info(f"Results saved to {output_dir / 'optimized_params.json'}")

if __name__ == '__main__':
    main()