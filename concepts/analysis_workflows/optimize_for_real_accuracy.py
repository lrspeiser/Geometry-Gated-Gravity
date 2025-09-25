#!/usr/bin/env python3
"""
Optimizer that ACTUALLY minimizes velocity prediction errors.
This optimizer uses the correct metric: relative error in predicted velocities.
"""

import numpy as np
import cupy as cp
import json
from cupyx.scipy.special import expit as cp_expit
import time
from datetime import datetime

def load_mw_data():
    """Load Milky Way Gaia data"""
    print("Loading Milky Way data...")
    data = np.load('data/mw_gaia_144k.npz')
    
    R_kpc = data['R_kpc']
    z_kpc = data['z_kpc']
    v_obs = data['v_obs_kms']
    v_err = data['v_err_kms']
    gN = data['gN_kms2_per_kpc']
    Sigma_loc = data['Sigma_loc_Msun_pc2']
    
    print(f"  Loaded {len(R_kpc):,} stars")
    print(f"  R range: {R_kpc.min():.1f} - {R_kpc.max():.1f} kpc")
    print(f"  Velocity range: {v_obs.min():.0f} - {v_obs.max():.0f} km/s")
    
    return R_kpc, z_kpc, v_obs, v_err, gN, Sigma_loc

def compute_loss_gpu(params, R_gpu, z_gpu, v_obs_gpu, v_err_gpu, gN_gpu, Sigma_loc_gpu, variant='thickness_gate'):
    """
    Compute the ACTUAL prediction error loss.
    
    Loss metrics computed:
    1. Median relative error (primary)
    2. Mean relative error
    3. Fraction within 10% error
    4. Weighted by observational uncertainty
    """
    v0, rc, r0, delta, Sigma0, alpha, hz, m = params
    
    # Enforce physical constraints
    if v0 <= 0 or rc <= 0 or delta <= 0 or Sigma0 <= 0 or hz <= 0:
        return 1e10
    if r0 < 0 or r0 > 20:
        return 1e10
    if alpha < 0 or alpha > 10:
        return 1e10
    if m < 0 or m > 5:
        return 1e10
    
    # Compute model predictions
    if variant == 'thickness_gate':
        rc_eff = rc * (1.0 + cp.power(cp.abs(z_gpu) / hz, m))
    else:
        rc_eff = rc
    
    rational = R_gpu / (R_gpu + rc_eff)
    x = (R_gpu - r0) / delta
    logistic = cp_expit(x)  # More numerically stable sigmoid
    
    sigma_ratio = Sigma0 / (Sigma_loc_gpu + 1e-6)
    sigma_screen = 1.0 / (1.0 + cp.power(sigma_ratio, alpha))
    
    g_additional = (v0**2 / R_gpu) * rational * logistic * sigma_screen
    g_total = gN_gpu + g_additional
    
    # Predicted velocities
    v_pred = cp.sqrt(cp.maximum(R_gpu * g_total, 0.0))
    
    # ACTUAL relative error for each star
    rel_error = cp.abs((v_pred - v_obs_gpu) / (v_obs_gpu + 1e-6))
    
    # Weight by observational uncertainty (give more weight to well-measured stars)
    weights = 1.0 / (1.0 + v_err_gpu / 10.0)  # Stars with <10 km/s error get full weight
    
    # Primary loss: Weighted median relative error
    weighted_errors = rel_error * weights
    loss_median = float(cp.median(weighted_errors))
    
    # Additional metrics for monitoring
    loss_mean = float(cp.mean(rel_error))
    frac_within_10pct = float(cp.mean(rel_error < 0.10))
    frac_within_5pct = float(cp.mean(rel_error < 0.05))
    
    # Combine metrics (primarily optimize median, with small penalty for mean)
    total_loss = loss_median + 0.1 * loss_mean
    
    # Add regularization to prevent extreme parameters
    reg_penalty = 0.001 * (cp.abs(cp.log(v0/200)) + cp.abs(cp.log(rc/3)) + 
                           cp.abs(cp.log(hz/0.3)) + cp.abs(cp.log(Sigma0/50)))
    
    total_loss += float(reg_penalty)
    
    return total_loss

def differential_evolution_gpu(R, z, v_obs, v_err, gN, Sigma_loc, 
                               variant='thickness_gate',
                               pop_size=128, max_iter=500, 
                               F=0.8, CR=0.9, sigma=0.3):
    """
    GPU-accelerated differential evolution optimizer.
    Optimizes for ACTUAL velocity prediction accuracy.
    """
    print(f"\n{'='*70}")
    print(f"OPTIMIZING FOR ACTUAL VELOCITY PREDICTION ACCURACY")
    print(f"{'='*70}")
    print(f"Variant: {variant}")
    print(f"Population size: {pop_size}")
    print(f"Max iterations: {max_iter}")
    print(f"Mutation factor F: {F}, Crossover rate CR: {CR}")
    print(f"Initial sigma: {sigma}")
    
    # Transfer data to GPU once
    R_gpu = cp.asarray(R, dtype=cp.float32)
    z_gpu = cp.asarray(z, dtype=cp.float32)
    v_obs_gpu = cp.asarray(v_obs, dtype=cp.float32)
    v_err_gpu = cp.asarray(v_err, dtype=cp.float32)
    gN_gpu = cp.asarray(gN, dtype=cp.float32)
    Sigma_loc_gpu = cp.asarray(Sigma_loc, dtype=cp.float32)
    
    # Parameter bounds (8 parameters for thickness_gate)
    bounds = [
        (100, 400),   # v0 [km/s] - asymptotic velocity
        (0.5, 10),    # rc [kpc] - core radius
        (2, 12),      # r0 [kpc] - transition radius
        (0.5, 5),     # delta [kpc] - transition width
        (10, 200),    # Sigma0 [Msun/pc^2] - critical density
        (0.1, 3),     # alpha - screening power
        (0.1, 2),     # hz [kpc] - vertical scale height
        (0.5, 3),     # m - vertical power index
    ]
    
    n_params = len(bounds)
    
    # Initialize population with Latin Hypercube Sampling for better coverage
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=n_params)
    sample = sampler.random(n=pop_size)
    
    # Scale to bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    population = qmc.scale(sample, lower, upper)
    
    # Add some focused initial guesses based on previous results
    if pop_size > 8:
        # Good starting points from previous runs
        population[0] = [268.5, 3.8, 6.2, 1.8, 65, 0.75, 0.35, 1.8]  # Previous best
        population[1] = [250, 4.0, 7.0, 2.0, 50, 1.0, 0.3, 2.0]      # Solar neighborhood focus
        population[2] = [280, 3.5, 5.5, 1.5, 80, 0.5, 0.4, 1.5]      # Inner galaxy focus
        population[3] = [300, 3.0, 8.0, 2.5, 40, 1.5, 0.3, 2.5]      # Outer galaxy focus
    
    population_gpu = cp.asarray(population, dtype=cp.float32)
    
    # Evaluate initial population
    fitness = cp.zeros(pop_size, dtype=cp.float32)
    for i in range(pop_size):
        fitness[i] = compute_loss_gpu(population_gpu[i], R_gpu, z_gpu, v_obs_gpu, 
                                      v_err_gpu, gN_gpu, Sigma_loc_gpu, variant)
    
    best_idx = cp.argmin(fitness)
    best_solution = population_gpu[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    print(f"\nInitial best fitness: {best_fitness:.6f}")
    print(f"Initial best params: {best_solution.get()}")
    
    # Evolution loop
    fitness_history = []
    param_history = []
    start_time = time.time()
    
    for generation in range(max_iter):
        # Adaptive parameters
        if generation > max_iter // 2:
            # Reduce exploration in later generations
            F_current = F * (1 - generation / max_iter) + 0.3
            CR_current = CR
        else:
            F_current = F
            CR_current = CR
        
        # For each individual in population
        for i in range(pop_size):
            # Select three random individuals (different from i)
            candidates = cp.arange(pop_size)
            candidates = candidates[candidates != i]
            idx = cp.random.choice(candidates, 3, replace=False)
            
            a, b, c = population_gpu[idx]
            
            # Mutation: use best solution as base occasionally
            if cp.random.random() < 0.1:  # 10% chance to use best
                mutant = best_solution + F_current * (a - b)
            else:
                mutant = a + F_current * (b - c)
            
            # Crossover
            cross_points = cp.random.random(n_params) < CR_current
            if not cp.any(cross_points):
                cross_points[cp.random.randint(n_params)] = True
            
            trial = cp.where(cross_points, mutant, population_gpu[i])
            
            # Boundary constraints with reflection
            for j in range(n_params):
                if trial[j] < bounds[j][0]:
                    trial[j] = bounds[j][0] + (bounds[j][0] - trial[j]) * 0.5
                elif trial[j] > bounds[j][1]:
                    trial[j] = bounds[j][1] - (trial[j] - bounds[j][1]) * 0.5
            
            # Selection
            trial_fitness = compute_loss_gpu(trial, R_gpu, z_gpu, v_obs_gpu, 
                                            v_err_gpu, gN_gpu, Sigma_loc_gpu, variant)
            
            if trial_fitness < fitness[i]:
                population_gpu[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < best_fitness:
                    best_solution = trial.copy()
                    best_fitness = trial_fitness
        
        # Log progress
        if generation % 10 == 0 or generation == max_iter - 1:
            # Compute actual metrics for best solution
            v0, rc, r0, delta, Sigma0, alpha, hz, m = best_solution.get()
            
            # Recompute with best params to get detailed metrics
            rc_eff = rc * (1.0 + cp.power(cp.abs(z_gpu) / hz, m))
            rational = R_gpu / (R_gpu + rc_eff)
            x = (R_gpu - r0) / delta
            logistic = cp_expit(x)
            sigma_ratio = Sigma0 / (Sigma_loc_gpu + 1e-6)
            sigma_screen = 1.0 / (1.0 + cp.power(sigma_ratio, alpha))
            g_additional = (v0**2 / R_gpu) * rational * logistic * sigma_screen
            g_total = gN_gpu + g_additional
            v_pred = cp.sqrt(cp.maximum(R_gpu * g_total, 0.0))
            
            rel_error = cp.abs((v_pred - v_obs_gpu) / v_obs_gpu) * 100
            median_error = float(cp.median(rel_error))
            mean_error = float(cp.mean(rel_error))
            frac_10 = float(cp.mean(rel_error < 10)) * 100
            frac_5 = float(cp.mean(rel_error < 5)) * 100
            
            elapsed = time.time() - start_time
            print(f"\nGen {generation:3d}/{max_iter} | Time: {elapsed:.1f}s")
            print(f"  Loss: {best_fitness:.6f}")
            print(f"  Median error: {median_error:.2f}% | Mean: {mean_error:.2f}%")
            print(f"  Stars <10%: {frac_10:.1f}% | <5%: {frac_5:.1f}%")
            print(f"  Best params: v0={v0:.1f}, rc={rc:.2f}, r0={r0:.2f}, delta={delta:.2f}")
            print(f"              Σ0={Sigma0:.1f}, α={alpha:.2f}, hz={hz:.3f}, m={m:.2f}")
        
        fitness_history.append(float(best_fitness))
        param_history.append(best_solution.get().tolist())
        
        # Early stopping if converged
        if len(fitness_history) > 50:
            recent = fitness_history[-50:]
            if max(recent) - min(recent) < 1e-6:
                print(f"\nConverged at generation {generation}")
                break
    
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    
    return best_solution.get(), best_fitness, fitness_history, param_history

def main():
    """Run optimization for actual velocity prediction accuracy"""
    
    # Load data
    R, z, v_obs, v_err, gN, Sigma_loc = load_mw_data()
    
    # Run optimization
    best_params, best_fitness, fitness_history, param_history = differential_evolution_gpu(
        R, z, v_obs, v_err, gN, Sigma_loc,
        variant='thickness_gate',
        pop_size=256,  # Larger population for better exploration
        max_iter=1000,  # More iterations
        F=0.8,
        CR=0.9,
        sigma=0.3
    )
    
    # Save results
    results = {
        'variant': 'thickness_gate_corrected',
        'timestamp': datetime.now().isoformat(),
        'params': best_params.tolist(),
        'param_names': ['v0', 'rc', 'r0', 'delta', 'Sigma0', 'alpha', 'hz', 'm'],
        'loss': float(best_fitness),
        'fitness_history': fitness_history,
        'param_history': param_history,
        'data_info': {
            'n_stars': len(R),
            'R_range': [float(R.min()), float(R.max())],
            'v_range': [float(v_obs.min()), float(v_obs.max())]
        }
    }
    
    # Compute final metrics
    v0, rc, r0, delta, Sigma0, alpha, hz, m = best_params
    
    # CPU computation for final metrics
    rc_eff = rc * (1.0 + np.power(np.abs(z) / hz, m))
    rational = R / (R + rc_eff)
    x = (R - r0) / delta
    logistic = 1.0 / (1.0 + np.exp(-x))
    sigma_ratio = Sigma0 / (Sigma_loc + 1e-6)
    sigma_screen = 1.0 / (1.0 + np.power(sigma_ratio, alpha))
    g_additional = (v0**2 / R) * rational * logistic * sigma_screen
    g_total = gN + g_additional
    v_pred = np.sqrt(np.maximum(R * g_total, 0.0))
    
    rel_error = np.abs((v_pred - v_obs) / v_obs) * 100
    
    results['final_metrics'] = {
        'median_error_pct': float(np.median(rel_error)),
        'mean_error_pct': float(np.mean(rel_error)),
        'std_error_pct': float(np.std(rel_error)),
        'frac_within_5pct': float(np.mean(rel_error < 5)),
        'frac_within_10pct': float(np.mean(rel_error < 10)),
        'frac_within_20pct': float(np.mean(rel_error < 20))
    }
    
    # Save to file
    output_file = 'out/mw_orchestrated/optimized_for_accuracy.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Median error: {results['final_metrics']['median_error_pct']:.2f}%")
    print(f"Mean error: {results['final_metrics']['mean_error_pct']:.2f}%")
    print(f"Stars within 5%: {results['final_metrics']['frac_within_5pct']*100:.1f}%")
    print(f"Stars within 10%: {results['final_metrics']['frac_within_10pct']*100:.1f}%")
    print(f"Stars within 20%: {results['final_metrics']['frac_within_20pct']*100:.1f}%")
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == '__main__':
    results = main()