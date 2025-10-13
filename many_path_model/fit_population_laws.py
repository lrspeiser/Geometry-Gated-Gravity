#!/usr/bin/env python3
"""
Population Law Fitter for SPARC Galaxies
=========================================

Replaces discrete morphological classes with smooth, continuous functions
of measurable galaxy properties (B/T, M_star, R_d, etc.).

Implements smooth laws:
- eta(B/T, M_star) = a0 + a1*log(M_star) + a2*B/T + a3*(B/T)^2
- ring_amp(B/T) = b0 * exp(-b1 * B/T)
- M_max(M_star, R_d) = c0 + c1*log(M_star) + c2*log(R_d)
- lambda_hat(R_d) = d0 + d1*R_d

Uses k-fold cross-validation to prevent overfitting and reports
generalization performance.

Usage:
    # Fit all population laws with 5-fold CV
    python many_path_model/fit_population_laws.py --sparc_dir data/Rotmod_LTG --output_dir results/pop_laws --cv_folds 5

    # Quick test with 3-fold CV
    python many_path_model/fit_population_laws.py --sparc_dir data/Rotmod_LTG --output_dir results/pop_laws_test --cv_folds 3
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import time

# Optimization
from scipy.optimize import minimize, differential_evolution

# GPU check
try:
    import cupy as cp
    _USING_CUPY = True
except ImportError:
    import numpy as cp
    _USING_CUPY = False

# Import galaxy loading and prediction
sys.path.insert(0, str(Path(__file__).parent))
from sparc_stratified_test import (
    load_sparc_galaxy, load_sparc_master_table,
    predict_rotation_curve_fast, compute_metrics
)


@dataclass
class PopulationLaws:
    """Container for population law parameters."""
    # eta(B/T, M_star) = a0 + a1*log(M_star) + a2*B/T + a3*(B/T)^2
    eta_a0: float
    eta_a1: float  # log(M_star) coefficient
    eta_a2: float  # B/T linear
    eta_a3: float  # B/T quadratic
    
    # ring_amp(B/T) = b0 * exp(-b1 * B/T)
    ring_amp_b0: float
    ring_amp_b1: float
    
    # M_max(M_star, R_d) = c0 + c1*log(M_star) + c2*log(R_d)
    M_max_c0: float
    M_max_c1: float
    M_max_c2: float
    
    # lambda_hat(R_d) = d0 + d1*R_d (but kept fixed for now)
    lambda_hat_fixed: float = 20.0
    
    # Bulge gate (keep near MW-anchored values)
    bulge_gate_power_fixed: float = 32.9
    
    def to_dict(self) -> Dict:
        return {
            'eta_a0': self.eta_a0,
            'eta_a1': self.eta_a1,
            'eta_a2': self.eta_a2,
            'eta_a3': self.eta_a3,
            'ring_amp_b0': self.ring_amp_b0,
            'ring_amp_b1': self.ring_amp_b1,
            'M_max_c0': self.M_max_c0,
            'M_max_c1': self.M_max_c1,
            'M_max_c2': self.M_max_c2,
            'lambda_hat_fixed': self.lambda_hat_fixed,
            'bulge_gate_power_fixed': self.bulge_gate_power_fixed
        }
    
    @staticmethod
    def from_params(params: np.ndarray) -> 'PopulationLaws':
        """Create from parameter array."""
        return PopulationLaws(
            eta_a0=params[0],
            eta_a1=params[1],
            eta_a2=params[2],
            eta_a3=params[3],
            ring_amp_b0=params[4],
            ring_amp_b1=params[5],
            M_max_c0=params[6],
            M_max_c1=params[7],
            M_max_c2=params[8]
        )
    
    def predict_params(self, bulge_frac: float, M_star: float, R_d: float) -> Dict:
        """
        Predict per-galaxy parameters from population laws.
        
        Args:
            bulge_frac: Bulge-to-total ratio (0-1)
            M_star: Stellar mass in solar masses
            R_d: Disk scale length in kpc
        
        Returns:
            Dictionary of predicted parameters for rotation curve model
        """
        # Clamp inputs to reasonable ranges
        bulge_frac = np.clip(bulge_frac, 0.0, 1.0)
        log_M_star = np.log10(np.clip(M_star, 1e8, 1e12))
        log_R_d = np.log10(np.clip(R_d, 0.1, 100.0))
        
        # Compute parameters from laws
        eta = self.eta_a0 + self.eta_a1 * log_M_star + \
              self.eta_a2 * bulge_frac + self.eta_a3 * bulge_frac**2
        
        ring_amp = self.ring_amp_b0 * np.exp(-self.ring_amp_b1 * bulge_frac)
        
        M_max = self.M_max_c0 + self.M_max_c1 * log_M_star + self.M_max_c2 * log_R_d
        
        # Clamp outputs to physical ranges
        eta = np.clip(eta, 0.01, 2.0)
        ring_amp = np.clip(ring_amp, 0.0, 15.0)
        M_max = np.clip(M_max, 0.5, 5.0)
        
        return {
            'eta': eta,
            'ring_amp': ring_amp,
            'M_max': M_max,
            'bulge_gate_power': self.bulge_gate_power_fixed,
            'lambda_hat': self.lambda_hat_fixed
        }


def estimate_stellar_mass(galaxy) -> float:
    """
    Estimate stellar mass from rotation curve data.
    Uses baryonic mass at fiducial radius as proxy.
    """
    # Use median radius point
    idx_mid = len(galaxy.r_kpc) // 2
    r_fid = galaxy.r_kpc[idx_mid]
    
    # Total velocity squared at fiducial radius
    v_tot_sq = galaxy.v_gas[idx_mid]**2 + \
               galaxy.v_disk[idx_mid]**2 + \
               galaxy.v_bulge[idx_mid]**2
    
    # Rough mass estimate: M = v^2 * r / G
    G = 4.30091e-6  # kpc (km/s)^2 / M_sun
    M_bary = v_tot_sq * r_fid / G
    
    # Stellar mass is typically 60-80% of baryonic mass
    M_star = 0.7 * M_bary
    
    return np.clip(M_star, 1e8, 1e12)


def estimate_disk_scale_length(galaxy) -> float:
    """
    Estimate disk scale length from surface brightness profile.
    Uses exponential fit to disk SB.
    """
    # Find where disk SB drops to 1/e
    sb_disk = galaxy.sb_disk
    if len(sb_disk) == 0 or np.max(sb_disk) == 0:
        return 3.0  # Default
    
    sb_max = np.max(sb_disk)
    sb_threshold = sb_max / np.e
    
    # Find radius where SB crosses threshold
    mask = sb_disk > sb_threshold
    if np.sum(mask) > 0:
        R_d = np.max(galaxy.r_kpc[mask])
    else:
        R_d = galaxy.r_kpc[len(galaxy.r_kpc)//2]
    
    return np.clip(R_d, 0.5, 50.0)


def evaluate_population_laws(params: np.ndarray, galaxies: List, 
                            train_indices: np.ndarray,
                            return_details: bool = False,
                            l2_lambda: float = 0.01) -> float:
    """
    Evaluate population laws on a set of galaxies with L2 regularization.
    
    Args:
        params: Population law parameters
        galaxies: List of all galaxies
        train_indices: Indices of galaxies to evaluate on
        return_details: If True, return per-galaxy errors
        l2_lambda: L2 regularization strength (ridge penalty)
    
    Returns:
        Median APE + L2 penalty across training galaxies (or list if return_details=True)
    """
    laws = PopulationLaws.from_params(params)
    errors = []
    
    for idx in train_indices:
        galaxy = galaxies[idx]
        
        try:
            # Get galaxy properties
            bulge_frac = galaxy.avg_bulge_frac
            M_star = estimate_stellar_mass(galaxy)
            R_d = estimate_disk_scale_length(galaxy)
            
            # Predict parameters from population laws
            pred_params = laws.predict_params(bulge_frac, M_star, R_d)
            
            # Compute rotation curve
            v_pred = predict_rotation_curve_fast(
                galaxy, pred_params,
                use_bulge_gate=True,
                n_particles=50000  # Faster evaluation
            )
            
            # Compute error
            metrics = compute_metrics(galaxy.v_obs, v_pred, galaxy.v_err)
            errors.append(metrics['ape'])
            
        except Exception as e:
            # Penalize failures
            errors.append(100.0)
    
    if return_details:
        return errors
    else:
        # Compute median APE
        median_ape = np.median(errors)
        
        # Add L2 regularization penalty (ridge regression)
        # Penalize large coefficients to prevent overfitting
        l2_penalty = l2_lambda * np.sum(params**2)
        
        return median_ape + l2_penalty


def fit_population_laws_cv(galaxies: List, n_folds: int = 5,
                          l2_lambda: float = 0.01,
                          verbose: bool = True) -> Tuple[PopulationLaws, Dict]:
    """
    Fit population laws using k-fold cross-validation with L2 regularization.
    
    Args:
        galaxies: List of all galaxies
        n_folds: Number of CV folds
        l2_lambda: L2 regularization strength (default: 0.01)
        verbose: Print progress
    
    Returns:
        (best_laws, results_dict)
    """
    n_galaxies = len(galaxies)
    indices = np.arange(n_galaxies)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    # Create stratified folds by morphology
    fold_size = n_galaxies // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_galaxies
        folds.append(indices[start:end])
    
    # Initial guess from class-wise medians
    # eta: late ~0.7, early ~0.03, intermediate ~0.015
    # ring_amp: late ~2.2, early ~1.5, intermediate ~0
    # M_max: all ~2.5
    params_init = np.array([
        0.5,   # eta_a0 (baseline)
        0.0,   # eta_a1 (mass dependence)
        -0.5,  # eta_a2 (B/T linear, negative = decreases with bulge)
        0.0,   # eta_a3 (B/T quadratic)
        2.5,   # ring_amp_b0 (max amplitude)
        2.0,   # ring_amp_b1 (decay rate with B/T)
        2.5,   # M_max_c0 (baseline)
        0.0,   # M_max_c1 (mass scaling)
        0.0,   # M_max_c2 (size scaling)
    ])
    
    # Parameter bounds (wide but physical)
    bounds = [
        (0.0, 2.0),    # eta_a0
        (-0.5, 0.5),   # eta_a1
        (-2.0, 2.0),   # eta_a2
        (-2.0, 2.0),   # eta_a3
        (0.1, 15.0),   # ring_amp_b0
        (0.0, 10.0),   # ring_amp_b1
        (0.5, 5.0),    # M_max_c0
        (-0.5, 0.5),   # M_max_c1
        (-0.5, 0.5),   # M_max_c2
    ]
    
    print(f"\n{'='*80}")
    print(f"POPULATION LAW FITTING ({n_folds}-FOLD CROSS-VALIDATION)")
    print(f"{'='*80}")
    print(f"Galaxies: {n_galaxies}")
    print(f"Folds: {n_folds}")
    print(f"L2 Lambda: {l2_lambda}")
    print(f"GPU: {'ENABLED' if _USING_CUPY else 'DISABLED'}")
    print()
    
    cv_results = []
    
    # Run k-fold CV
    for fold_idx in range(n_folds):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        
        # Split train/val
        val_indices = folds[fold_idx]
        train_indices = np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx])
        
        print(f"Train: {len(train_indices)} galaxies")
        print(f"Val:   {len(val_indices)} galaxies")
        
        # Optimize on training set
        print("\nOptimizing population laws...")
        start = time.time()
        
        result = differential_evolution(
            lambda p: evaluate_population_laws(p, galaxies, train_indices, 
                                             return_details=False, l2_lambda=l2_lambda),
            bounds=bounds,
            seed=42 + fold_idx,
            maxiter=50,  # Reduced for faster completion
            popsize=10,  # Reduced for faster completion
            workers=1,  # Avoid nested parallelism
            updating='deferred',
            polish=True
        )
        
        elapsed = time.time() - start
        
        # Evaluate on validation set (no L2 penalty for validation)
        val_errors = evaluate_population_laws(
            result.x, galaxies, val_indices, return_details=True, l2_lambda=0.0
        )
        
        train_median = result.fun
        val_median = np.median(val_errors)
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Train median APE: {train_median:.2f}%")
        print(f"  Val median APE:   {val_median:.2f}%")
        print(f"  Time: {elapsed:.1f}s")
        
        cv_results.append({
            'fold': fold_idx,
            'train_ape': train_median,
            'val_ape': val_median,
            'val_errors': val_errors,
            'params': result.x,
            'time': elapsed
        })
    
    # Aggregate results
    train_apes = [r['train_ape'] for r in cv_results]
    val_apes = [r['val_ape'] for r in cv_results]
    
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Train APE: {np.mean(train_apes):.2f}% ± {np.std(train_apes):.2f}%")
    print(f"Val APE:   {np.mean(val_apes):.2f}% ± {np.std(val_apes):.2f}%")
    print(f"Generalization gap: {np.mean(val_apes) - np.mean(train_apes):+.2f}%")
    
    # Use parameters from best validation fold
    best_fold_idx = np.argmin(val_apes)
    best_params = cv_results[best_fold_idx]['params']
    best_laws = PopulationLaws.from_params(best_params)
    
    print(f"\nBest fold: {best_fold_idx + 1} (val APE: {val_apes[best_fold_idx]:.2f}%)")
    print(f"\nLearned Population Laws:")
    print(f"  eta(B/T, M*) = {best_laws.eta_a0:.3f} + {best_laws.eta_a1:.3f}*log(M*) + "
          f"{best_laws.eta_a2:.3f}*B/T + {best_laws.eta_a3:.3f}*(B/T)^2")
    print(f"  ring_amp(B/T) = {best_laws.ring_amp_b0:.3f} * exp(-{best_laws.ring_amp_b1:.3f}*B/T)")
    print(f"  M_max(M*, Rd) = {best_laws.M_max_c0:.3f} + {best_laws.M_max_c1:.3f}*log(M*) + "
          f"{best_laws.M_max_c2:.3f}*log(Rd)")
    print(f"{'='*80}")
    
    results_summary = {
        'cv_results': cv_results,
        'train_ape_mean': float(np.mean(train_apes)),
        'train_ape_std': float(np.std(train_apes)),
        'val_ape_mean': float(np.mean(val_apes)),
        'val_ape_std': float(np.std(val_apes)),
        'best_fold': int(best_fold_idx),
        'best_val_ape': float(val_apes[best_fold_idx])
    }
    
    return best_laws, results_summary


def main():
    parser = argparse.ArgumentParser(description='Fit Population Laws for SPARC')
    parser.add_argument('--sparc_dir', default='data/Rotmod_LTG', help='SPARC data directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--l2_lambda', type=float, default=0.01, help='L2 regularization strength')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.sparc_dir)
    master_file = data_dir / "MasterSheet_SPARC.mrt"
    
    # Load galaxies
    print("Loading SPARC galaxies...")
    master_info = load_sparc_master_table(master_file)
    
    rotmod_files = sorted(data_dir.glob('*_rotmod.dat'))
    galaxies = []
    for rotmod_file in rotmod_files:
        try:
            galaxy = load_sparc_galaxy(rotmod_file, master_info)
            galaxies.append(galaxy)
        except Exception as e:
            print(f"Warning: Failed to load {rotmod_file.name}: {e}")
    
    print(f"Loaded {len(galaxies)} galaxies")
    
    # Fit population laws with CV
    best_laws, cv_summary = fit_population_laws_cv(
        galaxies,
        n_folds=args.cv_folds,
        l2_lambda=args.l2_lambda,
        verbose=True
    )
    
    # Save results
    output_data = {
        'population_laws': best_laws.to_dict(),
        'cv_summary': cv_summary,
        'n_galaxies': len(galaxies),
        'n_folds': args.cv_folds,
        'l2_lambda': args.l2_lambda
    }
    
    output_file = output_dir / 'population_laws.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nPopulation laws saved to: {output_file}")
    print("\nNext step: Test zero-shot prediction with:")
    print(f"  python many_path_model/sparc_zero_shot_population.py "
          f"--laws {output_file} --output_dir results/zero_shot_population")


if __name__ == '__main__':
    main()
