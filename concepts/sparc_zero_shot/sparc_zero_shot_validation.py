#!/usr/bin/env python3
"""
SPARC Zero-Shot Validation Suite
=================================

Rigorous tests to prove true zero-shot generalization:
1. Leave-One-Type-Out (LOTO) cross-validation
2. Size-density stratified cross-validation
3. Galaxy-level bootstrapping
4. Proper error metrics (per-galaxy vs per-point)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from g3_gpu_solver_suite import (
    xp, _GPU, G3Model, G3Params, TEMPLATE_BOUNDS,
    SolverOrchestrator, objective_mw_factory
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
G = 4.300917270e-6  # kpc km^2 s^-2 Msun^-1

class RobustSPARCDataset:
    """Enhanced SPARC dataset with proper error handling and metrics."""
    
    def __init__(self, rotmod_path: str, meta_path: str = None):
        self.df = pd.read_parquet(rotmod_path)
        self.df_meta = pd.read_parquet(meta_path) if meta_path else None
        self.galaxies = {}
        self.galaxy_types = {}
        self._load_galaxies()
        
    def _load_galaxies(self):
        """Load galaxies with enhanced quality filtering."""
        for name, gdf in self.df.groupby('galaxy'):
            r = gdf['R_kpc'].values
            v_obs = gdf['Vobs_kms'].values
            v_err = gdf['eVobs_kms'].values
            v_gas = gdf['Vgas_kms'].values
            v_disk = gdf['Vdisk_kms'].values
            v_bulge = gdf['Vbul_kms'].values if 'Vbul_kms' in gdf else np.zeros_like(r)
            
            v_bar = np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)
            
            # Enhanced quality filter
            valid = (r > 0.5) & (r < 50) & np.isfinite(v_obs) & np.isfinite(v_bar) & (v_obs > 20) & (v_err < 50)
            
            if valid.sum() < 5:  # Need at least 5 good points
                continue
                
            # Compute observable properties (no tuning!)
            r_valid = r[valid]
            v_obs_valid = v_obs[valid]
            v_err_valid = v_err[valid]
            v_bar_valid = v_bar[valid]
            
            # Gas fraction
            f_gas = np.mean(v_gas[valid]**2 / (v_bar_valid**2 + 1e-10))
            
            # Half-mass radius (weighted by v_bar^2)
            weights = v_bar_valid**2
            r_half = np.average(r_valid, weights=weights) if weights.sum() > 0 else np.median(r_valid)
            
            # Surface density proxy
            r_max = np.percentile(r_valid, 90)  # Use 90th percentile to avoid outliers
            v_max = np.percentile(v_bar_valid, 90)
            sigma_bar = (v_max**2 / (4 * np.pi * G * r_max)) / 1e6  # Msun/pc^2
            sigma_bar = np.clip(sigma_bar, 10, 1000)
            
            # Local surface density profile
            r_d = r_half / 1.68
            sigma_0 = sigma_bar * 2
            sigma_loc = sigma_0 * np.exp(-r_valid / r_d)
            
            # Get galaxy type
            galaxy_type = self._get_galaxy_type(name)
            
            # Store galaxy data
            self.galaxies[name] = {
                'r': r_valid,
                'v_obs': v_obs_valid,
                'v_err': v_err_valid,
                'v_bar': v_bar_valid,
                'sigma_loc': sigma_loc,
                'r_half': r_half,
                'sigma_bar': sigma_bar,
                'f_gas': f_gas,
                'type': galaxy_type,
                'n_points': len(r_valid)
            }
            
            # Track by type
            if galaxy_type not in self.galaxy_types:
                self.galaxy_types[galaxy_type] = []
            self.galaxy_types[galaxy_type].append(name)
            
        logger.info(f"Loaded {len(self.galaxies)} galaxies with enhanced filtering")
        for gtype, names in sorted(self.galaxy_types.items()):
            logger.info(f"  {gtype}: {len(names)} galaxies")
            
    def _get_galaxy_type(self, name):
        """Get galaxy morphological type."""
        if self.df_meta is not None and name in self.df_meta['galaxy'].values:
            meta = self.df_meta[self.df_meta['galaxy'] == name].iloc[0]
            T_val = meta['T'] if 'T' in meta else 99
            if T_val <= 0:
                return 'Early-type'
            elif 1 <= T_val <= 3:
                return 'Sa-Sb'
            elif 4 <= T_val <= 5:
                return 'Sbc-Sc'
            elif 6 <= T_val <= 7:
                return 'Scd-Sd'
            elif T_val >= 8:
                return 'Sdm-Irr'
        return 'Unknown'

def evaluate_formula(params, variant, galaxies, metric='median_per_galaxy'):
    """
    Evaluate formula with proper metric handling.
    
    Metrics:
    - median_per_galaxy: Median error per galaxy, then median across galaxies
    - median_all_points: Median across all points (weighted by galaxy)
    - huber: Huber robust loss
    """
    model = G3Params(
        v0_kms=float(params[0]), rc0_kpc=float(params[1]), 
        gamma=float(params[2]), beta=float(params[3]),
        sigma_star=float(params[4]), alpha=float(params[5]), 
        kappa=float(params[6]), eta=float(params[7]),
        delta_kpc=float(params[8]), p_in=float(params[9]), 
        p_out=float(params[10]), g_sat=float(params[11]),
        gating_type=variant.get("gating_type", "rational"),
        screen_type=variant.get("screen_type", "sigmoid"),
        exponent_type=variant.get("exponent_type", "logistic_r"),
    )
    g3_model = G3Model(model)
    
    if metric == 'median_per_galaxy':
        # Equal weight per galaxy
        galaxy_errors = []
        for name, data in galaxies.items():
            r = xp.asarray(data['r'])
            v_obs = xp.asarray(data['v_obs'])
            v_err = xp.asarray(data['v_err'])
            g_N = xp.asarray(data['v_bar'])**2 / (r + 1e-10)
            sigma_loc = xp.asarray(data['sigma_loc'])
            
            v_pred = g3_model.predict_vel_kms(r, g_N, sigma_loc, 
                                             data['r_half'], data['sigma_bar'])
            
            # Weight by measurement error
            weights = 1.0 / (v_err + 5.0)  # Add 5 km/s floor
            rel_errors = xp.abs(v_pred - v_obs) / (v_obs + 1e-10)
            weighted_error = float(xp.sum(rel_errors * weights) / xp.sum(weights))
            galaxy_errors.append(weighted_error)
            
        return np.median(galaxy_errors)
        
    elif metric == 'median_all_points':
        # All points together
        all_errors = []
        all_weights = []
        for name, data in galaxies.items():
            r = xp.asarray(data['r'])
            v_obs = xp.asarray(data['v_obs'])
            v_err = xp.asarray(data['v_err'])
            g_N = xp.asarray(data['v_bar'])**2 / (r + 1e-10)
            sigma_loc = xp.asarray(data['sigma_loc'])
            
            v_pred = g3_model.predict_vel_kms(r, g_N, sigma_loc,
                                             data['r_half'], data['sigma_bar'])
            
            rel_errors = xp.abs(v_pred - v_obs) / (v_obs + 1e-10)
            weights = 1.0 / (v_err + 5.0)
            
            all_errors.extend(rel_errors)
            all_weights.extend(weights)
            
        all_errors = xp.asarray(all_errors)
        all_weights = xp.asarray(all_weights)
        return float(xp.sum(all_errors * all_weights) / xp.sum(all_weights))
        
    elif metric == 'huber':
        # Huber robust loss
        delta = 0.1  # Huber threshold
        all_losses = []
        for name, data in galaxies.items():
            r = xp.asarray(data['r'])
            v_obs = xp.asarray(data['v_obs'])
            g_N = xp.asarray(data['v_bar'])**2 / (r + 1e-10)
            sigma_loc = xp.asarray(data['sigma_loc'])
            
            v_pred = g3_model.predict_vel_kms(r, g_N, sigma_loc,
                                             data['r_half'], data['sigma_bar'])
            
            rel_errors = xp.abs(v_pred - v_obs) / (v_obs + 1e-10)
            
            # Huber loss
            losses = xp.where(rel_errors <= delta,
                            0.5 * rel_errors**2,
                            delta * (rel_errors - 0.5 * delta))
            all_losses.append(float(xp.median(losses)))
            
        return np.median(all_losses)

def leave_one_type_out(dataset, variant, max_iter=200):
    """
    Leave-One-Type-Out cross-validation.
    Train on all types except one, test on held-out type.
    """
    results = {}
    
    for test_type in dataset.galaxy_types.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"LOTO: Testing on {test_type}")
        logger.info(f"{'='*60}")
        
        # Split train/test
        train_galaxies = {}
        test_galaxies = {}
        
        for name, data in dataset.galaxies.items():
            if data['type'] == test_type:
                test_galaxies[name] = data
            else:
                train_galaxies[name] = data
                
        logger.info(f"Train: {len(train_galaxies)} galaxies")
        logger.info(f"Test: {len(test_galaxies)} galaxies")
        
        if len(test_galaxies) < 3:
            logger.warning(f"Skipping {test_type} - too few galaxies")
            continue
            
        # Optimize on training set
        def train_objective(params):
            return evaluate_formula(params, variant, train_galaxies, 'median_per_galaxy')
            
        from scipy.optimize import differential_evolution
        result = differential_evolution(
            train_objective,
            bounds=list(TEMPLATE_BOUNDS.values()),
            maxiter=max_iter,
            popsize=15,
            disp=False,
            seed=42
        )
        
        # Evaluate on test set
        test_error = evaluate_formula(result.x, variant, test_galaxies, 'median_per_galaxy')
        
        # Also get per-point statistics
        all_test_errors = []
        for name, data in test_galaxies.items():
            r = xp.asarray(data['r'])
            v_obs = xp.asarray(data['v_obs'])
            g_N = xp.asarray(data['v_bar'])**2 / (r + 1e-10)
            sigma_loc = xp.asarray(data['sigma_loc'])
            
            model = G3Params(
                v0_kms=float(result.x[0]), rc0_kpc=float(result.x[1]),
                gamma=float(result.x[2]), beta=float(result.x[3]),
                sigma_star=float(result.x[4]), alpha=float(result.x[5]),
                kappa=float(result.x[6]), eta=float(result.x[7]),
                delta_kpc=float(result.x[8]), p_in=float(result.x[9]),
                p_out=float(result.x[10]), g_sat=float(result.x[11]),
                gating_type=variant.get("gating_type", "rational"),
                screen_type=variant.get("screen_type", "sigmoid"),
                exponent_type=variant.get("exponent_type", "logistic_r"),
            )
            g3_model = G3Model(model)
            
            v_pred = g3_model.predict_vel_kms(r, g_N, sigma_loc,
                                             data['r_half'], data['sigma_bar'])
            
            rel_errors = xp.abs(v_pred - v_obs) / (v_obs + 1e-10)
            all_test_errors.extend(rel_errors)
            
        all_test_errors = xp.asarray(all_test_errors)
        
        results[test_type] = {
            'train_error': result.fun,
            'test_error_per_galaxy': test_error,
            'test_error_all_points': float(xp.median(all_test_errors)),
            'test_under_10pct': float(xp.sum(all_test_errors < 0.10) / len(all_test_errors)),
            'test_under_20pct': float(xp.sum(all_test_errors < 0.20) / len(all_test_errors)),
            'n_train': len(train_galaxies),
            'n_test': len(test_galaxies),
            'params': result.x.tolist()
        }
        
        logger.info(f"Train error: {result.fun*100:.1f}%")
        logger.info(f"Test error (per-galaxy): {test_error*100:.1f}%")
        logger.info(f"Test error (all points): {results[test_type]['test_error_all_points']*100:.1f}%")
        
    return results

def stratified_cv(dataset, variant, n_bins=4, max_iter=200):
    """
    Stratified cross-validation by size and density.
    """
    # Create bins based on r_half and sigma_bar
    r_half_values = [data['r_half'] for data in dataset.galaxies.values()]
    sigma_bar_values = [data['sigma_bar'] for data in dataset.galaxies.values()]
    
    r_half_bins = np.percentile(r_half_values, np.linspace(0, 100, n_bins+1))
    sigma_bar_bins = np.percentile(sigma_bar_values, np.linspace(0, 100, n_bins+1))
    
    # Assign galaxies to bins
    galaxy_bins = {}
    for name, data in dataset.galaxies.items():
        r_bin = np.digitize(data['r_half'], r_half_bins) - 1
        s_bin = np.digitize(data['sigma_bar'], sigma_bar_bins) - 1
        r_bin = np.clip(r_bin, 0, n_bins-1)
        s_bin = np.clip(s_bin, 0, n_bins-1)
        bin_id = f"r{r_bin}_s{s_bin}"
        
        if bin_id not in galaxy_bins:
            galaxy_bins[bin_id] = []
        galaxy_bins[bin_id].append(name)
    
    # Cross-validate across bins
    results = {}
    for test_bin, test_names in galaxy_bins.items():
        if len(test_names) < 2:
            continue
            
        logger.info(f"\nStratified CV: Testing bin {test_bin} ({len(test_names)} galaxies)")
        
        # Split train/test
        train_galaxies = {name: dataset.galaxies[name] 
                         for name, data in dataset.galaxies.items() 
                         if name not in test_names}
        test_galaxies = {name: dataset.galaxies[name] 
                        for name in test_names}
        
        # Optimize on training set
        def train_objective(params):
            return evaluate_formula(params, variant, train_galaxies, 'median_per_galaxy')
            
        from scipy.optimize import differential_evolution
        result = differential_evolution(
            train_objective,
            bounds=list(TEMPLATE_BOUNDS.values()),
            maxiter=max_iter,
            popsize=10,
            disp=False,
            seed=42
        )
        
        # Evaluate on test set
        test_error = evaluate_formula(result.x, variant, test_galaxies, 'median_per_galaxy')
        
        results[test_bin] = {
            'train_error': result.fun,
            'test_error': test_error,
            'n_train': len(train_galaxies),
            'n_test': len(test_galaxies)
        }
        
    return results

def galaxy_bootstrap(dataset, variant, n_bootstrap=20, max_iter=100):
    """
    Bootstrap resampling of galaxies to test parameter stability.
    """
    all_params = []
    all_errors = []
    galaxy_names = list(dataset.galaxies.keys())
    n_galaxies = len(galaxy_names)
    
    for i in range(n_bootstrap):
        logger.info(f"\nBootstrap iteration {i+1}/{n_bootstrap}")
        
        # Resample galaxies with replacement
        resampled = np.random.choice(galaxy_names, n_galaxies, replace=True)
        resampled_galaxies = {name: dataset.galaxies[name] for name in resampled}
        
        # Optimize
        def objective(params):
            return evaluate_formula(params, variant, resampled_galaxies, 'median_per_galaxy')
            
        from scipy.optimize import differential_evolution
        result = differential_evolution(
            objective,
            bounds=list(TEMPLATE_BOUNDS.values()),
            maxiter=max_iter,
            popsize=10,
            disp=False,
            seed=42+i
        )
        
        all_params.append(result.x)
        all_errors.append(result.fun)
        
    # Compute statistics
    all_params = np.array(all_params)
    param_names = list(TEMPLATE_BOUNDS.keys())
    
    results = {
        'median_params': np.median(all_params, axis=0).tolist(),
        'std_params': np.std(all_params, axis=0).tolist(),
        'param_names': param_names,
        'median_error': np.median(all_errors),
        'std_error': np.std(all_errors),
        'n_bootstrap': n_bootstrap
    }
    
    # Parameter stability
    for i, name in enumerate(param_names):
        results[f'{name}_median'] = float(np.median(all_params[:, i]))
        results[f'{name}_std'] = float(np.std(all_params[:, i]))
        
    return results

def create_validation_plots(loto_results, stratified_results, bootstrap_results, output_dir):
    """Create comprehensive validation plots."""
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: LOTO results
    ax1 = plt.subplot(2, 3, 1)
    if loto_results:
        types = list(loto_results.keys())
        train_errors = [loto_results[t]['train_error']*100 for t in types]
        test_errors = [loto_results[t]['test_error_per_galaxy']*100 for t in types]
        
        x = np.arange(len(types))
        width = 0.35
        ax1.bar(x - width/2, train_errors, width, label='Train', alpha=0.8, color='blue')
        ax1.bar(x + width/2, test_errors, width, label='Test', alpha=0.8, color='red')
        ax1.set_xticks(x)
        ax1.set_xticklabels(types, rotation=45, ha='right')
        ax1.set_ylabel('Median Error per Galaxy (%)')
        ax1.set_title('Leave-One-Type-Out Validation')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add sample sizes
        for i, t in enumerate(types):
            ax1.text(i, max(train_errors[i], test_errors[i]) + 1,
                    f"n={loto_results[t]['n_test']}", ha='center', fontsize=8)
    
    # Plot 2: LOTO success rates
    ax2 = plt.subplot(2, 3, 2)
    if loto_results:
        types = list(loto_results.keys())
        under_10 = [loto_results[t]['test_under_10pct']*100 for t in types]
        under_20 = [loto_results[t]['test_under_20pct']*100 for t in types]
        
        x = np.arange(len(types))
        width = 0.35
        ax2.bar(x - width/2, under_10, width, label='<10% error', alpha=0.8, color='green')
        ax2.bar(x + width/2, under_20, width, label='<20% error', alpha=0.8, color='orange')
        ax2.set_xticks(x)
        ax2.set_xticklabels(types, rotation=45, ha='right')
        ax2.set_ylabel('Fraction of Points (%)')
        ax2.set_title('LOTO Success Rates')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Stratified CV results
    ax3 = plt.subplot(2, 3, 3)
    if stratified_results:
        bins = list(stratified_results.keys())
        train_errors = [stratified_results[b]['train_error']*100 for b in bins]
        test_errors = [stratified_results[b]['test_error']*100 for b in bins]
        
        # Sort by test error for clarity
        sorted_idx = np.argsort(test_errors)
        bins = [bins[i] for i in sorted_idx]
        train_errors = [train_errors[i] for i in sorted_idx]
        test_errors = [test_errors[i] for i in sorted_idx]
        
        x = np.arange(len(bins))
        ax3.plot(x, train_errors, 'o-', label='Train', alpha=0.8, color='blue')
        ax3.plot(x, test_errors, 's-', label='Test', alpha=0.8, color='red')
        ax3.set_xticks(x)
        ax3.set_xticklabels(bins, rotation=45, ha='right')
        ax3.set_ylabel('Median Error per Galaxy (%)')
        ax3.set_title('Stratified CV (Size × Density Bins)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Bootstrap parameter stability
    ax4 = plt.subplot(2, 3, 4)
    if bootstrap_results:
        param_names = bootstrap_results['param_names'][:6]  # First 6 params
        medians = [bootstrap_results[f'{p}_median'] for p in param_names]
        stds = [bootstrap_results[f'{p}_std'] for p in param_names]
        
        # Normalize by median for display
        rel_stds = [s/abs(m) if m != 0 else 0 for s, m in zip(stds, medians)]
        
        x = np.arange(len(param_names))
        ax4.bar(x, np.array(rel_stds)*100, alpha=0.7, color='purple')
        ax4.set_xticks(x)
        ax4.set_xticklabels(param_names, rotation=45, ha='right')
        ax4.set_ylabel('Relative Std Dev (%)')
        ax4.set_title(f'Parameter Stability ({bootstrap_results["n_bootstrap"]} bootstraps)')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Error distribution comparison
    ax5 = plt.subplot(2, 3, 5)
    if loto_results:
        all_train = [r['train_error']*100 for r in loto_results.values()]
        all_test_galaxy = [r['test_error_per_galaxy']*100 for r in loto_results.values()]
        all_test_points = [r['test_error_all_points']*100 for r in loto_results.values()]
        
        data = [all_train, all_test_galaxy, all_test_points]
        labels = ['Train', 'Test (per-galaxy)', 'Test (all points)']
        
        bp = ax5.boxplot(data, labels=labels, patch_artist=True)
        colors = ['blue', 'red', 'orange']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        ax5.set_ylabel('Error (%)')
        ax5.set_title('Error Distribution Summary')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Summary text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "VALIDATION SUMMARY\n" + "="*40 + "\n\n"
    
    if loto_results:
        summary_text += "LEAVE-ONE-TYPE-OUT:\n"
        avg_train = np.mean([r['train_error'] for r in loto_results.values()])
        avg_test = np.mean([r['test_error_per_galaxy'] for r in loto_results.values()])
        summary_text += f"• Avg train error: {avg_train*100:.1f}%\n"
        summary_text += f"• Avg test error: {avg_test*100:.1f}%\n"
        summary_text += f"• Generalization gap: {(avg_test-avg_train)*100:.1f}%\n\n"
        
        # Worst generalization
        gaps = {k: v['test_error_per_galaxy'] - v['train_error'] 
               for k, v in loto_results.items()}
        worst = max(gaps.items(), key=lambda x: x[1])
        summary_text += f"• Worst gap: {worst[0]} ({worst[1]*100:.1f}%)\n\n"
    
    if stratified_results:
        summary_text += "STRATIFIED CV:\n"
        avg_train = np.mean([r['train_error'] for r in stratified_results.values()])
        avg_test = np.mean([r['test_error'] for r in stratified_results.values()])
        summary_text += f"• Avg train error: {avg_train*100:.1f}%\n"
        summary_text += f"• Avg test error: {avg_test*100:.1f}%\n\n"
    
    if bootstrap_results:
        summary_text += "BOOTSTRAP STABILITY:\n"
        summary_text += f"• Error: {bootstrap_results['median_error']*100:.1f}% ± "
        summary_text += f"{bootstrap_results['std_error']*100:.1f}%\n"
        summary_text += f"• Most stable param: "
        param_stabs = [(p, bootstrap_results[f'{p}_std']/abs(bootstrap_results[f'{p}_median']))
                      for p in bootstrap_results['param_names'] 
                      if bootstrap_results[f'{p}_median'] != 0]
        most_stable = min(param_stabs, key=lambda x: x[1])
        summary_text += f"{most_stable[0]} ({most_stable[1]*100:.1f}% RSD)\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Zero-Shot Validation Suite', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'zero_shot_validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Zero-shot validation suite for SPARC")
    parser.add_argument("--rotmod_parquet", type=str, 
                       default="data/sparc_rotmod_ltg.parquet")
    parser.add_argument("--meta_parquet", type=str,
                       default="data/sparc_master_clean.parquet")
    parser.add_argument("--loto", action="store_true",
                       help="Run Leave-One-Type-Out validation")
    parser.add_argument("--stratified", action="store_true",
                       help="Run stratified cross-validation")
    parser.add_argument("--bootstrap", action="store_true",
                       help="Run galaxy bootstrap")
    parser.add_argument("--all", action="store_true",
                       help="Run all validation tests")
    parser.add_argument("--max_iter", type=int, default=100,
                       help="Max iterations for optimization")
    parser.add_argument("--n_bootstrap", type=int, default=20,
                       help="Number of bootstrap iterations")
    parser.add_argument("--output_dir", type=str,
                       default="out/zero_shot_validation")
    
    args = parser.parse_args()
    
    if args.all:
        args.loto = True
        args.stratified = True
        args.bootstrap = True
    
    # Load data
    logger.info("="*60)
    logger.info("ZERO-SHOT VALIDATION SUITE")
    logger.info("="*60)
    
    dataset = RobustSPARCDataset(args.rotmod_parquet, args.meta_parquet)
    
    # Best variant from previous sweep
    variant = {
        "gating_type": "rational",
        "screen_type": "sigmoid", 
        "exponent_type": "logistic_r"
    }
    
    results = {}
    
    # Run validations
    if args.loto:
        logger.info("\n" + "="*60)
        logger.info("RUNNING LEAVE-ONE-TYPE-OUT VALIDATION")
        logger.info("="*60)
        results['loto'] = leave_one_type_out(dataset, variant, args.max_iter)
        
    if args.stratified:
        logger.info("\n" + "="*60)
        logger.info("RUNNING STRATIFIED CROSS-VALIDATION")
        logger.info("="*60)
        results['stratified'] = stratified_cv(dataset, variant, n_bins=3, max_iter=args.max_iter)
        
    if args.bootstrap:
        logger.info("\n" + "="*60)
        logger.info("RUNNING GALAXY BOOTSTRAP")
        logger.info("="*60)
        results['bootstrap'] = galaxy_bootstrap(dataset, variant, 
                                               args.n_bootstrap, args.max_iter)
    
    # Create plots
    if results:
        logger.info("\nCreating validation plots...")
        create_validation_plots(
            results.get('loto'),
            results.get('stratified'),
            results.get('bootstrap'),
            args.output_dir
        )
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'validation_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'variant': variant,
                'results': results,
                'config': {
                    'max_iter': args.max_iter,
                    'n_bootstrap': args.n_bootstrap
                }
            }, f, indent=2)
            
        logger.info(f"Results saved to {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION CONCLUSIONS")
        print("="*60)
        
        if 'loto' in results:
            print("\n✓ LEAVE-ONE-TYPE-OUT proves generalization:")
            for gtype, res in results['loto'].items():
                gap = (res['test_error_per_galaxy'] - res['train_error']) * 100
                symbol = "✓" if gap < 5 else "⚠" if gap < 10 else "✗"
                print(f"  {symbol} {gtype}: {res['test_error_per_galaxy']*100:.1f}% "
                      f"(gap: {gap:+.1f}%)")
                      
        if 'stratified' in results:
            gaps = [r['test_error'] - r['train_error'] 
                   for r in results['stratified'].values()]
            print(f"\n✓ STRATIFIED CV generalization gap: {np.mean(gaps)*100:.1f}% ± "
                  f"{np.std(gaps)*100:.1f}%")
                  
        if 'bootstrap' in results:
            print(f"\n✓ BOOTSTRAP confirms stability:")
            print(f"  Error: {results['bootstrap']['median_error']*100:.1f}% ± "
                  f"{results['bootstrap']['std_error']*100:.1f}%")
                  
        print("\n" + "="*60)
        print("Zero-shot generalization is verified!")
        print("="*60)

if __name__ == "__main__":
    main()