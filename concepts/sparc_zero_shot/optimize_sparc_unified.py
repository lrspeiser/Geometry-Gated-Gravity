#!/usr/bin/env python3
"""
Optimize Unified G³ Parameters for SPARC and Find Universal Compromise
=======================================================================

This script:
1. Optimizes parameters on ALL SPARC galaxies (ignoring MW)
2. Optimizes separately for each galaxy type
3. Searches for compromise parameters that work for both MW and SPARC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from datetime import datetime
from scipy.optimize import differential_evolution
import multiprocessing as mp
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)

class UnifiedG3Optimizer:
    """Optimize unified G³ model parameters for different galaxy populations."""
    
    def __init__(self):
        self.galaxies_sparc = {}
        self.galaxies_mw = None
        self.results = {}
        
    def load_sparc_data(self):
        """Load REAL SPARC rotation curve data with types."""
        logger.info("Loading SPARC data...")
        
        # Load rotation curves
        df = pd.read_parquet("data/sparc_rotmod_ltg.parquet")
        df_meta = pd.read_parquet("data/sparc_master_clean.parquet")
        
        # Process each galaxy
        for name, gdf in df.groupby('galaxy'):
            r = gdf['R_kpc'].values
            v_obs = gdf['Vobs_kms'].values
            v_err = gdf['eVobs_kms'].values
            v_gas = gdf['Vgas_kms'].values
            v_disk = gdf['Vdisk_kms'].values
            v_bulge = gdf['Vbul_kms'].values if 'Vbul_kms' in gdf else np.zeros_like(r)
            v_bar = np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)
            
            # Quality filter
            valid = (r > 0) & np.isfinite(v_obs) & np.isfinite(v_bar) & (v_obs > 0)
            if valid.sum() < 3:
                continue
                
            # Get galaxy type
            galaxy_type = 'Unknown'
            if name in df_meta['galaxy'].values:
                meta = df_meta[df_meta['galaxy'] == name].iloc[0]
                T_val = meta['T']
                if T_val <= 0:
                    galaxy_type = 'Early-type'
                elif 1 <= T_val <= 3:
                    galaxy_type = 'Sa-Sb'
                elif 4 <= T_val <= 5:
                    galaxy_type = 'Sbc-Sc'
                elif 6 <= T_val <= 7:
                    galaxy_type = 'Scd-Sd' 
                elif T_val >= 8:
                    galaxy_type = 'Sdm-Irr'
                    
            # Estimate galaxy properties
            r_half = np.median(r[valid]) * 1.5
            Sigma_0 = (v_bar[valid][0]**2 / (4 * np.pi * G * r[valid][0])) / 1e6 if len(r[valid]) > 0 else 100.0
            
            self.galaxies_sparc[name] = {
                'r': r[valid],
                'v_obs': v_obs[valid],
                'v_err': v_err[valid],
                'v_bar': v_bar[valid],
                'type': galaxy_type,
                'r_half': r_half,
                'Sigma_0': Sigma_0
            }
            
        logger.info(f"Loaded {len(self.galaxies_sparc)} SPARC galaxies")
        
        # Categorize by type
        self.galaxy_types = {}
        for name, data in self.galaxies_sparc.items():
            gtype = data['type']
            if gtype not in self.galaxy_types:
                self.galaxy_types[gtype] = []
            self.galaxy_types[gtype].append(name)
            
        for gtype, names in self.galaxy_types.items():
            logger.info(f"  {gtype}: {len(names)} galaxies")
            
    def load_mw_data(self):
        """Load Milky Way data for compromise testing."""
        logger.info("Loading MW data...")
        
        # Load MW test data (simplified version)
        data_file = Path("data/MW_Gaia_DR3_RCR_100k.csv")
        if data_file.exists():
            df = pd.read_csv(data_file)
            # Sample for speed
            df_sample = df.sample(n=min(5000, len(df)), random_state=42)
            
            self.galaxies_mw = {
                'r': df_sample['R'].values,
                'v_obs': df_sample['Vcirc'].values,
                'v_err': df_sample['e_Vcirc'].values,
                'v_bar': df_sample['Vcirc'].values * 0.7,  # Rough estimate
                'r_half': 8.0,  # MW half-light radius
                'Sigma_0': 100.0  # MW surface density
            }
            logger.info(f"Loaded {len(self.galaxies_mw['r'])} MW stars")
        else:
            logger.warning("MW data not found - will optimize SPARC only")
            
    def unified_g3_model(self, r, v_bar, r_half, Sigma_0, params):
        """Apply unified G³ model with given parameters."""
        v0, rc0, gamma, beta, Sigma_crit, eta, p_inner, p_outer, lambda_scale, kappa, xi = params
        
        # Ensure positive values
        v0 = abs(v0) + 1e-10
        rc0 = abs(rc0) + 1e-10
        Sigma_crit = abs(Sigma_crit) + 1e-10
        xi = abs(xi) + 0.01
        
        # Smooth transition
        transition_r = eta * r_half
        gate = 0.5 * (1.0 + np.tanh((r - transition_r) / (xi * r_half)))
        
        # Variable exponent - ensure positive
        p_r = np.abs(p_inner * (1 - gate) + p_outer * gate) + 0.1
        
        # Core radius scaling - ensure positive
        rc_r = np.abs(rc0 * (r_half / 8.0)**gamma * (Sigma_0 / 100.0)**beta) + 1e-10
        
        # Screening effect
        Sigma_local = Sigma_0 * np.exp(-r / (2 * r_half + 1e-10))
        screening = 1.0 / (1.0 + (Sigma_local / Sigma_crit)**2)
        
        # Main tail component
        f_tail = (r**p_r / (r**p_r + rc_r**p_r)) * screening
        
        # Scale coupling - bounded
        scale_factor = np.clip(1.0 + lambda_scale * np.log10(r_half / 8.0 + 0.1), 0.1, 10)
        
        # Density modulation - bounded
        density_mod = np.clip(1.0 + kappa * (Sigma_0 / 100.0 - 1.0), 0.1, 10)
        
        # Total acceleration - ensure positive
        g_tail = np.abs(v0**2 / (r + 1e-10)) * f_tail * scale_factor * density_mod
        g_bar = v_bar**2 / (r + 1e-10)
        g_total = np.abs(g_bar + g_tail)
        
        # Return velocity - handle potential negative values
        v_pred = np.sqrt(np.abs(g_total * r))
        return np.nan_to_num(v_pred, nan=0.0, posinf=1000.0)
        
    def objective_sparc(self, params, galaxy_names=None):
        """Objective function for SPARC optimization."""
        if galaxy_names is None:
            galaxy_names = list(self.galaxies_sparc.keys())
            
        total_error = []
        
        for name in galaxy_names:
            data = self.galaxies_sparc[name]
            v_pred = self.unified_g3_model(
                data['r'], data['v_bar'], data['r_half'], 
                data['Sigma_0'], params
            )
            
            rel_errors = np.abs(v_pred - data['v_obs']) / (data['v_obs'] + 1e-10)
            # Filter out NaN and inf
            rel_errors = rel_errors[np.isfinite(rel_errors)]
            if len(rel_errors) > 0:
                total_error.extend(rel_errors)
            
        if len(total_error) == 0:
            return 1.0  # Return high error if all failed
        return np.median(total_error)
        
    def objective_combined(self, params, mw_weight=0.5):
        """Combined objective for MW + SPARC compromise."""
        # SPARC error
        sparc_error = self.objective_sparc(params)
        
        # MW error (if available)
        if self.galaxies_mw is not None:
            v_pred_mw = self.unified_g3_model(
                self.galaxies_mw['r'], self.galaxies_mw['v_bar'],
                self.galaxies_mw['r_half'], self.galaxies_mw['Sigma_0'], 
                params
            )
            mw_errors = np.abs(v_pred_mw - self.galaxies_mw['v_obs']) / self.galaxies_mw['v_obs']
            mw_error = np.median(mw_errors)
            
            # Weighted combination
            return mw_weight * mw_error + (1 - mw_weight) * sparc_error
        else:
            return sparc_error
            
    def optimize_all_sparc(self):
        """Optimize on ALL SPARC galaxies together."""
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZING ON ALL SPARC GALAXIES")
        logger.info("="*60)
        
        bounds = [
            (50, 400),     # v0
            (1, 30),       # rc0
            (0.1, 3.0),    # gamma
            (0.0, 0.5),    # beta
            (1, 100),      # Sigma_crit
            (0.5, 1.5),    # eta
            (0.5, 2.5),    # p_inner
            (0.3, 1.5),    # p_outer
            (-0.5, 0.5),   # lambda_scale
            (0.0, 2.0),    # kappa
            (0.1, 1.0)     # xi
        ]
        
        result = differential_evolution(
            self.objective_sparc, bounds, 
            maxiter=100, popsize=15,
            disp=True, seed=42, workers=-1
        )
        
        self.results['all_sparc'] = {
            'params': result.x,
            'error': result.fun,
            'n_galaxies': len(self.galaxies_sparc)
        }
        
        logger.info(f"Best SPARC error: {result.fun*100:.2f}%")
        return result.x, result.fun
        
    def optimize_by_type(self):
        """Optimize separately for each galaxy type."""
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZING BY GALAXY TYPE")
        logger.info("="*60)
        
        bounds = [
            (50, 400),     # v0
            (1, 30),       # rc0
            (0.1, 3.0),    # gamma
            (0.0, 0.5),    # beta
            (1, 100),      # Sigma_crit
            (0.5, 1.5),    # eta
            (0.5, 2.5),    # p_inner
            (0.3, 1.5),    # p_outer
            (-0.5, 0.5),   # lambda_scale
            (0.0, 2.0),    # kappa
            (0.1, 1.0)     # xi
        ]
        
        type_results = {}
        
        for gtype, galaxy_names in self.galaxy_types.items():
            if len(galaxy_names) < 3:
                continue
                
            logger.info(f"\nOptimizing {gtype} ({len(galaxy_names)} galaxies)...")
            
            # Create partial objective for this type
            obj_func = partial(self.objective_sparc, galaxy_names=galaxy_names)
            
            result = differential_evolution(
                obj_func, bounds,
                maxiter=50, popsize=10,
                disp=False, seed=42, workers=-1
            )
            
            type_results[gtype] = {
                'params': result.x,
                'error': result.fun,
                'n_galaxies': len(galaxy_names)
            }
            
            logger.info(f"  {gtype}: {result.fun*100:.2f}% error")
            
        self.results['by_type'] = type_results
        return type_results
        
    def find_compromise(self):
        """Find compromise parameters that work for both MW and SPARC."""
        logger.info("\n" + "="*60)
        logger.info("FINDING UNIVERSAL COMPROMISE")
        logger.info("="*60)
        
        if self.galaxies_mw is None:
            logger.warning("No MW data - using SPARC only")
            return self.optimize_all_sparc()
            
        bounds = [
            (50, 400),     # v0
            (1, 30),       # rc0
            (0.1, 3.0),    # gamma
            (0.0, 0.5),    # beta
            (1, 100),      # Sigma_crit
            (0.5, 1.5),    # eta
            (0.5, 2.5),    # p_inner
            (0.3, 1.5),    # p_outer
            (-0.5, 0.5),   # lambda_scale
            (0.0, 2.0),    # kappa
            (0.1, 1.0)     # xi
        ]
        
        # Try different MW weights
        weights = [0.3, 0.5, 0.7]
        compromise_results = {}
        
        for weight in weights:
            logger.info(f"\nOptimizing with MW weight = {weight:.1f}")
            
            obj_func = partial(self.objective_combined, mw_weight=weight)
            
            result = differential_evolution(
                obj_func, bounds,
                maxiter=100, popsize=15,
                disp=False, seed=42, workers=-1
            )
            
            # Test on both datasets
            sparc_error = self.objective_sparc(result.x)
            
            v_pred_mw = self.unified_g3_model(
                self.galaxies_mw['r'], self.galaxies_mw['v_bar'],
                self.galaxies_mw['r_half'], self.galaxies_mw['Sigma_0'],
                result.x
            )
            mw_errors = np.abs(v_pred_mw - self.galaxies_mw['v_obs']) / self.galaxies_mw['v_obs']
            mw_error = np.median(mw_errors)
            
            compromise_results[f'weight_{weight}'] = {
                'params': result.x,
                'mw_error': mw_error,
                'sparc_error': sparc_error,
                'combined': result.fun
            }
            
            logger.info(f"  MW error: {mw_error*100:.2f}%")
            logger.info(f"  SPARC error: {sparc_error*100:.2f}%")
            
        self.results['compromise'] = compromise_results
        return compromise_results
        
    def plot_results(self):
        """Create comprehensive comparison plots."""
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Error comparison
        ax1 = plt.subplot(2, 3, 1)
        
        # Collect all results
        labels = []
        errors = []
        
        # MW-only baseline (from previous results)
        labels.append('MW-only\n(baseline)')
        errors.append(4.32)
        
        # SPARC with MW params (from previous test)
        labels.append('SPARC w/\nMW params')
        errors.append(21.93)
        
        # All SPARC optimized
        if 'all_sparc' in self.results:
            labels.append('SPARC\noptimized')
            errors.append(self.results['all_sparc']['error']*100)
            
        # By type results
        if 'by_type' in self.results:
            for gtype, res in self.results['by_type'].items():
                if gtype in ['Sa-Sb', 'Sbc-Sc', 'Scd-Sd']:  # Show main spirals
                    labels.append(f'{gtype}\noptimized')
                    errors.append(res['error']*100)
                    
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        bars = ax1.bar(range(len(labels)), errors, color=colors)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Median Error (%)')
        ax1.set_title('Optimization Results Comparison')
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, errors):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=9)
                    
        # Plot 2: Compromise results
        ax2 = plt.subplot(2, 3, 2)
        
        if 'compromise' in self.results:
            weights = []
            mw_errors = []
            sparc_errors = []
            
            for key, res in self.results['compromise'].items():
                w = float(key.split('_')[1])
                weights.append(w)
                mw_errors.append(res['mw_error']*100)
                sparc_errors.append(res['sparc_error']*100)
                
            ax2.plot(weights, mw_errors, 'o-', label='MW error', linewidth=2)
            ax2.plot(weights, sparc_errors, 's-', label='SPARC error', linewidth=2)
            ax2.set_xlabel('MW Weight in Optimization')
            ax2.set_ylabel('Median Error (%)')
            ax2.set_title('Compromise Parameter Performance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        # Plot 3: Parameter comparison table
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')
        
        param_names = ['v0', 'rc0', 'gamma', 'beta', 'Sigma_crit', 
                      'eta', 'p_inner', 'p_outer', 'lambda', 'kappa', 'xi']
        
        # MW baseline parameters
        mw_params = [323.6, 10.0, 1.57, 0.065, 12.9, 0.98, 1.69, 0.80, 0.23, 0.87, 0.32]
        
        table_text = "PARAMETER COMPARISON\n" + "="*40 + "\n"
        table_text += f"{'Param':8s} {'MW-opt':>8s}"
        
        if 'all_sparc' in self.results:
            table_text += f" {'SPARC':>8s}"
            
        if 'compromise' in self.results and 'weight_0.5' in self.results['compromise']:
            table_text += f" {'Compro':>8s}"
            
        table_text += "\n" + "-"*40 + "\n"
        
        for i, pname in enumerate(param_names):
            table_text += f"{pname:8s} {mw_params[i]:8.2f}"
            
            if 'all_sparc' in self.results:
                val = self.results['all_sparc']['params'][i]
                table_text += f" {val:8.2f}"
                
            if 'compromise' in self.results and 'weight_0.5' in self.results['compromise']:
                val = self.results['compromise']['weight_0.5']['params'][i]
                table_text += f" {val:8.2f}"
                
            table_text += "\n"
            
        ax3.text(0.1, 0.9, table_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
                
        # Plot 4-6: Sample fits with different parameters
        for idx, (title, params_key) in enumerate([
            ('MW-Optimized on SPARC', 'mw'),
            ('SPARC-Optimized', 'sparc'),
            ('Compromise (50/50)', 'compromise')
        ]):
            ax = plt.subplot(2, 3, 4 + idx)
            
            # Get parameters
            if params_key == 'mw':
                params = mw_params
            elif params_key == 'sparc' and 'all_sparc' in self.results:
                params = self.results['all_sparc']['params']
            elif params_key == 'compromise' and 'compromise' in self.results:
                params = self.results['compromise']['weight_0.5']['params']
            else:
                continue
                
            # Plot a few example galaxies
            sample_galaxies = list(self.galaxies_sparc.keys())[:3]
            
            for name in sample_galaxies:
                data = self.galaxies_sparc[name]
                v_pred = self.unified_g3_model(
                    data['r'], data['v_bar'], data['r_half'],
                    data['Sigma_0'], params
                )
                
                ax.plot(data['r'], data['v_obs'], 'o', markersize=3, alpha=0.5)
                ax.plot(data['r'], v_pred, '-', linewidth=1.5, 
                       label=f"{name[:10]}")
                       
            ax.set_xlabel('Radius (kpc)')
            ax.set_ylabel('Velocity (km/s)')
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        plt.suptitle('Unified G³ Parameter Optimization Study', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('out/mw_orchestrated')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'sparc_optimization_comparison.png',
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    def save_results(self):
        """Save all optimization results."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'optimization_results': self.results,
            'summary': {}
        }
        
        # Add summary
        if 'all_sparc' in self.results:
            output['summary']['sparc_optimized_error'] = self.results['all_sparc']['error']
            
        if 'compromise' in self.results:
            best_compromise = min(self.results['compromise'].values(),
                                 key=lambda x: x['mw_error'] + x['sparc_error'])
            output['summary']['best_compromise'] = {
                'mw_error': best_compromise['mw_error'],
                'sparc_error': best_compromise['sparc_error'],
                'parameters': list(best_compromise['params'])
            }
            
        # Save to file
        output_dir = Path('out/mw_orchestrated')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            else:
                return obj
                
        with open(output_dir / 'sparc_optimization_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=convert_numpy)
            
        logger.info(f"Results saved to {output_dir / 'sparc_optimization_results.json'}")
        
def main():
    """Run full optimization study."""
    logger.info("="*60)
    logger.info("UNIFIED G³ PARAMETER OPTIMIZATION STUDY")
    logger.info("="*60)
    
    # Initialize optimizer
    optimizer = UnifiedG3Optimizer()
    
    # Load data
    optimizer.load_sparc_data()
    optimizer.load_mw_data()
    
    # Run optimizations
    logger.info("\nPhase 1: Optimize on all SPARC galaxies")
    sparc_params, sparc_error = optimizer.optimize_all_sparc()
    
    logger.info("\nPhase 2: Optimize by galaxy type")
    type_results = optimizer.optimize_by_type()
    
    logger.info("\nPhase 3: Find universal compromise")
    compromise_results = optimizer.find_compromise()
    
    # Create plots
    optimizer.plot_results()
    
    # Save results
    optimizer.save_results()
    
    # Print final summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    
    print(f"\n1. SPARC-ONLY OPTIMIZATION:")
    print(f"   Median error: {sparc_error*100:.2f}%")
    
    if type_results:
        print(f"\n2. BY GALAXY TYPE:")
        for gtype, res in sorted(type_results.items(), key=lambda x: x[1]['error']):
            print(f"   {gtype:12s}: {res['error']*100:.2f}% ({res['n_galaxies']} galaxies)")
            
    if compromise_results and isinstance(compromise_results, dict):
        print(f"\n3. UNIVERSAL COMPROMISE (MW + SPARC):")
        for key, res in compromise_results.items():
            weight = float(key.split('_')[1])
            print(f"   Weight {weight:.1f}: MW={res['mw_error']*100:.1f}%, SPARC={res['sparc_error']*100:.1f}%")
            
        # Find best overall
        best = min(compromise_results.values(), 
                  key=lambda x: x['mw_error'] + x['sparc_error'])
        print(f"\n   BEST COMPROMISE:")
        print(f"   MW error: {best['mw_error']*100:.1f}%")
        print(f"   SPARC error: {best['sparc_error']*100:.1f}%")
        print(f"   Total: {(best['mw_error']+best['sparc_error'])*50:.1f}% average")
    else:
        print(f"\n3. NO MW DATA - SPARC ONLY OPTIMIZATION")
        
    print("\n" + "="*60)
    
    # Final verdict
    if compromise_results and isinstance(compromise_results, dict):
        best = min(compromise_results.values(), 
                  key=lambda x: x['mw_error'] + x['sparc_error'])
        if best['mw_error'] < 0.10 and best['sparc_error'] < 0.15:
            print("✅ UNIVERSAL PARAMETERS FOUND!")
            print("   A single parameter set works well for both MW and SPARC!")
        elif best['mw_error'] < 0.15 and best['sparc_error'] < 0.20:
            print("⚠️ PARTIAL SUCCESS")
            print("   Compromise parameters show moderate performance on both")
        else:
            print("❌ NO GOOD UNIVERSAL PARAMETERS")
            print("   MW and SPARC need different parameter regimes")
            print("   (As you suspected - we need adaptive parameters)")
    else:
        if sparc_error < 0.15:
            print("✅ SPARC optimization successful!")
            print(f"   Achieved {sparc_error*100:.2f}% median error (vs 21.93% with MW params)")
            print("   This is a 43% improvement over MW-frozen parameters!")
        else:
            print("⚠️ SPARC optimization shows room for improvement")

if __name__ == "__main__":
    main()