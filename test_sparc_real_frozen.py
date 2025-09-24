#!/usr/bin/env python3
"""
Test Unified G³ Model on REAL SPARC Data with Frozen MW Parameters
===================================================================

This tests the unified model with frozen MW-optimized parameters on actual SPARC
rotation curves - TRUE zero-shot generalization test with no parameter tuning.

You're absolutely right - we need to test on REAL data to see if the model
actually generalizes or if we just backed into MW-specific parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from datetime import datetime
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants
G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)

class UnifiedG3RealSPARCTest:
    """Test unified G³ model on REAL SPARC galaxies with frozen MW parameters."""
    
    def __init__(self):
        # FROZEN MW-optimized parameters (4.32% error on MW)
        self.frozen_params = {
            'v0': 323.6,      # km/s - asymptotic velocity 
            'rc0': 10.0,      # kpc - reference core radius
            'gamma': 1.57,    # size scaling
            'beta': 0.065,    # density scaling  
            'Sigma_crit': 12.9,  # M_sun/pc^2 - screening threshold
            'eta': 0.98,      # transition at r_half
            'p_inner': 1.69,  # inner exponent
            'p_outer': 0.80,  # outer exponent
            'lambda_scale': 0.23,  # scale coupling
            'kappa': 0.87,    # density modulation
            'xi': 0.32        # transition sharpness
        }
        
        # Create hash for tracking
        param_str = json.dumps(self.frozen_params, sort_keys=True)
        self.param_hash = hashlib.md5(param_str.encode()).hexdigest()[:16]
        
        self.galaxies = {}
        self.results = {}
        
    def load_real_sparc_data(self):
        """Load ACTUAL SPARC rotation curve data."""
        logger.info("Loading REAL SPARC data...")
        
        # Load the real SPARC data
        data_file = Path("data/sparc_rotmod_ltg.parquet")
        meta_file = Path("data/sparc_master_clean.parquet")
        if not data_file.exists():
            raise FileNotFoundError(f"SPARC data not found at {data_file}")
            
        df = pd.read_parquet(data_file)
        # Load metadata for galaxy types
        df_meta = pd.read_parquet(meta_file) if meta_file.exists() else None
        logger.info(f"Loaded {len(df)} data points from {len(df['galaxy'].unique())} galaxies")
        
        # Process each galaxy
        galaxy_groups = df.groupby('galaxy')
        
        for name, gdf in galaxy_groups:
            # Extract data
            r = gdf['R_kpc'].values
            v_obs = gdf['Vobs_kms'].values
            v_err = gdf['eVobs_kms'].values
            
            # Baryonic components
            v_gas = gdf['Vgas_kms'].values
            v_disk = gdf['Vdisk_kms'].values
            v_bulge = gdf['Vbul_kms'].values if 'Vbul_kms' in gdf else np.zeros_like(r)
            
            # Total baryonic velocity
            v_bar = np.sqrt(v_gas**2 + v_disk**2 + v_bulge**2)
            
            # Quality filter
            valid = (r > 0) & np.isfinite(v_obs) & np.isfinite(v_bar) & (v_obs > 0)
            if valid.sum() < 3:
                continue
                
            # Get galaxy properties from metadata (if available)
            if df_meta is not None and name in df_meta['galaxy'].values:
                meta = df_meta[df_meta['galaxy'] == name].iloc[0]
                T_val = meta['T'] if 'T' in meta else 99
                # Classify by Hubble type
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
                else:
                    galaxy_type = 'Unknown'
                distance = meta['D'] if 'D' in meta else np.nan
            else:
                galaxy_type = 'Unknown'
                distance = np.nan
            
            # Estimate r_half (half-light radius) - critical for the model
            r_half = np.median(r[valid]) * 1.5  # Rough estimate
            
            # Estimate surface density
            if len(r[valid]) > 0:
                # Rough proxy from rotation curve
                Sigma_0 = (v_bar[valid][0]**2 / (4 * np.pi * G * r[valid][0])) / 1e6  # M_sun/pc^2
            else:
                Sigma_0 = 100.0  # Default
            
            self.galaxies[name] = {
                'r': r[valid],
                'v_obs': v_obs[valid],
                'v_err': v_err[valid],
                'v_bar': v_bar[valid],
                'v_gas': v_gas[valid],
                'v_disk': v_disk[valid],
                'v_bulge': v_bulge[valid] if 'Vbul_kms' in gdf else np.zeros(valid.sum()),
                'type': galaxy_type,
                'distance': distance,
                'r_half': r_half,
                'Sigma_0': Sigma_0
            }
            
        logger.info(f"Processed {len(self.galaxies)} galaxies with valid data")
        
        # Categorize by type
        self.galaxy_types = {}
        for name, data in self.galaxies.items():
            gtype = data['type']
            if gtype not in self.galaxy_types:
                self.galaxy_types[gtype] = []
            self.galaxy_types[gtype].append(name)
            
        logger.info("Galaxy types found:")
        for gtype, names in self.galaxy_types.items():
            logger.info(f"  {gtype}: {len(names)} galaxies")
            
    def unified_g3_model(self, r, v_bar, r_half, Sigma_0):
        """
        Apply the FROZEN unified G³ model - NO TUNING ALLOWED!
        
        This is the exact same model that achieved 4.32% on MW.
        """
        params = self.frozen_params
        
        # Smooth transition based on r_half
        transition_r = params['eta'] * r_half
        gate = 0.5 * (1.0 + np.tanh((r - transition_r) / (params['xi'] * r_half)))
        
        # Variable exponent - transitions from inner to outer
        p_r = params['p_inner'] * (1 - gate) + params['p_outer'] * gate
        
        # Core radius scaling with galaxy properties
        rc_r = params['rc0'] * (r_half / 8.0)**params['gamma'] * \
               (Sigma_0 / 100.0)**params['beta']
        
        # Screening effect from surface density
        Sigma_local = Sigma_0 * np.exp(-r / (2 * r_half))  # Exponential disk approximation
        screening = 1.0 / (1.0 + (Sigma_local / params['Sigma_crit'])**2)
        
        # Main tail component with variable profile
        f_tail = (r**p_r / (r**p_r + rc_r**p_r)) * screening
        
        # Scale coupling
        scale_factor = 1.0 + params['lambda_scale'] * np.log10(r_half / 8.0)
        
        # Density modulation 
        density_mod = 1.0 + params['kappa'] * (Sigma_0 / 100.0 - 1.0)
        
        # Total tail acceleration
        g_tail = (params['v0']**2 / r) * f_tail * scale_factor * density_mod
        
        # Combine with baryonic
        g_bar = v_bar**2 / r
        g_total = g_bar + g_tail
        
        # Return total velocity
        return np.sqrt(g_total * r)
        
    def test_galaxy(self, name):
        """Test a single galaxy with frozen parameters."""
        data = self.galaxies[name]
        
        # Predict with frozen model
        v_pred = self.unified_g3_model(
            data['r'], 
            data['v_bar'],
            data['r_half'],
            data['Sigma_0']
        )
        
        # Calculate errors
        residuals = v_pred - data['v_obs']
        rel_errors = np.abs(residuals) / data['v_obs']
        chi2 = np.sum((residuals / data['v_err'])**2) / len(residuals)
        
        return {
            'v_pred': v_pred,
            'residuals': residuals,
            'rel_errors': rel_errors,
            'median_error': np.median(rel_errors),
            'mean_error': np.mean(rel_errors),
            'chi2_reduced': chi2
        }
        
    def run_full_test(self):
        """Test all galaxies with frozen parameters."""
        logger.info(f"\nTesting with FROZEN parameters (hash: {self.param_hash})")
        logger.info("=" * 60)
        
        all_errors = []
        results_by_type = {}
        
        for name in self.galaxies:
            result = self.test_galaxy(name)
            self.results[name] = result
            all_errors.extend(result['rel_errors'])
            
            # Track by type
            gtype = self.galaxies[name]['type']
            if gtype not in results_by_type:
                results_by_type[gtype] = []
            results_by_type[gtype].append(result['median_error'])
            
        # Overall statistics
        overall_stats = {
            'median_error': np.median(all_errors),
            'mean_error': np.mean(all_errors),
            'std_error': np.std(all_errors),
            'n_galaxies': len(self.galaxies),
            'n_points': len(all_errors),
            'under_5pct': np.sum(np.array(all_errors) < 0.05) / len(all_errors),
            'under_10pct': np.sum(np.array(all_errors) < 0.10) / len(all_errors),
            'under_20pct': np.sum(np.array(all_errors) < 0.20) / len(all_errors)
        }
        
        # Statistics by type
        type_stats = {}
        for gtype, errors in results_by_type.items():
            type_stats[gtype] = {
                'median_error': np.median(errors),
                'mean_error': np.mean(errors),
                'n_galaxies': len(errors)
            }
            
        return overall_stats, type_stats
        
    def plot_results(self, overall_stats, type_stats):
        """Create comprehensive plots of results."""
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Error distribution
        ax1 = plt.subplot(2, 3, 1)
        all_errors = []
        for result in self.results.values():
            all_errors.extend(result['rel_errors'] * 100)
        
        ax1.hist(all_errors, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(overall_stats['median_error']*100, color='red', 
                   linestyle='--', label=f"Median: {overall_stats['median_error']*100:.1f}%")
        ax1.set_xlabel('Relative Error (%)')
        ax1.set_ylabel('Count')
        ax1.set_title('Error Distribution - ALL SPARC Galaxies')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error by galaxy type
        ax2 = plt.subplot(2, 3, 2)
        types = list(type_stats.keys())
        medians = [type_stats[t]['median_error']*100 for t in types]
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
        
        bars = ax2.bar(range(len(types)), medians, color=colors)
        ax2.set_xticks(range(len(types)))
        ax2.set_xticklabels(types, rotation=45, ha='right')
        ax2.set_ylabel('Median Error (%)')
        ax2.set_title('Performance by Galaxy Type')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, medians):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=9)
        
        # Plot 3: Sample fits - best cases
        ax3 = plt.subplot(2, 3, 3)
        # Find best performing galaxies
        sorted_galaxies = sorted(self.results.items(), 
                               key=lambda x: x[1]['median_error'])[:3]
        
        for i, (name, result) in enumerate(sorted_galaxies):
            data = self.galaxies[name]
            ax3.errorbar(data['r'], data['v_obs'], yerr=data['v_err'],
                        fmt='o', markersize=3, alpha=0.6, label=f'{name} (obs)')
            ax3.plot(data['r'], result['v_pred'], '-', linewidth=2,
                    label=f'{name} ({result["median_error"]*100:.1f}% err)')
        
        ax3.set_xlabel('Radius (kpc)')
        ax3.set_ylabel('Velocity (km/s)')
        ax3.set_title('Best Fits (Lowest Error)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sample fits - worst cases
        ax4 = plt.subplot(2, 3, 4)
        sorted_galaxies = sorted(self.results.items(), 
                               key=lambda x: x[1]['median_error'], reverse=True)[:3]
        
        for i, (name, result) in enumerate(sorted_galaxies):
            data = self.galaxies[name]
            ax4.errorbar(data['r'], data['v_obs'], yerr=data['v_err'],
                        fmt='o', markersize=3, alpha=0.6, label=f'{name} (obs)')
            ax4.plot(data['r'], result['v_pred'], '-', linewidth=2,
                    label=f'{name} ({result["median_error"]*100:.1f}% err)')
        
        ax4.set_xlabel('Radius (kpc)')
        ax4.set_ylabel('Velocity (km/s)')
        ax4.set_title('Worst Fits (Highest Error)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Error vs galaxy properties
        ax5 = plt.subplot(2, 3, 5)
        errors = [self.results[name]['median_error']*100 for name in self.galaxies]
        r_maxs = [np.max(self.galaxies[name]['r']) for name in self.galaxies]
        v_maxs = [np.max(self.galaxies[name]['v_obs']) for name in self.galaxies]
        
        scatter = ax5.scatter(r_maxs, errors, c=v_maxs, cmap='viridis', alpha=0.6)
        ax5.set_xlabel('Max Radius (kpc)')
        ax5.set_ylabel('Median Error (%)')
        ax5.set_title('Error vs Galaxy Size')
        ax5.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax5, label='Max V (km/s)')
        
        # Plot 6: Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
FROZEN PARAMETER TEST RESULTS
{'='*40}
Parameters Hash: {self.param_hash}
Source: MW optimization (4.32% error)

OVERALL STATISTICS:
• Galaxies tested: {overall_stats['n_galaxies']}
• Total data points: {overall_stats['n_points']}
• Median error: {overall_stats['median_error']*100:.2f}%
• Mean error: {overall_stats['mean_error']*100:.2f}%
• Std deviation: {overall_stats['std_error']*100:.2f}%

ERROR THRESHOLDS:
• < 5% error: {overall_stats['under_5pct']*100:.1f}% of points
• < 10% error: {overall_stats['under_10pct']*100:.1f}% of points  
• < 20% error: {overall_stats['under_20pct']*100:.1f}% of points

TOP PERFORMING TYPES:
"""
        # Add top types
        sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['median_error'])[:3]
        for gtype, stats in sorted_types:
            summary_text += f"• {gtype}: {stats['median_error']*100:.1f}% ({stats['n_galaxies']} galaxies)\n"
            
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Unified G³ Model - REAL SPARC Test (Frozen MW Parameters)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('out/mw_orchestrated')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'sparc_real_frozen_test_results.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
        
    def save_results(self, overall_stats, type_stats):
        """Save detailed results to JSON."""
        output = {
            'test_type': 'REAL_SPARC_FROZEN_PARAMS',
            'timestamp': datetime.now().isoformat(),
            'parameters_hash': self.param_hash,
            'frozen_parameters': self.frozen_params,
            'source': 'MW optimization (4.32% median error)',
            'overall_statistics': overall_stats,
            'type_statistics': type_stats,
            'individual_galaxies': {
                name: {
                    'median_error': result['median_error'],
                    'mean_error': result['mean_error'],
                    'chi2_reduced': result['chi2_reduced'],
                    'type': self.galaxies[name]['type'],
                    'n_points': len(self.galaxies[name]['r'])
                }
                for name, result in self.results.items()
            }
        }
        
        output_dir = Path('out/mw_orchestrated')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'sparc_real_frozen_test_results.json', 'w') as f:
            json.dump(output, f, indent=2)
            
        logger.info(f"Results saved to {output_dir / 'sparc_real_frozen_test_results.json'}")
        
def main():
    """Run the real SPARC test with frozen MW parameters."""
    logger.info("="*60)
    logger.info("TESTING UNIFIED G³ ON REAL SPARC DATA")
    logger.info("Using FROZEN MW-optimized parameters (NO TUNING)")
    logger.info("="*60)
    
    # Initialize tester
    tester = UnifiedG3RealSPARCTest()
    
    # Load real SPARC data
    tester.load_real_sparc_data()
    
    # Run tests
    overall_stats, type_stats = tester.run_full_test()
    
    # Print results
    print("\n" + "="*60)
    print("REAL SPARC TEST RESULTS (FROZEN PARAMETERS)")
    print("="*60)
    print(f"Median Error: {overall_stats['median_error']*100:.2f}%")
    print(f"Mean Error: {overall_stats['mean_error']*100:.2f}%")
    print(f"Galaxies Tested: {overall_stats['n_galaxies']}")
    print(f"Total Points: {overall_stats['n_points']}")
    print(f"\nFraction with error < 10%: {overall_stats['under_10pct']*100:.1f}%")
    print(f"Fraction with error < 20%: {overall_stats['under_20pct']*100:.1f}%")
    
    print("\nBy Galaxy Type:")
    for gtype, stats in sorted(type_stats.items(), key=lambda x: x[1]['median_error']):
        print(f"  {gtype:15s}: {stats['median_error']*100:5.1f}% median error ({stats['n_galaxies']} galaxies)")
    
    print("\nConclusion:")
    if overall_stats['median_error'] < 0.15:
        print("✅ Model GENERALIZES WELL to SPARC without retuning!")
    elif overall_stats['median_error'] < 0.30:
        print("⚠️ Model shows moderate generalization - some adjustment needed")
    else:
        print("❌ Model does NOT generalize well - parameters are MW-specific")
        print("   (As you suspected - we backed into MW-specific numbers)")
    
    # Create plots
    tester.plot_results(overall_stats, type_stats)
    
    # Save results
    tester.save_results(overall_stats, type_stats)
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()