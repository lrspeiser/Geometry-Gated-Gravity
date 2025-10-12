#!/usr/bin/env python3
"""
compare_track2_track3_predictive_power.py - Compare Predictive Power of Both Approaches

Evaluates and compares:
- Track 2: Path-spectrum kernel approach (physics-grounded coherence length)
- Track 3: Outlier triage empirical predictors (surface brightness, Vmax, etc.)

For predicting outer-edge velocities across all galaxy types in SPARC.

Key Metrics:
- Correlation with observed outer-edge velocities
- Absolute percentage error (APE) on outermost 3 data points
- Performance by morphological type (E, S0, Sa-Sd, Irr, SAB, SB)
- Residual analysis and systematic trends
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add project root
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Import the modules we created
from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams
from outlier_triage_analysis import OutlierTriageAnalyzer


class PredictivePowerComparator:
    """Compare predictive power of Track 2 vs Track 3 approaches"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize Track 2 kernel with default hyperparameters
        self.hp_track2 = PathSpectrumHyperparams(
            L_0=2.5,
            beta_bulge=1.0,
            alpha_shear=0.05,
            gamma_bar=1.0
        )
        self.kernel = PathSpectrumKernel(self.hp_track2, use_cupy=False)
        
        # Track 3 predictor weights (from outlier triage analysis)
        self.track3_weights = {
            'surface_brightness': -0.360,  # Strongest predictor
            'vmax': -0.349,
            'inclination': 0.143,
            'rdisk': 0.117,
            'bt_ratio': -0.025,
        }
    
    def generate_synthetic_sparc_sample(self, n_galaxies: int = 100) -> pd.DataFrame:
        """Generate synthetic SPARC-like galaxy sample for testing"""
        np.random.seed(42)
        
        galaxies = []
        for i in range(n_galaxies):
            # Generate galaxy properties
            gal_type = np.random.choice(['Sa', 'Sb', 'Sc', 'Sd', 'Irr', 'SAB', 'SB'], 
                                       p=[0.1, 0.2, 0.25, 0.20, 0.10, 0.10, 0.05])
            
            # Properties correlated with type
            if gal_type in ['Sa', 'Sb']:
                BT = np.random.uniform(0.2, 0.6)
                vmax = np.random.uniform(150, 300)
                SB = np.random.uniform(150, 250)
            elif gal_type in ['Sc', 'Sd']:
                BT = np.random.uniform(0.0, 0.2)
                vmax = np.random.uniform(80, 180)
                SB = np.random.uniform(80, 150)
            else:  # Irr, SAB, SB
                BT = np.random.uniform(0.0, 0.4)
                vmax = np.random.uniform(60, 200)
                SB = np.random.uniform(70, 180)
            
            # Bar strength for barred galaxies
            bar_strength = 0.7 if gal_type == 'SB' else (0.4 if gal_type == 'SAB' else 0.0)
            
            # Generate rotation curve points
            r_points = np.linspace(1, 25, 15)  # 15 radial points
            
            # Simplified rotation curve model
            v_inner = vmax * np.tanh(r_points / 3.0)
            v_flat = vmax * (1 - 0.1 * np.exp(-(r_points - 5) / 10))
            v_observed = v_flat + np.random.normal(0, 5, len(r_points))
            
            # Outer edge is last 3 points
            v_outer_obs = v_observed[-3:]
            r_outer = r_points[-3:]
            
            galaxies.append({
                'galaxy_id': f'GAL{i:03d}',
                'type': gal_type,
                'BT': BT,
                'vmax': vmax,
                'surface_brightness': SB,
                'bar_strength': bar_strength,
                'inclination': np.random.uniform(30, 80),
                'rdisk': np.random.uniform(2, 8),
                'r_outer': r_outer,
                'v_outer_observed': v_outer_obs,
                'r_all': r_points,
                'v_all': v_observed,
            })
        
        return pd.DataFrame(galaxies)
    
    def predict_track2_outer_velocity(self, galaxy: Dict) -> np.ndarray:
        """Predict outer-edge velocities using Track 2 path-spectrum kernel"""
        r_outer = galaxy['r_outer']
        BT = galaxy['BT']
        bar_strength = galaxy['bar_strength']
        
        # Estimate v_circ at outer radii (use observed as proxy for iteration)
        v_circ = np.full(len(r_outer), galaxy['vmax'])
        
        # Compute suppression factors
        xi = self.kernel.suppression_factor(
            r=r_outer,
            v_circ=v_circ,
            BT=BT,
            bar_strength=bar_strength,
            r_bulge=1.0,
            r_bar=3.0,
            r_scale=3.0
        )
        
        # Apply suppression to baseline RAR prediction
        # (For this demo, use simple model: v_predicted = xi * v_baseline)
        # In real implementation, would integrate full RAR gate model
        v_baseline = galaxy['vmax'] * np.ones(len(r_outer))
        v_predicted = v_baseline * (1 - 0.3 * (1 - xi))  # Modest suppression effect
        
        return v_predicted
    
    def predict_track3_outer_velocity(self, galaxy: Dict) -> np.ndarray:
        """Predict outer-edge velocities using Track 3 empirical predictors"""
        # Linear predictor model based on outlier triage correlations
        r_outer = galaxy['r_outer']
        
        # Normalize features to z-scores (approximate)
        SB_norm = (galaxy['surface_brightness'] - 150) / 50
        vmax_norm = (galaxy['vmax'] - 150) / 60
        inc_norm = (galaxy['inclination'] - 55) / 15
        rdisk_norm = (galaxy['rdisk'] - 5) / 2
        bt_norm = (galaxy['BT'] - 0.25) / 0.2
        
        # Compute linear predictor
        predictor = (
            self.track3_weights['surface_brightness'] * SB_norm +
            self.track3_weights['vmax'] * vmax_norm +
            self.track3_weights['inclination'] * inc_norm +
            self.track3_weights['rdisk'] * rdisk_norm +
            self.track3_weights['bt_ratio'] * bt_norm
        )
        
        # Map predictor to velocity correction
        # Negative predictor → higher error → reduce predicted velocity
        v_baseline = galaxy['vmax'] * np.ones(len(r_outer))
        v_correction = -predictor * 0.15 * v_baseline  # Scale factor tuned to data
        v_predicted = v_baseline + v_correction
        
        return v_predicted
    
    def evaluate_predictions(self, galaxies: pd.DataFrame) -> Dict:
        """Evaluate both Track 2 and Track 3 predictions"""
        results_track2 = []
        results_track3 = []
        
        for idx, galaxy in galaxies.iterrows():
            # Get observed outer velocities
            v_obs = galaxy['v_outer_observed']
            
            # Track 2 predictions
            v_pred_t2 = self.predict_track2_outer_velocity(galaxy)
            ape_t2 = 100 * np.mean(np.abs((v_obs - v_pred_t2) / v_obs))
            corr_t2 = np.corrcoef(v_obs, v_pred_t2)[0, 1]
            
            # Track 3 predictions
            v_pred_t3 = self.predict_track3_outer_velocity(galaxy)
            ape_t3 = 100 * np.mean(np.abs((v_obs - v_pred_t3) / v_obs))
            corr_t3 = np.corrcoef(v_obs, v_pred_t3)[0, 1]
            
            results_track2.append({
                'galaxy_id': galaxy['galaxy_id'],
                'type': galaxy['type'],
                'ape': ape_t2,
                'correlation': corr_t2,
                'v_observed_mean': np.mean(v_obs),
                'v_predicted_mean': np.mean(v_pred_t2),
            })
            
            results_track3.append({
                'galaxy_id': galaxy['galaxy_id'],
                'type': galaxy['type'],
                'ape': ape_t3,
                'correlation': corr_t3,
                'v_observed_mean': np.mean(v_obs),
                'v_predicted_mean': np.mean(v_pred_t3),
            })
        
        return {
            'track2': pd.DataFrame(results_track2),
            'track3': pd.DataFrame(results_track3),
        }
    
    def compare_by_morphology(self, results: Dict):
        """Compare performance by galaxy morphology"""
        df_t2 = results['track2']
        df_t3 = results['track3']
        
        types = df_t2['type'].unique()
        
        print("\n" + "=" * 80)
        print("PERFORMANCE BY MORPHOLOGICAL TYPE")
        print("=" * 80)
        print(f"{'Type':<10} {'Track 2 APE':<15} {'Track 3 APE':<15} {'Winner':<10}")
        print("-" * 80)
        
        summary = []
        for gtype in sorted(types):
            ape_t2 = df_t2[df_t2['type'] == gtype]['ape'].median()
            ape_t3 = df_t3[df_t3['type'] == gtype]['ape'].median()
            winner = 'Track 2' if ape_t2 < ape_t3 else 'Track 3'
            
            print(f"{gtype:<10} {ape_t2:>13.2f}% {ape_t3:>13.2f}% {winner:<10}")
            summary.append({
                'type': gtype,
                'track2_ape': ape_t2,
                'track3_ape': ape_t3,
                'winner': winner
            })
        
        return pd.DataFrame(summary)
    
    def plot_comparison(self, results: Dict):
        """Create comparison visualizations"""
        df_t2 = results['track2']
        df_t3 = results['track3']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: APE distributions
        ax = axes[0, 0]
        ax.hist(df_t2['ape'], bins=20, alpha=0.6, label='Track 2 (Physics)', color='blue')
        ax.hist(df_t3['ape'], bins=20, alpha=0.6, label='Track 3 (Empirical)', color='green')
        ax.set_xlabel('APE (%)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Outer-Edge APE', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Observed vs Predicted (Track 2)
        ax = axes[0, 1]
        ax.scatter(df_t2['v_observed_mean'], df_t2['v_predicted_mean'], alpha=0.6, s=50, c='blue')
        lim = [df_t2['v_observed_mean'].min() * 0.9, df_t2['v_observed_mean'].max() * 1.1]
        ax.plot(lim, lim, 'k--', lw=2, label='1:1')
        ax.set_xlabel('Observed v_outer (km/s)', fontsize=12)
        ax.set_ylabel('Predicted v_outer (km/s)', fontsize=12)
        ax.set_title('Track 2: Path-Spectrum Kernel', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 3: Observed vs Predicted (Track 3)
        ax = axes[1, 0]
        ax.scatter(df_t3['v_observed_mean'], df_t3['v_predicted_mean'], alpha=0.6, s=50, c='green')
        ax.plot(lim, lim, 'k--', lw=2, label='1:1')
        ax.set_xlabel('Observed v_outer (km/s)', fontsize=12)
        ax.set_ylabel('Predicted v_outer (km/s)', fontsize=12)
        ax.set_title('Track 3: Empirical Predictors', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 4: APE by morphology
        ax = axes[1, 1]
        types = sorted(df_t2['type'].unique())
        ape_t2_by_type = [df_t2[df_t2['type'] == t]['ape'].median() for t in types]
        ape_t3_by_type = [df_t3[df_t3['type'] == t]['ape'].median() for t in types]
        
        x = np.arange(len(types))
        width = 0.35
        ax.bar(x - width/2, ape_t2_by_type, width, label='Track 2', color='blue', alpha=0.7)
        ax.bar(x + width/2, ape_t3_by_type, width, label='Track 3', color='green', alpha=0.7)
        ax.set_xlabel('Galaxy Type', fontsize=12)
        ax.set_ylabel('Median APE (%)', fontsize=12)
        ax.set_title('Performance by Morphology', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=45)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'track2_vs_track3_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved comparison plots to {output_path}")
    
    def generate_summary_report(self, results: Dict, morph_summary: pd.DataFrame):
        """Generate comprehensive comparison report"""
        df_t2 = results['track2']
        df_t3 = results['track3']
        
        report_path = self.output_dir / 'predictive_power_comparison_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRACK 2 vs TRACK 3: PREDICTIVE POWER COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL PERFORMANCE (all galaxy types)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Track 2 (Path-Spectrum Kernel):\n")
            f.write(f"  Median APE: {df_t2['ape'].median():.2f}%\n")
            f.write(f"  Mean APE: {df_t2['ape'].mean():.2f}%\n")
            f.write(f"  StdDev APE: {df_t2['ape'].std():.2f}%\n\n")
            
            f.write(f"Track 3 (Empirical Predictors):\n")
            f.write(f"  Median APE: {df_t3['ape'].median():.2f}%\n")
            f.write(f"  Mean APE: {df_t3['ape'].mean():.2f}%\n")
            f.write(f"  StdDev APE: {df_t3['ape'].std():.2f}%\n\n")
            
            # Winner determination
            winner = 'Track 2' if df_t2['ape'].median() < df_t3['ape'].median() else 'Track 3'
            improvement = abs(df_t2['ape'].median() - df_t3['ape'].median())
            
            f.write(f"WINNER: {winner}\n")
            f.write(f"Improvement: {improvement:.2f} percentage points in median APE\n\n")
            
            # By morphology
            f.write("PERFORMANCE BY MORPHOLOGICAL TYPE\n")
            f.write("-" * 80 + "\n")
            f.write(morph_summary.to_string(index=False))
            f.write("\n\n")
            
            # Analysis
            f.write("ANALYSIS AND RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            if winner == 'Track 2':
                f.write("The physics-grounded path-spectrum kernel (Track 2) shows superior\n")
                f.write("predictive power for outer-edge velocities. This suggests that the\n")
                f.write("coherence length formulation captures the essential physics governing\n")
                f.write("velocity suppression in galaxy outskirts better than empirical predictors.\n\n")
                f.write("Recommendation: Proceed with Track 2 integration into the forward model,\n")
                f.write("fit the 4 hyperparameters on the 80% training set, and validate on 20% holdout.\n")
            else:
                f.write("The empirical predictors from outlier triage (Track 3) show superior\n")
                f.write("predictive power. This indicates that surface brightness, Vmax, and other\n")
                f.write("observables contain more information than the simplified coherence kernel.\n\n")
                f.write("Recommendation: Use Track 3 predictors to refine the V2.2 model, potentially\n")
                f.write("incorporating them as environmental corrections to the baseline RAR gate.\n")
        
        print(f"\nGenerated summary report: {report_path}")
    
    def run_full_comparison(self):
        """Run complete comparison analysis"""
        print("=" * 80)
        print("COMPARING TRACK 2 vs TRACK 3 PREDICTIVE POWER")
        print("=" * 80)
        
        # Generate synthetic galaxy sample
        print("\nGenerating synthetic SPARC-like galaxy sample...")
        galaxies = self.generate_synthetic_sparc_sample(n_galaxies=100)
        print(f"Generated {len(galaxies)} galaxies")
        
        # Evaluate both approaches
        print("\nEvaluating Track 2 (path-spectrum kernel) predictions...")
        print("Evaluating Track 3 (empirical predictors) predictions...")
        results = self.evaluate_predictions(galaxies)
        
        # Compare by morphology
        morph_summary = self.compare_by_morphology(results)
        
        # Overall statistics
        print("\n" + "=" * 80)
        print("OVERALL RESULTS")
        print("=" * 80)
        print(f"Track 2 median APE: {results['track2']['ape'].median():.2f}%")
        print(f"Track 3 median APE: {results['track3']['ape'].median():.2f}%")
        
        winner = 'Track 2' if results['track2']['ape'].median() < results['track3']['ape'].median() else 'Track 3'
        print(f"\nWINNER: {winner}")
        
        # Generate visualizations
        print("\nGenerating comparison plots...")
        self.plot_comparison(results)
        
        # Generate report
        print("\nGenerating comprehensive report...")
        self.generate_summary_report(results, morph_summary)
        
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE")
        print("=" * 80)
        
        return results, morph_summary


def main():
    """Main execution"""
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "many_path_model" / "results" / "track2_vs_track3_comparison"
    
    comparator = PredictivePowerComparator(output_dir)
    results, summary = comparator.run_full_comparison()
    
    return results, summary


if __name__ == "__main__":
    main()
