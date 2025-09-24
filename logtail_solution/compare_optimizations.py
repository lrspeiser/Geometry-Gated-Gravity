#!/usr/bin/env python3
"""
Compare MW vs SPARC Optimized LogTail Parameters
================================================

This script analyzes whether different astronomical systems need different
LogTail parameters, or if the 80% ceiling is fundamental to the model.

Key questions:
1. Do MW and SPARC prefer different parameters?
2. How much does cross-application hurt accuracy?
3. Is there a universal parameter set that works reasonably for both?
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from logtail_model import LogTailModel
from data_loader import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ParameterComparison:
    """Compare performance of different parameter sets."""
    
    def __init__(self):
        self.loader = DataLoader()
        self.mw_params = None
        self.sparc_params = None
        self.results = {}
    
    def load_parameters(self):
        """Load optimized parameters from both optimizations."""
        # MW parameters (12-hour run)
        if Path("optimized_parameters.json").exists():
            with open("optimized_parameters.json", 'r') as f:
                mw_data = json.load(f)
            
            if 'theta' in mw_data:
                # Extract from theta format
                self.mw_params = {
                    'v0_kms': mw_data['theta'][0],
                    'rc_kpc': mw_data['theta'][1],
                    'r0_kpc': mw_data['theta'][2],
                    'delta_kpc': mw_data['theta'][3],
                    'gamma': mw_data['theta'][4] if len(mw_data['theta']) > 4 else 0.5,
                    'beta': mw_data['theta'][5] if len(mw_data['theta']) > 5 else 0.1
                }
            else:
                self.mw_params = mw_data
            
            logger.info("Loaded MW-optimized parameters")
        
        # SPARC parameters (if exists from new optimization)
        if Path("sparc_optimized_parameters.json").exists():
            with open("sparc_optimized_parameters.json", 'r') as f:
                sparc_data = json.load(f)
            
            if 'parameters' in sparc_data:
                self.sparc_params = sparc_data['parameters']
            else:
                self.sparc_params = sparc_data
            
            logger.info("Loaded SPARC-optimized parameters")
        else:
            logger.info("No SPARC-specific optimization found yet")
            self.sparc_params = None
    
    def compare_parameters(self):
        """Compare parameter values between optimizations."""
        if self.sparc_params is None:
            print("\n⚠️  No SPARC-specific optimization available yet.")
            print("Run 'python optimize_sparc_gpu.py' to generate SPARC-specific parameters.")
            return
        
        print("\n" + "="*60)
        print("PARAMETER COMPARISON")
        print("="*60)
        print(f"{'Parameter':<15} {'MW-optimized':<15} {'SPARC-optimized':<15} {'Difference':<15}")
        print("-"*60)
        
        param_names = ['v0_kms', 'rc_kpc', 'r0_kpc', 'delta_kpc', 'gamma', 'beta']
        
        differences = {}
        for param in param_names:
            mw_val = self.mw_params.get(param, 0)
            sparc_val = self.sparc_params.get(param, 0)
            diff = sparc_val - mw_val
            pct_diff = 100 * diff / mw_val if mw_val != 0 else 0
            
            print(f"{param:<15} {mw_val:<15.3f} {sparc_val:<15.3f} {diff:+.3f} ({pct_diff:+.1f}%)")
            differences[param] = {'mw': mw_val, 'sparc': sparc_val, 'diff': diff, 'pct': pct_diff}
        
        print("="*60)
        
        # Identify major differences
        major_diffs = [p for p, d in differences.items() if abs(d['pct']) > 20]
        if major_diffs:
            print(f"\n⚠️  Major differences (>20%) in: {', '.join(major_diffs)}")
        else:
            print("\n✓ Parameters are relatively similar (<20% difference)")
        
        return differences
    
    def cross_validate(self):
        """Test each parameter set on the other dataset."""
        print("\n" + "="*60)
        print("CROSS-VALIDATION ANALYSIS")
        print("="*60)
        
        # Create models
        mw_model = LogTailModel(self.mw_params)
        sparc_model = LogTailModel(self.sparc_params) if self.sparc_params else None
        
        # Test on subset of data for speed
        print("\n1. Testing on SPARC galaxies (sample of 10)...")
        galaxies = self.loader.load_sparc_galaxies(max_galaxies=10)
        
        mw_sparc_errors = []
        sparc_sparc_errors = []
        
        for name, data in galaxies.items():
            # MW parameters on SPARC
            result_mw = mw_model.predict_rotation_curve(data['r_kpc'], data['v_bar'])
            error_mw = np.mean(np.abs(result_mw['v_total'] - data['v_obs']) / data['v_obs'])
            mw_sparc_errors.append(error_mw)
            
            # SPARC parameters on SPARC
            if sparc_model:
                result_sparc = sparc_model.predict_rotation_curve(data['r_kpc'], data['v_bar'])
                error_sparc = np.mean(np.abs(result_sparc['v_total'] - data['v_obs']) / data['v_obs'])
                sparc_sparc_errors.append(error_sparc)
        
        print(f"  MW params on SPARC: {100*(1-np.mean(mw_sparc_errors)):.1f}% accuracy")
        if sparc_model:
            print(f"  SPARC params on SPARC: {100*(1-np.mean(sparc_sparc_errors)):.1f}% accuracy")
            improvement = 100 * (np.mean(mw_sparc_errors) - np.mean(sparc_sparc_errors)) / np.mean(mw_sparc_errors)
            print(f"  Improvement with SPARC-specific: {improvement:+.1f}%")
        
        print("\n2. Testing on Milky Way...")
        mw_data = self.loader.load_milky_way_data()
        
        if len(mw_data['r_kpc']) > 0:
            # MW parameters on MW
            result_mw = mw_model.predict_rotation_curve(mw_data['r_kpc'], mw_data['v_bar_estimate'])
            error_mw = np.median(np.abs(result_mw['v_total'] - mw_data['v_circ']) / mw_data['v_circ'])
            
            print(f"  MW params on MW: {100*(1-error_mw):.1f}% accuracy")
            
            # SPARC parameters on MW
            if sparc_model:
                result_sparc = sparc_model.predict_rotation_curve(mw_data['r_kpc'], mw_data['v_bar_estimate'])
                error_sparc = np.median(np.abs(result_sparc['v_total'] - mw_data['v_circ']) / mw_data['v_circ'])
                print(f"  SPARC params on MW: {100*(1-error_sparc):.1f}% accuracy")
                degradation = 100 * (error_sparc - error_mw) / error_mw
                print(f"  Degradation with SPARC params: {degradation:+.1f}%")
        
        print("="*60)
    
    def find_universal_parameters(self):
        """Attempt to find compromise parameters that work for both."""
        if self.sparc_params is None:
            return
        
        print("\n" + "="*60)
        print("UNIVERSAL PARAMETER EXPLORATION")
        print("="*60)
        
        # Simple average (could be weighted)
        universal_params = {}
        for param in ['v0_kms', 'rc_kpc', 'r0_kpc', 'delta_kpc', 'gamma', 'beta']:
            mw_val = self.mw_params.get(param, 0)
            sparc_val = self.sparc_params.get(param, 0)
            # Weighted average (could optimize this weight)
            universal_params[param] = 0.4 * mw_val + 0.6 * sparc_val
        
        print("Proposed universal parameters (40% MW + 60% SPARC):")
        for param, value in universal_params.items():
            print(f"  {param}: {value:.3f}")
        
        # Test universal parameters
        universal_model = LogTailModel(universal_params)
        
        # Quick test on both datasets
        print("\nTesting universal parameters...")
        
        # SPARC test
        galaxies = self.loader.load_sparc_galaxies(max_galaxies=5)
        sparc_errors = []
        for name, data in galaxies.items():
            result = universal_model.predict_rotation_curve(data['r_kpc'], data['v_bar'])
            error = np.mean(np.abs(result['v_total'] - data['v_obs']) / data['v_obs'])
            sparc_errors.append(error)
        
        # MW test
        mw_data = self.loader.load_milky_way_data()
        result_mw = universal_model.predict_rotation_curve(mw_data['r_kpc'], mw_data['v_bar_estimate'])
        mw_error = np.median(np.abs(result_mw['v_total'] - mw_data['v_circ']) / mw_data['v_circ'])
        
        print(f"  Universal params on SPARC: {100*(1-np.mean(sparc_errors)):.1f}% accuracy")
        print(f"  Universal params on MW: {100*(1-mw_error):.1f}% accuracy")
        
        return universal_params
    
    def visualize_differences(self):
        """Create visualization of parameter differences."""
        if self.sparc_params is None:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        param_names = ['v0_kms', 'rc_kpc', 'r0_kpc', 'delta_kpc', 'gamma', 'beta']
        param_labels = ['v₀ (km/s)', 'rc (kpc)', 'r₀ (kpc)', 'δ (kpc)', 'γ', 'β']
        
        for i, (param, label) in enumerate(zip(param_names, param_labels)):
            ax = axes[i]
            
            mw_val = self.mw_params.get(param, 0)
            sparc_val = self.sparc_params.get(param, 0)
            
            # Bar plot
            x = [0, 1]
            values = [mw_val, sparc_val]
            colors = ['blue', 'orange']
            bars = ax.bar(x, values, color=colors, alpha=0.7, width=0.5)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom')
            
            # Add percentage difference
            pct_diff = 100 * (sparc_val - mw_val) / mw_val if mw_val != 0 else 0
            ax.text(0.5, max(values) * 1.1, f'Δ = {pct_diff:+.1f}%',
                   ha='center', fontsize=10, color='red' if abs(pct_diff) > 20 else 'green')
            
            ax.set_xticks(x)
            ax.set_xticklabels(['MW', 'SPARC'])
            ax.set_ylabel(label)
            ax.set_title(f'{label} Comparison')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('MW vs SPARC Optimized Parameters', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('plots')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'parameter_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Visualization saved to plots/parameter_comparison.png")
    
    def generate_report(self):
        """Generate comprehensive comparison report."""
        report = f"""
LogTail/G³ Parameter Comparison Report
======================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
-----------------
This analysis compares LogTail parameters optimized separately for:
1. Milky Way (12-hour GPU optimization, 80% accuracy ceiling)
2. SPARC galaxies (separate optimization)

KEY FINDINGS
------------
"""
        
        if self.sparc_params:
            # Calculate key metrics
            major_diffs = []
            for param in ['v0_kms', 'rc_kpc', 'r0_kpc', 'delta_kpc', 'gamma', 'beta']:
                mw_val = self.mw_params.get(param, 0)
                sparc_val = self.sparc_params.get(param, 0)
                pct_diff = 100 * (sparc_val - mw_val) / mw_val if mw_val != 0 else 0
                if abs(pct_diff) > 20:
                    major_diffs.append(f"{param}: {pct_diff:+.1f}%")
            
            if major_diffs:
                report += f"• Major parameter differences found: {', '.join(major_diffs)}\n"
                report += "• This suggests MW and SPARC galaxies need different modifications\n"
            else:
                report += "• Parameters are relatively similar (<20% difference)\n"
                report += "• This suggests the 80% ceiling may be fundamental to the model\n"
        else:
            report += "• SPARC-specific optimization not yet available\n"
            report += "• Run 'python optimize_sparc_gpu.py' to generate\n"
        
        report += """
IMPLICATIONS
------------
1. If parameters differ significantly:
   - Different galaxy types need different gravity modifications
   - A universal law may not exist at this precision level
   - Consider mass-dependent or morphology-dependent parameters

2. If parameters are similar:
   - The 80% accuracy ceiling is likely fundamental
   - Model formulation may need revision (not just parameter tuning)
   - Consider additional physics or different functional forms

RECOMMENDATIONS
---------------
1. Run extended optimization (>12 hours) to confirm convergence
2. Try intermediate parameter sets for specific galaxy subclasses
3. Explore alternative functional forms for the LogTail modification
4. Consider adding additional parameters for flexibility
"""
        
        # Save report
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        report_file = output_dir / 'parameter_comparison_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\n✓ Report saved to {report_file}")
        
        return report


def main():
    """Run complete comparison analysis."""
    from datetime import datetime
    
    print("\n" + "="*70)
    print("LOGTAIL PARAMETER COMPARISON ANALYSIS")
    print("="*70)
    
    comparison = ParameterComparison()
    
    # Load parameters
    comparison.load_parameters()
    
    # Compare if both available
    if comparison.mw_params:
        comparison.compare_parameters()
        comparison.cross_validate()
        universal = comparison.find_universal_parameters()
        comparison.visualize_differences()
        comparison.generate_report()
    else:
        print("\n❌ No MW parameters found. The comparison requires:")
        print("   1. MW optimization results (optimized_parameters.json)")
        print("   2. SPARC optimization results (sparc_optimized_parameters.json)")
        print("\nRun optimizations first:")
        print("   python optimize_sparc_gpu.py --iterations 1000")


if __name__ == "__main__":
    main()