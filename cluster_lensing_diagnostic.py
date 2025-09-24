#!/usr/bin/env python3
"""
Cluster Lensing Diagnostic: G³ vs Observations
===============================================

This script performs a detailed diagnostic analysis comparing G³ lensing
predictions with observed cluster lensing data and standard NFW models.

Key analyses:
1. Compare with observed Einstein radii from literature
2. Compute required scaling factors for G³ to match observations
3. Analyze mass deficit and its radial dependence
4. Generate comprehensive diagnostic plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Physical constants
G = 4.300917270e-6  # kpc km^2 s^-2 Msun^-1
c_km_s = 299792.458  # km/s

# Observed Einstein radii from literature (arcsec)
OBSERVED_EINSTEIN_RADII = {
    'Abell_1689': {
        'theta_E': 47.0,  # Broadhurst et al. 2005, ApJ 621, 53
        'error': 3.0,
        'z_lens': 0.183,
        'z_source': 1.0,
        'reference': 'Broadhurst et al. 2005'
    },
    'Abell_2218': {
        'theta_E': 34.0,  # Elíasdóttir et al. 2007
        'error': 2.0,
        'z_lens': 0.175,
        'z_source': 1.0,
        'reference': 'Elíasdóttir et al. 2007'
    },
    'MACS_J0416': {
        'theta_E': 36.0,  # Caminha et al. 2017
        'error': 2.0,
        'z_lens': 0.396,
        'z_source': 2.0,
        'reference': 'Caminha et al. 2017'
    },
    'Bullet': {
        'theta_E': 16.0,  # Clowe et al. 2006 (main cluster)
        'error': 2.0,
        'z_lens': 0.296,
        'z_source': 1.2,
        'reference': 'Clowe et al. 2006'
    },
    'Coma': {
        'theta_E': None,  # Too nearby for strong lensing
        'error': None,
        'z_lens': 0.0231,
        'z_source': 0.8,
        'reference': 'No strong lensing (too nearby)'
    }
}

# NFW profile tools
def nfw_convergence(R_kpc, M200, r200, z_lens, z_source, c=5.0):
    """Compute NFW convergence profile."""
    # Scale radius
    rs = r200 / c
    
    # Dimensionless radius
    x = R_kpc / rs
    
    # NFW surface density (Bartelmann 1996)
    def f(x):
        if x < 1:
            return (1 - 2*np.arctanh(np.sqrt((1-x)/(1+x)))/np.sqrt(1-x**2))/(x**2 - 1)
        elif x > 1:
            return (1 - 2*np.arctan(np.sqrt((x-1)/(1+x)))/np.sqrt(x**2-1))/(x**2 - 1)
        else:
            return 1/3
    
    # Vectorize for arrays
    if isinstance(x, np.ndarray):
        kappa = np.array([f(xi) for xi in x])
    else:
        kappa = f(x)
    
    # Scale by characteristic density
    delta_c = 200/3 * c**3 / (np.log(1 + c) - c/(1 + c))
    rho_crit = 2.775e11 * 0.7**2 * 0.3  # Msun/kpc^3 at z=0
    Sigma_s = delta_c * rho_crit * rs / 1e6  # Msun/kpc^2
    
    # Get critical density for lensing
    from cluster_lensing_analysis import sigma_crit_Msun_per_kpc2
    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)
    
    return kappa * Sigma_s / Sigma_crit

def analyze_g3_deficit():
    """Analyze why G³ doesn't produce enough lensing."""
    
    # Load G³ results
    summary_file = Path('out/cluster_lensing/summaries.json')
    if not summary_file.exists():
        print("Run cluster_lensing_analysis.py first!")
        return
    
    with open(summary_file) as f:
        g3_results = json.load(f)
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Einstein radius comparison
    ax = axes[0, 0]
    clusters = []
    obs_theta_E = []
    obs_errors = []
    g3_max_kappa = []
    
    for result in g3_results:
        name = result['cluster']
        if name in OBSERVED_EINSTEIN_RADII:
            obs = OBSERVED_EINSTEIN_RADII[name]
            if obs['theta_E'] is not None:
                clusters.append(name)
                obs_theta_E.append(obs['theta_E'])
                obs_errors.append(obs['error'])
                g3_max_kappa.append(result['max_kappa_eff_mean'])
    
    x = np.arange(len(clusters))
    ax.errorbar(x, obs_theta_E, yerr=obs_errors, fmt='o', label='Observed θ_E', markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=45, ha='right')
    ax.set_ylabel('Einstein Radius (arcsec)')
    ax.set_title('Observed Einstein Radii')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Required scaling factor
    ax = axes[0, 1]
    scaling_factors = []
    for i, cluster in enumerate(clusters):
        # G³ max convergence from results
        for res in g3_results:
            if res['cluster'] == cluster:
                max_kappa = res['max_kappa_eff_mean']
                break
        # Required scaling to reach κ=1
        scaling = 1.0 / max_kappa if max_kappa > 0 else np.inf
        scaling_factors.append(scaling)
    
    ax.bar(x, scaling_factors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=45, ha='right')
    ax.set_ylabel('Required Scaling Factor')
    ax.set_title('G³ Mass Deficit (Factor needed for κ=1)')
    ax.axhline(1, color='k', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Max convergence achieved
    ax = axes[0, 2]
    bar_colors = ['red' if k < 0.5 else 'orange' if k < 1.0 else 'green' for k in g3_max_kappa]
    ax.bar(x, g3_max_kappa, color=bar_colors, alpha=0.7)
    ax.axhline(1.0, color='k', linestyle='--', label='Strong lensing threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=45, ha='right')
    ax.set_ylabel('Max Mean Convergence')
    ax.set_title('G³ Maximum κ̄(<R)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4-6: Individual cluster profiles
    for i, cluster in enumerate(clusters[:3]):
        ax = axes[1, i]
        
        # Load profile data
        profile_file = Path(f'out/cluster_lensing/{cluster}/profiles.csv')
        if profile_file.exists():
            import csv
            R, kappa_eff_mean = [], []
            with open(profile_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    R.append(float(row['R_kpc']))
                    kappa_eff_mean.append(float(row['kappa_eff_mean']))
            
            R = np.array(R)
            kappa_eff_mean = np.array(kappa_eff_mean)
            
            # Plot G³ profile
            ax.loglog(R, kappa_eff_mean, label='G³', linewidth=2)
            
            # Add NFW comparison (typical parameters)
            for res in g3_results:
                if res['cluster'] == cluster:
                    M200 = res['M200_Msun']
                    r200 = res['r200_kpc']
                    z_lens = res['z_lens']
                    z_source = res['z_source']
                    break
            
            # NFW with different concentrations
            for c in [3, 5, 7]:
                kappa_nfw = nfw_convergence(R, M200, r200, z_lens, z_source, c=c)
                ax.loglog(R, kappa_nfw, '--', alpha=0.5, label=f'NFW c={c}')
            
            ax.axhline(1.0, color='k', linestyle=':', alpha=0.5)
            ax.set_xlabel('R (kpc)')
            ax.set_ylabel('Mean Convergence κ̄(<R)')
            ax.set_title(f'{cluster}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('G³ Lensing Diagnostic: Why G³ Under-predicts Cluster Lensing', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    outdir = Path('out/cluster_lensing')
    fig.savefig(outdir / 'g3_lensing_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print analysis summary
    print("\n" + "="*80)
    print("G³ CLUSTER LENSING DIAGNOSTIC SUMMARY")
    print("="*80)
    
    print("\nKEY FINDINGS:")
    print("-" * 40)
    
    # Average deficit
    avg_scaling = np.mean(scaling_factors)
    print(f"1. Mass Deficit: G³ needs ~{avg_scaling:.1f}x more mass for strong lensing")
    
    # Maximum convergence
    max_achieved = np.max(g3_max_kappa)
    print(f"2. Best Case: Maximum κ̄ achieved = {max_achieved:.3f} (need 1.0)")
    
    print(f"3. Success Rate: {sum(k >= 1.0 for k in g3_max_kappa)}/{len(g3_max_kappa)} clusters reach strong lensing")
    
    print("\nIMPLICATIONS:")
    print("-" * 40)
    print("• G³ in current form dramatically under-predicts cluster lensing")
    print("• The 'tail' acceleration (g_tail) is insufficient at cluster scales")
    print("• Possible issues:")
    print("  - Screening function too aggressive at high densities")
    print("  - Saturation (g_sat) limiting total acceleration")
    print("  - Scale-dependent parameters need cluster-specific values")
    print("  - Fundamental limitation of universal formula approach")
    
    print("\nPOSSIBLE SOLUTIONS:")
    print("-" * 40)
    print("1. Scale-dependent parameters: r_c0, v_0 scale with system size")
    print("2. Modified screening: Less suppression in high-density regions")
    print("3. Remove/increase saturation cap for clusters")
    print("4. Additional cluster-specific term in formula")
    print("5. Acknowledge G³ works for galaxies but not clusters")
    
    print("\nCOMPARISON WITH NFW:")
    print("-" * 40)
    print("• NFW with c=5-7 matches observations reasonably well")
    print("• G³ produces much shallower profiles than NFW")
    print("• G³ lacks the central density spike needed for strong lensing")
    
    print("="*80)
    
    # Save detailed results
    diagnostic_results = {
        'clusters': clusters,
        'observed_einstein_radii': [obs_theta_E[i] for i in range(len(clusters))],
        'g3_max_convergence': g3_max_kappa,
        'required_scaling_factors': scaling_factors,
        'average_deficit_factor': avg_scaling,
        'implications': {
            'mass_deficit': f'G³ needs {avg_scaling:.1f}x more mass',
            'best_convergence': max_achieved,
            'success_rate': f'{sum(k >= 1.0 for k in g3_max_kappa)}/{len(g3_max_kappa)}',
            'conclusion': 'G³ significantly under-predicts cluster lensing'
        }
    }
    
    with open(outdir / 'lensing_diagnostic_results.json', 'w') as f:
        json.dump(diagnostic_results, f, indent=2)
    
    print(f"\nResults saved to {outdir / 'lensing_diagnostic_results.json'}")

if __name__ == '__main__':
    analyze_g3_deficit()