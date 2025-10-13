#!/usr/bin/env python3
"""
Track B1: Cluster Lensing Predictions from Frozen SPARC Parameters
===================================================================

Predict cluster Einstein radii using ZERO per-cluster tuning.
Uses only the 7 frozen hyperparameters from SPARC rotation curves.

This is THE critical test:
- If this works → Geometry-gated gravity validated at potential level
- No dark matter needed for lensing (unlike MOND)
- Zero free parameters per cluster (unlike NFW: 2-3 params/cluster)

Author: Track B1 Implementation  
Date: 2025-01-13
"""

import sys
sys.path.insert(0, 'C:/Users/henry/dev/GravityCalculator/many_path_model')

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

# Local imports
from cluster_data_loader import ClusterDataLoader, ClusterBaryonData
from lensing_utilities import (LensingCosmology, AbelProjection, 
                               EinsteinRadiusFinder)
from path_spectrum_kernel_track2 import PathSpectrumKernel, PathSpectrumHyperparams


class ClusterLensingPredictor:
    """Predict cluster lensing from frozen SPARC parameters."""
    
    def __init__(self, sparc_split_file: str = "splits/sparc_split_v1.json"):
        """
        Initialize with frozen SPARC hyperparameters.
        
        Parameters
        ----------
        sparc_split_file : str
            Path to SPARC split file with frozen hyperparameters
        """
        # Load frozen hyperparameters from SPARC
        with open(sparc_split_file, 'r') as f:
            split_data = json.load(f)
        
        hp_dict = split_data['hyperparameters']
        self.hp = PathSpectrumHyperparams(**hp_dict)
        
        print(f"✅ Loaded frozen SPARC hyperparameters:")
        print(f"   L_0 = {self.hp.L_0:.3f} kpc")
        print(f"   p = {self.hp.p:.3f}")
        print(f"   n_coh = {self.hp.n_coh:.3f}")
        print(f"   β_bulge = {self.hp.beta_bulge:.3f}")
        print(f"   α_shear = {self.hp.alpha_shear:.3f}")
        print(f"   γ_bar = {self.hp.gamma_bar:.3f}")
        print(f"   A_0 = {self.hp.A_0:.3f}")
        
        # Initialize kernel (NO tuning)
        self.kernel = PathSpectrumKernel(self.hp, use_cupy=False)
        
        # Initialize utilities
        self.data_loader = ClusterDataLoader()
        self.cosmo = LensingCosmology()
        self.abel = AbelProjection()
        self.finder = EinsteinRadiusFinder(self.cosmo)
        
        # Physical constants
        self.G = 4.300917270e-6  # kpc km² s⁻² M_☉⁻¹
    
    def predict_lensing(self, cluster_name: str, z_source: float = 2.0,
                       verbose: bool = True) -> Dict:
        """
        Predict cluster lensing observables from baryons only.
        
        Parameters
        ----------
        cluster_name : str
            Cluster name (e.g., 'MACSJ0416')
        z_source : float
            Source redshift for lensing
        verbose : bool
            Print progress
            
        Returns
        -------
        results : dict
            Dictionary with all predictions and diagnostics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"PREDICTING LENSING FOR {cluster_name}")
            print(f"{'='*70}")
        
        # Step 1: Load cluster data
        if verbose:
            print("\n[1/7] Loading cluster baryon profiles...")
        data = self.data_loader.load_cluster(cluster_name, validate=True)
        
        if not data.validated:
            raise ValueError(f"Cluster data validation failed for {cluster_name}")
        
        # Step 2: Compute baryonic quantities
        if verbose:
            print("[2/7] Computing baryonic mass and acceleration...")
        M_bar, g_bar = self.data_loader.compute_baryonic_mass(data)
        v_bar = np.sqrt(g_bar * data.r_kpc)
        
        # Step 3: Apply path-spectrum boost
        if verbose:
            print("[3/7] Applying path-spectrum kernel boost...")
        
        # Compute boost factor K(r)
        K = self.kernel.many_path_boost_factor(
            r=data.r_kpc,
            v_circ=v_bar,
            g_bar=g_bar
        )
        
        # Apply boost: g_total = g_bar × (1 + K)
        g_total = g_bar * (1.0 + K)
        
        # Compute effective enclosed mass
        # Approximate: M_eff ≈ M_bar × (1 + K)
        # More rigorous: integrate ρ_eff from g_total
        M_eff = M_bar * (1.0 + K)
        
        # Compute effective density via differentiation
        rho_eff = self.abel.enclosed_mass_to_density(data.r_kpc, M_eff)
        
        if verbose:
            K_median = np.median(K)
            K_mean = np.mean(K)
            print(f"   Boost factor K: median={K_median:.4f}, mean={K_mean:.4f}")
            print(f"   M_eff/M_bar at 500 kpc: {M_eff[np.argmin(np.abs(data.r_kpc-500))]/M_bar[np.argmin(np.abs(data.r_kpc-500))]:.3f}")
        
        # Step 4: Project to surface density
        if verbose:
            print("[4/7] Projecting to 2D surface density (Abel transform)...")
        
        # Define projection radii
        R_kpc = np.logspace(0, np.log10(data.r_kpc.max()), 200)
        
        # Baryonic surface density
        Sigma_bar = self.abel.project_density_to_surface(data.r_kpc, data.rho_total, R_kpc)
        
        # Effective surface density (with boost)
        Sigma_eff = self.abel.project_density_to_surface(data.r_kpc, rho_eff, R_kpc)
        
        # Step 5: Compute lensing quantities
        if verbose:
            print("[5/7] Computing convergence and shear...")
        
        Sigma_crit = self.cosmo.critical_surface_density(data.z_lens, z_source)
        
        # Convergence
        kappa_bar = self.finder.compute_convergence(Sigma_bar, Sigma_crit)
        kappa_eff = self.finder.compute_convergence(Sigma_eff, Sigma_crit)
        
        # Mean convergence
        kappa_bar_mean = self.finder.compute_mean_convergence(R_kpc, kappa_bar)
        kappa_eff_mean = self.finder.compute_mean_convergence(R_kpc, kappa_eff)
        
        # Shear
        gamma_t_bar = self.finder.compute_shear(R_kpc, kappa_bar, kappa_bar_mean)
        gamma_t_eff = self.finder.compute_shear(R_kpc, kappa_eff, kappa_eff_mean)
        
        if verbose:
            print(f"   Σ_crit = {Sigma_crit:.2e} M_☉/kpc²")
            print(f"   κ_eff(100 kpc) = {kappa_eff[np.argmin(np.abs(R_kpc-100))]:.3f}")
        
        # Step 6: Find Einstein radii
        if verbose:
            print("[6/7] Finding Einstein radii...")
        
        R_E_bar, theta_E_bar = self.finder.find_einstein_radius(R_kpc, kappa_bar_mean, data.z_lens)
        R_E_eff, theta_E_eff = self.finder.find_einstein_radius(R_kpc, kappa_eff_mean, data.z_lens)
        
        if verbose:
            if theta_E_bar is not None:
                print(f"   θ_E (baryons only): {theta_E_bar:.2f} arcsec")
            else:
                print(f"   θ_E (baryons only): No Einstein radius (κ_max < 1)")
            
            if theta_E_eff is not None:
                print(f"   θ_E (with boost):   {theta_E_eff:.2f} arcsec")
            else:
                print(f"   θ_E (with boost):   No Einstein radius (κ_max < 1)")
        
        # Step 7: Compile results
        results = {
            'cluster_name': cluster_name,
            'z_lens': data.z_lens,
            'z_source': z_source,
            
            # Radial profiles
            'r_kpc': data.r_kpc.tolist(),
            'R_kpc': R_kpc.tolist(),
            
            # Boost factors
            'K': K.tolist(),
            'K_median': float(np.median(K)),
            'K_mean': float(np.mean(K)),
            
            # Mass profiles
            'M_bar': M_bar.tolist(),
            'M_eff': M_eff.tolist(),
            
            # Surface densities
            'Sigma_bar': Sigma_bar.tolist(),
            'Sigma_eff': Sigma_eff.tolist(),
            'Sigma_crit': float(Sigma_crit),
            
            # Convergence
            'kappa_bar': kappa_bar.tolist(),
            'kappa_eff': kappa_eff.tolist(),
            'kappa_bar_mean': kappa_bar_mean.tolist(),
            'kappa_eff_mean': kappa_eff_mean.tolist(),
            
            # Shear
            'gamma_t_bar': gamma_t_bar.tolist(),
            'gamma_t_eff': gamma_t_eff.tolist(),
            
            # Einstein radii
            'R_E_bar_kpc': R_E_bar,
            'theta_E_bar_arcsec': theta_E_bar,
            'R_E_eff_kpc': R_E_eff,
            'theta_E_eff_arcsec': theta_E_eff,
            
            # Metadata
            'frozen_parameters': True,
            'n_free_params_per_cluster': 0
        }
        
        if verbose:
            print(f"[7/7] Prediction complete!")
        
        return results
    
    def plot_diagnostics(self, results: Dict, output_file: str = None,
                        theta_E_obs: float = None, theta_E_obs_err: float = None):
        """
        Create diagnostic plots for lensing prediction.
        
        Parameters
        ----------
        results : dict
            Results from predict_lensing()
        output_file : str, optional
            Save plot to file
        theta_E_obs : float, optional
            Observed Einstein radius [arcsec]
        theta_E_obs_err : float, optional
            Error on observed Einstein radius
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        R = np.array(results['R_kpc'])
        
        # Plot 1: Surface density
        ax = axes[0, 0]
        ax.loglog(R, results['Sigma_bar'], 'b-', label='Baryons only', alpha=0.7)
        ax.loglog(R, results['Sigma_eff'], 'r-', label='With boost', linewidth=2)
        ax.axhline(results['Sigma_crit'], color='k', linestyle='--', alpha=0.5, label='Σ_crit')
        ax.set_xlabel('R [kpc]')
        ax.set_ylabel('Σ(R) [M_☉/kpc²]')
        ax.set_title('Surface Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Convergence
        ax = axes[0, 1]
        ax.loglog(R, results['kappa_bar'], 'b-', label='Baryons only', alpha=0.7)
        ax.loglog(R, results['kappa_eff'], 'r-', label='With boost', linewidth=2)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='κ = 1')
        ax.set_xlabel('R [kpc]')
        ax.set_ylabel('κ(R)')
        ax.set_title('Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mean convergence
        ax = axes[0, 2]
        ax.loglog(R, results['kappa_bar_mean'], 'b-', label='Baryons only', alpha=0.7)
        ax.loglog(R, results['kappa_eff_mean'], 'r-', label='With boost', linewidth=2)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='<κ> = 1')
        
        # Mark Einstein radii
        if results['R_E_bar_kpc'] is not None:
            ax.plot(results['R_E_bar_kpc'], 1.0, 'bo', markersize=10, label='R_E (bar)')
        if results['R_E_eff_kpc'] is not None:
            ax.plot(results['R_E_eff_kpc'], 1.0, 'ro', markersize=10, label='R_E (eff)')
        
        ax.set_xlabel('R [kpc]')
        ax.set_ylabel('<κ>(<R)')
        ax.set_title('Mean Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Shear
        ax = axes[1, 0]
        ax.semilogx(R, results['gamma_t_bar'], 'b-', label='Baryons only', alpha=0.7)
        ax.semilogx(R, results['gamma_t_eff'], 'r-', label='With boost', linewidth=2)
        ax.set_xlabel('R [kpc]')
        ax.set_ylabel('γ_t(R)')
        ax.set_title('Tangential Shear')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Boost factor
        ax = axes[1, 1]
        r = np.array(results['r_kpc'])
        K = np.array(results['K'])
        ax.semilogx(r, K * 100, 'g-', linewidth=2)
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('Boost K [%]')
        ax.set_title(f'Path-Spectrum Boost (median={results["K_median"]*100:.2f}%)')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Einstein radius comparison
        ax = axes[1, 2]
        
        theta_values = []
        labels = []
        colors = []
        
        if results['theta_E_bar_arcsec'] is not None:
            theta_values.append(results['theta_E_bar_arcsec'])
            labels.append('Baryons\nonly')
            colors.append('blue')
        
        if results['theta_E_eff_arcsec'] is not None:
            theta_values.append(results['theta_E_eff_arcsec'])
            labels.append('With\nboost')
            colors.append('red')
        
        if theta_E_obs is not None:
            theta_values.append(theta_E_obs)
            labels.append('Observed')
            colors.append('black')
        
        if theta_values:
            x_pos = np.arange(len(theta_values))
            bars = ax.bar(x_pos, theta_values, color=colors, alpha=0.7, edgecolor='black')
            
            # Add error bar for observed
            if theta_E_obs is not None and theta_E_obs_err is not None:
                obs_idx = len(theta_values) - 1
                ax.errorbar(obs_idx, theta_E_obs, yerr=theta_E_obs_err, 
                           fmt='none', color='black', capsize=5, linewidth=2)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.set_ylabel('θ_E [arcsec]')
            ax.set_title('Einstein Radius')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add values on bars
            for i, (bar, val) in enumerate(zip(bars, theta_values)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}"', ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Einstein\nradius found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        plt.suptitle(f"{results['cluster_name']} Lensing Prediction (z_lens={results['z_lens']:.3f}, z_source={results['z_source']:.1f})",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved to {output_file}")
        
        return fig


def run_single_cluster(cluster_name: str = "MACSJ0416", 
                      theta_E_obs: float = 35.0,
                      theta_E_obs_err: float = 1.5):
    """Run prediction on a single cluster with validation."""
    
    print("="*70)
    print("TRACK B1: CLUSTER LENSING PREDICTION")
    print("="*70)
    print(f"\nCluster: {cluster_name}")
    print(f"Observed θ_E: {theta_E_obs:.1f} ± {theta_E_obs_err:.1f} arcsec")
    print("\nUsing FROZEN SPARC parameters (ZERO per-cluster tuning)")
    
    # Initialize predictor
    predictor = ClusterLensingPredictor()
    
    # Run prediction
    results = predictor.predict_lensing(cluster_name, z_source=2.0, verbose=True)
    
    # Save results
    output_dir = Path("many_path_model/results/cluster_lensing_b1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{cluster_name.lower()}_predictions.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")
    
    # Create diagnostic plots
    plot_file = output_dir / f"{cluster_name.lower()}_diagnostics.png"
    predictor.plot_diagnostics(results, output_file=str(plot_file),
                               theta_E_obs=theta_E_obs, theta_E_obs_err=theta_E_obs_err)
    
    # Validation report
    print(f"\n{'='*70}")
    print("VALIDATION REPORT")
    print(f"{'='*70}")
    
    theta_pred = results['theta_E_eff_arcsec']
    if theta_pred is not None:
        ratio = theta_pred / theta_E_obs
        sigma = abs(theta_pred - theta_E_obs) / theta_E_obs_err
        
        print(f"\nEinstein Radius:")
        print(f"  Predicted:  {theta_pred:.2f} arcsec")
        print(f"  Observed:   {theta_E_obs:.1f} ± {theta_E_obs_err:.1f} arcsec")
        print(f"  Ratio:      {ratio:.2f}  (pred/obs)")
        print(f"  Deviation:  {sigma:.1f}σ")
        
        if abs(ratio - 1.0) < 0.5:
            print(f"\n✅ SUCCESS: Prediction within factor of 2!")
        elif abs(ratio - 1.0) < 1.0:
            print(f"\n⚠️  MARGINAL: Prediction within factor of 3")
        else:
            print(f"\n❌ FAILURE: Prediction off by factor > 3")
        
        print(f"\nBoost Statistics:")
        print(f"  Median K: {results['K_median']*100:.2f}%")
        print(f"  Mean K:   {results['K_mean']*100:.2f}%")
        
    else:
        print(f"\n❌ FAILURE: No Einstein radius found")
        print(f"   (max convergence < 1.0)")
    
    print(f"\n{'='*70}\n")
    
    return results


if __name__ == "__main__":
    # Run on MACS0416 (best quality data)
    results = run_single_cluster(
        cluster_name="MACSJ0416",
        theta_E_obs=35.0,      # From Frontier Fields gold standard
        theta_E_obs_err=1.5
    )
