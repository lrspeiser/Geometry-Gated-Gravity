#!/usr/bin/env python3
"""
Blind Cluster Validation Suite
================================

Systematically tests the 3D interior-chord kernel on multiple galaxy clusters
with FROZEN hyperparameters and physically-calibrated baryons.

Goal: Demonstrate universal validity of path-integral gravity from galaxies
      to clusters using baryon-only models (no dark matter).

Test Protocol:
--------------
1. Load master cluster catalog (12 systems)
2. For each cluster:
   a. Build physically-calibrated baryons (gNFW gas + BCG + ICL)
   b. Apply 3D shell kernel with FROZEN hyperparameters
   c. Compute Einstein radius and convergence profiles
   d. Record residuals vs observations
3. Generate summary statistics and validation figures

Frozen Hyperparameters (from MACS0416 calibration):
----------------------------------------------------
- A_c = 10.0         # Cluster amplitude
- ell0 = 180.0       # Coherence length [kpc]
- p_density = 1.2    # Density-dependent constructive interference
- w_interior = 1.0   # Interior chords FULL STRENGTH
- w_exterior = 0.0   # Exterior arcs DISABLED (optimal from tuning)
- n_coh = 1.5        # Coherence damping exponent

Acceptance Criteria:
--------------------
- Einstein radius: median error ≤ 15%
- ≥ 60% of clusters within ±20% of observed
- No catastrophic outliers (>50% error)

Author: GravityCalculator
Date: 2025-01-14
"""

import numpy as np
import pandas as pd
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Add core and many_path_model directories to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'core'))
sys.path.insert(0, str(repo_root / 'many_path_model'))
sys.path.insert(0, str(repo_root))

from build_cluster_baryons import (
    build_cluster_baryon_model,
    ClusterBaryonParams
)
from cluster_kernel_3d_shell import (
    Shell3DKernelParams,
    lensing_profiles_3d_shell
)


@dataclass
class ClusterResult:
    """Results for a single cluster."""
    cluster_name: str
    z_lens: float
    M_500: float
    R_500: float
    
    # Observations
    theta_E_obs: float
    theta_E_err: float
    
    # Predictions
    theta_E_pred: float
    theta_E_residual: float  # (pred - obs) / obs
    theta_E_sigma: float  # residual / err
    
    # Peak convergence
    kappa_max: float
    mean_kappa_max: float
    
    # Boost factor at Einstein radius
    K_Sigma_at_RE: float
    
    # Baryon diagnostics
    fgas_R500: float
    fbaryon_R500: float
    M_baryon: float
    
    # Status
    converged: bool
    notes: str


class ClusterValidationSuite:
    """Main driver for blind cluster validation."""
    
    def __init__(
        self,
        catalog_path: str,
        output_dir: str,
        kernel_params: Optional[Shell3DKernelParams] = None,
        holdout_fraction: float = 0.0,
        random_seed: int = 42,
        verbose: bool = True
    ):
        """
        Initialize validation suite.
        
        Parameters
        ----------
        catalog_path : str
            Path to master cluster catalog CSV
        output_dir : str
            Directory for output files
        kernel_params : Shell3DKernelParams, optional
            Kernel parameters (default: MACS0416 optimal)
        holdout_fraction : float
            Fraction of clusters to hold out for testing (default: 0.0)
        random_seed : int
            Random seed for reproducibility
        verbose : bool
            Print progress
        """
        self.catalog_path = catalog_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load catalog
        self.catalog = pd.read_csv(catalog_path)
        print(f"Loaded {len(self.catalog)} clusters from catalog")
        
        # Parse M_500 (handle scientific notation)
        self.catalog['M_500_Msun'] = self.catalog['M_500_Msun'].apply(
            lambda x: float(x) if not isinstance(x, str) else float(x.replace('e', 'e'))
        )
        
        # Kernel parameters (frozen from MACS0416)
        if kernel_params is None:
            self.kernel_params = Shell3DKernelParams(
                A_c=10.0,
                r_gate=5.0,
                n_gate=4,
                ell0=180.0,
                p_density=1.2,
                L1=1200.0,
                q_taper=2.0,
                w_interior=1.0,
                w_exterior=0.0,  # KEY: exterior arcs disabled (optimal)
                coherence_mode='power_law',
                n_coh=1.5
            )
        else:
            self.kernel_params = kernel_params
        
        # Train/holdout split
        if holdout_fraction > 0:
            np.random.seed(random_seed)
            n_holdout = int(len(self.catalog) * holdout_fraction)
            holdout_indices = np.random.choice(
                len(self.catalog), n_holdout, replace=False
            )
            self.catalog['is_holdout'] = False
            self.catalog.loc[holdout_indices, 'is_holdout'] = True
            print(f"Holdout: {n_holdout} clusters ({holdout_fraction*100:.0f}%)")
        else:
            self.catalog['is_holdout'] = False
        
        self.verbose = verbose
        self.results: List[ClusterResult] = []
    
    
    def process_cluster(self, row: pd.Series) -> ClusterResult:
        """
        Process a single cluster.
        
        Parameters
        ----------
        row : pd.Series
            Cluster catalog row
        
        Returns
        -------
        result : ClusterResult
            Cluster prediction results
        """
        cluster_name = row['cluster_name']
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Processing: {cluster_name}")
            print(f"{'='*70}")
        
        # Extract parameters
        M_500 = row['M_500_Msun']
        R_500 = row['R_500_kpc']
        z_lens = row['z_lens']
        z_src = row['z_source']
        theta_E_obs = row['theta_E_obs_arcsec']
        theta_E_err = row['theta_E_err_arcsec']
        fgas_target = row['fgas_R500']
        T_keV = row['TX_central_keV']
        
        try:
            # 1. Build baryon model
            r_grid = np.logspace(-1, 3.5, 2000)  # 0.1 to ~3000 kpc
            
            baryon_params = ClusterBaryonParams(
                M_500=M_500,
                R_500=R_500,
                z=z_lens,
                fgas_target=fgas_target,
                T_keV=T_keV
            )
            
            components = build_cluster_baryon_model(
                r_grid, baryon_params, verbose=False
            )
            
            # 2. Compute lensing observables
            # Define R grid for lensing (projected radii)
            R = np.logspace(1, 3.2, 300)  # 10 to ~1500 kpc
            
            # Import cosmology
            from many_path_model.lensing_utilities import LensingCosmology
            cosmo = LensingCosmology()
            
            lensing = lensing_profiles_3d_shell(
                R=R,
                z_lens=z_lens,
                z_src=z_src,
                r_grid=r_grid,
                rho_3d=components.rho_total,
                params=self.kernel_params,
                cosmo=cosmo,
                verbose=self.verbose
            )
            
            # 3. Extract results
            theta_E_pred = lensing['theta_E_arcsec']
            theta_E_residual = (theta_E_pred - theta_E_obs) / theta_E_obs
            theta_E_sigma = (theta_E_pred - theta_E_obs) / theta_E_err
            
            # Boost factor at Einstein radius (if found)
            K_Sigma_at_RE = 0.0
            if theta_E_pred > 0:
                # Convert theta_E (arcsec) to physical radius (kpc)
                # R = theta × D_A, where theta in radians
                D_A_kpc = cosmo.angular_diameter_distance_kpc(z_lens)
                theta_E_rad = theta_E_pred / (180.0 * 3600.0 / np.pi)  # arcsec to radians
                R_E = theta_E_rad * D_A_kpc  # physical radius in kpc
                # Find closest R grid point
                idx_RE = np.argmin(np.abs(R - R_E))
                K_Sigma_at_RE = lensing['K_Sigma'][idx_RE]
            
            result = ClusterResult(
                cluster_name=cluster_name,
                z_lens=z_lens,
                M_500=M_500,
                R_500=R_500,
                theta_E_obs=theta_E_obs,
                theta_E_err=theta_E_err,
                theta_E_pred=theta_E_pred,
                theta_E_residual=theta_E_residual,
                theta_E_sigma=theta_E_sigma,
                kappa_max=np.max(lensing['kappa']),
                mean_kappa_max=np.max(lensing['mean_kappa']),
                K_Sigma_at_RE=K_Sigma_at_RE,
                fgas_R500=components.info['fgas_R500'],
                fbaryon_R500=components.info['fbaryon_R500'],
                M_baryon=components.info['M_baryon_R500'],
                converged=True,
                notes="Success"
            )
            
            if self.verbose:
                print(f"\n[OK] {cluster_name} complete:")
                print(f"  theta_E(obs) = {theta_E_obs:.1f}\" +/- {theta_E_err:.1f}\"")
                print(f"  theta_E(pred) = {theta_E_pred:.1f}\"")
                print(f"  Residual = {theta_E_residual*100:+.1f}%")
                print(f"  K_Sigma(R_E) = {K_Sigma_at_RE:.2f}")
                print(f"  <kappa>_max = {result.mean_kappa_max:.3f}")
        
        except Exception as e:
            # Record failure
            result = ClusterResult(
                cluster_name=cluster_name,
                z_lens=z_lens,
                M_500=M_500,
                R_500=R_500,
                theta_E_obs=theta_E_obs,
                theta_E_err=theta_E_err,
                theta_E_pred=0.0,
                theta_E_residual=np.nan,
                theta_E_sigma=np.nan,
                kappa_max=0.0,
                mean_kappa_max=0.0,
                K_Sigma_at_RE=0.0,
                fgas_R500=0.0,
                fbaryon_R500=0.0,
                M_baryon=0.0,
                converged=False,
                notes=f"Error: {str(e)}"
            )
            
            if self.verbose:
                print(f"\n[FAIL] {cluster_name} FAILED:")
                print(f"  {str(e)}")
        
        return result
    
    
    def run_suite(self):
        """Process all clusters in catalog."""
        print(f"\n{'='*70}")
        print(f"Running Blind Cluster Validation Suite")
        print(f"{'='*70}")
        print(f"\nFrozen Kernel Parameters:")
        print(f"  A_c = {self.kernel_params.A_c}")
        print(f"  ell0 = {self.kernel_params.ell0} kpc")
        print(f"  p_density = {self.kernel_params.p_density}")
        print(f"  w_interior = {self.kernel_params.w_interior}")
        print(f"  w_exterior = {self.kernel_params.w_exterior}")
        print(f"  n_coh = {self.kernel_params.n_coh}")
        print()
        
        for idx, row in self.catalog.iterrows():
            result = self.process_cluster(row)
            self.results.append(result)
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    
    def save_results(self):
        """Save per-cluster results and summary statistics."""
        # Per-cluster JSON
        results_list = [asdict(r) for r in self.results]
        
        with open(self.output_dir / 'per_cluster_results.json', 'w') as f:
            json.dump(results_list, f, indent=2)
        
        # Per-cluster CSV
        df = pd.DataFrame(results_list)
        df.to_csv(self.output_dir / 'per_cluster_results.csv', index=False)
        
        # Summary statistics
        converged = [r for r in self.results if r.converged]
        
        if len(converged) > 0:
            residuals = [r.theta_E_residual for r in converged]
            abs_residuals = [abs(r) for r in residuals]
            
            summary = {
                'n_clusters': len(self.results),
                'n_converged': len(converged),
                'n_failed': len(self.results) - len(converged),
                'theta_E_residuals': {
                    'median': float(np.median(residuals)),
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'median_abs': float(np.median(abs_residuals)),
                    'within_10pct': int(np.sum(np.array(abs_residuals) < 0.10)),
                    'within_15pct': int(np.sum(np.array(abs_residuals) < 0.15)),
                    'within_20pct': int(np.sum(np.array(abs_residuals) < 0.20)),
                },
                'kernel_params': asdict(self.kernel_params)
            }
            
            with open(self.output_dir / 'summary_statistics.json', 'w') as f:
                json.dump(summary, f, indent=2)
        
        print(f"\n[OK] Results saved to {self.output_dir}")
    
    
    def print_summary(self):
        """Print summary statistics."""
        converged = [r for r in self.results if r.converged]
        
        if len(converged) == 0:
            print("\n[FAIL] No successful predictions!")
            return
        
        residuals = np.array([r.theta_E_residual for r in converged])
        abs_residuals = np.abs(residuals)
        
        print(f"\n{'='*70}")
        print(f"BLIND VALIDATION SUITE SUMMARY")
        print(f"{'='*70}")
        print(f"\nSample:")
        print(f"  Total clusters: {len(self.results)}")
        print(f"  Converged: {len(converged)}")
        print(f"  Failed: {len(self.results) - len(converged)}")
        
        print(f"\nEinstein Radius Residuals:")
        print(f"  Median: {np.median(residuals)*100:+.1f}%")
        print(f"  Mean: {np.mean(residuals)*100:+.1f}%")
        print(f"  Std: {np.std(residuals)*100:.1f}%")
        print(f"  Median |residual|: {np.median(abs_residuals)*100:.1f}%")
        
        print(f"\nFraction within tolerance:")
        print(f"  +/-10%: {np.sum(abs_residuals < 0.10) / len(converged)*100:.1f}%")
        print(f"  +/-15%: {np.sum(abs_residuals < 0.15) / len(converged)*100:.1f}%")
        print(f"  +/-20%: {np.sum(abs_residuals < 0.20) / len(converged)*100:.1f}%")
        
        print(f"\n{'='*70}")
        print(f"TARGET: Median <=15%, >=60% within +/-20%")
        
        # Check if target met
        median_ok = np.median(abs_residuals) <= 0.15
        coverage_ok = np.sum(abs_residuals < 0.20) / len(converged) >= 0.60
        
        if median_ok and coverage_ok:
            print(f"[OK] TARGETS MET!")
        else:
            print(f"[WARN] Targets not yet met - continue optimization")
        
        print(f"{'='*70}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run blind cluster validation suite')
    parser.add_argument('--catalog', type=str, 
                       default='data/clusters/master_catalog.csv',
                       help='Path to master cluster catalog')
    parser.add_argument('--out_dir', type=str,
                       default='results/cluster_suite',
                       help='Output directory')
    parser.add_argument('--holdout_fraction', type=float, default=0.0,
                       help='Fraction of clusters to hold out for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress per-cluster output')
    
    args = parser.parse_args()
    
    # Run suite
    suite = ClusterValidationSuite(
        catalog_path=args.catalog,
        output_dir=args.out_dir,
        holdout_fraction=args.holdout_fraction,
        random_seed=args.seed,
        verbose=not args.quiet
    )
    
    suite.run_suite()


if __name__ == '__main__':
    main()
