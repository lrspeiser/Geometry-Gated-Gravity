#!/usr/bin/env python3
"""
Hierarchical Cluster Calibration Framework
==========================================

Implements a rigorous, baryons-only, geometry-aware calibration for the
many-paths cluster kernel following best practices:

1. **Global (shared) kernel parameters** - Universal cluster physics:
   - A_c (amplitude), ell0 (coherence length), p_density, n_coh
   - Interior/exterior family weights with sparsity prior on w_exterior
   
2. **Per-cluster nuisance parameters** - Geometry & ICM systematics:
   - Triaxial shape: q_plane (in-plane), q_LOS (line-of-sight flattening)
   - Clumping: (C0, C_max, eta) with hierarchical priors
   - BCG/ICL masses with photometry-informed priors
   
3. **Joint loss function** breaking key degeneracies:
   - Strong lensing: theta_E (sets normalization)
   - Weak lensing: gamma_t(R) (breaks q_LOS vs A_c degeneracy)
   - Optional: X-ray/SZ gNFW prior (stabilizes outer profile)

4. **Hierarchical (partial-pooling) inference**:
   - Kernel params learned from ALL clusters (borrowing strength)
   - Cluster nuisances fitted per-cluster with population priors
   - Train/hold-out validation (no tuning on hold-outs)

Key Design:
-----------
- NO dark matter anywhere in the pipeline
- Keeps kernel universal (one physics for all clusters)
- Lets geometry vary cluster-by-cluster in controlled, interpretable way
- Uses independent data (weak lensing) to prevent shape-amplitude masquerade

References:
-----------
User guidance 2025-01-14: "Best practice hierarchical calibration"
Simionescu+ 2011: Clumping measurements
Eckert+ 2015: Radial clumping profiles

Author: GravityCalculator
Date: 2025-01-14
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, laplace
import json
from pathlib import Path


@dataclass
class GlobalKernelParams:
    """
    Universal cluster kernel hyperparameters.
    
    These are SHARED across all clusters and define the fundamental
    many-paths physics. Learned from entire population.
    """
    # Core kernel parameters
    A_c: float = 10.0  # Cluster amplitude
    ell0: float = 180.0  # Coherence length [kpc]
    p_density: float = 1.2  # Density-dependent constructive interference
    n_coh: float = 1.5  # Coherence damping exponent
    
    # Gate parameters (typically frozen)
    r_gate: float = 5.0  # Gate radius [kpc]
    n_gate: float = 4.0  # Gate sharpness
    
    # Taper parameters (typically frozen)
    L1: float = 1200.0  # Taper scale [kpc]
    q_taper: float = 2.0  # Taper power
    
    # Path family weights (KEY: fit w_exterior with sparsity prior)
    w_interior: float = 1.0  # Interior chords (typically = 1)
    w_exterior: float = 0.0  # Exterior arcs (fit with Laplace prior at 0)
    
    # Coherence mode
    coherence_mode: str = 'power_law'
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ClusterGeometryParams:
    """
    Per-cluster nuisance parameters for geometry and systematics.
    
    These vary cluster-by-cluster but share population-level priors
    (hierarchical inference / partial pooling).
    """
    # Triaxial shape
    q_plane: float = 0.9  # In-plane axis ratio (typically 0.6-1.0)
    q_LOS: float = 1.0  # Line-of-sight flattening (0.6-1.6 for mergers/cores)
    
    # Clumping (hierarchical with population mean ~ literature)
    C0: float = 1.3  # Core clumping
    C_max: float = 2.5  # Outskirts clumping
    eta_clump: float = 2.0  # Radial exponent
    
    # BCG/ICL systematics (mildly informative from photometry)
    f_BCG_scatter: float = 1.0  # Multiplicative scatter around scaling relation
    f_ICL_scatter: float = 1.0  # Multiplicative scatter around scaling relation
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ObservationalData:
    """
    Observational constraints for a single cluster.
    """
    # Strong lensing
    theta_E_obs: float  # Einstein radius [arcsec]
    theta_E_err: float  # Error [arcsec]
    
    # Weak lensing (optional but highly recommended)
    gamma_t_R: Optional[np.ndarray] = None  # Radii [kpc]
    gamma_t_obs: Optional[np.ndarray] = None  # Tangential shear
    gamma_t_err: Optional[np.ndarray] = None  # Errors
    
    # X-ray/SZ (optional, for gNFW prior)
    has_xray_sz: bool = False
    
    def has_weak_lensing(self) -> bool:
        return (self.gamma_t_R is not None and 
                self.gamma_t_obs is not None and
                self.gamma_t_err is not None)


@dataclass
class HierarchicalPriors:
    """
    Population-level priors for hierarchical inference.
    """
    # Global kernel priors (tight, from calibration cluster)
    A_c_prior: Tuple[float, float] = (10.0, 2.0)  # (mean, std)
    ell0_prior: Tuple[float, float] = (180.0, 30.0)
    p_density_prior: Tuple[float, float] = (1.2, 0.2)
    n_coh_prior: Tuple[float, float] = (1.5, 0.3)
    
    # w_exterior: Laplace (sparsity) prior centered at 0
    w_exterior_scale: float = 0.1  # Laplace scale (encourages 0)
    
    # Per-cluster geometry priors (broad, let data decide)
    q_plane_prior: Tuple[float, float] = (0.85, 0.15)  # Prefer mildly oblate
    q_LOS_prior: Tuple[float, float] = (1.0, 0.3)  # Broad around spherical
    
    # Clumping population priors (from literature)
    C0_prior: Tuple[float, float] = (1.3, 0.2)
    C_max_prior: Tuple[float, float] = (2.5, 0.3)
    eta_clump_prior: Tuple[float, float] = (2.0, 0.3)
    
    # BCG/ICL scatter (log-normal, ~30-50% scatter)
    f_scatter_prior: Tuple[float, float] = (1.0, 0.4)


class HierarchicalClusterCalibration:
    """
    Main calibration class implementing hierarchical inference.
    
    Usage:
    ------
    1. Initialize with cluster catalog and observational data
    2. Specify train/hold-out split
    3. Run hierarchical fit (global kernel + per-cluster geometry)
    4. Validate on hold-out set with frozen kernel
    5. Report posteriors and diagnostics
    """
    
    def __init__(
        self,
        catalog: pd.DataFrame,
        obs_data: Dict[str, ObservationalData],
        priors: Optional[HierarchicalPriors] = None,
        verbose: bool = True
    ):
        """
        Initialize hierarchical calibration.
        
        Parameters
        ----------
        catalog : DataFrame
            Cluster catalog with M_500, R_500, z, etc.
        obs_data : dict
            Maps cluster_name -> ObservationalData
        priors : HierarchicalPriors, optional
            Population priors (defaults to literature-motivated)
        verbose : bool
            Print progress
        """
        self.catalog = catalog
        self.obs_data = obs_data
        self.priors = priors if priors is not None else HierarchicalPriors()
        self.verbose = verbose
        
        # Will be populated by fit
        self.global_kernel_best: Optional[GlobalKernelParams] = None
        self.cluster_geometry_best: Dict[str, ClusterGeometryParams] = {}
        self.fit_results: Dict = {}
        
        # Train/hold-out split
        self.train_clusters: List[str] = []
        self.holdout_clusters: List[str] = []
    
    
    def set_train_holdout_split(
        self,
        holdout_fraction: float = 0.25,
        random_seed: int = 42
    ):
        """
        Randomly split clusters into train/hold-out sets.
        
        Parameters
        ----------
        holdout_fraction : float
            Fraction to hold out (default: 0.25 = 25%)
        random_seed : int
            Random seed for reproducibility
        """
        np.random.seed(random_seed)
        all_clusters = list(self.catalog['cluster_name'])
        n_holdout = max(1, int(len(all_clusters) * holdout_fraction))
        
        holdout_idx = np.random.choice(
            len(all_clusters), n_holdout, replace=False
        )
        
        self.holdout_clusters = [all_clusters[i] for i in holdout_idx]
        self.train_clusters = [c for c in all_clusters if c not in self.holdout_clusters]
        
        if self.verbose:
            print(f"\nTrain/Hold-out Split:")
            print(f"  Train: {len(self.train_clusters)} clusters")
            print(f"  Hold-out: {len(self.holdout_clusters)} clusters")
            print(f"  Hold-out names: {self.holdout_clusters}")
            print()
    
    
    def joint_loss(
        self,
        params_flat: np.ndarray,
        cluster_name: str,
        global_kernel: GlobalKernelParams,
        return_components: bool = False
    ) -> float:
        """
        Joint loss function for a single cluster.
        
        Combines:
        1. Strong lensing chi^2 (theta_E)
        2. Weak lensing chi^2 (gamma_t profile)
        3. gNFW prior chi^2 (optional, if X-ray/SZ available)
        4. Regularization from population priors
        
        Parameters
        ----------
        params_flat : array
            Flattened per-cluster geometry parameters
        cluster_name : str
            Cluster identifier
        global_kernel : GlobalKernelParams
            Current global kernel (frozen during per-cluster fit)
        return_components : bool
            If True, return dict of loss components
        
        Returns
        -------
        loss : float
            Total chi^2 + priors
        """
        # Unpack geometry params
        geom = self._unflatten_geometry_params(params_flat)
        
        # Get observations
        obs = self.obs_data[cluster_name]
        row = self.catalog[self.catalog['cluster_name'] == cluster_name].iloc[0]
        
        # Compute predictions (this will call lensing_profiles_3d_shell)
        theta_E_pred, gamma_t_pred = self._predict_lensing(
            row, global_kernel, geom
        )
        
        # 1. Strong lensing chi^2
        chi2_theta_E = ((theta_E_pred - obs.theta_E_obs) / obs.theta_E_err)**2
        
        # 2. Weak lensing chi^2 (if available)
        chi2_gamma_t = 0.0
        if obs.has_weak_lensing():
            residuals = (gamma_t_pred - obs.gamma_t_obs) / obs.gamma_t_err
            chi2_gamma_t = np.sum(residuals**2)
        
        # 3. gNFW prior chi^2 (optional, placeholder for now)
        chi2_gnfw = 0.0
        
        # 4. Geometry priors (population-level regularization)
        prior_penalty = 0.0
        
        # q_plane prior
        q_mean, q_std = self.priors.q_plane_prior
        prior_penalty += ((geom.q_plane - q_mean) / q_std)**2
        
        # q_LOS prior
        qlos_mean, qlos_std = self.priors.q_LOS_prior
        prior_penalty += ((geom.q_LOS - qlos_mean) / qlos_std)**2
        
        # Clumping priors
        c0_mean, c0_std = self.priors.C0_prior
        prior_penalty += ((geom.C0 - c0_mean) / c0_std)**2
        
        cmax_mean, cmax_std = self.priors.C_max_prior
        prior_penalty += ((geom.C_max - cmax_mean) / cmax_std)**2
        
        # BCG/ICL scatter priors (log-normal)
        f_mean, f_std = self.priors.f_scatter_prior
        prior_penalty += ((np.log(geom.f_BCG_scatter) - np.log(f_mean)) / (f_std/f_mean))**2
        prior_penalty += ((np.log(geom.f_ICL_scatter) - np.log(f_mean)) / (f_std/f_mean))**2
        
        # Total loss (with weighting)
        w_wl = 1.0 if obs.has_weak_lensing() else 0.0
        w_gnfw = 0.1 if obs.has_xray_sz else 0.0
        w_prior = 0.5  # Moderate regularization
        
        total_loss = (
            chi2_theta_E +
            w_wl * chi2_gamma_t +
            w_gnfw * chi2_gnfw +
            w_prior * prior_penalty
        )
        
        if return_components:
            return {
                'total': total_loss,
                'chi2_theta_E': chi2_theta_E,
                'chi2_gamma_t': chi2_gamma_t,
                'chi2_gnfw': chi2_gnfw,
                'prior_penalty': prior_penalty
            }
        
        return total_loss
    
    
    def _predict_lensing(
        self,
        cluster_row: pd.Series,
        global_kernel: GlobalKernelParams,
        geometry: ClusterGeometryParams
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Predict lensing observables for a cluster.
        
        This is a placeholder that should call the actual lensing code.
        In practice, this would:
        1. Build triaxial baryon model with geometry params
        2. Apply 3D shell kernel with global kernel params
        3. Compute theta_E and gamma_t(R)
        
        Parameters
        ----------
        cluster_row : Series
            Cluster properties (M_500, R_500, z, etc.)
        global_kernel : GlobalKernelParams
            Universal kernel parameters
        geometry : ClusterGeometryParams
            Per-cluster geometry/systematics
        
        Returns
        -------
        theta_E_pred : float
            Predicted Einstein radius [arcsec]
        gamma_t_pred : array or None
            Predicted tangential shear profile (if obs available)
        """
        # TODO: Implement actual lensing computation
        # For now, return placeholder
        theta_E_pred = 20.0  # Placeholder
        gamma_t_pred = None
        
        return theta_E_pred, gamma_t_pred
    
    
    def _flatten_geometry_params(self, geom: ClusterGeometryParams) -> np.ndarray:
        """Convert ClusterGeometryParams to flat array for optimization."""
        return np.array([
            geom.q_plane,
            geom.q_LOS,
            geom.C0,
            geom.C_max,
            geom.eta_clump,
            np.log(geom.f_BCG_scatter),  # Log-space for positivity
            np.log(geom.f_ICL_scatter)
        ])
    
    
    def _unflatten_geometry_params(self, params_flat: np.ndarray) -> ClusterGeometryParams:
        """Convert flat array back to ClusterGeometryParams."""
        return ClusterGeometryParams(
            q_plane=params_flat[0],
            q_LOS=params_flat[1],
            C0=params_flat[2],
            C_max=params_flat[3],
            eta_clump=params_flat[4],
            f_BCG_scatter=np.exp(params_flat[5]),
            f_ICL_scatter=np.exp(params_flat[6])
        )
    
    
    def fit_hierarchical(
        self,
        n_iterations: int = 3,
        global_method: str = 'differential_evolution'
    ):
        """
        Main hierarchical fitting procedure.
        
        Alternates between:
        1. Fitting global kernel (with per-cluster geometries frozen)
        2. Fitting per-cluster geometries (with global kernel frozen)
        
        Iterates until convergence or max iterations.
        
        Parameters
        ----------
        n_iterations : int
            Number of alternating iterations
        global_method : str
            Optimization method for global kernel ('differential_evolution' or 'minimize')
        """
        if len(self.train_clusters) == 0:
            raise ValueError("Must call set_train_holdout_split() first!")
        
        if self.verbose:
            print("="*70)
            print("HIERARCHICAL CLUSTER CALIBRATION")
            print("="*70)
            print(f"\nTraining on {len(self.train_clusters)} clusters")
            print(f"Iterations: {n_iterations}")
            print()
        
        # Initialize with priors
        self.global_kernel_best = GlobalKernelParams(
            A_c=self.priors.A_c_prior[0],
            ell0=self.priors.ell0_prior[0],
            p_density=self.priors.p_density_prior[0],
            n_coh=self.priors.n_coh_prior[0],
            w_exterior=0.0  # Start with interior-only
        )
        
        # Initialize per-cluster geometries
        for cluster_name in self.train_clusters:
            self.cluster_geometry_best[cluster_name] = ClusterGeometryParams()
        
        # Alternating optimization
        for iteration in range(n_iterations):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Iteration {iteration+1}/{n_iterations}")
                print(f"{'='*70}")
            
            # Step 1: Optimize per-cluster geometries (global kernel frozen)
            if self.verbose:
                print("\nStep 1: Fitting per-cluster geometries...")
            self._fit_cluster_geometries(self.global_kernel_best)
            
            # Step 2: Optimize global kernel (geometries frozen)
            if self.verbose:
                print("\nStep 2: Fitting global kernel...")
            self._fit_global_kernel(global_method)
        
        # Final evaluation
        self._evaluate_fit()
    
    
    def _fit_cluster_geometries(self, global_kernel: GlobalKernelParams):
        """
        Fit per-cluster geometry parameters with global kernel frozen.
        """
        for cluster_name in self.train_clusters:
            if self.verbose:
                print(f"  Fitting {cluster_name}...", end=' ')
            
            # Initial guess from current best
            geom_init = self.cluster_geometry_best[cluster_name]
            x0 = self._flatten_geometry_params(geom_init)
            
            # Bounds (physical constraints)
            bounds = [
                (0.6, 1.0),  # q_plane
                (0.6, 1.6),  # q_LOS
                (1.0, 1.8),  # C0
                (1.8, 3.5),  # C_max
                (1.0, 3.0),  # eta_clump
                (np.log(0.5), np.log(2.0)),  # log(f_BCG_scatter)
                (np.log(0.5), np.log(2.0))   # log(f_ICL_scatter)
            ]
            
            # Optimize
            result = minimize(
                lambda x: self.joint_loss(x, cluster_name, global_kernel),
                x0,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            # Store best
            self.cluster_geometry_best[cluster_name] = self._unflatten_geometry_params(result.x)
            
            if self.verbose:
                print(f"loss = {result.fun:.2f}")
    
    
    def _fit_global_kernel(self, method: str = 'differential_evolution'):
        """
        Fit global kernel parameters with per-cluster geometries frozen.
        
        Uses population-level loss (sum over all train clusters) plus
        sparsity prior on w_exterior.
        """
        def population_loss(params_flat):
            """Loss summed over all train clusters plus global priors."""
            # Unpack global kernel
            kernel = GlobalKernelParams(
                A_c=params_flat[0],
                ell0=params_flat[1],
                p_density=params_flat[2],
                n_coh=params_flat[3],
                w_exterior=params_flat[4]
            )
            
            # Sum per-cluster losses
            total_loss = 0.0
            for cluster_name in self.train_clusters:
                geom = self.cluster_geometry_best[cluster_name]
                geom_flat = self._flatten_geometry_params(geom)
                total_loss += self.joint_loss(geom_flat, cluster_name, kernel)
            
            # Add global priors
            # Gaussian priors on A_c, ell0, p_density, n_coh
            a_mean, a_std = self.priors.A_c_prior
            total_loss += ((kernel.A_c - a_mean) / a_std)**2
            
            ell_mean, ell_std = self.priors.ell0_prior
            total_loss += ((kernel.ell0 - ell_mean) / ell_std)**2
            
            p_mean, p_std = self.priors.p_density_prior
            total_loss += ((kernel.p_density - p_mean) / p_std)**2
            
            n_mean, n_std = self.priors.n_coh_prior
            total_loss += ((kernel.n_coh - n_mean) / n_std)**2
            
            # Laplace (sparsity) prior on w_exterior
            total_loss += np.abs(kernel.w_exterior) / self.priors.w_exterior_scale
            
            return total_loss
        
        # Bounds
        bounds = [
            (5.0, 20.0),  # A_c
            (100.0, 300.0),  # ell0
            (0.8, 1.8),  # p_density
            (1.0, 2.5),  # n_coh
            (0.0, 0.5)   # w_exterior (small values due to sparsity prior)
        ]
        
        if method == 'differential_evolution':
            result = differential_evolution(
                population_loss,
                bounds,
                maxiter=50,
                popsize=10,
                disp=self.verbose
            )
        else:
            x0 = np.array([
                self.global_kernel_best.A_c,
                self.global_kernel_best.ell0,
                self.global_kernel_best.p_density,
                self.global_kernel_best.n_coh,
                self.global_kernel_best.w_exterior
            ])
            result = minimize(
                population_loss,
                x0,
                method='L-BFGS-B',
                bounds=bounds
            )
        
        # Update global kernel
        self.global_kernel_best = GlobalKernelParams(
            A_c=result.x[0],
            ell0=result.x[1],
            p_density=result.x[2],
            n_coh=result.x[3],
            w_exterior=result.x[4]
        )
        
        if self.verbose:
            print(f"  Global kernel updated:")
            print(f"    A_c = {self.global_kernel_best.A_c:.2f}")
            print(f"    ell0 = {self.global_kernel_best.ell0:.1f} kpc")
            print(f"    w_exterior = {self.global_kernel_best.w_exterior:.3f}")
    
    
    def _evaluate_fit(self):
        """Evaluate final fit quality on train set."""
        if self.verbose:
            print(f"\n{'='*70}")
            print("FINAL FIT EVALUATION (Train Set)")
            print(f"{'='*70}\n")
        
        theta_E_residuals = []
        
        for cluster_name in self.train_clusters:
            obs = self.obs_data[cluster_name]
            row = self.catalog[self.catalog['cluster_name'] == cluster_name].iloc[0]
            geom = self.cluster_geometry_best[cluster_name]
            
            theta_E_pred, _ = self._predict_lensing(
                row, self.global_kernel_best, geom
            )
            
            residual = (theta_E_pred - obs.theta_E_obs) / obs.theta_E_obs
            theta_E_residuals.append(residual)
            
            if self.verbose:
                print(f"{cluster_name:12s}: "
                      f"pred={theta_E_pred:.2f}\" "
                      f"obs={obs.theta_E_obs:.2f}\" "
                      f"residual={residual*100:+.1f}%")
        
        theta_E_residuals = np.array(theta_E_residuals)
        
        if self.verbose:
            print(f"\nTrain Summary:")
            print(f"  Median residual: {np.median(theta_E_residuals)*100:.1f}%")
            print(f"  MAD: {np.median(np.abs(theta_E_residuals))*100:.1f}%")
            print(f"  Within ±10%: {np.sum(np.abs(theta_E_residuals)<0.10)}/{len(theta_E_residuals)}")
            print(f"  Within ±20%: {np.sum(np.abs(theta_E_residuals)<0.20)}/{len(theta_E_residuals)}")
    
    
    def validate_holdout(self):
        """
        Validate on hold-out set with FROZEN global kernel.
        
        Per-cluster geometries are fitted, but global kernel is fixed.
        This tests generalization of the universal physics.
        """
        if len(self.holdout_clusters) == 0:
            print("No hold-out clusters defined!")
            return
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("HOLD-OUT VALIDATION (Frozen Global Kernel)")
            print(f"{'='*70}\n")
        
        # Fit geometries for hold-out clusters (kernel frozen)
        for cluster_name in self.holdout_clusters:
            if cluster_name not in self.cluster_geometry_best:
                self.cluster_geometry_best[cluster_name] = ClusterGeometryParams()
        
        self._fit_cluster_geometries(self.global_kernel_best)
        
        # Evaluate
        theta_E_residuals = []
        
        for cluster_name in self.holdout_clusters:
            obs = self.obs_data[cluster_name]
            row = self.catalog[self.catalog['cluster_name'] == cluster_name].iloc[0]
            geom = self.cluster_geometry_best[cluster_name]
            
            theta_E_pred, _ = self._predict_lensing(
                row, self.global_kernel_best, geom
            )
            
            residual = (theta_E_pred - obs.theta_E_obs) / obs.theta_E_obs
            theta_E_residuals.append(residual)
            
            if self.verbose:
                print(f"{cluster_name:12s}: "
                      f"pred={theta_E_pred:.2f}\" "
                      f"obs={obs.theta_E_obs:.2f}\" "
                      f"residual={residual*100:+.1f}%")
        
        theta_E_residuals = np.array(theta_E_residuals)
        
        if self.verbose:
            print(f"\nHold-out Summary:")
            print(f"  Median residual: {np.median(theta_E_residuals)*100:.1f}%")
            print(f"  MAD: {np.median(np.abs(theta_E_residuals))*100:.1f}%")
    
    
    def save_results(self, output_dir: Path):
        """Save calibration results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Global kernel
        with open(output_dir / 'global_kernel.json', 'w') as f:
            json.dump(self.global_kernel_best.to_dict(), f, indent=2)
        
        # Per-cluster geometries
        geometries = {
            name: geom.to_dict()
            for name, geom in self.cluster_geometry_best.items()
        }
        with open(output_dir / 'cluster_geometries.json', 'w') as f:
            json.dump(geometries, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    print("Hierarchical Cluster Calibration Framework")
    print("=" * 70)
    print("\nThis module implements rigorous, baryons-only calibration")
    print("with global kernel parameters and per-cluster geometry.")
    print("\nSee docstrings for usage examples.")
