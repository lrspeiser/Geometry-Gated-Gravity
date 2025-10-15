#!/usr/bin/env python3
"""
Hierarchical Bayesian Calibration with Mass-Scaled Coherence Length
====================================================================

Extension of hierarchical_cluster_model.py that includes mass-scaling:
    ℓ₀ᵢ(M) = ℓ₀,⋆ × (R₅₀₀,ᵢ / 1 Mpc)^γ
    log A_cᵢ = μ_A + γ_cc·CC + γ_w·log(w) + γ_c·log(c₅₀₀) + εᵢ

Key Innovation:
- Population-level (ℓ₀,⋆, γ) instead of per-cluster ℓ₀ᵢ
- Reduces free parameters: 2 population params vs n_clusters individual params
- Enables cross-scale predictions (dwarf galaxies to superclusters)
- Tests whether coherence scales with halo mass

Mathematical Framework:
- γ = 0 → fixed coherence scale (fundamental length)
- γ > 0 → mass-dependent coherence (self-similar scaling)
- Hierarchical priors on (ℓ₀,⋆, γ) with physical constraints

Author: GravityCalculator - Mass-Scaling Extension
Date: 2025-01-19
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings

# Try to import PyMC for full Bayesian inference
try:
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn("PyMC not available - using scipy optimization fallback")

from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class GeometryPredictors:
    """Geometry and morphology predictors for each cluster."""
    
    # Mass proxy (NEW - required for mass-scaling)
    R_500: float = 1200.0        # R_500 [kpc] - cluster scale radius
    M_500: float = 1e15          # M_500 [Msun] - optional, can derive from R500
    
    # Shape/orientation
    q_los: float = 1.0           # LOS axis ratio (from lensing or prior)
    ellipticity: float = 0.0     # Projected ellipticity
    
    # Morphology/relaxation
    centroid_shift: float = 0.0  # w = offset/R_500
    power_ratio: float = 0.0     # P₃/P₀ (power in m=3 mode)
    cool_core: bool = False      # Cool-core flag
    
    # Thermodynamics  
    T_X: float = 5.0             # X-ray temperature [keV]
    sigma_v: float = 1000.0      # Velocity dispersion [km/s]
    
    # Structure
    c_500: float = 3.0           # Concentration c₅₀₀
    z: float = 0.3               # Redshift


@dataclass  
class ClusterKernelParams:
    """Cluster-specific kernel parameters (to be learned)."""
    L0: float = 180.0                # Coherence length [kpc] (computed from mass-scaling)
    log_A_c: float = np.log(10.0)    # log amplitude
    logit_w_ext: float = -3.0        # logit exterior weight (0.05)
    
    # Mass-scaling diagnostics (stored for provenance)
    ell0_star_used: Optional[float] = None
    gamma_used: Optional[float] = None
    R500_used: Optional[float] = None
    
    @property
    def A_c(self) -> float:
        return np.exp(self.log_A_c)
    
    @property
    def w_ext(self) -> float:
        return 1.0 / (1.0 + np.exp(-self.logit_w_ext))


@dataclass
class HyperParameters:
    """Population-level hyperparameters with mass-scaled coherence."""
    
    # Mass-scaled coherence length: ℓ₀(M) = ℓ₀,⋆ × (R₅₀₀/1Mpc)^γ
    ell0_star: float = 200.0          # Coherence at pivot mass (1 Mpc) [kpc]
    gamma: float = 0.5                # Mass-scaling exponent (0 = fixed, 1 = linear)
    R500_pivot: float = 1000.0        # Pivot scale [kpc] = 1 Mpc (fixed)
    
    # Additional coherence law (optional, can be removed for simplicity):
    beta_T: float = 0.0               # Temperature dependence (secondary effect)
    beta_e: float = 0.0               # Ellipticity dependence  
    beta_q: float = 0.0               # LOS shape dependence
    beta_z: float = 0.0               # Redshift evolution
    sigma_ell: float = 0.2            # Cluster-to-cluster scatter (residual)
    
    # Amplitude law: log A_c = μ_A + γ_cc·CC + γ_w·log(w) + γ_c·log(c)
    mu_A: float = np.log(10.0)       # Population mean log A_c
    gamma_cc: float = 0.2            # Cool-core boost
    gamma_w: float = -0.3            # Centroid shift penalty (disturbed)
    gamma_c: float = 0.1             # Concentration scaling
    sigma_A: float = 0.2             # Cluster-to-cluster scatter
    
    # Exterior weight law: logit(w_ext) = μ_w + η_q·(q-1) + η_e·e
    mu_w: float = -3.0               # Population mean logit w_ext (~0.05)
    eta_q: float = 2.0               # LOS elongation → less exterior
    eta_e: float = -1.0              # Sky elongation → more exterior
    sigma_w: float = 0.5             # Cluster-to-cluster scatter


class HierarchicalClusterModelMassScaled:
    """
    Hierarchical Bayesian model with mass-scaled coherence length.
    
    This model learns population-level (ℓ₀,⋆, γ, μ_A, σ_A) instead of
    per-cluster (ℓ₀ᵢ, A_cᵢ). The coherence length scales with halo mass:
    
        ℓ₀ᵢ = ℓ₀,⋆ × (R₅₀₀,ᵢ / 1 Mpc)^γ
    
    Key Features:
    - Fewer free parameters: 2 coherence params vs n_clusters
    - Testable prediction: γ > 0 indicates mass-dependent coherence
    - Cross-scale extrapolation: predict ℓ₀ for any halo mass
    - Physics discrimination: γ ≈ 0.5-1.0 favors self-similar many-paths
    """
    
    def __init__(
        self,
        predictors: List[GeometryPredictors],
        observations: Dict[str, np.ndarray],
        use_pymc: bool = True,
        include_secondary_effects: bool = False
    ):
        """
        Initialize hierarchical model with mass-scaling.
        
        Parameters
        ----------
        predictors : List[GeometryPredictors]
            Geometry predictors for each cluster (must include R_500)
        observations : Dict[str, np.ndarray]
            Observations with keys:
            - 'theta_E': Einstein radii [arcsec]
            - 'theta_E_err': Uncertainties [arcsec]
        use_pymc : bool
            Use PyMC for full Bayesian inference (vs scipy MLE)
        include_secondary_effects : bool
            Include secondary coherence modulations (T_X, ellipticity, etc.)
            Set False for simplest mass-scaling model
        """
        self.predictors = predictors
        self.observations = observations
        self.n_clusters = len(predictors)
        self.use_pymc = use_pymc and HAS_PYMC
        self.include_secondary_effects = include_secondary_effects
        
        # Validate R500 is provided for all clusters
        for i, pred in enumerate(predictors):
            if pred.R_500 is None or pred.R_500 <= 0:
                raise ValueError(f"Cluster {i}: R_500 must be provided and positive for mass-scaling")
        
        # Storage for inference results
        self.hyperparams_map: Optional[HyperParameters] = None
        self.cluster_params_map: Optional[List[ClusterKernelParams]] = None
        self.trace = None  # PyMC trace if available
        
    def compute_mass_scaled_coherence(
        self,
        R500: float,
        ell0_star: float,
        gamma: float,
        R500_pivot: float = 1000.0
    ) -> float:
        """
        Compute mass-scaled coherence length ℓ₀(M).
        
        Parameters
        ----------
        R500 : float
            Cluster scale radius [kpc]
        ell0_star : float
            Coherence at pivot mass [kpc]
        gamma : float
            Mass-scaling exponent
        R500_pivot : float
            Pivot scale [kpc], default 1 Mpc
        
        Returns
        -------
        ell0 : float
            Mass-scaled coherence length [kpc]
        """
        return ell0_star * (R500 / R500_pivot)**gamma
        
    def predict_kernel_params(
        self,
        pred: GeometryPredictors,
        hyper: HyperParameters
    ) -> ClusterKernelParams:
        """
        Predict kernel parameters from geometry using learned laws.
        
        Parameters
        ----------
        pred : GeometryPredictors
            Geometry predictors for target cluster
        hyper : HyperParameters
            Population-level hyperparameters (learned laws)
        
        Returns
        -------
        params : ClusterKernelParams
            Predicted kernel parameters
        """
        # Mass-scaled coherence length (PRIMARY effect)
        L0_base = self.compute_mass_scaled_coherence(
            pred.R_500, hyper.ell0_star, hyper.gamma, hyper.R500_pivot
        )
        
        # Optional secondary modulations (temperature, shape, etc.)
        if self.include_secondary_effects:
            log_L0_modulation = (
                hyper.beta_T * np.log(pred.T_X / 5.0)  # Normalized to 5 keV
                + hyper.beta_e * pred.ellipticity
                + hyper.beta_q * (pred.q_los - 1.0)
                + hyper.beta_z * np.log(1.0 + pred.z)
            )
            L0 = L0_base * np.exp(log_L0_modulation)
        else:
            L0 = L0_base
        
        # Amplitude law (unchanged from original)
        log_A_c = (
            hyper.mu_A
            + hyper.gamma_cc * (1.0 if pred.cool_core else 0.0)
            + hyper.gamma_w * np.log(pred.centroid_shift + 0.01)  # Avoid log(0)
            + hyper.gamma_c * np.log(pred.c_500 / 3.0)  # Normalized to c=3
        )
        
        # Exterior weight law (unchanged)
        logit_w_ext = (
            hyper.mu_w
            + hyper.eta_q * (pred.q_los - 1.0)
            + hyper.eta_e * pred.ellipticity
        )
        
        return ClusterKernelParams(
            L0=L0,
            log_A_c=log_A_c,
            logit_w_ext=logit_w_ext,
            ell0_star_used=hyper.ell0_star,
            gamma_used=hyper.gamma,
            R500_used=pred.R_500
        )
    
    def log_likelihood(
        self,
        hyper: HyperParameters,
        cluster_params: List[ClusterKernelParams],
        observations: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute log likelihood of observations given parameters.
        
        NOTE: This is a placeholder. In production, this should call
        the actual lensing forward model (build_cluster_baryons → 
        apply kernel → compute theta_E).
        
        Parameters
        ----------
        hyper : HyperParameters
            Population hyperparameters
        cluster_params : List[ClusterKernelParams]
            Cluster-specific parameters
        observations : Dict
            Observed data
        
        Returns
        -------
        log_like : float
            Log likelihood
        """
        # Simplified likelihood (replace with actual lensing predictions)
        # TODO: Replace with real forward model
        #   from core.build_cluster_baryons import build_baryon_profiles
        #   from core.kernel2d_sigma import convolve_sigma_with_kernel
        #   from lensing.einstein_radius import compute_einstein_radius
        
        log_like = 0.0
        
        # Einstein radius likelihood
        if 'theta_E' in observations:
            theta_E_obs = observations['theta_E']
            theta_E_err = observations['theta_E_err']
            
            # Mock predictions (replace with actual forward model)
            # Scaling: θ_E ∝ A_c × sqrt(L0/R500)
            theta_E_pred = np.array([
                30.0 * (p.A_c / 10.0) * (p.L0 / 180.0)**0.5
                for p in cluster_params
            ])
            
            residuals = (theta_E_pred - theta_E_obs) / theta_E_err
            log_like += -0.5 * np.sum(residuals**2)
        
        return log_like
    
    def log_prior(
        self,
        hyper: HyperParameters,
        cluster_params: List[ClusterKernelParams]
    ) -> float:
        """
        Compute log prior probability with mass-scaling priors.
        
        Parameters
        ----------
        hyper : HyperParameters
            Population hyperparameters
        cluster_params : List[ClusterKernelParams]
            Cluster-specific parameters
        
        Returns
        -------
        log_prior : float
            Log prior probability
        """
        log_p = 0.0
        
        # Mass-scaling priors (PRIMARY parameters)
        # ℓ₀,⋆: TruncatedNormal(200, 100) in [50, 500] kpc
        if 50 <= hyper.ell0_star <= 500:
            log_p += norm.logpdf(hyper.ell0_star, loc=200, scale=100)
        else:
            return -np.inf  # Hard truncation
        
        # γ: TruncatedNormal(0.5, 0.3) in [0, 1.5]
        if 0.0 <= hyper.gamma <= 1.5:
            log_p += norm.logpdf(hyper.gamma, loc=0.5, scale=0.3)
        else:
            return -np.inf
        
        # Secondary coherence modulation priors (if included)
        if self.include_secondary_effects:
            log_p += norm.logpdf(hyper.beta_T, loc=0.0, scale=0.3)
            log_p += norm.logpdf(hyper.beta_q, loc=0.0, scale=0.3)
            log_p += norm.logpdf(hyper.sigma_ell, loc=0.2, scale=0.1)
        
        # Amplitude priors (unchanged)
        log_p += norm.logpdf(hyper.gamma_cc, loc=0.2, scale=0.3)
        log_p += norm.logpdf(hyper.sigma_A, loc=0.2, scale=0.1)
        
        # Exterior weight priors (unchanged)
        log_p += norm.logpdf(hyper.eta_q, loc=2.0, scale=1.0)
        log_p += norm.logpdf(hyper.sigma_w, loc=0.5, scale=0.3)
        
        # Hierarchical priors: cluster params given hyperparams
        for i, (pred, params) in enumerate(zip(self.predictors, cluster_params)):
            # Predicted mean from geometry + mass-scaling
            pred_mean = self.predict_kernel_params(pred, hyper)
            
            # Scatter around predicted L0 (log-normal)
            log_p += norm.logpdf(
                np.log(params.L0), 
                loc=np.log(pred_mean.L0), 
                scale=hyper.sigma_ell
            )
            
            # Scatter around predicted A_c (log-normal)
            log_p += norm.logpdf(
                params.log_A_c, 
                loc=pred_mean.log_A_c, 
                scale=hyper.sigma_A
            )
            
            # Scatter around predicted w_ext (logit-normal)
            log_p += norm.logpdf(
                params.logit_w_ext, 
                loc=pred_mean.logit_w_ext, 
                scale=hyper.sigma_w
            )
        
        return log_p
    
    def fit_mcmc(
        self,
        n_samples: int = 2000,
        n_tune: int = 1000,
        n_chains: int = 4,
        target_accept: float = 0.95
    ):
        """
        Fit hierarchical model using PyMC MCMC sampling.
        
        Parameters
        ----------
        n_samples : int
            Number of posterior samples per chain
        n_tune : int
            Number of tuning steps
        n_chains : int
            Number of parallel chains
        target_accept : float
            Target acceptance rate for NUTS sampler
        
        Returns
        -------
        trace : az.InferenceData
            ArviZ inference data object with posterior samples
        """
        if not self.use_pymc:
            raise RuntimeError("PyMC not available - use fit_map() instead")
        
        print("Building PyMC hierarchical model with mass-scaling...")
        
        with pm.Model() as model:
            # Population-level hyperpriors (mass-scaling)
            ell0_star_pop = pm.TruncatedNormal(
                'ell0_star_pop', mu=200, sigma=100, lower=50, upper=500
            )
            gamma_pop = pm.TruncatedNormal(
                'gamma_pop', mu=0.5, sigma=0.3, lower=0.0, upper=1.5
            )
            
            # Amplitude hyperpriors
            mu_A_pop = pm.Normal('mu_A_pop', mu=np.log(10.0), sigma=0.5)
            sigma_A_pop = pm.HalfNormal('sigma_A_pop', sigma=0.2)
            
            # Extract R500 values
            R500_data = np.array([pred.R_500 for pred in self.predictors])
            
            # Compute mass-scaled ℓ₀ for each cluster (deterministic)
            ell0_clusters = pm.Deterministic(
                'ell0_clusters',
                ell0_star_pop * (R500_data / 1000.0)**gamma_pop
            )
            
            # Per-cluster amplitude (hierarchical)
            log_A_c_clusters = pm.Normal(
                'log_A_c_clusters',
                mu=mu_A_pop,
                sigma=sigma_A_pop,
                shape=self.n_clusters
            )
            
            # Likelihood: Compare predicted vs observed Einstein radii
            # NOTE: Replace this with actual forward model
            theta_E_pred = pm.Deterministic(
                'theta_E_pred',
                30.0 * pm.math.exp(log_A_c_clusters - np.log(10.0)) * (ell0_clusters / 180.0)**0.5
            )
            
            pm.Normal(
                'theta_E_obs',
                mu=theta_E_pred,
                sigma=self.observations['theta_E_err'],
                observed=self.observations['theta_E']
            )
            
            # Sample posterior
            print(f"Sampling {n_samples} × {n_chains} = {n_samples * n_chains} total draws...")
            trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=n_chains,
                target_accept=target_accept,
                return_inferencedata=True
            )
            
            # Posterior predictive checks
            print("Computing posterior predictive...")
            pm.sample_posterior_predictive(trace, extend_inferencedata=True)
        
        self.trace = trace
        
        # Extract MAP estimates for convenience
        self.hyperparams_map = HyperParameters(
            ell0_star=float(trace.posterior['ell0_star_pop'].mean()),
            gamma=float(trace.posterior['gamma_pop'].mean()),
            mu_A=float(trace.posterior['mu_A_pop'].mean()),
            sigma_A=float(trace.posterior['sigma_A_pop'].mean())
        )
        
        print("\nPosterior Summary:")
        print(az.summary(trace, var_names=['ell0_star_pop', 'gamma_pop', 'mu_A_pop', 'sigma_A_pop']))
        
        return trace
    
    def predict_holdout(
        self,
        pred: GeometryPredictors,
        use_posterior: bool = True,
        n_samples: int = 1000
    ) -> Tuple[float, float, float]:
        """
        Predict Einstein radius for hold-out cluster.
        
        Parameters
        ----------
        pred : GeometryPredictors
            Geometry predictors for hold-out cluster (must include R_500)
        use_posterior : bool
            Use full posterior (True) or MAP estimate (False)
        n_samples : int
            Number of posterior samples to use
        
        Returns
        -------
        theta_E_median : float
            Median prediction [arcsec]
        theta_E_lower : float
            16th percentile [arcsec]
        theta_E_upper : float
            84th percentile [arcsec]
        """
        if use_posterior and self.trace is not None:
            # Draw from posterior
            ell0_star_samples = self.trace.posterior['ell0_star_pop'].values.flatten()[:n_samples]
            gamma_samples = self.trace.posterior['gamma_pop'].values.flatten()[:n_samples]
            mu_A_samples = self.trace.posterior['mu_A_pop'].values.flatten()[:n_samples]
            
            # Compute mass-scaled ℓ₀ for hold-out cluster
            ell0_samples = ell0_star_samples * (pred.R_500 / 1000.0)**gamma_samples
            
            # Sample A_c from hierarchical prior
            A_c_samples = np.exp(np.random.normal(mu_A_samples, 0.2, size=n_samples))
            
            # Predict θ_E (mock model - replace with real forward model)
            theta_E_samples = 30.0 * (A_c_samples / 10.0) * (ell0_samples / 180.0)**0.5
            
            return (
                np.median(theta_E_samples),
                np.percentile(theta_E_samples, 16),
                np.percentile(theta_E_samples, 84)
            )
        else:
            # Use MAP estimate
            if self.hyperparams_map is None:
                raise ValueError("No MAP estimate available - run fit_map() or fit_mcmc() first")
            
            params = self.predict_kernel_params(pred, self.hyperparams_map)
            theta_E_pred = 30.0 * (params.A_c / 10.0) * (params.L0 / 180.0)**0.5
            
            # Return point estimate with nominal 20% uncertainty
            return theta_E_pred, theta_E_pred * 0.8, theta_E_pred * 1.2


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 70)
    print("Hierarchical Model with Mass-Scaled Coherence Length")
    print("=" * 70)
    
    # Load cluster metadata
    catalog_path = "C:/Users/henry/dev/GravityCalculator/data/clusters/master_catalog.csv"
    df = pd.read_csv(catalog_path)
    
    # Select Tier-1 + Tier-2 clusters for training (exclude hold-outs)
    train_mask = (df['tier'].isin([1, 2])) & (~df['cluster_name'].isin(['A1689', 'MACS1149']))
    df_train = df[train_mask]
    
    print(f"\nTraining clusters: {len(df_train)}")
    print(df_train[['cluster_name', 'R_500_kpc', 'theta_E_obs_arcsec']].to_string(index=False))
    
    # Build predictors
    predictors = []
    for _, row in df_train.iterrows():
        predictors.append(GeometryPredictors(
            R_500=row['R_500_kpc'],
            M_500=row['M_500_Msun'],
            z=row['z_lens'],
            cool_core=(row['dynamical_state'] == 'relaxed')
        ))
    
    # Observations
    observations = {
        'theta_E': df_train['theta_E_obs_arcsec'].values,
        'theta_E_err': df_train['theta_E_err_arcsec'].values
    }
    
    # Initialize model
    model = HierarchicalClusterModelMassScaled(
        predictors=predictors,
        observations=observations,
        use_pymc=HAS_PYMC,
        include_secondary_effects=False  # Simple mass-scaling only
    )
    
    if HAS_PYMC:
        print("\n" + "="*70)
        print("Running MCMC sampling...")
        print("="*70)
        trace = model.fit_mcmc(n_samples=1000, n_tune=500, n_chains=2)
        
        print("\n" + "="*70)
        print("Posterior Predictive Checks on Hold-Out Clusters")
        print("="*70)
        
        # A1689
        pred_a1689 = GeometryPredictors(
            R_500=df[df['cluster_name'] == 'A1689']['R_500_kpc'].values[0],
            z=df[df['cluster_name'] == 'A1689']['z_lens'].values[0]
        )
        theta_E_a1689_obs = df[df['cluster_name'] == 'A1689']['theta_E_obs_arcsec'].values[0]
        theta_E_a1689_med, theta_E_a1689_lo, theta_E_a1689_hi = model.predict_holdout(pred_a1689)
        
        print(f"\nA1689:")
        print(f"  Observed: {theta_E_a1689_obs:.1f}\"")
        print(f"  Predicted: {theta_E_a1689_med:.1f}\" [{theta_E_a1689_lo:.1f}, {theta_E_a1689_hi:.1f}]\"")
        print(f"  Z-score: {(theta_E_a1689_obs - theta_E_a1689_med) / (theta_E_a1689_hi - theta_E_a1689_lo):.2f}")
        
        # MACS1149
        pred_macs1149 = GeometryPredictors(
            R_500=df[df['cluster_name'] == 'MACS1149']['R_500_kpc'].values[0],
            z=df[df['cluster_name'] == 'MACS1149']['z_lens'].values[0]
        )
        theta_E_macs1149_obs = df[df['cluster_name'] == 'MACS1149']['theta_E_obs_arcsec'].values[0]
        theta_E_macs1149_med, theta_E_macs1149_lo, theta_E_macs1149_hi = model.predict_holdout(pred_macs1149)
        
        print(f"\nMACS1149:")
        print(f"  Observed: {theta_E_macs1149_obs:.1f}\"")
        print(f"  Predicted: {theta_E_macs1149_med:.1f}\" [{theta_E_macs1149_lo:.1f}, {theta_E_macs1149_hi:.1f}]\"")
        print(f"  Z-score: {(theta_E_macs1149_obs - theta_E_macs1149_med) / (theta_E_macs1149_hi - theta_E_macs1149_lo):.2f}")
    else:
        print("\nPyMC not available - skipping MCMC example")
        print("Install with: pip install pymc arviz")
