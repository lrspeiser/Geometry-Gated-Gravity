#!/usr/bin/env python3
"""
Hierarchical Bayesian Calibration for Cluster Path-Integral Kernel
===================================================================

Implements partial-pooling model that learns universal geometry laws:
    log L₀ᵢ = μ_ℓ + β_T·log(T_X/T₀) + β_e·e + β_q·(q_los-1) + εᵢ
    log A_cᵢ = μ_A + γ_cc·CC + γ_w·log(w) + γ_c·log(c₅₀₀) + εᵢ  
    logit(w_extᵢ) = μ_w + η_q·(q_los-1) + η_e·e + εᵢ

Key Innovation:
- Avoids per-cluster overfitting by sharing strength across sample
- Learns HOW geometry controls coherence (not just fits amplitudes)
- Provides blind predictions for new clusters from their geometry alone

Mathematical Framework:
- Partial pooling via hierarchical priors
- Geometry predictors from observables (morphology, temperature, shape)
- Proper regularization to prevent overfitting
- WAIC/LOO-CV for model selection

Author: GravityCalculator - Hierarchical Calibration
Date: 2025-01-14
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings

# Try to import PyMC for full Bayesian inference
# Fall back to scipy optimization if not available
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn("PyMC not available - using scipy optimization fallback")

from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class GeometryPredictors:
    """Geometry and morphology predictors for each cluster."""
    
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
    R_500: float = 1200.0        # R_500 [kpc]
    M_500: float = 1e15          # M_500 [Msun]
    z: float = 0.3               # Redshift


@dataclass  
class ClusterKernelParams:
    """Cluster-specific kernel parameters (to be learned)."""
    log_L0: float = np.log(180.0)    # log coherence length
    log_A_c: float = np.log(10.0)    # log amplitude
    logit_w_ext: float = -3.0        # logit exterior weight (0.05)
    
    @property
    def L0(self) -> float:
        return np.exp(self.log_L0)
    
    @property
    def A_c(self) -> float:
        return np.exp(self.log_A_c)
    
    @property
    def w_ext(self) -> float:
        return 1.0 / (1.0 + np.exp(-self.logit_w_ext))


@dataclass
class HyperParameters:
    """Population-level hyperparameters (geometry laws)."""
    
    # Coherence length law: log L₀ = μ_ℓ + β_T·log(T/T₀) + β_e·e + β_q·(q-1) + ...
    mu_ell: float = np.log(180.0)    # Population mean log L₀
    beta_T: float = -0.3              # Temperature dependence (hotter → shorter)
    beta_e: float = 0.0               # Ellipticity dependence  
    beta_q: float = -0.2              # LOS shape (more elongated → shorter)
    beta_z: float = 0.0               # Redshift evolution
    sigma_ell: float = 0.3            # Cluster-to-cluster scatter
    
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


class HierarchicalClusterModel:
    """
    Hierarchical Bayesian model for cluster kernel calibration.
    
    This model learns how cluster geometry (morphology, temperature, shape)
    controls the path-integral kernel parameters (coherence length, amplitude,
    path family weights).
    
    Key Features:
    - Partial pooling: clusters share strength via geometry laws
    - Regularization: prevents overfitting to individual clusters
    - Interpretability: coefficients have physical meaning
    - Generalization: predicts new clusters from geometry alone
    """
    
    def __init__(
        self,
        predictors: List[GeometryPredictors],
        observations: Dict[str, np.ndarray],
        use_pymc: bool = True
    ):
        """
        Initialize hierarchical model.
        
        Parameters
        ----------
        predictors : List[GeometryPredictors]
            Geometry predictors for each cluster
        observations : Dict[str, np.ndarray]
            Observations with keys:
            - 'theta_E': Einstein radii [arcsec]
            - 'theta_E_err': Uncertainties [arcsec]
            - 'M_enc' (optional): Enclosed masses [Msun]
            - 'M_enc_err' (optional): Uncertainties [Msun]
        use_pymc : bool
            Use PyMC for full Bayesian inference (vs scipy MLE)
        """
        self.predictors = predictors
        self.observations = observations
        self.n_clusters = len(predictors)
        self.use_pymc = use_pymc and HAS_PYMC
        
        # Storage for inference results
        self.hyperparams_map: Optional[HyperParameters] = None
        self.cluster_params_map: Optional[List[ClusterKernelParams]] = None
        self.trace = None  # PyMC trace if available
        
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
        # Coherence length law
        log_L0 = (
            hyper.mu_ell
            + hyper.beta_T * np.log(pred.T_X / 5.0)  # Normalized to 5 keV
            + hyper.beta_e * pred.ellipticity
            + hyper.beta_q * (pred.q_los - 1.0)
            + hyper.beta_z * np.log(1.0 + pred.z)
        )
        
        # Amplitude law
        log_A_c = (
            hyper.mu_A
            + hyper.gamma_cc * (1.0 if pred.cool_core else 0.0)
            + hyper.gamma_w * np.log(pred.centroid_shift + 0.01)  # Avoid log(0)
            + hyper.gamma_c * np.log(pred.c_500 / 3.0)  # Normalized to c=3
        )
        
        # Exterior weight law
        logit_w_ext = (
            hyper.mu_w
            + hyper.eta_q * (pred.q_los - 1.0)
            + hyper.eta_e * pred.ellipticity
        )
        
        return ClusterKernelParams(
            log_L0=log_L0,
            log_A_c=log_A_c,
            logit_w_ext=logit_w_ext
        )
    
    def log_likelihood(
        self,
        hyper: HyperParameters,
        cluster_params: List[ClusterKernelParams],
        observations: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute log likelihood of observations given parameters.
        
        This would call the actual lensing forward model in practice.
        For now, we use a simplified Gaussian likelihood.
        
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
        # In real implementation, this would:
        # 1. Build baryons for each cluster
        # 2. Apply kernel with cluster_params
        # 3. Compute theta_E predictions
        # 4. Compare to observations
        
        log_like = 0.0
        
        # Einstein radius likelihood
        if 'theta_E' in observations:
            theta_E_obs = observations['theta_E']
            theta_E_err = observations['theta_E_err']
            
            # Mock predictions (replace with actual forward model)
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
        Compute log prior probability.
        
        Includes:
        - Hyperparameter priors (weakly informative)
        - Hierarchical priors on cluster params given hyperparams
        
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
        
        # Hyperparameter priors (weakly informative)
        # Coherence coefficients
        log_p += norm.logpdf(hyper.beta_T, loc=-0.3, scale=0.5)  # Expect hotter → shorter
        log_p += norm.logpdf(hyper.beta_q, loc=-0.2, scale=0.3)  # Elongated → shorter
        log_p += norm.logpdf(hyper.sigma_ell, loc=0.3, scale=0.2)  # Moderate scatter
        
        # Amplitude coefficients  
        log_p += norm.logpdf(hyper.gamma_cc, loc=0.2, scale=0.3)  # Cool cores brighter
        log_p += norm.logpdf(hyper.sigma_A, loc=0.2, scale=0.1)
        
        # Exterior weight coefficients
        log_p += norm.logpdf(hyper.eta_q, loc=2.0, scale=1.0)  # Strong LOS dependence
        log_p += norm.logpdf(hyper.sigma_w, loc=0.5, scale=0.3)
        
        # Hierarchical priors: cluster params given hyperparams
        for i, (pred, params) in enumerate(zip(self.predictors, cluster_params)):
            # Predicted mean from geometry
            pred_mean = self.predict_kernel_params(pred, hyper)
            
            # Likelihood of observed params given predicted mean
            log_p += norm.logpdf(params.log_L0, loc=pred_mean.log_L0, scale=hyper.sigma_ell)
            log_p += norm.logpdf(params.log_A_c, loc=pred_mean.log_A_c, scale=hyper.sigma_A)
            log_p += norm.logpdf(params.logit_w_ext, loc=pred_mean.logit_w_ext, scale=hyper.sigma_w)
        
        return log_p
    
    def fit_map(self, verbose: bool = True) -> Tuple[HyperParameters, List[ClusterKernelParams]]:
        """
        Find maximum a posteriori (MAP) estimate using optimization.
        
        This is the fallback method when PyMC is not available,
        or for quick initial estimates.
        
        Parameters
        ----------
        verbose : bool
            Print progress
        
        Returns
        -------
        hyper_map : HyperParameters
            MAP estimate of hyperparameters
        cluster_params_map : List[ClusterKernelParams]
            MAP estimates of cluster parameters
        """
        if verbose:
            print("Fitting hierarchical model (MAP estimation)...")
        
        # Initialize
        hyper_init = HyperParameters()
        cluster_params_init = [
            self.predict_kernel_params(pred, hyper_init)
            for pred in self.predictors
        ]
        
        # Pack parameters into vector
        def pack_params(hyper, cluster_params):
            return np.array([
                hyper.mu_ell, hyper.beta_T, hyper.beta_e, hyper.beta_q,
                hyper.sigma_ell,
                hyper.mu_A, hyper.gamma_cc, hyper.gamma_w, hyper.sigma_A,
                hyper.mu_w, hyper.eta_q, hyper.eta_e, hyper.sigma_w,
                *[p.log_L0 for p in cluster_params],
                *[p.log_A_c for p in cluster_params],
                *[p.logit_w_ext for p in cluster_params]
            ])
        
        def unpack_params(x):
            n = self.n_clusters
            hyper = HyperParameters(
                mu_ell=x[0], beta_T=x[1], beta_e=x[2], beta_q=x[3],
                sigma_ell=abs(x[4]),  # Force positive
                mu_A=x[5], gamma_cc=x[6], gamma_w=x[7], sigma_A=abs(x[8]),
                mu_w=x[9], eta_q=x[10], eta_e=x[11], sigma_w=abs(x[12])
            )
            cluster_params = [
                ClusterKernelParams(
                    log_L0=x[13 + i],
                    log_A_c=x[13 + n + i],
                    logit_w_ext=x[13 + 2*n + i]
                )
                for i in range(n)
            ]
            return hyper, cluster_params
        
        # Negative log posterior
        def neg_log_posterior(x):
            hyper, cluster_params = unpack_params(x)
            log_like = self.log_likelihood(hyper, cluster_params, self.observations)
            log_prior = self.log_prior(hyper, cluster_params)
            return -(log_like + log_prior)
        
        # Optimize
        x0 = pack_params(hyper_init, cluster_params_init)
        result = minimize(neg_log_posterior, x0, method='L-BFGS-B')
        
        if verbose:
            print(f"Optimization {'converged' if result.success else 'FAILED'}")
            print(f"Final log posterior: {-result.fun:.2f}")
        
        # Unpack result
        hyper_map, cluster_params_map = unpack_params(result.x)
        
        self.hyperparams_map = hyper_map
        self.cluster_params_map = cluster_params_map
        
        return hyper_map, cluster_params_map
    
    def fit_mcmc(
        self,
        n_samples: int = 2000,
        n_tune: int = 1000,
        n_chains: int = 4,
        verbose: bool = True
    ):
        """
        Fit model using MCMC (PyMC).
        
        Provides full posterior distributions over hyperparameters
        and cluster parameters.
        
        Parameters
        ----------
        n_samples : int
            Number of posterior samples per chain
        n_tune : int
            Number of tuning steps
        n_chains : int
            Number of MCMC chains
        verbose : bool
            Print progress
        
        Returns
        -------
        trace : arviz.InferenceData
            Posterior samples
        """
        if not self.use_pymc:
            raise RuntimeError("PyMC not available - use fit_map() instead")
        
        if verbose:
            print("Fitting hierarchical model (MCMC)...")
        
        # Build PyMC model
        with pm.Model() as model:
            # Hyperparameters (population level)
            mu_ell = pm.Normal('mu_ell', mu=np.log(180), sigma=0.5)
            beta_T = pm.Normal('beta_T', mu=-0.3, sigma=0.5)
            beta_q = pm.Normal('beta_q', mu=-0.2, sigma=0.3)
            sigma_ell = pm.HalfNormal('sigma_ell', sigma=0.3)
            
            mu_A = pm.Normal('mu_A', mu=np.log(10), sigma=0.5)
            gamma_cc = pm.Normal('gamma_cc', mu=0.2, sigma=0.3)
            sigma_A = pm.HalfNormal('sigma_A', sigma=0.2)
            
            mu_w = pm.Normal('mu_w', mu=-3.0, sigma=1.0)
            eta_q = pm.Normal('eta_q', mu=2.0, sigma=1.0)
            sigma_w = pm.HalfNormal('sigma_w', sigma=0.5)
            
            # Cluster-level parameters (partial pooling)
            log_L0_raw = pm.Normal('log_L0_raw', mu=0, sigma=1, shape=self.n_clusters)
            log_A_c_raw = pm.Normal('log_A_c_raw', mu=0, sigma=1, shape=self.n_clusters)
            logit_w_ext_raw = pm.Normal('logit_w_ext_raw', mu=0, sigma=1, shape=self.n_clusters)
            
            # Transform via geometry predictors
            # (This is simplified - in practice you'd vectorize predictor computations)
            log_L0 = pm.Deterministic(
                'log_L0',
                mu_ell + beta_T * np.log([p.T_X/5.0 for p in self.predictors])
                + sigma_ell * log_L0_raw
            )
            
            # Likelihood
            theta_E_pred = pm.Deterministic(
                'theta_E_pred',
                30.0 * pm.math.exp(log_A_c_raw * sigma_A)  # Simplified
            )
            
            pm.Normal(
                'theta_E_obs',
                mu=theta_E_pred,
                sigma=self.observations['theta_E_err'],
                observed=self.observations['theta_E']
            )
            
            # Sample
            self.trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=n_chains,
                return_inferencedata=True
            )
        
        if verbose:
            print(az.summary(self.trace))
        
        return self.trace


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Hierarchical Cluster Model")
    print("=" * 70)
    print()
    
    # Create mock data
    np.random.seed(42)
    n_clusters = 5
    
    predictors = [
        GeometryPredictors(
            T_X=np.random.uniform(3, 12),
            q_los=np.random.uniform(0.8, 1.2),
            ellipticity=np.random.uniform(0, 0.3),
            cool_core=(i < 2)
        )
        for i in range(n_clusters)
    ]
    
    observations = {
        'theta_E': np.random.uniform(20, 40, n_clusters),
        'theta_E_err': np.ones(n_clusters) * 2.0
    }
    
    # Test MAP estimation
    model = HierarchicalClusterModel(predictors, observations, use_pymc=False)
    hyper_map, cluster_params_map = model.fit_map(verbose=True)
    
    print("\nMAP Hyperparameters:")
    print(f"  mu_ell = {hyper_map.mu_ell:.3f} (L0 ~ {np.exp(hyper_map.mu_ell):.1f} kpc)")
    print(f"  beta_T = {hyper_map.beta_T:.3f} (temperature dependence)")
    print(f"  mu_A = {hyper_map.mu_A:.3f} (A_c ~ {np.exp(hyper_map.mu_A):.1f})")
    print()
    print("✓ Hierarchical model test complete!")
