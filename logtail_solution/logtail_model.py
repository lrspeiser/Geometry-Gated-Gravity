#!/usr/bin/env python3
"""
LogTail/G³ (Geometry-Gated Gravity) Model Implementation
========================================================

This module implements the LogTail acceleration model, a geometric modification
to Newtonian gravity that responds to baryon density without requiring dark matter.

Mathematical Framework:
----------------------
The total acceleration is:
    g_total = g_Newton + g_tail

Where g_tail is the LogTail component:
    g_tail = (v0²/r) × f(r) × S(r) × Σ(r)
    
With:
    - f(r) = r/(r + rc): radial profile function
    - S(r) = 0.5 × (1 + tanh((r - r0)/δ)): smooth gate function
    - Σ(r): surface density coupling (optional)

Parameters:
----------
- v0: Asymptotic velocity scale (km/s)
- rc: Core radius (kpc)
- r0: Gate activation radius (kpc)
- δ: Transition width (kpc)
- γ: Radial profile power (extended model)
- β: Surface density coupling strength (extended model)

Author: LogTail Solution Framework
Date: September 2024
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import json
import logging

# Physical constants
G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)
MSUN_G = 1.98847e33
KPC_CM = 3.085677581491367e21
MP_G = 1.67262192369e-24
KEV_PER_J = 6.241509074e15
MU_E = 1.17  # mean molecular weight per electron

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class LogTailModel:
    """
    LogTail/G³ gravity model for galaxy rotation curves and cluster dynamics.
    """
    
    def __init__(self, parameters: Optional[Dict] = None):
        """
        Initialize the LogTail model.
        
        Parameters:
        -----------
        parameters : dict, optional
            Model parameters. If None, uses default values.
            Expected keys:
            - v0_kms: Asymptotic velocity (km/s)
            - rc_kpc: Core radius (kpc)
            - r0_kpc: Gate activation radius (kpc)
            - delta_kpc: Transition width (kpc)
            - gamma: Radial power (optional, default 1.0)
            - beta: Surface density coupling (optional, default 0.0)
        """
        if parameters is None:
            # Default parameters from SPARC optimization
            self.v0 = 140.0
            self.rc = 22.0
            self.r0 = 3.0
            self.delta = 3.0
            self.gamma = 0.5
            self.beta = 0.1
        else:
            self.v0 = parameters.get('v0_kms', 140.0)
            self.rc = parameters.get('rc_kpc', 22.0)
            self.r0 = parameters.get('r0_kpc', 3.0)
            self.delta = parameters.get('delta_kpc', 3.0)
            self.gamma = parameters.get('gamma', 0.5)
            self.beta = parameters.get('beta', 0.1)
        
        logger.info(f"LogTail model initialized with parameters:")
        logger.info(f"  v0 = {self.v0:.1f} km/s")
        logger.info(f"  rc = {self.rc:.1f} kpc")
        logger.info(f"  r0 = {self.r0:.1f} kpc")
        logger.info(f"  δ = {self.delta:.1f} kpc")
        logger.info(f"  γ = {self.gamma:.2f}")
        logger.info(f"  β = {self.beta:.3f}")
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load model parameters from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if 'theta' in data:
            # MW optimization format
            param_names = data.get('param_names', ['v0', 'rc', 'r0', 'delta', 'gamma', 'beta'])
            theta = data['theta']
            parameters = {}
            for i, name in enumerate(param_names[:6]):  # Use first 6 parameters
                if i < len(theta):
                    if 'v0' in name:
                        parameters['v0_kms'] = theta[i]
                    elif 'rc' in name:
                        parameters['rc_kpc'] = theta[i]
                    elif 'r0' in name:
                        parameters['r0_kpc'] = theta[i]
                    elif 'delta' in name or 'dlt' in name:
                        parameters['delta_kpc'] = theta[i]
                    elif 'gamma' in name:
                        parameters['gamma'] = theta[i]
                    elif 'beta' in name:
                        parameters['beta'] = theta[i]
        else:
            # Direct parameter format
            parameters = data
        
        return cls(parameters)
    
    def smooth_gate(self, r: np.ndarray) -> np.ndarray:
        """
        Smooth transition gate function.
        
        S(r) = 0.5 × (1 + tanh((r - r0)/δ))
        
        This function smoothly transitions from 0 to 1 around r0 with width δ.
        It suppresses the LogTail effect at small radii where Newtonian gravity dominates.
        """
        return 0.5 * (1.0 + np.tanh((r - self.r0) / max(self.delta, 1e-9)))
    
    def radial_profile(self, r: np.ndarray) -> np.ndarray:
        """
        Radial profile function.
        
        f(r) = (r / (r + rc))^γ
        
        This function determines how the LogTail strength varies with radius.
        """
        return np.power(r / (r + self.rc), self.gamma)
    
    def surface_density_factor(self, sigma: Optional[np.ndarray] = None,
                              sigma_ref: float = 100.0) -> Union[float, np.ndarray]:
        """
        Surface density coupling factor.
        
        Σ_factor = (Σ/Σ_ref)^β
        
        This couples the LogTail strength to local baryon density.
        """
        if sigma is None or self.beta == 0:
            return 1.0
        return np.power(sigma / sigma_ref, self.beta)
    
    def logtail_acceleration(self, r: np.ndarray, 
                            sigma: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate LogTail acceleration component.
        
        Parameters:
        -----------
        r : array
            Radii in kpc
        sigma : array, optional
            Surface density in Msun/pc²
        
        Returns:
        --------
        g_tail : array
            LogTail acceleration in km²/s²/kpc
        """
        # Avoid division by zero
        r_safe = np.maximum(r, 1e-12)
        
        # Base amplitude
        v2_tail = self.v0**2
        
        # Apply radial profile
        v2_tail *= self.radial_profile(r_safe)
        
        # Apply smooth gate
        v2_tail *= self.smooth_gate(r_safe)
        
        # Apply surface density coupling if provided
        if sigma is not None:
            v2_tail *= self.surface_density_factor(sigma)
        
        # Convert to acceleration
        return v2_tail / r_safe
    
    def predict_rotation_curve(self, r: np.ndarray, v_bar: np.ndarray,
                             sigma: Optional[np.ndarray] = None) -> Dict:
        """
        Predict rotation curve including LogTail modification.
        
        Parameters:
        -----------
        r : array
            Radii in kpc
        v_bar : array
            Baryonic velocity in km/s (from Newton)
        sigma : array, optional
            Surface density in Msun/pc²
        
        Returns:
        --------
        dict with keys:
            - v_total: Total rotation velocity (km/s)
            - v_bar: Input baryonic velocity (km/s)
            - v_tail: LogTail contribution (km/s)
            - g_bar: Baryonic acceleration (km²/s²/kpc)
            - g_tail: LogTail acceleration (km²/s²/kpc)
        """
        # Baryonic acceleration
        g_bar = (v_bar**2) / np.maximum(r, 1e-12)
        
        # LogTail acceleration
        g_tail = self.logtail_acceleration(r, sigma)
        
        # Total acceleration and velocity
        g_total = g_bar + g_tail
        v_total = np.sqrt(g_total * r)
        
        # LogTail contribution to velocity
        v_tail = np.sqrt(np.maximum(g_tail * r, 0))
        
        return {
            'v_total': v_total,
            'v_bar': v_bar,
            'v_tail': v_tail,
            'g_bar': g_bar,
            'g_tail': g_tail,
            'g_total': g_total
        }
    
    def predict_cluster_temperature(self, r: np.ndarray, rho_gas: np.ndarray,
                                   M_gas: np.ndarray) -> np.ndarray:
        """
        Predict cluster temperature profile from hydrostatic equilibrium.
        
        Parameters:
        -----------
        r : array
            Radii in kpc
        rho_gas : array
            Gas density in Msun/kpc³
        M_gas : array
            Enclosed gas mass in Msun
        
        Returns:
        --------
        kT : array
            Temperature in keV
        """
        # Baryonic acceleration
        g_bar = G * M_gas / r**2
        
        # LogTail acceleration (no surface density for clusters)
        g_tail = self.logtail_acceleration(r)
        
        # Total acceleration
        g_total = g_bar + g_tail
        
        # Temperature from hydrostatic equilibrium
        # kT = -μ m_p g / (d ln ρ / dr)
        dlnrho_dr = np.gradient(np.log(rho_gas + 1e-10), r)
        kT_J = 0.6 * MP_G * g_total * 1e6 / np.abs(dlnrho_dr + 1e-10)
        kT_keV = kT_J * KEV_PER_J
        
        return kT_keV
    
    def compute_chi2(self, r: np.ndarray, v_obs: np.ndarray, v_err: np.ndarray,
                    v_bar: np.ndarray, sigma: Optional[np.ndarray] = None) -> float:
        """
        Compute χ² for rotation curve fit.
        
        Parameters:
        -----------
        r : array
            Radii in kpc
        v_obs : array
            Observed velocities in km/s
        v_err : array
            Velocity errors in km/s
        v_bar : array
            Baryonic velocities in km/s
        sigma : array, optional
            Surface density in Msun/pc²
        
        Returns:
        --------
        chi2 : float
            Reduced χ² value
        """
        # Predict rotation curve
        result = self.predict_rotation_curve(r, v_bar, sigma)
        v_pred = result['v_total']
        
        # Compute χ²
        residuals = (v_pred - v_obs) / v_err
        chi2 = np.sum(residuals**2) / len(residuals)
        
        return chi2
    
    def analyze_performance(self, r: np.ndarray, v_obs: np.ndarray,
                           v_bar: np.ndarray, sigma: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze model performance on a galaxy.
        
        Returns detailed metrics about fit quality.
        """
        # Get predictions
        result = self.predict_rotation_curve(r, v_bar, sigma)
        v_pred = result['v_total']
        
        # Compute various metrics
        residuals = v_pred - v_obs
        percent_error = 100 * np.abs(residuals) / v_obs
        
        # Separate inner and outer regions
        r_half = np.median(r)
        inner_mask = r <= r_half
        outer_mask = r > r_half
        
        metrics = {
            'mean_percent_error': np.mean(percent_error),
            'median_percent_error': np.median(percent_error),
            'max_percent_error': np.max(percent_error),
            'rms_error': np.sqrt(np.mean(residuals**2)),
            'inner_mean_error': np.mean(percent_error[inner_mask]) if np.any(inner_mask) else np.nan,
            'outer_mean_error': np.mean(percent_error[outer_mask]) if np.any(outer_mask) else np.nan,
            'correlation': np.corrcoef(v_obs, v_pred)[0, 1],
            'r2_score': 1 - np.sum(residuals**2) / np.sum((v_obs - np.mean(v_obs))**2)
        }
        
        return metrics


def load_optimized_model(param_file: str = "optimized_parameters.json") -> LogTailModel:
    """
    Convenience function to load model with optimized parameters.
    """
    return LogTailModel.from_json(param_file)


if __name__ == "__main__":
    # Example usage
    print("LogTail/G³ Model Test")
    print("=" * 50)
    
    # Create model with default parameters
    model = LogTailModel()
    
    # Test on synthetic data
    r = np.linspace(0.1, 30, 50)  # kpc
    v_bar = 100 * np.sqrt(r / (r + 5))  # Synthetic baryonic curve
    
    # Predict with LogTail
    result = model.predict_rotation_curve(r, v_bar)
    
    print(f"\nAt r=10 kpc:")
    print(f"  Baryonic velocity: {result['v_bar'][20]:.1f} km/s")
    print(f"  LogTail contribution: {result['v_tail'][20]:.1f} km/s")
    print(f"  Total velocity: {result['v_total'][20]:.1f} km/s")
    
    print("\nModel ready for analysis!")