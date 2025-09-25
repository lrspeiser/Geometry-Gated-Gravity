#!/usr/bin/env python3
"""
Asymmetric Drift (AD) policy wrapper for consistent MW/SPARC treatment
Implements full Jeans equation for MW and bounded surrogate for SPARC
"""

import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

def deriv_loglog(R, y):
    """
    Compute d(ln y)/d(ln R) using finite differences
    
    Parameters:
    -----------
    R : array
        Radial positions (must be positive)
    y : array
        Values at R (must be positive)
        
    Returns:
    --------
    dlny_dlnR : array
        Logarithmic derivative
    """
    xp = cp if GPU_AVAILABLE and isinstance(R, cp.ndarray) else np
    
    # Ensure positive values
    R_safe = xp.maximum(R, 1e-10)
    y_safe = xp.maximum(y, 1e-10)
    
    # Log values
    log_R = xp.log(R_safe)
    log_y = xp.log(y_safe)
    
    # Gradient using numpy/cupy's built-in
    if len(R) > 1:
        dlny_dlnR = xp.gradient(log_y, log_R)
    else:
        dlny_dlnR = xp.zeros_like(R)
    
    return dlny_dlnR

def jeans_ad(R, v_c, nu, sigma_R, sigma_phi=None, tilt=None):
    """
    Full axisymmetric Jeans equation for asymmetric drift
    
    AD = sigma_R^2 * [d(ln nu)/d(ln R) + d(ln sigma_R^2)/d(ln R) + beta]
    where beta = 1 - sigma_phi^2/sigma_R^2
    
    Parameters:
    -----------
    R : array
        Galactocentric radius (kpc)
    v_c : array
        Circular velocity if known (km/s), used for epicyclic approximation
    nu : array
        Number density of tracers
    sigma_R : array
        Radial velocity dispersion (km/s)
    sigma_phi : array, optional
        Azimuthal velocity dispersion (km/s)
        If None, use epicyclic approximation
    tilt : array, optional
        Tilt term from vertical gradient (usually negligible)
        
    Returns:
    --------
    AD : array
        Asymmetric drift correction (km/s)^2
        Can be negative for steep exponential disks
    """
    xp = cp if GPU_AVAILABLE and isinstance(R, cp.ndarray) else np
    
    # Logarithmic derivatives
    dln_nu_dlnR = deriv_loglog(R, nu)
    dln_sigR2_dlnR = deriv_loglog(R, sigma_R**2)
    
    # Anisotropy parameter
    if sigma_phi is None:
        # Use epicyclic approximation: sigma_phi^2/sigma_R^2 = 0.5*(1 + d(ln v_c)/d(ln R))
        dln_vc_dlnR = deriv_loglog(R, v_c)
        sigma_phi2_over_sigR2 = 0.5 * (1.0 + dln_vc_dlnR)
    else:
        sigma_phi2_over_sigR2 = (sigma_phi / xp.maximum(sigma_R, 1e-6))**2
    
    beta = 1.0 - sigma_phi2_over_sigR2
    
    # Tilt term (usually small)
    if tilt is None:
        tilt_term = xp.zeros_like(R)
    else:
        tilt_term = tilt
    
    # Full AD formula
    AD = sigma_R**2 * (dln_nu_dlnR + dln_sigR2_dlnR + beta) + tilt_term
    
    return AD

def bounded_ad(R, v_phi, sigma_R=None, nu=None, f_max=0.15):
    """
    Bounded surrogate AD for SPARC galaxies without full kinematic data
    
    Uses simplified model with upper bound to prevent unphysical corrections
    
    Parameters:
    -----------
    R : array
        Galactocentric radius (kpc)
    v_phi : array
        Observed rotation velocity (km/s)
    sigma_R : array, optional
        If available, radial dispersion (km/s)
    nu : array, optional
        If available, stellar density profile
    f_max : float
        Maximum fractional AD correction (default 0.15 = 15%)
        
    Returns:
    --------
    AD : array
        Bounded asymmetric drift correction (km/s)^2
    """
    xp = cp if GPU_AVAILABLE and isinstance(R, cp.ndarray) else np
    
    if sigma_R is not None and nu is not None:
        # If we have dispersions, use simplified Jeans
        dln_nu_dlnR = deriv_loglog(R, nu)
        # Assume isothermal: beta ≈ 0.5, d(ln sigma_R^2)/d(ln R) ≈ 0
        AD_jeans = sigma_R**2 * (dln_nu_dlnR + 0.5)
    else:
        # No dispersion data - use empirical estimate
        # Typical AD ~ 10-20 km/s for disk galaxies
        AD_typical = 15.0**2  # (km/s)^2
        # Scale with surface brightness proxy (lower Σ → higher AD/v_c)
        scale_factor = xp.ones_like(R)
        if nu is not None:
            # Low density → higher relative AD
            nu_norm = nu / xp.median(nu)
            scale_factor = 1.0 / xp.sqrt(xp.maximum(nu_norm, 0.1))
        AD_jeans = AD_typical * scale_factor
    
    # Apply upper bound to prevent unphysical corrections
    AD_max = f_max * v_phi**2
    AD = xp.minimum(AD_jeans, AD_max)
    
    # Also apply lower bound (AD shouldn't be too negative)
    AD = xp.maximum(AD, -0.5 * AD_max)
    
    return AD

class ADPolicy:
    """
    Asymmetric drift policy manager for consistent MW/SPARC treatment
    """
    
    def __init__(self, policy='auto', f_max=0.15):
        """
        Parameters:
        -----------
        policy : str
            'jeans' - Full Jeans equation (requires dispersions)
            'bounded' - Bounded surrogate (for SPARC)
            'none' - No AD correction
            'auto' - Choose based on available data
        f_max : float
            Maximum fractional AD for bounded mode
        """
        self.policy = policy
        self.f_max = f_max
        self.last_AD = None
        self.diagnostics = {}
    
    def compute_ad(self, R, v_phi, sigma_R=None, sigma_phi=None, nu=None, v_c=None):
        """
        Compute AD correction based on policy and available data
        
        Returns:
        --------
        AD : array
            Asymmetric drift correction (km/s)^2 to add to v_phi^2
        """
        xp = cp if GPU_AVAILABLE and isinstance(R, cp.ndarray) else np
        
        # Determine policy
        if self.policy == 'none':
            AD = xp.zeros_like(R)
            method = 'none'
            
        elif self.policy == 'jeans' or (self.policy == 'auto' and sigma_R is not None):
            # Use full Jeans equation
            if nu is None:
                # Estimate density from exponential assumption
                R_d = 3.0  # Typical scale length
                nu = xp.exp(-R / R_d)
            
            if v_c is None:
                v_c = v_phi  # Approximate
                
            AD = jeans_ad(R, v_c, nu, sigma_R, sigma_phi)
            method = 'jeans'
            
        elif self.policy == 'bounded' or self.policy == 'auto':
            # Use bounded surrogate
            AD = bounded_ad(R, v_phi, sigma_R, nu, self.f_max)
            method = 'bounded'
            
        else:
            raise ValueError(f"Unknown AD policy: {self.policy}")
        
        # Store diagnostics
        self.last_AD = AD
        self.diagnostics = {
            'method': method,
            'median_AD': float(xp.median(xp.sqrt(xp.maximum(AD, 0)))),
            'max_AD': float(xp.max(xp.sqrt(xp.maximum(AD, 0)))),
            'negative_fraction': float(xp.mean(AD < 0)),
            'f_max_used': self.f_max if method == 'bounded' else None
        }
        
        return AD
    
    def get_circular_velocity(self, R, v_phi, **kwargs):
        """
        Convert observed v_phi to circular velocity v_c
        
        v_c^2 = v_phi^2 + AD
        """
        xp = cp if GPU_AVAILABLE and isinstance(R, cp.ndarray) else np
        
        AD = self.compute_ad(R, v_phi, **kwargs)
        v_c_squared = v_phi**2 + AD
        
        # Handle negative v_c^2 (can occur with negative AD)
        v_c_squared = xp.maximum(v_c_squared, 0.0)
        v_c = xp.sqrt(v_c_squared)
        
        return v_c
    
    def report(self):
        """
        Print diagnostics from last AD computation
        """
        if not self.diagnostics:
            print("No AD computation performed yet")
            return
            
        print(f"AD Policy Report:")
        print(f"  Method: {self.diagnostics['method']}")
        print(f"  Median AD: {self.diagnostics['median_AD']:.1f} km/s")
        print(f"  Max AD: {self.diagnostics['max_AD']:.1f} km/s")
        print(f"  Negative AD fraction: {self.diagnostics['negative_fraction']*100:.1f}%")
        if self.diagnostics['f_max_used'] is not None:
            print(f"  f_max bound: {self.diagnostics['f_max_used']:.2f}")

def test_ad_policies():
    """
    Test AD policies on synthetic data
    """
    print("="*60)
    print("TESTING AD POLICIES")
    print("="*60)
    
    # Create test data
    R = np.linspace(1, 20, 50)
    v_phi = 220.0 - 5.0 * (R - 8.0)  # Declining rotation curve
    sigma_R = 30.0 + 5.0 * (R - 8.0)  # Increasing dispersion
    sigma_phi = 0.7 * sigma_R
    nu = np.exp(-R / 3.0)  # Exponential disk
    
    # Test different policies
    policies = ['none', 'jeans', 'bounded', 'auto']
    
    for policy_name in policies:
        print(f"\nPolicy: {policy_name}")
        policy = ADPolicy(policy=policy_name)
        
        if policy_name == 'none':
            v_c = policy.get_circular_velocity(R, v_phi)
        elif policy_name == 'jeans':
            v_c = policy.get_circular_velocity(R, v_phi, sigma_R=sigma_R, 
                                              sigma_phi=sigma_phi, nu=nu)
        elif policy_name == 'bounded':
            v_c = policy.get_circular_velocity(R, v_phi, nu=nu)
        else:  # auto
            # Test with full data
            v_c = policy.get_circular_velocity(R, v_phi, sigma_R=sigma_R, nu=nu)
        
        policy.report()
        
        # Check results
        median_diff = np.median(v_c - v_phi)
        print(f"  Median v_c - v_phi: {median_diff:.1f} km/s")
        
        if policy_name == 'none':
            assert np.allclose(v_c, v_phi), "None policy should give v_c = v_phi"
            print("  ✅ PASSED: No correction applied")
        elif median_diff > 0:
            print("  ✅ PASSED: AD correction increases v_c")
        else:
            print("  ⚠️ WARNING: Unexpected AD behavior")
    
    print("\n" + "="*60)
    print("All AD policy tests completed")

if __name__ == '__main__':
    test_ad_policies()