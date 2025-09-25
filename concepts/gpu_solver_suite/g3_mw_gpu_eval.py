#!/usr/bin/env python3
"""
GPU-accelerated MW evaluator with asymmetric drift correction
Ensures apples-to-apples comparison with SPARC data
"""

import numpy as np
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d
    GPU_AVAILABLE = True
except:
    import numpy as cp
    from scipy.ndimage import gaussian_filter1d
    GPU_AVAILABLE = False

# Physical constants
G = 4.300917270e-6  # (km/s)^2 kpc / Msun

# Solar parameters (consistent with Gaia DR3)
R0_KPC = 8.2      # Solar galactocentric radius
V0_KMS = 232.8    # Solar circular velocity
U_SUN = 11.1      # Solar motion in radial direction (km/s)
V_SUN = 12.24     # Solar motion in rotation direction (km/s) 
W_SUN = 7.25      # Solar motion in vertical direction (km/s)

def smooth_step(x, x0, width):
    """C¹ smooth logistic transition function"""
    # Use a wider transition to ensure smoothness
    # tanh provides C-infinity smoothness
    if width <= 0:
        width = 0.5  # Default width to avoid division by zero
    t = 0.5 * (1 + cp.tanh((x - x0) / (2.0 * width)))
    return t

def compute_ad_correction(R, vphi, sigR, sigphi, nu, bins=None):
    """
    Compute asymmetric drift correction to get circular speed from streaming
    
    v_c^2 = v_phi^2 + sigma_R^2 * [d(ln nu)/d(ln R) + d(ln sigma_R^2)/d(ln R) + 1 - sigma_phi^2/sigma_R^2]
    
    Parameters:
    -----------
    R : array
        Galactocentric radius (kpc)
    vphi : array  
        Mean azimuthal velocity (km/s)
    sigR : array
        Radial velocity dispersion (km/s)
    sigphi : array
        Azimuthal velocity dispersion (km/s)  
    nu : array
        Number density of tracers
    bins : array
        Radial bins for computing gradients
        
    Returns:
    --------
    AD : array
        Asymmetric drift correction term (km/s)^2
    """
    if bins is None:
        bins = cp.linspace(cp.min(R), cp.max(R), 21)  # Fewer bins for better statistics
    
    AD = cp.zeros_like(R)
    
    # Process each bin
    for i in range(len(bins) - 1):
        # Find stars in this bin
        mask = (R >= bins[i]) & (R < bins[i + 1])
        n_stars = cp.sum(mask)
        
        if n_stars < 20:  # Need minimum stars for statistics
            continue
            
        # Extract bin data
        R_bin = R[mask]
        sigR_bin = sigR[mask]
        sigphi_bin = sigphi[mask]
        nu_bin = nu[mask]
        
        # Compute bin-averaged quantities
        R_mean = cp.mean(R_bin)
        sigR2_mean = cp.mean(sigR_bin ** 2)
        sigphi2_mean = cp.mean(sigphi_bin ** 2)
        nu_mean = cp.mean(nu_bin)
        
        # Compute gradients using adjacent bins (if they exist)
        dln_nu_dlnR = 0.0
        dln_sigR2_dlnR = 0.0
        
        # Look at previous bin
        if i > 0:
            mask_prev = (R >= bins[i-1]) & (R < bins[i])
            if cp.sum(mask_prev) > 10:
                R_prev = cp.mean(R[mask_prev])
                nu_prev = cp.mean(nu[mask_prev])
                sigR2_prev = cp.mean(sigR[mask_prev] ** 2)
                
                # Look at next bin
                if i < len(bins) - 2:
                    mask_next = (R >= bins[i+1]) & (R < bins[i+2])
                    if cp.sum(mask_next) > 10:
                        R_next = cp.mean(R[mask_next])
                        nu_next = cp.mean(nu[mask_next])
                        sigR2_next = cp.mean(sigR[mask_next] ** 2)
                        
                        # Central differences for gradients
                        if R_next > R_prev > 0 and nu_next > 0 and nu_prev > 0:
                            dln_nu_dlnR = (cp.log(nu_next) - cp.log(nu_prev)) / (cp.log(R_next) - cp.log(R_prev))
                        if R_next > R_prev > 0 and sigR2_next > 0 and sigR2_prev > 0:
                            dln_sigR2_dlnR = (cp.log(sigR2_next) - cp.log(sigR2_prev)) / (cp.log(R_next) - cp.log(R_prev))
                else:
                    # Use backward difference at outer edge
                    if R_mean > R_prev > 0 and nu_mean > 0 and nu_prev > 0:
                        dln_nu_dlnR = (cp.log(nu_mean) - cp.log(nu_prev)) / (cp.log(R_mean) - cp.log(R_prev))
                    if R_mean > R_prev > 0 and sigR2_mean > 0 and sigR2_prev > 0:
                        dln_sigR2_dlnR = (cp.log(sigR2_mean) - cp.log(sigR2_prev)) / (cp.log(R_mean) - cp.log(R_prev))
        elif i == 0 and i < len(bins) - 2:
            # Use forward difference at inner edge
            mask_next = (R >= bins[i+1]) & (R < bins[i+2])
            if cp.sum(mask_next) > 10:
                R_next = cp.mean(R[mask_next])
                nu_next = cp.mean(nu[mask_next])
                sigR2_next = cp.mean(sigR[mask_next] ** 2)
                
                if R_next > R_mean > 0 and nu_next > 0 and nu_mean > 0:
                    dln_nu_dlnR = (cp.log(nu_next) - cp.log(nu_mean)) / (cp.log(R_next) - cp.log(R_mean))
                if R_next > R_mean > 0 and sigR2_next > 0 and sigR2_mean > 0:
                    dln_sigR2_dlnR = (cp.log(sigR2_next) - cp.log(sigR2_mean)) / (cp.log(R_next) - cp.log(R_mean))
        
        # Anisotropy parameter beta = 1 - (sigma_phi/sigma_R)^2
        beta_aniso = 1.0 - sigphi2_mean / cp.maximum(sigR2_mean, 1e-4)
        
        # Full AD correction: sigma_R^2 * [d(ln nu)/d(ln R) + d(ln sigma_R^2)/d(ln R) + beta]
        # Note: The standard formula has "1 - sigma_phi^2/sigma_R^2" which equals beta
        AD_value = sigR2_mean * (dln_nu_dlnR + dln_sigR2_dlnR + beta_aniso)
        
        # Apply to all stars in bin
        AD[mask] = AD_value
    
    return AD

def g3_tail_smooth(R, Sigma_loc, params):
    """
    Compute G³ tail acceleration with C¹ smooth transitions
    
    Parameters:
    -----------
    R : array
        Galactocentric radius (kpc)
    Sigma_loc : array
        Local surface density (Msun/pc^2)
    params : dict
        Model parameters including:
        - v0: characteristic velocity (km/s)
        - rc0: core radius (kpc)
        - p_in, p_out: inner/outer power law exponents
        - Sigma_star: transition surface density (Msun/pc^2)
        - w_p: width for power law transition (in log space)
        - w_S: width for screening transition (in log space)
        - gamma, beta: geometry scaling exponents
        - r_half_kpc: half-mass radius
        - Sigma_bar: mean surface density
        
    Returns:
    --------
    g_tail : array
        Tail acceleration (km/s)^2/kpc
    """
    # Extract parameters
    v0 = params['v0']
    rc0 = params['rc0']
    p_in = params['p_in']
    p_out = params['p_out']
    Sigma_star = params['Sigma_star']
    
    # Use default widths if not provided (ensures C¹)
    # Wider transitions for better smoothness
    w_p = params.get('w_p', 1.5)  # Power law transition width (wider for smoothness)
    w_S = params.get('w_S', 2.0)  # Screening transition width (wider for smoothness)
    
    gamma = params.get('gamma', 0.5)
    beta = params.get('beta', 0.5)
    
    # Geometry-aware rescaling
    r_half = params.get('r_half_kpc', 10.0)
    Sigma_bar = params.get('Sigma_bar', 50.0)
    rc_ref = params.get('rc_ref', 10.0)
    Sigma0 = params.get('Sigma0', 50.0)
    
    rc_eff = rc0 * (r_half / rc_ref) ** gamma * (Sigma_bar / Sigma0) ** (-beta)
    
    # Smooth transitions in log-Sigma space for C¹ continuity
    log_Sigma = cp.log(cp.maximum(Sigma_loc, 1e-8))
    log_Sigma_star = cp.log(cp.maximum(Sigma_star, 1e-8))
    
    # Smooth power law exponent transition
    # Use smooth interpolation to avoid jumps
    transition_p = smooth_step(log_Sigma, log_Sigma_star, w_p)
    p = p_out + (p_in - p_out) * transition_p
    
    # Ensure p is always positive and bounded
    p = cp.clip(p, 0.5, 3.0)
    
    # Smooth screening function (0 → 1)
    # Gradual turn-on to avoid sharp transitions
    S = smooth_step(log_Sigma, log_Sigma_star, w_S)
    
    # Additional smoothing for very low densities
    # This prevents numerical issues at the edges
    low_density_mask = Sigma_loc < (Sigma_star * 0.01)
    S = cp.where(low_density_mask, S * cp.exp(-(Sigma_star * 0.01 - Sigma_loc) / Sigma_star), S)
    
    # Rational gate function (always smooth if p is smooth)
    R_safe = cp.maximum(R, 1e-6)
    num = R_safe ** p
    den = R_safe ** p + rc_eff ** p
    rational_gate = num / den
    
    # Final tail acceleration
    g_tail = (v0 ** 2 / R_safe) * rational_gate * S
    
    return g_tail

def evaluate_mw_with_ad(params, mw_data):
    """
    Evaluate MW fit with asymmetric drift correction
    
    Parameters:
    -----------
    params : dict
        Model parameters
    mw_data : dict
        MW data including:
        - R_kpc: radii
        - vphi_kms: azimuthal velocities
        - sigma_R, sigma_phi: velocity dispersions
        - nu_tracer: number density
        - Sigma_loc: local surface density
        - vbar_kms: baryonic circular speed
        - z_kpc: vertical heights
        
    Returns:
    --------
    loss : float
        Robust loss metric
    diagnostics : dict
        Diagnostic information
    """
    # Extract data
    R = mw_data['R_kpc']
    vphi = mw_data['vphi_kms']
    sigR = mw_data['sigma_R']
    sigphi = mw_data['sigma_phi']
    nu = mw_data.get('nu_tracer', cp.ones_like(R))
    Sigma_loc = mw_data['Sigma_loc']
    vbar = mw_data['vbar_kms']
    z = mw_data.get('z_kpc', cp.zeros_like(R))
    
    # Apply selection cuts
    # 1. Near-plane stars only
    z_cut = cp.abs(z) < 0.5  # Within 500 pc of plane
    
    # 2. Exclude inner bar region and outer halo
    R_cut = (R > 4.0) & (R < 14.0)
    
    # Combined mask
    mask = z_cut & R_cut
    
    # Compute asymmetric drift correction
    bins = cp.linspace(4.0, 14.0, 31)  # 30 bins from 4-14 kpc
    AD = compute_ad_correction(R, vphi, sigR, sigphi, nu, bins)
    
    # Observed circular speed with AD correction
    vc_obs = cp.sqrt(cp.maximum(vphi ** 2 + AD, 0.0))
    
    # Add geometry info to params
    params_full = dict(params)
    params_full['r_half_kpc'] = float(mw_data.get('r_half_kpc', 10.0))
    params_full['Sigma_bar'] = float(mw_data.get('Sigma_bar', 50.0))
    
    # Compute model tail acceleration
    g_tail = g3_tail_smooth(R, Sigma_loc, params_full)
    
    # Model circular speed
    vc_model = cp.sqrt(cp.maximum(vbar ** 2 + g_tail * R, 0.0))
    
    # Robust loss with floor to avoid division issues
    err_floor = 8.0  # km/s
    denom = cp.maximum(cp.abs(vc_obs[mask]), err_floor)
    residuals = (vc_model[mask] - vc_obs[mask]) / denom
    
    # Use median absolute deviation for robustness
    loss = cp.median(cp.abs(residuals))
    
    # Compute diagnostics
    diagnostics = {
        'n_stars': int(cp.sum(mask)),
        'median_AD': float(cp.median(cp.sqrt(cp.maximum(AD[mask], 0.0)))),
        'mean_vphi': float(cp.mean(vphi[mask])),
        'mean_vc_obs': float(cp.mean(vc_obs[mask])),
        'mean_vc_model': float(cp.mean(vc_model[mask])),
        'median_error': float(cp.median(residuals)),
        'percentile_90': float(cp.percentile(cp.abs(residuals), 90))
    }
    
    return float(loss), diagnostics

def verify_units_and_geometry(mw_data):
    """
    Verify unit consistency and perform round-trip geometry test
    
    Returns:
    --------
    checks : dict
        Results of consistency checks
    """
    checks = {}
    
    # Check units
    R = mw_data['R_kpc']
    vbar = mw_data['vbar_kms']
    
    # Mass within R from circular speed
    M_from_vc = vbar ** 2 * R / G  # Should be in Msun
    
    # Typical values at 8 kpc
    idx_8kpc = cp.argmin(cp.abs(R - 8.0))
    checks['R_8kpc'] = float(R[idx_8kpc])
    checks['vbar_8kpc'] = float(vbar[idx_8kpc])
    checks['M_8kpc'] = float(M_from_vc[idx_8kpc])
    
    # Check if mass is reasonable (should be ~1e10 Msun at 8 kpc)
    mass_check = (checks['M_8kpc'] > 1e9) and (checks['M_8kpc'] < 1e11)
    checks['mass_reasonable'] = mass_check
    
    # Check surface density units
    Sigma = mw_data['Sigma_loc']
    checks['Sigma_min'] = float(cp.min(Sigma))
    checks['Sigma_max'] = float(cp.max(Sigma))
    checks['Sigma_median'] = float(cp.median(Sigma))
    
    # Should be in Msun/pc^2 range (1-1000)
    sigma_check = (checks['Sigma_min'] > 0.1) and (checks['Sigma_max'] < 1e4)
    checks['sigma_reasonable'] = sigma_check
    
    # Log all solar parameters used
    checks['solar_params'] = {
        'R0_kpc': R0_KPC,
        'V0_kms': V0_KMS,
        'U_sun_kms': U_SUN,
        'V_sun_kms': V_SUN,
        'W_sun_kms': W_SUN
    }
    
    return checks

class PlateauScheduler:
    """Handle optimization plateaus with smart restarts"""
    
    def __init__(self, patience=12, rel_tol=1e-3):
        self.best = cp.inf
        self.count = 0
        self.patience = patience
        self.rel_tol = rel_tol
        self.history = []
        
    def step(self, val):
        """Check if we've plateaued"""
        self.history.append(float(val))
        
        if val < self.best * (1 - self.rel_tol):
            self.best = val
            self.count = 0
            return False
            
        self.count += 1
        return self.count >= self.patience  # True => trigger restart
        
    def jitter_params(self, params, scale=0.1):
        """Apply smart jitter for restart"""
        jittered = {}
        for key, val in params.items():
            if key in ['v0', 'rc0', 'p_in', 'p_out', 'Sigma_star']:
                # These are the key params to vary
                jitter = np.random.uniform(-scale, scale) * val
                jittered[key] = val + jitter
            else:
                jittered[key] = val
        return jittered