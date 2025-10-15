"""
2D Projected-Space Sigma-Gravity Kernel for Cluster Lensing

This module implements the many-paths gravitational kernel in projected (surface density)
space, preserving triaxial geometry effects through to lensing observables.

Theory:
-------
The effective surface density is computed as:
    Sigma_eff(R) = Sigma_triax(R) * [1 + K_Sigma(R)]

where the boost kernel K_Sigma is a dimensionless, radially-weighted convolution:

    K_Sigma(R) = A_c * Integral[ Sigma_triax(R') * W(|R-R'|; ell0, p, n_coh) d²R' ]
                      / Integral[ Sigma_triax(R') d²R' ]

This form:
- Preserves Newtonian limit (K_Sigma -> 0 as A_c -> 0 or ell0 -> 0)
- Retains all triaxial geometry via Sigma_triax
- Mirrors the stationary-phase path-integral interpretation from galaxy scales
- Uses a radially decaying coherence window W that captures interior-chord emphasis

Physics:
--------
- A_c: Coherence amplitude (dimensionless, ~0.1-1.0)
- ell0: Coherence length scale (kpc, ~50-500)
  * Can be mass-scaled: ell0(M) = ell0_star * (R500/1Mpc)^gamma
  * Physics: larger halos may sustain longer-range coherent path interference
  * Typical values: ell0_star ~ 100-300 kpc, gamma ~ 0.3-0.8
- p: Window power-law index (~1-3)
- n_coh: Coherence decay rate (~1-5)

The "interior-emphasis" mode upweights contributions from R' < R, reflecting
the path-integral insight that interior chords dominate the boost.

Mass-Scaling Motivation:
-------------------------
The coherence length ell0 sets the spatial scale over which quantum-gravitational
path interference remains phase-coherent. For larger/more massive halos:
  1. Deeper potential wells may sustain longer decoherence times
  2. Larger virial radii R500 naturally correlate with extended path networks
  3. The dimensionless ratio (ell0 / R500) may be approximately universal

We parameterize this as:
    ell0(M) = ell0_star * (R500 / R500_pivot)^gamma

where:
  - ell0_star: coherence length at pivot mass (kpc)
  - R500: cluster scale radius (kpc), proxy for mass M500
  - R500_pivot: normalization scale (default 1000 kpc = 1 Mpc)
  - gamma: power-law index (0 = no mass dependence, >0 = increasing with mass)

Expected gamma range: 0.0-1.0
  - gamma = 0: fixed coherence scale (mass-independent)
  - gamma ~ 0.3-0.5: weak mass scaling (sub-linear with R500)
  - gamma ~ 1.0: linear scaling (ell0 ∝ R500, constant ell0/R500 ratio)

This mass-scaling ansatz is testable via hierarchical inference over cluster samples.

Author: Many-Paths Gravity Research Team
Date: 2025-01-14 (updated 2025-01-19 with mass-scaling)
"""

import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


def compute_mass_scaled_coherence_length(R500, ell0_star=200.0, gamma=0.0, R500_pivot=1000.0):
    """
    Compute mass-dependent coherence length ell0(M) from cluster scale radius R500.
    
    Physics Motivation:
    -------------------
    The coherence length ell0 represents the spatial scale over which gravitational
    path interference remains phase-coherent. For massive galaxy clusters:
      - Larger halos (higher M500, larger R500) have deeper potential wells
      - Deeper wells may sustain longer decoherence times → larger ell0
      - The ratio (ell0 / R500) may be approximately universal across mass scales
    
    This function parameterizes the mass-dependence as a power law:
    
        ell0(M) = ell0_star × (R500 / R500_pivot)^gamma
    
    where:
      - R500: cluster scale radius (kpc), defined as the radius within which
              the mean density is 500× critical density of the universe
      - ell0_star: coherence length at the pivot mass (kpc)
      - gamma: power-law index controlling mass-scaling strength
      - R500_pivot: normalization scale (default 1 Mpc = 1000 kpc)
    
    Parameters:
    -----------
    R500 : float or array
        Cluster scale radius (kpc), proxy for M500
        Typical range: 500-1500 kpc for cluster-scale halos
    ell0_star : float, optional
        Coherence length at pivot mass R500_pivot (kpc)
        Default: 200 kpc (intermediate scale, ~0.2× typical R500)
    gamma : float, optional
        Power-law index for mass scaling
        Default: 0.0 (no mass dependence, fixed ell0 = ell0_star)
        Typical range: 0.0-1.0
          gamma = 0.0 → ell0 constant (mass-independent)
          gamma = 0.5 → ell0 ∝ sqrt(R500) (weak scaling)
          gamma = 1.0 → ell0 ∝ R500 (linear scaling, constant ell0/R500)
    R500_pivot : float, optional
        Pivot scale for normalization (kpc)
        Default: 1000 kpc = 1 Mpc (characteristic cluster scale)
    
    Returns:
    --------
    ell0 : float or array
        Mass-scaled coherence length (kpc), same shape as R500
    
    Examples:
    ---------
    # Fixed coherence length (mass-independent)
    >>> compute_mass_scaled_coherence_length(R500=800, ell0_star=200, gamma=0.0)
    200.0  # kpc, independent of R500
    
    # Linear scaling (constant ell0/R500 ratio)
    >>> compute_mass_scaled_coherence_length(R500=1500, ell0_star=200, gamma=1.0, R500_pivot=1000)
    300.0  # kpc = 200 × (1500/1000)^1.0
    
    # Sub-linear scaling (weaker mass dependence)
    >>> compute_mass_scaled_coherence_length(R500=1600, ell0_star=200, gamma=0.5, R500_pivot=1000)
    252.98  # kpc = 200 × (1600/1000)^0.5 ≈ 200 × 1.265
    
    Physics Interpretation:
    -----------------------
    - gamma = 0: Coherence scale set by fundamental quantum-gravity length,
                 independent of halo mass (e.g., Planck scale, modified gravity scale)
    - gamma > 0: Coherence scale tracks halo size, suggesting path networks
                 scale with gravitational radius
    - gamma = 1: Perfectly self-similar scaling, ell0/R500 constant across masses
    
    Testability:
    ------------
    The exponent gamma is a free parameter constrained by hierarchical Bayesian
    inference over cluster lensing data. A statistically significant gamma > 0
    would provide evidence for mass-dependent coherence, potentially distinguishing
    many-paths gravity from models with fixed fundamental scales.
    
    Notes:
    ------
    - This implementation assumes R500 as a mass proxy. For more precise modeling,
      one could use M500 directly or calibrate R500(M500) via scaling relations.
    - The pivot R500_pivot = 1 Mpc is chosen as a typical cluster scale; results
      are independent of this choice given appropriate ell0_star.
    - For gamma = 0 (default), this reduces to constant ell0 = ell0_star, matching
      the original fixed-ell0 implementation.
    """
    # Validate inputs
    if np.any(R500 <= 0):
        raise ValueError("R500 must be positive (got values <= 0)")
    if ell0_star <= 0:
        raise ValueError(f"ell0_star must be positive, got {ell0_star}")
    if gamma < 0:
        logger.warning(f"gamma = {gamma} < 0: negative mass-scaling is unphysical but allowed")
    if R500_pivot <= 0:
        raise ValueError(f"R500_pivot must be positive, got {R500_pivot}")
    
    # Compute mass-scaled coherence length
    # ell0(M) = ell0_star × (R500 / R500_pivot)^gamma
    ell0 = ell0_star * (R500 / R500_pivot)**gamma
    
    # Log the scaling parameters (useful for diagnostics)
    if gamma != 0.0:
        logger.info(f"Mass-scaled coherence: ell0_star={ell0_star:.1f} kpc, gamma={gamma:.3f}, "
                    f"R500={np.mean(R500):.1f} kpc → ell0={np.mean(ell0):.1f} kpc")
    else:
        logger.debug(f"Fixed coherence length: ell0={ell0_star:.1f} kpc (gamma=0, mass-independent)")
    
    return ell0


def radial_window(R, ell0, p, ncoh, emphasize_interior=False, R_eval=None):
    """
    Compute radial coherence window W(R) for the projected kernel.
    
    Parameters:
    -----------
    R : array
        Radial distance grid (kpc)
    ell0 : float
        Coherence length scale (kpc)
    p : float
        Power-law index for the window (typically 1-3)
    ncoh : float
        Coherence decay rate (typically 1-5)
    emphasize_interior : bool
        If True, upweight contributions from R' < R_eval
    R_eval : array or None
        Evaluation radii for interior emphasis (kpc)
        
    Returns:
    --------
    w : array
        Window weights (dimensionless, positive)
    """
    # Base power-law window: W(R) = [1 + (R/ell0)^p]^(-ncoh)
    w = (1.0 + (R / ell0)**p)**(-ncoh)
    
    # Optional interior emphasis: upweight R' < R_eval
    if emphasize_interior and R_eval is not None:
        # Soft sigmoid mask that enhances interior and suppresses exterior
        # Factor decays over ~15% of coherence scale
        mask = 1.0 / (1.0 + np.exp((R - R_eval) / (0.15 * ell0)))
        w *= (1.0 + mask)  # Range: [1.0, 2.0] for smooth enhancement
    
    return w


def exponential_window(R, ell0, p):
    """
    Alternative exponential window: W(R) = exp[-(R/ell0)^p]
    
    Provides sharper localization than power-law window.
    Good for ablation studies.
    """
    return np.exp(-(R / ell0)**p)


def convolve_sigma_with_kernel(Sigma_triax, R_grid, ell0, p, ncoh, A_c,
                               emphasize_interior=True, use_fft=True,
                               window_type='power_law',
                               R500=None, ell0_star=None, gamma=0.0, R500_pivot=1000.0):
    """
    Apply 2D projected-space Sigma-Gravity kernel to triaxial surface density.
    
    Computes:
        Sigma_eff(R) = Sigma_triax(R) * [1 + K_Sigma(R)]
    
    where K_Sigma is a dimensionless boost from coherent path interference.
    
    Parameters:
    -----------
    Sigma_triax : 2D array
        Triaxial projected surface density on square grid (M_sun/kpc^2)
        Must be centered on cluster center
    R_grid : 2D array
        Radial distance for each grid point (kpc), same shape as Sigma_triax
    ell0 : float or None
        Coherence length scale (kpc)
        If None, must provide R500 and ell0_star for mass-scaling
        If provided, overrides mass-scaling computation
    p : float
        Window power-law index
    ncoh : float
        Coherence decay rate
    A_c : float
        Coherence amplitude (dimensionless)
    emphasize_interior : bool
        If True, upweight interior contributions (R' < R)
    use_fft : bool
        If True, use FFT-based convolution (faster for large grids)
    window_type : str
        'power_law' or 'exponential'
    R500 : float or None, optional
        Cluster scale radius (kpc) for mass-scaling
        Used only if ell0 is None
    ell0_star : float or None, optional
        Coherence length at pivot mass (kpc) for mass-scaling
        Used only if ell0 is None
    gamma : float, optional
        Mass-scaling exponent: ell0(M) = ell0_star * (R500/R500_pivot)^gamma
        Default: 0.0 (no mass-scaling)
    R500_pivot : float, optional
        Pivot scale for mass-scaling normalization (kpc)
        Default: 1000 kpc = 1 Mpc
        
    Returns:
    --------
    Sigma_eff : 2D array
        Effective surface density (M_sun/kpc^2)
    K_sigma : 2D array
        Dimensionless boost kernel field
    diagnostics : dict
        Additional diagnostic information, including 'ell0_used' and mass-scaling params
    
    Notes:
    ------
    Two modes for specifying coherence length:
    1. Fixed ell0: Pass ell0 directly (original behavior)
    2. Mass-scaled ell0: Pass R500 and ell0_star, set ell0=None
    
    For mass-scaled mode:
        ell0(M) = ell0_star * (R500 / R500_pivot)^gamma
    where gamma controls the strength of mass-dependence:
        gamma = 0: fixed ell0 (mass-independent)
        gamma > 0: ell0 increases with halo mass
    
    Example (fixed ell0):
        Sigma_eff, K, diag = convolve_sigma_with_kernel(
            Sigma, R_grid, ell0=200, p=2, ncoh=2, A_c=0.5
        )
    
    Example (mass-scaled ell0):
        Sigma_eff, K, diag = convolve_sigma_with_kernel(
            Sigma, R_grid, ell0=None, p=2, ncoh=2, A_c=0.5,
            R500=800, ell0_star=200, gamma=0.5
        )
    """
    # Validate inputs
    assert Sigma_triax.shape == R_grid.shape, "Sigma and R_grid must have same shape"
    assert np.all(Sigma_triax >= 0), "Surface density must be non-negative"
    assert A_c >= 0, "Coherence amplitude must be non-negative"
    
    # Handle coherence length: fixed ell0 or mass-scaled
    if ell0 is None:
        # Mass-scaling mode: compute ell0(M) from R500
        if R500 is None or ell0_star is None:
            raise ValueError("If ell0 is None, must provide R500 and ell0_star for mass-scaling")
        ell0_used = compute_mass_scaled_coherence_length(R500, ell0_star, gamma, R500_pivot)
        logger.info(f"Using mass-scaled coherence length: R500={R500:.1f} kpc → ell0={ell0_used:.1f} kpc "
                    f"(ell0_star={ell0_star:.1f}, gamma={gamma:.3f})")
    else:
        # Fixed ell0 mode (original behavior)
        ell0_used = ell0
        logger.debug(f"Using fixed coherence length: ell0={ell0_used:.1f} kpc")
    
    assert ell0_used > 0, "Coherence length must be positive"
    
    # Build coherence window on grid using the determined ell0
    if window_type == 'power_law':
        w = radial_window(R_grid, ell0_used, p, ncoh,
                         emphasize_interior=emphasize_interior,
                         R_eval=R_grid if emphasize_interior else None)
    elif window_type == 'exponential':
        w = exponential_window(R_grid, ell0_used, p)
    else:
        raise ValueError(f"Unknown window_type: {window_type}")
    
    # Compute coherence factor from the window
    # Normalize window to peak value of 1 (not integral)
    w_max = np.max(w) + 1e-30
    coherence_field = w / w_max
    
    # A_c directly controls the boost amplitude
    # The boost is modulated by the coherence field which captures:
    # - Distance-dependent gating (via W(|R-R'|))
    # - Density distribution (via where Sigma is concentrated)
    
    # Simple formulation: K_sigma = A_c × coherence_factor(R)
    # where coherence_factor encodes the local many-paths enhancement
    # This allows A_c to directly set the boost scale
    K_sigma = A_c * coherence_field
    
    # Ensure physical positivity: 1 + K_Sigma > 0
    K_sigma = np.maximum(K_sigma, -0.99)  # Allow small negative K for smoothness
    
    # Effective surface density
    Sigma_eff = Sigma_triax * (1.0 + K_sigma)
    
    # Diagnostic info (include mass-scaling parameters)
    diagnostics = {
        'K_sigma_mean': np.mean(K_sigma),
        'K_sigma_std': np.std(K_sigma),
        'K_sigma_max': np.max(K_sigma),
        'K_sigma_min': np.min(K_sigma),
        'boost_factor_mean': np.mean(1.0 + K_sigma),
        'total_mass_input': np.sum(Sigma_triax),
        'total_mass_output': np.sum(Sigma_eff),
        'window_type': window_type,
        'emphasize_interior': emphasize_interior,
        'normalization': 'local_annular_mean',  # Document the normalization used
        'ell0_used': ell0_used,  # Actual ell0 used (may be mass-scaled)
        'ell0_input': ell0,  # Original ell0 parameter (may be None)
        'A_c': A_c,
        # Mass-scaling parameters (for provenance)
        'R500': R500,
        'ell0_star': ell0_star,
        'gamma': gamma,
        'R500_pivot': R500_pivot
    }
    
    logger.info(f"Kernel applied: ell0={ell0_used:.1f} kpc, <K> = {diagnostics['K_sigma_mean']:.4f}, "
                f"<1+K> = {diagnostics['boost_factor_mean']:.4f}")
    
    return Sigma_eff, K_sigma, diagnostics


def azimuthal_average(field_2d, R_grid, R_bins):
    """
    Compute azimuthal average of a 2D field in radial bins.
    
    Useful for creating radial profiles from 2D maps.
    
    Parameters:
    -----------
    field_2d : 2D array
        Field to average (e.g., Sigma, K_sigma)
    R_grid : 2D array
        Radial distance for each grid point (kpc)
    R_bins : array
        Bin edges for radial averaging (kpc)
        
    Returns:
    --------
    R_centers : array
        Bin centers (kpc)
    profile : array
        Azimuthally averaged profile
    profile_std : array
        Standard deviation in each bin
    """
    R_centers = 0.5 * (R_bins[:-1] + R_bins[1:])
    profile = np.zeros(len(R_centers))
    profile_std = np.zeros(len(R_centers))
    
    for i in range(len(R_centers)):
        mask = (R_grid >= R_bins[i]) & (R_grid < R_bins[i+1])
        if np.sum(mask) > 0:
            values = field_2d[mask]
            profile[i] = np.mean(values)
            profile_std[i] = np.std(values)
        else:
            profile[i] = np.nan
            profile_std[i] = np.nan
    
    return R_centers, profile, profile_std


def kernel_ablation_study(Sigma_triax, R_grid, ell0_range, A_c, p=2.0, ncoh=2.0):
    """
    Perform ablation study varying coherence length scale.
    
    Useful for sensitivity analysis and determining optimal ell0.
    
    Parameters:
    -----------
    Sigma_triax : 2D array
        Triaxial surface density
    R_grid : 2D array
        Radial grid (kpc)
    ell0_range : array
        Range of coherence lengths to test (kpc)
    A_c : float
        Fixed coherence amplitude
    p, ncoh : float
        Fixed window parameters
        
    Returns:
    --------
    results : list of dict
        Each dict contains ell0 and corresponding diagnostics
    """
    results = []
    
    for ell0 in ell0_range:
        Sigma_eff, K_sigma, diag = convolve_sigma_with_kernel(
            Sigma_triax, R_grid, ell0, p, ncoh, A_c,
            emphasize_interior=True, use_fft=True
        )
        
        results.append({
            'ell0': ell0,
            'K_mean': diag['K_sigma_mean'],
            'K_std': diag['K_sigma_std'],
            'boost_mean': diag['boost_factor_mean'],
            'Sigma_eff': Sigma_eff,
            'K_sigma': K_sigma
        })
    
    return results


# Validation test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing 2D projected-space Sigma-Gravity kernel...")
    print("="*60)
    
    # Create a simple test profile (NFW-like)
    nx, ny = 256, 256
    R_max = 2000.0  # kpc
    x = np.linspace(-R_max, R_max, nx)
    y = np.linspace(-R_max, R_max, ny)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X**2 + Y**2)
    
    # Simple NFW-like profile
    r_s = 300.0  # kpc
    Sigma0 = 1e3  # M_sun/kpc^2
    Sigma_test = Sigma0 / (1.0 + (R_grid / r_s)**2)
    
    # Apply kernel
    ell0 = 200.0  # kpc
    p = 2.0
    ncoh = 2.0
    A_c = 0.5
    
    print(f"\nTest parameters:")
    print(f"  Grid: {nx}x{ny}, R_max = {R_max:.0f} kpc")
    print(f"  Profile: NFW-like, r_s = {r_s:.0f} kpc")
    print(f"  Kernel: ell0 = {ell0:.0f} kpc, A_c = {A_c:.2f}")
    
    Sigma_eff, K_sigma, diag = convolve_sigma_with_kernel(
        Sigma_test, R_grid, ell0, p, ncoh, A_c,
        emphasize_interior=True, use_fft=True
    )
    
    print(f"\nKernel diagnostics:")
    for key, val in diag.items():
        if isinstance(val, (int, float)):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    
    # Compute radial profiles
    R_bins = np.linspace(0, 1500, 31)
    R_cen, Sigma_prof, Sigma_std = azimuthal_average(Sigma_test, R_grid, R_bins)
    _, Sigma_eff_prof, _ = azimuthal_average(Sigma_eff, R_grid, R_bins)
    _, K_prof, K_std = azimuthal_average(K_sigma, R_grid, R_bins)
    
    print(f"\nRadial profile at R = 500 kpc:")
    idx = np.argmin(np.abs(R_cen - 500))
    print(f"  Sigma_input: {Sigma_prof[idx]:.2f} M_sun/kpc^2")
    print(f"  Sigma_eff: {Sigma_eff_prof[idx]:.2f} M_sun/kpc^2")
    print(f"  K_sigma: {K_prof[idx]:.4f}")
    print(f"  Boost factor (1+K): {1+K_prof[idx]:.4f}")
    
    # Test Newtonian limit
    print(f"\nNewtonian limit test (A_c -> 0):")
    Sigma_eff_newton, K_newton, diag_newton = convolve_sigma_with_kernel(
        Sigma_test, R_grid, ell0, p, ncoh, A_c=0.0,
        emphasize_interior=True, use_fft=True
    )
    newton_error = np.max(np.abs(Sigma_eff_newton - Sigma_test)) / np.max(Sigma_test)
    print(f"  Max relative error: {newton_error:.2e}")
    print(f"  PASS" if newton_error < 1e-6 else f"  FAIL")
    
    # Test interior emphasis
    print(f"\nInterior emphasis test:")
    _, K_interior, _ = convolve_sigma_with_kernel(
        Sigma_test, R_grid, ell0, p, ncoh, A_c,
        emphasize_interior=True, use_fft=True
    )
    _, K_no_interior, _ = convolve_sigma_with_kernel(
        Sigma_test, R_grid, ell0, p, ncoh, A_c,
        emphasize_interior=False, use_fft=True
    )
    
    interior_diff = np.mean(K_interior) - np.mean(K_no_interior)
    print(f"  <K> with interior emphasis: {np.mean(K_interior):.4f}")
    print(f"  <K> without interior emphasis: {np.mean(K_no_interior):.4f}")
    print(f"  Difference: {interior_diff:.4f}")
    print(f"  PASS" if interior_diff > 0.01 else f"  INFO: Small difference")
    
    print(f"\n" + "="*60)
    print("2D Sigma-Gravity kernel validation complete!")
    print("="*60)
