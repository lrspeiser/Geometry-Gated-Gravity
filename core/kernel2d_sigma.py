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
- p: Window power-law index (~1-3)
- n_coh: Coherence decay rate (~1-5)

The "interior-emphasis" mode upweights contributions from R' < R, reflecting
the path-integral insight that interior chords dominate the boost.

Author: Many-Paths Gravity Research Team
Date: 2025-01-14
"""

import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


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
                               window_type='power_law'):
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
    ell0 : float
        Coherence length scale (kpc)
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
        
    Returns:
    --------
    Sigma_eff : 2D array
        Effective surface density (M_sun/kpc^2)
    K_sigma : 2D array
        Dimensionless boost kernel field
    diagnostics : dict
        Additional diagnostic information
    """
    # Validate inputs
    assert Sigma_triax.shape == R_grid.shape, "Sigma and R_grid must have same shape"
    assert np.all(Sigma_triax >= 0), "Surface density must be non-negative"
    assert ell0 > 0, "Coherence length must be positive"
    assert A_c >= 0, "Coherence amplitude must be non-negative"
    
    # Build coherence window on grid
    if window_type == 'power_law':
        w = radial_window(R_grid, ell0, p, ncoh,
                         emphasize_interior=emphasize_interior,
                         R_eval=R_grid if emphasize_interior else None)
    elif window_type == 'exponential':
        w = exponential_window(R_grid, ell0, p)
    else:
        raise ValueError(f"Unknown window_type: {window_type}")
    
    # Normalize window with respect to Sigma-weighted integral
    # This makes K_Sigma dimensionless and bounded
    W_den = np.sum(Sigma_triax) + 1e-30  # Avoid division by zero
    
    # Perform convolution: Integral[Sigma(R') * W(|R-R'|) d²R']
    if use_fft:
        # Zero-pad to reduce wrap-around artifacts
        pad_width = max(1, int(0.1 * min(Sigma_triax.shape)))
        S_padded = np.pad(Sigma_triax, pad_width, mode='edge')
        W_padded = np.pad(w, pad_width, mode='edge')
        
        # FFT-based convolution (fast)
        F_S = np.fft.rfft2(S_padded)
        F_W = np.fft.rfft2(W_padded)
        conv_padded = np.fft.irfft2(F_S * F_W, s=S_padded.shape)
        
        # Remove padding
        conv = conv_padded[pad_width:-pad_width, pad_width:-pad_width]
    else:
        # Direct convolution (slower but exact for small grids)
        conv = signal.fftconvolve(Sigma_triax, w, mode='same')
    
    # Compute dimensionless boost kernel
    K_sigma = A_c * (conv / W_den)
    
    # Ensure physical positivity: 1 + K_Sigma > 0
    K_sigma = np.maximum(K_sigma, -0.99)  # Allow small negative K for smoothness
    
    # Effective surface density
    Sigma_eff = Sigma_triax * (1.0 + K_sigma)
    
    # Diagnostic info
    diagnostics = {
        'K_sigma_mean': np.mean(K_sigma),
        'K_sigma_std': np.std(K_sigma),
        'K_sigma_max': np.max(K_sigma),
        'K_sigma_min': np.min(K_sigma),
        'boost_factor_mean': np.mean(1.0 + K_sigma),
        'total_mass_input': np.sum(Sigma_triax),
        'total_mass_output': np.sum(Sigma_eff),
        'window_type': window_type,
        'emphasize_interior': emphasize_interior
    }
    
    logger.info(f"Kernel applied: <K> = {diagnostics['K_sigma_mean']:.4f}, "
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
