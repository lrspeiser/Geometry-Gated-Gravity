#!/usr/bin/env python3
"""
Correct baryon primitives with analytic Hernquist projection and proper vertical normalization
"""

import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

def hernquist_sigma(R, M, a):
    """
    Analytic projected surface density for Hernquist profile
    Avoids numerical divergence at small R
    
    Parameters:
    -----------
    R : array
        Projected radius (kpc)
    M : float
        Total mass (Msun)
    a : float
        Scale radius (kpc)
        
    Returns:
    --------
    Sigma : array
        Surface density (Msun/kpc^2)
    """
    xp = cp if GPU_AVAILABLE and isinstance(R, cp.ndarray) else np
    
    # Dimensionless radius
    x = R / a
    x = xp.maximum(x, 1e-10)  # Avoid exact zero
    
    # Prefactor
    prefac = M / (2 * xp.pi * a**2)
    
    # Allocate output
    Sigma = xp.zeros_like(x)
    
    # Small-x series (x < 1e-3) to avoid catastrophic cancellation
    small_mask = x < 1e-3
    if xp.any(small_mask):
        x_small = x[small_mask]
        # Series: Σ ≈ (M/2πa²) * (1/3 + x²/10 + 3x⁴/140 + ...)
        Sigma[small_mask] = prefac * (1.0/3.0 + x_small**2/10.0 + 3*x_small**4/140.0)
    
    # x < 1 branch
    mask_lt1 = (x >= 1e-3) & (x < 1.0 - 1e-10)
    if xp.any(mask_lt1):
        x_lt1 = x[mask_lt1]
        # F(x) = arccos(x) / sqrt(1 - x²)
        # Use stable formula
        sqrt_term = xp.sqrt(1.0 - x_lt1**2)
        F = xp.arccos(x_lt1) / sqrt_term
        numerator = (2.0 + x_lt1**2) * F - 3.0
        denominator = (1.0 - x_lt1**2)**2
        Sigma[mask_lt1] = prefac * numerator / denominator
    
    # x = 1 special case
    mask_eq1 = xp.abs(x - 1.0) < 1e-10
    if xp.any(mask_eq1):
        Sigma[mask_eq1] = prefac * 2.0 / 3.0  # M/(3πa²) = (M/2πa²) * (2/3)
    
    # x > 1 branch
    mask_gt1 = x > 1.0 + 1e-10
    if xp.any(mask_gt1):
        x_gt1 = x[mask_gt1]
        # F(x) = arccosh(x) / sqrt(x² - 1)
        # Use log form for numerical stability: arccosh(x) = log(x + sqrt(x²-1))
        sqrt_term = xp.sqrt(x_gt1**2 - 1.0)
        F = xp.log(x_gt1 + sqrt_term) / sqrt_term
        numerator = (2.0 + x_gt1**2) * F - 3.0
        denominator = (x_gt1**2 - 1.0)**2
        Sigma[mask_gt1] = prefac * numerator / denominator
    
    return Sigma

def vert_profile(kind='exponential', h_z=0.3, use_gpu=None):
    """
    Factory for vertical density profiles with correct normalization
    
    Parameters:
    -----------
    kind : str
        'exponential', 'sech2', or 'gaussian'
    h_z : float
        Vertical scale height (kpc)
    use_gpu : bool
        Force GPU/CPU usage (None = auto-detect)
        
    Returns:
    --------
    kernel : function(z)
        Normalized vertical kernel such that ∫ kernel(z) dz = 1
    norm : float
        Normalization constant (1/integral of unnormalized kernel)
    """
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    
    if kind == 'exponential':
        # ρ(z) = (1/2h_z) * exp(-|z|/h_z)
        # ∫_{-∞}^∞ exp(-|z|/h_z) dz = 2h_z
        norm = 1.0 / (2.0 * h_z)
        def kernel(z):
            xp = cp if use_gpu and isinstance(z, cp.ndarray) else np
            return norm * xp.exp(-xp.abs(z) / h_z)
            
    elif kind == 'sech2':
        # ρ(z) = (1/2h_z) * sech²(z/h_z)
        # ∫_{-∞}^∞ sech²(z/h_z) dz = 2h_z
        norm = 1.0 / (2.0 * h_z)
        def kernel(z):
            xp = cp if use_gpu and isinstance(z, cp.ndarray) else np
            return norm / xp.cosh(z / h_z)**2
            
    elif kind == 'gaussian':
        # ρ(z) = (1/√(2π)h_z) * exp(-z²/2h_z²)
        # ∫_{-∞}^∞ exp(-z²/2h_z²) dz = √(2π) h_z
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * h_z)
        def kernel(z):
            xp = cp if use_gpu and isinstance(z, cp.ndarray) else np
            return norm * xp.exp(-z**2 / (2.0 * h_z**2))
    else:
        raise ValueError(f"Unknown vertical profile: {kind}")
    
    return kernel, norm

def build_3d_density(R_grid, z_grid, Sigma_R, vert_kind='exponential', h_z=0.3):
    """
    Build 3D density from surface density with proper vertical normalization
    
    ρ(R,z) = Σ(R) * kernel(z) where ∫ kernel(z) dz = 1
    
    Parameters:
    -----------
    R_grid : array (NR,)
        Radial grid points (kpc)
    z_grid : array (Nz,)
        Vertical grid points (kpc)
    Sigma_R : array (NR,) 
        Surface density profile (Msun/kpc^2)
    vert_kind : str
        Vertical profile type
    h_z : float or array
        Scale height (kpc), can be R-dependent
        
    Returns:
    --------
    rho : array (NR, Nz)
        3D density (Msun/kpc^3)
    """
    xp = cp if GPU_AVAILABLE and isinstance(R_grid, cp.ndarray) else np
    
    # Get vertical kernel
    use_gpu = GPU_AVAILABLE and isinstance(R_grid, cp.ndarray)
    kernel, norm = vert_profile(vert_kind, h_z, use_gpu=use_gpu)
    
    # Build 3D density
    NR = len(R_grid)
    Nz = len(z_grid)
    rho = xp.zeros((NR, Nz))
    
    for i in range(NR):
        # Each R slice has its surface density distributed vertically
        rho[i, :] = Sigma_R[i] * kernel(z_grid)
    
    return rho

def verify_vertical_normalization(vert_kind='exponential', h_z=0.3):
    """
    Test that vertical integration recovers surface density exactly
    """
    xp = np  # Use CPU for testing
    
    # Test grid
    z_grid = xp.linspace(-5*h_z, 5*h_z, 1001)
    
    # Get kernel
    kernel, norm = vert_profile(vert_kind, h_z, use_gpu=False)
    
    # Integrate
    integral = xp.trapz(kernel(z_grid), z_grid)
    
    error = abs(integral - 1.0)
    
    print(f"Vertical profile: {vert_kind}")
    print(f"  Scale height: {h_z} kpc")
    print(f"  Normalization: {norm}")
    print(f"  Integral: {integral:.6f}")
    print(f"  Error: {error:.2e}")
    
    if error < 1e-4:
        print("  ✅ PASSED: Normalization correct")
        return True
    else:
        print("  ❌ FAILED: Normalization error > 1e-4")
        return False

def test_hernquist_projection():
    """
    Test Hernquist analytic projection against high-precision numerical integration
    """
    import scipy.integrate as integrate
    
    M = 1e10  # Msun
    a = 0.5   # kpc
    
    # Test radii
    R_test = np.logspace(-4, 2, 50) * a  # 1e-4*a to 100*a
    
    # Analytic projection
    Sigma_analytic = hernquist_sigma(R_test, M, a)
    
    # High-precision numerical projection for comparison
    def hernquist_3d_density(r):
        """3D Hernquist density"""
        return M / (2 * np.pi) * a / (r * (r + a)**3)
    
    def los_integrand(s, R):
        """Line-of-sight integrand"""
        r = np.sqrt(R**2 + s**2)
        return hernquist_3d_density(r)
    
    Sigma_numerical = np.zeros_like(R_test)
    for i, R in enumerate(R_test):
        # Integrate from -∞ to +∞ along line of sight
        # In practice, truncate at large s where density is negligible
        s_max = 100 * a
        result, _ = integrate.quad(los_integrand, -s_max, s_max, args=(R,))
        Sigma_numerical[i] = result
    
    # Compare
    rel_error = np.abs((Sigma_analytic - Sigma_numerical) / Sigma_numerical)
    max_error = np.max(rel_error)
    
    print(f"\nHernquist projection test:")
    print(f"  Mass: {M:.2e} Msun")
    print(f"  Scale radius: {a} kpc")
    print(f"  Max relative error: {max_error:.2e}")
    
    # Check at specific radii
    for R_check in [1e-3*a, 0.1*a, 0.5*a, a, 2*a, 10*a]:
        idx = np.argmin(np.abs(R_test - R_check))
        print(f"  R = {R_check/a:.3f}*a: "
              f"analytic = {Sigma_analytic[idx]:.2e}, "
              f"numerical = {Sigma_numerical[idx]:.2e}, "
              f"error = {rel_error[idx]*100:.1f}%")
    
    if max_error < 0.02:
        print("  ✅ PASSED: Analytic projection accurate")
        return True
    else:
        print("  ❌ FAILED: Error > 2%")
        return False

def test_mass_conservation():
    """
    Test that volume → surface → mass round-trip is exact
    """
    xp = np
    
    # Test parameters
    M_total = 6e10  # Msun
    R_d = 3.0      # kpc
    h_z = 0.3      # kpc
    
    # Grids
    R_grid = xp.logspace(-1, 2, 100)  # 0.1 to 100 kpc
    z_grid = xp.linspace(-3*h_z, 3*h_z, 101)  # -0.9 to +0.9 kpc
    
    # Exponential disk surface density
    Sigma_0 = M_total / (2 * xp.pi * R_d**2)
    Sigma_R = Sigma_0 * xp.exp(-R_grid / R_d)
    
    # Build 3D density with proper normalization
    rho = build_3d_density(R_grid, z_grid, Sigma_R, 'exponential', h_z)
    
    # Integrate back to surface density
    Sigma_recovered = xp.trapz(rho, z_grid, axis=1)
    
    # Compare
    rel_error = xp.abs((Sigma_recovered - Sigma_R) / Sigma_R)
    max_error = xp.max(rel_error[R_grid < 30])
    
    print(f"\nMass conservation test:")
    print(f"  Vertical profile: exponential")
    print(f"  Scale height: {h_z} kpc")
    print(f"  Max error (R < 30 kpc): {max_error*100:.2f}%")
    
    if max_error < 0.005:
        print("  ✅ PASSED: Volume ↔ Surface conversion exact")
        return True
    else:
        print("  ❌ FAILED: Conversion error > 0.5%")
        return False

def main():
    """
    Run all baryon primitive tests
    """
    print("="*60)
    print("BARYON PRIMITIVES TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test vertical profiles
    for kind in ['exponential', 'sech2', 'gaussian']:
        results.append((f"Vertical {kind}", verify_vertical_normalization(kind, 0.3)))
    
    # Test Hernquist projection
    results.append(("Hernquist projection", test_hernquist_projection()))
    
    # Test mass conservation
    results.append(("Mass conservation", test_mass_conservation()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:25s}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 All baryon primitives correct!")
    else:
        print("\n⚠️ Some tests failed - check implementations")
    
    return all_passed

if __name__ == '__main__':
    main()