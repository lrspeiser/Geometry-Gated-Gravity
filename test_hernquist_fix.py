#!/usr/bin/env python3
"""
Test and debug Hernquist projection
"""

import numpy as np
import matplotlib.pyplot as plt

def hernquist_sigma_analytic(R, M, a):
    """
    Correct analytic Hernquist surface density projection.
    Based on Hernquist 1990, eq. 37.
    """
    x = R / a
    x = np.maximum(x, 1e-12)
    
    prefac = M / (2 * np.pi * a**2)
    Sigma = np.zeros_like(x)
    
    # Small x expansion (x << 1)
    small_mask = x < 0.001
    if np.any(small_mask):
        xs = x[small_mask]
        # For small x: Σ ≈ (M/2πa²) * [π/2 - 3x²/4 + O(x⁴)]
        Sigma[small_mask] = prefac * (np.pi/2 - 3*xs**2/4)
    
    # x < 1 branch
    mask_lt1 = (x >= 0.001) & (x < 0.999)
    if np.any(mask_lt1):
        x1 = x[mask_lt1]
        # Use stable formula with arccos
        u = 1.0 / x1
        F = np.arccosh(u) / np.sqrt(u**2 - 1)
        numerator = (2 + x1**2) * F - 3
        denominator = (1 - x1**2)**2
        Sigma[mask_lt1] = prefac * numerator / denominator
    
    # x = 1 exactly
    mask_eq1 = np.abs(x - 1.0) < 0.001
    if np.any(mask_eq1):
        Sigma[mask_eq1] = prefac * 2.0 / 3.0
    
    # x > 1 branch
    mask_gt1 = x > 1.001
    if np.any(mask_gt1):
        x2 = x[mask_gt1]
        u = 1.0 / x2
        # Use stable formula with arccos
        F = np.arccos(u) / np.sqrt(1 - u**2)
        numerator = (2 + x2**2) * F - 3
        denominator = (x2**2 - 1)**2
        Sigma[mask_gt1] = prefac * numerator / denominator
    
    return Sigma

def hernquist_sigma_fixed(R, M, a):
    """
    Fixed Hernquist projection with proper handling of all regimes.
    """
    x = R / a
    x = np.maximum(x, 1e-12)
    
    prefac = M / (2 * np.pi * a**2)
    Sigma = np.zeros_like(x)
    
    # Very small x (x < 0.001): Use series
    small_mask = x < 0.001
    if np.any(small_mask):
        xs = x[small_mask]
        # Proper series: Σ/Σ₀ = π/2 - 3x²/4 + 5x⁴/16 - ...
        Sigma[small_mask] = prefac * (np.pi/2 - 3*xs**2/4 + 5*xs**4/16)
    
    # Near x = 1: Use series around x=1
    near1_mask = np.abs(x - 1.0) < 0.01
    if np.any(near1_mask):
        dx = x[near1_mask] - 1.0
        # Taylor series: Σ/Σ₀ = 2/3 + 4dx/15 - 2dx²/35 + ...
        Sigma[near1_mask] = prefac * (2.0/3.0 + 4*dx/15 - 2*dx**2/35)
    
    # x < 1 (not small, not near 1)
    mask_lt1 = (x >= 0.001) & (x < 0.99) & ~near1_mask
    if np.any(mask_lt1):
        x1 = x[mask_lt1]
        # Use arccosh(1/x) for x < 1
        u = 1.0 / x1
        arccosh_u = np.log(u + np.sqrt(u**2 - 1))
        F = arccosh_u / np.sqrt(u**2 - 1)
        
        numerator = (2 + x1**2) * F - 3
        denominator = (1 - x1**2)**2
        Sigma[mask_lt1] = prefac * numerator / denominator
    
    # x > 1 (not near 1) 
    mask_gt1 = (x > 1.01) & ~near1_mask
    if np.any(mask_gt1):
        x2 = x[mask_gt1]
        # Use arccos(1/x) for x > 1
        u = 1.0 / x2
        F = np.arccos(u) / np.sqrt(1 - u**2)
        
        numerator = (2 + x2**2) * F - 3
        denominator = (x2**2 - 1)**2
        Sigma[mask_gt1] = prefac * numerator / denominator
    
    return Sigma

# Test the implementations
if __name__ == "__main__":
    print("Testing Hernquist projection implementations\n")
    
    # Test parameters
    M = 1e10  # Msun
    a = 1.0   # kpc
    
    # Test across wide range
    R_test = np.logspace(-4, 2, 1000)  # 0.0001 to 100 kpc
    
    # Compute with both methods
    print("Computing projections...")
    Sigma_fixed = hernquist_sigma_fixed(R_test, M, a)
    
    # Check for issues
    has_nan = np.any(np.isnan(Sigma_fixed))
    has_neg = np.any(Sigma_fixed < 0)
    has_inf = np.any(np.isinf(Sigma_fixed))
    
    print(f"Fixed implementation:")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has negative: {has_neg}")
    print(f"  Has inf: {has_inf}")
    
    if not has_nan and not has_neg and not has_inf:
        print("  ✓ All values finite and positive")
        
        # Check mass recovery
        M_recovered = 2 * np.pi * np.trapz(Sigma_fixed * R_test, R_test)
        mass_error = abs(M_recovered - M) / M
        print(f"  Mass recovery: {mass_error*100:.3f}% error")
        
        if mass_error < 0.01:
            print("  ✓ Mass conserved to 1%")
        
        # Check specific values
        print(f"\nSpot checks:")
        print(f"  Σ(0.001a) = {Sigma_fixed[np.argmin(np.abs(R_test - 0.001))]/prefac:.6f} × M/2πa²")
        print(f"  Σ(0.1a)   = {Sigma_fixed[np.argmin(np.abs(R_test - 0.1))]/prefac:.6f} × M/2πa²")
        print(f"  Σ(1.0a)   = {Sigma_fixed[np.argmin(np.abs(R_test - 1.0))]/prefac:.6f} × M/2πa² (should be 2/3)")
        print(f"  Σ(10a)    = {Sigma_fixed[np.argmin(np.abs(R_test - 10.0))]/prefac:.6f} × M/2πa²")
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        plt.subplot(121)
        plt.loglog(R_test/a, Sigma_fixed * (2*np.pi*a**2/M))
        plt.xlabel('R/a')
        plt.ylabel('Σ × (2πa²/M)')
        plt.title('Hernquist Surface Density')
        plt.grid(True, alpha=0.3)
        plt.axvline(1, color='r', linestyle='--', alpha=0.5, label='R=a')
        plt.legend()
        
        plt.subplot(122)
        # Check smoothness
        dSigma = np.gradient(Sigma_fixed, R_test)
        plt.semilogx(R_test[1:]/a, np.abs(np.diff(Sigma_fixed))/Sigma_fixed[1:])
        plt.xlabel('R/a')
        plt.ylabel('|ΔΣ|/Σ')
        plt.title('Relative Jump (smoothness check)')
        plt.grid(True, alpha=0.3)
        plt.axvline(1, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('hernquist_test.png', dpi=150)
        print(f"\nPlot saved as hernquist_test.png")
    
    # Now write the corrected version
    print("\n" + "="*50)
    print("Corrected Hernquist function for g3_universal_fix.py:")
    print("="*50)
    
    prefac = M / (2 * np.pi * a**2)
    
    print("""
def hernquist_sigma_stable(self, R, M, a):
    \"\"\"
    B. Numerically stable Hernquist projection with proper branches.
    \"\"\"
    array_mod = cp if GPU_AVAILABLE and isinstance(R, cp.ndarray) else np
    
    x = R / a
    x = array_mod.maximum(x, 1e-12)
    
    prefac = M / (2 * np.pi * a**2)
    Sigma = array_mod.zeros_like(x)
    
    # Very small x (x < 0.001): Use series
    small_mask = x < 0.001
    if array_mod.any(small_mask):
        xs = x[small_mask]
        # Series: Σ/Σ₀ = π/2 - 3x²/4 + 5x⁴/16
        Sigma[small_mask] = prefac * (np.pi/2 - 3*xs**2/4 + 5*xs**4/16)
    
    # Near x = 1: Use Taylor series
    near1_mask = array_mod.abs(x - 1.0) < 0.01
    if array_mod.any(near1_mask):
        dx = x[near1_mask] - 1.0
        # Taylor: Σ/Σ₀ = 2/3 + 4dx/15 - 2dx²/35
        Sigma[near1_mask] = prefac * (2.0/3.0 + 4*dx/15 - 2*dx**2/35)
    
    # Regular x < 1 branch
    mask_lt1 = (x >= 0.001) & (x < 0.99) & ~near1_mask
    if array_mod.any(mask_lt1):
        x1 = x[mask_lt1]
        u = 1.0 / x1
        # arccosh(u) = log(u + sqrt(u²-1))
        arccosh_u = array_mod.log(u + array_mod.sqrt(u**2 - 1))
        F = arccosh_u / array_mod.sqrt(u**2 - 1)
        
        numerator = (2 + x1**2) * F - 3
        denominator = (1 - x1**2)**2
        Sigma[mask_lt1] = prefac * numerator / denominator
    
    # Regular x > 1 branch
    mask_gt1 = (x > 1.01) & ~near1_mask
    if array_mod.any(mask_gt1):
        x2 = x[mask_gt1]
        u = 1.0 / x2
        F = array_mod.arccos(u) / array_mod.sqrt(1 - u**2)
        
        numerator = (2 + x2**2) * F - 3
        denominator = (x2**2 - 1)**2
        Sigma[mask_gt1] = prefac * numerator / denominator
    
    return Sigma
""")