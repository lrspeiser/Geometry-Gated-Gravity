#!/usr/bin/env python3
"""
Debug AD correction calculation
"""

import numpy as np
import matplotlib.pyplot as plt

# GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
    print("Using GPU")
except ImportError:
    xp = np
    GPU_AVAILABLE = False
    print("Using CPU")

def compute_ad_correction_debug(R, vphi, sigR, sigphi, nu):
    """
    Debug version of AD correction with detailed output
    """
    print(f"\nInput data shapes:")
    print(f"  R: {R.shape}, range [{xp.min(R):.1f}, {xp.max(R):.1f}] kpc")
    print(f"  vphi: mean={xp.mean(vphi):.1f} km/s")
    print(f"  sigR: mean={xp.mean(sigR):.1f} km/s")
    print(f"  sigphi: mean={xp.mean(sigphi):.1f} km/s")
    print(f"  nu: range [{xp.min(nu):.3f}, {xp.max(nu):.3f}]")
    
    # Create bins
    bins = xp.linspace(4.0, 14.0, 11)  # 10 bins
    print(f"\nBins: {bins}")
    
    AD_values = []
    R_centers = []
    
    for i in range(len(bins) - 1):
        mask = (R >= bins[i]) & (R < bins[i + 1])
        n_stars = int(xp.sum(mask))
        
        if n_stars < 20:
            continue
            
        R_bin = R[mask]
        R_mean = float(xp.mean(R_bin))
        
        sigR2_mean = float(xp.mean(sigR[mask] ** 2))
        sigphi2_mean = float(xp.mean(sigphi[mask] ** 2))
        nu_mean = float(xp.mean(nu[mask]))
        
        print(f"\nBin {i}: R=[{bins[i]:.1f}, {bins[i+1]:.1f}] kpc")
        print(f"  N stars: {n_stars}")
        print(f"  R_mean: {R_mean:.2f} kpc")
        print(f"  sigR: {np.sqrt(sigR2_mean):.1f} km/s")
        print(f"  sigphi: {np.sqrt(sigphi2_mean):.1f} km/s")
        print(f"  nu: {nu_mean:.3f}")
        
        # Compute gradients using finite differences
        dln_nu_dlnR = 0.0
        dln_sigR2_dlnR = 0.0
        
        # Need adjacent bins for gradients
        if i > 0:
            mask_prev = (R >= bins[i-1]) & (R < bins[i])
            if xp.sum(mask_prev) > 20:
                R_prev = float(xp.mean(R[mask_prev]))
                nu_prev = float(xp.mean(nu[mask_prev]))
                sigR2_prev = float(xp.mean(sigR[mask_prev] ** 2))
                
                if i < len(bins) - 2:
                    mask_next = (R >= bins[i+1]) & (R < bins[i+2])
                    if xp.sum(mask_next) > 20:
                        R_next = float(xp.mean(R[mask_next]))
                        nu_next = float(xp.mean(nu[mask_next]))
                        sigR2_next = float(xp.mean(sigR[mask_next] ** 2))
                        
                        # Central differences
                        dln_nu_dlnR = (np.log(nu_next) - np.log(nu_prev)) / (np.log(R_next) - np.log(R_prev))
                        dln_sigR2_dlnR = (np.log(sigR2_next) - np.log(sigR2_prev)) / (np.log(R_next) - np.log(R_prev))
        
        # Anisotropy parameter
        beta = 1.0 - sigphi2_mean / sigR2_mean
        
        print(f"  dln(nu)/dln(R): {dln_nu_dlnR:.3f}")
        print(f"  dln(sigR2)/dln(R): {dln_sigR2_dlnR:.3f}")
        print(f"  beta (anisotropy): {beta:.3f}")
        
        # AD correction
        AD_value = sigR2_mean * (dln_nu_dlnR + dln_sigR2_dlnR + beta)
        print(f"  AD = {np.sqrt(max(AD_value, 0)):.1f} km/s")
        
        AD_values.append(AD_value)
        R_centers.append(R_mean)
    
    return np.array(R_centers), np.array(AD_values)

def main():
    """Test AD correction with synthetic MW data"""
    
    print("="*60)
    print("DEBUGGING ASYMMETRIC DRIFT CORRECTION")
    print("="*60)
    
    # Create realistic MW test data
    np.random.seed(42)
    n_stars = 5000
    
    # Radial distribution
    R = np.random.uniform(4.0, 14.0, n_stars)
    
    # True circular velocity (flat rotation curve)
    v_c_true = 220.0 + 2.0 * (R - 8.0)
    
    # Velocity dispersions (realistic MW values)
    sigma_R = 30.0 + 10.0 * (R / 8.0)  # Increases outward
    sigma_phi = 0.7 * sigma_R  # Typical anisotropy
    
    # Number density (exponential disk)
    R_d = 2.5  # kpc
    nu = np.exp(-R / R_d)
    
    # Asymmetric drift: v_phi = sqrt(v_c^2 - AD)
    # For MW disk, AD ~ sigma_R^2 * (gradient terms + anisotropy)
    # Typical AD is 10-25 km/s, or AD ~ 100-625 (km/s)^2
    
    # Expected AD from Jeans equation
    dln_nu_dlnR_true = -R / R_d / np.log(R)  # d(ln nu)/d(ln R)
    dln_sigR2_dlnR_true = np.log(1 + 10.0/8.0) / np.log(14.0/4.0)  # Approximate
    beta_true = 1.0 - (0.7)**2  # = 0.51
    
    AD_true = sigma_R**2 * (-R/R_d/np.log(R) + 0.2 + beta_true)
    
    # Stellar streaming velocity
    v_phi = np.sqrt(np.maximum(v_c_true**2 - AD_true, 0.0))
    v_phi += np.random.normal(0, 3.0, n_stars)  # Add observational scatter
    
    print(f"\nTrue values at R=8 kpc:")
    idx_8 = np.argmin(np.abs(R - 8.0))
    print(f"  v_c (true): {v_c_true[idx_8]:.1f} km/s")
    print(f"  v_phi (streaming): {v_phi[idx_8]:.1f} km/s")
    print(f"  AD (true): {np.sqrt(AD_true[idx_8]):.1f} km/s")
    print(f"  sigma_R: {sigma_R[idx_8]:.1f} km/s")
    
    # Convert to GPU if available
    if GPU_AVAILABLE:
        R_gpu = cp.asarray(R)
        vphi_gpu = cp.asarray(v_phi)
        sigR_gpu = cp.asarray(sigma_R)
        sigphi_gpu = cp.asarray(sigma_phi)
        nu_gpu = cp.asarray(nu)
    else:
        R_gpu = R
        vphi_gpu = v_phi
        sigR_gpu = sigma_R
        sigphi_gpu = sigma_phi
        nu_gpu = nu
    
    # Compute AD correction
    R_centers, AD_computed = compute_ad_correction_debug(R_gpu, vphi_gpu, sigR_gpu, sigphi_gpu, nu_gpu)
    
    # Plot results
    if GPU_AVAILABLE:
        R_centers = cp.asnumpy(R_centers)
        AD_computed = cp.asnumpy(AD_computed)
        R = cp.asnumpy(R_gpu)
        v_phi = cp.asnumpy(vphi_gpu)
        v_c_true = cp.asnumpy(cp.asarray(v_c_true))
        AD_true = cp.asnumpy(cp.asarray(AD_true))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Velocity curves
    ax = axes[0, 0]
    ax.scatter(R, v_c_true, c='blue', alpha=0.1, s=1, label='v_c (true)')
    ax.scatter(R, v_phi, c='red', alpha=0.1, s=1, label='v_phi (streaming)')
    ax.plot(R_centers, np.sqrt(np.mean(v_phi)**2 + AD_computed), 'go-', label='v_c (AD corrected)')
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Velocity (km/s)')
    ax.legend()
    ax.set_title('Velocity Curves')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: AD correction
    ax = axes[0, 1]
    ax.scatter(R, np.sqrt(np.maximum(AD_true, 0)), c='blue', alpha=0.1, s=1, label='AD (true)')
    ax.plot(R_centers, np.sqrt(np.maximum(AD_computed, 0)), 'ro-', label='AD (computed)', markersize=8)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('AD correction (km/s)')
    ax.legend()
    ax.set_title('Asymmetric Drift')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Number density
    ax = axes[1, 0]
    ax.scatter(R, nu, c='gray', alpha=0.1, s=1)
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Number density')
    ax.set_yscale('log')
    ax.set_title('Stellar Density Profile')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Velocity dispersions
    ax = axes[1, 1]
    ax.scatter(R, sigma_R, c='red', alpha=0.1, s=1, label='sigma_R')
    ax.scatter(R, sigma_phi, c='blue', alpha=0.1, s=1, label='sigma_phi')
    ax.set_xlabel('R (kpc)')
    ax.set_ylabel('Dispersion (km/s)')
    ax.legend()
    ax.set_title('Velocity Dispersions')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('out/mw_ad_tests/ad_debug.png', dpi=150)
    print(f"\nPlot saved to out/mw_ad_tests/ad_debug.png")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if len(AD_computed) > 0:
        mean_AD = np.mean(np.sqrt(np.maximum(AD_computed, 0)))
        print(f"Mean AD correction: {mean_AD:.1f} km/s")
        
        if mean_AD < 5.0:
            print("❌ AD correction too small - gradient calculation may be failing")
        elif mean_AD > 50.0:
            print("❌ AD correction too large - check units or formula")
        else:
            print("✅ AD correction in reasonable range (5-50 km/s)")
    else:
        print("❌ No AD values computed - check binning")

if __name__ == '__main__':
    main()