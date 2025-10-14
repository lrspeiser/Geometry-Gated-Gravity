#!/usr/bin/env python3
"""
Tune Exterior Weighting Parameter
==================================

Systematic sweep of w_exterior to match MACS0416 θ_E = 30 arcsec.

Strategy:
1. Fix interior paths (working perfectly: θ_E=33" at w_int=1.0)
2. Vary w_exterior from 0.0 to 1.0
3. Find optimal value that brings total θ_E to 30"
4. Verify K_Σ remains physically reasonable (1-10 range)

Author: GravityCalculator Parameter Tuning
Date: 2025-01-13
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

# Import cluster models
from core.gnfw_gas_profiles import build_gnfw_gas_profile, integrate_gas_mass
from core.gas_profiles import (
    rho_hernquist,
    rho_icl_exponential,
    apply_clumping_correction
)
from core.cluster_kernel_3d_shell import (
    Shell3DKernelParams,
    lensing_profiles_3d_shell
)
from many_path_model.lensing_utilities import default_cosmology

def physical_to_angular(R_kpc: float, z: float, cosmo) -> float:
    """
    Convert physical radius to angular size.
    
    Parameters
    ----------
    R_kpc : float
        Physical radius in kpc
    z : float
        Redshift
    cosmo : astropy.cosmology
        Cosmology object
    
    Returns
    -------
    theta_arcsec : float
        Angular size in arcsec
    """
    # Angular diameter distance in kpc
    D_A = cosmo.angular_diameter_distance_kpc(z)
    
    # θ = R / D_A (in radians), convert to arcsec
    theta_rad = R_kpc / D_A
    theta_arcsec = theta_rad * (206265.0)  # radians to arcsec
    
    return theta_arcsec


# MACS0416 cluster properties
MACS0416_PROPS = {
    'M_500': 1.15e15,  # Msun
    'R_500': 1200.0,   # kpc
    'z_lens': 0.396,
    'z_src': 2.0,
    'theta_E_obs': 30.0,  # arcsec (observed)
    'f_gas_target': 0.11,
    'T_X': 10.0,  # keV
}

def test_w_exterior(w_ext: float, verbose: bool = False) -> dict:
    """
    Test a specific w_exterior value on MACS0416.
    
    Parameters
    ----------
    w_ext : float
        Exterior weighting parameter
    verbose : bool
        Print diagnostics
    
    Returns
    -------
    results : dict
        theta_E, error_percent, K_Sigma, etc.
    """
    # Cosmology
    cosmo = default_cosmology()
    
    # Build baryon profile
    r_grid = np.logspace(0, 3.5, 500)  # 1 to ~3000 kpc
    
    # gNFW gas profile
    rho_gas, gas_info = build_gnfw_gas_profile(
        r_grid,
        R_500=MACS0416_PROPS['R_500'],
        M_500=MACS0416_PROPS['M_500'],
        z=MACS0416_PROPS['z_lens'],
        fgas_target=MACS0416_PROPS['f_gas_target'],
        verbose=False
    )
    
    # Apply clumping
    R_200 = MACS0416_PROPS['R_500'] * 1.5
    rho_gas = apply_clumping_correction(
        r_grid, rho_gas, C0=0.30, eta=2.0, R_200=R_200
    )
    
    # BCG (Hernquist)
    M_bcg = 2.0e12  # Msun
    a_bcg = 25.0  # kpc
    rho_bcg = rho_hernquist(r_grid, M_bcg, a_bcg)
    
    # ICL (exponential)
    M_icl = 8.0e11  # Msun
    rs_icl = 150.0  # kpc
    rho_icl = rho_icl_exponential(r_grid, M_icl, rs_icl)
    
    # Total baryon density
    rho_3d = rho_gas + rho_bcg + rho_icl
    
    # 3D shell kernel parameters
    params = Shell3DKernelParams(
        A_c=10.0,
        r_gate=5.0,
        n_gate=4,
        ell0=180.0,
        p_density=1.2,
        L1=1200.0,
        q_taper=2.0,
        w_interior=1.0,  # FIXED (working perfectly)
        w_exterior=w_ext,  # VARIED
        coherence_mode='power_law',
        n_coh=1.5
    )
    
    # Compute lensing profiles
    R_eval = np.logspace(1, 3.2, 300)  # 10 to ~1500 kpc
    profiles = lensing_profiles_3d_shell(
        R_eval,
        MACS0416_PROPS['z_lens'],
        MACS0416_PROPS['z_src'],
        r_grid,
        rho_3d,
        params,
        cosmo,
        verbose=False
    )
    
    theta_E = profiles['theta_E_arcsec']
    K_Sigma_at_RE = 0.0
    
    # Get K_Sigma at Einstein radius
    if theta_E > 0:
        R_E_kpc = theta_E / physical_to_angular(1.0, MACS0416_PROPS['z_lens'], cosmo)
        if R_E_kpc > R_eval[0] and R_E_kpc < R_eval[-1]:
            K_Sigma_at_RE = np.interp(R_E_kpc, profiles['R'], profiles['K_Sigma'])
    
    error_percent = ((theta_E - MACS0416_PROPS['theta_E_obs']) / 
                     MACS0416_PROPS['theta_E_obs'] * 100)
    
    if verbose:
        print(f"w_exterior = {w_ext:.3f}:")
        print(f"  theta_E = {theta_E:.2f}\" (error: {error_percent:+.1f}%)")
        print(f"  K_Sigma(R_E) = {K_Sigma_at_RE:.2f}")
    
    return {
        'w_exterior': w_ext,
        'theta_E': theta_E,
        'error_percent': error_percent,
        'K_Sigma_RE': K_Sigma_at_RE,
        'profiles': profiles
    }


def sweep_w_exterior():
    """
    Sweep w_exterior from 0.0 to 1.0 and find optimal value.
    """
    print("=" * 70)
    print("EXTERIOR WEIGHTING PARAMETER SWEEP")
    print("=" * 70)
    print()
    print(f"Target: theta_E = {MACS0416_PROPS['theta_E_obs']:.1f}\" (MACS0416 observed)")
    print(f"Baseline: w_interior = 1.0 (gives theta_E approx 33\" alone)")
    print()
    print("Sweeping w_exterior from 0.0 to 1.0...")
    print()
    
    # Parameter sweep
    w_ext_values = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, 0.1, ..., 1.0
    results = []
    
    print(f"{'w_ext':<8} {'theta_E':>10} {'Error [%]':>12} {'K_Sigma':>10}")
    print("-" * 70)
    
    for w_ext in w_ext_values:
        result = test_w_exterior(w_ext, verbose=False)
        results.append(result)
        
        # Mark if within ±10% of target
        marker = "  ✓" if abs(result['error_percent']) < 10 else ""
        
        print(f"{w_ext:<8.3f} {result['theta_E']:>10.2f} "
              f"{result['error_percent']:>+12.1f} {result['K_Sigma_RE']:>10.2f}{marker}")
    
    print()
    
    # Find optimal value (closest to 30")
    errors = np.array([r['error_percent'] for r in results])
    abs_errors = np.abs(errors)
    idx_best = np.argmin(abs_errors)
    
    best_result = results[idx_best]
    
    print("=" * 70)
    print("OPTIMAL PARAMETERS")
    print("=" * 70)
    print()
    print(f"Best w_exterior = {best_result['w_exterior']:.3f}")
    print(f"  theta_E = {best_result['theta_E']:.2f}\" (error: {best_result['error_percent']:+.1f}%)")
    print(f"  K_Sigma(R_E) = {best_result['K_Sigma_RE']:.2f}")
    print()
    
    # Check if within acceptable range
    if abs(best_result['error_percent']) < 10:
        print("✓ SUCCESS: Within ±10% of observed Einstein radius!")
    elif abs(best_result['error_percent']) < 20:
        print("⚠ CLOSE: Within ±20%, may need finer tuning")
    else:
        print("✗ NEEDS MORE TUNING: Error > 20%")
    print()
    
    # Plot results
    plot_sweep_results(results)
    
    return results, best_result


def plot_sweep_results(results):
    """
    Plot w_exterior sweep results.
    """
    w_values = np.array([r['w_exterior'] for r in results])
    theta_E = np.array([r['theta_E'] for r in results])
    errors = np.array([r['error_percent'] for r in results])
    K_Sigma = np.array([r['K_Sigma_RE'] for r in results])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: θ_E vs w_exterior
    ax = axes[0, 0]
    ax.plot(w_values, theta_E, 'o-', linewidth=2, markersize=6, label='Predicted')
    ax.axhline(MACS0416_PROPS['theta_E_obs'], color='red', linestyle='--', 
               linewidth=2, label=f"Observed ({MACS0416_PROPS['theta_E_obs']}\")")
    ax.axhspan(MACS0416_PROPS['theta_E_obs'] * 0.9, 
               MACS0416_PROPS['theta_E_obs'] * 1.1,
               alpha=0.2, color='green', label='±10% range')
    ax.set_xlabel('w_exterior', fontsize=12)
    ax.set_ylabel('theta_E [arcsec]', fontsize=12)
    ax.set_title('Einstein Radius vs Exterior Weight', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error vs w_exterior
    ax = axes[0, 1]
    ax.plot(w_values, errors, 'o-', linewidth=2, markersize=6, color='C1')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.axhspan(-10, 10, alpha=0.2, color='green', label='±10% range')
    ax.set_xlabel('w_exterior', fontsize=12)
    ax.set_ylabel('Error [%]', fontsize=12)
    ax.set_title('Prediction Error vs Exterior Weight', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: K_Sigma vs w_exterior
    ax = axes[1, 0]
    ax.plot(w_values, K_Sigma, 'o-', linewidth=2, markersize=6, color='C2')
    ax.axhspan(1, 10, alpha=0.2, color='green', label='Physical range (1-10)')
    ax.set_xlabel('w_exterior', fontsize=12)
    ax.set_ylabel('K_Sigma(R_E)', fontsize=12)
    ax.set_title('Boost Factor vs Exterior Weight', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence profile for optimal w_exterior
    # Find best result
    abs_errors = np.abs(errors)
    idx_best = np.argmin(abs_errors)
    best_result = results[idx_best]
    
    ax = axes[1, 1]
    profiles = best_result['profiles']
    cosmo = default_cosmology()
    R_arcsec = profiles['R'] / physical_to_angular(1.0, MACS0416_PROPS['z_lens'], cosmo)
    
    ax.plot(R_arcsec, profiles['mean_kappa'], linewidth=2, 
            label=f"w_ext={best_result['w_exterior']:.3f}")
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='<kappa> = 1')
    ax.axvline(best_result['theta_E'], color='green', linestyle='--', 
               linewidth=2, label=f"theta_E={best_result['theta_E']:.1f}\"")
    ax.set_xlabel('Radius [arcsec]', fontsize=12)
    ax.set_ylabel('<kappa>(R)', fontsize=12)
    ax.set_title('Mean Convergence Profile (Optimal)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join('..', 'figures', 'w_exterior_tuning_sweep.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    print()


def test_optimal_configuration():
    """
    Test the optimal configuration in detail.
    """
    # First, find optimal w_exterior
    results, best = sweep_w_exterior()
    
    print("=" * 70)
    print("DETAILED TEST WITH OPTIMAL PARAMETERS")
    print("=" * 70)
    print()
    
    # Re-run with verbose output
    result = test_w_exterior(best['w_exterior'], verbose=True)
    
    print()
    print("Configuration:")
    print(f"  w_interior = 1.0 (interior chords)")
    print(f"  w_exterior = {best['w_exterior']:.3f} (exterior arcs)")
    print(f"  A_c = 10.0")
    print(f"  ell0 = 180.0 kpc")
    print(f"  p_density = 1.2")
    print()
    
    return result


if __name__ == '__main__':
    print()
    print("=" * 70)
    print("MACS0416: EXTERIOR WEIGHTING PARAMETER TUNING")
    print("=" * 70)
    print()
    
    result = test_optimal_configuration()
    
    print()
    print("=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Update default parameters in Shell3DKernelParams")
    print("  2. Test on A1689 and MACS0717 for universality")
    print("  3. Run ablation studies with tuned parameters")
    print("  4. Prepare figures for publication")
    print()
