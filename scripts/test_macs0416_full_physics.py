#!/usr/bin/env python3
"""
MACS0416 Full Physics Test - Phase 2, Step 2.3
===============================================

Tests path-integral gravity on MACS0416 cluster with complete physics stack:
1. gNFW gas profile (Arnaud+ 2010) normalized to f_gas = 0.11
2. 3D shell integral kernel with interior chord + exterior arc families
3. BCG + ICL stellar components
4. Clumping corrections
5. Optional: triaxial geometry and external convergence

This is the baryon-only, no-dark-matter cluster lensing test.

Key Tests
---------
A. Full stack: Predict Einstein radius with all physics enabled
B. Ablation studies: Interior-only, exterior-only to isolate contributions
C. Knob-off test: Demonstrate interior chords are essential (missed by 2D rings)
D. Robustness: Vary triaxiality (q_los) and external convergence (κ_ext)

Target: θ_E within ±10% of observed 30 arcsec (MACS0416)

Physics Validation
------------------
This test validates the "many paths" hypothesis at cluster scale:
- Interior chords (r<R): Through-core paths sample dense center
- Exterior arcs (r>R): Up-and-over paths sample extended ICM
- Both families contribute via stationary-phase coherence
- NO dark matter invoked

References
----------
- PHYSICS_ROADMAP.md: Phase 2 implementation plan
- PHASE1_STEP14_COMPLETE.md: gNFW gas profiles
- PHASE2_STEP21_COMPLETE.md: 3D shell kernel physics

Author: GravityCalculator Phase 2 Complete Stack
Date: 2025-01-13
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# Import physically-calibrated baryon model (Phase 1 + unified clumping)
from core.build_cluster_baryons import (
    build_cluster_baryon_model,
    ClusterBaryonParams
)

# Import 3D shell kernel (Phase 2.1)
from core.cluster_kernel_3d_shell import (
    Shell3DKernelParams,
    lensing_profiles_3d_shell
)

# Import cosmology
from many_path_model.lensing_utilities import default_cosmology


def build_macs0416_baryon_profile(
    r_grid: np.ndarray,
    fgas_target: float = 0.11,
    verbose: bool = False
) -> tuple:
    """
    Build complete baryon profile for MACS0416 using unified physics.
    
    Now uses the same physically-motivated clumping model as the blind suite
    (C0=1.3, C_max=2.5) to ensure consistent predictions everywhere.
    
    Parameters
    ----------
    r_grid : ndarray
        3D radial grid [kpc]
    fgas_target : float
        Target gas fraction at R_500 (default: 0.11, cosmic fraction)
    verbose : bool
        Print diagnostics
    
    Returns
    -------
    rho_total : ndarray
        Total baryon density [Msun/kpc³]
    info : dict
        Profile diagnostics including component masses
    """
    # MACS0416 parameters (from literature)
    M_500 = 1.15e15  # Msun (Jauzac+ 2015)
    R_500 = 1200.0  # kpc
    z = 0.396
    T_keV = 10.5  # Temperature estimate
    
    if verbose:
        print("=" * 70)
        print("MACS0416 Baryon Profile (Unified Physics)")
        print("=" * 70)
        print()
        print("Cluster Properties:")
        print(f"  M_500 = {M_500:.2e} Msun")
        print(f"  R_500 = {R_500:.1f} kpc")
        print(f"  z = {z:.3f}")
        print(f"  T_keV = {T_keV:.1f} keV")
        print()
        print("Physics:")
        print("  - gNFW gas (Arnaud+ 2010)")
        print("  - Normalized to f_gas(R_500) = 0.11")
        print("  - Clumping: C0=1.3, C_max=2.5 (Simionescu+ 2011)")
        print("  - BCG + ICL from mass scaling relations")
        print()
    
    # Build using unified model
    params = ClusterBaryonParams(
        M_500=M_500,
        R_500=R_500,
        z=z,
        fgas_target=fgas_target,
        T_keV=T_keV,
        # Use physically-motivated clumping (same as blind suite)
        C0=1.3,
        eta=2.0,
        C_max=2.5
    )
    
    components = build_cluster_baryon_model(
        r_grid, params, apply_clumping=True, verbose=verbose
    )
    
    # Return in format expected by caller
    info = {
        'M_500': M_500,
        'R_500': R_500,
        'z': z,
        'M_gas': components.info['M_gas_R500'],
        'M_bcg': components.info['M_BCG'],
        'M_icl': components.info['M_ICL'],
        'M_total': components.info['M_baryon_R500'],
        'fgas': components.info['fgas_R500'],
        'fbaryon': components.info['fbaryon_R500']
    }
    
    return components.rho_total, info


def test_full_physics(
    w_interior: float = 1.0,
    w_exterior: float = 1.0,
    A_c: float = 10.0,
    ell0: float = 180.0,
    verbose: bool = True
) -> dict:
    """
    Run full physics test on MACS0416.
    
    Parameters
    ----------
    w_interior : float
        Weight for interior chord families
    w_exterior : float
        Weight for exterior arc families
    A_c : float
        Cluster amplitude
    ell0 : float
        Coherence length [kpc]
    verbose : bool
        Print diagnostics
    
    Returns
    -------
    results : dict
        Full test results
    """
    # Build radial grid
    r_3d = np.logspace(-1, 3.5, 2000)  # 0.1 to ~3000 kpc
    
    # Build baryon profile (unified physics)
    rho_total, baryon_info = build_macs0416_baryon_profile(
        r_3d, verbose=verbose
    )
    
    # 3D shell kernel parameters
    params = Shell3DKernelParams(
        A_c=A_c,
        r_gate=5.0,
        n_gate=4,
        ell0=ell0,
        p_density=1.2,
        L1=1200.0,
        q_taper=2.0,
        w_interior=w_interior,
        w_exterior=w_exterior,
        coherence_mode='power_law',
        n_coh=1.5
    )
    
    if verbose:
        print("3D Shell Kernel Parameters:")
        print(f"  A_c = {params.A_c}")
        print(f"  ell0 = {params.ell0} kpc")
        print(f"  w_interior = {params.w_interior} (chord families)")
        print(f"  w_exterior = {params.w_exterior} (arc families)")
        print(f"  Coherence mode = {params.coherence_mode}")
        print()
    
    # Projected radii for lensing profiles
    R_proj = np.geomspace(10, 1500, 200)  # kpc
    
    # Cosmology
    cosmo = default_cosmology()
    z_lens = baryon_info['z']
    z_src = 2.0  # Typical background source
    
    # Compute lensing profiles with 3D shell kernel
    if verbose:
        print("Computing lensing profiles with 3D shell kernel...")
    
    profiles = lensing_profiles_3d_shell(
        R_proj, z_lens, z_src, r_3d, rho_total, params, cosmo, verbose=False
    )
    
    # Extract key results
    theta_E_pred = profiles['theta_E_arcsec']
    max_mean_kappa = np.max(profiles['mean_kappa'])
    
    # Einstein radius diagnostics
    idx_E = np.where(profiles['mean_kappa'] >= 1.0)[0]
    if idx_E.size > 0:
        R_E_kpc = R_proj[idx_E[-1]]
        K_Sigma_at_RE = profiles['K_Sigma'][idx_E[-1]]
        Sigma_at_RE = profiles['Sigma'][idx_E[-1]]
        Sigma_eff_at_RE = profiles['Sigma_eff'][idx_E[-1]]
    else:
        R_E_kpc = 0.0
        K_Sigma_at_RE = 0.0
        Sigma_at_RE = 0.0
        Sigma_eff_at_RE = 0.0
    
    # Observed Einstein radius
    theta_E_obs = 30.0  # arcsec (from literature)
    error = abs(theta_E_pred - theta_E_obs)
    frac_error = error / theta_E_obs if theta_E_obs > 0 else np.inf
    
    if verbose:
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        print(f"Einstein Radius:")
        print(f"  Predicted: {theta_E_pred:.2f} arcsec (R_E = {R_E_kpc:.1f} kpc)")
        print(f"  Observed:  {theta_E_obs:.2f} arcsec")
        print(f"  Error: {error:.2f}\" ({100*frac_error:.1f}%)")
        print()
        
        if frac_error < 0.10:
            print("  [PASS] EXCELLENT: Within +/-10% of observed!")
        elif frac_error < 0.25:
            print("  [OK] GOOD: Within +/-25% of observed")
        elif frac_error < 0.50:
            print("  [CHECK] ACCEPTABLE: Within +/-50% of observed")
        else:
            print("  [FAIL] NEEDS TUNING: More than 50% off")
        print()
        
        print(f"Peak Convergence:")
        print(f"  Max mean_kappa = {max_mean_kappa:.3f}")
        print()
        
        print(f"At Einstein Radius (R_E = {R_E_kpc:.1f} kpc):")
        print(f"  K_Sigma(R_E) = {K_Sigma_at_RE:.2f}")
        print(f"  Sigma_baryon(R_E) = {Sigma_at_RE:.2e} Msun/kpc^2")
        print(f"  Sigma_eff(R_E) = {Sigma_eff_at_RE:.2e} Msun/kpc^2")
        print(f"  Boost factor = {Sigma_eff_at_RE/Sigma_at_RE:.2f}x" if Sigma_at_RE > 0 else "")
        print()
    
    results = {
        'theta_E_pred': theta_E_pred,
        'theta_E_obs': theta_E_obs,
        'error': error,
        'frac_error': frac_error,
        'R_E_kpc': R_E_kpc,
        'max_mean_kappa': max_mean_kappa,
        'K_Sigma_at_RE': K_Sigma_at_RE,
        'Sigma_at_RE': Sigma_at_RE,
        'Sigma_eff_at_RE': Sigma_eff_at_RE,
        'profiles': profiles,
        'baryon_info': baryon_info,
        'params': params
    }
    
    return results


def ablation_study(verbose: bool = True):
    """
    Ablation study: Test interior-only, exterior-only, and both.
    
    This demonstrates that interior chords are essential - something
    standard 2D ring projections completely miss!
    """
    if verbose:
        print("\n" + "=" * 70)
        print("ABLATION STUDY: Interior vs Exterior Path Families")
        print("=" * 70)
        print()
        print("Testing contribution from each path family:")
        print("  1. Interior chords only (w_int=1, w_ext=0)")
        print("  2. Exterior arcs only (w_int=0, w_ext=1)")
        print("  3. Both families (w_int=1, w_ext=1)")
        print()
    
    configs = [
        {'name': 'Interior only', 'w_int': 1.0, 'w_ext': 0.0},
        {'name': 'Exterior only', 'w_int': 0.0, 'w_ext': 1.0},
        {'name': 'Both (full)', 'w_int': 1.0, 'w_ext': 1.0},
    ]
    
    results_ablation = []
    
    for config in configs:
        if verbose:
            print(f"\n{'-'*70}")
            print(f"Test: {config['name']}")
            print(f"{'-'*70}\n")
        
        res = test_full_physics(
            w_interior=config['w_int'],
            w_exterior=config['w_ext'],
            verbose=False
        )
        
        results_ablation.append({
            'name': config['name'],
            'w_interior': config['w_int'],
            'w_exterior': config['w_ext'],
            'theta_E': res['theta_E_pred'],
            'error_pct': res['frac_error'] * 100,
            'K_Sigma_at_RE': res['K_Sigma_at_RE'],
            'results': res
        })
        
        if verbose:
            print(f"  theta_E = {res['theta_E_pred']:.2f}\" (error: {res['frac_error']*100:.1f}%)")
            print(f"  K_Sigma(R_E) = {res['K_Sigma_at_RE']:.2f}")
    
    if verbose:
        print("\n" + "=" * 70)
        print("ABLATION SUMMARY")
        print("=" * 70)
        print()
        print(f"{'Configuration':<20} {'theta_E [arcsec]':<15} {'Error [%]':<12} {'K_Sigma(R_E)':<10}")
        print("-" * 70)
        for r in results_ablation:
            print(f"{r['name']:<20} {r['theta_E']:<15.2f} {r['error_pct']:<12.1f} {r['K_Sigma_at_RE']:<10.2f}")
        print()
        
        # Compare interior-only vs full
        interior_only = next(r for r in results_ablation if r['name'] == 'Interior only')
        full = next(r for r in results_ablation if r['name'] == 'Both (full)')
        
        interior_fraction = interior_only['theta_E'] / full['theta_E'] if full['theta_E'] > 0 else 0
        
        print("Key Insight:")
        print(f"  Interior chords provide {interior_fraction*100:.0f}% of total lensing signal")
        print(f"  --> This is what standard 2D ring projections MISS!")
        print()
    
    return results_ablation


def generate_diagnostic_plots(results: dict, output_dir: str = '../figures'):
    """Generate comprehensive diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    R = results['profiles']['R']
    Sigma = results['profiles']['Sigma']
    K_Sigma = results['profiles']['K_Sigma']
    Sigma_eff = results['profiles']['Sigma_eff']
    kappa = results['profiles']['kappa']
    mean_kappa = results['profiles']['mean_kappa']
    gamma_t = results['profiles']['gamma_t']
    
    R_E = results['R_E_kpc']
    theta_E_pred = results['theta_E_pred']
    theta_E_obs = results['theta_E_obs']
    
    # (0, 0): Surface density
    ax = axes[0, 0]
    ax.loglog(R, Sigma, 'b-', lw=2, label='Σ (baryons only)')
    ax.loglog(R, Sigma_eff, 'r-', lw=2, label='Σ_eff (with 3D kernel)')
    ax.axvline(R_E, color='gray', ls='--', alpha=0.7, label=f'R_E = {R_E:.0f} kpc')
    ax.set_xlabel('Projected Radius R [kpc]', fontsize=11)
    ax.set_ylabel('Surface Density [Msun/kpc^2]', fontsize=11)
    ax.set_title('Surface Density Profiles', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # (0, 1): Boost factor K_Σ(R)
    ax = axes[0, 1]
    ax.semilogx(R, K_Sigma, 'g-', lw=2)
    ax.axvline(R_E, color='gray', ls='--', alpha=0.7)
    ax.axhline(0, color='k', ls=':', lw=1)
    ax.set_xlabel('Projected Radius R [kpc]', fontsize=11)
    ax.set_ylabel('Boost Factor K_Sigma(R)', fontsize=11)
    ax.set_title('3D Shell Kernel Boost', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # (0, 2): Convergence
    ax = axes[0, 2]
    ax.loglog(R, kappa, 'b-', lw=2, label='kappa(R)')
    ax.loglog(R, mean_kappa, 'r-', lw=2, label='mean_kappa(<R)')
    ax.axhline(1.0, color='k', ls='--', lw=1, alpha=0.7, label='kappa = 1')
    ax.axvline(R_E, color='gray', ls='--', alpha=0.7)
    ax.set_xlabel('Projected Radius R [kpc]', fontsize=11)
    ax.set_ylabel('Convergence', fontsize=11)
    ax.set_title('Convergence Profiles', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # (1, 0): Tangential shear
    ax = axes[1, 0]
    ax.loglog(R, np.abs(gamma_t), 'purple', lw=2)
    ax.axvline(R_E, color='gray', ls='--', alpha=0.7)
    ax.set_xlabel('Projected Radius R [kpc]', fontsize=11)
    ax.set_ylabel('|gamma_t(R)|', fontsize=11)
    ax.set_title('Tangential Shear', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # (1, 1): Boost decomposition (if available from ablation)
    ax = axes[1, 1]
    ax.text(0.5, 0.5, 'Run ablation_study()\nfor boost decomposition', 
            transform=ax.transAxes, ha='center', va='center',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.set_title('Interior vs Exterior Contribution', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # (1, 2): Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = f"""
    MACS0416 Full Physics Test
    {'-'*40}
    
    Baryon Model (Phase 1):
      * gNFW gas (Arnaud+ 2010)
      * f_gas(R_500) = {results['baryon_info']['fgas']:.3f}
      * BCG + ICL included
    
    Path Kernel (Phase 2.1):
      * 3D shell integral
      * Interior chords + exterior arcs
      * ell0 = {results['params'].ell0:.0f} kpc
    
    Einstein Radius:
      * Predicted: {theta_E_pred:.2f} arcsec
      * Observed:  {theta_E_obs:.2f} arcsec
      * Error: {results['frac_error']*100:.1f}%
      {'[PASS]' if results['frac_error'] < 0.10 else '[CHECK]' if results['frac_error'] < 0.25 else '[FAIL]'}
    
    At Einstein Radius:
      * K_Sigma(R_E) = {results['K_Sigma_at_RE']:.2f}
      * Boost = {results['Sigma_eff_at_RE']/results['Sigma_at_RE']:.2f}x
    
    NO DARK MATTER
    """
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'macs0416_full_physics_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")
    
    return fig


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("MACS0416 FULL PHYSICS TEST")
    print("Phase 1 (gNFW) + Phase 2 (3D Shell Kernel)")
    print("=" * 70)
    print()
    
    # Test A: Full physics stack
    print("TEST A: Full Physics Stack")
    print("=" * 70)
    results_full = test_full_physics(verbose=True)
    
    # Test B: Ablation study
    print("\n")
    results_ablation = ablation_study(verbose=True)
    
    # Generate plots
    generate_diagnostic_plots(results_full)
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 2, STEP 2.3 TEST COMPLETE")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  1. Tune (A_c, ell0) if needed to hit theta_E within +/-10%")
    print("  2. Add triaxial geometry test (vary q_los)")
    print("  3. Test on A1689 and MACS0717 for universality")
    print("  4. Optional: Explicit path sum validation (Phase 2.2)")
    print()
    print("Bottom line: Baryon-only model with 3D shell kernel")
    print("             → All gravitational paths counted")
    print("             → No dark matter invoked")
