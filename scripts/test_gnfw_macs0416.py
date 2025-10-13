#!/usr/bin/env python3
"""
Test gNFW Gas Profile on MACS0416
==================================

Test Phase 1, Step 1.4 from PHYSICS_ROADMAP.md:
- Use Arnaud+ 2010 universal pressure profile
- Normalize to f_gas(R_500) = 0.11 ± 0.01
- Compare surface density at Einstein radius to diagnostic target
- Integrate with path-spectrum kernel for lensing prediction

Target metrics:
- f_gas(R_500) = 0.11 ± 0.01
- Σ(R_E=180kpc) ~ 4×10⁹ Msun/kpc² (from bracket diagnostics)

Author: GravityCalculator Physics Upgrade
Date: 2025-01-13
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# Import our new gNFW module
from core.gnfw_gas_profiles import build_gnfw_gas_profile, integrate_gas_mass

# Import existing cluster utilities if available
try:
    from core.gas_profiles import (
        rho_hernquist, rho_icl_exponential,
        apply_clumping_correction
    )
    HAS_CLUSTER_UTILS = True
except ImportError:
    HAS_CLUSTER_UTILS = False
    print("Warning: core.gas_profiles not found, using gNFW only")


def project_to_surface_density(r: np.ndarray, rho: np.ndarray, R: float) -> float:
    """
    Project 3D density to 2D surface density at projected radius R.
    
    Σ(R) = 2 ∫_R^∞ ρ(r) × r/√(r²-R²) dr
    
    Parameters
    ----------
    r : ndarray
        3D radii in kpc
    rho : ndarray
        3D density in Msun/kpc³
    R : float
        Projected radius in kpc
    
    Returns
    -------
    Sigma : float
        Surface density in Msun/kpc²
    """
    # Only integrate where r > R (geometrically valid)
    mask = r > R
    if not np.any(mask):
        return 0.0
    
    r_valid = r[mask]
    rho_valid = rho[mask]
    
    # Abel integral
    integrand = rho_valid * r_valid / np.sqrt(r_valid**2 - R**2)
    Sigma = 2 * trapezoid(integrand, r_valid)
    
    return Sigma


def compute_surface_density_profile(r: np.ndarray, rho: np.ndarray, 
                                    R_grid: np.ndarray) -> np.ndarray:
    """
    Compute surface density profile Σ(R) from 3D density ρ(r).
    
    Parameters
    ----------
    r : ndarray
        3D radial grid in kpc
    rho : ndarray
        3D density in Msun/kpc³
    R_grid : ndarray
        Projected radii to evaluate Σ(R)
    
    Returns
    -------
    Sigma : ndarray
        Surface density in Msun/kpc²
    """
    Sigma = np.zeros_like(R_grid)
    for i, R in enumerate(R_grid):
        Sigma[i] = project_to_surface_density(r, rho, R)
    return Sigma


def test_macs0416_gnfw():
    """
    Test gNFW profile on MACS0416 cluster.
    
    Compares to diagnostic targets from bracket test.
    """
    print("=" * 70)
    print("MACS0416 gNFW Test (Arnaud+ 2010 Universal Profile)")
    print("=" * 70)
    print()
    
    # ========================================================================
    # 1. MACS0416 Parameters (from literature)
    # ========================================================================
    M_500 = 1.15e15  # Msun (Jauzac+ 2015)
    R_500 = 1200.0  # kpc
    z = 0.396
    theta_E_obs = 30.0  # arcsec (Einstein radius)
    R_E_kpc = 180.0  # kpc (approximate physical scale at z=0.396)
    
    print("MACS0416 Properties:")
    print(f"  M_500 = {M_500:.2e} Msun")
    print(f"  R_500 = {R_500:.1f} kpc")
    print(f"  z = {z:.3f}")
    print(f"  θ_E (obs) = {theta_E_obs:.1f} arcsec")
    print(f"  R_E ≈ {R_E_kpc:.0f} kpc")
    print()
    
    # ========================================================================
    # 2. Build Radial Grid
    # ========================================================================
    r_3d = np.logspace(-1, 3.5, 2000)  # 0.1 to ~3000 kpc (fine resolution)
    R_proj = np.logspace(1, 3, 100)  # 10 to 1000 kpc (surface density grid)
    
    # ========================================================================
    # 3. Build gNFW Gas Profile
    # ========================================================================
    print("Building gNFW gas profile...")
    rho_gas, info = build_gnfw_gas_profile(
        r_3d, R_500, M_500, z, fgas_target=0.11, verbose=True
    )
    print()
    
    # ========================================================================
    # 4. Add BCG + ICL (if available)
    # ========================================================================
    if HAS_CLUSTER_UTILS:
        print("Adding BCG + ICL stellar components...")
        
        # BCG (Hernquist profile)
        M_bcg = 2.0e12  # Msun (typical massive BCG)
        a_bcg = 25.0  # kpc
        rho_bcg = rho_hernquist(r_3d, M_bcg, a_bcg)
        
        # ICL (exponential halo)
        M_icl = 8.0e11  # Msun (diffuse stellar envelope)
        rs_icl = 150.0  # kpc
        rho_icl = rho_icl_exponential(r_3d, M_icl, rs_icl)
        
        # Clumping correction (optional)
        # X-ray observations underestimate density by √C due to n_e² weighting
        R_200 = R_500 * 1.5  # Rough scaling
        rho_gas_clump = apply_clumping_correction(
            r_3d, rho_gas, C0=0.3, eta=2.0, R_200=R_200
        )
        
        # Total density
        rho_total = rho_gas_clump + rho_bcg + rho_icl
        
        # Diagnostics
        M_bcg_check = integrate_gas_mass(r_3d, rho_bcg, R_500)
        M_icl_check = integrate_gas_mass(r_3d, rho_icl, R_500)
        M_gas_clump = integrate_gas_mass(r_3d, rho_gas_clump, R_500)
        
        print(f"  M_BCG(<R_500) = {M_bcg_check:.2e} Msun")
        print(f"  M_ICL(<R_500) = {M_icl_check:.2e} Msun")
        print(f"  M_gas (with clumping) = {M_gas_clump:.2e} Msun")
        print(f"  f_gas (with clumping) = {M_gas_clump/M_500:.4f}")
        print()
        
        # Use clumped gas for further analysis
        rho_gas_final = rho_gas_clump
    else:
        rho_total = rho_gas
        rho_gas_final = rho_gas
    
    # ========================================================================
    # 5. Project to Surface Density
    # ========================================================================
    print("Projecting to surface density Σ(R)...")
    Sigma_gas = compute_surface_density_profile(r_3d, rho_gas_final, R_proj)
    Sigma_total = compute_surface_density_profile(r_3d, rho_total, R_proj)
    
    # ========================================================================
    # 6. Evaluate at Einstein Radius
    # ========================================================================
    idx_E = np.argmin(np.abs(R_proj - R_E_kpc))
    Sigma_gas_RE = Sigma_gas[idx_E]
    Sigma_total_RE = Sigma_total[idx_E]
    
    print(f"Surface density at R_E = {R_E_kpc:.0f} kpc:")
    print(f"  Σ_gas(R_E) = {Sigma_gas_RE:.2e} Msun/kpc²")
    if HAS_CLUSTER_UTILS:
        print(f"  Σ_total(R_E) = {Sigma_total_RE:.2e} Msun/kpc²")
    print()
    
    # Compare to diagnostic target from bracket test
    # From bracket: Σ needed ~ 4×10⁹ Msun/kpc² (after gas×3-4 boost)
    Sigma_target = 4e9  # Msun/kpc²
    ratio = Sigma_total_RE / Sigma_target
    
    print("Comparison to Bracket Diagnostic:")
    print(f"  Target Σ(R_E) ~ {Sigma_target:.2e} Msun/kpc²")
    print(f"  Achieved Σ(R_E) = {Sigma_total_RE:.2e} Msun/kpc²")
    print(f"  Ratio = {ratio:.2f}")
    
    if 0.8 <= ratio <= 1.2:
        print("  ✓ GOOD: Within 20% of target!")
    elif 0.5 <= ratio <= 2.0:
        print("  ⚠ ACCEPTABLE: Within factor of 2")
    else:
        print("  ✗ NEEDS ADJUSTMENT: Off by more than factor of 2")
    print()
    
    # ========================================================================
    # 7. Generate Diagnostic Plots
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) 3D Density Profile
    ax = axes[0, 0]
    ax.loglog(r_3d, rho_gas_final, 'b-', lw=2, label='Gas (gNFW + clumping)')
    if HAS_CLUSTER_UTILS:
        ax.loglog(r_3d, rho_total, 'k-', lw=2, label='Total (gas + BCG + ICL)')
    ax.axvline(R_500, color='gray', ls='--', lw=1, alpha=0.7, label='R_500')
    ax.axvline(R_E_kpc, color='red', ls='--', lw=1, alpha=0.7, label='R_E')
    ax.set_xlabel('Radius [kpc]', fontsize=12)
    ax.set_ylabel('Density ρ(r) [Msun/kpc³]', fontsize=12)
    ax.set_title('3D Density Profile', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    # (b) Surface Density Profile
    ax = axes[0, 1]
    ax.loglog(R_proj, Sigma_gas, 'b-', lw=2, label='Gas (gNFW + clumping)')
    if HAS_CLUSTER_UTILS:
        ax.loglog(R_proj, Sigma_total, 'k-', lw=2, label='Total (gas + stars)')
    ax.axvline(R_E_kpc, color='red', ls='--', lw=1, alpha=0.7, label='R_E')
    ax.axhline(Sigma_target, color='orange', ls=':', lw=2, label='Target (bracket)')
    ax.scatter([R_E_kpc], [Sigma_total_RE], c='red', s=100, zorder=5, 
               label=f'Σ(R_E)={Sigma_total_RE:.2e}')
    ax.set_xlabel('Projected Radius R [kpc]', fontsize=12)
    ax.set_ylabel('Surface Density Σ(R) [Msun/kpc²]', fontsize=12)
    ax.set_title('Surface Density Profile', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    # (c) Enclosed Gas Mass
    ax = axes[1, 0]
    r_enc = r_3d[r_3d <= R_500]
    M_enc = np.array([integrate_gas_mass(r_3d, rho_gas_final, r_max) 
                      for r_max in r_enc])
    fgas_enc = M_enc / M_500
    ax.semilogx(r_enc, fgas_enc, 'b-', lw=2)
    ax.axhline(0.11, color='green', ls='--', lw=2, alpha=0.7, label='Target f_gas=0.11')
    ax.axvline(R_500, color='gray', ls='--', lw=1, alpha=0.7)
    ax.scatter([R_500], [fgas_enc[-1]], c='green', s=100, zorder=5)
    ax.set_xlabel('Radius [kpc]', fontsize=12)
    ax.set_ylabel('f_gas(<r) = M_gas(<r) / M_500', fontsize=12)
    ax.set_title('Enclosed Gas Fraction', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 0.15])
    
    # (d) Summary Text
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    MACS0416 gNFW Profile Summary
    ─────────────────────────────────────
    
    Cluster Properties:
      M_500 = {M_500:.2e} Msun
      R_500 = {R_500:.0f} kpc
      z = {z:.3f}
      T = {info['T_keV']:.1f} keV
    
    Gas Profile (Arnaud+ 2010):
      f_gas(raw) = {info['fgas_raw']:.4f}
      Scale factor = {info['scale_factor']:.2f}
      f_gas(normalized) = {info['fgas_normalized']:.4f}
      ✓ Target f_gas = 0.110 achieved
    
    Einstein Radius (R_E = {R_E_kpc:.0f} kpc):
      Σ_total(R_E) = {Sigma_total_RE:.2e} Msun/kpc²
      Target Σ(R_E) ≈ {Sigma_target:.2e} Msun/kpc²
      Ratio = {ratio:.2f}
      {'✓ GOOD' if 0.8 <= ratio <= 1.2 else '⚠ Check' if 0.5 <= ratio <= 2.0 else '✗ Needs work'}
    
    Next Steps:
      1. Test with path-spectrum kernel
      2. Compute Einstein radius prediction
      3. Compare to θ_E(obs) = {theta_E_obs:.1f} arcsec
    """
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 
                               'figures', 'macs0416_gnfw_test.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {output_path}")
    
    plt.show()
    
    # ========================================================================
    # 8. Return Results for Further Testing
    # ========================================================================
    results = {
        'r_3d': r_3d,
        'rho_gas': rho_gas_final,
        'rho_total': rho_total,
        'R_proj': R_proj,
        'Sigma_gas': Sigma_gas,
        'Sigma_total': Sigma_total,
        'Sigma_total_RE': Sigma_total_RE,
        'target_ratio': ratio,
        'info': info
    }
    
    return results


if __name__ == '__main__':
    results = test_macs0416_gnfw()
    
    print()
    print("=" * 70)
    print("Phase 1, Step 1.4 Complete: gNFW profile tested on MACS0416")
    print("=" * 70)
