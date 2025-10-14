#!/usr/bin/env python3
"""
Test Physical Gas Models on MACS0416
=====================================

Uses double-β gas profile with f_gas = 0.11 normalization, BCG + ICL,
clumping correction, and LOS elongation to test if we naturally match
the observed Einstein radius.

This should replicate the bracket test success but with physical models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from core.gas_profiles import build_cluster_density_profile
from many_path_model.lensing_utilities import default_cosmology, AbelProjection
from core.cluster_first_kernel import lensing_profiles


def main():
    print("\n" + "="*70)
    print("PHYSICAL GAS MODEL TEST: MACS0416")
    print("="*70)
    
    # Cluster parameters from literature
    cluster_name = "MACS0416"
    z_lens = 0.396
    z_src = 2.0
    M_500 = 1.15e15  # Msun, Umetsu+ 2016
    R_500 = 1200.0  # kpc
    R_200 = 1500.0  # kpc, approximate
    theta_E_obs = 35.0  # arcsec
    
    # LOS geometry
    q_los = 0.75  # Elongation along line of sight
    kappa_ext = 0.08  # External sheet
    
    print(f"\nCluster: {cluster_name}")
    print(f"  z_lens = {z_lens}")
    print(f"  M_500 = {M_500:.2e} Msun")
    print(f"  R_500 = {R_500:.1f} kpc")
    print(f"  θ_E (observed) = {theta_E_obs:.1f} arcsec")
    
    print(f"\nPhysical Model Parameters:")
    print(f"  Gas: Double-β profile")
    print(f"  f_gas target = 0.11 ± 0.02")
    print(f"  Clumping: C₀ = 0.3, η = 2.0")
    print(f"  BCG: M = 2×10¹² Msun, a = 25 kpc")
    print(f"  ICL: M = 8×10¹¹ Msun, rs = 150 kpc")
    print(f"  q_los = {q_los:.2f} (LOS elongation)")
    print(f"  κ_ext = {kappa_ext:.2f} (external sheet)")
    
    # Build radial grid
    r = np.logspace(-1, 3.5, 2000)  # 0.1 to 3000 kpc, high resolution
    
    # Build physical baryon profile
    print(f"\nBuilding physical baryon profile...")
    rho_gas, rho_bcg, rho_icl, rho_total = build_cluster_density_profile(
        r=r,
        M_500=M_500,
        R_500=R_500,
        fgas_target=0.11,
        M_bcg=2e12,
        a_bcg=25.0,
        M_icl=8e11,
        rs_icl=150.0,
        C0_clump=0.3,
        eta_clump=2.0,
        R_200=R_200,
        use_gnfw=False  # Use double-β
    )
    
    # Check masses
    from core.gas_profiles import integrate_mass_spherical
    mask_500 = r <= R_500
    M_gas = integrate_mass_spherical(r[mask_500], rho_gas[mask_500])
    M_bcg_total = integrate_mass_spherical(r[mask_500], rho_bcg[mask_500])
    M_icl_total = integrate_mass_spherical(r[mask_500], rho_icl[mask_500])
    M_baryon = M_gas + M_bcg_total + M_icl_total
    f_gas = M_gas / M_500
    f_bar = M_baryon / M_500
    
    print(f"\n✓ Baryon Budget (<R_500):")
    print(f"  M_gas = {M_gas:.2e} Msun")
    print(f"  M_BCG = {M_bcg_total:.2e} Msun")
    print(f"  M_ICL = {M_icl_total:.2e} Msun")
    print(f"  M_baryon = {M_baryon:.2e} Msun")
    print(f"  f_gas = {f_gas:.3f} (target: 0.110)")
    print(f"  f_bar = {f_bar:.3f}")
    
    # Apply LOS elongation
    print(f"\nApplying LOS geometry...")
    rho_effective = rho_total / q_los
    
    # Compute lensing with cluster-first kernel
    print(f"\nComputing lensing profiles with G³ boost...")
    cosmo = default_cosmology()
    R_eval = np.geomspace(2.0, 1500.0, 200)
    
    K3D_params = {
        'A_c': 10.0,
        'r_gate': 5.0,
        'n_gate': 4,
        'ell0': 180.0,
        'p': 1.2,
        'L1': 1200.0,
        'q': 2.0,
    }
    
    prof = lensing_profiles(R_eval, z_lens, z_src, r, rho_effective, K3D_params, cosmo)
    
    # Add external sheet
    prof['kappa'] += kappa_ext
    prof['mean_kappa'] += kappa_ext
    
    # Recompute Einstein radius
    idx = np.where(prof['mean_kappa'] >= 1.0)[0]
    if idx.size > 0:
        i = idx[-1]
        theta_E_pred = cosmo.physical_to_angular(R_eval[i], z_lens)
    else:
        theta_E_pred = 0.0
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nEinstein Radius:")
    print(f"  Predicted: {theta_E_pred:.2f} arcsec")
    print(f"  Observed:  {theta_E_obs:.2f} arcsec")
    
    if theta_E_pred > 0:
        error = abs(theta_E_pred - theta_E_obs)
        frac_error = error / theta_E_obs
        print(f"  Error: {error:.2f}\" ({100*frac_error:.1f}%)")
        
        if frac_error < 0.15:
            print(f"\n🎉 EXCELLENT: Within ±15% of observed!")
        elif frac_error < 0.25:
            print(f"\n✅ PASS: Within ±25% of observed")
        elif frac_error < 0.5:
            print(f"\n⚠️  MARGINAL: Within ±50% of observed")
        else:
            print(f"\n❌ FAIL: More than 50% off")
    else:
        print(f"\n❌ FAIL: No Einstein radius found")
    
    # Diagnostics at R = 180 kpc
    idx_180 = np.argmin(np.abs(R_eval - 180))
    R_test = R_eval[idx_180]
    
    print(f"\nDiagnostics at R = {R_test:.1f} kpc:")
    print(f"  K_Σ = {prof['K_Sigma'][idx_180]:.2f}")
    print(f"  Σ = {prof['Sigma'][idx_180]:.2e} Msun/kpc²")
    print(f"  Σ_eff = {prof['Sigma_eff'][idx_180]:.2e} Msun/kpc²")
    print(f"  κ = {prof['kappa'][idx_180]:.3f}")
    print(f"  ⟨κ⟩ = {prof['mean_kappa'][idx_180]:.3f}")
    
    # Peak convergence
    max_kappa = np.max(prof['mean_kappa'])
    idx_max = np.argmax(prof['mean_kappa'])
    print(f"\nMax ⟨κ⟩: {max_kappa:.3f} at R = {R_eval[idx_max]:.1f} kpc")
    
    # Create diagnostic plot
    print(f"\nGenerating diagnostic plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Physical Gas Model Test: {cluster_name}', fontsize=14, fontweight='bold')
    
    # Plot 1: Density profiles
    ax = axes[0, 0]
    ax.loglog(r, rho_gas, 'b-', lw=2, label='Gas')
    ax.loglog(r, rho_bcg, 'r-', lw=2, label='BCG')
    ax.loglog(r, rho_icl, 'g-', lw=2, label='ICL')
    ax.loglog(r, rho_total, 'k-', lw=2, label='Total')
    ax.axvline(R_500, color='gray', ls='--', alpha=0.5, label=f'R_500')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('ρ [M☉/kpc³]')
    ax.set_title('3D Density Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Surface density
    ax = axes[0, 1]
    ax.loglog(R_eval, prof['Sigma'], 'k--', lw=2, label='Σ (baryons)')
    ax.loglog(R_eval, prof['Sigma_eff'], 'b-', lw=2, label='Σ_eff (with boost)')
    ax.axhline(prof['Sigma_crit'], color='red', ls=':', lw=2, label='Σ_crit')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Σ [M☉/kpc²]')
    ax.set_title('Surface Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Boost factor
    ax = axes[0, 2]
    ax.semilogx(R_eval, prof['K_Sigma'], 'purple', lw=2)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.axvspan(50, 300, alpha=0.1, color='green', label='Lensing zone')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('K_Σ(R)')
    ax.set_title('G³ Projected Boost')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence
    ax = axes[1, 0]
    ax.semilogx(R_eval, prof['kappa'], 'b-', lw=2, label='κ(R)')
    ax.semilogx(R_eval, prof['mean_kappa'], 'r-', lw=2, label='⟨κ⟩(<R)')
    ax.axhline(1.0, color='green', ls='--', lw=2, label='Einstein condition')
    if theta_E_pred > 0:
        R_E = theta_E_pred * cosmo.angular_diameter_distance_kpc(z_lens) / 206265
        ax.axvline(R_E, color='green', ls=':', alpha=0.7, label=f'θ_E = {theta_E_pred:.1f}"')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Convergence')
    ax.set_title('Lensing Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative mass
    ax = axes[1, 1]
    M_enc = np.array([integrate_mass_spherical(r[r<=Ri], rho_total[r<=Ri]) for Ri in R_eval])
    ax.loglog(R_eval, M_enc, 'k-', lw=2, label='M_baryon(<R)')
    ax.axhline(M_500, color='red', ls='--', label='M_500')
    ax.axvline(R_500, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('M(<R) [M☉]')
    ax.set_title('Enclosed Mass')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Mass fractions
    ax = axes[1, 2]
    f_gas_r = np.array([integrate_mass_spherical(r[r<=Ri], rho_gas[r<=Ri]) / 
                        max(integrate_mass_spherical(r[r<=Ri], rho_total[r<=Ri]), 1e-30) 
                        for Ri in R_eval])
    ax.semilogx(R_eval, f_gas_r, 'b-', lw=2, label='f_gas(R)')
    ax.axhline(0.11, color='green', ls='--', label='Target f_gas')
    ax.axvline(R_500, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('f_gas(<R)')
    ax.set_title('Gas Mass Fraction')
    ax.set_ylim(0, 0.2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = Path('results/plots/physical_gas_test_macs0416.png')
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved: {plot_path}")
    
    # Save results
    results = {
        'cluster': cluster_name,
        'z_lens': z_lens,
        'M_500': float(M_500),
        'R_500': float(R_500),
        'baryon_budget': {
            'M_gas': float(M_gas),
            'M_bcg': float(M_bcg_total),
            'M_icl': float(M_icl_total),
            'f_gas': float(f_gas),
            'f_bar': float(f_bar),
        },
        'geometry': {
            'q_los': q_los,
            'kappa_ext': kappa_ext,
        },
        'kernel': K3D_params,
        'results': {
            'theta_E_pred': float(theta_E_pred),
            'theta_E_obs': theta_E_obs,
            'error_arcsec': float(error) if theta_E_pred > 0 else None,
            'fractional_error': float(frac_error) if theta_E_pred > 0 else None,
            'max_kappa': float(max_kappa),
        }
    }
    
    out_path = Path('results/physical_gas_test_macs0416.json')
    out_path.parent.mkdir(exist_ok=True)
    import json
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved: {out_path}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
