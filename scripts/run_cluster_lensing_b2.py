# scripts/run_cluster_lensing_b2.py
"""
Track B2: Cluster-first lensing prediction runner.

Uses the isotropic, long-coherence cluster kernel (NOT galaxy-tuned parameters).
This script runs lensing predictions on a single cluster with user-specified
kernel parameters, producing Einstein radius and boost diagnostics.

Key differences from B1:
  - Clean restart with cluster-appropriate physics
  - Isotropic kernel (no disk anisotropy)
  - Long coherence length ell0 ~ 100-300 kpc
  - Boost cannot collapse to zero by construction

Usage:
    python scripts/run_cluster_lensing_b2.py --cluster MACSJ0416 --Ac 10 --ell0 180
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from many_path_model.cluster_data_loader import load_cluster_profile
from many_path_model.lensing_utilities import default_cosmology
from core.cluster_first_kernel import lensing_profiles, K3D_isotropic


def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description='B2 cluster lensing with cluster-first kernel')
    
    # Cluster selection
    ap.add_argument('--cluster', type=str, required=True,
                    help='Cluster name (e.g., MACSJ0416, MACSJ0717, A1689)')
    ap.add_argument('--zsrc', type=float, default=2.0,
                    help='Source redshift for lensing (default: 2.0)')
    
    # Kernel hyperparameters
    ap.add_argument('--Ac', type=float, default=8.0,
                    help='Cluster amplitude (dimensionless, default: 8.0)')
    ap.add_argument('--rg', type=float, default=5.0,
                    help='Gate radius in kpc (default: 5.0)')
    ap.add_argument('--ng', type=int, default=4,
                    help='Gate steepness (default: 4)')
    ap.add_argument('--ell0', type=float, default=150.0,
                    help='Coherence length in kpc (default: 150.0)')
    ap.add_argument('--p', type=float, default=1.2,
                    help='Growth power (default: 1.2)')
    ap.add_argument('--L1', type=float, default=1200.0,
                    help='Taper scale in kpc (default: 1200.0)')
    ap.add_argument('--q', type=float, default=2.0,
                    help='Taper steepness (default: 2.0)')
    
    # Profile evaluation parameters
    ap.add_argument('--Rmin', type=float, default=2.0,
                    help='Minimum radius in kpc (default: 2.0)')
    ap.add_argument('--Rmax', type=float, default=1500.0,
                    help='Maximum radius in kpc (default: 1500.0)')
    ap.add_argument('--nR', type=int, default=200,
                    help='Number of radial points (default: 200)')
    
    # Output
    ap.add_argument('--out', type=str, default='results/cluster_b2_single.json',
                    help='Output JSON file path')
    ap.add_argument('--plot', action='store_true',
                    help='Generate diagnostic plots')
    
    return ap.parse_args()


def print_diagnostics(cluster_name, z_lens, zsrc, prof, K3D_params):
    """Print key diagnostic outputs."""
    R = prof['R']
    K_Sigma = prof['K_Sigma']
    
    # Find median boost in 50-300 kpc range (where lensing is strong)
    mask = (R > 50) & (R < 300)
    if mask.any():
        median_boost = np.median(K_Sigma[mask])
    else:
        median_boost = np.nan
    
    # Max boost
    max_boost = np.max(K_Sigma)
    
    print("\n" + "="*70)
    print(f"CLUSTER-FIRST LENSING PREDICTION (Track B2)")
    print("="*70)
    print(f"\nCluster: {cluster_name}")
    print(f"Redshift: z_lens={z_lens:.3f}, z_src={zsrc:.2f}")
    print(f"\nKernel Parameters:")
    print(f"  A_c (amplitude):        {K3D_params['A_c']:.2f}")
    print(f"  ell0 (coherence, kpc):  {K3D_params['ell0']:.1f}")
    print(f"  r_gate (kpc):           {K3D_params['r_gate']:.1f}")
    print(f"  n_gate:                 {K3D_params['n_gate']}")
    print(f"  p (growth power):       {K3D_params['p']:.2f}")
    print(f"  L1 (taper, kpc):        {K3D_params['L1']:.1f}")
    print(f"  q (taper power):        {K3D_params['q']:.2f}")
    print(f"\nLensing Results:")
    print(f"  Einstein radius:        {prof['theta_E_arcsec']:.2f} arcsec")
    print(f"  Median boost (50-300 kpc): {median_boost:.3f}")
    print(f"  Max boost K_Sigma:      {max_boost:.3f}")
    print(f"\nPhysical Check:")
    if median_boost > 0.01:
        print(f"  ✓ Boost is NONZERO (as expected for cluster-first kernel)")
    else:
        print(f"  ✗ WARNING: Boost near zero (check parameters)")
    
    if prof['theta_E_arcsec'] > 0:
        print(f"  ✓ Einstein radius predicted")
    else:
        print(f"  ✗ No Einstein radius (boost too weak or Sigma too low)")
    print("="*70 + "\n")


def make_diagnostic_plots(cluster_name, prof, K3D_params, r_grid, rho_r):
    """Generate diagnostic plots for B2 results."""
    R = prof['R']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Track B2: Cluster-First Kernel — {cluster_name}', fontsize=14, fontweight='bold')
    
    # Plot 1: 3D boost kernel K_3D(r)
    ax = axes[0, 0]
    K_r = K3D_isotropic(r_grid, **K3D_params)
    ax.loglog(r_grid, K_r, 'b-', lw=2)
    ax.axvline(K3D_params['ell0'], color='orange', ls='--', label=f"ell0={K3D_params['ell0']:.0f} kpc")
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('K_3D(r)')
    ax.set_title('3D Boost Kernel')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Baryon density profile
    ax = axes[0, 1]
    ax.loglog(r_grid, rho_r, 'k-', lw=2, label='Total baryons')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('ρ(r) [M☉/kpc³]')
    ax.set_title('Baryon Density Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Projected boost K_Sigma(R)
    ax = axes[0, 2]
    ax.semilogx(R, prof['K_Sigma'], 'r-', lw=2)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.axvspan(50, 300, alpha=0.1, color='green', label='Strong lensing zone')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('K_Σ(R)')
    ax.set_title('Projected Boost Factor')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Surface densities
    ax = axes[1, 0]
    ax.loglog(R, prof['Sigma'], 'k--', lw=2, label='Σ (baryons only)')
    ax.loglog(R, prof['Sigma_eff'], 'b-', lw=2, label='Σ_eff (with boost)')
    ax.axhline(prof['Sigma_crit'], color='red', ls=':', lw=2, label='Σ_crit')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Σ [M☉/kpc²]')
    ax.set_title('Surface Density Profiles')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Convergence profiles
    ax = axes[1, 1]
    ax.semilogx(R, prof['kappa'], 'b-', lw=2, label='κ(R)')
    ax.semilogx(R, prof['mean_kappa'], 'r-', lw=2, label='⟨κ⟩(<R)')
    ax.axhline(1.0, color='green', ls='--', lw=2, label='Einstein condition')
    if prof['theta_E_arcsec'] > 0:
        # Find R corresponding to Einstein radius
        theta_E_kpc = prof['theta_E_arcsec'] / default_cosmology().kpc_to_arcsec(1.0, 0.4)  # approx
        ax.axvline(theta_E_kpc, color='green', ls=':', alpha=0.7, label=f"θ_E≈{prof['theta_E_arcsec']:.1f}''")
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Convergence')
    ax.set_title('Lensing Convergence')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 6: Tangential shear
    ax = axes[1, 2]
    ax.semilogx(R, prof['gamma_t'], 'purple', lw=2)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('γ_t(R)')
    ax.set_title('Tangential Shear')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path('results') / 'plots' / f'cluster_b2_{cluster_name.lower()}.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostic plot saved: {plot_path}")
    
    return fig


def main():
    """Main execution function."""
    args = parse_args()
    
    # Initialize cosmology
    cosmo = default_cosmology()
    
    # Load cluster baryon profile
    print(f"\nLoading cluster data: {args.cluster}...")
    z_lens, r_grid, rho_r = load_cluster_profile(args.cluster)
    print(f"  Loaded {len(r_grid)} radial points")
    print(f"  Radial range: {r_grid[0]:.2f} - {r_grid[-1]:.2f} kpc")
    print(f"  Total baryon mass: {np.trapz(4*np.pi*r_grid**2*rho_r, r_grid):.2e} Msun")
    
    # Set up radii for lensing evaluation
    R = np.geomspace(args.Rmin, args.Rmax, args.nR)
    
    # Package kernel parameters
    K3D_params = dict(
        A_c=args.Ac,
        r_gate=args.rg,
        n_gate=args.ng,
        ell0=args.ell0,
        p=args.p,
        L1=args.L1,
        q=args.q
    )
    
    # Compute lensing profiles
    print("\nComputing lensing profiles with cluster-first kernel...")
    prof = lensing_profiles(R, z_lens, args.zsrc, r_grid, rho_r, K3D_params, cosmo)
    
    # Print diagnostics
    print_diagnostics(args.cluster, z_lens, args.zsrc, prof, K3D_params)
    
    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare output (convert numpy arrays to lists for JSON)
    output = {
        'cluster': args.cluster,
        'z_lens': float(z_lens),
        'z_src': args.zsrc,
        'kernel_params': K3D_params,
        'theta_E_arcsec': float(prof['theta_E_arcsec']),
        'median_boost_50_300kpc': float(np.median(prof['K_Sigma'][(R>50)&(R<300)])),
        'max_boost': float(np.max(prof['K_Sigma'])),
        'profiles': {
            'R_kpc': R.tolist(),
            'K_Sigma': prof['K_Sigma'].tolist(),
            'kappa': prof['kappa'].tolist(),
            'mean_kappa': prof['mean_kappa'].tolist(),
            'gamma_t': prof['gamma_t'].tolist()
        }
    }
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved: {out_path}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating diagnostic plots...")
        make_diagnostic_plots(args.cluster, prof, K3D_params, r_grid, rho_r)
    
    print("\n✓ B2 single-cluster prediction complete.\n")


if __name__ == '__main__':
    main()
