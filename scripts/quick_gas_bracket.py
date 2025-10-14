#!/usr/bin/env python3
"""
Quick Gas Bracket Test
======================

Tests whether scaling gas mass to realistic f_gas values closes the lensing gap.
This is a "does it work in principle?" test before implementing full physics.

Usage:
    python scripts/quick_gas_bracket.py --cluster MACSJ0416 --scale 3.5
    python scripts/quick_gas_bracket.py --cluster MACSJ0416 --scale 3.5 --kappa_ext 0.08
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import argparse
import json
from many_path_model.cluster_data_loader import load_cluster_profile
from many_path_model.lensing_utilities import default_cosmology
from core.cluster_first_kernel import lensing_profiles

# Observed Einstein radii
OBSERVED_THETA_E = {
    'MACSJ0416': 35.0,
    'MACSJ0717': 55.0,
    'ABELL_1689': 45.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Quick gas scaling bracket test')
    parser.add_argument('--cluster', required=True, help='Cluster name')
    parser.add_argument('--scale', type=float, default=3.5, 
                       help='Gas mass multiplier (default: 3.5 to fix f_gas)')
    parser.add_argument('--kappa_ext', type=float, default=0.0,
                       help='External sheet convergence (default: 0.0)')
    parser.add_argument('--q_los', type=float, default=1.0,
                       help='LOS axis ratio (default: 1.0, use <1 for elongation)')
    
    # Kernel parameters
    parser.add_argument('--Ac', type=float, default=10.0)
    parser.add_argument('--ell0', type=float, default=180.0)
    parser.add_argument('--use_boost', action='store_true',
                       help='Apply cluster-first boost kernel')
    
    return parser.parse_args()


def bracket_test(cluster_name, gas_scale, kappa_ext=0.0, q_los=1.0, 
                 use_boost=True, K3D_params=None):
    """
    Run bracket test with scaled gas.
    
    Parameters
    ----------
    cluster_name : str
        Cluster to test
    gas_scale : float
        Factor to multiply gas density (e.g., 3.5 to fix f_gas)
    kappa_ext : float
        External sheet convergence to add
    q_los : float
        Line-of-sight axis ratio (< 1 means elongated)
    use_boost : bool
        Whether to apply cluster-first kernel boost
    K3D_params : dict
        Kernel parameters if using boost
    """
    
    # Load data
    z_lens, r_grid, rho_total = load_cluster_profile(cluster_name)
    
    # Get gas and star components separately
    from many_path_model.cluster_data_loader import ClusterDataLoader
    loader = ClusterDataLoader()
    data = loader.load_cluster(cluster_name, validate=False)
    
    # Scale gas only
    rho_gas_scaled = data.rho_gas * gas_scale
    rho_total_scaled = rho_gas_scaled + data.rho_stars
    
    # Apply LOS elongation (decreases projected radius, increases Σ)
    # Simple approximation: Σ ∝ 1/q_los for fixed M(<R)
    effective_rho = rho_total_scaled / q_los
    
    # Compute lensing
    cosmo = default_cosmology()
    R = np.geomspace(2.0, 1500.0, 200)
    
    if use_boost and K3D_params is not None:
        prof = lensing_profiles(R, z_lens, 2.0, r_grid, effective_rho, K3D_params, cosmo)
    else:
        # No boost case - just baryons
        from many_path_model.lensing_utilities import AbelProjection
        projector = AbelProjection()
        Sigma = projector.project_density_to_surface(r_grid, effective_rho, R)
        Sigma_crit = cosmo.critical_surface_density(z_lens, 2.0)
        
        kappa = Sigma / Sigma_crit
        
        # Mean convergence
        area = np.pi * R**2
        cum = np.cumsum((Sigma[1:] + Sigma[:-1]) / 2.0 * np.diff(area))
        mean_kappa = np.zeros_like(kappa)
        mean_kappa[1:] = cum / area[1:] / Sigma_crit
        mean_kappa[0] = mean_kappa[1]
        
        prof = {
            'R': R,
            'kappa': kappa,
            'mean_kappa': mean_kappa,
            'theta_E_arcsec': 0.0,
            'K_Sigma': np.zeros_like(R),
        }
        
        # Find Einstein radius
        idx = np.where(mean_kappa >= 1.0)[0]
        if idx.size > 0:
            i = idx[-1]
            prof['theta_E_arcsec'] = cosmo.physical_to_angular(R[i], z_lens)
    
    # Add external sheet
    prof['kappa'] += kappa_ext
    prof['mean_kappa'] += kappa_ext
    
    # Recompute Einstein radius with external sheet
    idx = np.where(prof['mean_kappa'] >= 1.0)[0]
    if idx.size > 0:
        i = idx[-1]
        prof['theta_E_arcsec'] = cosmo.physical_to_angular(R[i], z_lens)
    
    return prof


def main():
    args = parse_args()
    
    K3D_params = {
        'A_c': args.Ac,
        'r_gate': 5.0,
        'n_gate': 4,
        'ell0': args.ell0,
        'p': 1.2,
        'L1': 1200.0,
        'q': 2.0,
    }
    
    print("\n" + "="*70)
    print("QUICK GAS BRACKET TEST")
    print("="*70)
    print(f"\nCluster: {args.cluster}")
    print(f"Gas scale factor: {args.scale:.2f}×")
    print(f"External κ: {args.kappa_ext:.3f}")
    print(f"LOS axis ratio q: {args.q_los:.2f}")
    print(f"Use boost: {args.use_boost}")
    if args.use_boost:
        print(f"  A_c = {args.Ac:.1f}, ℓ₀ = {args.ell0:.1f} kpc")
    
    # Run test
    prof = bracket_test(
        args.cluster, 
        args.scale, 
        args.kappa_ext, 
        args.q_los,
        args.use_boost,
        K3D_params if args.use_boost else None
    )
    
    # Results
    theta_E_obs = OBSERVED_THETA_E.get(args.cluster, None)
    
    print(f"\n{'─'*70}")
    print("RESULTS")
    print(f"{'─'*70}")
    print(f"\nPredicted Einstein radius: {prof['theta_E_arcsec']:.2f} arcsec")
    if theta_E_obs:
        print(f"Observed Einstein radius:  {theta_E_obs:.2f} arcsec")
        error = abs(prof['theta_E_arcsec'] - theta_E_obs)
        frac_error = error / theta_E_obs
        print(f"Error: {error:.2f}\" ({100*frac_error:.1f}%)")
        
        if frac_error < 0.25:
            print(f"\n✅ PASS: Within ±25% of observed")
        elif frac_error < 0.5:
            print(f"\n⚠️  MARGINAL: Within ±50% of observed")
        else:
            print(f"\n❌ FAIL: More than 50% off")
    
    # Peak convergence
    max_kappa = np.max(prof['mean_kappa'])
    idx_max = np.argmax(prof['mean_kappa'])
    print(f"\nMax ⟨κ⟩: {max_kappa:.3f} at R = {prof['R'][idx_max]:.1f} kpc")
    
    # Boost check
    if args.use_boost:
        idx_180 = np.argmin(np.abs(prof['R'] - 180))
        print(f"\nAt R=180 kpc:")
        print(f"  K_Σ = {prof['K_Sigma'][idx_180]:.2f}")
        print(f"  κ = {prof['kappa'][idx_180]:.3f}")
        print(f"  ⟨κ⟩ = {prof['mean_kappa'][idx_180]:.3f}")
    
    # Save results
    output = {
        'cluster': args.cluster,
        'gas_scale': args.scale,
        'kappa_ext': args.kappa_ext,
        'q_los': args.q_los,
        'use_boost': args.use_boost,
        'theta_E_pred': float(prof['theta_E_arcsec']),
        'theta_E_obs': theta_E_obs,
        'max_kappa': float(max_kappa),
    }
    
    if args.use_boost:
        output['kernel'] = K3D_params
    
    out_path = Path('results') / f'bracket_test_{args.cluster.lower()}.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {out_path}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
