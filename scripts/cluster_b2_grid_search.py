# scripts/cluster_b2_grid_search.py
"""
Track B2: Parameter grid search for cluster-first kernel.

Sweeps (A_c, ell0) parameter space to find optimal cluster-first parameters
that match observed Einstein radii for clusters with strong lensing data.

Usage:
    python scripts/cluster_b2_grid_search.py --clusters MACSJ0416 MACSJ0717 A1689
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from many_path_model.cluster_data_loader import load_cluster_profile
from many_path_model.lensing_utilities import default_cosmology
from core.cluster_first_kernel import lensing_profiles


# Known Einstein radii for priority clusters (arcsec)
OBSERVED_EINSTEIN_RADII = {
    'MACSJ0416': 35.0,  # Frontier Fields
    'MACSJ0717': 55.0,  # Frontier Fields  
    'ABELL_1689': 45.0  # Frontier Fields (alternate name: A1689)
}

# Alternative names
CLUSTER_NAME_MAP = {
    'A1689': 'ABELL_1689',
    'MACS0416': 'MACSJ0416',
    'MACS0717': 'MACSJ0717'
}


def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description='B2 cluster parameter grid search')
    
    # Cluster selection
    ap.add_argument('--clusters', nargs='+', required=True,
                    help='Cluster names to fit')
    ap.add_argument('--zsrc', type=float, default=2.0,
                    help='Source redshift')
    
    # Grid parameters
    ap.add_argument('--Ac_min', type=float, default=5.0)
    ap.add_argument('--Ac_max', type=float, default=30.0)
    ap.add_argument('--Ac_n', type=int, default=15,
                    help='Number of A_c values')
    
    ap.add_argument('--ell0_min', type=float, default=100.0)
    ap.add_argument('--ell0_max', type=float, default=400.0)
    ap.add_argument('--ell0_n', type=int, default=15,
                    help='Number of ell0 values')
    
    # Fixed parameters
    ap.add_argument('--rg', type=float, default=5.0)
    ap.add_argument('--ng', type=int, default=4)
    ap.add_argument('--p', type=float, default=1.2)
    ap.add_argument('--L1', type=float, default=1200.0)
    ap.add_argument('--q', type=float, default=2.0)
    
    # Output
    ap.add_argument('--out', type=str, default='results/cluster_b2_grid_search.json')
    ap.add_argument('--plot', action='store_true', help='Generate diagnostic plots')
    
    return ap.parse_args()


def normalize_cluster_name(name):
    """Normalize cluster name to standard form."""
    return CLUSTER_NAME_MAP.get(name, name.upper())


def compute_cluster_lensing(cluster_name, K3D_params, zsrc=2.0):
    """
    Compute lensing prediction for a cluster with given parameters.
    
    Returns theta_E_arcsec and median boost.
    """
    try:
        cosmo = default_cosmology()
        z_lens, r_grid, rho_r = load_cluster_profile(cluster_name)
        
        # Evaluation radii
        R = np.geomspace(2.0, 1500.0, 200)
        
        # Compute profiles
        prof = lensing_profiles(R, z_lens, zsrc, r_grid, rho_r, K3D_params, cosmo)
        
        # Extract metrics
        theta_E = prof['theta_E_arcsec']
        median_boost = np.median(prof['K_Sigma'][(R>50)&(R<300)])
        
        return {
            'theta_E_arcsec': float(theta_E),
            'median_boost': float(median_boost),
            'success': True
        }
    except Exception as e:
        return {
            'theta_E_arcsec': 0.0,
            'median_boost': 0.0,
            'success': False,
            'error': str(e)
        }


def run_grid_search(cluster_name, Ac_grid, ell0_grid, fixed_params, zsrc=2.0):
    """
    Run grid search over (A_c, ell0) for a single cluster.
    
    Returns grid of theta_E predictions and best-fit parameters.
    """
    print(f"\n{'='*70}")
    print(f"Grid Search: {cluster_name}")
    print(f"{'='*70}")
    
    n_Ac = len(Ac_grid)
    n_ell0 = len(ell0_grid)
    
    theta_E_grid = np.zeros((n_Ac, n_ell0))
    boost_grid = np.zeros((n_Ac, n_ell0))
    
    # Get observed Einstein radius
    theta_E_obs = OBSERVED_EINSTEIN_RADII.get(cluster_name, None)
    if theta_E_obs is None:
        print(f"⚠️  WARNING: No observed Einstein radius for {cluster_name}")
        theta_E_obs = 35.0  # default
    
    print(f"Observed Einstein radius: {theta_E_obs:.1f} arcsec")
    print(f"Grid: A_c ∈ [{Ac_grid[0]:.1f}, {Ac_grid[-1]:.1f}] ({n_Ac} points)")
    print(f"      ell0 ∈ [{ell0_grid[0]:.1f}, {ell0_grid[-1]:.1f}] kpc ({n_ell0} points)")
    print(f"Total evaluations: {n_Ac * n_ell0}")
    
    # Grid search with progress bar
    pbar = tqdm(total=n_Ac * n_ell0, desc="Grid search")
    
    for i, Ac in enumerate(Ac_grid):
        for j, ell0 in enumerate(ell0_grid):
            K3D_params = {
                'A_c': Ac,
                'ell0': ell0,
                **fixed_params
            }
            
            result = compute_cluster_lensing(cluster_name, K3D_params, zsrc)
            
            theta_E_grid[i, j] = result['theta_E_arcsec']
            boost_grid[i, j] = result['median_boost']
            
            pbar.update(1)
    
    pbar.close()
    
    # Find best fit (minimize |theta_E_pred - theta_E_obs|)
    residuals = np.abs(theta_E_grid - theta_E_obs)
    
    # Mask where theta_E = 0 (failed to find Einstein radius)
    residuals_masked = np.where(theta_E_grid > 0, residuals, np.inf)
    
    if np.all(np.isinf(residuals_masked)):
        print("❌ No valid Einstein radii found in grid!")
        best_idx = (0, 0)
        best_Ac = Ac_grid[0]
        best_ell0 = ell0_grid[0]
        best_theta_E = 0.0
        best_residual = np.inf
    else:
        best_idx = np.unravel_index(np.argmin(residuals_masked), residuals.shape)
        best_Ac = Ac_grid[best_idx[0]]
        best_ell0 = ell0_grid[best_idx[1]]
        best_theta_E = theta_E_grid[best_idx]
        best_residual = residuals[best_idx]
    
    print(f"\n✅ Best Fit:")
    print(f"  A_c = {best_Ac:.2f}")
    print(f"  ell0 = {best_ell0:.1f} kpc")
    print(f"  θ_E (predicted) = {best_theta_E:.2f} arcsec")
    print(f"  θ_E (observed) = {theta_E_obs:.2f} arcsec")
    print(f"  |Δθ_E| = {best_residual:.2f} arcsec ({100*best_residual/theta_E_obs:.1f}%)")
    
    return {
        'cluster': cluster_name,
        'Ac_grid': Ac_grid,
        'ell0_grid': ell0_grid,
        'theta_E_grid': theta_E_grid,
        'boost_grid': boost_grid,
        'theta_E_obs': theta_E_obs,
        'best_fit': {
            'A_c': float(best_Ac),
            'ell0': float(best_ell0),
            'theta_E_pred': float(best_theta_E),
            'residual': float(best_residual),
            'fractional_error': float(best_residual / theta_E_obs)
        }
    }


def plot_grid_results(results, output_dir='results/plots'):
    """Generate diagnostic plots for grid search results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cluster = results['cluster']
    Ac_grid = results['Ac_grid']
    ell0_grid = results['ell0_grid']
    theta_E_grid = results['theta_E_grid']
    boost_grid = results['boost_grid']
    theta_E_obs = results['theta_E_obs']
    best_fit = results['best_fit']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'B2 Grid Search: {cluster}', fontsize=14, fontweight='bold')
    
    # Plot 1: Predicted Einstein radius
    ax = axes[0]
    # Mask zeros for better visualization
    theta_E_plot = np.where(theta_E_grid > 0, theta_E_grid, np.nan)
    im = ax.contourf(ell0_grid, Ac_grid, theta_E_plot, levels=20, cmap='viridis')
    ax.contour(ell0_grid, Ac_grid, theta_E_plot, levels=[theta_E_obs], colors='red', 
               linewidths=2, linestyles='--')
    ax.plot(best_fit['ell0'], best_fit['A_c'], 'r*', markersize=20, label='Best fit')
    ax.set_xlabel('ell0 [kpc]')
    ax.set_ylabel('A_c')
    ax.set_title(f'Predicted θ_E [arcsec]\n(Observed: {theta_E_obs:.1f}")')
    plt.colorbar(im, ax=ax)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residuals |θ_E_pred - θ_E_obs|
    ax = axes[1]
    residuals = np.abs(theta_E_grid - theta_E_obs)
    residuals_plot = np.where(theta_E_grid > 0, residuals, np.nan)
    im = ax.contourf(ell0_grid, Ac_grid, residuals_plot, levels=20, cmap='RdYlGn_r')
    ax.plot(best_fit['ell0'], best_fit['A_c'], 'r*', markersize=20)
    ax.set_xlabel('ell0 [kpc]')
    ax.set_ylabel('A_c')
    ax.set_title(f'|Δθ_E| [arcsec]\n(Best: {best_fit["residual"]:.2f}")')
    plt.colorbar(im, ax=ax)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Median boost K_Sigma
    ax = axes[2]
    im = ax.contourf(ell0_grid, Ac_grid, boost_grid, levels=20, cmap='plasma')
    ax.plot(best_fit['ell0'], best_fit['A_c'], 'r*', markersize=20)
    ax.set_xlabel('ell0 [kpc]')
    ax.set_ylabel('A_c')
    ax.set_title('Median Boost K_Σ\n(50-300 kpc)')
    plt.colorbar(im, ax=ax)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / f'cluster_b2_grid_{cluster.lower()}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Grid plot saved: {plot_path}")
    
    plt.close(fig)


def main():
    """Main execution function."""
    args = parse_args()
    
    # Normalize cluster names
    clusters = [normalize_cluster_name(c) for c in args.clusters]
    
    # Set up parameter grids
    Ac_grid = np.linspace(args.Ac_min, args.Ac_max, args.Ac_n)
    ell0_grid = np.linspace(args.ell0_min, args.ell0_max, args.ell0_n)
    
    fixed_params = {
        'r_gate': args.rg,
        'n_gate': args.ng,
        'p': args.p,
        'L1': args.L1,
        'q': args.q
    }
    
    print("\n" + "="*70)
    print("TRACK B2: CLUSTER-FIRST PARAMETER GRID SEARCH")
    print("="*70)
    print(f"\nClusters: {', '.join(clusters)}")
    print(f"Fixed parameters: rg={args.rg}, ng={args.ng}, p={args.p}, L1={args.L1}, q={args.q}")
    
    # Run grid search for each cluster
    all_results = []
    
    for cluster in clusters:
        results = run_grid_search(cluster, Ac_grid, ell0_grid, fixed_params, args.zsrc)
        all_results.append(results)
        
        if args.plot:
            plot_grid_results(results)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for results in all_results:
        bf = results['best_fit']
        print(f"\n{results['cluster']}:")
        print(f"  Best: A_c={bf['A_c']:.2f}, ell0={bf['ell0']:.1f} kpc")
        print(f"  θ_E: {bf['theta_E_pred']:.2f}\" (obs: {results['theta_E_obs']:.2f}\")")
        print(f"  Error: {bf['residual']:.2f}\" ({100*bf['fractional_error']:.1f}%)")
    
    # Compute mean best-fit parameters
    mean_Ac = np.mean([r['best_fit']['A_c'] for r in all_results])
    mean_ell0 = np.mean([r['best_fit']['ell0'] for r in all_results])
    mean_error = np.mean([r['best_fit']['fractional_error'] for r in all_results])
    
    print(f"\nMean Best-Fit Parameters:")
    print(f"  A_c = {mean_Ac:.2f}")
    print(f"  ell0 = {mean_ell0:.1f} kpc")
    print(f"  Mean fractional error: {100*mean_error:.1f}%")
    
    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        'grid_params': {
            'Ac_range': [args.Ac_min, args.Ac_max, args.Ac_n],
            'ell0_range': [args.ell0_min, args.ell0_max, args.ell0_n],
            'fixed_params': fixed_params
        },
        'clusters': [],
        'summary': {
            'mean_Ac': float(mean_Ac),
            'mean_ell0': float(mean_ell0),
            'mean_fractional_error': float(mean_error)
        }
    }
    
    for results in all_results:
        output['clusters'].append({
            'name': results['cluster'],
            'theta_E_obs': results['theta_E_obs'],
            'best_fit': results['best_fit'],
            'Ac_grid': Ac_grid.tolist(),
            'ell0_grid': ell0_grid.tolist(),
            'theta_E_grid': results['theta_E_grid'].tolist()
        })
    
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Grid search results saved: {out_path}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
