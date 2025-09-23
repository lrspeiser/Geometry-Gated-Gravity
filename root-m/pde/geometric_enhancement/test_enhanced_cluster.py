#!/usr/bin/env python3
"""
Test the enhanced PDE solver on clusters with geometric enhancement.
Compares with baseline total-baryon results.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from solve_phi_enhanced import solve_axisym_enhanced, EnhancedSolverParams, predict_temperature_hse
import baryon_maps as _maps

# Constants
G = 4.301e-6  # km^2/s^2 * kpc / Msun

def test_cluster_enhanced(cluster_name='ABELL_0426'):
    """Test enhanced solver on a cluster."""
    
    print(f"\nTesting enhanced solver on {cluster_name}...")
    
    # Setup paths
    base_path = Path("C:/Users/henry/dev/GravityCalculator/data/clusters")
    cluster_dir = base_path / cluster_name
    if not cluster_dir.exists():
        print(f"Cluster directory not found: {cluster_dir}")
        return None
    
    out_dir = Path("results") / cluster_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Grid parameters
    NR, NZ = 64, 64
    rmax = 600.0  # kpc
    
    # Check for clumping and stellar profiles
    clump_profile_csv = cluster_dir / "clump_profile.csv"
    if not clump_profile_csv.exists():
        clump_profile_csv = None
        clump = 3.0  # default uniform clumping
    else:
        clump = 1.0  # profile-based
    
    stellar_profile_csv = cluster_dir / "stars_profile.csv"
    if not stellar_profile_csv.exists():
        stellar_profile_csv = None
    
    # Get baryon density maps using cluster_maps_from_csv
    Z, R, rho_total_map, rho_gas_map = _maps.cluster_maps_from_csv(
        cluster_dir, R_max=rmax, Z_max=rmax, NR=NR, NZ=NZ,
        clump=clump, clump_profile_csv=clump_profile_csv,
        stars_csv=stellar_profile_csv
    )
    
    # Use total baryon density for the PDE
    rho = rho_total_map
    
    # Apply edge taper to avoid boundary issues
    def taper_1d(x, frac=0.1):
        xabs = np.abs(x)
        xmax = np.max(xabs)
        x0 = (1.0 - frac) * xmax
        w = np.ones_like(xabs)
        mask = xabs > x0
        if np.any(mask):
            xi = (xabs[mask] - x0) / (xmax - x0 + 1e-12)
            w[mask] = 0.5 * (1.0 + np.cos(np.pi * xi))
        return w
    
    wZ = taper_1d(Z, frac=0.1).reshape(-1,1)
    wR = taper_1d(R, frac=0.1).reshape(1,-1)
    rho = rho * (wZ * wR)
    
    print(f"  Grid: {NR}x{NZ}, Rmax={rmax} kpc")
    print(f"  Max density: {np.max(rho):.2e} Msun/kpc^3")
    print(f"  Total mass: {np.sum(rho * 2*np.pi*R.reshape(1,-1) * (R[1]-R[0]) * (Z[1]-Z[0])):.2e} Msun")
    
    # Test multiple parameter sets
    test_configs = [
        # Baseline (no enhancement)
        {"name": "baseline", "lambda0": 0.0, "use_geometric_enhancement": False},
        # Weak enhancement
        {"name": "weak", "lambda0": 0.3, "alpha_grad": 1.0},
        # Moderate enhancement
        {"name": "moderate", "lambda0": 0.5, "alpha_grad": 1.5},
        # Strong enhancement  
        {"name": "strong", "lambda0": 0.8, "alpha_grad": 2.0},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n[Testing config: {config['name']}]")
        
        # Setup parameters
        params = EnhancedSolverParams(
            S0_base=1.4e-4,
            rc_kpc=22.0,
            g0_kms2_per_kpc=1200.0,
            use_geometric_enhancement=config.get("use_geometric_enhancement", True),
            lambda0=config.get("lambda0", 0.5),
            alpha_grad=config.get("alpha_grad", 1.5),
            rho_crit_Msun_kpc3=1e6,
            r_enhance_kpc=50.0,
            use_saturating_mobility=False,
            max_iter=5000,  # Increase max iterations
            rtol=1e-4,  # Relax tolerance for faster testing
            verbose=True
        )
        
        # Solve enhanced PDE
        phi, gR, gZ = solve_axisym_enhanced(R, Z, rho, params)
        
        # Project to spherical shells for temperature prediction
        NZ_grid, NR_grid = gR.shape
        R2D = np.broadcast_to(R.reshape(1,-1), (NZ_grid, NR_grid))
        Z2D = np.broadcast_to(Z.reshape(-1,1), (NZ_grid, NR_grid))
        r_cell = np.sqrt(R2D*R2D + Z2D*Z2D)
        
        # Radial components
        r_hat_R = np.divide(R2D, np.maximum(r_cell, 1e-9))
        r_hat_Z = np.divide(Z2D, np.maximum(r_cell, 1e-9))
        g_r_field = gR * r_hat_R + gZ * r_hat_Z
        
        # Average on spherical shells
        g_phi_R = np.zeros_like(R)
        dr = 0.5 * max(R[1]-R[0], Z[1]-Z[0])
        for i, rsh in enumerate(R):
            if rsh <= 0:
                mask = (r_cell <= max(dr, 1e-6))
            else:
                mask = (np.abs(r_cell - rsh) <= dr)
            if np.any(mask):
                g_phi_R[i] = float(np.mean(np.abs(g_r_field[mask])))
            else:
                z0_idx = len(Z)//2
                g_phi_R[i] = float(np.abs(gR[z0_idx, i]))
        
        # Build Newtonian comparator
        gas_df = pd.read_csv(cluster_dir / "gas_profile.csv")
        r_obs = gas_df['r_kpc'].values
        
        if 'n_e_cm3' in gas_df.columns:
            n_e = gas_df['n_e_cm3'].values
            rho_gas_1d = _maps.ne_to_rho_gas_Msun_kpc3(n_e)
        else:
            rho_gas_1d = gas_df['rho_gas_Msun_per_kpc3'].values
        
        # Apply clumping
        if clump_profile_csv is not None:
            try:
                clump_df = pd.read_csv(clump_profile_csv)
                C = np.interp(r_obs, clump_df['r_kpc'].values, clump_df['C'].values)
                rho_gas_1d *= np.sqrt(C)
            except:
                pass
        else:
            rho_gas_1d *= np.sqrt(clump)
        
        # Add stars
        rho_total_1d = rho_gas_1d.copy()
        if stellar_profile_csv is not None:
            try:
                stars_df = pd.read_csv(stellar_profile_csv)
                rho_star_1d = np.interp(r_obs, stars_df['r_kpc'].values, 
                                        stars_df['rho_star_Msun_per_kpc3'].values,
                                        left=0, right=0)
                rho_total_1d = rho_gas_1d + rho_star_1d
            except:
                pass
        
        # Cumulative mass and g_N
        integrand = 4.0*np.pi * (r_obs**2) * rho_total_1d
        M = np.concatenate(([0.0], np.cumsum(0.5*(integrand[1:]+integrand[:-1]) * np.diff(r_obs))))
        M = M[:len(r_obs)]
        M_R = np.interp(R, r_obs, M, left=M[0], right=M[-1])
        g_N_R = G * M_R / np.maximum(R**2, 1e-9)
        
        # Total acceleration
        g_tot_R = g_N_R + g_phi_R
        
        # Temperature prediction via HSE
        temp_df = pd.read_csv(cluster_dir / "temp_profile.csv")
        r_T = temp_df['r_kpc'].values
        kT_obs = temp_df['kT_keV'].values
        
        # Interpolate n_e to R grid
        if 'n_e_cm3' in gas_df.columns:
            ne_R = np.interp(R, r_obs, gas_df['n_e_cm3'].values)
        else:
            # Convert from density back to n_e
            MU_E = 1.17
            M_P_G = 1.67262192369e-24
            KPC_CM = 3.0856775814913673e21
            MSUN_G = 1.988409870698051e33
            ne_cm3 = (rho_gas_1d * MSUN_G / (KPC_CM**3)) / (MU_E * M_P_G)
            ne_R = np.interp(R, r_obs, ne_cm3)
        
        # Predict temperatures
        kT_pred = predict_temperature_hse(R, ne_R, g_tot_R)
        kT_pred_GR = predict_temperature_hse(R, ne_R, g_N_R)
        
        # Interpolate to observation radii
        kT_pred_obs = np.interp(r_T, R, kT_pred)
        kT_pred_GR_obs = np.interp(r_T, R, kT_pred_GR)
        
        # Compute residuals
        res_enhanced = np.abs(kT_pred_obs - kT_obs) / kT_obs
        res_GR = np.abs(kT_pred_GR_obs - kT_obs) / kT_obs
        
        median_enhanced = np.median(res_enhanced)
        median_GR = np.median(res_GR)
        
        print(f"  Median |DT|/T (Enhanced): {median_enhanced:.3f}")
        print(f"  Median |DT|/T (GR only): {median_GR:.3f}")
        print(f"  Max g_phi/g_N: {np.max(g_phi_R/np.maximum(g_N_R, 1e-12)):.2f}")
        
        # Store results
        results[config['name']] = {
            'g_N_R': g_N_R,
            'g_phi_R': g_phi_R,
            'g_tot_R': g_tot_R,
            'kT_pred': kT_pred,
            'kT_pred_obs': kT_pred_obs,
            'median_residual': median_enhanced,
            'params': config
        }
    
    # Plotting comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Density map
    ax = axes[0, 0]
    im = ax.pcolormesh(R, Z, np.log10(np.maximum(rho, 1e-10)), shading='auto', cmap='viridis')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Z [kpc]')
    ax.set_title(f'{cluster_name}: log₁₀(ρ_total) [Msun/kpc³]')
    plt.colorbar(im, ax=ax)
    
    # Panel 2: Accelerations comparison
    ax = axes[0, 1]
    ax.loglog(R[R>0], results['baseline']['g_N_R'][R>0], 'k-', label='g_N (Newtonian)', lw=2)
    for name, res in results.items():
        if name != 'baseline':
            ax.loglog(R[R>0], res['g_phi_R'][R>0], '-', label=f'g_φ ({name})', alpha=0.7)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('|g| [km/s²/kpc]')
    ax.legend()
    ax.set_title('Acceleration components')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Temperature predictions
    ax = axes[1, 0]
    ax.semilogy(r_T, kT_obs, 'ko', label='Observed', markersize=8)
    ax.semilogy(R, results['baseline']['kT_pred'], 'k--', label='GR only', alpha=0.5)
    for name, res in results.items():
        if name != 'baseline':
            ax.semilogy(R, res['kT_pred'], '-', label=f'Enhanced ({name})', alpha=0.7)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('kT [keV]')
    ax.legend()
    ax.set_title('Temperature profiles')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Residuals summary
    ax = axes[1, 1]
    config_names = list(results.keys())
    residuals = [results[name]['median_residual'] for name in config_names]
    colors = ['gray' if name == 'baseline' else 'blue' for name in config_names]
    bars = ax.bar(config_names, residuals, color=colors, alpha=0.7)
    ax.set_ylabel('Median |DT|/T')
    ax.set_title('Temperature residuals')
    ax.axhline(y=median_GR, color='k', linestyle='--', alpha=0.5, label='GR baseline')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / 'enhanced_comparison.png', dpi=150)
    plt.show()
    
    # Save results
    metrics = {
        'cluster': cluster_name,
        'configs': {name: {
            'params': res['params'],
            'median_residual': float(res['median_residual']),
            'max_g_phi_over_gN': float(np.max(res['g_phi_R']/np.maximum(res['g_N_R'], 1e-12)))
        } for name, res in results.items()}
    }
    
    with open(out_dir / 'enhanced_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {out_dir}")
    return metrics


if __name__ == "__main__":
    # Test on Perseus cluster
    metrics_perseus = test_cluster_enhanced('ABELL_0426')
    
    # Test on additional clusters if available
    other_clusters = ['A1795', 'A2029', 'A478']
    base_path = Path("C:/Users/henry/dev/GravityCalculator/data/clusters")
    for cluster in other_clusters:
        cluster_dir = base_path / cluster
        if cluster_dir.exists():
            test_cluster_enhanced(cluster)
