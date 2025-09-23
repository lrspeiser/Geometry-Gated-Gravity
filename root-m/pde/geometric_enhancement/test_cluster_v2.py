#!/usr/bin/env python3
"""
test_cluster_v2.py

Test the improved enhanced PDE solver on clusters with proper scaling.
Compares geometric enhancement with pure Newtonian and observations.
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

from solve_phi_v2 import solve_axisym_sor, EnhancedSolverParams, compute_geometric_enhancement_v2
import baryon_maps as _maps

# Constants
G = 4.301e-6  # km^2/s^2 * kpc / Msun
mu = 0.61  # mean molecular weight
m_p_keV = 938272.0  # proton mass in keV/c^2
c_km_s = 299792.458
kpc_to_cm = 3.086e21


def predict_temperature_from_g(r: np.ndarray, ne: np.ndarray, g_tot: np.ndarray) -> np.ndarray:
    """
    Predict temperature from HSE given density and total acceleration.
    Simplified version for testing.
    """
    kT = np.zeros_like(r)
    
    # Integrate from outside in
    for i in range(len(r)-2, -1, -1):
        dr = r[i+1] - r[i]
        g_avg = 0.5 * (g_tot[i] + g_tot[i+1])
        ne_ratio = ne[i] / (ne[i+1] + 1e-30)
        
        # Temperature gradient from HSE
        dT = (mu * m_p_keV / c_km_s**2) * g_avg * dr * kpc_to_cm
        kT[i] = kT[i+1] * ne_ratio + dT
    
    return np.maximum(kT, 0.0)


def test_cluster_enhanced_v2(cluster_name='ABELL_0426'):
    """Test improved enhanced solver on a cluster."""
    
    print(f"\n{'='*60}")
    print(f"Testing cluster: {cluster_name}")
    print('='*60)
    
    # Setup paths
    base_path = Path("C:/Users/henry/dev/GravityCalculator/data/clusters")
    cluster_dir = base_path / cluster_name
    if not cluster_dir.exists():
        print(f"Cluster directory not found: {cluster_dir}")
        return None
    
    out_dir = Path("results") / cluster_name / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Grid parameters - use smaller grid for faster testing
    NR, NZ = 32, 32
    rmax = 300.0  # kpc - focus on inner region
    
    # Check for profiles
    clump_profile_csv = cluster_dir / "clump_profile.csv"
    if not clump_profile_csv.exists():
        clump_profile_csv = None
        clump = 3.0  # default uniform clumping
    else:
        clump = 1.0  # profile-based
    
    stellar_profile_csv = cluster_dir / "stars_profile.csv"
    if not stellar_profile_csv.exists():
        stellar_profile_csv = None
    
    # Get baryon density maps
    Z, R, rho_total_map, rho_gas_map = _maps.cluster_maps_from_csv(
        cluster_dir, R_max=rmax, Z_max=rmax, NR=NR, NZ=NZ,
        clump=clump, clump_profile_csv=clump_profile_csv,
        stars_csv=stellar_profile_csv
    )
    
    # Use total baryon density
    rho = rho_total_map
    
    # Apply edge taper
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
    
    print(f"\nGrid Setup:")
    print(f"  Resolution: {NR}x{NZ}")
    print(f"  Domain: R=[0,{rmax}], Z=[{-rmax},{rmax}] kpc")
    print(f"  Max density: {np.max(rho):.2e} Msun/kpc^3")
    print(f"  Total mass: {np.sum(rho * 2*np.pi*R.reshape(1,-1) * (R[1]-R[0]) * (Z[1]-Z[0])):.2e} Msun")
    
    # Build Newtonian baseline first
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
    
    # Cumulative mass and Newtonian g
    integrand = 4.0*np.pi * (r_obs**2) * rho_total_1d
    M = np.concatenate(([0.0], np.cumsum(0.5*(integrand[1:]+integrand[:-1]) * np.diff(r_obs))))
    M = M[:len(r_obs)]
    M_R = np.interp(R, r_obs, M, left=M[0], right=M[-1])
    g_N_R = G * M_R / np.maximum(R**2, 1e-9)
    
    print(f"\nNewtonian Baseline:")
    print(f"  M(<100 kpc) = {np.interp(100, R, M_R):.2e} Msun")
    print(f"  g_N(100 kpc) = {np.interp(100, R, g_N_R):.2e} km/s^2/kpc")
    
    # Test multiple parameter sets with calibrated values
    test_configs = [
        {
            "name": "weak",
            "gamma": 0.1,  # Very weak coupling
            "lambda0": 0.3,
            "alpha_grad": 1.0
        },
        {
            "name": "moderate", 
            "gamma": 0.3,  # Moderate coupling
            "lambda0": 0.5,
            "alpha_grad": 1.5
        },
        {
            "name": "strong",
            "gamma": 0.5,  # Stronger coupling
            "lambda0": 0.7,
            "alpha_grad": 2.0
        },
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n[Testing: {config['name']}]")
        
        # Setup parameters with calibrated values
        params = EnhancedSolverParams(
            gamma=config.get("gamma", 0.3),
            beta=1.0,
            use_geometric_enhancement=True,
            lambda0=config.get("lambda0", 0.5),
            alpha_grad=config.get("alpha_grad", 1.5),
            rho_crit_factor=0.01,  # Enhancement below 1% of max density
            r_enhance_kpc=50.0,
            max_iter=2000,
            rtol=1e-5,
            omega_init=1.5,
            adaptive_omega=True,
            verbose=False,
            smooth_source=True
        )
        
        # Show enhancement statistics
        Lambda = compute_geometric_enhancement_v2(R, Z, rho, params)
        print(f"  Enhancement: min={np.min(Lambda):.2f}, max={np.max(Lambda):.2f}, mean={np.mean(Lambda):.2f}")
        
        # Solve enhanced PDE
        try:
            phi, gR, gZ = solve_axisym_sor(R, Z, rho, params)
        except Exception as e:
            print(f"  Solver failed: {e}")
            continue
        
        # Project to spherical shells
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
        
        # Scale the scalar field contribution
        # This is the key: we need g_phi to be comparable to g_N
        g_phi_scale = np.median(g_N_R[R > 10]) / (np.median(g_phi_R[R > 10]) + 1e-12)
        g_phi_R = g_phi_R * g_phi_scale * 0.1  # Start with 10% contribution
        
        # Total acceleration
        g_tot_R = g_N_R + g_phi_R
        
        # Temperature prediction
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
        kT_pred = predict_temperature_from_g(R, ne_R, g_tot_R)
        kT_pred_GR = predict_temperature_from_g(R, ne_R, g_N_R)
        
        # Interpolate to observation radii
        kT_pred_obs = np.interp(r_T, R, kT_pred)
        kT_pred_GR_obs = np.interp(r_T, R, kT_pred_GR)
        
        # Compute residuals
        res_enhanced = np.abs(kT_pred_obs - kT_obs) / kT_obs
        res_GR = np.abs(kT_pred_GR_obs - kT_obs) / kT_obs
        
        median_enhanced = np.median(res_enhanced)
        median_GR = np.median(res_GR)
        
        print(f"  Results:")
        print(f"    max(g_phi/g_N) = {np.max(g_phi_R/np.maximum(g_N_R, 1e-12)):.2f}")
        print(f"    Median |DT|/T (Enhanced): {median_enhanced:.3f}")
        print(f"    Median |DT|/T (GR only): {median_GR:.3f}")
        print(f"    Improvement: {(1 - median_enhanced/median_GR)*100:.1f}%")
        
        # Store results
        results[config['name']] = {
            'g_N_R': g_N_R,
            'g_phi_R': g_phi_R,
            'g_tot_R': g_tot_R,
            'kT_pred': kT_pred,
            'kT_pred_obs': kT_pred_obs,
            'median_residual': median_enhanced,
            'Lambda': Lambda,
            'params': config
        }
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: Density map
    ax = axes[0, 0]
    im = ax.pcolormesh(R, Z, np.log10(np.maximum(rho, 1e-10)), shading='auto', cmap='viridis')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Z [kpc]')
    ax.set_title(f'{cluster_name}: log₁₀(ρ_total) [Msun/kpc³]')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    
    # Panel 2: Enhancement map (for moderate case)
    if 'moderate' in results:
        ax = axes[0, 1]
        Lambda = results['moderate']['Lambda']
        im = ax.pcolormesh(R, Z, Lambda, shading='auto', cmap='RdYlBu_r', vmin=1, vmax=3)
        ax.set_xlabel('R [kpc]')
        ax.set_ylabel('Z [kpc]')
        ax.set_title('Enhancement Factor Λ')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Λ')
    
    # Panel 3: Acceleration profiles
    ax = axes[0, 2]
    ax.loglog(R[R>0], g_N_R[R>0], 'k-', label='g_N (Newtonian)', lw=2)
    for name, res in results.items():
        ax.loglog(R[R>0], res['g_phi_R'][R>0], '--', label=f'g_φ ({name})', alpha=0.7)
        ax.loglog(R[R>0], res['g_tot_R'][R>0], '-', label=f'g_tot ({name})', alpha=0.5)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('|g| [km/s²/kpc]')
    ax.legend(fontsize=8)
    ax.set_title('Acceleration components')
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Temperature predictions
    ax = axes[1, 0]
    ax.semilogy(r_T, kT_obs, 'ko', label='Observed', markersize=8)
    ax.semilogy(R, kT_pred_GR, 'k--', label='GR only', alpha=0.5, lw=2)
    for name, res in results.items():
        ax.semilogy(R, res['kT_pred'], '-', label=f'{name}', alpha=0.7)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('kT [keV]')
    ax.legend(fontsize=8)
    ax.set_title('Temperature profiles')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(300, max(r_T))])
    
    # Panel 5: Temperature residuals
    ax = axes[1, 1]
    ax.semilogy(r_T, res_GR, 'k--', label='GR only', lw=2)
    for name, res in results.items():
        res_vals = np.abs(res['kT_pred_obs'] - kT_obs) / kT_obs
        ax.semilogy(r_T, res_vals, '-', label=f'{name}', alpha=0.7)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('|ΔT|/T')
    ax.legend(fontsize=8)
    ax.set_title('Temperature residuals')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.1, color='g', linestyle=':', alpha=0.5, label='10% target')
    
    # Panel 6: Summary metrics
    ax = axes[1, 2]
    config_names = ['GR'] + list(results.keys())
    residuals = [median_GR] + [results[name]['median_residual'] for name in results.keys()]
    colors = ['black'] + ['blue', 'green', 'red'][:len(results)]
    bars = ax.bar(config_names, residuals, color=colors, alpha=0.7)
    ax.set_ylabel('Median |ΔT|/T')
    ax.set_title('Performance Summary')
    ax.axhline(y=0.1, color='g', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(residuals) * 1.2])
    
    plt.suptitle(f'{cluster_name} - Enhanced Gravity Test (v2)', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'enhanced_test_v2.png', dpi=150)
    plt.show()
    
    # Save metrics
    metrics = {
        'cluster': cluster_name,
        'median_GR': float(median_GR),
        'configs': {name: {
            'params': res['params'],
            'median_residual': float(res['median_residual']),
            'improvement_pct': float((1 - res['median_residual']/median_GR)*100)
        } for name, res in results.items()}
    }
    
    with open(out_dir / 'metrics_v2.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {out_dir}")
    print('='*60)
    
    return metrics


if __name__ == "__main__":
    # Test on available clusters
    clusters = ['ABELL_0426', 'A1795', 'A2029', 'A478']
    
    all_metrics = {}
    for cluster in clusters:
        cluster_dir = Path("C:/Users/henry/dev/GravityCalculator/data/clusters") / cluster
        if cluster_dir.exists():
            metrics = test_cluster_enhanced_v2(cluster)
            if metrics:
                all_metrics[cluster] = metrics
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL CLUSTERS")
    print("="*60)
    
    for cluster, metrics in all_metrics.items():
        print(f"\n{cluster}:")
        print(f"  GR baseline: {metrics['median_GR']:.3f}")
        for config_name, config_data in metrics['configs'].items():
            print(f"  {config_name}: {config_data['median_residual']:.3f} (improvement: {config_data['improvement_pct']:.1f}%)")