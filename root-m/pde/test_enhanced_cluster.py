#!/usr/bin/env python3
"""
Test the enhanced PDE solver on clusters with proper normalization.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
sys.path.append(str(Path(__file__).parent.parent.parent))

from solve_phi_enhanced import solve_axisym_enhanced, EnhancedSolverParams
import baryon_maps as _maps
from predict_hse import kT_from_ne_and_gtot

# Constants
G = 4.301e-6  # km^2/s^2 * kpc / Msun

def test_cluster_enhanced(cluster_name='ABELL_0426'):
    """Test enhanced solver on a cluster."""
    
    # Setup paths
    base_dir = Path("../../data/clusters") / cluster_name
    out_dir = Path("../geometric_enhancement/results") / cluster_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Grid parameters
    NR, NZ = 64, 64
    Rmax, Zmax = 600.0, 600.0
    R = np.linspace(0, Rmax, NR)
    Z = np.linspace(-Zmax, Zmax, NZ)
    
    # Build 3D baryon map (total baryons)
    rho_gas_map = _maps.build_axisym_gas_map(
        R, Z, 
        cluster=cluster_name,
        clump_profile_csv=base_dir / "clump_profile.csv"
    )
    
    rho_star_map = _maps.build_axisym_star_map(
        R, Z,
        cluster=cluster_name,
        stars_csv=base_dir / "stars_profile.csv"
    )
    
    # Total baryon density
    rho = rho_gas_map + rho_star_map
    
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
    
    print(f"\n[Test] {cluster_name}")
    print(f"  Grid: {NR}x{NZ}, Rmax={Rmax}, Zmax={Zmax}")
    print(f"  Max density: {np.max(rho):.2e} Msun/kpc^3")
    print(f"  Total mass: {np.sum(rho * 2*np.pi*R.reshape(1,-1) * (R[1]-R[0]) * (Z[1]-Z[0])):.2e} Msun")
    
    # Solve with enhanced PDE
    params = EnhancedSolverParams(
        S0_base=1.4e-4,
        rc_kpc=22.0,
        g0_kms2_per_kpc=1200.0,
        use_geometric_enhancement=True,
        lambda0=0.5,
        alpha_grad=1.5,
        use_saturating_mobility=True,
        gsat_kms2_per_kpc=2500.0,
        n_sat=2.0
    )
    
    print("\n[Solving] Enhanced PDE with geometric enhancement...")
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
    gas_df = pd.read_csv(base_dir / "gas_profile.csv")
    r_obs = gas_df['r_kpc'].values
    
    if 'n_e_cm3' in gas_df.columns:
        n_e = gas_df['n_e_cm3'].values
        rho_gas_1d = _maps.ne_to_rho_gas_Msun_kpc3(n_e)
    else:
        rho_gas_1d = gas_df['rho_gas_Msun_per_kpc3'].values
    
    # Apply clumping
    try:
        clump_df = pd.read_csv(base_dir / "clump_profile.csv")
        C = np.interp(r_obs, clump_df['r_kpc'].values, clump_df['C'].values)
        rho_gas_1d *= np.sqrt(C)
    except:
        pass
    
    # Add stars
    try:
        stars_df = pd.read_csv(base_dir / "stars_profile.csv")
        rho_star_1d = np.interp(r_obs, stars_df['r_kpc'].values, 
                                stars_df['rho_star_Msun_per_kpc3'].values)
        rho_total_1d = rho_gas_1d + rho_star_1d
    except:
        rho_total_1d = rho_gas_1d
    
    # Cumulative mass and g_N
    integrand = 4.0*np.pi * (r_obs**2) * rho_total_1d
    M = np.concatenate(([0.0], np.cumsum(0.5*(integrand[1:]+integrand[:-1]) * np.diff(r_obs))))
    M = M[:len(r_obs)]
    M_R = np.interp(R, r_obs, M, left=M[0], right=M[-1])
    g_N_R = G * M_R / np.maximum(R**2, 1e-9)
    
    # Total acceleration
    g_tot_R = g_N_R + g_phi_R
    
    # Temperature prediction via HSE
    temp_df = pd.read_csv(base_dir / "temp_profile.csv")
    r_T = temp_df['r_kpc'].values
    kT_obs = temp_df['kT_keV'].values
    
    # Interpolate n_e to R grid
    if 'n_e_cm3' in gas_df.columns:
        ne_R = np.interp(R, r_obs, gas_df['n_e_cm3'].values)
    else:
        MU_E = 1.17
        M_P_G = 1.67262192369e-24
        KPC_CM = 3.0856775814913673e21
        MSUN_G = 1.988409870698051e33
        ne_R = (rho_gas_1d * MSUN_G / (KPC_CM**3)) / (MU_E * M_P_G)
        ne_R = np.interp(R, r_obs, ne_R)
    
    # Predict temperatures
    kT_pred = kT_from_ne_and_gtot(R, ne_R, g_tot_R, f_nt=None)
    kT_pred_GR = kT_from_ne_and_gtot(R, ne_R, g_N_R, f_nt=None)
    
    # Interpolate to observation radii
    kT_pred_obs = np.interp(r_T, R, kT_pred)
    kT_pred_GR_obs = np.interp(r_T, R, kT_pred_GR)
    
    # Compute residuals
    res_enhanced = np.abs(kT_pred_obs - kT_obs) / kT_obs
    res_GR = np.abs(kT_pred_GR_obs - kT_obs) / kT_obs
    
    median_enhanced = np.median(res_enhanced)
    median_GR = np.median(res_GR)
    
    print(f"\n[Results]")
    print(f"  Median |ΔT|/T (Enhanced): {median_enhanced:.3f}")
    print(f"  Median |ΔT|/T (GR only): {median_GR:.3f}")
    print(f"  Improvement factor: {median_GR/median_enhanced:.2f}x")
    print(f"  Max g_phi/g_N: {np.max(g_phi_R/np.maximum(g_N_R, 1e-12)):.2f}")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Density map with enhancement
    ax = axes[0, 0]
    im = ax.pcolormesh(R, Z, np.log10(np.maximum(rho, 1e-10)), shading='auto', cmap='viridis')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Z [kpc]')
    ax.set_title('log₁₀(ρ_total) [Msun/kpc³]')
    plt.colorbar(im, ax=ax)
    
    # Panel 2: Accelerations
    ax = axes[0, 1]
    ax.loglog(R[R>0], g_N_R[R>0], 'g-', label='g_N (Newtonian)')
    ax.loglog(R[R>0], g_phi_R[R>0], 'b-', label='g_φ (enhanced)')
    ax.loglog(R[R>0], g_tot_R[R>0], 'r-', linewidth=2, label='g_total')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('g [(km/s)²/kpc]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Temperature comparison
    ax = axes[1, 0]
    ax.plot(r_T, kT_obs, 'ko', markersize=8, label='Observed')
    ax.plot(R, kT_pred, 'r-', linewidth=2, label=f'Enhanced (med={median_enhanced:.3f})')
    ax.plot(R, kT_pred_GR, 'g--', label=f'GR only (med={median_GR:.3f})')
    ax.set_xscale('log')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('kT [keV]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Residuals
    ax = axes[1, 1]
    ax.semilogx(r_T, res_enhanced, 'r-', linewidth=2, label='Enhanced')
    ax.semilogx(r_T, res_GR, 'g--', label='GR only')
    ax.axhline(0.3, color='k', linestyle=':', alpha=0.5)
    ax.axhline(0.6, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('|ΔT|/T')
    ax.set_ylim(0, 2.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{cluster_name}: Enhanced PDE Test')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(out_dir / f"enhanced_test_{NR}x{NZ}.png", dpi=150)
    plt.close()
    
    # Save metrics
    metrics = {
        'cluster': cluster_name,
        'NR': NR,
        'NZ': NZ,
        'median_residual_enhanced': float(median_enhanced),
        'median_residual_GR': float(median_GR),
        'improvement_factor': float(median_GR / median_enhanced) if median_enhanced > 0 else np.inf,
        'max_g_phi_over_g_N': float(np.max(g_phi_R/np.maximum(g_N_R, 1e-12)))
    }
    
    with open(out_dir / f"metrics_enhanced_{NR}x{NZ}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    # Test on Perseus
    print("Testing enhanced solver on ABELL_0426...")
    metrics_perseus = test_cluster_enhanced('ABELL_0426')
    
    # Test on A1689
    print("\nTesting enhanced solver on ABELL_1689...")
    metrics_a1689 = test_cluster_enhanced('ABELL_1689')
    
    print("\n=== SUMMARY ===")
    print(f"ABELL_0426: Enhanced={metrics_perseus['median_residual_enhanced']:.3f}, "
          f"GR={metrics_perseus['median_residual_GR']:.3f}, "
          f"Improvement={metrics_perseus['improvement_factor']:.2f}x")
    print(f"ABELL_1689: Enhanced={metrics_a1689['median_residual_enhanced']:.3f}, "
          f"GR={metrics_a1689['median_residual_GR']:.3f}, "
          f"Improvement={metrics_a1689['improvement_factor']:.2f}x")