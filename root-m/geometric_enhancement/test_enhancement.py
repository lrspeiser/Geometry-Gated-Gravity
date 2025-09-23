#!/usr/bin/env python3
"""
Test geometric enhancement on cluster profiles.
The enhancement factor responds to density gradients, naturally amplifying
gravity where baryons thin out.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Constants
G = 4.301e-6  # km^2/s^2 * kpc / Msun
KB_KEV_TO_KM2_S2 = 1.16045e7  # keV to (km/s)^2
MU_MP_KEV = 0.6  # mean molecular weight * proton mass in keV/(km/s)^2

def compute_density_gradient_enhancement(r, rho, lambda0=0.5, alpha=1.5, beta=1.0, 
                                         r_half=None, r_ref=30.0, gamma=0.5):
    """
    Compute geometric enhancement factor based on density gradient.
    
    κ(r) = 1 + λ_eff * |∇log(ρ)|^α / (1 + |∇log(ρ)|^α/β)
    
    where λ_eff = λ_0 * (r_half/r_ref)^γ
    """
    # Compute log density gradient
    log_rho = np.log(np.maximum(rho, 1e-30))
    grad_log_rho = np.gradient(log_rho, r)
    abs_grad = np.abs(grad_log_rho)
    
    # Scale enhancement with system size if r_half provided
    if r_half is not None:
        lambda_eff = lambda0 * (r_half / r_ref)**gamma
    else:
        lambda_eff = lambda0
    
    # Compute enhancement factor
    grad_term = abs_grad**alpha
    kappa = 1.0 + lambda_eff * grad_term / (1.0 + grad_term/beta)
    
    return kappa

def test_cluster_with_enhancement(cluster_name, lambda0=0.5, alpha=1.5):
    """Test geometric enhancement on a cluster."""
    
    base_dir = Path("data/clusters") / cluster_name
    
    # Load profiles
    gas_df = pd.read_csv(base_dir / "gas_profile.csv")
    temp_df = pd.read_csv(base_dir / "temp_profile.csv")
    
    r = gas_df['r_kpc'].values
    
    # Convert gas density
    if 'n_e_cm3' in gas_df.columns:
        # Convert n_e to mass density
        n_e = gas_df['n_e_cm3'].values
        mu_e = 1.17  # mean molecular weight per electron
        m_p_msun_kpc3 = 8.4e-58  # proton mass in Msun/kpc^3
        rho_gas = n_e * mu_e * m_p_msun_kpc3 * 1e9  # convert cm^-3 to kpc^-3
    else:
        rho_gas = gas_df['rho_gas_Msun_per_kpc3'].values
    
    # Apply clumping if available
    try:
        clump_df = pd.read_csv(base_dir / "clump_profile.csv")
        C = np.interp(r, clump_df['r_kpc'].values, clump_df['C'].values)
        rho_gas *= np.sqrt(C)
    except:
        pass
    
    # Add stars if available
    try:
        stars_df = pd.read_csv(base_dir / "stars_profile.csv")
        rho_star = np.interp(r, stars_df['r_kpc'].values, 
                            stars_df['rho_star_Msun_per_kpc3'].values)
        rho_total = rho_gas + rho_star
    except:
        rho_total = rho_gas
    
    # Compute half-mass radius
    M_cumul = np.cumsum(4*np.pi * r**2 * rho_total * np.gradient(r))
    r_half = np.interp(0.5*M_cumul[-1], M_cumul, r)
    
    # Compute enhancement factor
    kappa = compute_density_gradient_enhancement(r, rho_total, 
                                                 lambda0=lambda0, 
                                                 alpha=alpha,
                                                 r_half=r_half)
    
    # Compute gravitational accelerations
    M_r = M_cumul
    g_N = G * M_r / r**2
    
    # Simple field approximation: g_phi ~ S0 * rho * r_c^2 / r^2
    # This is a placeholder - actual PDE solution would be more complex
    S0 = 1e-4
    rc = 22.0
    g_phi_base = S0 * G * M_r / r**2 * (rc/r)**2
    
    # Apply enhancement
    g_phi = kappa * g_phi_base
    g_tot = g_N + g_phi
    
    # Predict temperature via HSE
    kT_obs = temp_df['kT_keV'].values
    r_T = temp_df['r_kpc'].values
    
    # Interpolate densities and accelerations to temperature radii
    if 'n_e_cm3' in gas_df.columns:
        n_e_T = np.interp(r_T, r, gas_df['n_e_cm3'].values)
    else:
        # Approximate n_e from gas density
        n_e_T = rho_gas / (1.17 * 8.4e-58 * 1e9)
        n_e_T = np.interp(r_T, r, n_e_T)
    
    g_tot_T = np.interp(r_T, r, g_tot)
    g_N_T = np.interp(r_T, r, g_N)
    kappa_T = np.interp(r_T, r, kappa)
    
    # HSE: dP/dr = -rho*g => kT = integral(-g * mu*m_p * n_e * dr) / n_e
    # Simplified: kT ~ mu*m_p * g * r / (dlnP/dlnr)
    # For isothermal approximation: kT ~ mu*m_p * g * r
    
    # Better approach: integrate from outside in
    kT_pred = np.zeros_like(r_T)
    for i in range(len(r_T)-1, -1, -1):
        if i == len(r_T)-1:
            # Boundary condition: match observed at outermost point
            kT_pred[i] = kT_obs[i]
        else:
            # Integrate inward
            dr = r_T[i+1] - r_T[i]
            dlnP = -MU_MP_KEV * g_tot_T[i] * dr / kT_pred[i+1]
            kT_pred[i] = kT_pred[i+1] * np.exp(dlnP)
    
    # Similar for GR-only
    kT_GR = np.zeros_like(r_T)
    for i in range(len(r_T)-1, -1, -1):
        if i == len(r_T)-1:
            kT_GR[i] = kT_obs[i]
        else:
            dr = r_T[i+1] - r_T[i]
            dlnP = -MU_MP_KEV * g_N_T[i] * dr / kT_GR[i+1]
            kT_GR[i] = kT_GR[i+1] * np.exp(dlnP)
    
    # Calculate residuals
    residual_enhanced = np.abs(kT_pred - kT_obs) / kT_obs
    residual_GR = np.abs(kT_GR - kT_obs) / kT_obs
    
    median_enhanced = np.median(residual_enhanced)
    median_GR = np.median(residual_GR)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Density and enhancement
    ax = axes[0, 0]
    ax.loglog(r, rho_total, 'b-', label='ρ_total')
    ax2 = ax.twinx()
    ax2.semilogx(r, kappa, 'r-', label='κ enhancement')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('ρ [Msun/kpc³]', color='b')
    ax2.set_ylabel('κ factor', color='r')
    ax.axvline(r_half, color='k', linestyle='--', alpha=0.3, label=f'r_half={r_half:.1f} kpc')
    ax.legend(loc='upper right')
    ax2.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Accelerations
    ax = axes[0, 1]
    ax.loglog(r, g_N, 'g-', label='g_N (Newtonian)')
    ax.loglog(r, g_phi, 'b-', label='g_φ (enhanced field)')
    ax.loglog(r, g_tot, 'r-', linewidth=2, label='g_total')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('g [(km/s)²/kpc]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Temperature profiles
    ax = axes[1, 0]
    ax.plot(r_T, kT_obs, 'ko', markersize=8, label='Observed')
    ax.plot(r_T, kT_pred, 'r-', linewidth=2, label=f'Enhanced (med={median_enhanced:.3f})')
    ax.plot(r_T, kT_GR, 'g--', label=f'GR only (med={median_GR:.3f})')
    ax.set_xscale('log')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('kT [keV]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Residuals
    ax = axes[1, 1]
    ax.semilogx(r_T, residual_enhanced, 'r-', linewidth=2, label='Enhanced')
    ax.semilogx(r_T, residual_GR, 'g--', label='GR only')
    ax.axhline(0.3, color='k', linestyle=':', alpha=0.5)
    ax.axhline(0.6, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('|ΔT|/T')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{cluster_name}: Geometric Enhancement Test (λ₀={lambda0}, α={alpha})')
    plt.tight_layout()
    
    # Save results
    out_dir = Path("root-m/geometric_enhancement/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_dir / f"{cluster_name}_enhancement_test.png", dpi=150)
    
    # Save metrics
    metrics = {
        'cluster': cluster_name,
        'lambda0': lambda0,
        'alpha': alpha,
        'r_half_kpc': float(r_half),
        'median_residual_enhanced': float(median_enhanced),
        'median_residual_GR': float(median_GR),
        'improvement_factor': float(median_GR / median_enhanced) if median_enhanced > 0 else np.inf,
        'max_kappa': float(np.max(kappa)),
        'kappa_at_r_half': float(np.interp(r_half, r, kappa))
    }
    
    with open(out_dir / f"{cluster_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def scan_parameters():
    """Scan over enhancement parameters for multiple clusters."""
    
    clusters = ['ABELL_0426', 'ABELL_1689']
    lambda_grid = [0.3, 0.5, 0.7, 1.0]
    alpha_grid = [1.0, 1.5, 2.0]
    
    results = []
    
    for cluster in clusters:
        for lambda0 in lambda_grid:
            for alpha in alpha_grid:
                try:
                    metrics = test_cluster_with_enhancement(cluster, lambda0, alpha)
                    results.append(metrics)
                    print(f"{cluster} λ={lambda0} α={alpha}: "
                          f"enhanced={metrics['median_residual_enhanced']:.3f} "
                          f"GR={metrics['median_residual_GR']:.3f}")
                except Exception as e:
                    print(f"Error with {cluster} λ={lambda0} α={alpha}: {e}")
    
    # Find best parameters
    results_df = pd.DataFrame(results)
    best_idx = results_df['median_residual_enhanced'].idxmin()
    best = results_df.iloc[best_idx]
    
    print("\n=== BEST PARAMETERS ===")
    print(f"λ₀ = {best['lambda0']}")
    print(f"α = {best['alpha']}")
    print(f"Cluster: {best['cluster']}")
    print(f"Median residual: {best['median_residual_enhanced']:.3f}")
    print(f"Improvement over GR: {best['improvement_factor']:.2f}x")
    
    # Save scan results
    results_df.to_csv(Path("root-m/geometric_enhancement/results/scan_results.csv"), index=False)
    
    return best

if __name__ == "__main__":
    # Test individual cluster
    print("Testing ABELL_0426 with default parameters...")
    test_cluster_with_enhancement('ABELL_0426')
    
    # Run parameter scan
    print("\nRunning parameter scan...")
    best = scan_parameters()