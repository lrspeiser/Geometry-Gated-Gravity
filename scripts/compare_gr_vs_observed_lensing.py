#!/usr/bin/env python3
"""
Compare GR (baryons-only) vs Observed Lensing for HFF Clusters

Computes deflection from OBSERVED baryons (gas + stars, NO dark matter) using standard GR,
and compares to the accepted lensing reconstruction from HFF teams.

For each cluster, generates:
- Deflection curve: α(θ) for GR-baryons vs Observed-total
- Mean convergence: k̄(<θ) = α/θ
- Ratio plot: α_observed / α_GR-baryons (should be >> 1, showing DM dominance)

Usage:
    python scripts/compare_gr_vs_observed_lensing.py --cluster macs0416 --team cats --version v4.1 --zs 2.0
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    sigma_crit_Msun_per_kpc2, angular_diameter_distance_kpc, ne_to_rho_gas_Msun_kpc3
)
from scripts.lensing_utils import CLASH, alpha_fun_ACCEPTED

# Constants
G_kpc_km2_s2_Msun = 4.300917270e-6  # kpc (km/s)^2 / Msun
c_km_s = 299792.458  # km/s

def load_baryon_profiles(cluster_name: str):
    """Load observed gas and star profiles, return combined 3D density ρ(r)."""
    base = ROOT / 'data' / 'clusters' / cluster_name
    gas_path = base / 'gas_profile.csv'
    stars_path = base / 'stars_profile.csv'
    
    # Load data
    gas_df = pd.read_csv(gas_path)
    stars_df = pd.read_csv(stars_path)
    
    # Convert gas electron density to mass density
    r_gas = gas_df['r_kpc'].values
    n_e = gas_df['n_e_cm3'].values
    rho_gas = ne_to_rho_gas_Msun_kpc3(n_e)
    
    # Stars already in correct units
    r_stars = stars_df['r_kpc'].values
    rho_stars = stars_df['rho_star_Msun_per_kpc3'].values
    
    # Filter out NaN/inf values
    mask_gas = np.isfinite(r_gas) & np.isfinite(rho_gas) & (r_gas > 0)
    r_gas = r_gas[mask_gas]
    rho_gas = rho_gas[mask_gas]
    
    mask_stars = np.isfinite(r_stars) & np.isfinite(rho_stars) & (r_stars > 0)
    r_stars = r_stars[mask_stars]
    rho_stars = rho_stars[mask_stars]
    
    # Sort arrays (np.interp requires ascending x coordinates)
    idx_gas = np.argsort(r_gas)
    r_gas = r_gas[idx_gas]
    rho_gas = rho_gas[idx_gas]
    
    idx_stars = np.argsort(r_stars)
    r_stars = r_stars[idx_stars]
    rho_stars = rho_stars[idx_stars]
    
    # Merge onto common grid (use unique sorted radii from both)
    r_all = np.unique(np.concatenate([r_gas, r_stars]))
    
    # Interpolate both profiles onto common grid
    rho_gas_interp = np.interp(r_all, r_gas, rho_gas, left=0, right=0)
    rho_stars_interp = np.interp(r_all, r_stars, rho_stars, left=0, right=0)
    
    # Total baryon density
    rho_baryons = rho_gas_interp + rho_stars_interp
    
    print(f"  Loaded {len(r_gas)} gas + {len(r_stars)} star points")
    print(f"  Combined grid: {len(r_all)} radii from {r_all.min():.2f} to {r_all.max():.2f} kpc")
    print(f"  ρ_baryons range: {rho_baryons.min():.2e} to {rho_baryons.max():.2e} Msun/kpc^3")
    
    return r_all, rho_baryons


def abel_project_clean(r: np.ndarray, rho: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Abel projection: Σ(R) = 2∫_R^∞ ρ(r) r/√(r²-R²) dr
    
    Uses clean, regularly-spaced R grid.
    """
    Sigma = np.zeros_like(R)
    for i, Rp in enumerate(R):
        # Integrate from Rp to r_max
        mask = r >= Rp
        if not mask.any():
            continue
        rr = r[mask]
        rh = rho[mask]
        if len(rr) < 2:
            continue
        # Integrand
        denom = np.sqrt(np.maximum(rr**2 - Rp**2, 1e-30))
        integrand = rh * rr / denom
        Sigma[i] = 2.0 * np.trapezoid(integrand, rr)
    return Sigma


def compute_gr_baryons_deflection(cluster_name: str, z_lens: float, z_source: float, 
                                  theta_max_arcsec: float = 100.0, n_theta: int = 200):
    """Compute GR deflection from observed baryons only (no dark matter).
    
    Returns:
        theta_grid: angular radii in arcsec
        alpha_gr: deflection in arcsec at each theta
    """
    # Load baryon density ρ(r)
    r, rho_baryons = load_baryon_profiles(cluster_name)
    
    # Create clean R grid for projection (log-spaced from r_min to r_max)
    r_min = max(0.1, r[r > 0].min())
    r_max = r.max()
    R = np.logspace(np.log10(r_min), np.log10(r_max), 600)
    
    # Project to surface density Σ(R)
    Sigma_baryons = abel_project_clean(r, rho_baryons, R)
    
    # Compute enclosed mass M(<R) = 2π ∫_0^R Σ(R') R' dR'
    M_enc = np.array([2 * np.pi * np.trapezoid(Sigma_baryons[:i+1] * R[:i+1], R[:i+1]) 
                      for i in range(len(R))])
    
    # Compute mean convergence k̄(<R) = M(<R) / (π R² Σ_crit)
    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)
    kbar = M_enc / (np.pi * R**2 * Sigma_crit)
    
    # Convert R (kpc) to θ (arcsec)
    D_d = angular_diameter_distance_kpc(z_lens)
    theta_R_arcsec = (R / D_d) * 206265.0
    
    # Deflection: α(θ) = k̄(θ) × θ
    alpha_R_arcsec = kbar * theta_R_arcsec
    
    # Interpolate onto desired theta grid
    theta_grid = np.linspace(0.5, theta_max_arcsec, n_theta)
    alpha_gr = np.interp(theta_grid, theta_R_arcsec, alpha_R_arcsec, left=0, right=alpha_R_arcsec[-1])
    
    # Verify units by checking total baryon mass
    M_tot = M_enc[-1]
    print(f"  Total baryon mass within {r_max:.0f} kpc: {M_tot:.2e} Msun")
    print(f"  k̄_baryon at {theta_grid[-1]:.0f}″: {kbar[-1]:.4f}")
    print(f"  α_GR at {theta_grid[-1]:.0f}″: {alpha_gr[-1]:.2f}″")
    
    return theta_grid, alpha_gr


def plot_comparison(cluster: str, team: str, version: str, z_lens: float, z_source: float,
                    theta_grid: np.ndarray, alpha_gr: np.ndarray, alpha_obs: np.ndarray,
                    alpha_tanh: np.ndarray, alpha_edge: np.ndarray,
                    outdir: Path):
    """Create comparison plots: deflection, convergence, and ratio."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Panel 1: Deflection α(θ)
    ax1 = axes[0]
    ax1.plot(theta_grid, theta_grid, 'k--', lw=1.0, alpha=0.5, label='α=θ')
    ax1.plot(theta_grid, alpha_obs, 'b-', lw=2.5, label='Observed (DM+baryons)', alpha=0.8)
    ax1.plot(theta_grid, alpha_gr, 'k-', lw=2.0, label='GR (baryons only)', alpha=0.9)
    ax1.plot(theta_grid, alpha_tanh, color='#8c564b', lw=2.0, label='Slip: tanh', alpha=0.9)
    ax1.plot(theta_grid, alpha_edge, color='#17becf', lw=2.0, label='Slip: edge-peaked', alpha=0.9)
    ax1.set_xlabel('θ (arcsec)', fontsize=11)
    ax1.set_ylabel('α(θ) (arcsec)', fontsize=11)
    ax1.set_title(f'{cluster.upper()} {team} {version}: Deflection', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3, ls=':')
    ax1.set_xlim(0, theta_grid[-1])
    
    # Panel 2: Mean convergence k̄(<θ) = α/θ
    ax2 = axes[1]
    eps = 1e-6
    kbar_obs = alpha_obs / np.maximum(theta_grid, eps)
    kbar_gr = alpha_gr / np.maximum(theta_grid, eps)
    kbar_tanh = alpha_tanh / np.maximum(theta_grid, eps)
    kbar_edge = alpha_edge / np.maximum(theta_grid, eps)
    ax2.plot(theta_grid, kbar_obs, 'b-', lw=2.5, label='Observed k̄(<θ)', alpha=0.8)
    ax2.plot(theta_grid, kbar_gr, 'k-', lw=2.0, label='GR k̄(<θ)', alpha=0.9)
    ax2.plot(theta_grid, kbar_tanh, color='#8c564b', lw=2.0, label='tanh k̄(<θ)', alpha=0.9)
    ax2.plot(theta_grid, kbar_edge, color='#17becf', lw=2.0, label='edge k̄(<θ)', alpha=0.9)
    ax2.axhline(1.0, color='k', ls='--', lw=1.0, alpha=0.5)
    ax2.set_xlabel('θ (arcsec)', fontsize=11)
    ax2.set_ylabel('k̄(<θ)', fontsize=11)
    ax2.set_title('Mean Convergence', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3, ls=':')
    ax2.set_xlim(0, theta_grid[-1])
    
    # Panel 3: Ratio α_obs / α_GR (shows DM contribution)
    ax3 = axes[2]
    ratio_gr = np.divide(alpha_obs, np.maximum(alpha_gr, eps))
    ratio_tanh = np.divide(alpha_obs, np.maximum(alpha_tanh, eps))
    ratio_edge = np.divide(alpha_obs, np.maximum(alpha_edge, eps))
    ax3.plot(theta_grid, ratio_gr, 'purple', lw=2.5, alpha=0.8, label='obs/GR')
    ax3.plot(theta_grid, ratio_tanh, color='#8c564b', lw=2.0, alpha=0.9, label='obs/tanh')
    ax3.plot(theta_grid, ratio_edge, color='#17becf', lw=2.0, alpha=0.9, label='obs/edge')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.axhline(1.0, color='k', ls='--', lw=1.0, alpha=0.5, label='No DM (ratio=1)')
    ax3.set_xlabel('θ (arcsec)', fontsize=11)
    ax3.set_ylabel('α_observed / α_GR-baryons', fontsize=11)
    ax3.set_title('Lensing Amplification (DM dominance)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(alpha=0.3, ls=':')
    ax3.set_xlim(0, theta_grid[-1])
    # choose percentile based on GR ratio for scaling
    safe = ratio_gr[np.isfinite(ratio_gr)]
    ymax = 10 if safe.size == 0 else max(10, float(np.nanpercentile(safe, 95) * 1.1))
    ax3.set_ylim(0, ymax)
    
    # Add text box with key metrics
    theta_50 = 50.0  # reference radius
    idx_50 = np.argmin(np.abs(theta_grid - theta_50))
    ratio_50 = ratio_gr[idx_50]
    textstr = f'z_lens = {z_lens:.3f}\nz_source = {z_source:.1f}\n'
    textstr += f'At θ={theta_50:.0f}″:\n'
    textstr += f'  α_obs = {alpha_obs[idx_50]:.2f}″\n'
    textstr += f'  α_GR = {alpha_gr[idx_50]:.2f}″\n'
    textstr += f'  α_tanh = {alpha_tanh[idx_50]:.2f}″\n'
    textstr += f'  α_edge = {alpha_edge[idx_50]:.2f}″\n'
    textstr += f'  obs/GR = {ratio_gr[idx_50]:.1f}x'
    ax3.text(0.97, 0.97, textstr, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    outpath = outdir / f'{cluster}_{team}_{version}_gr_vs_observed.png'
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cluster', required=True, help='Cluster ID (e.g., macs0416)')
    ap.add_argument('--team', required=True, help='Modeling team (e.g., cats)')
    ap.add_argument('--version', required=True, help='Model version (e.g., v4.1)')
    ap.add_argument('--zs', type=float, default=2.0, help='Source redshift')
    ap.add_argument('--theta-max', type=float, default=100.0, help='Max angle in arcsec')
    args = ap.parse_args()
    
    # Get redshift
    if args.cluster.lower() in CLASH:
        local_name, z_lens = CLASH[args.cluster.lower()]
    else:
        raise ValueError(f"Unknown cluster: {args.cluster}")
    
    print(f"\n{'='*60}")
    print(f"Cluster: {args.cluster.upper()} (z={z_lens:.3f})")
    print(f"Team: {args.team}, Version: {args.version}")
    print(f"Source redshift: {args.zs}")
    print(f"{'='*60}\n")
    
    # Compute GR baryons-only deflection
    print("Computing GR (baryons-only) deflection...")
    theta_grid, alpha_gr = compute_gr_baryons_deflection(
        local_name, z_lens, args.zs, 
        theta_max_arcsec=args.theta_max
    )
    
    # Get observed deflection from HLSP
    print("\nLoading observed deflection from HLSP maps...")
    alpha_obs_func = alpha_fun_ACCEPTED(args.cluster.lower(), args.team.lower(), 
                                        args.version, z_lens, args.zs)
    if alpha_obs_func is None:
        raise RuntimeError("Could not load HLSP deflection maps")
    
    alpha_obs = np.array([alpha_obs_func(th) for th in theta_grid])
    print(f"  α_observed at {theta_grid[-1]:.0f}″: {alpha_obs[-1]:.2f}″")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    outdir = ROOT / 'out' / 'plots'
    outdir.mkdir(parents=True, exist_ok=True)
    # Build slips
    # Σ(R) from Abel projection already computed inside compute_gr_baryons_deflection via rho_baryons.
    # Re-compute here for slip gates to ensure consistent R grid
    r, rho_baryons = load_baryon_profiles(local_name)
    R = np.logspace(np.log10(max(0.1, r[r>0].min())), np.log10(r.max()), 600)
    Sigma_baryons = abel_project_clean(r, rho_baryons, R)

    # Map Σ(R) to θ grid
    D_d_kpc = angular_diameter_distance_kpc(z_lens)
    theta_R_arcsec = (R / D_d_kpc) * 206265.0

    # Slip A) tanh activation
    def slip_tanh_density(Sigma_bar_kpc2, Sigma_min_pc2=50.0, steepness=500.0, floor=0.0, cap=50.0):
        Sigma_pc2 = np.maximum(Sigma_bar_kpc2/1e6, 0.0)
        x = (Sigma_pc2 - Sigma_min_pc2) / max(Sigma_min_pc2, 1e-12)
        S = 0.5 * (1.0 + np.tanh(steepness * x))
        S = floor + (1.0 - floor) * S
        S = np.clip(S, floor, cap)
        # preserve core and monotone export
        S = np.maximum.accumulate(S)
        return S

    # Slip B) edge-peaked
    def slip_edge_peaked(R_kpc, Sigma_bar_kpc2,
                         Sigma_min_pc2=50.0, S_max=30.0, q=6.0, eps_frac=1e-6,
                         use_window=True, Rs_kpc=None, p=1.2,
                         use_grad_gate=True, gamma=2.0,
                         floor=0.0, cap=50.0):
        Sigma_pc2 = np.maximum(Sigma_bar_kpc2/1e6, 0.0)
        eps = eps_frac * Sigma_min_pc2
        Delta = np.maximum(Sigma_pc2 - Sigma_min_pc2, 0.0)
        core = 1.0 + S_max * (1.0 - np.exp(- (Sigma_min_pc2 / (Delta + eps))**q))
        S = np.where(Sigma_pc2 <= Sigma_min_pc2, floor, core)
        if use_window:
            if Rs_kpc is None:
                Rs_kpc = 0.3 * np.max(R_kpc)
            ramp = 1.0 - np.exp(-(np.maximum(R_kpc,1e-6)/Rs_kpc)**p)
            S = 1.0 + (S - 1.0) * ramp
        if use_grad_gate:
            dlnR = np.gradient(np.log(np.maximum(R_kpc,1e-6)))
            dlnSigma = np.gradient(np.log(np.maximum(Sigma_pc2,1e-12)))
            G = np.abs(dlnSigma / dlnR)
            gate = (G**gamma) / (1.0 + G**gamma)
            S = 1.0 + (S - 1.0) * gate
        S = np.clip(S, floor, cap)
        S = np.maximum.accumulate(S)
        return S

    S_tanh_R = slip_tanh_density(Sigma_baryons, Sigma_min_pc2=50.0, steepness=500.0, floor=0.0, cap=50.0)
    S_edge_R = slip_edge_peaked(R, Sigma_baryons, Sigma_min_pc2=50.0, S_max=30.0, q=6.0,
                                use_window=True, Rs_kpc=0.3*R.max(), p=1.2,
                                use_grad_gate=True, gamma=2.0, floor=0.0, cap=50.0)

    S_tanh_theta = np.interp(theta_grid, theta_R_arcsec, S_tanh_R, left=S_tanh_R[0], right=S_tanh_R[-1])
    S_edge_theta = np.interp(theta_grid, theta_R_arcsec, S_edge_R, left=S_edge_R[0], right=S_edge_R[-1])

    alpha_tanh = alpha_gr * S_tanh_theta
    alpha_edge = alpha_gr * S_edge_theta

    plot_comparison(args.cluster, args.team, args.version, z_lens, args.zs,
                   theta_grid, alpha_gr, alpha_obs, alpha_tanh, alpha_edge,
                   outdir)
    
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
