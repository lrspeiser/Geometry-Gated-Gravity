#!/usr/bin/env python3
"""
Compare all alternate gravity (GE) formula variants against observed lensing.

This script:
1. Loads observed baryon profiles (gas + stars) for HFF clusters
2. Computes GR deflection from baryons only (baseline)
3. Applies each GE formula variant to amplify the baryon lensing
4. Compares all variants against the observed deflection (DM+baryons)
5. Generates comparison plots showing which formula gets closest

GE Formula Variants:
- ratio: fX = x²/(a - b·Σ̂)
- ratio_curv: fX = x²/(a - b·Σ̂ - d·|∇ln Σ|)  [BEST for galaxies]
- ratio_curv_gbar: fX = x²/(a - b·Σ̂ - d·|∇ln Σ| + e·√gbar)
- exp: fX = α·x²·(exp(Σ̂) + c)
- exp_curv: fX = α·x²·(exp(Σ̂) + c + d·|∇ln Σ|)

Reference: PAPER_O2_RATIO_CURV.md
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import interp1d

# Constants
M_SUN = 1.98847e30  # kg
KPC_TO_M = 3.085677581e19  # m
PC_TO_M = 3.085677581e16  # m
C_LIGHT = 2.99792458e8  # m/s
G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2
ARCSEC_TO_RAD = 4.84813681e-6  # radians

# Best-fit parameters from galaxy rotation curves (mape_median optimization)
# Source: gravity_learn/experiments/eval/global_fit/mape_median_20250926_2259/
GE_PARAMS = {
    "ratio": {
        "a": 0.669,
        "b": 0.140,
    },
    "ratio_curv": {
        "a": 0.669,
        "b": 0.140,
        "d": 0.087,
    },
    "ratio_curv_gbar": {
        "a": 0.600,
        "b": 0.120,
        "d": 0.100,
        "e": 0.050,
    },
    "exp": {
        "alpha": 1.0,
        "c": 0.5,
    },
    "exp_curv": {
        "alpha": 1.0,
        "c": 0.5,
        "d": 0.1,
    },
}


def load_cluster_baryon_profile(
    cluster_name: str, data_dir: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load gas + stellar baryon density profiles for a cluster.
    
    Returns:
        radius_kpc: Ascending sorted radii in kpc
        rho_baryon: Total baryon mass density in Msun/kpc³
    """
    cluster_path = data_dir / cluster_name
    
    # Load gas profile (electron density)
    gas_file = cluster_path / "gas_profile.csv"
    if not gas_file.exists():
        raise FileNotFoundError(f"Gas profile not found: {gas_file}")
    
    gas_df = pd.read_csv(gas_file)
    # Try different column name variants
    r_col = "r_kpc" if "r_kpc" in gas_df.columns else "radius_kpc"
    n_col = "n_e_cm3" if "n_e_cm3" in gas_df.columns else "electron_density_cm3"
    
    r_gas_kpc = gas_df[r_col].values
    n_e_cm3 = gas_df[n_col].values
    
    # Convert electron density to gas mass density
    # Assume cosmic abundance: n_H ≈ n_e (ionized), mass per H atom ≈ 1.4 m_p
    m_p = 1.67262192e-27  # kg
    rho_gas_kg_m3 = n_e_cm3 * 1e6 * 1.4 * m_p  # kg/m³
    rho_gas_Msun_kpc3 = rho_gas_kg_m3 * (KPC_TO_M**3) / M_SUN
    
    # Load stellar profile - try multiple filenames
    stars_file = cluster_path / "stars_profile.csv"
    if not stars_file.exists():
        stars_file = cluster_path / "stellar_profile.csv"
    
    if not stars_file.exists():
        print(f"Warning: No stellar profile found for {cluster_name}, using gas only")
        rho_stars_Msun_kpc3 = np.zeros_like(rho_gas_Msun_kpc3)
        r_stars_kpc = r_gas_kpc
    else:
        stars_df = pd.read_csv(stars_file)
        r_col_s = "r_kpc" if "r_kpc" in stars_df.columns else "radius_kpc"
        rho_col = "rho_star_Msun_per_kpc3" if "rho_star_Msun_per_kpc3" in stars_df.columns else "stellar_density_Msun_kpc3"
        
        r_stars_kpc = stars_df[r_col_s].values
        rho_stars_Msun_kpc3 = stars_df[rho_col].values
    
    # Sort gas radii
    sort_idx_gas = np.argsort(r_gas_kpc)
    r_gas_kpc = r_gas_kpc[sort_idx_gas]
    rho_gas_Msun_kpc3 = rho_gas_Msun_kpc3[sort_idx_gas]
    
    # Sort stellar radii
    sort_idx_stars = np.argsort(r_stars_kpc)
    r_stars_kpc = r_stars_kpc[sort_idx_stars]
    rho_stars_Msun_kpc3 = rho_stars_Msun_kpc3[sort_idx_stars]
    
    # Interpolate both onto a common radius grid
    r_min = max(r_gas_kpc.min(), r_stars_kpc.min() if len(r_stars_kpc) > 0 else 0)
    r_max = min(r_gas_kpc.max(), r_stars_kpc.max() if len(r_stars_kpc) > 0 else r_gas_kpc.max())
    
    r_common = np.geomspace(max(r_min, 1.0), r_max, 200)
    
    interp_gas = interp1d(r_gas_kpc, rho_gas_Msun_kpc3, kind='linear', 
                          bounds_error=False, fill_value=0.0)
    rho_gas_common = interp_gas(r_common)
    
    if len(r_stars_kpc) > 0:
        interp_stars = interp1d(r_stars_kpc, rho_stars_Msun_kpc3, kind='linear',
                               bounds_error=False, fill_value=0.0)
        rho_stars_common = interp_stars(r_common)
    else:
        rho_stars_common = np.zeros_like(r_common)
    
    rho_baryon = rho_gas_common + rho_stars_common
    
    return r_common, rho_baryon


def compute_surface_density_abel(
    r_3d_kpc: np.ndarray, rho_3d: np.ndarray, R_2d_kpc: np.ndarray
) -> np.ndarray:
    """
    Compute projected surface density Σ(R) via Abel projection.
    
    Σ(R) = 2 ∫_R^∞ ρ(r) r dr / √(r² - R²)
    
    Args:
        r_3d_kpc: 3D radii (ascending)
        rho_3d: 3D density at r_3d
        R_2d_kpc: 2D projected radii where to compute Σ
    
    Returns:
        Sigma: Surface density in Msun/kpc²
    """
    Sigma = np.zeros_like(R_2d_kpc)
    
    for i, R in enumerate(R_2d_kpc):
        mask = r_3d_kpc > R
        if not np.any(mask):
            Sigma[i] = 0.0
            continue
        
        r_int = r_3d_kpc[mask]
        rho_int = rho_3d[mask]
        
        integrand = rho_int * r_int / np.sqrt(r_int**2 - R**2)
        Sigma[i] = 2.0 * simpson(integrand, x=r_int)
    
    return Sigma


def compute_gr_deflection(
    R_2d_kpc: np.ndarray,
    Sigma_Msun_kpc2: np.ndarray,
    z_lens: float,
    z_source: float,
    H0: float = 70.0,
    Om0: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute GR lensing deflection from surface density.
    
    Returns:
        theta_arcsec: Angular radii
        alpha_GR_arcsec: GR deflection angle
        kbar: Mean convergence from baryons
    """
    from astropy.cosmology import FlatLambdaCDM
    
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    D_l = cosmo.angular_diameter_distance(z_lens).value  # Mpc
    D_s = cosmo.angular_diameter_distance(z_source).value
    D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value
    
    # Convert R from kpc to arcsec
    theta_arcsec = R_2d_kpc / D_l / 1e3 * 206265  # kpc -> rad -> arcsec
    
    # Critical surface density
    Sigma_crit_Msun_kpc2 = (C_LIGHT**2 / (4 * np.pi * G_NEWTON)) * \
                           (D_s / (D_l * D_ls)) * \
                           (KPC_TO_M**2 / M_SUN) / 1e6  # Msun/kpc²
    
    # Convergence
    kappa = Sigma_Msun_kpc2 / Sigma_crit_Msun_kpc2
    
    # Mean convergence within R
    kbar = np.zeros_like(kappa)
    for i in range(len(R_2d_kpc)):
        if i == 0:
            kbar[i] = kappa[i]
        else:
            # Integrate kappa weighted by 2πR
            R_int = R_2d_kpc[:i+1]
            integrand = kappa[:i+1] * R_int
            M_enc = 2.0 * np.pi * simpson(integrand, x=R_int)
            kbar[i] = M_enc / (np.pi * R_2d_kpc[i]**2)
    
    # Deflection angle: α = k̄ × θ
    alpha_GR_arcsec = kbar * theta_arcsec
    
    return theta_arcsec, alpha_GR_arcsec, kbar


def sigma_hat(Sigma_Msun_pc2: np.ndarray) -> np.ndarray:
    """Normalized surface density: Σ̂ = log10(Σ / 100 Msun/pc²)"""
    return np.log10(np.maximum(Sigma_Msun_pc2, 1e-8) / 100.0)


def grad_log_sigma(R_kpc: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Logarithmic gradient: ∇ln Σ = d(ln Σ)/dR in kpc⁻¹"""
    dlnS = np.zeros_like(R_kpc)
    lnS = np.log(np.maximum(Sigma, 1e-12))
    
    # Centered finite differences
    dlnS[1:-1] = (lnS[2:] - lnS[:-2]) / (R_kpc[2:] - R_kpc[:-2])
    # Forward/backward at edges
    dlnS[0] = (lnS[1] - lnS[0]) / (R_kpc[1] - R_kpc[0])
    dlnS[-1] = (lnS[-1] - lnS[-2]) / (R_kpc[-1] - R_kpc[-2])
    
    return dlnS


def dimensionless_radius(R_kpc: np.ndarray, Rd_kpc: Optional[float] = None) -> np.ndarray:
    """x = R / R_d (dimensionless radius)"""
    if Rd_kpc is None or Rd_kpc <= 0:
        Rd_kpc = np.median(R_kpc)  # Fallback
    return R_kpc / Rd_kpc


def compute_fX_ratio(x: np.ndarray, Sh: np.ndarray, params: Dict) -> np.ndarray:
    """fX = x²/(a - b·Σ̂)"""
    a = params["a"]
    b = params["b"]
    denom = a - b * Sh
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
    return np.maximum(0.0, (x * x) / denom)


def compute_fX_ratio_curv(
    x: np.ndarray, Sh: np.ndarray, dlnS: np.ndarray, params: Dict
) -> np.ndarray:
    """fX = x²/(a - b·Σ̂ - d·|∇ln Σ|)"""
    a = params["a"]
    b = params["b"]
    d = params["d"]
    denom = a - b * Sh - d * np.abs(dlnS)
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
    return np.maximum(0.0, (x * x) / denom)


def compute_fX_ratio_curv_gbar(
    x: np.ndarray,
    Sh: np.ndarray,
    dlnS: np.ndarray,
    gbar: np.ndarray,
    params: Dict,
) -> np.ndarray:
    """fX = x²/(a - b·Σ̂ - d·|∇ln Σ| + e·√gbar)"""
    a = params["a"]
    b = params["b"]
    d = params["d"]
    e = params["e"]
    denom = a - b * Sh - d * np.abs(dlnS) + e * np.sqrt(np.maximum(gbar, 0.0))
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
    return np.maximum(0.0, (x * x) / denom)


def compute_fX_exp(x: np.ndarray, Sh: np.ndarray, params: Dict) -> np.ndarray:
    """fX = α·x²·(exp(Σ̂) + c)"""
    alpha = params["alpha"]
    c = params["c"]
    return np.maximum(0.0, alpha * (x * x) * (np.exp(Sh) + c))


def compute_fX_exp_curv(
    x: np.ndarray, Sh: np.ndarray, dlnS: np.ndarray, params: Dict
) -> np.ndarray:
    """fX = α·x²·(exp(Σ̂) + c + d·|∇ln Σ|)"""
    alpha = params["alpha"]
    c = params["c"]
    d = params["d"]
    return np.maximum(0.0, alpha * (x * x) * (np.exp(Sh) + c + d * np.abs(dlnS)))


def apply_ge_boost(
    formula: str,
    R_kpc: np.ndarray,
    Sigma_Msun_kpc2: np.ndarray,
    kbar_GR: np.ndarray,
    params: Dict,
    Rd_kpc: Optional[float] = None,
) -> np.ndarray:
    """
    Apply GE formula to boost mean convergence.
    
    GE model: κ_GE(R) = κ_bar(R) × (1 + fX(R))
    
    Returns:
        kbar_GE: Boosted mean convergence
    """
    # Compute geometry features
    Sigma_Msun_pc2 = Sigma_Msun_kpc2 * 1e6  # kpc² -> pc²
    Sh = sigma_hat(Sigma_Msun_pc2)
    dlnS = grad_log_sigma(R_kpc, Sigma_Msun_kpc2)
    x = dimensionless_radius(R_kpc, Rd_kpc)
    
    # For gbar variant, need baryon acceleration (use circular velocity approximation)
    # g_bar ≈ V_bar² / R, but we don't have V_bar for clusters
    # Use enclosed mass: g_bar ≈ G M(<R) / R²
    M_enc_Msun = np.zeros_like(R_kpc)
    for i in range(len(R_kpc)):
        if i == 0:
            M_enc_Msun[i] = np.pi * R_kpc[0]**2 * Sigma_Msun_kpc2[0]
        else:
            integrand = Sigma_Msun_kpc2[:i+1] * R_kpc[:i+1]
            M_enc_Msun[i] = 2.0 * np.pi * simpson(integrand, x=R_kpc[:i+1])
    
    G_SI = 6.67430e-11  # m³ kg⁻¹ s⁻²
    M_enc_kg = M_enc_Msun * M_SUN
    R_m = R_kpc * KPC_TO_M
    gbar_SI = G_SI * M_enc_kg / (R_m**2 + 1e-30)
    gbar = gbar_SI * 1e10  # Convert to 10^-10 m/s² (MOND units)
    
    # Compute fX based on formula
    if formula == "ratio":
        fX = compute_fX_ratio(x, Sh, params)
    elif formula == "ratio_curv":
        fX = compute_fX_ratio_curv(x, Sh, dlnS, params)
    elif formula == "ratio_curv_gbar":
        fX = compute_fX_ratio_curv_gbar(x, Sh, dlnS, gbar, params)
    elif formula == "exp":
        fX = compute_fX_exp(x, Sh, params)
    elif formula == "exp_curv":
        fX = compute_fX_exp_curv(x, Sh, dlnS, params)
    else:
        raise ValueError(f"Unknown formula: {formula}")
    
    # Boost convergence
    kappa_GE = kbar_GR * (1.0 + fX)
    
    # Recompute mean convergence (self-consistent)
    # This is approximate - proper treatment needs iterative projection
    # For now, use kbar_GE ≈ kbar_GR × (1 + fX)
    kbar_GE = kbar_GR * (1.0 + fX)
    
    return kbar_GE


def load_observed_deflection(
    cluster_name: str, team: str = "cats", version: str = "v4.1", z_lens: float = 0.4, z_source: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Load observed deflection using lensing_utils."""
    # Import the HLSP loader
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from scripts.lensing_utils import alpha_fun_ACCEPTED
    
    # Map short names to full cluster names
    cluster_map = {
        "macs0416": "macs0416",
        "macs0717": "macs0717",
        "macs1149": "macs1149",
        "MACSJ0416": "macs0416",
        "MACSJ0717": "macs0717",
        "MACSJ1149": "macs1149",
    }
    
    cluster_hlsp = cluster_map.get(cluster_name, cluster_name.lower())
    
    # Get deflection function from HLSP
    alpha_func = alpha_fun_ACCEPTED(cluster_hlsp, team, version, z_lens, z_source)
    if alpha_func is None:
        raise RuntimeError(f"Could not load HLSP deflection for {cluster_name}")
    
    # Sample at theta points
    theta_arcsec = np.linspace(0.5, 100.0, 200)
    alpha_arcsec = np.array([alpha_func(th) for th in theta_arcsec])
    
    return theta_arcsec, alpha_arcsec


def plot_comparison(
    cluster_name: str,
    theta_arcsec: np.ndarray,
    alpha_GR: np.ndarray,
    alpha_formulas: Dict[str, np.ndarray],
    alpha_observed: np.ndarray,
    out_dir: Path,
    team: str,
    version: str,
):
    """Create comparison plot of all GE formulas vs observed."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)
    
    # Color scheme
    colors = {
        "ratio": "#1f77b4",
        "ratio_curv": "#ff7f0e",
        "ratio_curv_gbar": "#2ca02c",
        "exp": "#d62728",
        "exp_curv": "#9467bd",
    }
    
    # Panel 1: Deflection angles
    ax1 = axes[0]
    ax1.plot(theta_arcsec, theta_arcsec, '--', color='gray', alpha=0.5, label='α=θ (no lensing)')
    ax1.plot(theta_arcsec, alpha_GR, color='black', linewidth=2, label='GR (baryons only)')
    
    for formula, alpha_ge in alpha_formulas.items():
        ax1.plot(theta_arcsec, alpha_ge, linewidth=2, label=f'GE: {formula}', color=colors[formula])
    
    ax1.plot(theta_arcsec, alpha_observed, 'o', markersize=4, color='magenta', 
             markerfacecolor='none', markeredgewidth=1.5, label='Observed (DM+baryons)')
    
    ax1.set_xlabel('θ (arcsec)', fontsize=12)
    ax1.set_ylabel('α(θ) (arcsec)', fontsize=12)
    ax1.set_title(f'{cluster_name} {team} {version}: Deflection Angle Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, theta_arcsec.max())
    
    # Panel 2: Ratio to observed (log scale)
    ax2 = axes[1]
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect match')
    
    # Avoid division by zero
    alpha_obs_safe = np.where(alpha_observed > 0, alpha_observed, np.nan)
    
    ratio_gr = alpha_GR / alpha_obs_safe
    ax2.plot(theta_arcsec, ratio_gr, color='black', linewidth=2, label='GR / Observed')
    
    for formula, alpha_ge in alpha_formulas.items():
        ratio_ge = alpha_ge / alpha_obs_safe
        ax2.plot(theta_arcsec, ratio_ge, linewidth=2, label=f'{formula} / Observed', color=colors[formula])
    
    ax2.set_xlabel('θ (arcsec)', fontsize=12)
    ax2.set_ylabel('α_model / α_observed', fontsize=12)
    ax2.set_title('Ratio to Observed Deflection', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(0, theta_arcsec.max())
    
    # Panel 3: Fractional residuals
    ax3 = axes[2]
    ax3.axhline(0.0, color='gray', linestyle='--', alpha=0.5)
    
    frac_gr = (alpha_GR - alpha_observed) / alpha_obs_safe
    ax3.plot(theta_arcsec, frac_gr, color='black', linewidth=2, label='GR residual')
    
    for formula, alpha_ge in alpha_formulas.items():
        frac_ge = (alpha_ge - alpha_observed) / alpha_obs_safe
        ax3.plot(theta_arcsec, frac_ge, linewidth=2, label=f'{formula} residual', color=colors[formula])
    
    ax3.set_xlabel('θ (arcsec)', fontsize=12)
    ax3.set_ylabel('(α_model - α_obs) / α_obs', fontsize=12)
    ax3.set_title('Fractional Residuals', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, theta_arcsec.max())
    
    # Save
    out_file = out_dir / f"{cluster_name}_{team}_{version}_ge_formulas_comparison.png"
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_file}")


def compute_metrics(
    theta_arcsec: np.ndarray,
    alpha_model: np.ndarray,
    alpha_observed: np.ndarray,
    theta_min: float = 10.0,
    theta_max: float = 100.0,
) -> Dict[str, float]:
    """Compute goodness-of-fit metrics over a radial range."""
    mask = (theta_arcsec >= theta_min) & (theta_arcsec <= theta_max)
    
    if not np.any(mask):
        return {"rmse": np.nan, "mape": np.nan, "max_ratio": np.nan}
    
    alpha_m = alpha_model[mask]
    alpha_o = alpha_observed[mask]
    
    # RMSE
    rmse = np.sqrt(np.mean((alpha_m - alpha_o)**2))
    
    # MAPE
    mape = np.mean(np.abs((alpha_m - alpha_o) / np.maximum(alpha_o, 1e-9)))
    
    # Max ratio
    max_ratio = np.max(np.abs(alpha_m / np.maximum(alpha_o, 1e-9)))
    
    return {"rmse": float(rmse), "mape": float(mape), "max_ratio": float(max_ratio)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clusters", nargs="+", 
                       default=["MACSJ0416", "MACSJ0717", "MACSJ1149"],
                       help="Cluster names to analyze")
    parser.add_argument("--data-dir", type=Path, 
                       default=Path("data/clusters"),
                       help="Directory with cluster baryon profiles")
    parser.add_argument("--team", default="cats", help="Lensing team")
    parser.add_argument("--version", default="v4.1", help="Lensing model version")
    parser.add_argument("--out-dir", type=Path, default=Path("out/plots"),
                       help="Output directory for plots")
    parser.add_argument("--z-lens", type=float, default=0.4,
                       help="Lens redshift (default for HFF clusters)")
    parser.add_argument("--z-source", type=float, default=2.0,
                       help="Source redshift")
    parser.add_argument("--Rd-kpc", type=float, default=100.0,
                       help="Scale length for dimensionless radius (kpc)")
    
    args = parser.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary results
    summary = {}
    
    for cluster_name in args.clusters:
        print(f"\n{'='*60}")
        print(f"Processing: {cluster_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # 1. Load baryon profiles
            print("Loading baryon profiles...")
            r_3d_kpc, rho_baryon = load_cluster_baryon_profile(
                cluster_name, args.data_dir
            )
            
            # 2. Project to surface density
            print("Computing surface density via Abel projection...")
            R_2d_kpc = np.geomspace(1.0, r_3d_kpc.max() * 0.95, 150)
            Sigma_Msun_kpc2 = compute_surface_density_abel(r_3d_kpc, rho_baryon, R_2d_kpc)
            
            # 3. Compute GR deflection
            print("Computing GR deflection from baryons...")
            theta_arcsec, alpha_GR, kbar_GR = compute_gr_deflection(
                R_2d_kpc, Sigma_Msun_kpc2, args.z_lens, args.z_source
            )
            
            # 4. Apply all GE formulas
            print("Applying GE formulas...")
            alpha_formulas = {}
            
            for formula, params in GE_PARAMS.items():
                print(f"  - {formula}: {params}")
                kbar_GE = apply_ge_boost(
                    formula, R_2d_kpc, Sigma_Msun_kpc2, kbar_GR, params, args.Rd_kpc
                )
                alpha_GE = kbar_GE * theta_arcsec
                alpha_formulas[formula] = alpha_GE
            
            # 5. Load observed deflection
            print("Loading observed deflection...")
            theta_obs, alpha_obs = load_observed_deflection(
                cluster_name, args.team, args.version, args.z_lens, args.z_source
            )
            
            # Interpolate observed onto model grid
            from scipy.interpolate import interp1d
            interp_obs = interp1d(theta_obs, alpha_obs, kind='linear',
                                 bounds_error=False, fill_value=np.nan)
            alpha_observed = interp_obs(theta_arcsec)
            
            # 6. Compute metrics
            print("\nMetrics (10-100 arcsec):")
            cluster_metrics = {}
            
            # GR baseline
            metrics_gr = compute_metrics(theta_arcsec, alpha_GR, alpha_observed)
            print(f"  GR: RMSE={metrics_gr['rmse']:.2f}\", MAPE={metrics_gr['mape']:.2%}, Max ratio={metrics_gr['max_ratio']:.1f}x")
            cluster_metrics["GR"] = metrics_gr
            
            # GE formulas
            for formula, alpha_ge in alpha_formulas.items():
                metrics = compute_metrics(theta_arcsec, alpha_ge, alpha_observed)
                print(f"  {formula}: RMSE={metrics['rmse']:.2f}\", MAPE={metrics['mape']:.2%}, Max ratio={metrics['max_ratio']:.1f}x")
                cluster_metrics[formula] = metrics
            
            summary[cluster_name] = cluster_metrics
            
            # 7. Plot comparison
            print("Creating comparison plot...")
            plot_comparison(
                cluster_name, theta_arcsec, alpha_GR, alpha_formulas,
                alpha_observed, args.out_dir, args.team, args.version
            )
            
        except Exception as e:
            print(f"ERROR processing {cluster_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary_file = args.out_dir / "ge_formulas_comparison_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    # Print best formula
    print(f"\n{'='*60}")
    print("BEST FORMULA BY CLUSTER (lowest MAPE):")
    print(f"{'='*60}")
    for cluster, metrics in summary.items():
        formula_mapes = {k: v["mape"] for k, v in metrics.items() if k != "GR"}
        best_formula = min(formula_mapes, key=formula_mapes.get)
        best_mape = formula_mapes[best_formula]
        print(f"{cluster:15s}: {best_formula:20s} (MAPE={best_mape:.2%})")


if __name__ == "__main__":
    main()
