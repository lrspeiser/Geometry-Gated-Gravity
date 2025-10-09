#!/usr/bin/env python3
"""
Cluster Lensing Comparison: Observed vs GR Baryons + Geometry-Tied Enhancements
================================================================================

Compares observed CLASH lensing deflection profiles with:
1. GR baseline (baryons only)
2. Mean-Σ gated slip (scales deflection based on interior mean surface density)
3. Scale-dependent response halo (convolution with ramping coupling)
4. Band-pass response (DoG kernel for MACS0717 dip-and-rise structure)

Key improvements:
- Uses mean surface density inside R for gating instead of local Σ
- Scale-dependent ε(R) that ramps with mean-Σ gate
- DoG (difference of Gaussians) kernel for MACS0717 to capture dip-and-rise
- Single-anchor calibration of S_inf from observed data

Usage:
    python concepts/cluster_lensing/cluster_baryon_lensing_comparison.py --cluster MACS0416
    python concepts/cluster_lensing/cluster_baryon_lensing_comparison.py --cluster all
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
G = 4.300917270e-6  # kpc (km/s)^2 Msun^-1
c_km_s = 299792.458
MU_E = 1.17
M_P_G = 1.67262192369e-24  # g
KPC_CM = 3.0856775814913673e21
MSUN_G = 1.988409870698051e33

# Cosmology
try:
    from astropy.cosmology import Planck18 as COSMO
    import astropy.units as u
    ASTROPY_OK = True
except Exception:
    ASTROPY_OK = False
    print("WARNING: astropy not available, using fallback cosmology")

# Root paths
ROOT = Path(__file__).resolve().parents[2]
CLASH_DIR = ROOT / 'data' / 'clash' / 'processed'
CLUSTER_DATA_DIR = ROOT / 'data' / 'clusters'
OUT_DIR = ROOT / 'out' / 'cluster_lensing_comparison'

# Cluster specifications (z_lens, z_source, theta_E_obs in arcsec)
CLUSTER_SPECS = {
    'MACS0416': {'z_lens': 0.396, 'z_source': 2.0, 'theta_E_obs': 36.0, 'data_dir': 'MACSJ0416'},
    'MACS0717': {'z_lens': 0.548, 'z_source': 2.5, 'theta_E_obs': 55.0, 'data_dir': 'MACSJ0717'},
    'MACS1149': {'z_lens': 0.544, 'z_source': 2.0, 'theta_E_obs': 32.0, 'data_dir': 'MACSJ1149'},
}


def ne_to_rho_gas_Msun_kpc3(ne_cm3: np.ndarray) -> np.ndarray:
    """Convert electron number density to gas mass density."""
    rho_g_cm3 = MU_E * M_P_G * np.asarray(ne_cm3)
    return rho_g_cm3 * (KPC_CM**3) / MSUN_G


def load_observed_deflection(cluster_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load observed deflection angle α(θ) from CLASH processed profiles.
    
    Returns:
        theta_arcsec: angular radius in arcsec
        alpha_arcsec: deflection angle in arcsec (derived from κ̄(<R))
    """
    cluster_id = cluster_name.lower().replace('macs', 'macs')
    profile_path = CLASH_DIR / 'profiles' / f'{cluster_id}_kappa_profile.csv'
    
    if not profile_path.exists():
        raise FileNotFoundError(f"Observed profile not found: {profile_path}")
    
    theta = []
    kappa_mean = []
    
    with open(profile_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = row['radius_arcsec']
            km = row['kappa_mean_within']
            if r and km:
                try:
                    theta.append(float(r))
                    kappa_mean.append(float(km))
                except ValueError:
                    continue
    
    theta = np.array(theta)
    kappa_mean = np.array(kappa_mean)
    
    # Deflection angle: α(θ) = θ * κ̄(<θ)
    alpha = theta * kappa_mean
    
    return theta, alpha


def load_cluster_baryon_profile(cluster_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load baryon density profile from gas + stars data.
    
    Returns:
        r_kpc: 3D radius in kpc
        rho_Msun_kpc3: total baryon density (gas + stars)
    """
    import pandas as pd
    
    # Use data_dir from specs if available
    data_dir_name = CLUSTER_SPECS.get(cluster_name, {}).get('data_dir', cluster_name)
    cdir = CLUSTER_DATA_DIR / data_dir_name
    
    # Load gas profile
    gas_path = cdir / 'gas_profile.csv'
    if not gas_path.exists():
        raise FileNotFoundError(f"Gas profile not found: {gas_path}")
    
    g = pd.read_csv(gas_path)
    r_g = np.asarray(g['r_kpc'], float)
    
    if 'rho_gas_Msun_per_kpc3' in g.columns:
        rho_gas = np.asarray(g['rho_gas_Msun_per_kpc3'], float)
    elif 'n_e_cm3' in g.columns:
        rho_gas = ne_to_rho_gas_Msun_kpc3(np.asarray(g['n_e_cm3'], float))
    else:
        raise ValueError('gas_profile.csv missing required density column')
    
    # Load stars profile if available
    stars_path = cdir / 'stars_profile.csv'
    if stars_path.exists():
        s = pd.read_csv(stars_path)
        r_s = np.asarray(s['r_kpc'], float)
        rho_star = np.asarray(s['rho_star_Msun_per_kpc3'], float)
    else:
        r_s = r_g
        rho_star = np.zeros_like(r_g)
    
    # Unify grid
    r_all = np.union1d(r_g, r_s)
    rho_total = np.interp(r_all, r_g, rho_gas) + np.interp(r_all, r_s, rho_star)
    
    # Clean
    mask = np.isfinite(r_all) & np.isfinite(rho_total) & (r_all > 0)
    r = r_all[mask]
    rho = np.clip(rho_total[mask], 0.0, None)
    
    # Ensure sorted
    idx = np.argsort(r)
    return r[idx], rho[idx]


def abel_project_to_sigma(r_kpc: np.ndarray, rho_kpc3: np.ndarray, 
                          R_kpc: np.ndarray) -> np.ndarray:
    """Project 3D density to 2D surface density via Abel transform.
    
    Σ(R) = 2 ∫_R^∞ ρ(r) r / sqrt(r² - R²) dr
    
    Note: Requires r_kpc and rho_kpc3 to be sorted in ascending order by r.
    """
    from scipy.integrate import simpson
    
    # Ensure ascending order
    if len(r_kpc) > 1 and r_kpc[1] < r_kpc[0]:
        idx = np.argsort(r_kpc)
        r_kpc = r_kpc[idx]
        rho_kpc3 = rho_kpc3[idx]
    
    # Clip negative densities
    rho_kpc3 = np.maximum(rho_kpc3, 0.0)
    
    Sigma = np.zeros_like(R_kpc)
    for j, Rv in enumerate(R_kpc):
        # Integrate from Rv outward
        mask = r_kpc >= Rv
        rr = r_kpc[mask]
        rh = rho_kpc3[mask]
        if rr.size < 2:
            Sigma[j] = 0.0
            continue
        # Guard against numerical issues at Rv
        sqrt_term = np.sqrt(np.maximum(rr**2 - Rv**2, 1e-30))
        integrand = 2.0 * rh * rr / sqrt_term
        # Filter out infinities
        valid = np.isfinite(integrand)
        if not np.any(valid):
            Sigma[j] = 0.0
            continue
        Sigma[j] = simpson(integrand[valid], rr[valid])
    
    # Final check for non-negative and finite
    Sigma = np.maximum(Sigma, 0.0)
    Sigma = np.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    return Sigma  # Msun/kpc²


def deflection_from_sigma(R_kpc: np.ndarray, Sigma_pc2: np.ndarray, 
                          z_lens: float, z_source: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute deflection angle α(θ) from surface density profile.
    
    α(θ) = (4GM(<R))/(c²R) * (D_ls/D_s)
    where M(<R) = 2π ∫_0^R Σ(R') R' dR'
    
    Args:
        R_kpc: Radii in kpc
        Sigma_pc2: Surface density in Msun/pc² (NOTE: pc², not kpc²!)
        z_lens, z_source: Redshifts
    """
    from scipy.integrate import cumulative_trapezoid
    
    # Convert Sigma from pc² to kpc² for mass integration
    Sigma_kpc2 = Sigma_pc2 * 1e6  # Msun/kpc²
    
    # Projected enclosed mass
    integrand = 2 * np.pi * Sigma_kpc2 * R_kpc
    M_proj = cumulative_trapezoid(integrand, R_kpc, initial=0.0)
    
    # Angular diameter distances
    if ASTROPY_OK:
        D_l = COSMO.angular_diameter_distance(z_lens).to(u.kpc).value
        D_s = COSMO.angular_diameter_distance(z_source).to(u.kpc).value
        D_ls = COSMO.angular_diameter_distance_z1z2(z_lens, z_source).to(u.kpc).value
    else:
        # Fallback simple cosmology
        D_l = 1500.0  # kpc (rough)
        D_s = 3000.0
        D_ls = D_s - D_l
    
    # Deflection in radians
    alpha_rad = (4 * G * M_proj) / (c_km_s**2 * np.maximum(R_kpc, 1e-12)) * (D_ls / D_s)
    
    # Convert R to theta (arcsec) and alpha to arcsec
    theta_arcsec = (R_kpc / D_l) * (180.0/np.pi) * 3600.0
    alpha_arcsec = alpha_rad * (180.0/np.pi) * 3600.0
    
    return theta_arcsec, alpha_arcsec


# --- New helper functions for mean-Σ gating and scale-dependent response ---

def mean_sigma_inside_R(R_kpc, Sigma_pc2):
    """Compute mean surface density inside each radius R.
    
    Σ̄(<R) = M_proj(<R) / (π R²)
    
    Args:
        R_kpc: Radii in kpc
        Sigma_pc2: Surface density in Msun/pc²
    
    Returns:
        Mean surface density in Msun/pc²
    """
    # Convert to kpc² for integration
    Sigma_kpc2 = Sigma_pc2 * 1e6
    Menc = np.array([2*np.pi*np.trapezoid(Sigma_kpc2[:i+1]*R_kpc[:i+1], R_kpc[:i+1])
                     for i in range(len(R_kpc))])
    # Return in pc²
    return Menc / (np.pi * np.maximum(R_kpc, 1e-9)**2) / 1e6


def logistic(x, x0=0.3, w=0.3):
    """Logistic sigmoid function."""
    return 1.0/(1.0 + np.exp(-(x - x0)/w))


def slip_meanSigma_gate(R_kpc, Sigma_pc2,
                        S_inf=10.0, Rs_kpc=100.0, p=1.2,
                        Sigma0_pc2=100.0, x0=0.3, w=0.3,
                        cap=50.0):
    """Compute slip factor S(R) based on mean surface density inside R.
    
    S(R) = 1 + S_∞ * [1 - exp(-(R/Rs)^p)] * g(R)
    
    where g(R) is a logistic gate on log(Σ̄(<R)/Σ₀) that turns on when
    the mean surface density drops (void interface).
    
    Args:
        R_kpc: Radii in kpc
        Sigma_pc2: Surface density in Msun/pc²
        S_inf: Outer amplitude (asymptotic boost)
        Rs_kpc: Scale radius for radial ramp
        p: Power for radial ramp
        Sigma0_pc2: Reference surface density in Msun/pc²
        x0: Logistic center for Σ̂ transition
        w: Logistic width
        cap: Maximum slip value
    
    Returns:
        S(R): Slip multiplier (≥1, monotonic)
    """
    # Gate on mean Σ inside R (both in pc²)
    Sigma_bar_pc2 = mean_sigma_inside_R(R_kpc, Sigma_pc2)
    Shat = np.log10(np.maximum(Sigma_bar_pc2, 1e-8) / Sigma0_pc2)
    g = 1.0 - logistic(Shat, x0=x0, w=w)  # near 0 in cores, →1 in outskirts
    
    # Radial ramp
    ramp = 1.0 - np.exp(-(np.maximum(R_kpc, 1e-6)/Rs_kpc)**p)
    
    # Combined slip
    S = 1.0 + S_inf * ramp * g
    S = np.clip(S, 1.0, cap)
    S = np.maximum.accumulate(S)  # enforce monotonic export
    
    return S


def epsilon_running(R_kpc, Sigma_pc2, eps0=15.0, Rs_kpc=120.0, p=1.2,
                    Sigma0_pc2=100.0, x0=0.3, w=0.3):
    """Compute scale-dependent response coupling ε(R).
    
    ε(R) = ε₀ * [1 - exp(-(R/Rs)^p)] * g(R)
    
    where g(R) uses the same mean-Σ gate as the slip.
    
    Args:
        R_kpc: Radii in kpc
        Sigma_pc2: Surface density in Msun/pc²
    """
    Sigma_bar_pc2 = mean_sigma_inside_R(R_kpc, Sigma_pc2)
    Shat = np.log10(np.maximum(Sigma_bar_pc2, 1e-8) / Sigma0_pc2)
    g = 1.0 - logistic(Shat, x0=x0, w=w)
    ramp = 1.0 - np.exp(-(np.maximum(R_kpc, 1e-6)/Rs_kpc)**p)
    return eps0 * ramp * g


def build_sigma_response(R_kpc, Sigma_pc2, lam_kpc=150.0, nu=1.8):
    """Build response halo via power-tail kernel convolution.
    
    K(ΔR) ∝ (1 + ΔR/λ)^(-ν)
    
    Σ_resp(R) = ∫ K(|R - R'|) Σ(R') 2π R' dR' / normalization
    
    Args:
        R_kpc: Radii in kpc
        Sigma_pc2: Surface density in Msun/pc²
    
    Returns:
        Response surface density in Msun/pc²
    """
    R = np.asarray(R_kpc)
    S = np.asarray(Sigma_pc2)
    
    # Distance matrix
    dR = np.abs(R[:, None] - R[None, :])
    
    # Power-tail kernel
    K = np.power(1.0 + dR/lam_kpc, -nu)
    
    # Weights for integration
    dRj = np.gradient(R)
    wts = 2.0 * np.pi * (R[None, :]) * (dRj[None, :])
    
    # Convolve
    num = (K * S[None, :] * wts).sum(axis=1)
    denom = np.maximum((K * wts).sum(axis=1), 1e-30)
    
    return num / denom


def build_sigma_response_DoG(R_kpc, Sigma_pc2,
                             lam1=70.0, lam2=220.0, nu1=1.8, nu2=1.8, beta=0.6):
    """Build band-pass response via Difference of Gaussians (DoG) kernel.
    
    K = K₂ - β*K₁
    where K_i(ΔR) = (1 + ΔR/λᵢ)^(-νᵢ)
    
    This produces a ring-like enhancement (dip at mid-radii, rise at outer radii).
    
    Args:
        R_kpc: Radii in kpc
        Sigma_pc2: Surface density in Msun/pc²
    
    Returns:
        DoG response surface density in Msun/pc²
    """
    R = np.asarray(R_kpc)
    S = np.asarray(Sigma_pc2)
    
    dR = np.abs(R[:, None] - R[None, :])
    
    K1 = np.power(1.0 + dR/lam1, -nu1)
    K2 = np.power(1.0 + dR/lam2, -nu2)
    K = K2 - beta*K1
    
    dRj = np.gradient(R)
    wts = 2.0 * np.pi * (R[None, :]) * (dRj[None, :])
    
    num = (K * S[None, :] * wts).sum(axis=1)
    denom = np.maximum((K * wts).sum(axis=1), 1e-30)
    
    return num / denom


def compare_cluster(cluster_name: str, output_plots: bool = True) -> Dict:
    """Compare observed lensing with GR baryons + geometry-tied models.
    
    Args:
        cluster_name: e.g., 'MACS0416'
        output_plots: Whether to generate and save plots
    
    Returns:
        Dictionary with comparison metrics
    """
    print(f"\n{'='*70}")
    print(f"Processing {cluster_name}")
    print(f"{'='*70}")
    
    spec = CLUSTER_SPECS[cluster_name]
    z_lens = spec['z_lens']
    z_source = spec['z_source']
    theta_E_obs = spec['theta_E_obs']
    
    # Load observed deflection from CLASH
    try:
        theta_obs, alpha_obs = load_observed_deflection(cluster_name)
        print(f"✓ Loaded observed deflection: {len(theta_obs)} points")
    except FileNotFoundError as e:
        print(f"✗ Could not load observed data: {e}")
        return {'error': str(e)}
    
    # Load baryon density profile
    try:
        r, rho_baryons = load_cluster_baryon_profile(cluster_name)
        print(f"✓ Loaded baryon profile: {len(r)} points, r = {r[0]:.1f} to {r[-1]:.1f} kpc")
    except FileNotFoundError as e:
        print(f"✗ Could not load baryon profile: {e}")
        return {'error': str(e)}
    
    # Compute surface density via Abel projection
    # Subsample the radial grid for projection (too many points cause numerical issues)
    n_proj = min(500, len(r))
    if len(r) > n_proj:
        # Use log-spaced sampling to preserve structure at all scales
        r_proj_idx = np.unique(np.logspace(0, np.log10(len(r)-1), n_proj).astype(int))
        r_proj = r[r_proj_idx]
        rho_proj = rho_baryons[r_proj_idx]
    else:
        r_proj = r
        rho_proj = rho_baryons
    
    print(f"  Projecting {len(r_proj)} radial points...")
    R = r_proj.copy()
    Sigma_baryons_kpc2 = abel_project_to_sigma(r_proj, rho_proj, R)
    # Convert to pc² for lensing calculations (1 kpc = 1000 pc, so 1 kpc² = 10^6 pc²)
    Sigma_baryons = Sigma_baryons_kpc2 / 1e6  # now in Msun/pc²
    print(f"✓ Computed Σ_baryons: range {Sigma_baryons.min():.2e} to {Sigma_baryons.max():.2e} Msun/pc²")
    print(f"  Typical values: Σ(50 kpc) = {np.interp(50, R, Sigma_baryons):.2e}, Σ(200 kpc) = {np.interp(200, R, Sigma_baryons):.2e} Msun/pc²")
    
    # Compute GR baseline deflection
    theta_R_arcsec, alpha_gr = deflection_from_sigma(R, Sigma_baryons, z_lens, z_source)
    alpha_gr_50 = np.interp(50.0, theta_R_arcsec, alpha_gr) if len(theta_R_arcsec) > 0 else 0.0
    print(f"✓ Computed GR deflection: α_GR(50\") = {alpha_gr_50:.3f} arcsec")
    
    # Sanity check: GR deflection should be much smaller than observed for baryons alone
    if alpha_gr_50 > 1000.0:
        print(f"  WARNING: GR deflection seems unreasonably large! Check data units.")
        print(f"  This suggests a problem with the baryon surface density calculation.")
    
    # Interpolate observed and GR onto common grid for comparison
    theta_min = max(theta_obs.min(), theta_R_arcsec.min(), 1.0)
    theta_max = min(theta_obs.max(), theta_R_arcsec.max(), 200.0)
    theta_grid = np.linspace(theta_min, theta_max, 300)
    
    alpha_obs_interp = np.interp(theta_grid, theta_obs, alpha_obs)
    alpha_gr_interp = np.interp(theta_grid, theta_R_arcsec, alpha_gr)
    
    # --- NEW: Mean-Σ gated slip ---
    # Calibrate S_inf at 50" reference point
    theta_ref = 50.0
    idx_ref = np.argmin(np.abs(theta_grid - theta_ref))
    S_inf_guess = max(alpha_obs_interp[idx_ref]/max(alpha_gr_interp[idx_ref], 1e-6) - 1.0, 0.0)
    
    print(f"\n--- Mean-Σ Gated Slip ---")
    print(f"  Calibrated S_inf from 50\" anchor: {S_inf_guess:.1f}")
    
    # Cluster-specific parameters (from your recommendations)
    if cluster_name == 'MACS0416':
        S_inf = S_inf_guess if S_inf_guess > 0 else 25.0
        Rs_kpc = 100.0
        p = 1.2
        x0 = 0.3
        w = 0.3
    elif cluster_name == 'MACS0717':
        S_inf = S_inf_guess if S_inf_guess > 0 else 6.0
        Rs_kpc = 100.0
        p = 1.2
        x0 = 0.3
        w = 0.3
    elif cluster_name == 'MACS1149':
        S_inf = S_inf_guess if S_inf_guess > 0 else 3.0
        Rs_kpc = 90.0
        p = 1.2
        x0 = 0.3
        w = 0.3
    else:
        S_inf = S_inf_guess if S_inf_guess > 0 else 10.0
        Rs_kpc = 100.0
        p = 1.2
        x0 = 0.3
        w = 0.3
    
    S_mean_R = slip_meanSigma_gate(R, Sigma_baryons,
                                   S_inf=S_inf, Rs_kpc=Rs_kpc, p=p,
                                   Sigma0_pc2=100.0, x0=x0, w=w, cap=50.0)
    S_mean_theta = np.interp(theta_grid, theta_R_arcsec, S_mean_R,
                             left=S_mean_R[0], right=S_mean_R[-1])
    alpha_slip_mean = alpha_gr_interp * S_mean_theta
    
    print(f"  S_mean(50\"): {np.interp(50.0, theta_grid, S_mean_theta):.2f}")
    print(f"  α_slip(50\"): {np.interp(50.0, theta_grid, alpha_slip_mean):.2f} vs obs {np.interp(50.0, theta_grid, alpha_obs_interp):.2f}")
    
    # --- NEW: Running-ε response (single-kernel) ---
    print(f"\n--- Scale-Dependent Response Halo ---")
    
    if cluster_name == 'MACS0416':
        eps0 = 20.0
        Rs_resp = 120.0
        lam_kpc = 200.0
        nu = 1.8
    elif cluster_name == 'MACS0717':
        eps0 = 25.0
        Rs_resp = 120.0
        lam_kpc = 200.0
        nu = 1.8
    elif cluster_name == 'MACS1149':
        eps0 = 12.0
        Rs_resp = 100.0
        lam_kpc = 150.0
        nu = 1.8
    else:
        eps0 = 15.0
        Rs_resp = 120.0
        lam_kpc = 150.0
        nu = 1.8
    
    eps_R = epsilon_running(R, Sigma_baryons, eps0=eps0, Rs_kpc=Rs_resp, p=p,
                            Sigma0_pc2=100.0, x0=x0, w=w)
    Sigma_resp = build_sigma_response(R, Sigma_baryons, lam_kpc=lam_kpc, nu=nu)
    Sigma_eff_run = Sigma_baryons + eps_R * Sigma_resp
    theta_R_run, alpha_R_run = deflection_from_sigma(R, Sigma_eff_run, z_lens, z_source)
    alpha_resp_run = np.interp(theta_grid, theta_R_run, alpha_R_run, 
                               left=alpha_R_run[0], right=alpha_R_run[-1])
    
    print(f"  ε(50\"): {np.interp(50.0, theta_R_arcsec, eps_R):.2f}")
    print(f"  α_resp_run(50\"): {np.interp(50.0, theta_grid, alpha_resp_run):.2f}")
    
    # --- NEW: DoG response for MACS0717 (band-pass for dip-and-rise) ---
    if cluster_name == 'MACS0717':
        print(f"\n--- DoG Band-Pass Response (MACS0717 special) ---")
        lam1 = 75.0
        lam2 = 240.0
        beta_dog = 0.6
        nu1 = nu2 = 1.8
        
        Sigma_DoG = build_sigma_response_DoG(R, Sigma_baryons, 
                                            lam1=lam1, lam2=lam2, 
                                            nu1=nu1, nu2=nu2, beta=beta_dog)
        Sigma_eff_dog = Sigma_baryons + eps_R * Sigma_DoG
        theta_R_dog, alpha_R_dog = deflection_from_sigma(R, Sigma_eff_dog, z_lens, z_source)
        alpha_resp_dog = np.interp(theta_grid, theta_R_dog, alpha_R_dog,
                                   left=alpha_R_dog[0], right=alpha_R_dog[-1])
        print(f"  DoG parameters: λ₁={lam1}, λ₂={lam2}, β={beta_dog}")
        print(f"  α_resp_DoG(50\"): {np.interp(50.0, theta_grid, alpha_resp_dog):.2f}")
    else:
        alpha_resp_dog = None
    
    # --- Compute errors at key radii ---
    test_radii = [20.0, 50.0, 100.0]
    errors = {}
    
    for rad in test_radii:
        obs_val = np.interp(rad, theta_grid, alpha_obs_interp)
        gr_val = np.interp(rad, theta_grid, alpha_gr_interp)
        slip_val = np.interp(rad, theta_grid, alpha_slip_mean)
        resp_val = np.interp(rad, theta_grid, alpha_resp_run)
        
        errors[f'{rad}arcsec'] = {
            'observed': float(obs_val),
            'gr': float(gr_val),
            'slip_mean': float(slip_val),
            'resp_run': float(resp_val),
            'err_gr': float(abs(gr_val - obs_val)/obs_val) if obs_val > 0 else float('nan'),
            'err_slip': float(abs(slip_val - obs_val)/obs_val) if obs_val > 0 else float('nan'),
            'err_resp': float(abs(resp_val - obs_val)/obs_val) if obs_val > 0 else float('nan'),
        }
        
        if cluster_name == 'MACS0717' and alpha_resp_dog is not None:
            dog_val = np.interp(rad, theta_grid, alpha_resp_dog)
            errors[f'{rad}arcsec']['resp_dog'] = float(dog_val)
            errors[f'{rad}arcsec']['err_dog'] = float(abs(dog_val - obs_val)/obs_val) if obs_val > 0 else float('nan')
    
    print(f"\n--- Relative Errors at Key Radii ---")
    for rad_key, vals in errors.items():
        print(f"  {rad_key}:")
        print(f"    Observed: {vals['observed']:.3f} arcsec")
        print(f"    GR error: {vals['err_gr']*100:.1f}%")
        print(f"    Slip error: {vals['err_slip']*100:.1f}%")
        print(f"    Response error: {vals['err_resp']*100:.1f}%")
        if 'err_dog' in vals:
            print(f"    DoG error: {vals['err_dog']*100:.1f}%")
    
    # --- Plotting ---
    if output_plots:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cluster_out = OUT_DIR / cluster_name
        cluster_out.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Observed data (blue)
        ax.plot(theta_obs, alpha_obs, 'o-', color='#1f77b4', linewidth=2.5, 
                markersize=4, label='Observed (CLASH)', zorder=10)
        
        # GR baseline (gray)
        ax.plot(theta_grid, alpha_gr_interp, '--', color='gray', linewidth=2,
                label='GR Baryons Only', alpha=0.7)
        
        # Mean-Σ slip (orange)
        ax.plot(theta_grid, alpha_slip_mean, '-', color='#ff7f0e', linewidth=2,
                label=f'Mean-Σ Slip (S∞={S_inf:.1f})', alpha=0.9)
        
        # Response run (green)
        ax.plot(theta_grid, alpha_resp_run, '-', color='#2ca02c', linewidth=2,
                label=f'Response Halo (ε₀={eps0:.0f})', alpha=0.9)
        
        # DoG for MACS0717 (purple)
        if cluster_name == 'MACS0717' and alpha_resp_dog is not None:
            ax.plot(theta_grid, alpha_resp_dog, '-', color='#9467bd', linewidth=2,
                    label='DoG Band-Pass', alpha=0.9)
        
        ax.set_xlabel('Angular Radius θ (arcsec)', fontsize=12)
        ax.set_ylabel('Deflection Angle α (arcsec)', fontsize=12)
        ax.set_title(f'{cluster_name}: Lensing Deflection Comparison\n'
                    f'z_lens={z_lens:.3f}, z_source={z_source:.1f}', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(theta_min, theta_max)
        
        # Mark Einstein radius
        ax.axvline(theta_E_obs, color='red', linestyle=':', linewidth=1.5, 
                  alpha=0.6, label=f'θ_E obs = {theta_E_obs:.0f}\"')
        
        fig.tight_layout()
        plot_path = cluster_out / f'{cluster_name}_deflection_comparison.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved plot: {plot_path}")
        plt.close(fig)
    
    # --- Save summary ---
    summary = {
        'cluster': cluster_name,
        'z_lens': z_lens,
        'z_source': z_source,
        'theta_E_obs': theta_E_obs,
        'parameters': {
            'slip': {'S_inf': float(S_inf), 'Rs_kpc': float(Rs_kpc), 'p': float(p), 'x0': float(x0), 'w': float(w)},
            'response': {'eps0': float(eps0), 'Rs_kpc': float(Rs_resp), 'lam_kpc': float(lam_kpc), 'nu': float(nu)},
        },
        'errors': errors,
    }
    
    if cluster_name == 'MACS0717':
        summary['parameters']['dog'] = {
            'lam1': float(lam1), 'lam2': float(lam2), 
            'beta': float(beta_dog), 'nu1': float(nu1), 'nu2': float(nu2)
        }
    
    summary_path = cluster_out / f'{cluster_name}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Compare observed cluster lensing with GR baryons + geometry-tied models'
    )
    parser.add_argument('--cluster', type=str, default='all',
                       choices=['all', 'MACS0416', 'MACS0717', 'MACS1149'],
                       help='Cluster to process (default: all)')
    args = parser.parse_args()
    
    if args.cluster == 'all':
        clusters = list(CLUSTER_SPECS.keys())
    else:
        clusters = [args.cluster]
    
    all_summaries = {}
    
    for cluster in clusters:
        try:
            summary = compare_cluster(cluster, output_plots=True)
            all_summaries[cluster] = summary
        except Exception as e:
            print(f"\n✗ ERROR processing {cluster}: {e}")
            import traceback
            traceback.print_exc()
            all_summaries[cluster] = {'error': str(e)}
    
    # Save aggregate summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg_path = OUT_DIR / 'all_clusters_summary.json'
    with open(agg_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n{'='*70}")
    print(f"✓ Saved aggregate summary: {agg_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
