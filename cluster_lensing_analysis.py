#!/usr/bin/env python3
"""
Cluster Lensing Analysis for Universal G³ Model
================================================

This script performs an academic-level gravitational lensing analysis using the
current Universal G³ model (no per-object tuning). It computes:
- Effective enclosed mass from the predicted total acceleration g_total(r)
- 3D effective density via numerical differentiation
- Projected surface density Σ(R) via Abel transform
- Lensing convergence κ(R), mean convergence \bar{κ}(<R), and tangential shear γ_t(R)
- Einstein radius θ_E solving \bar{κ}(<R_E)=1 (or M_proj(<R_E) = π R_E^2 Σ_crit)

It provides a transparent comparison between:
- Baryonic-only lensing (from Σ_bar)
- G³-effective lensing (from Σ_eff corresponding to g_total)

Outputs:
- Plots under out/cluster_lensing/<cluster_name>/*.png
- JSON summary with key lensing metrics

Notes:
- Cosmology: flat ΛCDM with H0=70 km/s/Mpc, Ω_m=0.3 (no external deps)
- Distances are angular-diameter distances in kpc
- Units: G taken from project conventions (kpc km^2 s^-2 Msun^-1)
- Assumes spherical symmetry for Abel projection

"""

import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# Project-local G constant and a copy of the Universal G³ model
G = 4.300917270e-6  # kpc km^2 s^-2 Msun^-1
c_km_s = 299792.458  # km/s

# Default parameters; will be overwritten by sweep best.json if present
UNIVERSAL_PARAMS = {
    "v0_kms": 237.27,
    "rc0_kpc": 28.98,
    "gamma": 0.77,
    "beta": 0.22,
    "sigma_star": 84.49,
    "alpha": 1.04,
    "kappa": 1.75,
    "eta": 0.96,
    "delta_kpc": 1.50,
    "p_in": 1.71,
    "p_out": 0.88,
    "g_sat": 2819.45
}
BEST_VARIANT = {
    "gating_type": "rational",
    "screen_type": "sigmoid",
    "exponent_type": "logistic_r"
}

class UniversalG3Model:
    def __init__(self, params=None, variant=None):
        self.params = params or UNIVERSAL_PARAMS
        self.variant = variant or BEST_VARIANT

    def predict_v(self, r_kpc: np.ndarray, v_bar: np.ndarray, r_half: float, sigma_bar: float) -> np.ndarray:
        p = self.params
        r = r_kpc
        # Exponential disk approximation for sigma_loc
        r_d = r_half / 1.68
        sigma_0 = sigma_bar * 2.0
        sigma_loc = sigma_0 * np.exp(-r / (r_d + 1e-12))

        # Effective core radius scaling
        rc_eff = p["rc0_kpc"] * (r_half / 8.0)**p["gamma"] * (sigma_bar / 100.0)**(-p["beta"])

        # Variable exponent (logistic_r)
        transition_r = p["eta"] * r_half
        x = (r - transition_r) / (p["delta_kpc"] + 1e-12)
        gate_exp = 1.0 / (1.0 + np.exp(-x))
        p_r = p["p_in"] * (1 - gate_exp) + p["p_out"] * gate_exp

        # Gating (rational)
        with np.errstate(divide='ignore', invalid='ignore'):
            gate = (r**p_r) / (r**p_r + rc_eff**p_r + 1e-20)

        # Screening (sigmoid)
        screen = 1.0 / (1.0 + (sigma_loc / (p["sigma_star"] + 1e-12))**p["alpha"])**p["kappa"]

        # Tail acceleration
        g_tail = (p["v0_kms"]**2 / (r + 1e-12)) * gate * screen
        g_tail = p["g_sat"] * np.tanh(g_tail / (p["g_sat"] + 1e-12))

        g_bar = v_bar**2 / (r + 1e-12)
        g_total = g_bar + g_tail
        v_pred = np.sqrt(np.abs(g_total * r))
        return np.nan_to_num(v_pred, nan=0.0, posinf=1e4)

    def predict_g(self, r_kpc: np.ndarray, v_bar: np.ndarray, r_half: float, sigma_bar: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (g_bar, g_tail, g_total) in km^2 s^-2 kpc^-1."""
        p = self.params
        r = r_kpc
        # Exponential disk approximation for sigma_loc
        r_d = r_half / 1.68
        sigma_0 = sigma_bar * 2.0
        sigma_loc = sigma_0 * np.exp(-r / (r_d + 1e-12))

        # Effective core radius scaling
        rc_eff = p["rc0_kpc"] * (r_half / 8.0)**p["gamma"] * (sigma_bar / 100.0)**(-p["beta"])

        # Variable exponent (logistic_r)
        transition_r = p["eta"] * r_half
        x = (r - transition_r) / (p["delta_kpc"] + 1e-12)
        gate_exp = 1.0 / (1.0 + np.exp(-x))
        p_r = p["p_in"] * (1 - gate_exp) + p["p_out"] * gate_exp

        # Gating (rational)
        with np.errstate(divide='ignore', invalid='ignore'):
            gate = (r**p_r) / (r**p_r + rc_eff**p_r + 1e-20)

        # Screening (sigmoid)
        screen = 1.0 / (1.0 + (sigma_loc / (p["sigma_star"] + 1e-12))**p["alpha"])**p["kappa"]

        # Tail acceleration
        g_tail = (p["v0_kms"]**2 / (r + 1e-12)) * gate * screen
        g_tail = p["g_sat"] * np.tanh(g_tail / (p["g_sat"] + 1e-12))

        g_bar = v_bar**2 / (r + 1e-12)
        g_total = g_bar + g_tail
        return g_bar, g_tail, g_total

# Simple cosmology utilities (flat LCDM)
H0 = 70.0  # km/s/Mpc
Omega_m = 0.3
Omega_L = 1.0 - Omega_m
Mpc_to_kpc = 1_000.0


def Ez(z):
    """E(z) function for flat LCDM cosmology, handles scalar or array input."""
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_L)


def comoving_distance_Mpc(z: float, n: int = 2048) -> float:
    """Simple numeric integral for comoving distance (flat)."""
    if z <= 0:
        return 0.0
    zs = np.linspace(0.0, z, n)
    integrand = 1.0 / Ez(zs)
    Dc = (c_km_s / H0) * np.trapezoid(integrand, zs)  # in Mpc
    return Dc


def angular_diameter_distance_kpc(z: float) -> float:
    Dc = comoving_distance_Mpc(z)
    Da = Dc / (1 + z)  # Mpc
    return Da * Mpc_to_kpc


def sigma_crit_Msun_per_kpc2(z_d: float, z_s: float) -> float:
    if z_s <= z_d:
        return np.inf
    Dd = angular_diameter_distance_kpc(z_d)
    Ds = angular_diameter_distance_kpc(z_s)
    Dds = angular_diameter_distance_kpc(z_s) - angular_diameter_distance_kpc(z_d)  # flat geometry: Dc additive
    # Convert Dds to angular diameter: Dds_ang = Dds / (1+z_s)
    # But since we computed in kpc, use the angular form directly via Da's:
    # In flat cosmology, Dds_ang = (1/(1+z_s)) (Dc(z_s)-Dc(z_d))
    Dc_s = comoving_distance_Mpc(z_s) * Mpc_to_kpc
    Dc_d = comoving_distance_Mpc(z_d) * Mpc_to_kpc
    Dds_ang = (Dc_s - Dc_d) / (1 + z_s)
    # Critical density Σ_crit = c^2/(4πG) * Ds/(Dd*Dds)
    return (c_km_s**2) / (4 * math.pi * G) * (Ds / (Dd * max(Dds_ang, 1e-12)))

# Spherical mass/Abel tools

def enclosed_mass_to_density(r: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
    """Compute ρ(r) = (1/4πr^2) dM/dr from enclosed mass profile.
    r: kpc, M_enc: Msun. Returns ρ in Msun/kpc^3.
    """
    dM_dr = np.gradient(M_enc, r, edge_order=2)
    rho = dM_dr / (4 * math.pi * np.clip(r, 1e-9, None)**2)
    rho = np.maximum(rho, 0.0)  # enforce non-negative
    return rho


def abel_project_sigma(r: np.ndarray, rho: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Project spherical density ρ(r) to Σ(R) using Abel integral:
    Σ(R) = 2 ∫_R^∞ ρ(r) r / sqrt(r^2 - R^2) dr.
    Numeric evaluation via discrete integration per R.
    """
    # Ensure monotonic grids
    r = np.asarray(r)
    rho = np.asarray(rho)
    R = np.asarray(R)
    Sigma = np.zeros_like(R)

    # Precompute for speed
    for i, Rp in enumerate(R):
        mask = r >= max(Rp, r[0])
        rr = r[mask]
        rh = rho[mask]
        if rr.size < 2:
            Sigma[i] = 0.0
            continue
        integrand = rh * rr / np.sqrt(np.clip(rr**2 - Rp**2, 1e-20, None))
        Sigma[i] = 2.0 * np.trapezoid(integrand, rr)
    return Sigma


def sigma_to_Mproj(R: np.ndarray, Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cumulative projected mass M_proj(<R) and mean Σbar(<R)."""
    # Cumulative integral: M_proj(<R) = 2π ∫_0^R Σ(R') R' dR'
    Mproj = 2 * math.pi * np.cumsum(Sigma * R) * np.gradient(R)
    # Fix integration via more stable trapezoid cumulative
    Mproj = np.array([2 * math.pi * np.trapezoid(Sigma[:i+1] * R[:i+1], R[:i+1]) for i in range(len(R))])
    area = math.pi * R**2
    Sbar = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area>0)
    return Mproj, Sbar

# Cluster baryon model (simplified), used only to seed v_bar
@dataclass
class ClusterSpec:
    name: str
    z_lens: float
    z_source: float
    r200_kpc: float
    M200_Msun: float
    rhalf_kpc: float  # used by G³ model
    sigma_bar: float  # Msun/pc^2 average surface density proxy


def baryon_mass_profile(r: np.ndarray, r200: float, M200: float) -> np.ndarray:
    """Simplified baryon mass profile: scaled NFW-like baryon-only
    so that M_bar(r200) ~ f_b * M200 with f_b ~ 0.15 (cosmic baryon fraction).
    """
    f_b = 0.15
    Mb_tot = f_b * M200
    # Use an NFW cumulative form with reduced concentration for baryons
    c = 3.0
    rs = r200 / c
    # NFW enclosed mass (normalized to Mb_tot at r200)
    def m_nfw(x):
        return np.log(1 + x) - x / (1 + x)
    norm = m_nfw(c)
    M_enc = Mb_tot * m_nfw(r / rs) / max(norm, 1e-12)
    # Guard
    M_enc = np.clip(M_enc, 0, Mb_tot)
    return M_enc


def compute_lensing_for_cluster(spec: ClusterSpec, model: UniversalG3Model, outdir: Path) -> Dict:
    # Radial and projected grids
    r = np.logspace(np.log10(5), np.log10(spec.r200_kpc), 800)  # 5 kpc to r200
    R = np.logspace(np.log10(1), np.log10(spec.r200_kpc), 600)

    # Baryon mass and velocity
    M_bar = baryon_mass_profile(r, spec.r200_kpc, spec.M200_Msun)
    v_bar = np.sqrt(G * np.maximum(M_bar, 0.0) / np.maximum(r, 1e-12))

    # G³ accelerations
    g_bar, g_tail, g_total = model.predict_g(r, v_bar, spec.rhalf_kpc, spec.sigma_bar)

    # Effective enclosed mass from g_total
    M_eff = g_total * r**2 / G  # Msun

    # Densities via derivative
    rho_bar = enclosed_mass_to_density(r, M_bar)
    rho_eff = enclosed_mass_to_density(r, M_eff)

    # Project to Σ(R)
    Sigma_bar = abel_project_sigma(r, rho_bar, R)
    Sigma_eff = abel_project_sigma(r, rho_eff, R)

    # Lensing quantities
    Sigma_crit = sigma_crit_Msun_per_kpc2(spec.z_lens, spec.z_source)
    kappa_bar = Sigma_bar / Sigma_crit
    kappa_eff = Sigma_eff / Sigma_crit

    Mproj_bar, Sbar_bar = sigma_to_Mproj(R, Sigma_bar)
    Mproj_eff, Sbar_eff = sigma_to_Mproj(R, Sigma_eff)

    kappa_bar_mean = Sbar_bar / Sigma_crit
    kappa_eff_mean = Sbar_eff / Sigma_crit

    # Einstein radius where mean kappa crosses 1
    def find_theta_E(R_kpc, kappa_mean):
        idx = np.where(kappa_mean >= 1.0)[0]
        if idx.size == 0:
            return None
        i = idx[0]
        if i == 0:
            R_E = R_kpc[0]
        else:
            # linear interpolation in (R, kappa_mean)
            x0, y0 = R_kpc[i-1], kappa_mean[i-1]
            x1, y1 = R_kpc[i], kappa_mean[i]
            if y1 == y0:
                R_E = x1
            else:
                R_E = x0 + (1 - y0) * (x1 - x0) / (y1 - y0)
        # Convert to arcsec: theta = R / D_d
        Dd = angular_diameter_distance_kpc(spec.z_lens)
        theta_rad = R_E / max(Dd, 1e-12)
        theta_arcsec = theta_rad * (180/math.pi) * 3600
        return R_E, theta_arcsec

    RE_bar = find_theta_E(R, kappa_bar_mean)
    RE_eff = find_theta_E(R, kappa_eff_mean)

    # Tangential shear γ_t = \bar{κ} - κ
    gamma_bar = kappa_bar_mean - kappa_bar
    gamma_eff = kappa_eff_mean - kappa_eff

    # Deflection angle profiles (arcsec): α(θ) = 4GM_proj(<R)/(c^2 R) * D_ds/D_s
    Dc_s = comoving_distance_Mpc(spec.z_source) * Mpc_to_kpc
    Dc_d = comoving_distance_Mpc(spec.z_lens) * Mpc_to_kpc
    Dds_ang = (Dc_s - Dc_d) / (1 + spec.z_source)
    Ds_ang = angular_diameter_distance_kpc(spec.z_source)
    geom = max(Dds_ang / max(Ds_ang, 1e-12), 0.0)
    alpha_bar_rad = 4 * G * Mproj_bar / (c_km_s**2 * np.maximum(R, 1e-12)) * geom
    alpha_eff_rad = 4 * G * Mproj_eff / (c_km_s**2 * np.maximum(R, 1e-12)) * geom
    alpha_bar_arcsec = alpha_bar_rad * (180/math.pi) * 3600
    alpha_eff_arcsec = alpha_eff_rad * (180/math.pi) * 3600

    # Plots
    outdir.mkdir(parents=True, exist_ok=True)

    # κ and γ profiles
    fig1, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].loglog(R, kappa_bar, label='κ_bar (baryons only)', lw=2)
    ax[0].loglog(R, kappa_eff, label='κ_eff (G³ effective)', lw=2)
    ax[0].loglog(R, kappa_bar_mean, '--', label='\u0304κ_bar(<R)')
    ax[0].loglog(R, kappa_eff_mean, '--', label='\u0304κ_eff(<R)')
    ax[0].axhline(1.0, color='k', ls=':', lw=1)
    ax[0].set_xlabel('R (kpc)')
    ax[0].set_ylabel('Convergence κ')
    ax[0].set_title(f'{spec.name}: Convergence profiles')
    ax[0].grid(True, which='both', alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].loglog(R, gamma_bar, label='γ_t (baryons)', lw=2)
    ax[1].loglog(R, gamma_eff, label='γ_t (G³ effective)', lw=2)
    ax[1].set_xlabel('R (kpc)')
    ax[1].set_ylabel('Tangential shear γ_t')
    ax[1].set_title(f'{spec.name}: Tangential shear')
    ax[1].grid(True, which='both', alpha=0.3)
    ax[1].legend(fontsize=8)
    fig1.tight_layout()
    fig1.savefig(outdir / 'kappa_gamma_profiles.png', dpi=150, bbox_inches='tight')

    # Deflection
    fig2, ax2 = plt.subplots(figsize=(6,5))
    ax2.loglog(R, alpha_bar_arcsec, label='α (baryons)', lw=2)
    ax2.loglog(R, alpha_eff_arcsec, label='α (G³ effective)', lw=2)
    ax2.set_xlabel('R (kpc)')
    ax2.set_ylabel('Deflection α (arcsec)')
    ax2.set_title(f'{spec.name}: Deflection angle')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(outdir / 'deflection_angle.png', dpi=150, bbox_inches='tight')

    # Surface density
    fig3, ax3 = plt.subplots(figsize=(6,5))
    ax3.loglog(R, Sigma_bar, label='Σ_bar', lw=2)
    ax3.loglog(R, Sigma_eff, label='Σ_eff (G³ effective)', lw=2)
    ax3.axhline(Sigma_crit, color='k', ls=':', label='Σ_crit')
    ax3.set_xlabel('R (kpc)')
    ax3.set_ylabel('Surface density Σ (Msun/kpc^2)')
    ax3.set_title(f'{spec.name}: Surface density profiles')
    ax3.grid(True, which='both', alpha=0.3)
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    fig3.savefig(outdir / 'surface_density.png', dpi=150, bbox_inches='tight')

    # Persist radial profiles for deeper analysis
    try:
        import csv
        with open(outdir / 'profiles.csv', 'w', newline='') as fcsv:
            w = csv.writer(fcsv)
            w.writerow(['R_kpc','Sigma_bar','Sigma_eff','kappa_bar','kappa_eff','kappa_bar_mean','kappa_eff_mean','gamma_bar','gamma_eff','alpha_bar_arcsec','alpha_eff_arcsec'])
            for i in range(len(R)):
                w.writerow([float(R[i]), float(Sigma_bar[i]), float(Sigma_eff[i]), float(kappa_bar[i]), float(kappa_eff[i]), float(kappa_bar_mean[i]), float(kappa_eff_mean[i]), float(gamma_bar[i]), float(gamma_eff[i]), float(alpha_bar_arcsec[i]), float(alpha_eff_arcsec[i])])
    except Exception as e:
        print(f"Warning: failed to write profiles.csv for {spec.name}: {e}")

    # Summaries
    summary = {
        'cluster': spec.name,
        'z_lens': spec.z_lens,
        'z_source': spec.z_source,
        'r200_kpc': spec.r200_kpc,
        'M200_Msun': spec.M200_Msun,
        'Sigma_crit_Msun_per_kpc2': float(Sigma_crit),
        'Einstein_radius_arcsec_baryons': None if RE_bar is None else float(RE_bar[1]),
        'Einstein_radius_kpc_baryons': None if RE_bar is None else float(RE_bar[0]),
        'Einstein_radius_arcsec_G3': None if RE_eff is None else float(RE_eff[1]),
        'Einstein_radius_kpc_G3': None if RE_eff is None else float(RE_eff[0]),
        'max_kappa_eff_mean': float(np.nanmax(kappa_eff_mean)),
        'max_kappa_bar_mean': float(np.nanmax(kappa_bar_mean)),
        'note': 'G³-effective lensing computed by mapping g_total to M_eff(r)=g_total r^2/G and projecting.'
    }
    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def load_best_params_if_available():
    best_path = Path('out/sparc_formula_sweep/best.json')
    if best_path.exists():
        with open(best_path, 'r') as f:
            best = json.load(f)
        if 'x' in best:
            names = ["v0_kms", "rc0_kpc", "gamma", "beta", "sigma_star",
                     "alpha", "kappa", "eta", "delta_kpc", "p_in", "p_out", "g_sat"]
            for i, nm in enumerate(names):
                if i < len(best['x']):
                    UNIVERSAL_PARAMS[nm] = best['x'][i]
        if 'variant' in best:
            BEST_VARIANT.update(best['variant'])


def main():
    load_best_params_if_available()
    model = UniversalG3Model()

    # Cluster presets (can be replaced by real catalogs later)
    clusters = [
        ClusterSpec(name='Abell_1689', z_lens=0.183, z_source=1.0, r200_kpc=2200.0, M200_Msun=2.0e15,
                    rhalf_kpc=700.0, sigma_bar=100.0),
        ClusterSpec(name='Coma', z_lens=0.0231, z_source=0.8, r200_kpc=2000.0, M200_Msun=1.5e15,
                    rhalf_kpc=650.0, sigma_bar=90.0),
        ClusterSpec(name='Bullet', z_lens=0.296, z_source=1.2, r200_kpc=1800.0, M200_Msun=1.0e15,
                    rhalf_kpc=600.0, sigma_bar=95.0),
    ]

    out_root = Path('out/cluster_lensing')
    summaries: List[Dict] = []

    for spec in clusters:
        outdir = out_root / spec.name
        summary = compute_lensing_for_cluster(spec, model, outdir)
        summaries.append(summary)
        print(f"Completed lensing for {spec.name}: θ_E(G3)={summary['Einstein_radius_arcsec_G3']} arcsec")

    # Aggregate
    with open(out_root / 'summaries.json', 'w') as f:
        json.dump(summaries, f, indent=2)

    # Quick comparison table print
    print("\nG³ Lensing Summary (Einstein Radii):")
    for s in summaries:
        print(f"- {s['cluster']}: θ_E(G3)={s['Einstein_radius_arcsec_G3']} arcsec, θ_E(baryons)={s['Einstein_radius_arcsec_baryons']} arcsec")


if __name__ == '__main__':
    main()
