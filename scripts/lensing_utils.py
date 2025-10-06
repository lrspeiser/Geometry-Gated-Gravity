#!/usr/bin/env python3
from __future__ import annotations
"""
Shared lensing utilities for interactive visualizations.

Provides:
- CLASH: mapping from short cluster_id -> (HLSP name, z_lens)
- find_hlsp_kappa_with_scale(cluster_id): load HLSP kappa map and infer arcsec/pixel via WCS
- alpha_fun_GR_baryons(local_name, z_lens, z_source): GR deflection from baryonic mass only
- alpha_fun_HLSP(cluster_id, theta_obs_arcsec): HLSP-based deflection field, returning alpha_y(theta)

Notes:
- alpha_* functions return deflection in radians.
- HLSP alpha is normalized to match |alpha|=theta_obs (rad) on a ring at theta_obs arcsec for visualization unless the
  caller further rescales. This keeps visualizations comparable when the map has differing native normalizations.
"""
from pathlib import Path
import glob
import numpy as np

from astropy.io import fits  # type: ignore
from astropy.wcs import WCS  # type: ignore
from astropy.wcs.utils import proj_plane_pixel_scales  # type: ignore

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    load_real_cluster_profiles, abel_project_sigma, sigma_crit_Msun_per_kpc2,
    angular_diameter_distance_kpc,
)
from concepts.squared_gravity.geometric_exponent import GeometricExponentGravity

# Short-name to (HLSP folder name, redshift)
CLASH = {
    'a1423': ('ABELL_1423', 0.213), 'a209': ('ABELL_0209', 0.206), 'a2261': ('ABELL_2261', 0.224),
    'a383': ('ABELL_0383', 0.187), 'a611': ('ABELL_0611', 0.288), 'clj1226': ('CLJ1226', 0.890),
    'macs0329': ('MACSJ0329', 0.450), 'macs0416': ('MACSJ0416', 0.396), 'macs0429': ('MACSJ0429', 0.399),
    'macs0647': ('MACSJ0647', 0.584), 'macs0717': ('MACSJ0717', 0.548), 'macs0744': ('MACSJ0744', 0.686),
    'macs1115': ('MACSJ1115', 0.352), 'macs1149': ('MACSJ1149', 0.544), 'macs1206': ('MACSJ1206', 0.440),
    'macs1311': ('MACSJ1311', 0.494), 'macs1423': ('MACSJ1423', 0.545), 'macs1720': ('MACSJ1720', 0.391),
    'macs1931': ('MACSJ1931', 0.352), 'macs2129': ('MACSJ2129', 0.570), 'ms2137': ('MS2137', 0.313),
    'rxj1347': ('RXJ1347', 0.451), 'rxj1532': ('RXJ1532', 0.345), 'rxj2129': ('RXJ2129', 0.234), 'rxj2248': ('RXJ2248', 0.348),
}

ROOT = Path(__file__).resolve().parents[1]


def load_gold_standard_catalog():
    """Load gold-standard cluster metadata if present.

    Returns mapping: cluster_id -> { 'z_lens': float, 'accepted': { 'zs': float, 'theta_E_arcsec': float }, 'notes': str }
    """
path = ROOT / 'data' / 'frontier' / 'gold_standard' / 'gold_standard_clusters.json'
    if path.exists():
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def find_hlsp_kappa_with_scale(cluster_id: str):
    """Locate a kappa FITS map and derive arcsec/pixel via WCS.

    Searches both CLASH and Frontier HLSP folder conventions:
      - data/clash/hlsp/<cluster_id>/**/kappa*.fits
      - data/frontier/hlsp/<cluster_id>/**/kappa*.fits
    """
    search_roots = [ROOT / 'data' / 'clash' / 'hlsp' / cluster_id.lower(),
                    ROOT / 'data' / 'frontier' / 'hlsp' / cluster_id.lower()]
    files = []
    for base in search_roots:
        pats = [str(base / '**' / '*kappa*.fits'), str(base / '**' / '*kappa1*.fits')]
        for p in pats:
            files.extend(glob.glob(p, recursive=True))
    files_sorted = sorted(files, key=lambda s: (0 if 'kappa.fits' in Path(s).name else 1, len(s)))
    for fp in files_sorted:
        try:
            with fits.open(fp) as hdul:
                hdr = hdul[0].header
                arr = hdul[0].data
                if arr is None or arr.size == 0:
                    continue
                arr = np.array(arr, dtype=float)
                arcsec_per_pix = None
                try:
                    w = WCS(hdr)
                    scales = proj_plane_pixel_scales(w)  # deg/pix
                    if scales is not None and len(scales) >= 2:
                        arcsec_per_pix = float(np.mean(scales[:2]) * 3600.0)
                except Exception:
                    arcsec_per_pix = None
                return arr, arcsec_per_pix
        except Exception:
            continue
    return None, None


def alpha_fun_GR_baryons(local_name: str, z_lens: float, z_source: float):
    r, rho = load_real_cluster_profiles(local_name)
    R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
    Sigma_bar = abel_project_sigma(r, rho, R)
    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)
    Mproj = np.array([2*np.pi*np.trapezoid(Sigma_bar[:i+1]*R[:i+1], R[:i+1]) for i in range(len(R))])
    area = np.pi * R**2
    Sbar = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area>0)
    kbar_mean = Sbar / Sigma_crit
    Dd = angular_diameter_distance_kpc(z_lens)
    theta_arcsec_grid = (R / max(Dd, 1e-12)) * 206265.0
    def alpha_of(theta_arcsec: float) -> float:
        kbar = float(np.interp(theta_arcsec, theta_arcsec_grid, kbar_mean,
                               left=kbar_mean[0], right=kbar_mean[-1]))
        theta_rad = theta_arcsec / 206265.0
        return kbar * theta_rad
    return alpha_of


def alpha_fun_HLSP(cluster_id: str, theta_obs_arcsec: float):
    kappa, arcsec_per_pix = find_hlsp_kappa_with_scale(cluster_id)
    if kappa is None or arcsec_per_pix is None or not np.isfinite(arcsec_per_pix) or arcsec_per_pix <= 0:
        return None
    # FFT Poisson solve: ∇²ψ = 2κ, α = ∇ψ
    img = np.asarray(kappa, float)
    # Replace NaNs in the HLSP map to avoid contaminating FFT with NaNs
    img = np.nan_to_num(img, nan=0.0)
    ny, nx = img.shape
    K = np.fft.fft2(img)
    fx = np.fft.fftfreq(nx)
    fy = np.fft.fftfreq(ny)
    kx = (2*np.pi) * fx
    ky = (2*np.pi) * fy
    kxg, kyg = np.meshgrid(kx, ky)
    k2 = kxg*kxg + kyg*kyg
    mask0 = (k2 == 0)
    k2_safe = k2.copy(); k2_safe[mask0] = 1.0
    Psi = -2.0 * K / k2_safe
    Psi[mask0] = 0.0
    Ax = (1j) * kxg * Psi
    Ay = (1j) * kyg * Psi
    ax_pix = np.real(np.fft.ifft2(Ax))
    ay_pix = np.real(np.fft.ifft2(Ay))
    # Convert to arcsec units
    ax_arcsec = ax_pix * arcsec_per_pix
    ay_arcsec = ay_pix * arcsec_per_pix
    # Re-center to κ peak
    peak_idx = np.unravel_index(int(np.nanargmax(img)), img.shape)
    cy, cx = float(peak_idx[0]), float(peak_idx[1])
    # Normalize deflection magnitude so mean |α| at θ_obs equals θ_obs (rad)
    rp = float(theta_obs_arcsec / arcsec_per_pix)
    nang = 360
    ang = np.linspace(0, 2*np.pi, nang, endpoint=False)
    xs = cx + rp * np.cos(ang)
    ys = cy + rp * np.sin(ang)
    def bilinear(a, xp, yp):
        x0 = np.clip(np.floor(xp).astype(int), 0, nx-1)
        y0 = np.clip(np.floor(yp).astype(int), 0, ny-1)
        x1 = np.clip(x0+1, 0, nx-1)
        y1 = np.clip(y0+1, 0, ny-1)
        tx = xp - x0; ty = yp - y0
        return ( (1-tx)*(1-ty)*a[y0, x0] + tx*(1-ty)*a[y0, x1] + (1-tx)*ty*a[y1, x0] + tx*ty*a[y1, x1] )
    ax_ring = bilinear(ax_arcsec, xs, ys)
    ay_ring = bilinear(ay_arcsec, xs, ys)
    alpha_mag_ring_rad = np.sqrt(ax_ring**2 + ay_ring**2) / 206265.0
    target = float(theta_obs_arcsec / 206265.0)
    current = float(np.nanmean(alpha_mag_ring_rad)) if np.isfinite(alpha_mag_ring_rad).all() else 1.0
    scale = target / current if current > 0 else 1.0
    ax_arcsec *= scale; ay_arcsec *= scale
    # Interpolator for α_y along vertical through peak
    def alpha_y_of(y_arcsec: float) -> float:
        y_pix = cy + (y_arcsec / arcsec_per_pix)
        x_pix = cx
        # Clamp indices to valid range to avoid negative or overflow indexing
        x0f = np.floor(x_pix)
        y0f = np.floor(y_pix)
        x0 = int(np.clip(x0f, 0, nx - 2))
        y0 = int(np.clip(y0f, 0, ny - 2))
        x1 = x0 + 1
        y1 = y0 + 1
        tx = float(np.clip(x_pix - x0, 0.0, 1.0))
        ty = float(np.clip(y_pix - y0, 0.0, 1.0))
        v00 = ay_arcsec[y0, x0]
        v10 = ay_arcsec[y0, x1]
        v01 = ay_arcsec[y1, x0]
        v11 = ay_arcsec[y1, x1]
        # All values should be finite after nan_to_num; extra guard just in case
        v00 = 0.0 if not np.isfinite(v00) else v00
        v10 = 0.0 if not np.isfinite(v10) else v10
        v01 = 0.0 if not np.isfinite(v01) else v01
        v11 = 0.0 if not np.isfinite(v11) else v11
        ay_here_arcsec = ( (1-tx)*(1-ty)*v00 + tx*(1-ty)*v10 + (1-tx)*ty*v01 + tx*ty*v11 )
        return float(ay_here_arcsec / 206265.0)
    return alpha_y_of


def alpha_fun_GE(local_name: str, z_lens: float, z_source: float,
                 a: float, b: float, d: float, gamma1: float, gamma2: float,
                 Rd_kpc: float = 1000.0, R_scale_kpc: float = 100.0,
                 beta_clip=(1.0, 5.0)):
    """Deflection alpha(theta) for the geometry-exponent model using Σ_eff.

    Returns a callable alpha_of(theta_arcsec) -> radians.
    """
    try:
        r, rho = load_real_cluster_profiles(local_name)
        R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
        Sigma_bar = abel_project_sigma(r, rho, R)
        model = GeometricExponentGravity(a=a, b=b, d=d, gamma1=gamma1, gamma2=gamma2,
                                         R_scale_kpc=R_scale_kpc, beta_clip=beta_clip)
        Sigma_eff, _, _ = model.Sigma_effective(R, Sigma_bar, Rd_kpc=Rd_kpc)
        Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)
        # Mean Σ_eff within R
        Mproj = np.array([2*np.pi*np.trapezoid(Sigma_eff[:i+1]*R[:i+1], R[:i+1]) for i in range(len(R))])
        area = np.pi * R**2
        Sbar_eff = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area>0)
        kbar_mean = Sbar_eff / Sigma_crit
        Dd = angular_diameter_distance_kpc(z_lens)
        theta_arcsec_grid = (R / max(Dd, 1e-12)) * 206265.0
        def alpha_of(theta_arcsec: float) -> float:
            kbar = float(np.interp(theta_arcsec, theta_arcsec_grid, kbar_mean,
                                   left=kbar_mean[0], right=kbar_mean[-1]))
            theta_rad = theta_arcsec / 206265.0
            return kbar * theta_rad
        return alpha_of
    except Exception:
        return None
