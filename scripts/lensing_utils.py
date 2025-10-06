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

def find_hlsp_kappa_with_scale(cluster_id: str):
    base = ROOT / 'data' / 'clash' / 'hlsp' / cluster_id.lower()
    pats = [str(base / '**' / '*kappa*.fits'), str(base / '**' / '*kappa1*.fits')]
    files = []
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
        x0 = int(np.floor(x_pix)); y0 = int(np.floor(y_pix))
        x1 = min(x0 + 1, nx - 1); y1 = min(y0 + 1, ny - 1)
        tx = x_pix - x0; ty = y_pix - y0
        ay_here_arcsec = ( (1-tx)*(1-ty)*ay_arcsec[y0, x0] + tx*(1-ty)*ay_arcsec[y0, x1] + (1-tx)*ty*ay_arcsec[y1, x0] + tx*ty*ay_arcsec[y1, x1] )
        return float(ay_here_arcsec / 206265.0)
    return alpha_y_of
