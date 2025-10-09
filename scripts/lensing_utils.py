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

from typing import Optional, Tuple, Dict, List

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    load_real_cluster_profiles, abel_project_sigma, sigma_crit_Msun_per_kpc2,
    angular_diameter_distance_kpc, comoving_distance_Mpc, Mpc_to_kpc,
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
        # Return deflection in ARCSEC: α_arcsec = k̄(θ) * θ_arcsec
        kbar = float(np.interp(theta_arcsec, theta_arcsec_grid, kbar_mean,
                               left=kbar_mean[0], right=kbar_mean[-1]))
        return kbar * theta_arcsec
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
        # Return ARCSEC (not radians!) - this is the normalized deflection magnitude
        return float(abs(ay_here_arcsec))
    return alpha_y_of


def alpha_fun_GE(local_name: str, z_lens: float, z_source: float,
                 a: float, b: float, d: float, gamma1: float, gamma2: float,
                 Rd_kpc: float = 1000.0, R_scale_kpc: float = 100.0,
                 beta_clip=(1.0, 5.0),
                 A_core: float = 0.25, p_core: float = 2.0,
                 Sigma0_hat: float = 0.0, beta_core: float = 0.4,
                 smooth_R_kpc: float = 5.0):
    """Deflection α(θ) for GE using Σ_eff (ARCSEC) with interior-anchored export."""
    try:
        r, rho = load_real_cluster_profiles(local_name)
        R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
        Sigma_bar = abel_project_sigma(r, rho, R)
        model = GeometricExponentGravity(a=a, b=b, d=d, gamma1=gamma1, gamma2=gamma2,
                                         R_scale_kpc=R_scale_kpc, beta_clip=beta_clip,
                                         A_core=A_core, p_core=p_core, Sigma0_hat=Sigma0_hat,
                                         beta_core=beta_core, smooth_R_kpc=smooth_R_kpc)
        Sigma_eff, _, _ = model.Sigma_effective(R, Sigma_bar, Rd_kpc=Rd_kpc)
        Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)
        Mproj = np.array([2*np.pi*np.trapezoid(Sigma_eff[:i+1]*R[:i+1], R[:i+1]) for i in range(len(R))])
        area = np.pi * R**2
        Sbar_eff = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area>0)
        kbar_mean = Sbar_eff / Sigma_crit
        Dd = angular_diameter_distance_kpc(z_lens)
        theta_arcsec_grid = (R / max(Dd, 1e-12)) * 206265.0
        def alpha_of(theta_arcsec: float) -> float:
            kbar = float(np.interp(theta_arcsec, theta_arcsec_grid, kbar_mean,
                                   left=kbar_mean[0], right=kbar_mean[-1]))
            return kbar * theta_arcsec
        return alpha_of
    except Exception:
        return None


# Frontier HFF utilities

def list_frontier_models(cluster_id: str) -> Dict[str, List[str]]:
    """Return mapping of team -> versions present under data/frontier/hlsp/<cluster>."""
    root = ROOT / 'data' / 'frontier' / 'hlsp' / cluster_id.lower()
    out: Dict[str, List[str]] = {}
    if not root.exists():
        return out
    for team_dir in root.iterdir():
        if not team_dir.is_dir():
            continue
        versions = []
        for v in team_dir.iterdir():
            if v.is_dir():
                versions.append(v.name)
        if not versions:
            versions = ['unversioned']
        out[team_dir.name] = versions
    return out


def find_model_file(cluster_id: str, team: str, version: str, key_parts: List[str]) -> Optional[Path]:
    """Find a file that contains all substrings in key_parts (case-insensitive)."""
    base = ROOT / 'data' / 'frontier' / 'hlsp' / cluster_id.lower() / team / version
    if version == 'unversioned':
        base = ROOT / 'data' / 'frontier' / 'hlsp' / cluster_id.lower() / team
    if not base.exists():
        return None
    parts_lc = [p.lower() for p in key_parts]
    for p in base.iterdir():
        if p.is_file():
            name = p.name.lower()
            if all(s in name for s in parts_lc):
                return p
    return None


def beta_Dls_over_Ds(z_lens: float, z_source: float) -> float:
    if z_source <= z_lens:
        return 0.0
    Ds = angular_diameter_distance_kpc(z_source)
    Dc_s = comoving_distance_Mpc(z_source) * Mpc_to_kpc
    Dc_d = comoving_distance_Mpc(z_lens) * Mpc_to_kpc
    Dds = (Dc_s - Dc_d) / (1.0 + z_source)
    return float(Dds / max(Ds, 1e-12))

def beta_from_header(hdr) -> float | None:
    for key in ('BETA','DLS_DS','DLSDS'):
        if key in hdr:
            try:
                return float(hdr[key])
            except Exception:
                pass
    zl = None; zs = None
    for k in ('ZL','LENSZ','Z_L','REDSHIFT_L','ZLENS'):
        if k in hdr:
            try:
                zl = float(hdr[k]); break
            except Exception:
                pass
    for k in ('ZS','Z_S','ZSOURCE','REDSHIFT_S','Z_SOURCE'):
        if k in hdr:
            try:
                zs = float(hdr[k]); break
            except Exception:
                pass
    if zl is not None and zs is not None:
        return beta_Dls_over_Ds(zl, zs)
    return None

def alpha_fun_ACCEPTED(cluster_id: str, team: str, version: str,
                       z_lens: float, z_source: float) -> Optional[callable]:
    """Return alpha_y(theta_arcsec) based on HFF deflection maps.

    Prefers *-arcsec-deflect.fits; falls back to *-pixels-deflect.fits and converts with WCS.
    Uses the central vertical cut through the κ peak (or deflection map center if κ absent).
    """
    # Try arcsec-deflect
    fx = find_model_file(cluster_id, team, version, ['x-arcsec-deflect', '.fits'])
    fy = find_model_file(cluster_id, team, version, ['y-arcsec-deflect', '.fits'])
    scale_arcsec = 1.0
    if fx is None or fy is None:
        # try pixel-deflect with WCS scale
        fx = find_model_file(cluster_id, team, version, ['x-pixels-deflect', '.fits'])
        fy = find_model_file(cluster_id, team, version, ['y-pixels-deflect', '.fits'])
        scale_arcsec = None  # determine from WCS
    if fx is None or fy is None:
        return None
    try:
        with fits.open(str(fy)) as hy:
            ay = np.array(hy[0].data, dtype=float)
            hdr = hy[0].header
        with fits.open(str(fx)) as hx:
            ax = np.array(hx[0].data, dtype=float)
        if scale_arcsec is None:
            try:
                w = WCS(hdr)
                sc = proj_plane_pixel_scales(w)
                scale_arcsec = float(np.mean(sc[:2]) * 3600.0)
            except Exception:
                return None
        # Convert to arcsec
        ax_arcsec = ax if scale_arcsec == 1.0 else (ax * scale_arcsec)
        ay_arcsec = ay if scale_arcsec == 1.0 else (ay * scale_arcsec)
        # Geometry scaling s = β_user/β_ref
        beta_ref = beta_from_header(hdr)
        beta_user = beta_Dls_over_Ds(z_lens, z_source)
        s = beta_user if beta_ref is None else (beta_user / max(beta_ref, 1e-12))
        ax_arcsec *= s; ay_arcsec *= s
        # center from κ peak if available
        kappa_fp = find_model_file(cluster_id, team, version, ['kappa', '.fits'])
        if kappa_fp is not None:
            with fits.open(str(kappa_fp)) as hk:
                kap = np.array(hk[0].data, dtype=float)
                cy, cx = np.unravel_index(int(np.nanargmax(np.nan_to_num(kap, nan=-1e30))), kap.shape)
        else:
            cy, cx = int(ay_arcsec.shape[0]//2), int(ay_arcsec.shape[1]//2)
        ny, nx = ay_arcsec.shape
        cx = int(np.clip(cx, 0, nx-1)); cy = int(np.clip(cy, 0, ny-1))
        # Determine arcsec->pixel direction from WCS (y positive upward)
        try:
            w = WCS(hdr)
            sc = proj_plane_pixel_scales(w) * 3600.0
            arcsec_per_pix = float(np.mean(sc[:2]))
            cd = w.wcs.cd if w.wcs.cd is not None else w.wcs.pc
            sign_y = -1.0 if cd is None else (1.0 if cd[1,1] > 0 else -1.0)
        except Exception:
            arcsec_per_pix = scale_arcsec
            sign_y = -1.0
        def alpha_y_of(theta_arcsec: float) -> float:
            # Return magnitude in ARCSEC along central vertical cut
            y_pix = cy + sign_y * (theta_arcsec / arcsec_per_pix)
            x_pix = float(cx)
            x0 = int(np.clip(np.floor(x_pix), 0, nx-1)); x1 = min(x0 + 1, nx - 1)
            y0 = int(np.clip(np.floor(y_pix), 0, ny-1)); y1 = min(y0 + 1, ny - 1)
            tx = x_pix - x0; ty = y_pix - y0
            v00 = ay_arcsec[y0, x0]; v10 = ay_arcsec[y0, x1]
            v01 = ay_arcsec[y1, x0]; v11 = ay_arcsec[y1, x1]
            vals = np.array([v00, v10, v01, v11], float)
            m = np.nanmean(np.where(np.isfinite(vals), vals, np.nan))
            if not np.isfinite(m):
                m = 0.0
            v00 = v00 if np.isfinite(v00) else m
            v10 = v10 if np.isfinite(v10) else m
            v01 = v01 if np.isfinite(v01) else m
            v11 = v11 if np.isfinite(v11) else m
            a_arcsec = ((1-tx)*(1-ty)*v00 + tx*(1-ty)*v10 + (1-tx)*ty*v01 + tx*ty*v11)
            # Return ARCSEC (not radians!)
            return float(abs(a_arcsec))
        return alpha_y_of
    except Exception:
        return None


def compute_thetaEcrit_from_maps(cluster_id: str, team: str, version: str) -> Optional[float]:
    """Compute area-equivalent tangential critical radius from κ and γ maps.

    T = 1 - κ - |γ|. We take the connected T<=0 component around the κ peak,
    compute its area in arcsec^2 and return θE = sqrt(area/pi) in arcsec.
    """
    kappa_fp = find_model_file(cluster_id, team, version, ['kappa', '.fits'])
    gamma_fp = find_model_file(cluster_id, team, version, ['gamma.fits'])
    g1_fp = find_model_file(cluster_id, team, version, ['gamma1', '.fits'])
    g2_fp = find_model_file(cluster_id, team, version, ['gamma2', '.fits'])
    if kappa_fp is None or (gamma_fp is None and (g1_fp is None or g2_fp is None)):
        return None
    try:
        with fits.open(str(kappa_fp)) as hk:
            kap = np.array(hk[0].data, dtype=float)
            hdr = hk[0].header
        if gamma_fp is not None:
            with fits.open(str(gamma_fp)) as hg:
                gam = np.array(hg[0].data, dtype=float)
        else:
            with fits.open(str(g1_fp)) as h1, fits.open(str(g2_fp)) as h2:
                g1 = np.array(h1[0].data, dtype=float)
                g2 = np.array(h2[0].data, dtype=float)
                gam = np.sqrt(np.square(g1) + np.square(g2))
        # arcsec/pixel from WCS
        try:
            w = WCS(hdr); sc = proj_plane_pixel_scales(w); arcsec_per_pix = float(np.mean(sc[:2]) * 3600.0)
        except Exception:
            return None
        T = 1.0 - kap - gam
        # seed near κ peak with T<=0
        cy, cx = np.unravel_index(int(np.nanargmax(np.nan_to_num(kap, nan=-1e30))), kap.shape)
        allowed = np.isfinite(T) & (T <= 0)
        if not allowed.any():
            return None
        # find nearest allowed to (cy,cx)
        if not allowed[cy, cx]:
            # search small window
            rad = 1
            found = False
            ny, nx = allowed.shape
            while rad < max(ny, nx):
                y0 = max(0, cy - rad); y1 = min(ny, cy + rad + 1)
                x0 = max(0, cx - rad); x1 = min(nx, cx + rad + 1)
                sub = allowed[y0:y1, x0:x1]
                if sub.any():
                    ys, xs = np.where(sub)
                    # pick closest
                    dy = ys + y0 - cy; dx = xs + x0 - cx
                    i = int(np.argmin(dy*dy + dx*dx))
                    cy = int(ys[i] + y0); cx = int(xs[i] + x0)
                    found = True
                    break
                rad *= 2
            if not found:
                return None
        core = np.zeros_like(allowed, dtype=bool)
        core[cy, cx] = True
        # iterative dilations using 8-neighborhood
        changed = True
        it = 0
        while changed and it < 5000:
            it += 1
            nbr = core.copy()
            nbr |= np.roll(core, 1, axis=0)
            nbr |= np.roll(core, -1, axis=0)
            nbr |= np.roll(core, 1, axis=1)
            nbr |= np.roll(core, -1, axis=1)
            nbr |= np.roll(np.roll(core, 1, axis=0), 1, axis=1)
            nbr |= np.roll(np.roll(core, 1, axis=0), -1, axis=1)
            nbr |= np.roll(np.roll(core, -1, axis=0), 1, axis=1)
            nbr |= np.roll(np.roll(core, -1, axis=0), -1, axis=1)
            new_core = (nbr & allowed)
            changed = bool(np.any(new_core & (~core)))
            core = new_core
        area_pix = float(np.count_nonzero(core))
        if area_pix <= 0:
            return None
        area_arcsec2 = area_pix * (arcsec_per_pix ** 2)
        thetaE = (area_arcsec2 / np.pi) ** 0.5
        return float(thetaE)
    except Exception:
        return None


def solve_theta_E_from_alpha(alpha_fun, theta_guess_arcsec: float, theta_min_arcsec: float, theta_max_arcsec: float) -> float | None:
    """Solve α(θ) = θ for θ_E using a safe bracket and bisection.

    alpha_fun takes θ in arcsec and returns ARCSEC.
    Returns θ_E in arcsec or None if not bracketed.
    """
    try:
        def f(theta_arcsec: float) -> float:
            return alpha_fun(theta_arcsec) - theta_arcsec
        a, b = float(theta_min_arcsec), float(theta_max_arcsec)
        fa, fb = f(a), f(b)
        # Expand bracket if needed up to a factor
        expand = 0
        while fa * fb > 0 and expand < 5:
            span = b - a
            a = max(1e-3, a - 0.5*span)
            b = b + 0.5*span
            fa, fb = f(a), f(b)
            expand += 1
        if fa * fb > 0:
            return None
        # Bisection
        for _ in range(60):
            m = 0.5*(a + b)
            fm = f(m)
            if abs(fm) < 1e-6:
                return float(m)
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return float(0.5*(a + b))
    except Exception:
        return None
