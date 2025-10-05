#!/usr/bin/env python3
from __future__ import annotations
"""
Visualize cluster baryons in 3D and compare GR baryon-only κ map to HLSP model κ.

Usage examples:
  python scripts/visualize_cluster_3d_and_kappa.py --cluster_id a209 --zs 2.0
  python scripts/visualize_cluster_3d_and_kappa.py --cluster_id a1423 --zs 2.0

What it makes:
  out/visualizations/<cluster_id>/baryons_3d.png
  out/visualizations/<cluster_id>/kappa_bar_vs_hlsp.png

Notes
- Baryon profiles are spherical (1D r, rho). We generate a 3D point cloud by
  sampling r with PDF ∝ r^2 rho(r) and random angles (uniform on sphere).
- κ_bar(R) is from Σ_bar(R)/Σ_crit using Abel projection.
- HLSP κ map is loaded if a kappa FITS is found under data/clash/hlsp/<cluster_id>.
- We do not attempt precise WCS alignment to the instrument frame here; HLSP κ is shown as-is for qualitative comparison.
"""
import argparse
from pathlib import Path
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    load_real_cluster_profiles, abel_project_sigma, sigma_crit_Msun_per_kpc2,
    angular_diameter_distance_kpc,
)

# Optional: astropy for FITS+WCS
try:
    from astropy.io import fits  # type: ignore
    from astropy.wcs import WCS  # type: ignore
    from astropy.wcs.utils import proj_plane_pixel_scales  # type: ignore
    ASTROPY_OK = True
except Exception:
    ASTROPY_OK = False

# Map cluster_id -> (local_name, z_lens)
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


def explain_mape():
    print("MAPE = mean absolute percentage error; here it's the average of |θE_pred - θE_obs| / θE_obs × 100% across clusters.")


def sample_baryons_3d(r: np.ndarray, rho: np.ndarray, n: int = 20000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.asarray(r, float)
    rho = np.maximum(np.asarray(rho, float), 0.0)
    # PDF over radius ∝ r^2 rho(r)
    w = r**2 * rho
    w = np.maximum(w, 0.0)
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.zeros((0, 3))
    cw /= cw[-1]
    u = rng.random(n)
    # Invert CDF via interp
    r_samp = np.interp(u, cw, r)
    # Uniform on sphere
    cos_t = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2*np.pi, size=n)
    sin_t = np.sqrt(1.0 - cos_t**2)
    x = r_samp * sin_t * np.cos(phi)
    y = r_samp * sin_t * np.sin(phi)
    z = r_samp * cos_t
    return np.vstack([x, y, z]).T


def kappa_bar_grid(Rmax_kpc: float, npx: int, R_kpc: np.ndarray, kappa_radial: np.ndarray) -> np.ndarray:
    # Build square grid [-Rmax, Rmax]^2
    xs = np.linspace(-Rmax_kpc, Rmax_kpc, npx)
    ys = np.linspace(-Rmax_kpc, Rmax_kpc, npx)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X*X + Y*Y)
    # Interpolate radial kappa to grid
    kappa_img = np.interp(R, R_kpc, kappa_radial, left=kappa_radial[0], right=kappa_radial[-1])
    return kappa_img


def find_hlsp_kappa(cluster_id: str):
    """Return (kappa2d, arcsec_per_pix) if available, else (None, None)."""
    if not ASTROPY_OK:
        return None, None
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
                    # proj_plane_pixel_scales returns deg/pix; take mean for square pixels
                    scales = proj_plane_pixel_scales(w)  # deg/pix
                    if scales is not None and len(scales) >= 2:
                        arcsec_per_pix = float(np.mean(scales[:2]) * 3600.0)
                except Exception:
                    arcsec_per_pix = None
                return arr, arcsec_per_pix
        except Exception:
            continue
    return None, None


def _find_theta_E_from_kbar_mean(R: np.ndarray, kbar_mean: np.ndarray) -> float | None:
    idx = np.where(kbar_mean >= 1.0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(R[0])
    x0, y0 = float(R[i-1]), float(kbar_mean[i-1])
    x1, y1 = float(R[i]), float(kbar_mean[i])
    if y1 == y0:
        return float(x1)
    return float(x0 + (1 - y0) * (x1 - x0) / (y1 - y0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster_id', required=True, help='e.g., a209, a1423, macs0416...')
    ap.add_argument('--zs', type=float, default=2.0, help='Source redshift for Σ_crit')
    ap.add_argument('--npoints', type=int, default=20000, help='Points in 3D baryon cloud')
    args = ap.parse_args()

    explain_mape()

    cid = args.cluster_id.lower()
    if cid not in CLASH:
        print(f'Unknown cluster_id {cid}. Available keys: {sorted(CLASH.keys())[:8]} ...')
        sys.exit(1)

    local_name, z_lens = CLASH[cid]

    # Load observed θE (arcsec) if available
    theta_obs = None
    obs_csv = ROOT / 'data' / 'clash' / 'einstein_radii_observed.csv'
    if obs_csv.exists():
        try:
            import pandas as pd  # type: ignore
            df = pd.read_csv(obs_csv)
            df['cluster_id'] = df['cluster_id'].str.lower()
            row = df[df['cluster_id'] == cid]
            if len(row) > 0:
                theta_obs = float(row.iloc[0]['theta_E_observed_arcsec'])
        except Exception:
            theta_obs = None

    # Load 1D baryon profiles
    r, rho = load_real_cluster_profiles(local_name)

    # 3D sampling
    pts = sample_baryons_3d(r, rho, n=args.npoints, seed=0)

    # Projected Σ and κ_bar
    R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
    Sigma_bar = abel_project_sigma(r, rho, R)
    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, args.zs)
    kappa_radial = Sigma_bar / Sigma_crit

    # Mean kappa inside R and GR θE
    Mproj = np.array([2*np.pi*np.trapezoid(Sigma_bar[:i+1]*R[:i+1], R[:i+1]) for i in range(len(R))])
    area = np.pi * R**2
    Sbar = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area>0)
    kbar_mean = Sbar / Sigma_crit
    R_E_GR = _find_theta_E_from_kbar_mean(R, kbar_mean)

    # Convert R and R_E_GR to arcsec
    Dd_kpc = angular_diameter_distance_kpc(z_lens)
    R_arcsec = (R / max(Dd_kpc, 1e-12)) * (180.0/np.pi) * 3600.0
    theta_GR_arcsec = None if R_E_GR is None else float((R_E_GR / max(Dd_kpc, 1e-12)) * (180.0/np.pi) * 3600.0)

    # Build 2D κ_bar image in arcsec coordinates (using arcsec extent)
    Rmax_kpc = float(R[-1])
    Rmax_arcsec = float(R_arcsec[-1])
    kappa_img = kappa_bar_grid(Rmax_kpc=Rmax_kpc, npx=512, R_kpc=R, kappa_radial=kappa_radial)

    # Try to load HLSP κ map + arcsec per pixel
    kappa_hlsp, arcsec_per_pix = find_hlsp_kappa(cid)

    outdir = ROOT / 'out' / 'visualizations' / cid
    outdir.mkdir(parents=True, exist_ok=True)

    # Figure 1: 3D baryons
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    if pts.shape[0] > 0:
        # Color by radius
        rr = np.sqrt(np.sum(pts**2, axis=1))
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=np.log10(rr+1e-6), s=1, cmap='viridis', alpha=0.5)
    ax.set_title(f'{cid}: 3D baryon cloud (samples), r_max ≈ {Rmax_kpc:.0f} kpc')
    ax.set_xlabel('x (kpc)'); ax.set_ylabel('y (kpc)'); ax.set_zlabel('z (kpc)')
    ax.set_box_aspect([1,1,1])
    fig.tight_layout(); fig.savefig(outdir / 'baryons_3d.png', dpi=150)
    plt.close(fig)

    # Figure 2: κ_bar vs HLSP κ with shared annotations
    cols = 2 if kappa_hlsp is not None else 1
    fig2, axes = plt.subplots(1, cols, figsize=(7*cols, 6))
    if cols == 1:
        axes = [axes]

    # Choose a common color scale using percentiles
    vmax_list = [np.nanpercentile(kappa_img, 99)]
    if kappa_hlsp is not None:
        vmax_list.append(np.nanpercentile(kappa_hlsp, 99))
    vmin, vmax = 0.0, float(max(vmax_list))

    # Left: GR baryons in arcsec extent
    im0 = axes[0].imshow(kappa_img, origin='lower', extent=[-Rmax_arcsec, Rmax_arcsec, -Rmax_arcsec, Rmax_arcsec], cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title('Predicted: GR (baryons only) [arcsec]')
    axes[0].set_xlabel('x (arcsec)'); axes[0].set_ylabel('y (arcsec)')
    # Overlay observed θE and GR θE
    if theta_obs is not None:
        c = Circle((0,0), theta_obs, fill=False, color='red', lw=1.8, label='Observed θE')
        axes[0].add_patch(c)
    if theta_GR_arcsec is not None:
        c2 = Circle((0,0), theta_GR_arcsec, fill=False, color='cyan', lw=1.5, ls='--', label='GR-baryons θE')
        axes[0].add_patch(c2)
    axes[0].legend(loc='upper right', fontsize=8)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='κ')

    # Right: HLSP κ map with arcsec axes if pixel scale known
    if kappa_hlsp is not None:
        if arcsec_per_pix is not None and np.isfinite(arcsec_per_pix) and arcsec_per_pix > 0:
            ny, nx = kappa_hlsp.shape
            halfx = (nx/2.0) * arcsec_per_pix
            halfy = (ny/2.0) * arcsec_per_pix
            extent = [-halfx, halfx, -halfy, halfy]
            axes[1].set_xlabel('x (arcsec)'); axes[1].set_ylabel('y (arcsec)')
        else:
            extent = None
            axes[1].set_xlabel('x (pix)'); axes[1].set_ylabel('y (pix)')
        im1 = axes[1].imshow(kappa_hlsp, origin='lower', cmap='magma', vmin=vmin, vmax=vmax, extent=extent)
        axes[1].set_title('Actual: HLSP κ (observational model)')
        # Overlay observed θE ring if we have arcsec axes and θE_obs
        if extent is not None and theta_obs is not None:
            c3 = Circle((0,0), theta_obs, fill=False, color='red', lw=1.8, label='Observed θE')
            axes[1].add_patch(c3)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='κ')

    fig2.suptitle(f'{cid}: Predicted vs Actual (Einstein-radius overlays)')
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(outdir / 'kappa_bar_vs_hlsp.png', dpi=150)
    plt.close(fig2)

    print(f'Wrote {outdir / "baryons_3d.png"}')
    print(f'Wrote {outdir / "kappa_bar_vs_hlsp.png"}')

if __name__ == '__main__':
    main()
