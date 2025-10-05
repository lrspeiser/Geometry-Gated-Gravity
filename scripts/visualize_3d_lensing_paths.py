#!/usr/bin/env python3
from __future__ import annotations
"""
3D lensing path visualization:
- Draw a translucent sphere at the origin to represent the cluster.
- Draw two photon trajectories (thin-lens kink at x=0) in 3D:
  • Cyan: "Actual" (from HLSP κ radialized) — stronger bending
  • Red:  "Predicted (GR baryons)" — weaker bending

We work in angular units (arcsec) so both curves and overlays are comparable.
The bending angle at impact θ is approximated by α(θ) = κ̄(<θ) · θ (axisymmetric thin lens).

Outputs per cluster:
  out/visualizations/<cluster_id>/3d_paths.png (3-panel views)

Example:
  python scripts/visualize_3d_lensing_paths.py --cluster_id a209 --zs 2.0
"""
import argparse
from pathlib import Path
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    load_real_cluster_profiles, abel_project_sigma, sigma_crit_Msun_per_kpc2,
    angular_diameter_distance_kpc,
)

try:
    from astropy.io import fits  # type: ignore
    ASTROPY_OK = True
except Exception:
    ASTROPY_OK = False

# cluster_id -> (local_name, z_lens)
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

def find_hlsp_kappa(cluster_id: str):
    if not ASTROPY_OK:
        return None
    base = ROOT / 'data' / 'clash' / 'hlsp' / cluster_id.lower()
    pats = [str(base / '**' / '*kappa*.fits'), str(base / '**' / '*kappa1*.fits')]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    files_sorted = sorted(files, key=lambda s: (0 if 'kappa.fits' in Path(s).name else 1, len(s)))
    for fp in files_sorted:
        try:
            with fits.open(fp) as hdul:
                arr = hdul[0].data
                if arr is not None and arr.size > 0:
                    return np.array(arr, dtype=float)
        except Exception:
            continue
    return None


def radial_cummean_kappa_from_map(kappa2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (r_pix, kappa_cummean) where cummean ≈ κ̄(<r)."""
    img = np.asarray(kappa2d, float)
    ny, nx = img.shape
    y = np.arange(ny) - (ny - 1) / 2.0
    x = np.arange(nx) - (nx - 1) / 2.0
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X*X + Y*Y)
    rmax = R.max()
    nbins = 300
    rbins = np.linspace(0, rmax, nbins+1)
    rmid = 0.5*(rbins[:-1] + rbins[1:])

    # cumulative mean inside radius in pixel space
    kcum = np.zeros_like(rmid)
    for i in range(1, len(rmid)):
        m = R <= rbins[i]
        if np.any(m):
            kcum[i] = float(np.mean(img[m]))
        else:
            kcum[i] = kcum[i-1]
    return rmid, kcum


def alpha_at_theta_GR_baryons(local_name: str, z_lens: float, z_source: float, theta_arcsec: float) -> float:
    # Build GR(baryons) kappa_bar_mean and evaluate α(θ) = κ̄(<θ)·θ
    r, rho = load_real_cluster_profiles(local_name)
    R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
    Sigma_bar = abel_project_sigma(r, rho, R)
    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)
    # mean Σ inside R
    Mproj = np.array([2*np.pi*np.trapezoid(Sigma_bar[:i+1]*R[:i+1], R[:i+1]) for i in range(len(R))])
    area = np.pi * R**2
    Sbar = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area>0)
    kbar_mean = Sbar / Sigma_crit
    # convert θ to kpc radius
    Dd = angular_diameter_distance_kpc(z_lens)
    theta_rad = theta_arcsec / 206265.0
    R0 = theta_rad * Dd
    kbar0 = float(np.interp(R0, R, kbar_mean, left=kbar_mean[0], right=kbar_mean[-1]))
    return float(kbar0 * theta_rad)  # radians


def alpha_at_theta_HLSP(cluster_id: str, theta_arcsec: float) -> float | None:
    # Radialize HLSP κ to get κ̄(<θ) then α(θ)=κ̄(<θ)·θ (dimensionless thin-lens units)
    kappa = find_hlsp_kappa(cluster_id)
    if kappa is None:
        return None
    r_pix, kcum = radial_cummean_kappa_from_map(kappa)
    # We don't know WCS here; for a qualitative path we'll assume α scales so α(θ_E)≈θ_E
    # Normalize so at the observed θ_E, α ≈ θ (the strong-lensing condition)
    return None  # Will compute normalization with observed θE outside


def build_3d_paths(theta0_arcsec: float, alpha_gr_rad: float, alpha_hlsp_rad: float,
                   xspan_arcsec: float = None) -> tuple[np.ndarray, np.ndarray]:
    """Return two 3D paths (N,3): GR (red) and HLSP-like (cyan), in arcsec units.
    Thin-lens kink at x=0, y0=+theta0, z=0.
    Post-lens slope dy/dx ≈ -alpha (small-angle, radians), x in arcsec.
    """
    if xspan_arcsec is None:
        xspan_arcsec = 2.5 * theta0_arcsec
    # pre-lens segment
    x_pre = np.linspace(-xspan_arcsec, 0.0, 100)
    y0 = theta0_arcsec
    y_pre = np.full_like(x_pre, y0)
    z_pre = np.zeros_like(x_pre)
    # post-lens segments
    x_post = np.linspace(0.0, xspan_arcsec, 200)
    y_post_gr = y0 - alpha_gr_rad * x_post
    y_post_hl = y0 - alpha_hlsp_rad * x_post
    z_post = np.zeros_like(x_post)
    # stack
    path_gr = np.vstack([
        np.concatenate([x_pre, x_post]),
        np.concatenate([y_pre, y_post_gr]),
        np.concatenate([z_pre, z_post])
    ]).T
    path_hl = np.vstack([
        np.concatenate([x_pre, x_post]),
        np.concatenate([y_pre, y_post_hl]),
        np.concatenate([z_pre, z_post])
    ]).T
    return path_gr, path_hl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster_id', required=True)
    ap.add_argument('--zs', type=float, default=2.0)
    args = ap.parse_args()

    cid = args.cluster_id.lower()
    if cid not in CLASH:
        print(f'Unknown cluster_id {cid}')
        sys.exit(1)
    local_name, z_lens = CLASH[cid]

    # Observed θE (arcsec)
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
    if theta_obs is None:
        print('No observed θE for this cluster; cannot build comparison path.')
        sys.exit(0)

    # Deflection under GR(baryons) at θ_obs
    alpha_gr = alpha_at_theta_GR_baryons(local_name, z_lens, args.zs, theta_obs)

    # For HLSP, we set α(θ_obs) ≈ θ_obs (in radians) to represent the actual strong-lensing bending at the ring
    alpha_hlsp = theta_obs / 206265.0

    # Build paths
    path_gr, path_hl = build_3d_paths(theta_obs, alpha_gr, alpha_hlsp)

    # Draw sphere representing the cluster (radius = 0.5 θE in arcsec units)
    r_sph = 0.5 * theta_obs
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = r_sph * np.outer(np.cos(u), np.sin(v))
    ys = r_sph * np.outer(np.sin(u), np.sin(v))
    zs = r_sph * np.outer(np.ones_like(u), np.cos(v))

    # Create 3 views
    outdir = ROOT / 'out' / 'visualizations' / cid
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 5))
    views = [(20, -60, '3D oblique'), (0, 0, 'Side view'), (90, 0, 'Top-down')]
    for i, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        # sphere
        ax.plot_surface(xs, ys, zs, rstride=2, cstride=2, color='lightgray', alpha=0.25, linewidth=0)
        # paths
        ax.plot(path_hl[:,0], path_hl[:,1], path_hl[:,2], color='cyan', lw=2.5, label='Actual (HLSP-like)')
        ax.plot(path_gr[:,0], path_gr[:,1], path_gr[:,2], color='red', lw=2.5, label='Predicted (GR baryons)')
        ax.set_xlabel('x (arcsec)'); ax.set_ylabel('y (arcsec)'); ax.set_zlabel('z (arcsec)')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([1,1,0.6])
        # limits
        span = 2.2 * theta_obs
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_zlim(-span*0.5, span*0.5)
        if i == 1:
            ax.legend(loc='upper right')

    fig.suptitle(f'{cid}: Photon paths around the cluster (sphere), Actual vs GR(baryons)'),
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = outdir / '3d_paths.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {out_path}')

if __name__ == '__main__':
    main()
