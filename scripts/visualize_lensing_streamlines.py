#!/usr/bin/env python3
from __future__ import annotations
"""
Lensing streamlines visualization: show photon paths (thin-lens kink) under
(1) GR(baryons-only, spherical) and (2) HLSP κ model (if available, radialized).

Outputs (per cluster):
  out/visualizations/<cluster_id>/rays_topdown.png
  out/visualizations/<cluster_id>/lens_equation_beta_vs_theta.png

Notes
- Top-down: incoming parallel rays (from left) bend at the lens plane by deflection
  angle α(R). We draw piecewise-linear paths with a single kink at x=0.
- For GR(baryons): we build Σ_bar(R) via Abel projection, κ̄(<R) = Sbar/Σ_crit,
  and use α(R) = κ̄(<R) * R (axisymmetric scaled units).
- For HLSP: we load a κ map (if present), circularly average to κ(r) (pixel units),
  and compute κ̄(<r) by cumulative area average in pixels; α(r) = κ̄(<r) * r in those units.
  It’s qualitative (no WCS), so we plot in separate panels with unit labels.
- Lens-equation panel (baryons-only): show β(θ) = θ - α(θ) to illustrate Einstein
  radius (β=0). HLSP panel is analogous but in pixel units.
"""
import argparse
from pathlib import Path
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    load_real_cluster_profiles, abel_project_sigma, sigma_crit_Msun_per_kpc2,
)

try:
    from astropy.io import fits  # type: ignore
    ASTROPY_OK = True
except Exception:
    ASTROPY_OK = False

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

def find_hlsp_kappa(cluster_id: str) -> np.ndarray | None:
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


def radial_cummean_kappa_from_map(kappa2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (r_pix, kappa_radial_mean, kappa_cummean) where cummean ≈ κ̄(<r)."""
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

    # Azimuthal mean per annulus
    kmean = np.zeros_like(rmid)
    counts = np.zeros_like(rmid)
    for i in range(nbins):
        m = (R >= rbins[i]) & (R < rbins[i+1])
        if np.any(m):
            kmean[i] = float(np.mean(img[m]))
            counts[i] = int(np.sum(m))
        else:
            kmean[i] = np.nan
    # Cumulative area-weighted mean: κ̄(<r) ~ sum(κ * area) / area_total
    # area per pixel is constant; cumulative mean reduces to cumulative average by counts
    kfill = np.nan_to_num(kmean, nan=0.0)
    csum = np.cumsum(kfill * np.maximum(counts, 0))
    ccount = np.cumsum(np.maximum(counts, 0))
    with np.errstate(invalid='ignore', divide='ignore'):
        kcum = np.where(ccount > 0, csum / ccount, 0.0)
    return rmid, kmean, kcum


def alpha_from_kappa_bar(R: np.ndarray, kappa_bar_mean: np.ndarray) -> np.ndarray:
    R = np.asarray(R, float)
    kbar = np.asarray(kappa_bar_mean, float)
    # Axisymmetric scaled deflection: α(R) = κ̄(<R) * R
    return kbar * R


def draw_topdown_rays(ax, alpha_of_R, R_grid, label: str, color: str = 'C0', n_rays: int = 9, Xspan: float = 2.0):
    """Draw piecewise-linear rays with a kink at x=0. Units follow R_grid.
    - alpha_of_R: callable or sampled pairs to interpolate α(R).
    - R_grid: radii for interpolation (monotonic).
    - Xspan: half-span in x to draw (from -Xspan to +Xspan).
    """
    Rg = np.asarray(R_grid, float)
    Ag = np.asarray(alpha_of_R(Rg) if callable(alpha_of_R) else alpha_of_R, float)
    # Build interpolator
    def a_of(r):
        return float(np.interp(abs(r), Rg, Ag, left=Ag[0], right=Ag[-1]))
    ys = np.linspace(-0.9*Rg[-1], 0.9*Rg[-1], n_rays)
    for y0 in ys:
        # pre-lens: from x=-Xspan to 0 at constant y=y0
        ax.plot([-Xspan, 0.0], [y0, y0], color=color, lw=1.5, alpha=0.9)
        # kink at lens plane: slope change dy/dx ≈ -sign(y0) * α(|y0|)
        alpha = a_of(abs(y0))
        slope = -np.sign(y0) * alpha
        # post-lens: y(x) = y0 + slope * x for x in [0, Xspan]
        xs = np.linspace(0.0, Xspan, 100)
        ys2 = y0 + slope * xs
        ax.plot(xs, ys2, color=color, lw=1.5, alpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-Xspan, Xspan)
    ax.set_ylim(-Rg[-1], Rg[-1])
    ax.set_title(label)
    ax.set_xlabel('x (arb.)'); ax.set_ylabel('y (same units as R)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster_id', required=True)
    ap.add_argument('--zs', type=float, default=2.0)
    ap.add_argument('--rays', type=int, default=9)
    args = ap.parse_args()

    cid = args.cluster_id.lower()
    if cid not in CLASH:
        print(f'Unknown cluster_id {cid}')
        sys.exit(1)
    local_name, z_lens = CLASH[cid]

    outdir = ROOT / 'out' / 'visualizations' / cid
    outdir.mkdir(parents=True, exist_ok=True)

    # Baryon-based (physical kpc grid)
    r, rho = load_real_cluster_profiles(local_name)
    R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
    Sigma_bar = abel_project_sigma(r, rho, R)
    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, args.zs)

    # Compute mean Σ inside R and kappa_bar_mean
    Mproj = np.array([2*np.pi*np.trapezoid(Sigma_bar[:i+1]*R[:i+1], R[:i+1]) for i in range(len(R))])
    area = np.pi * R**2
    Sbar = np.divide(Mproj, area, out=np.zeros_like(Mproj), where=area>0)
    kbar_mean = Sbar / Sigma_crit
    alpha_b = alpha_from_kappa_bar(R, kbar_mean)

    # HLSP radialization (pixel units)
    kappa_map = find_hlsp_kappa(cid)
    r_pix = None; alpha_h = None
    if kappa_map is not None:
        r_pix, kmean_pix, kcum_pix = radial_cummean_kappa_from_map(kappa_map)
        alpha_h = alpha_from_kappa_bar(r_pix, kcum_pix)

    # Figure: top-down rays
    cols = 2 if alpha_h is not None else 1
    fig, axes = plt.subplots(1, cols, figsize=(6*cols, 5))
    if cols == 1:
        axes = [axes]
    draw_topdown_rays(axes[0], (lambda rr: np.interp(rr, R, alpha_b)), R,
                      label='GR (baryons-only, spherical)', color='C0', n_rays=args.rays, Xspan=R[-1])
    if alpha_h is not None:
        # Use pixel-units panel, separate scale
        Xspan_pix = float(r_pix[-1])
        draw_topdown_rays(axes[1], (lambda rr: np.interp(rr, r_pix, alpha_h)), r_pix,
                          label='HLSP κ (radialized, pixel units)', color='C3', n_rays=args.rays, Xspan=Xspan_pix)
    fig.suptitle(f'{cid}: Photon ray paths (thin-lens kink). Left=GR(baryons). Right=Observed model')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(outdir / 'rays_topdown.png', dpi=150)
    plt.close(fig)

    # Figure: lens equation β(θ) = θ - α(θ)
    fig2, axes2 = plt.subplots(1, cols, figsize=(6*cols, 5))
    if cols == 1:
        axes2 = [axes2]
    theta = R
    beta_b = theta - alpha_b
    axes2[0].plot(theta, beta_b, color='C0', lw=2)
    axes2[0].axhline(0, color='k', ls=':')
    axes2[0].set_title('Lens equation (GR baryons): β(θ) = θ - α(θ)')
    axes2[0].set_xlabel('θ (kpc units)'); axes2[0].set_ylabel('β (same units)')
    if alpha_h is not None:
        theta_p = r_pix
        beta_h = theta_p - alpha_h
        axes2[1].plot(theta_p, beta_h, color='C3', lw=2)
        axes2[1].axhline(0, color='k', ls=':')
        axes2[1].set_title('Lens eq. (HLSP radialized, pixels)')
        axes2[1].set_xlabel('θ (pixels)'); axes2[1].set_ylabel('β (pixels)')
    fig2.suptitle(f'{cid}: Lens-equation mapping (Einstein radius where β=0)')
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(outdir / 'lens_equation_beta_vs_theta.png', dpi=150)
    plt.close(fig2)

    print(f'Wrote {outdir / "rays_topdown.png"}')
    print(f'Wrote {outdir / "lens_equation_beta_vs_theta.png"}')

if __name__ == '__main__':
    main()
