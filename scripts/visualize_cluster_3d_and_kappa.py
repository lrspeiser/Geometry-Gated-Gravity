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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    load_real_cluster_profiles, abel_project_sigma, sigma_crit_Msun_per_kpc2,
)

# Optional: astropy for FITS
try:
    from astropy.io import fits  # type: ignore
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


def find_hlsp_kappa(cluster_id: str) -> np.ndarray | None:
    if not ASTROPY_OK:
        return None
    base = ROOT / 'data' / 'clash' / 'hlsp' / cluster_id.lower()
    pats = [str(base / '**' / '*kappa*.fits'), str(base / '**' / '*kappa1*.fits')]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    # Prefer files with 'kappa.fits' (not kappa1/2 component) if available
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

    # Load 1D baryon profiles
    r, rho = load_real_cluster_profiles(local_name)

    # 3D sampling
    pts = sample_baryons_3d(r, rho, n=args.npoints, seed=0)

    # Projected Σ and κ_bar
    R = np.logspace(np.log10(max(1.0, r[0])), np.log10(max(1.0, r[-1])), 600)
    Sigma_bar = abel_project_sigma(r, rho, R)
    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, args.zs)
    kappa_radial = Sigma_bar / Sigma_crit

    # Build 2D κ_bar image
    Rmax = float(R[-1])
    kappa_img = kappa_bar_grid(Rmax_kpc=Rmax, npx=512, R_kpc=R, kappa_radial=kappa_radial)

    # Try to load HLSP κ map
    kappa_hlsp = find_hlsp_kappa(cid)

    outdir = ROOT / 'out' / 'visualizations' / cid
    outdir.mkdir(parents=True, exist_ok=True)

    # Figure 1: 3D baryons
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    if pts.shape[0] > 0:
        # Color by radius
        rr = np.sqrt(np.sum(pts**2, axis=1))
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=np.log10(rr+1e-6), s=1, cmap='viridis', alpha=0.5)
    ax.set_title(f'{cid}: 3D baryon cloud (samples), r_max ≈ {Rmax:.0f} kpc')
    ax.set_xlabel('x (kpc)'); ax.set_ylabel('y (kpc)'); ax.set_zlabel('z (kpc)')
    ax.set_box_aspect([1,1,1])
    fig.tight_layout(); fig.savefig(outdir / 'baryons_3d.png', dpi=150)
    plt.close(fig)

    # Figure 2: κ_bar vs HLSP κ
    cols = 2 if kappa_hlsp is not None else 1
    fig2, axes = plt.subplots(1, cols, figsize=(6*cols, 5))
    if cols == 1:
        axes = [axes]
    im0 = axes[0].imshow(kappa_img, origin='lower', extent=[-Rmax, Rmax, -Rmax, Rmax], cmap='magma')
    axes[0].set_title('κ_bar (GR, baryons) [radial, model]')
    axes[0].set_xlabel('x (kpc)'); axes[0].set_ylabel('y (kpc)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    if kappa_hlsp is not None:
        im1 = axes[1].imshow(kappa_hlsp, origin='lower', cmap='magma')
        axes[1].set_title('κ (CLASH HLSP model) [pixel grid]')
        axes[1].set_xlabel('x (pix)'); axes[1].set_ylabel('y (pix)')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig2.suptitle(f'{cid}: κ comparison (no WCS alignment)')
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(outdir / 'kappa_bar_vs_hlsp.png', dpi=150)
    plt.close(fig2)

    print(f'Wrote {outdir / "baryons_3d.png"}')
    print(f'Wrote {outdir / "kappa_bar_vs_hlsp.png"}')

if __name__ == '__main__':
    main()
