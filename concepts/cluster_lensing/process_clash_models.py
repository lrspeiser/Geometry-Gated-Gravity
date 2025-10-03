#!/usr/bin/env python3
"""
Process CLASH lensing models (derived products) to extract Einstein radii and κ(R)
===================================================================================

Inputs (downloaded beforehand):
- data/clash/hlsp/<cluster>/models/.../*kappa*.fits, *gamma*.fits (or gamma1/gamma2)
- Optionally catalogs, params.txt/readme.txt for metadata (not required for core outputs)

Outputs:
- data/clash/processed/einstein_radii_clash.csv
    Columns: cluster_id,cluster_label,z_lens,method_detA_arcsec,method_kappaMean_arcsec,pixel_scale_arcsec,model_dir
- data/clash/processed/profiles/<cluster>_kappa_profile.csv
    Columns: radius_arcsec,kappa_mean_within,kappa_ring_mean
- data/clash/processed/summaries.json (array of dicts per cluster)

Notes
- Critical curve method: uses detA = (1-κ)^2 - |γ|^2; the equivalent Einstein radius is sqrt(Area_tangential/π)
  for the connected component containing the κ peak.
- Mean-κ method: solves for radius where \bar{κ}(<R)=1 using cumulative mean over circular apertures.
- Pixel scale: estimated from FITS WCS CD matrix (deg/pixel) converted to arcsec; falls back to CDELT.
- Source-z normalization: CLASH “base” κ,γ maps may correspond to a particular normalization; both methods
  operate self-consistently on the provided maps. If you require θ_E at a specific source redshift, supply
  scale factors per CLASH documentation and re-scale (future extension).

Usage
- python concepts/cluster_lensing/process_clash_models.py
"""
from __future__ import annotations
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

try:
    from skimage.measure import label, regionprops
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

# Root paths
ROOT = Path(__file__).resolve().parents[2]
HLSP_DIR = ROOT / 'data' / 'clash' / 'hlsp'
OUT_DIR = ROOT / 'data' / 'clash' / 'processed'
PROFILES_DIR = OUT_DIR / 'profiles'

# CLASH cluster labels and redshifts (from MAST CLASH page)
CLASH_CLUSTERS: Dict[str, Tuple[str, float]] = {
    'a1423':   ('Abell 1423', 0.213),
    'a209':    ('Abell 209', 0.206),
    'a2261':   ('Abell 2261', 0.224),
    'a383':    ('Abell 383', 0.187),
    'a611':    ('Abell 611', 0.288),
    'clj1226': ('CLJ1226+3332', 0.890),
    'macs0329':('MACSJ0329-02', 0.450),
    'macs0416':('MACSJ0416-24', 0.396),
    'macs0429':('MACSJ0429-02', 0.399),
    'macs0647':('MACSJ0647+70', 0.584),
    'macs0717':('MACSJ0717+37', 0.548),
    'macs0744':('MACSJ0744+39', 0.686),
    'macs1115':('MACSJ1115+01', 0.352),
    'macs1149':('MACSJ1149+22', 0.544),
    'macs1206':('MACSJ1206-08', 0.440),
    'macs1311':('MACSJ1311-03', 0.494),
    'macs1423':('MACSJ1423+24', 0.545),
    'macs1720':('MACSJ1720+35', 0.391),
    'macs1931':('MACSJ1931-26', 0.352),
    'macs2129':('MACSJ2129-07', 0.570),
    'ms2137':  ('MS 2137.3-2353', 0.313),
    'rxj1347': ('RXJ1347-1145', 0.451),
    'rxj1532': ('RXJ1532.9+3021', 0.345),
    'rxj2129': ('RXJ2129+0005', 0.234),
    'rxj2248': ('RXJ2248-4431', 0.348),
}


def pixel_scale_arcsec(h: fits.Header) -> Optional[float]:
    # Prefer CD matrix
    for a, b in [('CD1_1','CD2_2'), ('CDELT1','CDELT2')]:
        if a in h and b in h and h[a] not in (0, None) and h[b] not in (0, None):
            try:
                # degrees per pixel to arcsec per pixel
                sx = abs(float(h[a])) * 3600.0
                sy = abs(float(h[b])) * 3600.0
                return float(np.sqrt(max(sx*sy, 1e-20)))
            except Exception:
                continue
    # Try CD matrix full form
    if 'CD1_1' in h and 'CD1_2' in h and 'CD2_1' in h and 'CD2_2' in h:
        try:
            cd11, cd12, cd21, cd22 = [float(h[k]) for k in ('CD1_1','CD1_2','CD2_1','CD2_2')]
            # pixel area scaling: sqrt(|det(CD)|) in deg/pix, convert to arcsec
            det = cd11*cd22 - cd12*cd21
            s = math.sqrt(abs(det)) * 3600.0
            if s > 0:
                return s
        except Exception:
            pass
    return None


def load_fits_data(path: Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = np.array(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header
    return data, hdr


def find_model_maps(cluster_dir: Path) -> Optional[Dict[str, Path]]:
    """Find base kappa and gamma maps for a cluster under its models/ directory.
    Returns dict with keys: kappa, gamma or gamma1, gamma2 (paths).
    Chooses a preferred set by prioritizing zitrin/nfw/*/v2 if available.
    """
    if not cluster_dir.exists():
        return None
    candidates: List[Tuple[int, Path]] = []
    for p in cluster_dir.rglob("*kappa*.fits"):
        # Skip heavy parameter sweep subtrees named 'range'
        if re.search(r"\\/range(\\/|$)", str(p).replace('\\', '/')):
            continue
        # Prefer zitrin/nfw/v2 paths (rank 0), else rank by path length
        rank = 1
        s = str(p).replace('\\', '/')
        if '/zitrin/' in s and '/nfw/' in s and '/v2/' in s:
            rank = 0
        candidates.append((rank, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], len(str(x[1]))))
    kappa_path = candidates[0][1]
    base_dir = kappa_path.parent
    # Try to locate gamma components in same dir
    gamma1 = None
    gamma2 = None
    gamma = None
    for q in base_dir.glob("*gamma1*.fits"):
        gamma1 = q; break
    for q in base_dir.glob("*gamma2*.fits"):
        gamma2 = q; break
    if gamma1 is None or gamma2 is None:
        for q in base_dir.glob("*gamma.fits"):
            gamma = q; break
    return {
        'kappa': kappa_path,
        **({'gamma1': gamma1} if gamma1 is not None else {}),
        **({'gamma2': gamma2} if gamma2 is not None else {}),
        **({'gamma': gamma} if gamma is not None else {}),
    }


def compute_einstein_radius_from_detA(kappa: np.ndarray, gamma1: Optional[np.ndarray], gamma2: Optional[np.ndarray],
                                       gamma_amp: Optional[np.ndarray], px_scale_arcsec: float) -> Optional[float]:
    # Center = pixel of maximal kappa
    if not np.isfinite(kappa).any():
        return None
    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    center_idx = np.unravel_index(np.argmax(kappa), kappa.shape)
    if gamma1 is not None and gamma2 is not None:
        g1 = np.nan_to_num(gamma1)
        g2 = np.nan_to_num(gamma2)
        gamma2_mag = g1*g1 + g2*g2
    elif gamma_amp is not None:
        g = np.nan_to_num(gamma_amp)
        gamma2_mag = g*g
    else:
        # Without gamma we cannot form detA reliably
        return None
    detA = (1.0 - kappa)**2 - gamma2_mag
    mask = detA <= 0.0
    if not mask.any():
        return None
    # Use connected component that contains the kappa-peak pixel; else fall back to largest region
    if HAVE_SKIMAGE:
        lbl = label(mask)
        cid = lbl[center_idx]
        if cid == 0:
            # pick largest
            props = regionprops(lbl)
            if not props:
                return None
            cid = max(props, key=lambda r: r.area).label
        area_px = int((lbl == cid).sum())
    else:
        # Fallback: take all tangential region as one (overestimates if disjoint)
        area_px = int(mask.sum())
    if area_px <= 0:
        return None
    area_arcsec2 = area_px * (px_scale_arcsec**2)
    theta_E = math.sqrt(area_arcsec2 / math.pi)
    return float(theta_E)


def compute_einstein_radius_from_mean_kappa(kappa: np.ndarray, px_scale_arcsec: float) -> Optional[float]:
    # Center at kappa peak
    k = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    cy, cx = np.unravel_index(np.argmax(k), k.shape)
    yy, xx = np.indices(k.shape)
    rr_px = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    # Sort pixels by radius and compute cumulative mean inside radius
    order = np.argsort(rr_px.ravel())
    r_sorted = rr_px.ravel()[order]
    k_sorted = k.ravel()[order]
    csum = np.cumsum(k_sorted)
    counts = np.arange(1, k_sorted.size + 1)
    kmean_inside = csum / counts
    # Find first index where mean >= 1
    idx = np.where(kmean_inside >= 1.0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    r0 = r_sorted[i-1] if i > 0 else r_sorted[i]
    r1 = r_sorted[i]
    y0 = kmean_inside[i-1] if i > 0 else kmean_inside[i]
    y1 = kmean_inside[i]
    if y1 == y0:
        r_px = r1
    else:
        # linear interpolation on mean-kappa
        t = (1.0 - y0) / (y1 - y0)
        r_px = r0 + t * (r1 - r0)
    return float(r_px * px_scale_arcsec)


def radial_profile(kappa: np.ndarray, px_scale_arcsec: float, nbins: int = 150) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    cy, cx = np.unravel_index(np.argmax(k), k.shape)
    yy, xx = np.indices(k.shape)
    rr_px = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    r_arc = rr_px * px_scale_arcsec
    rmax = np.percentile(r_arc, 95)  # ignore extreme corners
    edges = np.linspace(0.0, rmax, nbins + 1)

    ring_mean = np.zeros(nbins, dtype=np.float64)
    inside_mean = np.zeros(nbins, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    flat_k = k.ravel()
    flat_r = r_arc.ravel()

    # Pre-sort by radius for cumulative inside mean
    order = np.argsort(flat_r)
    r_sorted = flat_r[order]
    k_sorted = flat_k[order]
    csum = np.cumsum(k_sorted)
    counts = np.arange(1, k_sorted.size + 1)
    kmean_inside_all = csum / counts

    for i in range(nbins):
        m = (flat_r >= edges[i]) & (flat_r < edges[i+1])
        ring_mean[i] = float(np.mean(flat_k[m])) if np.any(m) else np.nan
        # inside mean at upper edge
        j = np.searchsorted(r_sorted, edges[i+1], side='right') - 1
        if j >= 0:
            inside_mean[i] = float(kmean_inside_all[j])
        else:
            inside_mean[i] = np.nan

    return centers, inside_mean, ring_mean


def process_cluster(cluster_id: str, cluster_root: Path) -> Optional[Dict]:
    models_dir = cluster_root / 'models'
    maps = find_model_maps(models_dir)
    if not maps or 'kappa' not in maps:
        return None
    kappa, hdr = load_fits_data(maps['kappa'])
    ps = pixel_scale_arcsec(hdr)
    if not ps or ps <= 0:
        # Attempt rough fallback from header keywords typical in CLASH
        ps = 0.065  # arcsec/pixel (65mas) as conservative default; updated if WCS present
    gamma1 = gamma2 = gamma = None
    if 'gamma1' in maps and 'gamma2' in maps:
        gamma1, _ = load_fits_data(maps['gamma1'])
        gamma2, _ = load_fits_data(maps['gamma2'])
    elif 'gamma' in maps:
        gamma, _ = load_fits_data(maps['gamma'])

    # Compute θ_E via detA and mean-κ
    theta_detA = compute_einstein_radius_from_detA(kappa, gamma1, gamma2, gamma, ps)
    theta_kmean = compute_einstein_radius_from_mean_kappa(kappa, ps)

    # Build radial profile CSV
    r_arc, kmean_inside, kmean_ring = radial_profile(kappa, ps)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    prof_path = PROFILES_DIR / f"{cluster_id}_kappa_profile.csv"
    with prof_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['radius_arcsec', 'kappa_mean_within', 'kappa_ring_mean'])
        for R, km, kr in zip(r_arc, kmean_inside, kmean_ring):
            w.writerow([f"{R:.6f}", f"{km:.6f}" if np.isfinite(km) else '', f"{kr:.6f}" if np.isfinite(kr) else ''])

    label_name, z_lens = CLASH_CLUSTERS.get(cluster_id, (cluster_id, None))
    return {
        'cluster_id': cluster_id,
        'cluster_label': label_name,
        'z_lens': z_lens,
        'pixel_scale_arcsec': ps,
        'theta_E_arcsec_detA': theta_detA,
        'theta_E_arcsec_kappaMean': theta_kmean,
        'model_dir': str(models_dir)
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []

    def _fmt(x: Optional[float]) -> str:
        try:
            if x is None:
                return 'nan'
            xf = float(x)
            if not math.isfinite(xf):
                return 'nan'
            return f"{xf:.2f}"
        except Exception:
            return 'nan'

    for cluster_id in sorted(CLASH_CLUSTERS.keys()):
        cdir = HLSP_DIR / cluster_id
        if not cdir.exists():
            continue
        try:
            res = process_cluster(cluster_id, cdir)
            if res:
                results.append(res)
                detA_s = _fmt(res.get('theta_E_arcsec_detA'))
                kmean_s = _fmt(res.get('theta_E_arcsec_kappaMean'))
                print(f"Processed {cluster_id}: θ_E(detA)={detA_s} arcsec, θ_E(κ̄=1)={kmean_s} arcsec")
            else:
                print(f"Skipped {cluster_id}: no maps found")
        except Exception as e:
            print(f"ERROR {cluster_id}: {e}")

    # Write CSV summary
    csv_path = OUT_DIR / 'einstein_radii_clash.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cluster_id','cluster_label','z_lens','method_detA_arcsec','method_kappaMean_arcsec','pixel_scale_arcsec','model_dir'])
        for r in results:
            w.writerow([
                r['cluster_id'], r['cluster_label'], r['z_lens'] if r['z_lens'] is not None else '',
                f"{r['theta_E_arcsec_detA']:.6f}" if r['theta_E_arcsec_detA'] is not None else '',
                f"{r['theta_E_arcsec_kappaMean']:.6f}" if r['theta_E_arcsec_kappaMean'] is not None else '',
                f"{r['pixel_scale_arcsec']:.6f}", r['model_dir']
            ])

    # JSON summary
    json_path = OUT_DIR / 'summaries.json'
    with json_path.open('w') as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Profiles: {PROFILES_DIR}")


if __name__ == '__main__':
    main()