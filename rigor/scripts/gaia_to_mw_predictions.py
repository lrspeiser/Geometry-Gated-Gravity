#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

G_KPC_KM2_S2_MSUN = 4.300917270e-6  # kpc (km/s)^2 / Msun

# -----------------------------
# Baryonic baseline (MN disk + Hernquist bulge) in midplane (z=0)
# -----------------------------

def v_circ_mn_disk_kms(R_kpc: np.ndarray, M_d_Msun: float, a_kpc: float, b_kpc: float) -> np.ndarray:
    R = np.asarray(R_kpc)
    A = a_kpc + b_kpc
    denom = np.power(R*R + A*A, 1.5)
    v2 = G_KPC_KM2_S2_MSUN * M_d_Msun * (R*R) / np.maximum(denom, 1e-12)
    return np.sqrt(np.clip(v2, 0.0, None))


def v_circ_hernquist_kms(R_kpc: np.ndarray, M_b_Msun: float, a_kpc: float) -> np.ndarray:
    R = np.asarray(R_kpc)
    v2 = G_KPC_KM2_S2_MSUN * M_b_Msun * R / np.maximum((R + a_kpc) ** 2, 1e-12)
    return np.sqrt(np.clip(v2, 0.0, None))


def vbar_mn_hern_kms(R_kpc: np.ndarray,
                     M_d_Msun: float, a_d_kpc: float, b_d_kpc: float,
                     M_b_Msun: float, a_b_kpc: float) -> np.ndarray:
    vd = v_circ_mn_disk_kms(R_kpc, M_d_Msun, a_d_kpc, b_d_kpc)
    vb = v_circ_hernquist_kms(R_kpc, M_b_Msun, a_b_kpc)
    return np.sqrt(np.clip(vd*vd + vb*vb, 0.0, None))

# -----------------------------
# LogTail preview (same functional form as analysis script)
# -----------------------------

def smooth_gate_r(r_kpc: np.ndarray, r0_kpc: float, delta_kpc: float) -> np.ndarray:
    x = (np.asarray(r_kpc) - r0_kpc) / max(delta_kpc, 1e-6)
    return 0.5 * (1.0 + np.tanh(x))


def v_model_logtail_preview(vbar_kms: np.ndarray, r_kpc: np.ndarray,
                            v0_kms: float = 140.0, rc_kpc: float = 15.0,
                            r0_kpc: float = 8.0, delta_kpc: float = 3.0) -> np.ndarray:
    vbar2 = np.asarray(vbar_kms) ** 2
    r = np.asarray(r_kpc)
    S = smooth_gate_r(r, r0_kpc, delta_kpc)
    add = (v0_kms ** 2) * (r / np.maximum(r + rc_kpc, 1e-12)) * S
    v2 = vbar2 + add
    return np.sqrt(np.clip(v2, 0.0, None))

# -----------------------------
# Data ingestion & binning
# -----------------------------

def list_slices(glob_pattern: str) -> List[Path]:
    import glob
    files = sorted(glob.glob(glob_pattern))
    return [Path(f) for f in files]


def _phi_degrees_from_df(df: pd.DataFrame) -> np.ndarray | None:
    # Try to find a phi column; support 'phi_rad', 'phi_deg', 'phi'
    cols = set(df.columns)
    phi_col = None
    for c in ('phi_rad','phi_deg','phi','Phi','PHI'):
        if c in cols:
            phi_col = c
            break
    if phi_col is None:
        return None
    phi_vals = pd.to_numeric(df[phi_col], errors='coerce').to_numpy()
    # Heuristically infer units: if values exceed ~2π in magnitude, treat as degrees
    finite = phi_vals[np.isfinite(phi_vals)]
    if finite.size == 0:
        return None
    if np.nanmax(np.abs(finite)) > 6.5:
        phi_deg = phi_vals  # already degrees
    else:
        phi_deg = np.degrees(phi_vals)
    # Normalize into [0,360)
    phi_deg = np.mod(phi_deg, 360.0)
    phi_deg[phi_deg < 0] += 360.0
    return phi_deg


def stream_bins(files: List[Path],
                r_edges: np.ndarray,
                z_max: float,
                sigma_v_max: float,
                vR_max: float | None = None,
                prefer_v_obs: bool = True,
                phi_bins: int = 1,
                phi_bin_index: int | None = None) -> Dict[str, Any]:
    n_bins = len(r_edges) - 1
    sum_w = np.zeros(n_bins)
    sum_wx = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    used_files = []

    # Precompute wedge bounds if requested
    use_wedge = (phi_bins is not None) and (phi_bins > 1) and (phi_bin_index is not None)
    if use_wedge:
        wedge_width = 360.0 / float(phi_bins)
        phi0 = wedge_width * int(phi_bin_index)
        phi1 = phi0 + wedge_width

    for p in files:
        try:
            # Read only necessary columns, fallback gracefully
            cols_try = ['R_kpc','z_kpc','v_phi_kms','sigma_v','v_obs','quality_flag','v_R_kms','phi_rad','phi_deg','phi']
            # Get available columns quickly
            try:
                available = pd.read_parquet(p, engine='pyarrow').columns
            except Exception:
                available = None
            if available is not None:
                df = pd.read_parquet(p, columns=[c for c in cols_try if c in available])
            else:
                df = pd.read_parquet(p)
        except Exception:
            continue

        # Normalize column names that may vary
        cols = set(df.columns)
        col_R = 'R_kpc' if 'R_kpc' in cols else next((c for c in cols if c.lower() == 'r_kpc'), None)
        col_z = 'z_kpc' if 'z_kpc' in cols else next((c for c in cols if c.lower() == 'z_kpc'), None)
        col_vphi = 'v_phi_kms' if 'v_phi_kms' in cols else next((c for c in cols if c.lower() in ('vphi_kms','v_phi','vphi')), None)
        col_vobs = 'v_obs' if 'v_obs' in cols else next((c for c in cols if c.lower() in ('vobs','v_obs_kms')), None)
        col_sv = 'sigma_v' if 'sigma_v' in cols else next((c for c in cols if c.lower() in ('sigma','sigma_v_kms','sigmav')), None)
        col_q = 'quality_flag' if 'quality_flag' in cols else next((c for c in cols if 'quality' in c.lower() and 'flag' in c.lower()), None)
        col_vR = 'v_R_kms' if 'v_R_kms' in cols else next((c for c in cols if c.lower() in ('v_r_kms','vr_kms','v_r')), None)

        if col_R is None or (col_vphi is None and col_vobs is None) or col_sv is None:
            continue

        R = pd.to_numeric(df[col_R], errors='coerce').to_numpy()
        z = pd.to_numeric(df[col_z], errors='coerce').to_numpy() if (col_z in cols) else np.zeros_like(R)
        sv = pd.to_numeric(df[col_sv], errors='coerce').to_numpy()
        # ϕ wedge if available
        if use_wedge:
            phi_deg = _phi_degrees_from_df(df)
        else:
            phi_deg = None

        x_raw = None
        if prefer_v_obs and (col_vobs in cols):
            x_raw = pd.to_numeric(df[col_vobs], errors='coerce').to_numpy()
        elif col_vphi in cols:
            x_raw = np.abs(pd.to_numeric(df[col_vphi], errors='coerce').to_numpy())
        else:
            continue
        if x_raw is None:
            continue

        mask = np.isfinite(R) & np.isfinite(x_raw) & np.isfinite(sv)
        if z_max is not None:
            mask &= (np.abs(z) <= float(z_max))
        if sigma_v_max is not None:
            mask &= (sv <= float(sigma_v_max))
        if (col_q in cols):
            q = pd.to_numeric(df[col_q], errors='coerce').to_numpy()
            mask &= (q == 0)
        if (vR_max is not None) and (col_vR in cols):
            vR = pd.to_numeric(df[col_vR], errors='coerce').to_numpy()
            mask &= (np.abs(vR) <= float(vR_max))
        if use_wedge and (phi_deg is not None):
            mask &= (phi_deg >= phi0) & (phi_deg < phi1)

        Rm = R[mask]; Xm = x_raw[mask]; Svm = sv[mask]
        if Rm.size == 0:
            continue

        # Weighted accumulation per bin
        idx = np.digitize(Rm, r_edges) - 1
        valid = (idx >= 0) & (idx < n_bins)
        if not np.any(valid):
            continue
        idx = idx[valid]; Xm = Xm[valid]; Svm = Svm[valid]
        # weights (inverse variance) with small floor to avoid blow-ups
        w = 1.0 / np.maximum(Svm*Svm, 1e-6)
        np.add.at(sum_w, idx, w)
        np.add.at(sum_wx, idx, w * Xm)
        np.add.at(counts, idx, 1)

        used_files.append(str(p))

    centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    with np.errstate(invalid='ignore', divide='ignore'):
        v_obs = sum_wx / np.maximum(sum_w, 1e-12)
        v_err = 1.0 / np.sqrt(np.maximum(sum_w, 1e-12))

    return {
        'r_centers': centers,
        'counts': counts,
        'v_obs': v_obs,
        'v_err': v_err,
        'used_files': sorted(set(used_files)),
    }

# -----------------------------
# Inner-fit of baryonic baseline (coarse grid + local jitter)
# -----------------------------

def fit_baryon_baseline(R_kpc: np.ndarray, V_obs_kms: np.ndarray, V_err_kms: np.ndarray,
                        r_fit_min: float, r_fit_max: float,
                        seed: int = 1337) -> Dict[str, float]:
    R = np.asarray(R_kpc)
    V = np.asarray(V_obs_kms)
    E = np.asarray(V_err_kms)
    m = np.isfinite(R) & np.isfinite(V) & (R >= r_fit_min) & (R <= r_fit_max)
    if m.sum() < 6:
        # Fallback simple curve: power law
        return {
            'M_d_Msun': 5.0e10, 'a_d_kpc': 3.0, 'b_d_kpc': 0.3,
            'M_b_Msun': 8.0e9,  'a_b_kpc': 0.5,
        }
    Rf, Vf, Ef = R[m], V[m], np.maximum(E[m], 5.0)

    def sse(params: Tuple[float,float,float,float,float]) -> float:
        Md, ad, bd, Mb, ab = params
        Vbar = vbar_mn_hern_kms(Rf, Md, ad, bd, Mb, ab)
        w = 1.0/np.maximum(Ef*Ef, 25.0)
        return float(np.sum(w * (Vbar - Vf)**2))

    # Coarse grid
    Md_grid = np.array([3e10, 4e10, 5e10, 6e10, 7e10, 8e10])
    ad_grid = np.array([2.0, 3.0, 4.0, 5.0])
    bd_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    Mb_grid = np.array([3e9, 6e9, 1e10, 1.5e10])
    ab_grid = np.array([0.2, 0.3, 0.5, 0.8, 1.2])

    best = None
    best_loss = math.inf
    for Md in Md_grid:
        for ad in ad_grid:
            for bd in bd_grid:
                for Mb in Mb_grid:
                    for ab in ab_grid:
                        loss = sse((Md, ad, bd, Mb, ab))
                        if loss < best_loss:
                            best_loss = loss
                            best = (Md, ad, bd, Mb, ab)

    # Local random search around best
    rng = np.random.default_rng(seed)
    Md0, ad0, bd0, Mb0, ab0 = best
    for _ in range(300):
        Md = float(Md0 * np.exp(rng.normal(0.0, 0.15)))
        ad = float(max(0.3, ad0 * np.exp(rng.normal(0.0, 0.15))))
        bd = float(np.clip(bd0 * np.exp(rng.normal(0.0, 0.15)), 0.05, 1.0))
        Mb = float(Mb0 * np.exp(rng.normal(0.0, 0.2)))
        ab = float(max(0.1, ab0 * np.exp(rng.normal(0.0, 0.2))))
        loss = sse((Md, ad, bd, Mb, ab))
        if loss < best_loss:
            best_loss = loss
            best = (Md, ad, bd, Mb, ab)

    Md, ad, bd, Mb, ab = best
    return {
        'M_d_Msun': float(Md), 'a_d_kpc': float(ad), 'b_d_kpc': float(bd),
        'M_b_Msun': float(Mb), 'a_b_kpc': float(ab),
    }

# -----------------------------
# Main CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description='Build Milky Way predictions_by_radius.csv from Gaia Parquet slices for LogTail pipeline.')
    ap.add_argument('--slices', default=str(Path('data')/'gaia_sky_slices'/'processed_*.parquet'), help='Glob for Gaia slice Parquet files')
    ap.add_argument('--out_csv', default=str(Path('out')/'mw'/'mw_predictions_by_radius.csv'))
    ap.add_argument('--meta_out', default=str(Path('out')/'mw'/'mw_predictions_by_radius.meta.json'))
    ap.add_argument('--gal_id', default='MilkyWay')
    # radius range (accept lowercase and uppercase synonyms)
    ap.add_argument('--r_min', type=float, default=None)
    ap.add_argument('--r_max', type=float, default=None)
    ap.add_argument('--dr', type=float, default=None)
    ap.add_argument('--R_min', type=float, default=None)
    ap.add_argument('--R_max', type=float, default=None)
    ap.add_argument('--dR_bin', type=float, default=None, help='Alias for dr (bin width)')
    ap.add_argument('--z_max', type=float, default=1.0)
    ap.add_argument('--sigma_v_max', type=float, default=20.0)
    ap.add_argument('--vR_max', type=float, default=None)
    ap.add_argument('--phi_bins', type=int, default=1, help='Number of azimuthal wedges (ϕ). If >1, use with --phi_bin_index to select one wedge.')
    ap.add_argument('--phi_bin_index', type=int, default=None, help='Index of wedge [0..phi_bins-1] to include')
    ap.add_argument('--inner_fit_min', type=float, default=3.0)
    ap.add_argument('--inner_fit_max', type=float, default=8.0)
    ap.add_argument('--min_bin_count', type=int, default=20)
    ap.add_argument('--compute_logtail_preview', action='store_true')
    ap.add_argument('--v0', type=float, default=140.0)
    ap.add_argument('--rc', type=float, default=15.0)
    ap.add_argument('--R0', type=float, default=8.0)
    ap.add_argument('--dR', type=float, default=3.0, help='LogTail gate width (Δ)')
    ap.add_argument('--write_meta', action='store_true', help='If set, write meta JSON (default on)')
    args = ap.parse_args()

    # Resolve range args with synonyms
    r_min = args.r_min if args.r_min is not None else (args.R_min if args.R_min is not None else 3.0)
    r_max = args.r_max if args.r_max is not None else (args.R_max if args.R_max is not None else 25.0)
    dr_bin = args.dr if args.dr is not None else (args.dR_bin if args.dR_bin is not None else 0.5)

    files = list_slices(args.slices)
    if not files:
        raise SystemExit(f"No Parquet slices matched: {args.slices}")

    r_edges = np.arange(float(r_min), float(r_max) + float(dr_bin) + 1e-9, float(dr_bin))
    bins = stream_bins(files, r_edges, z_max=args.z_max, sigma_v_max=args.sigma_v_max, vR_max=args.vR_max,
                       phi_bins=int(args.phi_bins) if args.phi_bins is not None else 1,
                       phi_bin_index=int(args.phi_bin_index) if args.phi_bin_index is not None else None)

    R = bins['r_centers']
    Vobs = bins['v_obs']
    Verr = bins['v_err']
    N = bins['counts']

    # Filter bins by occupancy and finiteness
    good = np.isfinite(R) & np.isfinite(Vobs) & (N >= int(args.min_bin_count))
    Rg, Vg, Eg = R[good], Vobs[good], np.where(np.isfinite(Verr[good]), Verr[good], np.nan)

    # Fit baryonic baseline on inner window
    pars = fit_baryon_baseline(Rg, Vg, Eg, r_fit_min=float(args.inner_fit_min), r_fit_max=float(args.inner_fit_max))
    Vbar = vbar_mn_hern_kms(Rg, pars['M_d_Msun'], pars['a_d_kpc'], pars['b_d_kpc'], pars['M_b_Msun'], pars['a_b_kpc'])

    # Outer mask (top 30% of available good bins by radius)
    if len(Rg) > 0:
        k = max(1, int(0.3 * len(Rg)))
        order = np.argsort(Rg)
        outer_idx = np.zeros(len(Rg), dtype=bool)
        outer_idx[order[-k:]] = True
    else:
        outer_idx = np.zeros(0, dtype=bool)

    data = {
        'gal_id': [str(args.gal_id)] * len(Rg),
        'r_kpc': Rg,
        'vbar_kms': Vbar,
        'v_obs_kms': Vg,
        'is_outer': outer_idx,
    }

    if args.compute_logtail_preview:
        Vprev = v_model_logtail_preview(Vbar, Rg, v0_kms=float(args.v0), rc_kpc=float(args.rc), r0_kpc=float(args.R0), delta_kpc=float(args.dR))
        data['v_LogTail_preview_kms'] = Vprev

    out_df = pd.DataFrame(data)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    meta = {
        'slices_glob': args.slices,
        'used_files': bins['used_files'],
        'r_min': float(r_min), 'r_max': float(r_max), 'dr': float(dr_bin),
        'z_max': float(args.z_max), 'sigma_v_max': float(args.sigma_v_max), 'vR_max': None if args.vR_max is None else float(args.vR_max),
        'phi_bins': int(args.phi_bins) if args.phi_bins is not None else 1,
        'phi_bin_index': None if args.phi_bin_index is None else int(args.phi_bin_index),
        'inner_fit_min': float(args.inner_fit_min), 'inner_fit_max': float(args.inner_fit_max), 'min_bin_count': int(args.min_bin_count),
        'fit_params': pars,
        'logtail_preview': {
            'enabled': bool(args.compute_logtail_preview),
            'v0': float(args.v0), 'rc': float(args.rc), 'R0': float(args.R0), 'dR': float(args.dR),
        },
        'gal_id': str(args.gal_id),
    }
    meta_path = Path(args.meta_out)
    if args.write_meta or True:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Wrote {out_path} (rows={len(out_df)})")
    if args.write_meta or True:
        print(f"Meta: {meta_path}")


if __name__ == '__main__':
    main()
