# run_pipeline.py
# Local Gaia MW pipeline: ingest local data, bin, fit baryons (inner), detect boundary,
# fit anchored saturated-well and NFW, and plot.

from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd

from .utils import setup_logging, write_json, xp_name, get_xp, G_KPC
from .data_io import detect_source, load_slices, load_mw_csv
from .rotation import bin_rotation_curve
from .models import v_c_baryon, v2_saturated_extra, v_c_nfw, v_flat_from_anchor, lensing_alpha_arcsec
from .boundary import fit_baryons_inner, find_boundary_bic, find_boundary_consecutive, bootstrap_boundary, fit_saturated_well, fit_nfw, compute_metrics
from .plotting import make_plot


def main():
    ap = argparse.ArgumentParser(description='Gaia MW max-depth test (local data only).')
    ap.add_argument('--use_source', choices=['auto','slices','mw_csv'], default='auto')
    ap.add_argument('--slices_glob', default=os.path.join('data','gaia_sky_slices','processed_*.parquet'))
    ap.add_argument('--mw_csv_path', default=os.path.join('data','gaia_mw_real.csv'))
    ap.add_argument('--zmax', type=float, default=0.5)
    ap.add_argument('--sigma_vmax', type=float, default=30.0)
    ap.add_argument('--vRmax', type=float, default=40.0)
    ap.add_argument('--phi_bins', type=int, default=1)
    ap.add_argument('--phi_bin_index', type=int, default=None)

    ap.add_argument('--rmin', type=float, default=3.0)
    ap.add_argument('--rmax', type=float, default=20.0)
    ap.add_argument('--nbins', type=int, default=24)
    ap.add_argument('--ad_correction', action='store_true')

    ap.add_argument('--inner_fit_min', type=float, default=3.0)
    ap.add_argument('--inner_fit_max', type=float, default=8.0)
    ap.add_argument('--boundary_method', choices=['bic_changepoint','consecutive_excess','both'], default='both')

    ap.add_argument('--saveplot', default=os.path.join('maxdepth_gaia','outputs','mw_rotation_curve_maxdepth.png'))
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    out_dir = os.path.dirname(args.saveplot) if args.saveplot else os.path.join('maxdepth_gaia','outputs')
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(out_dir, debug=args.debug)

    xp = get_xp(prefer_gpu=True)
    logger.info(f"Backend: {xp_name(xp)}")

    # Ingest data from local sources
    try:
        src = args.use_source
        if src == 'auto':
            src = detect_source(args.slices_glob, args.mw_csv_path)
        if src == 'slices':
            stars_df, meta = load_slices(args.slices_glob, zmax=args.zmax, sigma_vmax=args.sigma_vmax, vRmax=args.vRmax,
                                         phi_bins=args.phi_bins, phi_bin_index=args.phi_bin_index, logger=logger)
            # write used files for provenance
            write_json(os.path.join(out_dir,'used_files.json'), dict(files=meta['files']))
            # sample for plotting
            star_sample_df = stars_df.sample(n=min(len(stars_df), 200000), random_state=42)
        elif src == 'mw_csv':
            stars_df, meta = load_mw_csv(args.mw_csv_path, zmax=args.zmax, sigma_vmax=args.sigma_vmax, vRmax=args.vRmax, logger=logger)
            write_json(os.path.join(out_dir,'used_files.json'), dict(files=meta['files']))
            star_sample_df = stars_df.sample(n=min(len(stars_df), 200000), random_state=42)
        else:
            raise ValueError(f"Unsupported use_source: {src}")
    except Exception as e:
        logger.exception(f"Data ingestion failed: {e}")
        raise SystemExit(1)

    logger.info(f"Stars after filters: {len(stars_df):,}")

    # Bin rotation curve
    bins_df = bin_rotation_curve(stars_df, rmin=args.rmin, rmax=args.rmax, nbins=args.nbins, ad_correction=args.ad_correction, logger=logger)
    bins_path = os.path.join(out_dir, 'rotation_curve_bins.csv')
    bins_df.to_csv(bins_path, index=False)
    logger.info(f"Saved binned curve: {bins_path} (rows={len(bins_df)})")

    # Fit baryons in the inner region
    inner = fit_baryons_inner(bins_df, Rmin=args.inner_fit_min, Rmax=args.inner_fit_max, logger=logger)
    R_bins = bins_df['R_kpc_mid'].to_numpy()
    vbar_all = v_c_baryon(R_bins, inner.params)

    # Detect boundary
    chosen = None
    boundary_obj = None
    if args.boundary_method in ('both','consecutive_excess'):
        out1 = find_boundary_consecutive(bins_df, vbar_all, K=3, S_thresh=2.0, logger=logger)
        if out1.get('found'):
            chosen = out1
            boundary_obj = out1
            logger.info(f"Boundary (consecutive): R_b = {out1['R_boundary']:.2f} kpc")
    if args.boundary_method in ('both','bic_changepoint'):
        out2 = find_boundary_bic(bins_df, vbar_all, logger=logger)
        if out2.get('found') and (boundary_obj is None or out2['delta_bic_vs_baryons'] > 6.0):
            chosen = out2
            boundary_obj = out2
            logger.info(f"Boundary (BIC): R_b = {out2['R_boundary']:.2f} kpc  (ΔBIC={out2['delta_bic_vs_baryons']:.1f})")

    if boundary_obj is None:
        logger.warning("Boundary not detected robustly; using inner_fit_max as a provisional boundary.")
        boundary_obj = dict(found=True, R_boundary=float(args.inner_fit_max), method='provisional')
    else:
        boundary_obj['method'] = boundary_obj.get('method', 'bic' if 'delta_bic_vs_baryons' in boundary_obj else 'consecutive')

    # Bootstrap boundary uncertainty
    boot = bootstrap_boundary(bins_df, vbar_all, method='bic' if boundary_obj['method']=='bic' else 'consecutive', nboot=200, seed=42, logger=logger)
    if boot.get('success'):
        boundary_obj['R_unc_lo'] = boot['lo']
        boundary_obj['R_unc_hi'] = boot['hi']

    R_boundary = float(boundary_obj['R_boundary'])

    # Compute M_enclosed at boundary for anchor
    Vb = np.interp(R_boundary, R_bins, vbar_all)
    M_enclosed = float((Vb**2) * R_boundary / G_KPC)

    # Outer fits
    sat = fit_saturated_well(bins_df, vbar_all, R_boundary, logger=logger)
    nfw = fit_nfw(bins_df, vbar_all, logger=logger)

    # Build dense curves for plotting
    Rf = np.linspace(bins_df['R_lo'].min(), bins_df['R_hi'].max(), 300)
    vbar_curve = v_c_baryon(Rf, inner.params)
    # Saturated-well curve across all R (no tail inside boundary)
    xi = sat.params.get('xi', np.nan)
    R_s = sat.params.get('R_s', np.nan)
    m = sat.params.get('m', np.nan)
    vflat = sat.params.get('v_flat', np.nan)
    v2_extra = v2_saturated_extra(Rf, vflat, R_s, m) if np.isfinite(vflat) else np.zeros_like(Rf)
    # Apply tail only beyond the detected boundary (convert only excess beyond GR)
    if np.isfinite(R_boundary):
        v2_extra[Rf < R_boundary] = 0.0
    v_satwell = np.sqrt(np.clip(vbar_curve**2 + v2_extra, 0.0, None))

    v_nfw = np.sqrt(np.clip(vbar_curve**2 + v_c_nfw(Rf, nfw.params.get('V200', 200.0), nfw.params.get('c', 10.0))**2, 0.0, None))

    curves_df = pd.DataFrame(dict(R_kpc=Rf, v_baryon=vbar_curve, v_baryon_satwell=v_satwell, v_baryon_nfw=v_nfw))
    curves_path = os.path.join(out_dir, 'model_curves.csv')
    curves_df.to_csv(curves_path, index=False)

    # Metrics for baryons-only on bins_df
    stats_bary = compute_metrics(bins_df['vphi_kms'].to_numpy(), v_c_baryon(bins_df['R_kpc_mid'].to_numpy(), inner.params), np.maximum(bins_df['vphi_err_kms'].to_numpy(), 2.0), k_params=5)

    # Compose fit_params JSON
    fit_params = dict(
        data_source=dict(mode=meta.get('mode'), files=meta.get('files', [])),
        backend=dict(name=xp_name(xp)),
        baryon_params=inner.params,
        inner_fit_stats=inner.stats,
        boundary=boundary_obj,
        M_enclosed=M_enclosed,
        saturated_well=dict(
            params=sat.params,
            chi2=sat.stats.get('chi2'), aic=sat.stats.get('aic'), bic=sat.stats.get('bic'),
            v_flat=sat.params.get('v_flat'),
            lensing_alpha_arcsec=lensing_alpha_arcsec(sat.params.get('v_flat')) if np.isfinite(sat.params.get('v_flat', np.nan)) else np.nan,
        ),
        nfw=dict(
            params=nfw.params,
            chi2=nfw.stats.get('chi2'), aic=nfw.stats.get('aic'), bic=nfw.stats.get('bic'),
        ),
        baryons_only=dict(
            chi2=stats_bary.get('chi2'), aic=stats_bary.get('aic'), bic=stats_bary.get('bic')
        ),
        bins=dict(n_bins=int(len(bins_df)), rmin=float(args.rmin), rmax=float(args.rmax)),
        ad_correction=bool(args.ad_correction),
    )

    fit_path = os.path.join(out_dir, 'fit_params.json')
    write_json(fit_path, fit_params)
    logger.info(f"Saved fit params: {fit_path}")

    # Plot
    make_plot(bins_df, star_sample_df, curves_df, fit_params, args.saveplot, logger=logger)


if __name__ == '__main__':
    main()
