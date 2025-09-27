# run_pipeline.py
# Local Gaia MW pipeline: ingest local data, bin, fit baryons (inner), detect boundary,
# fit anchored saturated-well and NFW, and plot.

from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import setup_logging, write_json, xp_name, get_xp, G_KPC
from .data_io import detect_source, load_slices, load_mw_csv, load_sparc_catalog
from .rotation import bin_rotation_curve
from .models import v_c_baryon, v_c_baryon_multi, MW_DEFAULT, v2_saturated_extra, v_c_nfw, v_flat_from_anchor, lensing_alpha_arcsec, gate_c1, v_c_mond_from_vbar
from .boundary import fit_baryons_inner, find_boundary_bic, find_boundary_consecutive, bootstrap_boundary, fit_saturated_well, fit_nfw, compute_metrics
from .plotting import make_plot


def main():
    ap = argparse.ArgumentParser(description='Gaia MW max-depth test (local data only).')
    ap.add_argument('--use_source', choices=['auto','slices','mw_csv','sparc'], default='auto')
    ap.add_argument('--slices_glob', default=os.path.join('data','gaia_sky_slices','processed_*.parquet'))
    ap.add_argument('--mw_csv_path', default=os.path.join('data','gaia_mw_real.csv'))
    ap.add_argument('--sparc_dir', default=os.path.join('data','Rotmod_LTG'))
    ap.add_argument('--sparc_names', default=None, help='Comma-separated list of SPARC galaxy names to run (defaults to first 10).')
    ap.add_argument('--zmax', type=float, default=0.5)
    ap.add_argument('--sigma_vmax', type=float, default=30.0)
    ap.add_argument('--vRmax', type=float, default=40.0)
    ap.add_argument('--phi_bins', type=int, default=1)
    ap.add_argument('--phi_bin_index', type=int, default=None)

    ap.add_argument('--rmin', type=float, default=3.0)
    ap.add_argument('--rmax', type=float, default=20.0)
    ap.add_argument('--nbins', type=int, default=24)
    ap.add_argument('--ad_correction', action='store_true')
    ap.add_argument('--ad_poly_deg', type=int, default=2)
    ap.add_argument('--ad_frac_err', type=float, default=0.3)
    ap.add_argument('--baryon_model', choices=['single','mw_multi'], default='mw_multi', help='Select GR baseline: single MN disk + bulge or MW-like multi-disk + gas.')

    ap.add_argument('--inner_fit_min', type=float, default=3.0)
    ap.add_argument('--inner_fit_max', type=float, default=8.0)
    ap.add_argument('--boundary_method', choices=['bic_changepoint','consecutive_excess','both'], default='both')
    ap.add_argument('--baryon_priors', choices=['mw','wide'], default='mw')

    ap.add_argument('--saveplot', default=os.path.join('maxdepth_gaia','outputs','mw_rotation_curve_maxdepth.png'))
    ap.add_argument('--gate_width_kpc', type=float, default=None, help='Optional: fix smooth gate width (kpc) for the tail (C1, exact zero inside).')
    ap.add_argument('--fix_m', type=float, default=None, help='Optional: fix m (tail sharpness) globally for cross-galaxy tests.')
    ap.add_argument('--eta_rs', type=float, default=None, help='Optional: fix R_s as eta_rs * R_boundary (global tail shape).')
    ap.add_argument('--mond_kind', choices=['simple','standard'], default='simple')
    ap.add_argument('--mond_a0', type=float, default=1.2e-10)
    ap.add_argument('--anchor_kappa', type=float, default=1.0, help='Global scale factor on v_flat^2 anchor (>=1.0 increases tail amplitude).')
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
        if src == 'sparc':
            # SPARC mode: iterate galaxies and process per-galaxy outputs
            names = [s.strip() for s in args.sparc_names.split(',')] if args.sparc_names else None
            gal_list = load_sparc_catalog(args.sparc_dir, master_sheet=None, names=names, logger=logger)
            agg_rows = []
            for gal in gal_list:
                gname = gal['name']
                gdf = gal['df'].copy().sort_values('R_kpc')
                # Build a bins_df-compatible table
                R = gdf['R_kpc'].to_numpy(); V = gdf['vphi_kms'].to_numpy(); S = gdf['vphi_err_kms'].to_numpy(); vbar = gdf['vbar_kms'].to_numpy()
                # Create edges from midpoints
                edges = np.zeros(len(R)+1)
                edges[1:-1] = 0.5*(R[:-1] + R[1:])
                dr0 = R[1]-R[0] if len(R)>1 else 0.1
                drN = R[-1]-R[-2] if len(R)>1 else 0.1
                edges[0] = max(0.0, R[0] - dr0/2.0)
                edges[-1] = R[-1] + drN/2.0
                bins_df = pd.DataFrame(dict(R_lo=edges[:-1], R_hi=edges[1:], R_kpc_mid=R, vphi_kms=V, vphi_err_kms=S, N=np.ones_like(R)))

                # No inner renormalization in SPARC mode; use vbar as-is
                R_bins = R
                vbar_all = vbar

                # Boundary detection
                out1 = find_boundary_consecutive(bins_df, vbar_all, K=3, S_thresh=2.0, logger=logger)
                out2 = find_boundary_bic(bins_df, vbar_all, gate_width_fixed=args.gate_width_kpc, fixed_m=args.fix_m, logger=logger)
                boundary_obj = out2 if out2.get('found') else (out1 if out1.get('found') else dict(found=False))
                if not boundary_obj.get('found'):
                    logger.warning(f"[{gname}] Boundary not detected; skipping galaxy")
                    continue
                boundary_obj['method'] = boundary_obj.get('method','bic' if 'delta_bic_vs_baryons' in boundary_obj else 'consecutive')
                R_boundary = float(boundary_obj['R_boundary'])

                # Fit saturated well and NFW
                sat = fit_saturated_well(bins_df, vbar_all, R_boundary, gate_width_fixed=args.gate_width_kpc, fixed_m=args.fix_m, eta_rs=args.eta_rs, anchor_kappa=args.anchor_kappa, logger=logger)
                nfw = fit_nfw(bins_df, vbar_all, logger=logger)

                # Dense curves for plotting (use observed R grid)
                Rf = np.linspace(edges[0], edges[-1], 300)
                # Interpolate vbar onto Rf
                vbar_curve = np.interp(Rf, R, vbar)
                # SatWell on Rf
                xi = sat.params.get('xi', np.nan)
                R_s = sat.params.get('R_s', np.nan)
                m = sat.params.get('m', np.nan)
                vflat = sat.params.get('v_flat', np.nan)
                if np.isfinite(vflat) and np.isfinite(R_s) and np.isfinite(m):
                    gw = sat.params.get('gate_width_kpc', args.gate_width_kpc if args.gate_width_kpc is not None else 0.8)
                    v2_extra = v2_saturated_extra(Rf, vflat, R_s, m) * gate_c1(Rf, R_boundary, gw)
                else:
                    v2_extra = np.zeros_like(Rf)
                v_satwell = np.sqrt(np.clip(vbar_curve**2 + v2_extra, 0.0, None))
                v_nfw = np.sqrt(np.clip(vbar_curve**2 + v_c_nfw(Rf, nfw.params.get('V200', 200.0), nfw.params.get('c', 10.0))**2, 0.0, None))
                # MOND on SPARC
                v_mond = v_c_mond_from_vbar(Rf, vbar_curve, a0_m_s2=args.mond_a0, kind=args.mond_kind)
                curves_df = pd.DataFrame(dict(R_kpc=Rf, v_baryon=vbar_curve, v_baryon_satwell=v_satwell, v_baryon_nfw=v_nfw, v_baryon_mond=v_mond))

                # Metrics on SPARC bins
                stats_bary = compute_metrics(V, vbar, np.maximum(S, 2.0), k_params=5)
                v_mond_bins = v_c_mond_from_vbar(R, vbar, a0_m_s2=args.mond_a0, kind=args.mond_kind)
                stats_mond = compute_metrics(V, v_mond_bins, np.maximum(S, 2.0), k_params=5)

                # Write outputs per galaxy
                gal_dir = os.path.join(out_dir, 'sparc', gname)
                os.makedirs(gal_dir, exist_ok=True)
                bins_path = os.path.join(gal_dir, 'rotation_curve_bins.csv')
                bins_df.to_csv(bins_path, index=False)
                curves_df.to_csv(os.path.join(gal_dir, 'model_curves.csv'), index=False)
                fit_params = dict(
                    data_source=dict(mode='sparc', name=gname, file=os.path.basename(name_to_file) if 'name_to_file' in locals() else ''),
                    baryon_model='sparc_baryons',
                    boundary=boundary_obj,
                    saturated_well=dict(params=sat.params, chi2=sat.stats.get('chi2'), aic=sat.stats.get('aic'), bic=sat.stats.get('bic'), v_flat=sat.params.get('v_flat')),
                    nfw=dict(params=nfw.params, chi2=nfw.stats.get('chi2'), aic=nfw.stats.get('aic'), bic=nfw.stats.get('bic')),
                    mond=dict(kind=args.mond_kind, a0_m_s2=float(args.mond_a0), chi2=stats_mond.get('chi2'), aic=stats_mond.get('aic'), bic=stats_mond.get('bic')),
                    baryons_only=dict(chi2=stats_bary.get('chi2'), aic=stats_bary.get('aic'), bic=stats_bary.get('bic')),
                )
                write_json(os.path.join(gal_dir, 'fit_params.json'), fit_params)
                make_plot(bins_df, None, curves_df, fit_params, os.path.join(gal_dir, f'{gname}_rotation.png'), logger=logger)
                # Aggregate row
                agg_rows.append(dict(name=gname, Rb=R_boundary, xi=sat.params.get('xi'), sw_bic=sat.stats.get('bic'), nfw_bic=nfw.stats.get('bic'), mond_bic=stats_mond.get('bic'), bary_bic=stats_bary.get('bic')))

            # write aggregate
            if agg_rows:
                pd.DataFrame(agg_rows).to_csv(os.path.join(out_dir, 'sparc_summary.csv'), index=False)
            logger.info(f"SPARC processing complete ({len(agg_rows)} galaxies)")
            return
        elif src == 'slices':
            stars_df, meta = load_slices(args.slices_glob, zmax=args.zmax, sigma_vmax=args.sigma_vmax, vRmax=args.vRmax,
                                         phi_bins=args.phi_bins, phi_bin_index=args.phi_bin_index, logger=logger)
            write_json(os.path.join(out_dir,'used_files.json'), dict(files=meta['files']))
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
    bins_df = bin_rotation_curve(stars_df, rmin=args.rmin, rmax=args.rmax, nbins=args.nbins,
                                 ad_correction=args.ad_correction, ad_poly_deg=args.ad_poly_deg, ad_frac_err=args.ad_frac_err,
                                 logger=logger)
    bins_path = os.path.join(out_dir, 'rotation_curve_bins.csv')
    bins_df.to_csv(bins_path, index=False)
    logger.info(f"Saved binned curve: {bins_path} (rows={len(bins_df)})")

    # Fit baryons in the inner region (for diagnostics; baseline may be overridden below)
    inner = fit_baryons_inner(bins_df, Rmin=args.inner_fit_min, Rmax=args.inner_fit_max, priors=args.baryon_priors, logger=logger)

    # Select GR baseline and renormalize inner errors to reduced chi^2 ~ 1 relative to that baseline
    R_bins = bins_df['R_kpc_mid'].to_numpy()
    if args.baryon_model == 'mw_multi':
        vbar_all = v_c_baryon_multi(R_bins, MW_DEFAULT)
        # Compute inner stats vs MW baseline
        m_inner = (R_bins >= args.inner_fit_min) & (R_bins <= args.inner_fit_max)
        stats_inner_vs_mw = compute_metrics(bins_df['vphi_kms'].to_numpy()[m_inner], vbar_all[m_inner], np.maximum(bins_df['vphi_err_kms'].to_numpy()[m_inner], 2.0), k_params=5)
        if stats_inner_vs_mw and stats_inner_vs_mw.get('dof', 0) > 0 and stats_inner_vs_mw.get('chi2', None) is not None:
            f = float(np.sqrt(max(stats_inner_vs_mw['chi2']/max(stats_inner_vs_mw['dof'],1.0), 1.0)))
            bins_df['vphi_err_kms'] = bins_df['vphi_err_kms'] * f
            logger.info(f"Rescaled bin errors by factor f={f:.3f} vs MW-like baseline to target inner reduced chi2 ~ 1")
    else:
        # single-component baseline from inner fit
        if inner.stats and inner.stats.get('dof', 0) > 0 and inner.stats.get('chi2', None) is not None:
            f = float(np.sqrt(max(inner.stats['chi2']/max(inner.stats['dof'],1.0), 1.0)))
            bins_df['vphi_err_kms'] = bins_df['vphi_err_kms'] * f
            logger.info(f"Rescaled bin errors by factor f={f:.3f} to target inner reduced chi2 ~ 1")
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
        out2 = find_boundary_bic(bins_df, vbar_all, gate_width_fixed=args.gate_width_kpc, fixed_m=args.fix_m, logger=logger)
        if out2.get('found') and (boundary_obj is None or out2['delta_bic_vs_baryons'] > 6.0):
            chosen = out2
            boundary_obj = out2
            logger.info(f"Boundary (BIC): R_b = {out2['R_boundary']:.2f} kpc  (dBIC={out2['delta_bic_vs_baryons']:.1f})")

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
    sat = fit_saturated_well(bins_df, vbar_all, R_boundary, gate_width_fixed=args.gate_width_kpc, fixed_m=args.fix_m, eta_rs=args.eta_rs, anchor_kappa=args.anchor_kappa, logger=logger)
    nfw = fit_nfw(bins_df, vbar_all, logger=logger)

    # Build dense curves for plotting
    Rf = np.linspace(bins_df['R_lo'].min(), bins_df['R_hi'].max(), 300)
    if args.baryon_model == 'mw_multi':
        vbar_curve = v_c_baryon_multi(Rf, MW_DEFAULT)
    else:
        vbar_curve = v_c_baryon(Rf, inner.params)
    # Saturated-well curve across all R (no tail inside boundary)
    xi = sat.params.get('xi', np.nan)
    R_s = sat.params.get('R_s', np.nan)
    m = sat.params.get('m', np.nan)
    vflat = sat.params.get('v_flat', np.nan)
    # Dense curve uses the same smooth gate as the fit
    if np.isfinite(vflat):
        gw = sat.params.get('gate_width_kpc', args.gate_width_kpc if args.gate_width_kpc is not None else 0.8)
        v2_extra = v2_saturated_extra(Rf, vflat, R_s, m) * gate_c1(Rf, R_boundary, gw)
    else:
        v2_extra = np.zeros_like(Rf)
    v_satwell = np.sqrt(np.clip(vbar_curve**2 + v2_extra, 0.0, None))

    v_nfw = np.sqrt(np.clip(vbar_curve**2 + v_c_nfw(Rf, nfw.params.get('V200', 200.0), nfw.params.get('c', 10.0))**2, 0.0, None))
    # MOND curve based on the same GR baseline
    v_mond = v_c_mond_from_vbar(Rf, vbar_curve, a0_m_s2=args.mond_a0, kind=args.mond_kind)

    curves_df = pd.DataFrame(dict(R_kpc=Rf, v_baryon=vbar_curve, v_baryon_satwell=v_satwell, v_baryon_nfw=v_nfw, v_baryon_mond=v_mond))
    curves_path = os.path.join(out_dir, 'model_curves.csv')
    curves_df.to_csv(curves_path, index=False)

    # Metrics for baryons-only on bins_df
    if args.baryon_model == 'mw_multi':
        vb_bins = v_c_baryon_multi(bins_df['R_kpc_mid'].to_numpy(), MW_DEFAULT)
        stats_bary = compute_metrics(bins_df['vphi_kms'].to_numpy(), vb_bins, np.maximum(bins_df['vphi_err_kms'].to_numpy(), 2.0), k_params=5)
    else:
        vb_bins = v_c_baryon(bins_df['R_kpc_mid'].to_numpy(), inner.params)
        stats_bary = compute_metrics(bins_df['vphi_kms'].to_numpy(), vb_bins, np.maximum(bins_df['vphi_err_kms'].to_numpy(), 2.0), k_params=5)

    # MOND metrics on bins
    v_mond_bins = v_c_mond_from_vbar(bins_df['R_kpc_mid'].to_numpy(), vb_bins, a0_m_s2=args.mond_a0, kind=args.mond_kind)
    stats_mond = compute_metrics(bins_df['vphi_kms'].to_numpy(), v_mond_bins, np.maximum(bins_df['vphi_err_kms'].to_numpy(), 2.0), k_params=5)

    # Compose fit_params JSON
    fit_params = dict(
        data_source=dict(mode=meta.get('mode'), files=meta.get('files', [])),
        backend=dict(name=xp_name(xp)),
        baryon_model=args.baryon_model,
        baryon_params=(MW_DEFAULT if args.baryon_model == 'mw_multi' else inner.params),
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
        mond=dict(
            kind=args.mond_kind, a0_m_s2=float(args.mond_a0),
            chi2=stats_mond.get('chi2'), aic=stats_mond.get('aic'), bic=stats_mond.get('bic')
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

    # Budget audit: used vs allowed v_flat^2 across R (saved separately)
    try:
        out_dir = os.path.dirname(args.saveplot)
        gw = sat.params.get('gate_width_kpc', args.gate_width_kpc if args.gate_width_kpc is not None else 0.8)
        used_frac = gate_c1(Rf, R_boundary, gw) * (1.0 - np.exp(-np.power(np.maximum(Rf, 1e-9)/np.maximum(R_s, 1e-9), m)))
        budget_csv = os.path.join(out_dir, 'budget_audit.csv')
        pd.DataFrame(dict(R_kpc=Rf, used_fraction=np.clip(used_frac, 0.0, 1.0))).to_csv(budget_csv, index=False)
        plt.figure(figsize=(8,4))
        plt.plot(Rf, used_frac, color='teal', lw=2, label='Used fraction (v_extra^2 / v_flat^2)')
        plt.axhline(1.0, color='gray', ls='--', lw=1)
        plt.axvline(R_boundary, color='gray', ls='--', lw=1)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('R [kpc]')
        plt.ylabel('Used / Allowed')
        plt.title('Budget audit: tail budget usage vs radius')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right', frameon=True)
        budget_png = os.path.join(out_dir, 'budget_audit.png')
        plt.tight_layout(); plt.savefig(budget_png, dpi=220, bbox_inches='tight')
        if logger:
            logger.info(f"Saved budget audit plot to {budget_png} and CSV to {budget_csv}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to generate budget audit: {e}")


if __name__ == '__main__':
    main()
