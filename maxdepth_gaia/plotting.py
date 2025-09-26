# plotting.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .models import v_c_baryon_multi, MW_DEFAULT  # reference GR overlay


def make_plot(bins_df: pd.DataFrame,
              star_sample_df: pd.DataFrame | None,
              curves_df: pd.DataFrame,
              fit_json: dict,
              savepath: str,
              logger=None):
    plt.figure(figsize=(9,6))

    # Star-level scatter (light)
    if star_sample_df is not None and len(star_sample_df) > 0:
        plt.scatter(star_sample_df['R_kpc'], np.abs(star_sample_df['vphi_kms']), s=1, c='0.8', alpha=0.25, label='Stars (sample)')

    # Binned data with errors
    plt.errorbar(bins_df['R_kpc_mid'], bins_df['vphi_kms'], yerr=bins_df['vphi_err_kms'], fmt='o', ms=4,
                 color='black', ecolor='black', elinewidth=1, capsize=2, label='Gaia (binned)')

    # Curves
    if 'v_baryon' in curves_df:
        plt.plot(curves_df['R_kpc'], curves_df['v_baryon'], color='steelblue', lw=2, label='Baryons only')
    if 'v_baryon_nfw' in curves_df:
        plt.plot(curves_df['R_kpc'], curves_df['v_baryon_nfw'], color='darkorange', lw=2, label='Baryons + NFW')
    if 'v_baryon_mond' in curves_df:
        plt.plot(curves_df['R_kpc'], curves_df['v_baryon_mond'], color='purple', lw=2, label='Baryons + MOND')
    if 'v_baryon_satwell' in curves_df:
        plt.plot(curves_df['R_kpc'], curves_df['v_baryon_satwell'], color='forestgreen', lw=2, label='Baryons + Saturated-well')

    # Reference MW GR overlay for visual sanity check
    try:
        Rf = curves_df['R_kpc'].to_numpy()
        v_ref = v_c_baryon_multi(Rf, MW_DEFAULT)
        plt.plot(Rf, v_ref, ls='--', lw=1.2, alpha=0.6, color='gray', label='MWPotential2014-like (GR)')
    except Exception:
        pass

    # Boundary shading and line
    boundary = fit_json.get('boundary', {})
    if boundary and 'R_boundary' in boundary and np.isfinite(boundary['R_boundary']):
        Rb = boundary['R_boundary']
        plt.axvline(Rb, color='gray', ls='--', lw=1.5)
        plt.axvspan(bins_df['R_lo'].min(), Rb, color='gray', alpha=0.1, label='Inner (baryon fit)')
        lo = boundary.get('R_unc_lo', None)
        hi = boundary.get('R_unc_hi', None)
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            plt.fill_betweenx([0, 1e6], lo, hi, color='gray', alpha=0.06, transform=plt.gca().get_xaxis_transform())

    # Text box with stats
    sw = fit_json.get('saturated_well', {})
    nfw = fit_json.get('nfw', {})
    mond = fit_json.get('mond', {})
    bo = fit_json.get('baryons_only', {})
    txt = []
    if sw:
        txt.append(f"SatWell: v_flat={sw.get('v_flat', np.nan):.1f} km/s, R_s={sw.get('R_s', np.nan):.2f} kpc, m={sw.get('m', np.nan):.2f}")
        if 'gate_width_kpc' in sw.get('params', {}):
            txt.append(f"gate ΔR={sw['params'].get('gate_width_kpc', np.nan):.2f} kpc")
        if 'lensing_alpha_arcsec' in sw:
            txt.append(f"alpha≈{sw['lensing_alpha_arcsec']:.3f}\" (log-tail)")
        txt.append(f"AIC/BIC/chi2: {sw.get('aic', np.nan):.1f} / {sw.get('bic', np.nan):.1f} / {sw.get('chi2', np.nan):.1f}")
    if nfw:
        txt.append(f"NFW: V200={nfw.get('params',{}).get('V200', np.nan):.0f}, c={nfw.get('params',{}).get('c', np.nan):.1f}")
        txt.append(f"AIC/BIC/chi2: {nfw.get('aic', np.nan):.1f} / {nfw.get('bic', np.nan):.1f} / {nfw.get('chi2', np.nan):.1f}")
    if mond:
        txt.append(f"MOND(simple): a0=1.2e-10 m/s^2")
        txt.append(f"AIC/BIC/chi2: {mond.get('aic', np.nan):.1f} / {mond.get('bic', np.nan):.1f} / {mond.get('chi2', np.nan):.1f}")
    if bo:
        txt.append(f"Baryons-only: AIC/BIC/chi2: {bo.get('aic', np.nan):.1f} / {bo.get('bic', np.nan):.1f} / {bo.get('chi2', np.nan):.1f}")

    if txt:
        plt.gca().text(0.02, 0.98, "\n".join(txt), transform=plt.gca().transAxes,
                       va='top', ha='left', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, lw=0.5))

    plt.xlabel('R [kpc]')
    plt.ylabel('v_c [km/s]')
    plt.title('Milky Way Rotation Curve (Gaia)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', frameon=True)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath, dpi=250, bbox_inches='tight')
    if logger:
        logger.info(f"Saved plot to {savepath}")

    # Second chart: gravity ratio vs GR (model/GR)
    try:
        out_dir = os.path.dirname(savepath)
        # Only compute where curves are defined and R>0
        R = curves_df['R_kpc'].to_numpy()
        vb = curves_df['v_baryon'].to_numpy()
        vs = curves_df['v_baryon_satwell'].to_numpy() if 'v_baryon_satwell' in curves_df else None
        if vs is not None:
            m = (np.isfinite(R) & np.isfinite(vb) & np.isfinite(vs) & (R > 0) & (vb > 1e-6))
            ratio = np.full_like(R, np.nan, dtype=float)
            ratio[m] = np.power(vs[m]/vb[m], 2)
            plt.figure(figsize=(9,4))
            plt.plot(R, ratio, color='purple', lw=2)
            if boundary and 'R_boundary' in boundary and np.isfinite(boundary['R_boundary']):
                plt.axvline(boundary['R_boundary'], color='gray', ls='--', lw=1)
            plt.xlabel('R [kpc]')
            plt.ylabel('g_model / g_GR')
            plt.title('Predicted gravity strength vs GR')
            plt.grid(True, alpha=0.3)
            ratio_path = os.path.join(out_dir, 'g_ratio_vs_GR.png')
            plt.tight_layout()
            plt.savefig(ratio_path, dpi=250, bbox_inches='tight')
            if logger:
                logger.info(f"Saved gravity ratio plot to {ratio_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to generate gravity ratio plot: {e}")
