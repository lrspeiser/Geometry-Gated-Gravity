# rotation.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

from .utils import robust_stats


def bin_rotation_curve(stars_df: pd.DataFrame,
                       rmin: float = 3.0,
                       rmax: float = 20.0,
                       nbins: int = 24,
                       use_weights: bool = True,
                       ad_correction: bool = False,
                       logger=None) -> pd.DataFrame:
    """Bin star-level data into a rotation curve with robust uncertainties.

    stars_df columns expected:
      R_kpc, vphi_kms, optionally vphi_err_kms, vR_kms
    """
    cols = set(stars_df.columns)
    if not {'R_kpc','vphi_kms'}.issubset(cols):
        raise ValueError("stars_df must contain R_kpc and vphi_kms")

    df = stars_df.copy()
    mask = (np.isfinite(df['R_kpc'])) & (np.isfinite(df['vphi_kms']))
    df = df.loc[mask]

    bins = np.linspace(float(rmin), float(rmax), int(nbins)+1)
    centers = 0.5*(bins[:-1] + bins[1:])

    out_rows = []
    # prefetch presence of columns
    has_err = 'vphi_err_kms' in cols
    has_vR  = 'vR_kms' in cols

    # compute per-bin stats
    for i in range(nbins):
        lo, hi = bins[i], bins[i+1]
        sub = df[(df['R_kpc'] >= lo) & (df['R_kpc'] < hi)]
        N = len(sub)
        if N == 0:
            continue
        # weights from errors if available
        if use_weights and has_err and sub['vphi_err_kms'].notna().all():
            w = 1.0/np.maximum(sub['vphi_err_kms'].to_numpy()**2, 1e-6)
            v_med = float(np.average(sub['vphi_kms'].to_numpy(), weights=w))
            v_err = float(np.sqrt(1.0/np.maximum(np.sum(w), 1e-12)))
        else:
            s = robust_stats(sub['vphi_kms'].to_numpy())
            v_med = s['median']
            v_err = s['stderr'] if np.isfinite(s['stderr']) else 5.0

        # dispersions
        sigma_phi = float(np.std(sub['vphi_kms'].to_numpy(), ddof=1)) if N > 1 else np.nan
        sigma_R = float(np.std(sub['vR_kms'].to_numpy(), ddof=1)) if (has_vR and sub['vR_kms'].notna().all() and N>1) else np.nan

        out_rows.append(dict(R_lo=lo, R_hi=hi, R_kpc_mid=centers[i],
                             vphi_kms=v_med, vphi_err_kms=v_err, N=N,
                             sigma_phi=sigma_phi, sigma_R=sigma_R))

    rc = pd.DataFrame(out_rows)
    if rc.empty:
        raise RuntimeError("No bins populated; adjust rmin/rmax/nbins")

    # tracer surface density proxy nu ~ N / area of annulus
    area = np.pi*(np.power(rc['R_hi'],2) - np.power(rc['R_lo'],2))
    rc['nu'] = rc['N'] / np.maximum(area, 1e-6)

    # Optional asymmetric drift correction at bin level
    rc['ad_applied'] = False
    if ad_correction:
        # Precompute gradients in ln space; handle edges with 1-sided differences
        Rm = rc['R_kpc_mid'].to_numpy()
        nu = rc['nu'].to_numpy()
        sigR2 = np.power(rc['sigma_R'].to_numpy(), 2)
        sigphi2 = np.power(rc['sigma_phi'].to_numpy(), 2)

        # Only apply where we have finite dispersions
        valid = np.isfinite(nu) & np.isfinite(sigR2) & np.isfinite(sigphi2) & (sigR2 > 0) & (sigphi2 > 0)
        if np.count_nonzero(valid) >= 5:
            dlnnu = np.full_like(nu, np.nan, dtype=float)
            dlnsigR2 = np.full_like(sigR2, np.nan, dtype=float)
            # central differences
            for i in range(len(Rm)):
                if i == 0:
                    j, k = i, i+1
                elif i == len(Rm)-1:
                    j, k = i-1, i
                else:
                    j, k = i-1, i+1
                if valid[j] and valid[k] and Rm[k] > 0 and Rm[j] > 0 and nu[j] > 0 and nu[k] > 0 and sigR2[j] > 0 and sigR2[k] > 0:
                    dlnnu[i] = (np.log(nu[k]) - np.log(nu[j])) / (np.log(Rm[k]) - np.log(Rm[j]))
                    dlnsigR2[i] = (np.log(sigR2[k]) - np.log(sigR2[j])) / (np.log(Rm[k]) - np.log(Rm[j]))

            beta = 1.0 - sigphi2/np.maximum(sigR2, 1e-6)
            AD = sigR2 * (np.nan_to_num(dlnnu, nan=0.0) + np.nan_to_num(dlnsigR2, nan=0.0) + beta)

            # Apply AD correction where valid
            apply = np.isfinite(AD)
            idx = np.where(apply)[0]
            if idx.size > 0:
                vc2 = np.power(rc.loc[apply,'vphi_kms'].to_numpy(), 2) + AD[apply]
                vc = np.sqrt(np.clip(vc2, 0.0, None))
                rc.loc[apply, 'vphi_kms'] = vc
                rc.loc[apply, 'ad_applied'] = True
                # keep same error bars; could be propagated in a later refinement
        else:
            if logger:
                logger.info("AD correction requested but insufficient dispersions present; skipping.")

    return rc