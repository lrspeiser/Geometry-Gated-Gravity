# boundary.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from scipy.optimize import least_squares, curve_fit

from .utils import G_KPC
from .models import v_c_baryon, v2_saturated_extra, v_c_nfw, v_flat_from_anchor


@dataclass
class FitResult:
    params: Dict[str, float]
    cov: np.ndarray | None
    stats: Dict[str, Any]


def compute_metrics(y: np.ndarray, y_model: np.ndarray, sigma: np.ndarray, k_params: int) -> Dict[str, float]:
    m = np.isfinite(y) & np.isfinite(y_model) & np.isfinite(sigma) & (sigma > 0)
    y = y[m]; y_model = y_model[m]; s = sigma[m]
    n = len(y)
    if n == 0:
        return dict(chi2=np.nan, aic=np.nan, bic=np.nan, dof=np.nan)
    chi2 = float(np.sum(((y - y_model) / s) ** 2))
    aic = chi2 + 2 * k_params
    bic = chi2 + k_params * np.log(max(n, 1))
    dof = max(n - k_params, 1)
    return dict(chi2=chi2, aic=float(aic), bic=float(bic), dof=float(dof))


# -----------------------------
# Inner baryonic fit
# -----------------------------

def fit_baryons_inner(bins_df: pd.DataFrame,
                      Rmin: float = 3.0,
                      Rmax: float = 8.0,
                      logger=None) -> FitResult:
    d = bins_df[(bins_df['R_kpc_mid'] >= Rmin) & (bins_df['R_kpc_mid'] <= Rmax)].copy()
    if len(d) < 6:
        # fall back to default priors
        params = dict(M_d=6e10, a_d=5.0, b_d=0.3, M_b=8e9, a_b=0.6)
        vbar = v_c_baryon(bins_df['R_kpc_mid'].to_numpy(), params)
        stats = compute_metrics(d['vphi_kms'].to_numpy(), v_c_baryon(d['R_kpc_mid'].to_numpy(), params), np.maximum(d['vphi_err_kms'].to_numpy(), 2.0), 5)
        return FitResult(params=params, cov=None, stats=stats)

    R = d['R_kpc_mid'].to_numpy()
    V = d['vphi_kms'].to_numpy()
    S = np.maximum(d['vphi_err_kms'].to_numpy(), 2.0)

    def resid(theta: np.ndarray) -> np.ndarray:
        Md, ad, bd, Mb, ab = theta
        params = dict(M_d=float(Md), a_d=float(ad), b_d=float(bd), M_b=float(Mb), a_b=float(ab))
        Vbar = v_c_baryon(R, params)
        return (Vbar - V)/S

    # Bounds
    lb = np.array([2e10, 2.0, 0.05, 1e9, 0.1], dtype=float)
    ub = np.array([1.5e11, 7.0, 1.0, 2e10, 1.5], dtype=float)
    x0 = np.array([6e10, 5.0, 0.3, 8e9, 0.6], dtype=float)

    res = least_squares(resid, x0=x0, bounds=(lb, ub), max_nfev=20000)
    Md, ad, bd, Mb, ab = res.x
    params = dict(M_d=float(Md), a_d=float(ad), b_d=float(bd), M_b=float(Mb), a_b=float(ab))

    # Approximate covariance from jacobian
    try:
        _, s, VT = np.linalg.svd(res.jac, full_matrices=False)
        cov = VT.T @ np.diag(1.0/np.maximum(s*s, 1e-12)) @ VT
    except Exception:
        cov = None

    stats = compute_metrics(V, v_c_baryon(R, params), S, 5)

    return FitResult(params=params, cov=cov, stats=stats)


# -----------------------------
# Boundary detection
# -----------------------------

def compute_residual_excess(bins_df: pd.DataFrame, vbar_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    V = bins_df['vphi_kms'].to_numpy()
    S = np.maximum(bins_df['vphi_err_kms'].to_numpy(), 2.0)
    dV2 = np.power(V, 2) - np.power(vbar_all, 2)
    # significance per bin (approx): S_i = dV^2 / (2 V sigma)
    denom = 2.0 * np.maximum(V, 1e-6) * np.maximum(S, 1e-6)
    Sig = dV2 / denom
    return dV2, Sig


def find_boundary_consecutive(bins_df: pd.DataFrame, vbar_all: np.ndarray, K: int = 3, S_thresh: float = 2.0,
                              logger=None) -> Dict[str, Any]:
    _, Sig = compute_residual_excess(bins_df, vbar_all)
    R_edges = bins_df['R_lo'].to_numpy()
    hit_idx = None
    run = 0
    for i, s in enumerate(Sig):
        if np.isfinite(s) and s >= S_thresh:
            run += 1
            if run >= K:
                hit_idx = i - K + 1
                break
        else:
            run = 0
    if hit_idx is None:
        return dict(found=False)
    Rb = float(R_edges[hit_idx])
    return dict(found=True, R_boundary=Rb, first_index=int(hit_idx), K=int(K), S_thresh=float(S_thresh))


def find_boundary_bic(bins_df: pd.DataFrame, vbar_all: np.ndarray, logger=None) -> Dict[str, Any]:
    R = bins_df['R_kpc_mid'].to_numpy()
    V = bins_df['vphi_kms'].to_numpy()
    S = np.maximum(bins_df['vphi_err_kms'].to_numpy(), 2.0)

    # baseline metrics with baryons only
    base_stats = compute_metrics(V, vbar_all, S, k_params=5)

    best = None
    best_bic = np.inf
    best_idx = None

    # Candidate boundary at each bin edge beyond median
    start = max(2, len(R)//4)
    for j in range(start, len(R)-2):
        Rb = bins_df['R_lo'].iloc[j]
        # Anchored saturated-well fit beyond Rb with 3 params (xi, R_s, m)
        mask_out = R >= Rb
        if np.count_nonzero(mask_out) < 4:
            continue
        Rout = R[mask_out]; Vout = V[mask_out]; Sout = S[mask_out]

        def model_out(Rx, xi, R_s, m):
            # Derive v_flat via anchor from baryon mass within Rb
            Vb = np.interp(Rb, R, vbar_all)
            M_encl = (Vb**2) * Rb / G_KPC
            vflat = v_flat_from_anchor(M_encl, Rb, xi)
            v2_extra = v2_saturated_extra(Rx, vflat, R_s, m)
            return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))

        try:
            popt, pcov = curve_fit(model_out, Rout, Vout, sigma=Sout, absolute_sigma=True,
                                   p0=[1.0, 5.0, 1.5], bounds=([0.5, 1.0, 0.5], [2.0, 30.0, 4.0]), maxfev=20000)
            Vmod = np.interp(R, R, vbar_all)
            Vmod_out = model_out(Rout, *popt)
            # Compose full model by combining inside (no tail) and outside
            Vfull = np.array(Vmod)
            Vfull[mask_out] = Vmod_out
            stats = compute_metrics(V, Vfull, S, k_params=8)  # 5 baryon + 3 tail
            if stats['bic'] < best_bic:
                best_bic = stats['bic']
                best = dict(R_boundary=float(Rb), params=dict(xi=float(popt[0]), R_s=float(popt[1]), m=float(popt[2])), stats=stats)
                best_idx = j
        except Exception:
            continue

    if best is None:
        return dict(found=False)

    best['found'] = True
    best['index'] = int(best_idx)
    best['delta_bic_vs_baryons'] = float(base_stats['bic'] - best['stats']['bic'])
    return best


def bootstrap_boundary(bins_df: pd.DataFrame, vbar_all: np.ndarray, method: str = 'bic', nboot: int = 200, seed: int = 42,
                       logger=None) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    Rb_vals = []
    for _ in range(int(nboot)):
        # bootstrap over bins
        resample = bins_df.sample(n=len(bins_df), replace=True, random_state=int(rng.integers(0, 1e9)))
        if method == 'bic':
            out = find_boundary_bic(resample.sort_values('R_kpc_mid'), np.interp(resample['R_kpc_mid'].to_numpy(), bins_df['R_kpc_mid'].to_numpy(), vbar_all))
        else:
            out = find_boundary_consecutive(resample.sort_values('R_kpc_mid'), np.interp(resample['R_kpc_mid'].to_numpy(), bins_df['R_kpc_mid'].to_numpy(), vbar_all))
        if out.get('found'):
            Rb_vals.append(out['R_boundary'])
    if len(Rb_vals) == 0:
        return dict(success=False)
    Rb_vals = np.asarray(Rb_vals)
    return dict(success=True, median=float(np.median(Rb_vals)), lo=float(np.percentile(Rb_vals, 16)), hi=float(np.percentile(Rb_vals, 84)))


# -----------------------------
# Outer fits: anchored saturated-well and NFW
# -----------------------------

def fit_saturated_well(bins_df: pd.DataFrame, vbar_all: np.ndarray, R_boundary: float, logger=None) -> FitResult:
    R = bins_df['R_kpc_mid'].to_numpy()
    V = bins_df['vphi_kms'].to_numpy()
    S = np.maximum(bins_df['vphi_err_kms'].to_numpy(), 2.0)

    mask_out = R >= R_boundary
    Rout = R[mask_out]; Vout = V[mask_out]; Sout = S[mask_out]
    if np.count_nonzero(mask_out) < 4:
        return FitResult(params=dict(xi=np.nan, R_s=np.nan, m=np.nan, v_flat=np.nan), cov=None, stats=dict(chi2=np.nan, aic=np.nan, bic=np.nan, dof=np.nan))

    Vb = np.interp(R_boundary, R, vbar_all)
    M_encl = (Vb**2) * R_boundary / G_KPC

    def model_out(Rx, xi, R_s, m):
        vflat = v_flat_from_anchor(M_encl, R_boundary, xi)
        v2_extra = v2_saturated_extra(Rx, vflat, R_s, m)
        return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))

    popt, pcov = curve_fit(model_out, Rout, Vout, sigma=Sout, absolute_sigma=True,
                           p0=[1.0, 5.0, 1.5], bounds=([0.5, 1.0, 0.5], [2.0, 30.0, 4.0]), maxfev=20000)
    xi, R_s, m = popt
    vflat = v_flat_from_anchor(M_encl, R_boundary, xi)

    # Compose full curve
    Vmodel = np.interp(R, R, vbar_all)
    Vmodel[mask_out] = model_out(Rout, *popt)

    stats = compute_metrics(V, Vmodel, S, k_params=8)  # 5 baryon + 3 tail
    return FitResult(params=dict(xi=float(xi), R_s=float(R_s), m=float(m), v_flat=float(vflat)), cov=pcov, stats=stats)


def fit_nfw(bins_df: pd.DataFrame, vbar_all: np.ndarray, logger=None) -> FitResult:
    R = bins_df['R_kpc_mid'].to_numpy()
    V = bins_df['vphi_kms'].to_numpy()
    S = np.maximum(bins_df['vphi_err_kms'].to_numpy(), 2.0)

    def model_all(Rx, V200, c):
        return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + np.power(v_c_nfw(Rx, V200, c), 2), 0.0, None))

    p0 = [200.0, 10.0]
    bounds = ([100.0, 4.0], [300.0, 20.0])
    popt, pcov = curve_fit(model_all, R, V, sigma=S, absolute_sigma=True, p0=p0, bounds=bounds, maxfev=30000)

    Vmodel = model_all(R, *popt)
    stats = compute_metrics(V, Vmodel, S, k_params=7)  # 5 baryon + 2 halo
    return FitResult(params=dict(V200=float(popt[0]), c=float(popt[1])), cov=pcov, stats=stats)