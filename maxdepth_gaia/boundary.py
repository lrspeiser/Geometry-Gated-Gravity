# boundary.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from scipy.optimize import least_squares, curve_fit

from .utils import G_KPC
from .models import v_c_baryon, v2_saturated_extra, v_c_nfw, v_flat_from_anchor, gate_c1


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
                      priors: str = 'mw',
                      logger=None) -> FitResult:
    d = bins_df[(bins_df['R_kpc_mid'] >= Rmin) & (bins_df['R_kpc_mid'] <= Rmax)].copy()
    if len(d) < 6:
        # fall back to typical MW-like priors
        params = dict(M_d=6e10, a_d=3.0, b_d=0.3, M_b=8e9, a_b=0.6)
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
    if priors == 'mw':
        lb = np.array([3e10, 2.0, 0.1, 5e9, 0.3], dtype=float)
        ub = np.array([8e10, 4.0, 0.5, 1.5e10, 1.2], dtype=float)
        x0 = np.array([6e10, 3.0, 0.3, 8e9, 0.6], dtype=float)
    else:
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


def find_boundary_bic(bins_df: pd.DataFrame, vbar_all: np.ndarray, gate_width_fixed: float | None = None, fixed_m: float | None = None, logger=None) -> Dict[str, Any]:
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
        # Anchored saturated-well fit beyond Rb
        mask_out = R >= Rb
        if np.count_nonzero(mask_out) < 4:
            continue
        Rout = R[mask_out]; Vout = V[mask_out]; Sout = S[mask_out]

        if gate_width_fixed is None and fixed_m is None:
            def model_out(Rx, xi, R_s, m, dR):
                Vb = np.interp(Rb, R, vbar_all)
                M_encl = (Vb**2) * Rb / G_KPC
                vflat = v_flat_from_anchor(M_encl, Rb, xi)
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, m) * gate_c1(Rx, Rb, dR)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            p0=[0.8, 10.0, 2.0, 0.8]; lb=[0.1, 1.0, 0.5, 0.1]; ub=[1.0, 50.0, 8.0, 2.0]
            k_params = 9
        elif gate_width_fixed is not None and fixed_m is None:
            def model_out(Rx, xi, R_s, m):
                Vb = np.interp(Rb, R, vbar_all)
                M_encl = (Vb**2) * Rb / G_KPC
                vflat = v_flat_from_anchor(M_encl, Rb, xi)
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, m) * gate_c1(Rx, Rb, gate_width_fixed)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            p0=[0.8, 10.0, 2.0]; lb=[0.1, 1.0, 0.5]; ub=[1.0, 50.0, 8.0]
            k_params = 8
        elif gate_width_fixed is None and fixed_m is not None:
            def model_out(Rx, xi, R_s, dR):
                Vb = np.interp(Rb, R, vbar_all)
                M_encl = (Vb**2) * Rb / G_KPC
                vflat = v_flat_from_anchor(M_encl, Rb, xi)
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, fixed_m) * gate_c1(Rx, Rb, dR)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            p0=[0.8, 10.0, 0.8]; lb=[0.1, 1.0, 0.1]; ub=[1.0, 50.0, 2.0]
            k_params = 8
        else:
            def model_out(Rx, xi, R_s):
                Vb = np.interp(Rb, R, vbar_all)
                M_encl = (Vb**2) * Rb / G_KPC
                vflat = v_flat_from_anchor(M_encl, Rb, xi)
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, fixed_m) * gate_c1(Rx, Rb, gate_width_fixed)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            p0=[0.8, 10.0]; lb=[0.1, 1.0]; ub=[1.0, 50.0]
            k_params = 7

        try:
            popt, pcov = curve_fit(model_out, Rout, Vout, sigma=Sout, absolute_sigma=True,
                                   p0=p0, bounds=(lb, ub), maxfev=50000)
            Vmod = np.interp(R, R, vbar_all)
            Vmod_out = model_out(Rout, *popt)
            Vfull = np.array(Vmod)
            Vfull[mask_out] = Vmod_out
            stats = compute_metrics(V, Vfull, S, k_params=k_params)
            if stats['bic'] < best_bic:
                best_bic = stats['bic']
                params_out = dict(xi=float(popt[0]), R_s=float(popt[1]))
                if k_params >= 8:
                    if fixed_m is None:
                        params_out['m'] = float(popt[2])
                else:
                    params_out['m'] = float(fixed_m) if fixed_m is not None else np.nan
                best = dict(R_boundary=float(Rb), params=params_out, stats=stats)
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

def fit_saturated_well(bins_df: pd.DataFrame, vbar_all: np.ndarray, R_boundary: float, gate_width_fixed: float | None = None, fixed_m: float | None = None, eta_rs: float | None = None, anchor_kappa: float = 1.0, logger=None) -> FitResult:
    R = bins_df['R_kpc_mid'].to_numpy()
    V = bins_df['vphi_kms'].to_numpy()
    S = np.maximum(bins_df['vphi_err_kms'].to_numpy(), 2.0)

    mask_out = R >= R_boundary
    Rout = R[mask_out]; Vout = V[mask_out]; Sout = S[mask_out]
    if np.count_nonzero(mask_out) < 4:
        return FitResult(params=dict(xi=np.nan, R_s=np.nan, m=np.nan, v_flat=np.nan), cov=None, stats=dict(chi2=np.nan, aic=np.nan, bic=np.nan, dof=np.nan))

    Vb = np.interp(R_boundary, R, vbar_all)
    M_encl = (Vb**2) * R_boundary / G_KPC

    R_s_fixed = None
    if eta_rs is not None and np.isfinite(eta_rs):
        R_s_fixed = max(eta_rs * R_boundary, 1e-6)

    # Build model variants depending on which parameters are fixed
    if R_s_fixed is None:
        # R_s is free
        if gate_width_fixed is None and fixed_m is None:
            def model_out(Rx, xi, R_s, m, dR):
                vflat = v_flat_from_anchor(M_encl, R_boundary, xi) * np.sqrt(max(anchor_kappa, 1e-12))
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, m) * gate_c1(Rx, R_boundary, dR)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            p0 = [0.8, 10.0, 2.0, 0.8]
            lb = [0.1, 1.0, 0.5, 0.1]; ub = [1.0, 50.0, 8.0, 2.0]
            popt, pcov = curve_fit(model_out, Rout, Vout, sigma=Sout, absolute_sigma=True,
                                   p0=p0, bounds=(lb, ub), maxfev=50000)
            xi, R_s, m, dR = popt
            k_params = 9
        elif gate_width_fixed is not None and fixed_m is None:
            def model_out(Rx, xi, R_s, m):
                vflat = v_flat_from_anchor(M_encl, R_boundary, xi) * np.sqrt(max(anchor_kappa, 1e-12))
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, m) * gate_c1(Rx, R_boundary, gate_width_fixed)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            popt, pcov = curve_fit(model_out, Rout, Vout, sigma=Sout, absolute_sigma=True,
                                   p0=[0.8, 10.0, 2.0], bounds=([0.1, 1.0, 0.5], [1.0, 50.0, 8.0]), maxfev=40000)
            xi, R_s, m = popt; dR = gate_width_fixed
            k_params = 8
        elif gate_width_fixed is None and fixed_m is not None:
            def model_out(Rx, xi, R_s, dR):
                vflat = v_flat_from_anchor(M_encl, R_boundary, xi) * np.sqrt(max(anchor_kappa, 1e-12))
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, fixed_m) * gate_c1(Rx, R_boundary, dR)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            popt, pcov = curve_fit(model_out, Rout, Vout, sigma=Sout, absolute_sigma=True,
                                   p0=[0.8, 10.0, 0.8], bounds=([0.1, 1.0, 0.1], [1.0, 50.0, 2.0]), maxfev=40000)
            xi, R_s, dR = popt; m = fixed_m
            k_params = 8
        else:  # both gate_width and m fixed
            def model_out(Rx, xi, R_s):
                vflat = v_flat_from_anchor(M_encl, R_boundary, xi)
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, fixed_m) * gate_c1(Rx, R_boundary, gate_width_fixed)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            popt, pcov = curve_fit(model_out, Rout, Vout, sigma=Sout, absolute_sigma=True,
                                   p0=[0.8, 10.0], bounds=([0.1, 1.0], [1.0, 50.0]), maxfev=30000)
            xi, R_s = popt; m = fixed_m; dR = gate_width_fixed
            k_params = 7
    else:
        # R_s is fixed to eta_rs * R_boundary
        R_s = R_s_fixed
        if gate_width_fixed is None and fixed_m is None:
            def model_out(Rx, xi, m, dR):
                vflat = v_flat_from_anchor(M_encl, R_boundary, xi)
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, m) * gate_c1(Rx, R_boundary, dR)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            popt, pcov = curve_fit(model_out, Rout, Vout, sigma=Sout, absolute_sigma=True,
                                   p0=[0.8, 2.0, 0.8], bounds=([0.1, 0.5, 0.1], [1.0, 8.0, 2.0]), maxfev=40000)
            xi, m, dR = popt
            k_params = 8  # one fewer since R_s fixed
        elif gate_width_fixed is not None and fixed_m is None:
            def model_out(Rx, xi, m):
                vflat = v_flat_from_anchor(M_encl, R_boundary, xi)
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, m) * gate_c1(Rx, R_boundary, gate_width_fixed)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            popt, pcov = curve_fit(model_out, Rout, Vout, sigma=Sout, absolute_sigma=True,
                                   p0=[0.8, 2.0], bounds=([0.1, 0.5], [1.0, 8.0]), maxfev=30000)
            xi, m = popt; dR = gate_width_fixed
            k_params = 7
        elif gate_width_fixed is None and fixed_m is not None:
            def model_out(Rx, xi, dR):
                vflat = v_flat_from_anchor(M_encl, R_boundary, xi)
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, fixed_m) * gate_c1(Rx, R_boundary, dR)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            popt, pcov = curve_fit(model_out, Rout, Vout, sigma=Sout, absolute_sigma=True,
                                   p0=[0.8, 0.8], bounds=([0.1, 0.1], [1.0, 2.0]), maxfev=30000)
            xi, dR = popt; m = fixed_m
            k_params = 7
        else:  # R_s fixed, m fixed, gate fixed
            def model_out(Rx, xi):
                vflat = v_flat_from_anchor(M_encl, R_boundary, xi)
                v2_extra = v2_saturated_extra(Rx, vflat, R_s, fixed_m) * gate_c1(Rx, R_boundary, gate_width_fixed)
                return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + v2_extra, 0.0, None))
            popt, pcov = curve_fit(model_out, Rout, Vout, sigma=Sout, absolute_sigma=True,
                                   p0=[0.8], bounds=([0.1], [1.0]), maxfev=20000)
            xi = popt[0]; dR = gate_width_fixed; m = fixed_m
            k_params = 6

    vflat = v_flat_from_anchor(M_encl, R_boundary, xi) * np.sqrt(max(anchor_kappa, 1e-12))

    # Compose full curve
    Vmodel = np.interp(R, R, vbar_all)
    if R_s_fixed is None:
        # use the same function variant as above
        if gate_width_fixed is None and fixed_m is None:
            Vmodel[mask_out] = model_out(Rout, xi, R_s, m, dR)
        elif gate_width_fixed is not None and fixed_m is None:
            Vmodel[mask_out] = model_out(Rout, xi, R_s, m)
        elif gate_width_fixed is None and fixed_m is not None:
            Vmodel[mask_out] = model_out(Rout, xi, R_s, dR)
        else:
            Vmodel[mask_out] = model_out(Rout, xi, R_s)
    else:
        if gate_width_fixed is None and fixed_m is None:
            Vmodel[mask_out] = model_out(Rout, xi, m, dR)
        elif gate_width_fixed is not None and fixed_m is None:
            Vmodel[mask_out] = model_out(Rout, xi, m)
        elif gate_width_fixed is None and fixed_m is not None:
            Vmodel[mask_out] = model_out(Rout, xi, dR)
        else:
            Vmodel[mask_out] = model_out(Rout, xi)

    stats = compute_metrics(V, Vmodel, S, k_params=k_params)
    return FitResult(params=dict(xi=float(xi), R_s=float(R_s if R_s_fixed is not None else R_s), m=float(m), gate_width_kpc=float(dR), v_flat=float(vflat)), cov=pcov, stats=stats)


def fit_nfw(bins_df: pd.DataFrame, vbar_all: np.ndarray, logger=None) -> FitResult:
    R = bins_df['R_kpc_mid'].to_numpy()
    V = bins_df['vphi_kms'].to_numpy()
    S = np.maximum(bins_df['vphi_err_kms'].to_numpy(), 2.0)

    def model_all(Rx, V200, c):
        return np.sqrt(np.clip(np.power(np.interp(Rx, R, vbar_all), 2) + np.power(v_c_nfw(Rx, V200, c), 2), 0.0, None))

    # MW-like priors to avoid extreme/unphysical corners
    p0 = [150.0, 12.0]
    bounds = ([120.0, 8.0], [180.0, 20.0])
    popt, pcov = curve_fit(model_all, R, V, sigma=S, absolute_sigma=True, p0=p0, bounds=bounds, maxfev=30000)

    Vmodel = model_all(R, *popt)
    stats = compute_metrics(V, Vmodel, S, k_params=7)  # 5 baryon + 2 halo (effective)
    return FitResult(params=dict(V200=float(popt[0]), c=float(popt[1])), cov=pcov, stats=stats)


def fit_baryon_plus_nfw_joint(bins_df: pd.DataFrame, priors: str = 'mw', logger=None) -> FitResult:
    """Jointly fit single-disk baryons (MN+Hernquist) and NFW on all bins.
    This is an optional helper for future experiments; not wired to CLI by default.
    """
    R = bins_df['R_kpc_mid'].to_numpy()
    V = bins_df['vphi_kms'].to_numpy()
    S = np.maximum(bins_df['vphi_err_kms'].to_numpy(), 2.0)

    # Parameter vector: [Md, ad, bd, Mb, ab, V200, c]
    if priors == 'mw':
        lb = np.array([3e10, 2.0, 0.1, 5e9, 0.3, 120.0, 8.0], dtype=float)
        ub = np.array([8e10, 4.0, 0.5, 1.5e10, 1.2, 180.0, 20.0], dtype=float)
        x0 = np.array([6e10, 3.0, 0.3, 8e9, 0.6, 150.0, 12.0], dtype=float)
    else:
        lb = np.array([2e10, 1.5, 0.05, 1e9, 0.1, 100.0, 5.0], dtype=float)
        ub = np.array([1.2e11, 6.0, 1.0, 2e10, 1.5, 220.0, 25.0], dtype=float)
        x0 = np.array([5e10, 3.5, 0.3, 8e9, 0.6, 150.0, 12.0], dtype=float)

    def resid(theta: np.ndarray) -> np.ndarray:
        Md, ad, bd, Mb, ab, V200, c = theta
        params = dict(M_d=float(Md), a_d=float(ad), b_d=float(bd), M_b=float(Mb), a_b=float(ab))
        Vbar = v_c_baryon(R, params)
        Vhalo = v_c_nfw(R, float(V200), float(c))
        Vmod = np.sqrt(np.clip(Vbar**2 + Vhalo**2, 0.0, None))
        return (Vmod - V) / S

    res = least_squares(resid, x0=x0, bounds=(lb, ub), max_nfev=30000)

    # Extract and compute stats
    Md, ad, bd, Mb, ab, V200, c = res.x
    params = dict(M_d=float(Md), a_d=float(ad), b_d=float(bd), M_b=float(Mb), a_b=float(ab), V200=float(V200), c=float(c))
    Vbar = v_c_baryon(R, dict(M_d=params['M_d'], a_d=params['a_d'], b_d=params['b_d'], M_b=params['M_b'], a_b=params['a_b']))
    Vhalo = v_c_nfw(R, params['V200'], params['c'])
    Vmod = np.sqrt(np.clip(Vbar**2 + Vhalo**2, 0.0, None))
    stats = compute_metrics(V, Vmod, S, k_params=7)

    try:
        _, s, VT = np.linalg.svd(res.jac, full_matrices=False)
        cov = VT.T @ np.diag(1.0/np.maximum(s*s, 1e-12)) @ VT
    except Exception:
        cov = None

    return FitResult(params=params, cov=cov, stats=stats)
