from __future__ import annotations
import numpy as _np

try:
    import cupy as _cp  # type: ignore
except Exception:  # CuPy optional
    _cp = None


def get_xp(use_cupy: bool = False):
    """Return (xp, to_cpu) where xp is numpy or cupy, and to_cpu(x) -> numpy array.
    CuPy is optional; if unavailable or use_cupy=False, fall back to numpy.
    """
    if use_cupy and (_cp is not None):
        xp = _cp
        def to_cpu(x):
            return _cp.asnumpy(x)
        return xp, to_cpu
    else:
        xp = _np
        def to_cpu(x):
            return x
        return xp, to_cpu


def dimensionless_radius(R, r_half=None, Rd=None, use_cupy: bool = False):
    """x = R / r_half (preferred) or R / Rd; fallback to R / max(R) if both unknown.
    R: array-like (kpc)
    r_half, Rd: scalars (kpc) or None
    Returns x with same backend as input decision.
    """
    xp, _ = get_xp(use_cupy)
    R = xp.asarray(R)
    denom = None
    if r_half is not None and r_half > 0:
        denom = r_half
    elif Rd is not None and Rd > 0:
        denom = Rd
    else:
        # normalize by max(R) as a graceful fallback
        denom = xp.maximum(R.max(), 1e-6)
    return R / denom


def sigma_hat(Sigma, Sigma0=None, use_cupy: bool = False):
    """Dimensionless surface density Sigma_hat = Sigma / Sigma0.
    If Sigma0 is None, use the finite median of Sigma as the scale.
    """
    xp, _ = get_xp(use_cupy)
    S = xp.asarray(Sigma)
    if Sigma0 is None:
        # median over finite entries
        if xp is _np:
            finite = _np.isfinite(S)
            Sigma0 = _np.median(S[finite]) if _np.any(finite) else 1.0
        else:
            finite = xp.isfinite(S)
            Sigma0 = xp.asnumpy(xp.median(S[finite])) if xp.any(finite) else 1.0
    Sigma0 = max(float(Sigma0), 1e-8)
    return S / Sigma0


def grad_log_sigma(R, Sigma, use_cupy: bool = False):
    """Compute d ln Sigma / d ln R using centered differences in log-space.
    Returns an array with same shape as R; endpoints use one-sided diffs.
    """
    xp, _ = get_xp(use_cupy)
    R = xp.asarray(R)
    S = xp.asarray(Sigma)
    eps = 1e-12
    R = xp.maximum(R, eps)
    S = xp.maximum(S, eps)
    lnR = xp.log(R)
    lnS = xp.log(S)
    dlnR = xp.zeros_like(lnR)
    dlnS = xp.zeros_like(lnS)
    # centered for interior
    dlnR[1:-1] = (lnR[2:] - lnR[:-2]) / 2.0
    dlnS[1:-1] = (lnS[2:] - lnS[:-2]) / 2.0
    # one-sided at edges
    dlnR[0] = lnR[1] - lnR[0]
    dlnR[-1] = lnR[-1] - lnR[-2]
    dlnS[0] = lnS[1] - lnS[0]
    dlnS[-1] = lnS[-1] - lnS[-2]
    with _np.errstate(divide='ignore', invalid='ignore'):
        slope = dlnS / xp.maximum(dlnR, eps)
    return slope


def kappa_rho(R, rho, use_cupy: bool = False):
    """Curvature of ln rho vs ln r: d^2(ln rho) / d(ln r)^2 (discrete approximation).
    """
    xp, _ = get_xp(use_cupy)
    r = xp.asarray(R)
    rh = xp.asarray(rho)
    eps = 1e-12
    r = xp.maximum(r, eps)
    rh = xp.maximum(rh, eps)
    lnR = xp.log(r)
    lnR_diff = lnR[1:] - lnR[:-1]
    lnR_mid = (lnR[1:] + lnR[:-1]) / 2.0
    lnR_mid = xp.concatenate([lnR_mid[:1], lnR_mid, lnR_mid[-1:]], axis=0)
    ln_rho = xp.log(rh)
    # first derivative at centers
    d1 = xp.zeros_like(r)
    d1[1:-1] = (ln_rho[2:] - ln_rho[:-2]) / (lnR[2:] - lnR[:-2] + eps)
    d1[0] = d1[1]
    d1[-1] = d1[-2]
    # second derivative via centered differences on d1
    d2 = xp.zeros_like(r)
    d2[1:-1] = (d1[2:] - d1[:-2]) / (lnR[2:] - lnR[:-2] + eps)
    d2[0] = d2[1]
    d2[-1] = d2[-2]
    return d2
