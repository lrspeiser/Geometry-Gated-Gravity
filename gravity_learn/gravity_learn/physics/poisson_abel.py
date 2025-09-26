from __future__ import annotations
import numpy as _np

try:
    import cupy as _cp  # type: ignore
except Exception:
    _cp = None


_DEF_EPS = 1e-12


def get_xp(use_cupy: bool = False):
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


def enclosed_mass(r, rho, use_cupy: bool = False):
    """Cumulative M(r) = 4π ∫_0^r rho(r') r'^2 dr' (discrete trapezoid).
    r: radii (monotonic increasing), shape (N,)
    rho: density at r, shape (N,)
    Returns array M with shape (N,), same backend as chosen.
    """
    xp, _ = get_xp(use_cupy)
    r = xp.asarray(r)
    rho = xp.asarray(rho)
    r = xp.maximum(r, _DEF_EPS)
    # mid-interval contributions: (f[i] + f[i-1]) / 2 * (r[i] - r[i-1])
    f = 4.0 * _np.pi * rho * r * r
    dr = r[1:] - r[:-1]
    f_mid = 0.5 * (f[1:] + f[:-1])
    integral = xp.zeros_like(r)
    integral[1:] = xp.cumsum(f_mid * dr)
    return integral


def g_of_r(r, M, G: float = 1.0, use_cupy: bool = False):
    """g(r) = G M(r) / r^2. Use consistent units; default G=1 for dimensionless work.
    """
    xp, _ = get_xp(use_cupy)
    r = xp.asarray(r)
    M = xp.asarray(M)
    r2 = xp.maximum(r * r, _DEF_EPS)
    return G * M / r2


def sigma_from_rho_abel(R, r, rho, use_cupy: bool = False):
    """Project spherical rho(r) to surface density Σ(R) via Abel integral:
    Σ(R) = 2 ∫_R^∞ rho(r) r / sqrt(r^2 - R^2) dr.
    Midpoint discretization is used to avoid the endpoint singularity at r→R.
    R: shape (NR,), r: shape (N,), rho: shape (N,)
    """
    xp, _ = get_xp(use_cupy)
    R = xp.asarray(R)
    r = xp.asarray(r)
    rho = xp.asarray(rho)
    R = xp.maximum(R, _DEF_EPS)
    r = xp.maximum(r, _DEF_EPS)
    # For each Rj, integrate on intervals [ri, ri+1] with midpoint rule to avoid singularity
    Sigma = xp.zeros_like(R)
    for j in range(R.shape[0]):
        Rj = R[j]
        # select bins whose upper edge is above Rj
        mask = r > Rj
        rr = r[mask]
        if rr.size < 2:
            Sigma[j] = 0.0
            continue
        # build midpoints limited to segments strictly above Rj
        r_lo = rr[:-1]
        r_hi = rr[1:]
        # discard the first segment if its lower edge is too close to Rj to avoid large kernel error
        # keep segments where r_lo > Rj*(1+1e-9)
        keep = r_lo > (Rj * (1.0 + 1e-9))
        if not xp.any(keep):
            Sigma[j] = 0.0
            continue
        r_lo = r_lo[keep]
        r_hi = r_hi[keep]
        rmid = 0.5 * (r_lo + r_hi)
        dr = (r_hi - r_lo)
        kern_mid = rmid / xp.sqrt(xp.maximum(rmid * rmid - Rj * Rj, _DEF_EPS))
        # interpolate rho at midpoints with simple linear interpolation on log grid assumption
        # fallback: average of endpoints if interpolation not available
        # compute index mapping by searching positions; approximate with local average
        rho_lo = rho[mask][:-1][keep]
        rho_hi = rho[mask][1:][keep]
        rho_mid = 0.5 * (rho_lo + rho_hi)
        integrand_mid = rho_mid * kern_mid
        Sigma[j] = 2.0 * xp.sum(integrand_mid * dr)
    return Sigma


def delta_sigma(R, Sigma, use_cupy: bool = False):
    """Compute ΔΣ(R) = Σ̄(<R) - Σ(R) given Σ(R) on a grid R.
    Assumes R is increasing and Sigma values correspond to those R bins.
    """
    xp, _ = get_xp(use_cupy)
    R = xp.asarray(R)
    S = xp.asarray(Sigma)
    R = xp.maximum(R, _DEF_EPS)
    # Σ̄(<R) = 2/R^2 ∫_0^R Σ(R') R' dR'
    # discrete cumulative integral
    # Build mid-bin areas
    mid_val = 0.5 * (S[1:] + S[:-1])
    dR = R[1:] - R[:-1]
    cum = xp.zeros_like(R)
    cum[1:] = xp.cumsum(mid_val * dR)
    Sbar = xp.zeros_like(R)
    Sbar[1:] = 2.0 * cum[1:] / xp.maximum(R[1:] * R[1:], _DEF_EPS)
    Sbar[0] = Sbar[1]
    dS = Sbar - S
    return dS


# NFW helpers (analytic Σ only, ΔΣ obtainable via delta_sigma numerical path)

def nfw_rho(r, rho_s, r_s, use_cupy: bool = False):
    xp, _ = get_xp(use_cupy)
    r = xp.asarray(r)
    x = xp.maximum(r / r_s, _DEF_EPS)
    return rho_s / (x * (1.0 + x) * (1.0 + x))


def _nfw_F(x):
    # piecewise helper appearing in Σ(R) analytic expression
    out = _np.zeros_like(x)
    xm = x < 1.0
    xp = x > 1.0
    xe = _np.isclose(x, 1.0, rtol=1e-12, atol=1e-12)
    out[xm] = _np.arctanh(_np.sqrt((1.0 - x[xm]) / (1.0 + x[xm]))) / _np.sqrt(1.0 - x[xm] * x[xm])
    out[xp] = _np.arctan(_np.sqrt((x[xp] - 1.0) / (x[xp] + 1.0))) / _np.sqrt(x[xp] * x[xp] - 1.0)
    out[xe] = 1.0
    return out


def nfw_sigma_analytic(R, rho_s, r_s):
    """Projected Σ(R) for NFW; see Bartelmann (1996). Returns numpy array.
    Σ(R) = 2 r_s ρ_s / (x^2 - 1) * [1 - 2 F(x)] with x = R / r_s.
    """
    x = _np.asarray(R, dtype=float) / float(r_s)
    x2m1 = x * x - 1.0
    F = _nfw_F(x)
    # Handle x ~ 1 separately to avoid 0/0
    out = _np.empty_like(x)
    close = _np.isclose(x, 1.0, rtol=1e-8, atol=1e-12)
    out[close] = (2.0 * rho_s * r_s) / 3.0
    notc = ~close
    out[notc] = (2.0 * rho_s * r_s / x2m1[notc]) * (1.0 - 2.0 * F[notc])
    return out
