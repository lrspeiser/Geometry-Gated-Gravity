try:
    import cupy as cp
    XP = cp
    HAS_CUPY = True
except Exception:  # pragma: no cover
    import numpy as cp  # type: ignore
    XP = cp
    HAS_CUPY = False


def xp():
    """Return the active array module (cupy if available else numpy)."""
    return XP


def to_xp(a):
    xp = XP
    return xp.asarray(a)


def xp_trapz(y, x):
    xp = XP
    # Cupy/numpy both support trapz with the same API
    return xp.trapz(y, x)


def xp_cumtrapz(y, x):
    xp = XP
    # Manual cumulative trapezoid: cumsum of 0.5*(y[i]+y[i-1])*(x[i]-x[i-1])
    dy = y[1:] + y[:-1]
    dx = x[1:] - x[:-1]
    integ = 0.5 * dy * dx
    csum = xp.concatenate([xp.array([0], dtype=y.dtype), xp.cumsum(integ)])
    return csum
