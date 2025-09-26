# utils.py
from __future__ import annotations
import logging
import os
import json
import numpy as np

# Optional CuPy backend
try:
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False

G_KPC = 4.300917270e-6  # (kpc km^2 s^-2) / Msun
C_KMS = 299792.458      # km/s
KPC_M = 3.0856775814913673e19  # meters in a kiloparsec
A0_M_S2 = 1.2e-10              # MOND characteristic acceleration (m/s^2)


def get_xp(prefer_gpu: bool = True):
    """Return array module (CuPy if available and GPU visible, else NumPy)."""
    if prefer_gpu and _CUPY_AVAILABLE:
        try:
            ndev = cp.cuda.runtime.getDeviceCount()  # type: ignore[attr-defined]
            if ndev > 0:
                return cp
        except Exception:
            pass
    return np


def xp_name(xp) -> str:
    return 'cupy' if (xp is not None and getattr(xp, '__name__', '') == 'cupy') else 'numpy'


def to_cpu(arr):
    """Convert xp array to NumPy if necessary."""
    if _CUPY_AVAILABLE and 'cupy' in str(type(arr)):
        return cp.asnumpy(arr)  # type: ignore
    return arr


def robust_stats(x: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Robust central tendency and error.

    Returns: dict(median, mad, stderr)
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(median=np.nan, mad=np.nan, stderr=np.nan)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    n = max(len(x), 1)
    if w is not None:
        w = np.asarray(w)
        w = w[np.isfinite(w)]
        if w.size == x.size and np.any(w > 0):
            stderr = np.sqrt(1.0 / np.maximum(np.sum(w), 1e-12))
        else:
            stderr = mad / np.sqrt(n)
    else:
        stderr = mad / np.sqrt(n)
    return dict(median=float(med), mad=float(mad), stderr=float(stderr))


def setup_logging(out_dir: str, debug: bool = False) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger('maxdepth_gaia')
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    fh = logging.FileHandler(os.path.join(out_dir, 'pipeline.log'))
    fh.setLevel(logging.DEBUG if debug else logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.debug('Logger initialized')
    return logger


def write_json(path: str, obj: dict):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)