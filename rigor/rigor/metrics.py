
from __future__ import annotations
import numpy as np

def weighted_rmse(y, yhat, sigma):
    w = 1.0/np.maximum(sigma, 1e-3)**2
    num = np.sum(w*(y-yhat)**2)
    den = np.sum(w)
    return np.sqrt(num/np.maximum(den, 1e-12))

def percent_off(y, yhat):
    return 100.0*np.abs(y-yhat)/np.maximum(np.abs(y), 1e-6)

def closeness(y, yhat):
    return 100.0*(1.0 - np.abs(y-yhat)/np.maximum(np.abs(y), 1e-6))

def summarize_outer(y, yhat, sigma, mask):
    if mask is None:
        mask = slice(None)
    rmse = weighted_rmse(y[mask], yhat[mask], sigma[mask])
    off  = percent_off(y[mask], yhat[mask])
    return {
        "rmse": float(rmse),
        "mean_off": float(np.mean(off)),
        "median_off": float(np.median(off)),
        "pct_close_90": float(np.mean((100.0 - off) >= 90.0)*100.0),
    }
