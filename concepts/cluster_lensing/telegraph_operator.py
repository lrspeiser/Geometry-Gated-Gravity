#!/usr/bin/env python3
from __future__ import annotations
"""
Telegraph operator for baryon-driven nonlocal response in lensing.

This implements a nonlocal, geometry-aware mapping that increases central
convergence without inserting an ad hoc NFW halo. It operates on the
projected baryon surface density Σ_bar(R) to produce an effective Σ_eff(R)
that is more centrally concentrated while remaining tied to the baryon map.

Core idea (log-radius transport):
- Work on a log-spaced radial grid (as used elsewhere in the pipeline).
- Smooth Σ_bar in logR by a Gaussian of width sigma_logR, then shift inward
  by mu_logR (< 0), yielding Σ_in.
- Form a difference δΣ = Σ_in - Σ_bar. Clip relative to Σ_bar to maintain
  positivity and guard against pathologies.
- Σ_eff = Σ_bar + lambda_amp * δΣ_clipped.

Notes:
- This is not an NFW fit. It is an operator whose response depends on the
  geometry of Σ_bar itself, with only a few global hyperparameters.
- We optionally renormalize Σ_eff to preserve total projected mass.
- All computations are 1D in R with vectorized NumPy operations.

Parameters
- lambda_amp ∈ [0, 1]: strength of telegraphing (fraction of inward-shifted δΣ added)
- mu_logR < 0: inward log-radius shift magnitude (e.g., -0.5)
- sigma_logR > 0: smoothing width in log-radius (e.g., 0.4)
- clip_up, clip_down ≥ 0: relative clipping caps for δΣ; δΣ ∈ [-clip_down*Σ, +clip_up*Σ]
- renorm_total_mass: if True, rescale Σ_eff by scalar s to match total projected
  mass of Σ_bar (helps mass conservation under 2π R weighting)

Outputs
- Σ_eff, and a diag dict with integrals and guardrail info.
"""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class TelegraphParams:
    lambda_amp: float = 0.35
    mu_logR: float = -0.6
    sigma_logR: float = 0.45
    clip_up: float = 1.0
    clip_down: float = 0.8
    renorm_total_mass: bool = True


def _gaussian_kernel(n: int, sigma_bins: float) -> np.ndarray:
    """Return a normalized 1D Gaussian kernel of length n in index space (circular)."""
    # Build symmetric kernel over indices [-m, ..., 0, ..., +m]
    m = max(1, int(np.ceil(3.0 * sigma_bins)))
    x = np.arange(-m, m + 1, dtype=float)
    k = np.exp(-0.5 * (x / max(sigma_bins, 1e-6)) ** 2)
    k /= np.sum(k)
    return k


def _circular_convolve(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular convolution for 1D arrays using FFT or direct mode depending on size."""
    n = arr.size
    if n == 0:
        return arr
    # Use FFT for large n
    if n >= 256:
        fa = np.fft.rfft(arr)
        # Pad kernel to length n
        k = np.zeros(n)
        k0 = kernel.size // 2
        k[:kernel.size] = np.roll(kernel, -k0)
        fk = np.fft.rfft(k)
        out = np.fft.irfft(fa * fk, n)
        return out
    # Direct mode for small n
    out = np.zeros_like(arr)
    m = kernel.size // 2
    for i in range(n):
        s = 0.0
        for j in range(-m, m + 1):
            s += kernel[j + m] * arr[(i + j) % n]
        out[i] = s
    return out


def apply_telegraph(R_kpc: np.ndarray, Sigma_bar_kpc2: np.ndarray, p: TelegraphParams
                    ) -> Tuple[np.ndarray, Dict]:
    """Apply the telegraph operator to Σ_bar(R) and return Σ_eff and diagnostics.

    Inputs must be length-matched 1D arrays (R strictly increasing, log-spaced preferred).
    """
    R = np.asarray(R_kpc, float)
    S = np.asarray(Sigma_bar_kpc2, float)
    assert R.ndim == 1 and S.ndim == 1 and R.size == S.size and R.size > 4

    # Work in log-radius index space
    logR = np.log(np.maximum(R, 1e-9))
    dlogR = float(np.median(np.diff(logR)))

    # Smooth in index space with Gaussian of width sigma_bins
    sigma_bins = abs(p.sigma_logR / max(dlogR, 1e-9))
    kernel = _gaussian_kernel(len(S), sigma_bins)
    S_smooth = _circular_convolve(S, kernel)

    # Shift inward in logR by mu_logR (negative), using linear interpolation on logR grid
    shift_bins = p.mu_logR / max(dlogR, 1e-9)
    idx = np.arange(len(S), dtype=float)
    idx_shift = idx + shift_bins
    # Circular wrap indices, then linear interpolation between neighbors
    idx0 = np.floor(idx_shift).astype(int) % len(S)
    idx1 = (idx0 + 1) % len(S)
    t = idx_shift - np.floor(idx_shift)
    S_in = (1.0 - t) * S_smooth[idx0] + t * S_smooth[idx1]

    # Compute delta and clip relative to local Σ to maintain positivity
    dS = S_in - S
    up = float(max(p.clip_up, 0.0))
    dn = float(max(p.clip_down, 0.0))
    dS_clipped = np.minimum(np.maximum(dS, -dn * S), up * S)

    # Apply amplitude and build Σ_eff
    lam = float(np.clip(p.lambda_amp, 0.0, 1.0))
    S_eff = S + lam * dS_clipped

    # Optional mass renormalization under 2π R weighting
    # Mproj ∝ ∫ Σ R dR (we compute discrete equivalent)
    def mass_proj(S_):
        return float(np.trapezoid(S_ * R, R))

    M0 = mass_proj(S)
    M1 = mass_proj(S_eff)
    scale = 1.0
    if p.renorm_total_mass and M1 > 0 and M0 > 0:
        scale = M0 / M1
        S_eff = S_eff * scale
        M1 = mass_proj(S_eff)

    # Diagnostics
    diag = dict(
        lambda_amp=lam,
        mu_logR=float(p.mu_logR),
        sigma_logR=float(p.sigma_logR),
        clip_up=up,
        clip_down=dn,
        renorm_total_mass=bool(p.renorm_total_mass),
        mass_proj_before=M0,
        mass_proj_after=M1,
        renorm_scale=scale,
        dS_min=float(np.min(dS)),
        dS_max=float(np.max(dS)),
        dS_clip_min=float(np.min(dS_clipped)),
        dS_clip_max=float(np.max(dS_clipped)),
        S_eff_min=float(np.min(S_eff)),
        S_eff_max=float(np.max(S_eff)),
    )
    return S_eff, diag
