from __future__ import annotations
import math
import numpy as _np

try:
    import cupy as _cp  # type: ignore
    xp = _cp
    HAS_CUPY = True
except Exception:
    xp = _np  # fallback
    HAS_CUPY = False


def to_xp(a):
    return xp.asarray(a)


def xi_shell_logistic_radius(R_kpc, Mbar, *, xi_max=3.0, lnR0_base=math.log(3.0), width=0.6, alpha_M=-0.2, Mref=1e10):
    R = xp.asarray(R_kpc)
    lnR = xp.log(xp.maximum(R, 1e-6))
    xi_max = xp.maximum(xp.asarray(xi_max), 1.0)
    width = xp.maximum(xp.asarray(width), 1e-3)
    alpha_M = xp.asarray(alpha_M)
    Mbar = xp.asarray(Mbar if (Mbar is not None) else Mref)
    lnR0_g = lnR0_base + alpha_M * xp.log(xp.maximum(Mbar / Mref, 1e-12))
    sigm = 1.0 / (1.0 + xp.exp(-(lnR - lnR0_g) / width))
    xi = 1.0 + (xi_max - 1.0) * sigm
    return xi


def xi_logistic_density(R_kpc, Sigma_bar, Mbar, *, xi_max=3.0, lnSigma_c=math.log(10.0), width_sigma=0.6, n_sigma=1.0):
    if Sigma_bar is None:
        return xp.ones_like(xp.asarray(R_kpc))
    S = xp.asarray(Sigma_bar)
    xi_max = xp.maximum(xp.asarray(xi_max), 1.0)
    width_sigma = xp.maximum(xp.asarray(width_sigma), 1e-3)
    n = xp.maximum(xp.asarray(n_sigma), 1e-3)
    lnS = xp.log(xp.maximum(S, 1e-8))
    z = (lnSigma_c - lnS) / width_sigma
    sigm = 1.0 / (1.0 + xp.exp(-z))
    xi = 1.0 + (xi_max - 1.0) * xp.power(sigm, n)
    xi = xp.where(xp.isfinite(S), xi, 1.0)
    return xi


def gate_fixed(xi, R_kpc, *, gate_R_kpc=3.0, gate_width=0.4):
    R = xp.asarray(R_kpc)
    lnR = xp.log(xp.maximum(R, 1e-6))
    lnR_gate = math.log(max(gate_R_kpc, 1e-6))
    w = max(gate_width, 1e-3)
    H = 1.0 / (1.0 + xp.exp(-(lnR - lnR_gate) / w))
    return 1.0 + H * (xi - 1.0)


def gate_learned(xi, R_kpc, Mbar, *, lnR_gate_base=math.log(3.0), width_gate=0.4, alpha_gate_M=-0.2, Mref=1e10):
    R = xp.asarray(R_kpc)
    lnR = xp.log(xp.maximum(R, 1e-6))
    width_gate = max(width_gate, 1e-3)
    mass_term = 0.0
    if (Mbar is not None) and _np.isfinite(Mbar):
        mass_term = math.log(max((float(Mbar) / Mref), 1e-12))
    lnR_gate = lnR_gate_base + alpha_gate_M * mass_term
    H = 1.0 / (1.0 + xp.exp(-(lnR - lnR_gate) / width_gate))
    return 1.0 + H * (xi - 1.0)


def vpred_from_xi(Vbar_kms, xi):
    Vbar = xp.asarray(Vbar_kms)
    xi = xp.asarray(xi)
    return Vbar * xp.sqrt(xp.clip(xi, 1.0, 100.0))