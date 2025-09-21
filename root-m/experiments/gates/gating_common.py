# -*- coding: utf-8 -*-
"""
root-m/experiments/gates/gating_common.py

Gating variants for Root-M-like tails that respond to baryon thermodynamics.
This module is self-contained and does not modify existing LogTail or Root-M code.
"""
import numpy as np

EPS = 1e-30


def _moving_avg(y, win):
    win = max(1, int(win))
    y = np.asarray(y, float)
    if win <= 1 or y.size == 0:
        return y
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(y, kernel, mode='same')


def smooth_profile(r_kpc, y, s_kpc=50.0):
    """Robust 1D smoother without SciPy; boxcar with width ~ s_kpc."""
    r = np.asarray(r_kpc, float)
    y = np.asarray(y, float)
    if r.size < 2:
        return y
    dr = max(EPS, float(np.median(np.diff(r))))
    win = int(max(1, round(s_kpc / dr)))
    return _moving_avg(y, win)


def v_tail2_rootm_rhoaware(r_kpc, Mb_encl_Msun, rho_b_Msun_kpc3,
                           A_kms=160.0, Mref_Msun=6e10, rc_kpc=10.0,
                           rho0_Msun_kpc3=1e7, q=1.0, s_kpc=75.0):
    """
    ρ-aware: multiplicative boost that saturates to 1 in high-density regions
    and increases in diffuse ICM.
    Returns v_tail^2 in (km/s)^2.
    """
    r = np.asarray(r_kpc, float)
    Mb = np.clip(np.asarray(Mb_encl_Msun, float), 0.0, None)
    rho = np.clip(np.asarray(rho_b_Msun_kpc3, float), 0.0, None)
    L = min(r.size, Mb.size, rho.size)
    r = r[:L]; Mb = Mb[:L]; rho = rho[:L]
    rho_sm = smooth_profile(r, rho, s_kpc)
    dens_fac = (rho0_Msun_kpc3 / (rho_sm + rho0_Msun_kpc3))**float(q)
    base = (A_kms**2) * np.sqrt(Mb / Mref_Msun) * (r / (r + rc_kpc))
    return base * dens_fac


def v_tail2_rootm_gradaware(r_kpc, Mb_encl_Msun, rho_b_Msun_kpc3,
                            A_kms=160.0, Mref_Msun=6e10, rc_kpc=10.0,
                            qg=0.8, m=1.0, s_kpc=75.0):
    """
    Gradient-aware: boost keyed to |d ln rho/d ln r| of smoothed baryon density.
    Returns v_tail^2 in (km/s)^2.
    """
    r = np.asarray(r_kpc, float)
    Mb = np.clip(np.asarray(Mb_encl_Msun, float), 0.0, None)
    rho = np.clip(np.asarray(rho_b_Msun_kpc3, float), 0.0, None)
    L = min(r.size, Mb.size, rho.size)
    r = r[:L]; Mb = Mb[:L]; rho = rho[:L]
    rho_sm = smooth_profile(r, rho, s_kpc)
    with np.errstate(divide='ignore', invalid='ignore'):
        dlnrho = np.gradient(np.log(rho_sm + EPS), np.log(np.maximum(r, EPS)))
    boost = 1.0 + float(qg) * np.abs(dlnrho)**float(m)
    base = (A_kms**2) * np.sqrt(Mb / Mref_Msun) * (r / (r + rc_kpc))
    return base * boost


def v_tail2_rootm_pressaware(r_kpc, Mb_encl_Msun, ne_cm3, kT_keV,
                             A_kms=160.0, Mref_Msun=6e10, rc_kpc=10.0,
                             eta=0.5, s_kpc=75.0):
    """
    Pressure-aware: boost keyed to |d ln P/d ln r| where P ~ n_e kT (units cancel in log).
    Returns v_tail^2 in (km/s)^2.
    """
    r = np.asarray(r_kpc, float)
    Mb = np.clip(np.asarray(Mb_encl_Msun, float), 0.0, None)
    ne = np.clip(np.asarray(ne_cm3, float), 0.0, None)
    T = np.clip(np.asarray(kT_keV, float), 0.0, None)
    L = min(r.size, Mb.size, ne.size, T.size)
    r = r[:L]; Mb = Mb[:L]; ne = ne[:L]; T = T[:L]
    P = smooth_profile(r, ne, s_kpc) * smooth_profile(r, T, s_kpc)
    with np.errstate(divide='ignore', invalid='ignore'):
        dlnP = np.gradient(np.log(P + EPS), np.log(np.maximum(r, EPS)))
    boost = 1.0 + float(eta) * np.abs(dlnP)
    base = (A_kms**2) * np.sqrt(Mb / Mref_Msun) * (r / (r + rc_kpc))
    return base * boost
