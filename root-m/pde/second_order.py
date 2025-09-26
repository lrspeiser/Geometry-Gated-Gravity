# -*- coding: utf-8 -*-
"""
root-m/pde/second_order.py

Second-Order Gravity (SOG) utilities:
- Environmental gate S_env(Sigma_loc, g1)
- g2 variants: Field-Energy (FE), Rho^2 (local proxy), Running-G (RG)
- Lensing-only tracer-mass boost (mass-gated, environment-gated)

Units and conventions:
- Radii r in kpc
- Accelerations g in (km/s)^2 per kpc
- Densities rho in Msun / kpc^3
- Surface densities Sigma in Msun / pc^2
- Gravitational constant G in kpc (km/s)^2 Msun^-1

All functions are vectorized for 1D radial arrays.
"""
from __future__ import annotations
import numpy as np
from typing import Dict

# G in (kpc km^2 s^-2 Msun^-1)
G = 4.300917270e-6


def S_env(
    Sigma_loc: np.ndarray,
    g1: np.ndarray,
    Sigma_star: float = 100.0,
    g_star: float = 1200.0,
    aSigma: float = 2.0,
    ag: float = 2.0,
) -> np.ndarray:
    """Environment gate in [0,1] that turns SOG off in dense/strong-field regions.

    S_env = [1 + (Sigma/Sigma_*)^aSigma]^-1 * [1 + (g1/g_*)^ag]^-1

    Inputs
    - Sigma_loc: local surface-density proxy [Msun/pc^2]
    - g1: 1st-order (Newtonian) acceleration [km^2 s^-2 / kpc]
    - Sigma_star, g_star: thresholds controlling where the gate turns on
    - aSigma, ag: gate sharpness exponents
    """
    Sigma = np.asarray(Sigma_loc, dtype=float)
    g = np.asarray(g1, dtype=float)
    s1 = 1.0 / (1.0 + (np.maximum(Sigma, 0.0) / max(Sigma_star, 1e-12)) ** float(aSigma))
    s2 = 1.0 / (1.0 + (np.maximum(g, 0.0) / max(g_star, 1e-12)) ** float(ag))
    return s1 * s2


def _mass_from_rho(r_kpc: np.ndarray, rho_Msun_kpc3: np.ndarray) -> np.ndarray:
    """Enclosed mass M(<r) = 4π ∫_0^r rho(r') r'^2 dr'.
    Returns array M(r) with same length as r.
    """
    r = np.asarray(r_kpc, dtype=float)
    rho = np.asarray(rho_Msun_kpc3, dtype=float)
    if r.ndim != 1:
        raise ValueError("r must be 1D")
    if rho.shape != r.shape:
        raise ValueError("rho must have same shape as r")
    integrand = 4.0 * np.pi * rho * (r ** 2)
    # cumulative trapezoid integration
    dr = np.diff(r)
    if np.any(dr <= 0):
        # Ensure strictly increasing radii
        order = np.argsort(r)
        r = r[order]
        integrand = integrand[order]
        dr = np.diff(r)
    I = np.zeros_like(r)
    if r.size > 1:
        I[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dr)
    return I


def g2_field_energy(
    r_kpc: np.ndarray,
    g1_kms2_per_kpc: np.ndarray,
    Sigma_loc_Msun_pc2: np.ndarray,
    params: Dict[str, float],
) -> np.ndarray:
    """SOG-FE: g2 sourced by field energy density ~ g1^2 with environmental gating.

    ∇^2 Φ2 = λ S_env g1^2 / g_*^2  ⇒  g2(r) = λ r^-2 ∫_0^r S_env(r') [g1(r')^2/g_*^2] r'^2 dr'

    Required params: {'lambda', 'g_star', 'Sigma_star', 'aSigma', 'ag'}
    """
    r = np.asarray(r_kpc, dtype=float)
    g1 = np.asarray(g1_kms2_per_kpc, dtype=float)
    Sigma = np.asarray(Sigma_loc_Msun_pc2, dtype=float)
    lam = float(params.get('lambda', 1.0))
    g_star = float(params.get('g_star', 1200.0))
    S = S_env(Sigma, g1, float(params.get('Sigma_star', 100.0)), g_star,
              float(params.get('aSigma', 2.0)), float(params.get('ag', 2.0)))
    integrand = S * (g1 ** 2) / (g_star ** 2) * (r ** 2)
    dr = np.diff(r)
    I = np.zeros_like(r)
    if r.size > 1:
        I[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dr)
    g2 = np.zeros_like(g1)
    with np.errstate(divide='ignore', invalid='ignore'):
        g2[1:] = lam * I[1:] / (r[1:] ** 2)
    g2[0] = g2[1] if g2.size > 1 else 0.0
    return g2


def g2_rho2_local(
    r_kpc: np.ndarray,
    rho_b_Msun_kpc3: np.ndarray,
    g1_kms2_per_kpc: np.ndarray,
    Sigma_loc_Msun_pc2: np.ndarray,
    params: Dict[str, float],
) -> np.ndarray:
    """SOG-ρ²: g2 from local quadratic baryon density with environmental gating.

    ∇^2 Φ2 = 4π G η S_env ρ_b^2  ⇒  interpret as effective density ρ2 = η S ρ_b^2
    Then g2(r) = G M2(<r) / r^2 with M2 = 4π ∫ ρ2 r'^2 dr'

    Required params: {'eta', 'Sigma_star', 'g_star', 'aSigma', 'ag'}
    """
    r = np.asarray(r_kpc, dtype=float)
    rho = np.asarray(rho_b_Msun_kpc3, dtype=float)
    g1 = np.asarray(g1_kms2_per_kpc, dtype=float)
    Sigma = np.asarray(Sigma_loc_Msun_pc2, dtype=float)
    eta = float(params.get('eta', 0.01))
    S = S_env(Sigma, g1, float(params.get('Sigma_star', 100.0)), float(params.get('g_star', 1200.0)),
              float(params.get('aSigma', 2.0)), float(params.get('ag', 2.0)))
    rho2 = eta * S * (np.maximum(rho, 0.0) ** 2)
    M2 = _mass_from_rho(r, rho2)
    g2 = np.zeros_like(r)
    with np.errstate(divide='ignore', invalid='ignore'):
        g2[1:] = G * M2[1:] / (r[1:] ** 2)
    g2[0] = g2[1] if g2.size > 1 else 0.0
    return g2


def g_runningG(
    r_kpc: np.ndarray,
    g1_kms2_per_kpc: np.ndarray,
    Sigma_loc_Msun_pc2: np.ndarray,
    params: Dict[str, float],
) -> np.ndarray:
    """SOG-RG: Running-G style algebraic enhancement that vanishes in dense regions.

    G_eff = G [1 + A S_env (g_*/(g1+g0))^n]  ⇒  g_tot = (G_eff/G) g1  ⇒  g2 = g_tot - g1

    Required params: {'A','n','g0','g_star','Sigma_star','aSigma','ag'}
    """
    g1 = np.asarray(g1_kms2_per_kpc, dtype=float)
    Sigma = np.asarray(Sigma_loc_Msun_pc2, dtype=float)
    A = float(params.get('A', 1.0))
    n = float(params.get('n', 1.0))
    g0 = float(params.get('g0', 5.0))
    g_star = float(params.get('g_star', 1200.0))
    S = S_env(Sigma, g1, float(params.get('Sigma_star', 100.0)), g_star,
              float(params.get('aSigma', 2.0)), float(params.get('ag', 2.0)))
    factor = 1.0 + A * S * (g_star / np.maximum(g1 + g0, 1e-12)) ** n
    return g1 * (factor - 1.0)


def lensing_boost(
    g_dyn_kms2_per_kpc: np.ndarray,
    Sigma_loc_Msun_pc2: np.ndarray,
    g_gate_kms2_per_kpc: np.ndarray | None,
    m_tr_Msun: float,
    params: Dict[str, float],
) -> np.ndarray:
    """Environment- and tracer-mass-gated lensing amplifier.

    g_lens = g_dyn * [1 + xi_gamma S_env(g_gate, Sigma) * (m_ref/(m_tr+m_floor))^chi]

    Inputs
    - g_dyn_kms2_per_kpc: dynamical acceleration (g1+g2), used as base for lensing
    - Sigma_loc_Msun_pc2: local surface-density proxy for gating
    - g_gate_kms2_per_kpc: field to use in S_env for the g-term of the gate (e.g., g1 or g_dyn). If None, uses g_dyn.
    - m_tr_Msun: tracer mass (0 for photons)
    - params: {'xi_gamma','chi','m_ref','m_floor','Sigma_star','g_star','aSigma','ag'}
    """
    g_dyn = np.asarray(g_dyn_kms2_per_kpc, dtype=float)
    Sigma = np.asarray(Sigma_loc_Msun_pc2, dtype=float)
    g_gate = g_dyn if g_gate_kms2_per_kpc is None else np.asarray(g_gate_kms2_per_kpc, dtype=float)
    xi_gamma = float(params.get('xi_gamma', 0.0))
    chi = float(params.get('chi', 1.0))
    m_ref = float(params.get('m_ref', 1.0))
    m_floor = float(params.get('m_floor', 1e-8))
    S = S_env(Sigma, g_gate, float(params.get('Sigma_star', 100.0)), float(params.get('g_star', 1200.0)),
              float(params.get('aSigma', 2.0)), float(params.get('ag', 2.0)))
    mass_factor = (m_ref / (float(m_tr_Msun) + m_floor)) ** chi
    return g_dyn * (1.0 + xi_gamma * S * mass_factor)
