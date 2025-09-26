# models.py
# Active model implementations for the max-depth Gaia pipeline
# Units: kpc, km/s, Msun

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

from .utils import G_KPC, C_KMS, KPC_M, A0_M_S2

# -----------------------------
# Baryonic components
# -----------------------------

@dataclass
class HernquistBulge:
    M_b: float = 8e9  # Msun
    a_b: float = 0.6  # kpc
    def v2(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        return G_KPC * self.M_b * R / np.maximum((R + self.a_b) ** 2, 1e-12)


@dataclass
class MiyamotoNagaiDisk:
    M_d: float = 6e10  # Msun
    a: float = 5.0     # kpc (radial scale)
    b: float = 0.3     # kpc (vertical scale)
    def v2(self, R: np.ndarray) -> np.ndarray:
        # Midplane circular speed of MN: v_c^2 = G M R^2 / (R^2 + (a+b)^2)^{3/2}
        R = np.asarray(R, dtype=float)
        A = self.a + self.b
        denom = np.power(R*R + A*A, 1.5)
        return G_KPC * self.M_d * (R*R) / np.maximum(denom, 1e-12)


@dataclass
class BaryonicMW:
    bulge: HernquistBulge
    disk: MiyamotoNagaiDisk
    def v2(self, R: np.ndarray) -> np.ndarray:
        return self.bulge.v2(R) + self.disk.v2(R)


@dataclass
class MultiDiskMW:
    bulge: HernquistBulge
    disks: List[MiyamotoNagaiDisk]
    def v2(self, R: np.ndarray) -> np.ndarray:
        v2 = self.bulge.v2(R)
        for d in self.disks:
            v2 += d.v2(R)
        return v2


# Widely used MW-like defaults (order-of-magnitude consistent with MWPotential2014 / McMillan)
MW_DEFAULT: Dict[str, float] = dict(
    M_b=5e9, a_b=0.6,
    M_thin=4.5e10, a_thin=3.0, b_thin=0.3,
    M_thick=1.0e10, a_thick=2.5, b_thick=0.9,
    M_HI=1.1e10, a_HI=7.0, b_HI=0.1,
    M_H2=1.2e9,  a_H2=1.5, b_H2=0.05,
)


# -----------------------------
# NFW halo benchmark
# -----------------------------
@dataclass
class NFW:
    V200: float = 200.0  # km/s
    c: float = 10.0      # concentration
    R200_kpc: float = 220.0  # derived or set; approximate MW scale
    def v2(self, R: np.ndarray) -> np.ndarray:
        # Express NFW via V200, c. R_s = R200/c; v^2(R) = V200^2 * [ln(1+x)-x/(1+x)]/[x(ln(1+c)-c/(1+c))]
        R = np.asarray(R, dtype=float)
        c = max(self.c, 1e-3)
        R_s = self.R200_kpc / c
        x = np.maximum(R/np.maximum(R_s, 1e-6), 1e-12)
        g_c = np.log(1.0 + c) - c/(1.0 + c)
        num = np.log(1.0 + x) - x/(1.0 + x)
        v2 = (self.V200**2) * (num / np.maximum(x, 1e-12)) / np.maximum(g_c, 1e-12)
        return np.clip(v2, 0.0, None)


# -----------------------------
# Saturated-well (anchored)
# -----------------------------

def v2_saturated_extra(R: np.ndarray, v_flat: float, R_s: float, m: float) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    R_s = max(R_s, 1e-6)
    return (v_flat**2) * (1.0 - np.exp(-np.power(R/R_s, m)))


def gate_c1(R: np.ndarray, Rb: float, dR: float) -> np.ndarray:
    """C1 smooth gate that is exactly 0 for R <= Rb and exactly 1 for R >= Rb + dR.
    Uses Hermite smoothstep: g(s)=3s^2-2s^3 on s in [0,1].
    """
    R = np.asarray(R, dtype=float)
    dR = max(dR, 1e-6)
    s = (R - Rb) / dR
    s = np.clip(s, 0.0, 1.0)
    return 3.0 * s**2 - 2.0 * s**3


def lensing_alpha_arcsec(v_flat: float) -> float:
    # alpha ~ 2π (v_flat/c)^2 radians
    a_rad = 2.0 * np.pi * (v_flat**2) / (C_KMS**2)
    return a_rad * (180.0/np.pi) * 3600.0


# -----------------------------
# Wrappers used by the fitter
# -----------------------------

def v_c_baryon(R: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    bulge = HernquistBulge(M_b=params['M_b'], a_b=params['a_b'])
    disk = MiyamotoNagaiDisk(M_d=params['M_d'], a=params['a_d'], b=params['b_d'])
    return np.sqrt(np.clip(BaryonicMW(bulge, disk).v2(R), 0.0, None))


def v_c_baryon_multi(R: np.ndarray, p: Dict[str, float]) -> np.ndarray:
    bulge = HernquistBulge(M_b=p['M_b'], a_b=p['a_b'])
    disks = [
        MiyamotoNagaiDisk(M_d=p['M_thin'],  a=p['a_thin'],  b=p['b_thin']),
        MiyamotoNagaiDisk(M_d=p['M_thick'], a=p['a_thick'], b=p['b_thick']),
        MiyamotoNagaiDisk(M_d=p['M_HI'],    a=p['a_HI'],    b=p['b_HI']),
        MiyamotoNagaiDisk(M_d=p['M_H2'],    a=p['a_H2'],    b=p['b_H2']),
    ]
    return np.sqrt(np.clip(MultiDiskMW(bulge, disks).v2(R), 0.0, None))


def v_c_nfw(R: np.ndarray, V200: float, c: float, R200_kpc: float = 220.0) -> np.ndarray:
    halo = NFW(V200=V200, c=c, R200_kpc=R200_kpc)
    return np.sqrt(np.clip(halo.v2(R), 0.0, None))


def v_flat_from_anchor(M_enclosed: float, R_boundary: float, xi: float) -> float:
    # From M_encl and R_b: v_flat^2 ~ G M / R_b up to factor xi
    return np.sqrt(np.clip(xi * G_KPC * M_enclosed / max(R_boundary, 1e-6), 0.0, None))


# -----------------------------
# MOND baseline (from baryon GR curve)
# -----------------------------

def nu_simple(y: np.ndarray) -> np.ndarray:
    # Simple mu: mu(x)=x/(1+x) => nu(y)=0.5+sqrt(0.25+1/y)
    y = np.asarray(y, dtype=float)
    eps = 1e-30
    return 0.5 + np.sqrt(0.25 + 1.0 / np.maximum(y, eps))


def nu_standard(y: np.ndarray) -> np.ndarray:
    # Standard mu: mu(x)=x/sqrt(1+x^2) => nu(y)=sqrt(0.5+0.5*sqrt(1+4/y^2))
    y = np.asarray(y, dtype=float)
    eps = 1e-30
    return np.sqrt(0.5 + 0.5 * np.sqrt(1.0 + 4.0 / (np.maximum(y, eps) ** 2)))


def v_c_mond_from_vbar(R: np.ndarray, vbar_kms: np.ndarray, a0_m_s2: float = A0_M_S2, kind: str = 'simple') -> np.ndarray:
    """Compute MOND circular speed from baryonic GR speed vbar(R).
    Units: R [kpc], vbar [km/s], a0 [m/s^2]. Returns v_mond [km/s].
    g_N = vbar^2 / R, with consistent units; MOND: g = nu(y) g_N, y=g_N/a0; v^2 = g R = nu * vbar^2.
    """
    R = np.asarray(R, dtype=float)
    vbar_kms = np.asarray(vbar_kms, dtype=float)
    # Compute y = g_N / a0 with consistent units: g_N(m/s^2) = vbar(km/s)^2 * 1e6 / (R(kpc) * KPC_M)
    y = (np.power(vbar_kms, 2) * 1.0e6) / (np.maximum(R, 1e-9) * KPC_M * max(a0_m_s2, 1e-30))
    if kind == 'standard':
        nu = nu_standard(y)
    else:
        nu = nu_simple(y)
    v_mond2 = np.maximum(nu, 0.0) * np.power(vbar_kms, 2)
    return np.sqrt(np.clip(v_mond2, 0.0, None))
