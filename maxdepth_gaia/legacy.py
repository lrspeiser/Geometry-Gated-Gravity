# legacy.py
# Deprecated: Not used by the new maxdepth_gaia pipeline. Kept for reference.
# This file preserves earlier model functions (BulgeHernquist, DiskExponential, HaloNFW, SaturatedWell)
# Do not import this in new code. See models.py for the active implementations.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.special import iv, kv

G_KPC = 4.30091e-6  # (kpc km^2 s^-2) / Msun

@dataclass
class BulgeHernquist:
    M: float = 5e9      # Msun
    a: float = 0.7      # kpc
    def v2(self, R: np.ndarray) -> np.ndarray:
        # v^2 = G M(<R) / R ; M(<R) = M R^2 / (R + a)^2
        return G_KPC * self.M * R / (R + self.a)**2

@dataclass
class DiskExponential:
    M: float = 6e10     # Msun
    Rd: float = 2.6     # kpc
    def v2(self, R: np.ndarray) -> np.ndarray:
        # Freeman disk: v^2(R) = 4π G Σ0 Rd y^2 [I0(y)K0(y) - I1(y)K1(y)], y=R/(2Rd)
        Sigma0 = self.M / (2.0 * np.pi * self.Rd**2)
        y = R / (2.0 * self.Rd)
        bessel = iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y)
        return 4.0 * np.pi * G_KPC * Sigma0 * self.Rd * (y**2) * bessel

@dataclass
class HaloNFW:
    rho_s: float = 0.008  # Msun / pc^3  (will be converted to Msun/kpc^3)
    r_s: float = 16.0     # kpc
    def v2(self, R: np.ndarray) -> np.ndarray:
        rho_s_kpc = self.rho_s * 1.0e9  # Msun/kpc^3
        x = R / self.r_s
        M_enclosed = 4.0 * np.pi * rho_s_kpc * self.r_s**3 * (np.log(1.0 + x) - x/(1.0 + x))
        return G_KPC * M_enclosed / np.maximum(R, 1e-6)

@dataclass
class SaturatedWell:
    v_flat: float = 180.0  # km/s
    R_s: float = 3.0       # kpc
    m: float = 2.0         # dimensionless
    def v2_extra(self, R: np.ndarray) -> np.ndarray:
        return (self.v_flat**2) * (1.0 - np.exp(-(R/np.maximum(self.R_s,1e-6))**self.m))
    def lensing_deflection_rad(self) -> float:
        # Heuristic for a log-like potential: alpha ~ 2π v_flat^2 / c^2
        c = 2.99792458e5  # km/s
        return 2.0 * np.pi * (self.v_flat**2) / (c**2)

@dataclass
class Baryons:
    bulge: BulgeHernquist
    disk: DiskExponential
    def v2(self, R: np.ndarray) -> np.ndarray:
        return self.bulge.v2(R) + self.disk.v2(R)

# Legacy model wrappers (square root applied)
def v_model_baryons_only(R, M_bulge, a_bulge, M_disk, R_d):
    bulge = BulgeHernquist(M=M_bulge, a=a_bulge)
    disk = DiskExponential(M=M_disk, Rd=R_d)
    return np.sqrt(Baryons(bulge, disk).v2(R))

def v_model_baryons_nfw(R, M_bulge, a_bulge, M_disk, R_d, rho_s, r_s):
    bulge = BulgeHernquist(M=M_bulge, a=a_bulge)
    disk = DiskExponential(M=M_disk, Rd=R_d)
    halo = HaloNFW(rho_s=rho_s, r_s=r_s)
    v2 = Baryons(bulge, disk).v2(R) + halo.v2(R)
    return np.sqrt(v2)

def v_model_baryons_saturated(R, M_bulge, a_bulge, M_disk, R_d, v_flat, R_s, m):
    bulge = BulgeHernquist(M=M_bulge, a=a_bulge)
    disk = DiskExponential(M=M_disk, Rd=R_d)
    sat = SaturatedWell(v_flat=v_flat, R_s=R_s, m=m)
    v2 = Baryons(bulge, disk).v2(R) + sat.v2_extra(R)
    return np.sqrt(v2)
