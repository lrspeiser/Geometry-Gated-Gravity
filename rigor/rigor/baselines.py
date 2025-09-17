
"""Simple comparative baselines: GR, MOND (simple/standard mu), Burkert halo.

We implement MOND using the algebraic relation a * mu(a/a0) = a_N, with:
- simple:   mu(x) = x/(1+x)
- standard: mu(x) = x / sqrt(1+x^2)

Given a_N = Vbar^2 / R, we solve analytically for 'a' for both choices and return Vpred^2 = a * R.

For Burkert, we implement the enclosed mass analytic form to compute V_halo^2.
"""
from __future__ import annotations
import numpy as np

KPC = 3.085677581e19
def _aN_from_VbarR(Vbar_kms, R_kpc):
    V2 = (Vbar_kms*1e3)**2
    Rm = np.maximum(R_kpc, 1e-6)*KPC
    return V2 / Rm

def mond_simple(Vbar_kms, R_kpc, a0=1.2e-10):
    aN = _aN_from_VbarR(Vbar_kms, R_kpc)
    # a = 0.5 * aN * (1 + sqrt(1 + 4 a0/aN))
    a = 0.5*aN*(1.0 + np.sqrt(1.0 + 4.0*a0/np.maximum(aN,1e-30)))
    V2 = a * (R_kpc*KPC)
    return np.sqrt(np.maximum(V2, 0.0))/1e3

def mond_standard(Vbar_kms, R_kpc, a0=1.2e-10):
    aN = _aN_from_VbarR(Vbar_kms, R_kpc)
    # Solve a * (a/sqrt(a^2 + a0^2)) = aN => a^2 / sqrt(a^2 + a0^2) = aN
    # Numeric solve is safer
    a = np.zeros_like(aN)
    for i,x in enumerate(aN):
        # Newton iteration
        ai = x
        for _ in range(20):
            f  = (ai**2)/np.sqrt(ai**2 + a0**2) - x
            df = (2*ai)/np.sqrt(ai**2+a0**2) - (ai**3)/( (ai**2+a0**2)**(3/2) )
            ai = ai - f/np.maximum(df, 1e-30)
            ai = np.maximum(ai, 1e-20)
        a[i] = ai
    V2 = a * (R_kpc*KPC)
    return np.sqrt(np.maximum(V2, 0.0))/1e3

def burkert_velocity_kms(R_kpc, rho0, r0_kpc):
    # Enclosed mass for Burkert: M(r) = pi*r0^3*rho0*[ ln(1+r/r0)^2*(1+r^2/r0^2)/2 + ln(1 + r/r0) - arctan(r/r0) + 0.5*ln(1 + (r/r0)^2) ]
    # We'll use the standard expression for V^2 = G M(r)/r
    G = 4.30091e-6  # kpc * (km/s)^2 / Msun
    r = np.maximum(R_kpc, 1e-6)
    x = r/np.maximum(r0_kpc, 1e-6)
    term1 = 0.5*np.log(1+x**2) + np.log(1+x) - np.arctan(x)
    M = np.pi * (r0_kpc**3) * rho0 * (np.log(1+x) + 0.5*np.log(1+x**2) - np.arctan(x))
    # Some literature differ in constants; we adopt a widely used form; robustness test later.
    V2 = G * M / r
    return np.sqrt(np.maximum(V2,0.0))

def gr_baseline(Vbar_kms):
    return Vbar_kms  # in our notation, GR ==> xi=1
