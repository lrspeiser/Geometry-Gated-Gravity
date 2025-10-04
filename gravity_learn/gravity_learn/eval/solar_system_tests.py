#!/usr/bin/env python
from __future__ import annotations
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from gravity_learn.features.geometry import sigma_hat, grad_log_sigma

# Physical constants (consistent with repository usage)
G = 4.300917270e-6  # kpc km^2 s^-2 Msun^-1
AU_PER_KPC = 206265000.0  # 1 kpc = 206,265,000 AU (exact by definition of pc)
AU_TO_KPC = 1.0 / AU_PER_KPC


@dataclass
class PlanetSpec:
    name: str
    R_AU: float


PLANETS_DEFAULT: List[PlanetSpec] = [
    PlanetSpec("Mercury", 0.387),
    PlanetSpec("Earth", 1.000),
    PlanetSpec("Jupiter", 5.204),
    PlanetSpec("Saturn", 9.537),
    PlanetSpec("Neptune", 30.07),
]


def build_sigma_profile_for_solar_system(R_AU_min: float = 0.3, R_AU_max: float = 40.0, n: int = 400):
    """Construct a smooth Σ(R) profile for the Solar System dominated by the Sun.
    We take Σ(R) ∝ 1 / R^2 (Sun mass spread over disk of radius R) up to a constant.
    The constant cancels under Σ̂ = Σ / median(Σ).

    Returns:
      R_kpc_grid, Sigma_grid (arbitrary units, positive), Sigma_hat_grid, dlnS_grid
    """
    R_AU_grid = np.logspace(math.log10(R_AU_min), math.log10(R_AU_max), n)
    R_kpc_grid = R_AU_grid * AU_TO_KPC
    # Σ ∝ 1 / R^2 (constant factor irrelevant for Σ̂)
    Sigma_grid = 1.0 / np.maximum(R_kpc_grid, 1e-30) ** 2
    Sh_grid = sigma_hat(Sigma_grid)  # dimensionless
    dlnS_grid = grad_log_sigma(R_kpc_grid, Sigma_grid)  # dimensionless slope d ln Σ / d ln R
    return R_kpc_grid, Sh_grid, dlnS_grid


def interpolate_features(R_kpc_target: float, R_kpc_grid: np.ndarray, Sh_grid: np.ndarray, dlnS_grid: np.ndarray):
    Sh = float(np.interp(R_kpc_target, R_kpc_grid, Sh_grid))
    dlnS = float(np.interp(R_kpc_target, R_kpc_grid, dlnS_grid))
    return Sh, dlnS


def fx_ratio_curv(x: float, Sh: float, dlnS: float, a: float, b: float, d: float):
    denom = (a - b * Sh - d * abs(dlnS))
    # Sign-preserving clip per repo practice
    if abs(denom) < 1e-6:
        denom = math.copysign(1e-6, denom if denom != 0.0 else 1.0)
    fX_raw = (x * x) / denom
    fX_nonneg = max(0.0, fX_raw)
    return fX_raw, fX_nonneg, denom


def orbital_speed_kms(R_kpc: float, M_sun_Msun: float = 1.0):
    return math.sqrt(max(0.0, G * M_sun_Msun / max(R_kpc, 1e-30)))


def run_solar_system_tests(
    planets: List[PlanetSpec],
    Rd_AU: float = 30.0,
    a: float = 0.6687,
    b: float = 0.1401,
    d: float = 0.0871,
    cassini_threshold_mps: float = 1e-3,
):
    Rd_kpc = Rd_AU * AU_TO_KPC
    R_kpc_grid, Sh_grid, dlnS_grid = build_sigma_profile_for_solar_system()

    print("Solar System tests for O2 ratio_curv (geometry-gated) — Cassini focus")
    print(f"Params: a={a}, b={b}, d={d} | Rd={Rd_AU} AU | gradient uses d ln Σ / d ln R (dimensionless)")
    print("")
    header = (
        f"{'Body':<10} {'R(AU)':>8} {'x=R/Rd':>10} {'Sh=Σ̂':>10} {'|dlnΣ|':>8} "
        f"{'denom':>12} {'fX_raw':>12} {'fX>=0':>12} {'vN km/s':>10} {'Δv m/s':>10} {'Cassini?':>10}"
    )
    print(header)
    print("-" * len(header))

    out: List[Dict[str, float]] = []

    for p in planets:
        R_kpc = p.R_AU * AU_TO_KPC
        x = R_kpc / max(Rd_kpc, 1e-30)
        Sh, dlnS = interpolate_features(R_kpc, R_kpc_grid, Sh_grid, dlnS_grid)
        fX_raw, fX_nonneg, denom = fx_ratio_curv(x, Sh, dlnS, a, b, d)
        vN = orbital_speed_kms(R_kpc)
        vmod = vN * math.sqrt(max(0.0, 1.0 + fX_nonneg))
        dv_mps = (vmod - vN) * 1000.0
        cassini_flag = (p.name == "Saturn") and (abs(dv_mps) > cassini_threshold_mps)

        print(f"{p.name:<10} {p.R_AU:8.3f} {x:10.4f} {Sh:10.4f} {abs(dlnS):8.3f} {denom:12.6g} {fX_raw:12.6g} {fX_nonneg:12.6g} {vN:10.3f} {dv_mps:10.4f} {str(cassini_flag):>10}")

        out.append({
            "name": p.name,
            "R_AU": p.R_AU,
            "x": x,
            "Sh": Sh,
            "abs_dlnS": abs(dlnS),
            "denom": denom,
            "fX_raw": fX_raw,
            "fX_nonneg": fX_nonneg,
            "vN_kms": vN,
            "delta_v_mps": dv_mps,
            "cassini_detectable": cassini_flag,
        })

    print("")
    sat = [r for r in out if r["name"] == "Saturn"]
    if sat:
        s = sat[0]
        verdict = "DETECTED" if s["cassini_detectable"] else "NOT detected"
        print(f"Cassini Doppler threshold ~1 mm/s: Δv(Saturn) = {s['delta_v_mps']:.6f} m/s → {verdict}")
    else:
        print("Saturn not in planet list; no Cassini check performed.")


def main():
    ap = argparse.ArgumentParser(description="Solar System tests for O2 ratio_curv (Cassini focus)")
    ap.add_argument("--Rd_AU", type=float, default=30.0, help="Scale length Rd in AU (default 30 AU)")
    ap.add_argument("--a", type=float, default=0.6687, help="Model parameter a")
    ap.add_argument("--b", type=float, default=0.1401, help="Model parameter b")
    ap.add_argument("--d", type=float, default=0.0871, help="Model parameter d")
    ap.add_argument("--cassini_mm_s", type=float, default=1.0, help="Cassini threshold in mm/s (default 1.0)")
    args = ap.parse_args()

    cassini_threshold_mps = args.cassini_mm_s / 1000.0
    run_solar_system_tests(
        planets=PLANETS_DEFAULT,
        Rd_AU=args.Rd_AU,
        a=args.a,
        b=args.b,
        d=args.d,
        cassini_threshold_mps=cassini_threshold_mps,
    )


if __name__ == "__main__":
    main()