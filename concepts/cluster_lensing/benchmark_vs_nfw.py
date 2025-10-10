#!/usr/bin/env python3
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# Ensure project root on sys.path so we can import concepts.* modules
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reuse cosmology and projection helpers from existing module
from concepts.cluster_lensing.cluster_lensing_analysis_real_sigma import (
    Ez,
    angular_diameter_distance_kpc,
    sigma_crit_Msun_per_kpc2,
    abel_project_sigma,
    sigma_to_Mproj,
    G,
    H0,
    Omega_m,
    Omega_L,
)

Mpc_to_kpc = 1000.0

TRAINING_CLUSTERS = ["MACSJ0416", "MACSJ0717", "MACSJ1149"]
OUT_DIR = Path("out/cluster_lensing_comparison")

# Map cluster_id -> directory name used under out/cluster_lensing_real/
OUT_REAL_DIR_MAP = {
    "MACSJ0416": "macs0416",
    "MACSJ0717": "macs0717",
    "MACSJ1149": "macs1149",
}


def load_nfw_params(p: Path) -> Dict[str, dict]:
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for entry in data.get("clusters", []):
        cid = entry["cluster_id"].upper()
        out[cid] = entry
    return out


def load_gold_standard(p: Path) -> Dict[str, dict]:
    data = json.loads(p.read_text(encoding="utf-8"))
    # Keys may be lower-case like "macs0416"
    out = {}
    for k, v in data.items():
        out[k.upper()] = v
    return out


def critical_density_Msun_kpc3(z: float) -> float:
    # H(z) in km/s/Mpc
    Hz_Mpc = H0 * Ez(z)
    # convert to km/s/kpc
    Hz_kpc = Hz_Mpc / Mpc_to_kpc
    # rho_c = 3 H^2 / (8 pi G)
    return 3.0 * (Hz_kpc ** 2) / (8.0 * math.pi * G)


def nfw_r200c_from_M(M200c: float, z: float) -> float:
    rho_c = critical_density_Msun_kpc3(z)
    r200c_kpc = (3.0 * M200c / (4.0 * math.pi * 200.0 * rho_c)) ** (1.0 / 3.0)
    return float(r200c_kpc)


def nfw_rho_s_delta_c(c200c: float, z: float) -> Tuple[float, float]:
    rho_c = critical_density_Msun_kpc3(z)
    f = math.log(1.0 + c200c) - c200c / (1.0 + c200c)
    delta_c = (200.0 / 3.0) * (c200c ** 3) / f
    rho_s = delta_c * rho_c
    return rho_s, delta_c


def nfw_density_profile(r_kpc: np.ndarray, M200c: float, c200c: float, z: float) -> Tuple[np.ndarray, float, float, float]:
    r200 = nfw_r200c_from_M(M200c, z)
    r_s = r200 / c200c
    rho_s, _ = nfw_rho_s_delta_c(c200c, z)
    x = np.maximum(r_kpc / r_s, 1e-12)
    rho = rho_s / (x * (1.0 + x) ** 2)
    return rho, r200, r_s, rho_s


def _g_function(x: float) -> float:
    # Piecewise g(x) per Wright & Brainerd (2000)
    if x < 1.0:
        t = math.sqrt((1.0 - x) / (1.0 + x))
        val = math.log(max(x / 2.0, 1e-300)) + (2.0 / math.sqrt(1.0 - x * x)) * math.atanh(t)
        return val
    elif x > 1.0:
        t = math.sqrt((x - 1.0) / (1.0 + x))
        val = math.log(x / 2.0) + (2.0 / math.sqrt(x * x - 1.0)) * math.atan(t)
        return val
    else:
        # x == 1
        return 1.0 - math.log(2.0)


def nfw_theta_E_arcsec(M200c: float, c200c: float, z_lens: float, z_source: float) -> Optional[Tuple[float, float]]:
    # Analytic NFW mean convergence: kappa_bar(x) = 4 * kappa_s * g(x) / x^2,
    # where kappa_s = rho_s * r_s / Sigma_crit, x = R / r_s.
    r200 = nfw_r200c_from_M(M200c, z_lens)
    # Avoid zero or negative
    if r200 <= 0 or c200c <= 0:
        return None
    r_s = r200 / c200c
    rho_s, _ = nfw_rho_s_delta_c(c200c, z_lens)

    Sigma_crit = sigma_crit_Msun_per_kpc2(z_lens, z_source)
    if not np.isfinite(Sigma_crit) or Sigma_crit <= 0:
        return None
    kappa_s = (rho_s * r_s) / Sigma_crit

    # Scan x over a log grid to find where kappa_bar crosses unity
    x_grid = np.logspace(-4, 2, 2000)
    g_vals = np.array([_g_function(float(x)) for x in x_grid])
    kappa_bar = 4.0 * kappa_s * g_vals / (x_grid ** 2)

    # Find first index where kappa_bar >= 1.0
    idx = np.where(kappa_bar >= 1.0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    if i == 0:
        x_E = x_grid[0]
    else:
        x0, y0 = x_grid[i - 1], kappa_bar[i - 1]
        x1, y1 = x_grid[i], kappa_bar[i]
        if y1 == y0:
            x_E = x1
        else:
            # Linear interpolation in log x for stability
            lx0, lx1 = math.log(x0), math.log(x1)
            ly0, ly1 = y0, y1
            # Interpolate to y=1
            t = (1.0 - ly0) / (ly1 - ly0)
            lxE = lx0 + t * (lx1 - lx0)
            x_E = math.exp(lxE)

    R_E = x_E * r_s
    Dd = angular_diameter_distance_kpc(z_lens)
    theta_rad = R_E / max(Dd, 1e-12)
    theta_arcsec = theta_rad * (180.0 / math.pi) * 3600.0
    return float(R_E), float(theta_arcsec)


def read_baryon_theta_arcsec(cluster_id: str) -> Optional[float]:
    out_root = Path("out/cluster_lensing_real")
    sub = OUT_REAL_DIR_MAP.get(cluster_id.upper())
    if not sub:
        return None
    p = out_root / sub / "summary_realSigma.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        v = data.get("Einstein_radius_arcsec_realSigma")
        return None if v is None else float(v)
    except Exception:
        return None


def main():
    root = Path(__file__).resolve().parents[2]
    nfw_path = root / "data" / "literature" / "nfw_params.json"
    gold_path = root / "data" / "frontier" / "gold_standard" / "gold_standard_clusters.json"

    nfw = load_nfw_params(nfw_path)
    gold = load_gold_standard(gold_path)

    rows: List[Dict] = []

    for cid in TRAINING_CLUSTERS:
        # Observed θ_E and z_s from gold standard if available
        gold_key = cid.replace("MACSJ", "MACS")  # keys in gold_standard use lowercase macs0416; but we'll handle below
        # gold_standard keys like "macs0416"; unify to uppercase
        obs_entry = None
        for k, v in gold.items():
            if cid[-4:] in k:  # match by trailing digits e.g., 0416
                obs_entry = v
                break
        z_l = nfw[cid]["z_lens"] if cid in nfw else None
        z_s = obs_entry["accepted"]["zs"] if obs_entry else 2.0
        theta_obs = obs_entry["accepted"]["theta_E_arcsec"] if obs_entry else None

        # NFW prediction
        M200 = nfw[cid]["M_200c_Msun"]
        c200 = nfw[cid]["c_200c"]
        res = nfw_theta_E_arcsec(M200, c200, z_l, z_s)
        REnfw_kpc = None
        theta_nfw = None
        if res is not None:
            REnfw_kpc, theta_nfw = res

        # Our baryon-only prediction (current pipeline)
        theta_baryon = read_baryon_theta_arcsec(cid)

        rows.append({
            "cluster_id": cid,
            "z_lens": z_l,
            "z_source": z_s,
            "theta_E_obs_arcsec": theta_obs,
            "theta_E_pred_baryon_arcsec": theta_baryon,
            "theta_E_pred_nfw_arcsec": theta_nfw,
            "RE_nfw_kpc": REnfw_kpc,
            "M200c_Msun": M200,
            "c200c": c200,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "benchmark_vs_nfw.csv"
    json_path = OUT_DIR / "benchmark_vs_nfw.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")

    # Print small summary
    print("\nSummary (arcsec):")
    for r in rows:
        print(f"{r['cluster_id']}: obs={r['theta_E_obs_arcsec']}, baryon={r['theta_E_pred_baryon_arcsec']}, nfw={r['theta_E_pred_nfw_arcsec']}")


if __name__ == "__main__":
    main()
