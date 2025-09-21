# rigor/scripts/cluster_hse_muphi_plus.py
from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- constants (kpc, Msun, km/s, keV) ----
G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)
MU = 0.61           # mean molecular weight (particle), dimensionless
MU_E = 1.17         # mean molecular weight per electron, dimensionless
M_P = 1.67262192369e-27  # kg
J_PER_KEV = 1.602176634e-16
KM2_PER_S2_TO_J_PER_KG = 1.0e6  # 1 (km/s)^2 = 1e6 J/kg

# kT [keV] = (MU * M_P) * (potential [km^2/s^2]) * 1e6 / (1 keV in J)
KMS2_TO_KEV = (MU * M_P * KM2_PER_S2_TO_J_PER_KG) / J_PER_KEV


@dataclass
class MuPhiPlus:
    eps: float = 2.0         # dimensionless amplitude multiplier
    v_c_kms: float = 140.0   # km/s, sqrt(Phi_c)
    p: float = 2.0           # shape exponent in f(x) = 1 / (1 + x^p)
    eta: float = 0.20        # weak global mass-coupling exponent
    Mref_Msun: float = 6.0e10  # reference mass for mass-coupling

    def f(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.power(np.clip(x, 0.0, None), self.p))


# ---- I/O helpers ----
def load_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns {missing}. Found {list(df.columns)}")
    return df


def ne_to_rho_gas_Msun_kpc3(n_e_cm3: np.ndarray, mu_e: float = MU_E) -> np.ndarray:
    # rho_gas = mu_e * m_p * n_e
    # convert (g/cm^3) to (Msun/kpc^3): factor = (kpc/cm)^3 / (Msun/g)
    kpc_in_cm = 3.0856775814913673e21
    Msun_in_g = 1.988409870698051e33
    g_per_cm3 = mu_e * 1.67262192369e-24 * np.asarray(n_e_cm3)  # g/cm^3
    return g_per_cm3 * (kpc_in_cm**3) / Msun_in_g   # Msun/kpc^3


# ---- numerics ----
def enclosed_mass(r_kpc: np.ndarray, rho_Msun_kpc3: np.ndarray) -> np.ndarray:
    r = np.asarray(r_kpc); rho = np.asarray(rho_Msun_kpc3)
    integrand = 4.0 * math.pi * (r**2) * rho
    M = np.zeros_like(r)
    if r.size >= 2:
        dr = np.diff(r)
        area = 0.5 * (integrand[1:] + integrand[:-1]) * dr
        M[1:] = np.cumsum(area)
    return M


def reversed_cumtrapz(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x); y = np.asarray(y)
    S = np.zeros_like(y)
    if x.size >= 2:
        dx = np.diff(x)
        area = 0.5 * (y[1:] + y[:-1]) * dx
        S[:-1] = area[::-1].cumsum()[::-1]
    return S


# ---- hydrostatic temperature (integral form, robust) ----
def hydrostatic_kT_keV(r_kpc: np.ndarray, n_e_cm3: np.ndarray, g_tot_kms2_per_kpc: np.ndarray):
    # n_g = (MU_E / MU) * n_e
    n_g_cm3 = (MU_E / MU) * np.asarray(n_e_cm3)
    # I(r) = \int_r^{Rmax} n_g(r') g_tot(r') dr'  [ (km/s)^2 * cm^-3 * kpc ]
    I = reversed_cumtrapz(r_kpc, n_g_cm3 * g_tot_kms2_per_kpc)
    potential_kms2 = I / np.maximum(n_g_cm3, 1e-99)
    kT_keV = KMS2_TO_KEV * potential_kms2
    return kT_keV, potential_kms2


# ---- pipeline ----
def run_cluster(cluster: str = "ABELL_1689",
                base: str = "data/clusters",
                outdir: str = "out/clusters",
                params: MuPhiPlus = MuPhiPlus()):
    cdir = Path(base)/cluster
    g = load_csv(cdir/"gas_profile.csv", ["r_kpc","n_e_cm3"])
    t = load_csv(cdir/"temp_profile.csv", ["r_kpc","kT_keV"])
    s_path = cdir/"stars_profile.csv"
    s = pd.read_csv(s_path) if s_path.exists() else None

    # common radius grid
    r = np.unique(np.concatenate([g["r_kpc"].values, t["r_kpc"].values]))
    r = r[(r>0) & np.isfinite(r)]
    r = np.sort(r)
    if r.size < 2:
        raise ValueError("Not enough radial points to run HSE.")

    # interpolate n_e and optional rho_star
    ne = np.interp(r, g["r_kpc"], g["n_e_cm3"])
    rho_gas = ne_to_rho_gas_Msun_kpc3(ne)
    if s is not None and "rho_star_Msun_per_kpc3" in s.columns:
        rho_star = np.interp(r, s["r_kpc"], s["rho_star_Msun_per_kpc3"])
    else:
        rho_star = np.zeros_like(r)
    rho_b = rho_gas + rho_star

    # enclosed baryon mass and baryon-only acceleration
    Mb = enclosed_mass(r, rho_b)                # Msun
    g_N = G * Mb / np.maximum(r**2, 1e-12)      # (km/s)^2 / kpc

    # Newtonian potential depth from baryons-only: Phi_N(r) = \int_r^{Rmax} g_N(r') dr' (km^2/s^2)
    Phi_N = reversed_cumtrapz(r, g_N)

    # MuPhi+ response
    Phi_c = params.v_c_kms**2
    x = np.abs(Phi_N) / max(Phi_c, 1e-12)
    f = params.f(x)
    mass_boost = np.power(np.clip(Mb / max(params.Mref_Msun, 1e-30), 1e-30, None), params.eta)
    mu = 1.0 + params.eps * f * mass_boost

    # total acceleration
    g_tot = g_N * mu

    # hydrostatic temperature prediction
    kT_pred, phi_eff = hydrostatic_kT_keV(r, ne, g_tot)

    # lensing-equivalent mass (if one interprets g_tot as GM(<r)/r^2)
    M_pred = (r**2)*g_tot / G

    # observed T on this grid
    kT_obs = np.interp(r, t["r_kpc"], t["kT_keV"])
    mask = np.isfinite(kT_obs) & (kT_obs>0)
    frac_err = np.abs(kT_pred[mask]-kT_obs[mask]) / np.maximum(kT_obs[mask], 1e-12)
    med_frac = float(np.median(frac_err)) if mask.any() else float("nan")

    # save metrics and plot
    od = Path(outdir)/cluster
    od.mkdir(parents=True, exist_ok=True)
    with open(od/"cluster_muphi_plus_metrics.json","w") as f:
        json.dump({
            "cluster": cluster,
            "params": {
                "eps": params.eps, "v_c_kms": params.v_c_kms, "p": params.p,
                "eta": params.eta, "Mref_Msun": params.Mref_Msun
            },
            "r_kpc_min": float(r.min()), "r_kpc_max": float(r.max()), "n_r": int(len(r)),
            "temp_median_frac_err": med_frac
        }, f, indent=2)

    # plot
    plt.figure(figsize=(11,5))
    # Mass panel
    plt.subplot(1,2,1)
    plt.loglog(r, Mb, label="Baryons (gas+stars)")
    plt.loglog(r, M_pred, ls="--", label="Total (MuPhi+)")
    lm_path = cdir/"lensing_mass.csv"
    if lm_path.exists():
        try:
            lm = pd.read_csv(lm_path)
            plt.errorbar(lm["r_kpc"], lm["M_enclosed_Msun"], yerr=lm.get("M_err_Msun"), fmt="o", ms=3, label="Lensing")
        except Exception:
            pass
    plt.xlabel("r [kpc]"); plt.ylabel("M(<r) [Msun]"); plt.title(f"{cluster}: enclosed mass")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)

    # Temperature panel
    plt.subplot(1,2,2)
    plt.semilogx(r, kT_pred, ls="--", label="kT (pred, HSE+MuPhi+)")
    plt.errorbar(t["r_kpc"], t["kT_keV"], yerr=t.get("kT_err_keV"), fmt="o", ms=3, label="kT (X-ray)")
    plt.xlabel("r [kpc]"); plt.ylabel("kT [keV]"); plt.title(f"{cluster}: temperature (HSE)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(od/"cluster_muphi_plus_results.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--base", default="data/clusters")
    ap.add_argument("--outdir", default="out/clusters")
    ap.add_argument("--eps", type=float, default=2.0)
    ap.add_argument("--v_c_kms", type=float, default=140.0)
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--eta", type=float, default=0.20)
    ap.add_argument("--Mref_Msun", type=float, default=6.0e10)
    args = ap.parse_args()

    mp = MuPhiPlus(eps=args.eps, v_c_kms=args.v_c_kms, p=args.p, eta=args.eta, Mref_Msun=args.Mref_Msun)
    run_cluster(cluster=args.cluster, base=args.base, outdir=args.outdir, params=mp)
