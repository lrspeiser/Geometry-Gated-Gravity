# rigor/scripts/cluster_hse_logtail.py
import json, math, numpy as np, pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
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
class LogTail:
    v0: float = 140.0   # km/s
    rc: float = 15.0    # kpc
    r0: float = 3.0     # kpc
    delta: float = 4.0  # kpc
    mass_coupled_A: Optional[float] = None  # if not None, v0 = A*(Mb_tot/1e10)^(1/4)

    def gate(self, r):
        r = np.asarray(r)
        return 0.5*(1.0 + np.tanh((r - self.r0)/max(self.delta,1e-9)))

    def g_tail(self, r, v0=None):
        r = np.asarray(r)
        v02 = (self.v0 if v0 is None else v0)**2
        return v02 * self.gate(r) / (np.maximum(r + self.rc, 1e-12))

# ---- I/O helpers ----
def load_csv(path, cols):
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns {missing}. Found {list(df.columns)}")
    return df

def ne_to_rho_gas_Msun_kpc3(n_e_cm3, mu_e=MU_E):
    # rho_gas = mu_e * m_p * n_e
    # convert (g/cm^3) to (Msun/kpc^3): factor = (kpc/cm)^3 / (Msun/g)
    kpc_in_cm = 3.0856775814913673e21
    Msun_in_g = 1.988409870698051e33
    g_per_cm3 = mu_e * 1.67262192369e-24 * np.asarray(n_e_cm3)  # g/cm^3
    return g_per_cm3 * (kpc_in_cm**3) / Msun_in_g   # Msun/kpc^3

# ---- numerics ----
def enclosed_mass(r_kpc, rho_Msun_kpc3):
    r = np.asarray(r_kpc); rho = np.asarray(rho_Msun_kpc3)
    integrand = 4.0 * math.pi * (r**2) * rho
    M = np.zeros_like(r)
    if r.size >= 2:
        dr = np.diff(r)
        area = 0.5 * (integrand[1:] + integrand[:-1]) * dr
        M[1:] = np.cumsum(area)
    return M

def reversed_cumtrapz(x, y):
    x = np.asarray(x); y = np.asarray(y)
    S = np.zeros_like(y)
    if x.size >= 2:
        dx = np.diff(x)
        area = 0.5 * (y[1:] + y[:-1]) * dx
        S[:-1] = area[::-1].cumsum()[::-1]
    return S

# ---- hydrostatic temperature (integral form, robust) ----
def hydrostatic_kT_keV(r_kpc, n_e_cm3, g_tot_kms2_per_kpc):
    # n_g = (MU_E / MU) * n_e
    n_g_cm3 = (MU_E / MU) * np.asarray(n_e_cm3)
    # I(r) = \int_r^{Rmax} n_g(r') g_tot(r') dr'  [ (km/s)^2 * cm^-3 * kpc ]
    I = reversed_cumtrapz(r_kpc, n_g_cm3 * g_tot_kms2_per_kpc)
    potential_kms2 = I / np.maximum(n_g_cm3, 1e-99)
    kT_keV = KMS2_TO_KEV * potential_kms2
    return kT_keV, potential_kms2

# ---- pipeline ----
def run_cluster(cluster="ABELL_1689",
                base="data/clusters",
                outdir="out/clusters",
                logtail=LogTail(),
                mass_coupled=False,
                A_kms: float = 160.0,
                Mb_for_masscouple: Optional[float]=None,
                clump_profile_csv: Optional[str]=None):
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
    # Optional radial clumping profile: n_e -> sqrt(C(r)) * n_e
    if clump_profile_csv is not None:
        try:
            cp = pd.read_csv(clump_profile_csv)
            r_cp = np.asarray(cp['r_kpc'], float)
            C_cp = np.asarray(cp['C'], float)
            C_interp = np.interp(r, r_cp, C_cp, left=C_cp[0], right=C_cp[-1])
            ne = np.sqrt(np.maximum(C_interp, 1.0)) * ne
        except Exception as e:
            print(f"[LogTail-HSE] Warning: failed to apply clump_profile_csv ({e}); proceeding without profile")
    rho_gas = ne_to_rho_gas_Msun_kpc3(ne)
    if s is not None and "rho_star_Msun_per_kpc3" in s.columns:
        rho_star = np.interp(r, s["r_kpc"], s["rho_star_Msun_per_kpc3"])
    else:
        rho_star = np.zeros_like(r)
    rho_b = rho_gas + rho_star

    # enclosed baryon mass and baryon-only acceleration
    Mb = enclosed_mass(r, rho_b)                # Msun
    g_b = G * Mb / np.maximum(r**2, 1e-12)      # (km/s)^2 / kpc

    # decide v0 if mass-coupled is requested
    v0_use = None
    if mass_coupled:
        Mb_tot = Mb_for_masscouple if Mb_for_masscouple is not None else float(Mb[-1])
        v0_use = A_kms * (max(Mb_tot, 1e6)/1e10)**0.25  # km/s

    # total acceleration
    g_t = g_b + logtail.g_tail(r, v0=v0_use)

    # hydrostatic temperature prediction
    kT_pred, phi_eff = hydrostatic_kT_keV(r, ne, g_t)

    # lensing-equivalent mass
    M_pred = (r**2)*g_t / G

    # observed T on this grid
    kT_obs = np.interp(r, t["r_kpc"], t["kT_keV"])
    mask = np.isfinite(kT_obs) & (kT_obs>0)
    frac_err = np.abs(kT_pred[mask]-kT_obs[mask]) / np.maximum(kT_obs[mask], 1e-12)
    med_frac = float(np.median(frac_err)) if mask.any() else float("nan")

    # save metrics and plot
    od = Path(outdir)/cluster
    od.mkdir(parents=True, exist_ok=True)
    with open(od/"cluster_logtail_metrics.json","w") as f:
        json.dump({
            "cluster": cluster,
            "params": {"v0": (logtail.v0 if v0_use is None else v0_use), "rc": logtail.rc, "r0": logtail.r0, "delta": logtail.delta,
                       "mass_coupled": bool(mass_coupled), "A_kms": A_kms},
            "r_kpc_min": float(r.min()), "r_kpc_max": float(r.max()), "n_r": int(len(r)),
            "temp_median_frac_err": med_frac
        }, f, indent=2)

    # plot
    plt.figure(figsize=(11,5))
    # Mass panel
    plt.subplot(1,2,1)
    plt.loglog(r, Mb, label="Baryons (gas+stars)")
    plt.loglog(r, M_pred, ls="--", label="Total (LogTail)")
    lm_path = cdir/"lensing_mass.csv"
    if lm_path.exists():
        lm = pd.read_csv(lm_path)
        plt.errorbar(lm["r_kpc"], lm["M_enclosed_Msun"], yerr=lm.get("M_err_Msun"), fmt="o", ms=3, label="Lensing")
    plt.xlabel("r [kpc]"); plt.ylabel("M(<r) [Msun]"); plt.title(f"{cluster}: enclosed mass")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)

    # Temperature panel
    plt.subplot(1,2,2)
    plt.semilogx(r, kT_pred, ls="--", label="kT (pred, HSE+LogTail)")
    plt.errorbar(t["r_kpc"], t["kT_keV"], yerr=t.get("kT_err_keV"), fmt="o", ms=3, label="kT (X-ray)")
    plt.xlabel("r [kpc]"); plt.ylabel("kT [keV]"); plt.title(f"{cluster}: temperature (HSE)")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(od/"cluster_logtail_results.png", dpi=140)
    plt.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--base", default="data/clusters")
    ap.add_argument("--outdir", default="out/clusters")
    ap.add_argument("--mass_coupled", action="store_true")
    ap.add_argument("--A_kms", type=float, default=160.0, help="A for v0(Mb)=A*(Mb/1e10)^(1/4)")
    ap.add_argument("--clump_profile_csv", type=str, default=None,
                    help="CSV with columns r_kpc,C; applies n_e -> sqrt(C(r)) * n_e")
    args = ap.parse_args()

    lt = LogTail(v0=140.0, rc=15.0, r0=3.0, delta=4.0,
                 mass_coupled_A=(args.A_kms if args.mass_coupled else None))
    run_cluster(cluster=args.cluster, base=args.base, outdir=args.outdir,
                logtail=lt, mass_coupled=args.mass_coupled, A_kms=args.A_kms,
                clump_profile_csv=args.clump_profile_csv)
