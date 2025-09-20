#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster LogTail testbed

Applies the same LogTail law used in the galaxy pipeline (SPARC-global by default)
to a cluster's baryonic profiles (gas + stars), predicting enclosed mass and a
0th-order hydrostatic kT(r). Produces a two-panel PNG and a small JSON summary.

Schemas (place under data/clusters/<NAME>/):
- gas_profile.csv with either (r_kpc, rho_gas_Msun_per_kpc3) or (r_kpc, n_e_cm3)
- stars_profile.csv with (r_kpc, rho_stars_Msun_per_kpc3)
Optional:
- temp_profile.csv with (r_kpc, kT_keV[, kT_err_keV])
- lensing_mass.csv with (r_kpc, M_enclosed_Msun)
- gas_params.json (β-model) with {"ne0_cm3", "rc_kpc", "beta"} if no gas_profile.csv
"""

import json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.integrate import cumtrapz

# constants
G = 4.300917270e-6  # (kpc km^2 s^-2 Msun^-1)
MSUN_G = 1.98847e33
KPC_CM = 3.085677581491367e21
MP_G   = 1.67262192369e-24
KEV_PER_J = 6.241509074e15

# ICM composition
MU = 0.60   # mean molecular weight
MU_E = 1.17 # per free electron

# ---- LogTail (same as galaxy pipeline) ----
def _smooth_gate(r, r0, d):
    return 0.5*(1.0 + np.tanh((r - r0)/max(d, 1e-9)))

def logtail_v2(r_kpc, v0=140.0, rc=15.0, r0=3.0, delta=4.0):
    r = np.asarray(r_kpc, dtype=float)
    return (v0**2) * (r/(r + rc)) * _smooth_gate(r, r0, delta)

def total_accel(r_kpc, M_b_enclosed, p):
    r = np.asarray(r_kpc, dtype=float)
    g_b = G * np.asarray(M_b_enclosed, float) / np.maximum(r, 1e-12)**2
    g_t = logtail_v2(r, **p) / np.maximum(r, 1e-12)
    return g_b + g_t

# ---- IO helpers ----
def _col(df, *names):
    for n in names:
        if n in df.columns: return n
    raise KeyError(f"Missing one of {names}; columns={df.columns.tolist()}")

def load_gas(path_dir: Path) -> pd.DataFrame:
    csv = path_dir/"gas_profile.csv"
    if csv.exists():
        return pd.read_csv(csv)
    j = path_dir/"gas_params.json"
    if j.exists():
        prm = json.loads(j.read_text())
        n0 = float(prm["ne0_cm3"]); rc = float(prm["rc_kpc"]); beta = float(prm["beta"])
        r = np.geomspace(1, 3000, 400)
        ne = n0 * (1.0 + (r/rc)**2)**(-1.5*beta)
        return pd.DataFrame({"r_kpc": r, "n_e_cm3": ne})
    raise FileNotFoundError(f"No gas_profile.csv or gas_params.json in {path_dir}")

def gas_density_msun_kpc3(df: pd.DataFrame):
    if "rho_gas_Msun_per_kpc3" in df.columns:
        r = df[_col(df, "r_kpc", "R_kpc", "r")].to_numpy(float)
        rho = df["rho_gas_Msun_per_kpc3"].to_numpy(float)
        return r, rho
    # convert n_e to rho
    r = df[_col(df, "r_kpc", "R_kpc", "r")].to_numpy(float)
    ne = df[_col(df, "n_e_cm3", "ne_cm3", "ne")].to_numpy(float)
    rho_g_cm3 = ne * MU_E * MP_G
    rho_msun_kpc3 = rho_g_cm3 * (KPC_CM**3) / MSUN_G
    return r, rho_msun_kpc3

def load_stars(path_dir: Path):
    p = path_dir/"stars_profile.csv"
    df = pd.read_csv(p)
    r = df[_col(df, "r_kpc", "R_kpc", "r")].to_numpy(float)
    rho = df[_col(df, "rho_stars_Msun_per_kpc3", "rho_Msun_per_kpc3")].to_numpy(float)
    return r, rho

# ---- physics helpers ----
def enclosed_mass_from_rho(r_kpc, rho_msun_kpc3):
    r = np.asarray(r_kpc, float); rho = np.asarray(rho_msun_kpc3, float)
    integrand = 4.0*np.pi*(r**2)*rho
    return cumtrapz(integrand, r, initial=0.0)

def smooth_log_derivative(x, y, s=0.0, k=3):
    mask = np.isfinite(x) & np.isfinite(y) & (x>0) & (y>0)
    xs, ys = x[mask], y[mask]
    if xs.size < (k+2):
        # fallback finite diff
        xs = xs; d = np.gradient(np.log(ys), xs)
        return xs, d
    spl = UnivariateSpline(xs, np.log(ys), s=s, k=k)
    return xs, spl.derivative(1)(xs)

def kT_keV_from_g_and_dlnrho(g_tot, dlnrho_dr, mu=MU):
    g = np.asarray(g_tot, float)          # (km/s)^2/kpc
    slope = np.asarray(dlnrho_dr, float)  # 1/kpc
    kT_kms2 = - g / np.maximum(slope, 1e-12)
    kT_J = (mu * 1.67262192369e-27) * (kT_kms2 * 1e6)
    return kT_J * KEV_PER_J

# ---- runner ----
def run_cluster(cluster_dir, out_dir, logtail_params, r_min=1.0, r_max=3000.0, n_r=500):
    cdir = Path(cluster_dir); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    gas_df = load_gas(cdir)
    r_g, rho_g = gas_density_msun_kpc3(gas_df)
    r_s, rho_s = load_stars(cdir)

    r_lo = max(r_min, float(np.nanmin([np.nanmin(r_g), np.nanmin(r_s)])))
    r_hi = min(r_max, float(np.nanmax([np.nanmax(r_g), np.nanmax(r_s)])))
    r = np.geomspace(max(1.0, r_lo), max(r_lo+1.0, r_hi), n_r)

    rho_gi = interp1d(r_g, rho_g, kind="linear", bounds_error=False, fill_value="extrapolate")(r)
    rho_si = interp1d(r_s, rho_s, kind="linear", bounds_error=False, fill_value=0.0)(r)
    rho_b = np.clip(rho_gi + rho_si, 0.0, None)

    M_b = enclosed_mass_from_rho(r, rho_b)
    g_tot = total_accel(r, M_b, logtail_params)
    M_pred = g_tot * (r**2) / G

    # hydrostatic temp prediction
    rg, dlnrho = smooth_log_derivative(r, rho_gi, s=0.0, k=3)
    g_rg = interp1d(r, g_tot, kind="linear", bounds_error=False, fill_value="extrapolate")(rg)
    kT_pred = kT_keV_from_g_and_dlnrho(g_rg, dlnrho)

    # load optional obs
    temp_p = cdir/"temp_profile.csv"; T_obs=None
    if temp_p.exists():
        tdf = pd.read_csv(temp_p)
        rT = tdf[_col(tdf,"r_kpc","R_kpc","r")].to_numpy(float)
        kT = tdf[_col(tdf,"kT_keV","T_keV","kT")].to_numpy(float)
        kTerr = tdf[tdf.columns[2]].to_numpy(float) if len(tdf.columns)>=3 else None
        T_obs = dict(r=rT, kT=kT, kTerr=kTerr)

    lens_p = cdir/"lensing_mass.csv"; L_obs=None
    if lens_p.exists():
        ldf = pd.read_csv(lens_p)
        rL = ldf[_col(ldf,"r_kpc","R_kpc","r")].to_numpy(float)
        ML = ldf[_col(ldf,"M_enclosed_Msun","M_Msun","M")].to_numpy(float)
        L_obs = dict(r=rL, M=ML)

    # plots
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.loglog(r, M_b, label="Baryonic mass (gas+stars)")
    plt.loglog(r, M_pred, "--", label="Total mass (LogTail)")
    if L_obs is not None:
        plt.errorbar(L_obs["r"], L_obs["M"], fmt="o", ms=3, label="Lensing mass")
    plt.xlabel("r [kpc]"); plt.ylabel("M(<r) [Msun]"); plt.title("Enclosed mass"); plt.grid(True, which="both", alpha=0.25); plt.legend()

    plt.subplot(1,2,2)
    plt.semilogx(rg, kT_pred, "--", label="kT (LogTail hydrostatic)")
    if T_obs is not None:
        if T_obs["kTerr"] is not None:
            plt.errorbar(T_obs["r"], T_obs["kT"], yerr=T_obs["kTerr"], fmt="o", ms=3, label="kT (obs)")
        else:
            plt.plot(T_obs["r"], T_obs["kT"], "o", ms=3, label="kT (obs)")
    plt.xlabel("r [kpc]"); plt.ylabel("kT [keV]"); plt.title("Gas temperature"); plt.grid(True, which="both", alpha=0.25); plt.legend()

    png = out/"cluster_logtail_results.png"
    plt.tight_layout(); plt.savefig(png, dpi=160); plt.close()

    metrics = {"logtail_params": logtail_params, "files": {
        "gas": str(gas_df.columns.tolist()),
        "stars_file": str((Path(cluster_dir)/"stars_profile.csv").resolve()),
        "temp_file": str((temp_p.resolve())) if temp_p.exists() else None,
        "lens_mass_file": str((lens_p.resolve())) if lens_p.exists() else None
    }, "grid": {"r_min_kpc": float(r.min()), "r_max_kpc": float(r.max()), "n_r": int(n_r)}, "png": str(png)}

    if T_obs is not None:
        kT_pred_on_obs = interp1d(rg, kT_pred, bounds_error=False, fill_value="extrapolate")(T_obs["r"])
        if T_obs["kTerr"] is not None:
            chi2 = np.sum(((kT_pred_on_obs - T_obs["kT"])/np.maximum(T_obs["kTerr"],1e-6))**2)
            metrics["temp_chi2"] = float(chi2)
        frac = np.median(np.abs(kT_pred_on_obs - T_obs["kT"]) / np.maximum(T_obs["kT"], 1e-6))
        metrics["temp_median_frac_err"] = float(frac)
    if L_obs is not None:
        M_pred_on_obs = interp1d(r, M_pred, bounds_error=False, fill_value="extrapolate")(L_obs["r"])
        fracM = np.median(np.abs(M_pred_on_obs - L_obs["M"]) / np.maximum(L_obs["M"], 1e-6))
        metrics["mass_median_frac_err"] = float(fracM)

    (Path(out_dir)/"cluster_logtail_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster_dir", required=True, help="data/clusters/<NAME>")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--logtail", default="v0=140,rc=15,r0=3,delta=4")
    ap.add_argument("--r_min", type=float, default=1.0)
    ap.add_argument("--r_max", type=float, default=3000.0)
    ap.add_argument("--n_r", type=int, default=500)
    args = ap.parse_args()

    # parse logtail params
    p = {}
    for kv in args.logtail.split(","):
        k,v = kv.split("="); k=k.strip(); v=float(v)
        if k=="v0": p["v0"]=v
        elif k=="rc": p["rc"]=v
        elif k=="r0": p["r0"]=v
        elif k=="delta": p["delta"]=v
    if not p: p=dict(v0=140.0, rc=15.0, r0=3.0, delta=4.0)

    run_cluster(args.cluster_dir, args.out_dir, p, r_min=args.r_min, r_max=args.r_max, n_r=args.n_r)
