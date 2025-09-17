
from __future__ import annotations
import os, json, numpy as np
from typing import Dict, List
from .data import load_sparc
from .baselines import gr_baseline, mond_simple, burkert_velocity_kms
from .metrics import summarize_outer

def optimize_global_mond_a0(dataset, use_outer=True):
    # Grid search over a0 in (0.5e-10 .. 3e-10)
    grid = np.linspace(0.5e-10, 3.0e-10, 40)
    best = None
    best_score = np.inf
    for a0 in grid:
        total = 0.0; denom=0.0
        for g in dataset.galaxies:
            mask = g.outer_mask if use_outer else np.isfinite(g.R_kpc)
            Vpred = mond_simple(g.Vbar_kms[mask], g.R_kpc[mask], a0=a0)
            off = np.abs(g.Vobs_kms[mask]-Vpred)
            w = 1.0/np.maximum(g.eVobs_kms[mask], 1e-3)**2
            total += np.sum(w*off**2); denom += np.sum(w)
        score = total/np.maximum(denom,1e-12)
        if score < best_score:
            best_score = score; best = a0
    return float(best)

def burkert_fit_galaxy(g, use_outer=True):
    mask = g.outer_mask if use_outer else np.isfinite(g.R_kpc)
    R = g.R_kpc[mask]; Vobs = g.Vobs_kms[mask]; eV = g.eVobs_kms[mask]; Vbar = g.Vbar_kms[mask]
    if len(R) < 6: return None
    # Coarse grid over rho0 (Msun/kpc^3) and r0 (kpc)
    r0_grid = np.geomspace(0.5, 30.0, 30)
    rho0_grid= np.geomspace(1e6, 1e10, 30)
    best=None; best_sse=np.inf
    for r0 in r0_grid:
        Vh = np.array([burkert_velocity_kms(r, rho0_grid[:,None], r0) for r in R])  # grid over rho0
        # Vpred^2 = Vbar^2 + Vh^2; choose rho0 that minimizes weighted SSE for each r0
        for rho0 in rho0_grid:
            Vh = np.array([burkert_velocity_kms(r, rho0, r0) for r in R])
            Vpred = np.sqrt(np.maximum(Vbar**2 + Vh**2, 0.0))
            w = 1.0/np.maximum(eV, 1e-3)**2
            sse = np.sum(w*(Vobs - Vpred)**2)
            if sse < best_sse:
                best_sse = sse; best=(rho0, r0)
    return best

def compare_baselines(parquet="data/sparc_rotmod_ltg.parquet", master="data/Rotmod_LTG/MasterSheet_SPARC.csv",
                      out_json="out/baselines_summary.json", use_outer=True):
    ds = load_sparc(parquet, master)
    # GR
    gr_stats = []
    for g in ds.galaxies:
        mask = g.outer_mask if use_outer else np.isfinite(g.R_kpc)
        stats = summarize_outer(g.Vobs_kms, g.Vbar_kms, g.eVobs_kms, mask)
        gr_stats.append(stats)
    # MOND (global a0)
    a0_hat = optimize_global_mond_a0(ds, use_outer=use_outer)
    mond_stats = []
    for g in ds.galaxies:
        mask = g.outer_mask if use_outer else np.isfinite(g.R_kpc)
        Vpred = mond_simple(g.Vbar_kms[mask], g.R_kpc[mask], a0=a0_hat)
        stats = summarize_outer(g.Vobs_kms, np.where(mask, Vpred, g.Vbar_kms), g.eVobs_kms, mask)  # compute only on mask
        mond_stats.append(stats)
    # Burkert per galaxy
    burkert_stats = []
    for g in ds.galaxies[:20]:  # limit for speed
        fit = burkert_fit_galaxy(g, use_outer=use_outer)
        if fit is None:
            continue
        rho0, r0 = fit
        mask = g.outer_mask if use_outer else np.isfinite(g.R_kpc)
        Vh = np.array([burkert_velocity_kms(r, rho0, r0) for r in g.R_kpc[mask]])
        Vpred = np.sqrt(np.maximum(g.Vbar_kms[mask]**2 + Vh**2, 0.0))
        stats = summarize_outer(g.Vobs_kms, np.where(mask, Vpred, g.Vbar_kms), g.eVobs_kms, mask)
        burkert_stats.append(stats)
    out = {
        "N_galaxies": ds.meta["N_galaxies"],
        "GR": {"mean_off": float(np.mean([s["mean_off"] for s in gr_stats]))},
        "MOND_simple": {"a0_hat": a0_hat, "mean_off": float(np.mean([s["mean_off"] for s in mond_stats]))},
        "Burkert_subset": {"N": len(burkert_stats), "mean_off": float(np.mean([s["mean_off"] for s in burkert_stats]))},
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    return out
