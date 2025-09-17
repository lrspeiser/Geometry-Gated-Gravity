#!/usr/bin/env python3
"""
Auto-experiments to search for a good xi formula and gating strategy across SPARC and MW.

This script will:
- Load SPARC (and optionally MW binned curve) via rigor.data
- For each xi variant and gating mode, run a short NumPyro fit (CPU-friendly defaults)
- Score on outer regions and write a leaderboard CSV
- Optionally run PySR symbolic regression on outer-region xi targets and evaluate top formulas

Usage examples:
  # SPARC only, quick sweep
  python -m rigor.auto_search --outdir out/auto --platform cpu --samples 200 --warmup 200 --chains 1

  # Joint with MW binned curve
  python -m rigor.auto_search --mw_csv data/gaia_mw_binned.csv --outdir out/auto_mw

Notes:
- Requires `pip install pysr` for the symbolic regression step (optional).
- See README for installing JAX (CPU/GPU) and NumPyro.
"""
from __future__ import annotations
import os, argparse, json, time
import numpy as np
import pandas as pd

from .data import load_sparc, Dataset
from .inference_numpyro import fit_hierarchical
from .plotting import overlay_with_posterior
from .metrics import summarize_outer
from .xi import shell_logistic_radius, logistic_density
from jax.nn import sigmoid as j_sigmoid
import numpy as _np

XI_CHOICES = [
    "shell_logistic_radius",
    "logistic_density",
    # You can add combined variants by wrapping in xi.py and listing here
]

GATING_CHOICES = ["none", "fixed", "learned"]


def _post_median(arr):
    # Accepts array-like possibly with shape (samples, ...) and returns median over axis 0
    a = _np.asarray(arr)
    if a.ndim == 0:
        return float(a)
    if a.ndim == 1:
        return float(_np.median(a))
    return _np.median(a, axis=0)


def _predict_median(ds: Dataset, post: dict, xi_name: str, gating: str, gate_R_kpc: float, gate_width: float):
    # Build deterministic median prediction per galaxy from posterior
    # Global params
    xi_max = _post_median(post.get("xi_max"))
    lnR0_base = _post_median(post.get("lnR0_base"))
    width = _post_median(post.get("width"))
    alpha_M = _post_median(post.get("alpha_M"))
    lnSigma_c = _post_median(post.get("lnSigma_c"))
    width_sigma = _post_median(post.get("width_sigma"))
    n_sigma = _post_median(post.get("n_sigma"))
    params = {"xi_max": xi_max, "lnR0_base": lnR0_base, "width": width, "alpha_M": alpha_M,
              "lnSigma_c": lnSigma_c, "width_sigma": width_sigma, "n_sigma": n_sigma, "Mref": 1e10}

    # Gating params (learned)
    lnR_gate_base = _post_median(post.get("lnR_gate_base")) if (gating == "learned" and "lnR_gate_base" in post) else None
    width_gate = _post_median(post.get("width_gate")) if (gating == "learned" and "width_gate" in post) else None
    alpha_gate_M = _post_median(post.get("alpha_gate_M")) if (gating == "learned" and "alpha_gate_M" in post) else None
    dlnR_gate_g = _post_median(post.get("dlnR_gate_g")) if (gating == "learned" and "dlnR_gate_g" in post) else None

    # Per-galaxy nuisance (Vbar mass-to-light tilt)
    dlogML_g = _post_median(post.get("dlogML_g")) if ("dlogML_g" in post) else 0.0

    preds = []
    for gi, g in enumerate(ds.galaxies):
        R = _np.asarray(g.R_kpc)
        Vbar = _np.asarray(g.Vbar_kms)
        # Adjust Vbar using per-galaxy dlogML if available
        if _np.ndim(dlogML_g) == 0:
            Vbar_adj = Vbar * _np.exp(0.5 * float(dlogML_g))
        else:
            Vbar_adj = Vbar * _np.exp(0.5 * float(_np.asarray(dlogML_g)[gi]))

        # Compute xi according to selected registry
        if xi_name == "shell_logistic_radius":
            xi_r = _np.asarray(shell_logistic_radius(R, None, g.Mbar_Msun or 1e10, params))
            xi = xi_r
        elif xi_name == "logistic_density":
            xi = _np.asarray(logistic_density(R, g.Sigma_bar, g.Mbar_Msun or 1e10, params))
        else:
            # Default to radial logistic if unknown
            xi = _np.asarray(shell_logistic_radius(R, None, g.Mbar_Msun or 1e10, params))

        # Apply gating
        if gating == "fixed":
            lnR = _np.log(_np.maximum(R, 1e-6))
            lnR_gate = _np.log(max(gate_R_kpc, 1e-6))
            H = 1.0/(1.0 + _np.exp(-(lnR - lnR_gate)/max(gate_width, 1e-3)))
            xi = 1.0 + H * (xi - 1.0)
        elif gating == "learned" and (lnR_gate_base is not None):
            lnR = _np.log(_np.maximum(R, 1e-6))
            mass_term = _np.log(max((g.Mbar_Msun or 1e10)/1e10, 1e-12)) if (g.Mbar_Msun and _np.isfinite(g.Mbar_Msun)) else 0.0
            lnR_gate_gi = lnR_gate_base + (alpha_gate_M or 0.0)*mass_term
            if dlnR_gate_g is not None and _np.ndim(dlnR_gate_g) == 1 and gi < len(dlnR_gate_g):
                lnR_gate_gi += float(dlnR_gate_g[gi])
            w = max(width_gate or 0.4, 1e-3)
            H = 1.0/(1.0 + _np.exp(-(lnR - lnR_gate_gi)/w))
            xi = 1.0 + H * (xi - 1.0)

        Vpred = Vbar_adj * _np.sqrt(_np.clip(xi, 1.0, 100.0))
        preds.append((g, Vpred))
    return preds


def _score_predictions(preds):
    # Aggregate outer metrics across all galaxies
    summary = {"galaxies": 0, "outer_points": 0, "sum_mean_off": 0.0}
    details = []
    for g, Vpred in preds:
        mask = g.outer_mask
        if not _np.any(mask):
            continue
        res = summarize_outer(g.Vobs_kms, Vpred, g.eVobs_kms, mask)
        summary["galaxies"] += 1
        summary["outer_points"] += int(_np.sum(mask))
        summary["sum_mean_off"] += float(res["mean_off"])  # avg per galaxy
        details.append({"galaxy": g.name, **res})
    if summary["galaxies"]:
        summary["avg_mean_off"] = summary["sum_mean_off"]/summary["galaxies"]
    else:
        summary["avg_mean_off"] = _np.inf
    return summary, details


def run_variant(ds: Dataset, xi: str, gating: str, outdir: str, platform: str,
                samples: int, warmup: int, chains: int, gate_R_kpc: float, gate_width: float,
                seed: int = 0):
    subdir = os.path.join(outdir, f"xi_{xi}__gate_{gating}")
    os.makedirs(subdir, exist_ok=True)
    t0 = time.time()
    post, names = fit_hierarchical(ds, xi_name=xi, rng_key=seed, use_outer_only=True,
                                   num_samples=samples, num_warmup=warmup, num_chains=chains,
                                   platform=platform, save_dir=subdir,
                                   gating_mode=gating, gate_R_kpc=gate_R_kpc, gate_width=gate_width)
    dt = time.time() - t0
    # Save quick overlays for first few galaxies
    figs = os.path.join(subdir, "figs"); os.makedirs(figs, exist_ok=True)
    for idx, gal in enumerate(ds.galaxies[:6]):
        overlay_with_posterior(gal, post, xi_name=xi, out_path=os.path.join(figs, f"{gal.name}.png"), gal_index=idx,
                               title_suffix=f"xi={xi}, gate={gating}")
    # Compute median predictions and score
    preds = _predict_median(ds, post, xi, gating, gate_R_kpc, gate_width)
    summary, details = _score_predictions(preds)
    with open(os.path.join(subdir, "scores.json"), "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=2)
    return {"xi": xi, "gating": gating, "seconds": dt, **summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/sparc_rotmod_ltg.parquet")
    ap.add_argument("--master", default="data/Rotmod_LTG/MasterSheet_SPARC.csv")
    ap.add_argument("--mw_csv", default=None)
    ap.add_argument("--outer", choices=["sigma","kRd","slope"], default="sigma")
    ap.add_argument("--sigma_th", type=float, default=10.0)
    ap.add_argument("--k_Rd", type=float, default=3.0)
    ap.add_argument("--outdir", default="out/auto_search")
    ap.add_argument("--platform", choices=["gpu","cpu","mps"], default="cpu")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--gate_R_kpc", type=float, default=3.0)
    ap.add_argument("--gate_width", type=float, default=0.4)
    ap.add_argument("--xi", nargs="*", default=XI_CHOICES)
    ap.add_argument("--gating", nargs="*", default=GATING_CHOICES)
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()

    ds = load_sparc(args.parquet, args.master, outer_method=args.outer, sigma_th=args.sigma_th, k_Rd=args.k_Rd)
    os.makedirs(args.outdir, exist_ok=True)

    leaderboard = []
    for xi in args.xi:
        for g in args.gating:
            try:
                rec = run_variant(ds, xi, g, args.outdir, args.platform,
                                  args.samples, args.warmup, args.chains,
                                  args.gate_R_kpc, args.gate_width, seed=args.rng)
                leaderboard.append(rec)
            except Exception as e:
                leaderboard.append({"xi": xi, "gating": g, "error": str(e)})
    # Rank by avg_mean_off ascending
    leaderboard_sorted = sorted(leaderboard, key=lambda r: r.get("avg_mean_off", float("inf")))
    lb_path = os.path.join(args.outdir, "leaderboard.json")
    with open(lb_path, "w") as f:
        json.dump(leaderboard_sorted, f, indent=2)
    # Also write CSV
    import csv
    csv_path = os.path.join(args.outdir, "leaderboard.csv")
    cols = ["xi","gating","avg_mean_off","outer_points","galaxies","seconds"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in leaderboard_sorted:
            w.writerow({c: row.get(c, "") for c in cols})
    print("Wrote leaderboard to", lb_path, "and", csv_path)

if __name__ == "__main__":
    main()
