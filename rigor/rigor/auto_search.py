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

XI_CHOICES = [
    "shell_logistic_radius",
    "logistic_density",
    # You can add combined variants by wrapping in xi.py and listing here
]

GATING_CHOICES = ["none", "fixed", "learned"]


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
    return {"xi": xi, "gating": gating, "seconds": dt}


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
    lb_path = os.path.join(args.outdir, "leaderboard.json")
    with open(lb_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    print("Wrote leaderboard to", lb_path)

if __name__ == "__main__":
    main()
