
from __future__ import annotations
import argparse, json, os, numpy as np
import pandas as pd
from .data import load_sparc
from .plotting import overlay_with_posterior

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", required=True, help="posterior_samples.npz path")
    ap.add_argument("--galaxy_names", default=None, help="galaxy_names.json (if present alongside posterior)")
    ap.add_argument("--parquet", default="data/sparc_rotmod_ltg.parquet")
    ap.add_argument("--master",  default="data/Rotmod_LTG/MasterSheet_SPARC.csv")
    ap.add_argument("--outer",   choices=["sigma","kRd","slope"], default="sigma")
    ap.add_argument("--sigma_th", type=float, default=10.0)
    ap.add_argument("--k_Rd", type=float, default=3.0)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--xi", default="shell_logistic_radius")
    ap.add_argument("--limit", type=int, default=12, help="max number of galaxies to plot")
    args = ap.parse_args()

    ds = load_sparc(args.parquet, args.master, outer_method=args.outer, sigma_th=args.sigma_th, k_Rd=args.k_Rd)
    post = dict(np.load(args.post))
    # Try to read galaxy_names.json
    names_path = args.galaxy_names or os.path.join(os.path.dirname(args.post), "galaxy_names.json")
    names = None
    if os.path.exists(names_path):
        with open(names_path, "r") as f:
            names = json.load(f)

    os.makedirs(args.figdir, exist_ok=True)
    count = 0
    for idx, g in enumerate(ds.galaxies):
        if names and g.name not in names:
            continue
        out = os.path.join(args.figdir, f"{g.name}.png")
        overlay_with_posterior(g, post, xi_name=args.xi, out_path=out, gal_index=idx)
        count += 1
        if count >= args.limit:
            break
    print(f"Wrote {count} overlay(s) to {args.figdir}")

if __name__ == "__main__":
    main()
