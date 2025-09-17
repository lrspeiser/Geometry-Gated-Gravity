
"""Command-line driver to:
1) Load SPARC
2) Fit hierarchical model with NumPyro
3) Save posterior and generate overlays for a few galaxies
4) (Optional) Run PySR symbolic search for xi

Usage:
  python -m rigor.fit_sparc --parquet data/sparc_rotmod_ltg.parquet --master data/Rotmod_LTG/MasterSheet_SPARC.csv \

                            --xi shell_logistic_radius --outer sigma --sigma_th 10 \

                            --outdir out/xi_shell_logistic_radius
"""
from __future__ import annotations
import os, argparse, json
import numpy as np

from .data import load_sparc
from .inference_numpyro import fit_hierarchical
from .plotting import overlay_with_posterior

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/sparc_rotmod_ltg.parquet")
    ap.add_argument("--master", default="data/Rotmod_LTG/MasterSheet_SPARC.csv")
    ap.add_argument("--outer", choices=["sigma","kRd","slope"], default="sigma")
    ap.add_argument("--sigma_th", type=float, default=10.0)
    ap.add_argument("--k_Rd", type=float, default=3.0)
    ap.add_argument("--xi", default="shell_logistic_radius")
    ap.add_argument("--use_outer_only", action="store_true")
    ap.add_argument("--outdir", default="out/xi_shell")
    ap.add_argument("--platform", choices=["gpu","cpu","mps"], default="gpu")
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=1200)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()

    ds = load_sparc(args.parquet, args.master, outer_method=args.outer, sigma_th=args.sigma_th, k_Rd=args.k_Rd)
    print(f"Loaded {ds.meta['N_galaxies']} galaxies with outer_method={ds.meta['outer_method']}")

    posterior, names = fit_hierarchical(ds, xi_name=args.xi, rng_key=args.rng, use_outer_only=args.use_outer_only,
                                        num_samples=args.samples, num_warmup=args.warmup, num_chains=args.chains,
                                        platform=("gpu" if args.platform in ["gpu","mps"] else "cpu"), save_dir=args.outdir)
    # Save a couple of overlays
    os.makedirs(os.path.join(args.outdir, "figs"), exist_ok=True)
    for g in ds.galaxies[:6]:
        overlay_with_posterior(g, posterior, xi_name=args.xi,
                               out_path=os.path.join(args.outdir, "figs", f"{g.name}.png"))
    print("Done. Results in", args.outdir)

if __name__ == "__main__":
    main()
