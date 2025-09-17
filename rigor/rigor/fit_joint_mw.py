
from __future__ import annotations
import os, argparse, json
import numpy as np
import pandas as pd

from .data import load_sparc, Dataset
from .mw import bin_mw_stars
from .inference_numpyro import fit_hierarchical
from .plotting import overlay_with_posterior

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/sparc_rotmod_ltg.parquet")
    ap.add_argument("--master", default="data/Rotmod_LTG/MasterSheet_SPARC.csv")
    ap.add_argument("--mw_csv", required=True, help="CSV of Gaia stars with columns R_kpc and Vphi_kms (or binned curve)")
    ap.add_argument("--mw_is_binned", action="store_true", help="Set if the CSV already contains binned curve columns R_kpc,V_kms,eV_kms")
    ap.add_argument("--outer", choices=["sigma","kRd","slope"], default="sigma")
    ap.add_argument("--sigma_th", type=float, default=10.0)
    ap.add_argument("--k_Rd", type=float, default=3.0)
    ap.add_argument("--xi", default="shell_logistic_radius")
    ap.add_argument("--outdir", default="out/joint_mw_sparc")
    ap.add_argument("--platform", choices=["gpu","cpu","mps"], default="gpu")
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=1200)
    ap.add_argument("--chains", type=int, default=4)
    args = ap.parse_args()

    ds = load_sparc(args.parquet, args.master, outer_method=args.outer, sigma_th=args.sigma_th, k_Rd=args.k_Rd)
    # Load MW
    df = pd.read_csv(args.mw_csv)
    if args.mw_is_binned:
        from .data import GalaxyData
        g = GalaxyData(
            name="Milky Way (binned)",
            R_kpc=df["R_kpc"].to_numpy(float),
            Vobs_kms=df["V_kms"].to_numpy(float),
            eVobs_kms=df["eV_kms"].to_numpy(float) if "eV_kms" in df else np.full(len(df), 5.0),
            Vbar_kms=np.full(len(df), np.nan),  # add MW baryon model later if desired
            Sigma_bar=None,
            Rd_kpc=None, Mbar_Msun=None,
            outer_mask=(df["R_kpc"]>=df["R_kpc"].median()).to_numpy(bool),
            meta={"source": "MW_binned_CSV"}
        )
    else:
        g = bin_mw_stars(df)
    # Build joint dataset
    joint = Dataset(galaxies = ds.galaxies + [g], meta={**ds.meta, "joint_with_mw": True})
    post, names = fit_hierarchical(joint, xi_name=args.xi, rng_key=args.rng, use_outer_only=True,
                                   num_samples=args.samples, num_warmup=args.warmup, num_chains=args.chains,
                                   platform=args.platform, save_dir=args.outdir)
    # Save overlays including MW (last entry)
    import os
    os.makedirs(os.path.join(args.outdir, "figs"), exist_ok=True)
    for idx, gal in enumerate(joint.galaxies[:6] + [joint.galaxies[-1]]):
        overlay_with_posterior(gal, post, xi_name=args.xi, out_path=os.path.join(args.outdir, "figs", f"{gal.name}.png"), gal_index=idx)
    print("Done. Joint results in", args.outdir)

if __name__ == "__main__":
    main()
