
from __future__ import annotations
import os, json, numpy as np, argparse
import matplotlib.pyplot as plt
from .xi import shell_logistic_radius, logistic_density
from .data import load_sparc

def overlay_with_posterior(gal, posterior, xi_name="shell_logistic_radius", out_path=None, gal_index=None):
    post = {k: np.asarray(v) for k,v in posterior.items()}
    nsamp_all = len(post[list(post.keys())[0]])
    nsamp = min(400, nsamp_all)
    idx = np.random.choice(np.arange(nsamp_all), size=nsamp, replace=False)

    Vpred_samps = []
    for i in idx:
        params = {k: (float(post[k][i]) if k in post and np.ndim(post[k])==1 else None)
                  for k in ["xi_max","lnR0_base","width","alpha_M","lnSigma_c","width_sigma","n_sigma"]}
        dlogML = 0.0
        if "dlogML_g" in post and gal_index is not None:
            dlogML = float(post["dlogML_g"][i, gal_index])
        Vbar_adj = gal.Vbar_kms * np.exp(0.5*dlogML)

        xi_r = shell_logistic_radius(gal.R_kpc, None, gal.Mbar_Msun or 1e10, params)
        if gal.Sigma_bar is not None:
            xi_d = logistic_density(gal.R_kpc, gal.Sigma_bar, gal.Mbar_Msun or 1e10, params)
            xi = np.minimum(xi_r * xi_d, 10.0)
        else:
            xi = xi_r
        Vpred = Vbar_adj * np.sqrt(np.clip(xi, 1.0, 100.0))
        Vpred_samps.append(np.asarray(Vpred))
    Vpred_samps = np.array(Vpred_samps)

    lo = np.percentile(Vpred_samps, 16, axis=0)
    hi = np.percentile(Vpred_samps, 84, axis=0)
    med = np.median(Vpred_samps, axis=0)

    plt.figure(figsize=(8,5))
    plt.scatter(gal.R_kpc[~gal.outer_mask], gal.Vobs_kms[~gal.outer_mask], s=18, alpha=0.6, label="Observed (inner)")
    plt.scatter(gal.R_kpc[gal.outer_mask], gal.Vobs_kms[gal.outer_mask], s=28, alpha=0.9, label="Observed (outer)")
    plt.plot(gal.R_kpc, med, lw=2, label="Model median", color="C3")
    plt.fill_between(gal.R_kpc, lo, hi, alpha=0.2, label="68% band")
    plt.plot(gal.R_kpc, gal.Vbar_kms, lw=2, ls="--", label="GR baseline", color="C2")
    if np.any(gal.outer_mask):
        rb = np.nanmedian(gal.R_kpc[gal.outer_mask])
        plt.axvline(rb, ls=":", color="k", alpha=0.6)
        plt.text(rb, 0.98*max(np.nanmax(gal.Vobs_kms), np.nanmax(med)), "outer boundary", rotation=90, va="top", ha="right", fontsize=8, alpha=0.7)
    plt.xlabel("Radius R (kpc)"); plt.ylabel("Speed (km/s)")
    plt.title(gal.name)
    plt.legend()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", required=True, help="Path to posterior_samples.npz produced by fit_sparc/fit_joint_mw")
    ap.add_argument("--figdir", required=True, help="Directory to save overlays")
    ap.add_argument("--parquet", default="data/sparc_rotmod_ltg.parquet")
    ap.add_argument("--master", default="data/Rotmod_LTG/MasterSheet_SPARC.csv")
    ap.add_argument("--outer", choices=["sigma","kRd","slope"], default="sigma")
    ap.add_argument("--sigma_th", type=float, default=10.0)
    ap.add_argument("--k_Rd", type=float, default=3.0)
    args = ap.parse_args()

    # Load posterior and names (if available)
    post = np.load(args.post)
    names_path = os.path.join(os.path.dirname(args.post), "galaxy_names.json")
    names = None
    if os.path.exists(names_path):
        with open(names_path, "r") as f: names = json.load(f)

    from .data import load_sparc
    ds = load_sparc(args.parquet, args.master, outer_method=args.outer, sigma_th=args.sigma_th, k_Rd=args.k_Rd)

    os.makedirs(args.figdir, exist_ok=True)
    for idx, g in enumerate(ds.galaxies):
        out_path = os.path.join(args.figdir, f"{g.name}.png")
        overlay_with_posterior(g, post, out_path=out_path, gal_index=idx)
    print("Saved overlays to", args.figdir)

if __name__ == "__main__":
    main()
