
from __future__ import annotations
import os, json, copy, numpy as np
from typing import List
from .data import load_sparc, Dataset
from .inference_numpyro import fit_hierarchical
from .metrics import summarize_outer

def leave_one_out_cv(parquet="data/sparc_rotmod_ltg.parquet", master="data/Rotmod_LTG/MasterSheet_SPARC.csv",
                     xi_name="shell_logistic_radius", outer_method="sigma", sigma_th=10.0,
                     k_Rd=3.0, platform="gpu", max_gal=None, out_json="out/loo_summary.json"):
    ds_full = load_sparc(parquet, master, outer_method=outer_method, sigma_th=sigma_th, k_Rd=k_Rd)
    galaxies = ds_full.galaxies[:max_gal] if max_gal else ds_full.galaxies
    results = []
    for i, g_hold in enumerate(galaxies):
        # training set = all except i
        train = Dataset(galaxies=[g for j,g in enumerate(galaxies) if j!=i], meta=ds_full.meta)
        post, names = fit_hierarchical(train, xi_name=xi_name, use_outer_only=True,
                                       num_warmup=800, num_samples=800, num_chains=2,
                                       platform=platform, save_dir=f"out/loo_{i:03d}")
        # Evaluate on held-out galaxy using posterior medians
        from .xi import shell_logistic_radius, logistic_density
        g = g_hold
        import numpy as np
        med = {k: float(np.median(np.asarray(v))) for k,v in post.items() if k in ["xi_max","lnR0_base","width","alpha_M","lnSigma_c","width_sigma","n_sigma","sigma_ML"]}
        Vbar_adj = g.Vbar_kms  # ignore M/L for held-out estimate
        xi_r = shell_logistic_radius(g.R_kpc, None, g.Mbar_Msun or 1e10, med)
        if g.Sigma_bar is not None:
            xi_d = logistic_density(g.R_kpc, g.Sigma_bar, g.Mbar_Msun or 1e10, med)
            xi = np.minimum(xi_r*xi_d, 10.0)
        else:
            xi = xi_r
        Vpred = Vbar_adj * np.sqrt(np.clip(xi, 1.0, 100.0))
        stats = summarize_outer(g.Vobs_kms, Vpred, g.eVobs_kms, g.outer_mask)
        results.append({"galaxy": g.name, **stats})
        print(f"[{i+1}/{len(galaxies)}] {g.name}: mean_off={stats['mean_off']:.2f}%")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    import json
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    return results
