from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from rigor.rigor.data import load_sparc
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma

OUT_DIR = os.path.join("gravity_learn", "experiments", "sr")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_sparc()
    rows = []
    for g in ds.galaxies:
        R = g.R_kpc
        Vobs = g.Vobs_kms
        Vbar = g.Vbar_kms
        if R is None or Vobs is None or Vbar is None:
            continue
        mask = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
        if not np.any(mask):
            continue
        R = R[mask]
        Vobs = Vobs[mask]
        Vbar = Vbar[mask]
        Sigma = None
        if g.Sigma_bar is not None:
            Sigma = g.Sigma_bar[mask]
        else:
            Sigma = np.maximum(1e-3, np.exp(-R / np.maximum(1.0, np.nanmedian(R))))
        eps = 1e-9
        g_req = (Vobs * Vobs) / (R + eps)
        g_bar = (Vbar * Vbar) / (R + eps)
        gX = g_req - g_bar
        xi_emp = (Vobs / np.maximum(Vbar, 1e-6)) ** 2
        x = dimensionless_radius(R, Rd=(g.Rd_kpc or None))
        Sh = sigma_hat(Sigma)
        dlnS = grad_log_sigma(R, Sigma)
        df = pd.DataFrame({
            "Galaxy": g.name,
            "R_kpc": R,
            "gX": gX,
            "xi_emp": xi_emp,
            "x_dimless": np.asarray(x),
            "Sigma": Sigma,
            "Sigma_hat": np.asarray(Sh),
            "grad_ln_Sigma": np.asarray(dlnS),
        })
        rows.append(df)
    if not rows:
        raise RuntimeError("No SPARC rows to analyze.")
    full = pd.concat(rows, ignore_index=True)
    full = full.replace([np.inf, -np.inf], np.nan).dropna()

    features = ["x_dimless", "Sigma", "Sigma_hat", "grad_ln_Sigma", "R_kpc"]
    targets = ["gX", "xi_emp"]
    corrs = []
    for tgt in targets:
        y = full[tgt].to_numpy()
        for feat in features:
            x = full[feat].to_numpy()
            if _HAVE_SCIPY:
                rho, p = spearmanr(x, y)
                corrs.append({"target": tgt, "feature": feat, "spearman": float(rho), "p_value": float(p)})
            else:
                xm = x - x.mean()
                ym = y - y.mean()
                denom = np.sqrt((xm * xm).sum() * (ym * ym).sum()) + 1e-12
                r = float((xm * ym).sum() / denom)
                corrs.append({"target": tgt, "feature": feat, "pearson": r})

    out_json = os.path.join(OUT_DIR, "correlations.json")
    out_csv = os.path.join(OUT_DIR, "correlations.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(corrs, f, indent=2)
    pd.DataFrame(corrs).to_csv(out_csv, index=False)
    print(f"[correlations] wrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()