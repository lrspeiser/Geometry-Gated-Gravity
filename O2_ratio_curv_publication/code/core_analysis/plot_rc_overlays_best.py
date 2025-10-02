from __future__ import annotations
import os, argparse, json, datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from rigor.rigor.data import load_sparc
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma


def fX_from_family(family: str, params, x, Sh, dlnS):
    if family == "ratio":
        a, b = params
        denom = (a - b * Sh)
        denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
        return np.maximum(0.0, (x * x) / denom)
    elif family == "exp":
        alpha, c = params
        alpha = max(alpha, 0.0)
        return np.maximum(0.0, alpha * (x * x) * (np.exp(Sh) + c))
    elif family == "ratio_curv":
        a, b, d = params
        denom = (a - b * Sh - d * np.abs(dlnS))
        denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
        return np.maximum(0.0, (x * x) / denom)
    elif family == "exp_curv":
        alpha, c, d = params
        alpha = max(alpha, 0.0)
        return np.maximum(0.0, alpha * (x * x) * (np.exp(Sh) + c + d * np.abs(dlnS)))
    else:
        # fallback: linear
        k = params[0] if params else 2.56
        return np.maximum(0.0, k * x)


def load_best(path_json: str):
    with open(path_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    fam = d.get("best_family", "ratio")
    params = d.get("params", [])
    return fam, params


def plot_grid(ds, fam: str, params, out_png: str, limit: int = 16):
    galaxies = ds.galaxies[:limit]
    n = len(galaxies)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    rows = []
    for i, g in enumerate(galaxies):
        ax = axes[i]
        R = g.R_kpc; Vobs = g.Vobs_kms; Vbar = g.Vbar_kms
        if R is None or Vobs is None or Vbar is None:
            continue
        mask = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
        R = np.asarray(R)[mask]; Vobs = np.asarray(Vobs)[mask]; Vbar = np.asarray(Vbar)[mask]
        if R.size < 4:
            continue
        Sigma = g.Sigma_bar[mask] if (g.Sigma_bar is not None) else np.maximum(1e-3, np.exp(-R/np.maximum(1.0, np.nanmedian(R))))
        x = np.asarray(dimensionless_radius(R, Rd=(g.Rd_kpc or None)))
        Sh = np.asarray(sigma_hat(Sigma))
        dlnS = np.asarray(grad_log_sigma(R, Sigma))
        fX = fX_from_family(fam, params, x, Sh, dlnS)
        Vmod = Vbar * np.sqrt(np.maximum(0.0, 1.0 + fX))
        ax.plot(R, Vobs, 'k.', label='Observed')
        ax.plot(R, Vbar, color='#1f77b4', alpha=0.8, label='Baryons')
        ax.plot(R, Vmod, color='#d62728', alpha=0.9, label='Model')
        ax.set_title(f"{g.name} [{fam}]")
        ax.set_xlabel('R [kpc]'); ax.set_ylabel('V [km/s]'); ax.grid(True, alpha=0.3)
        rmse = float(np.sqrt(np.mean((Vmod - Vobs) ** 2)))
        mape = float(np.median(np.abs((Vmod - Vobs) / np.maximum(np.abs(Vobs), 1e-9))))
        rows.append({"Galaxy": g.name, "rmse": rmse, "median_ape": mape, "n_points": int(len(R))})
    for j in range(len(galaxies), len(axes)):
        fig.delaxes(axes[j])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f"Global best ({fam}) overlays")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_json", type=str, default=os.path.join("gravity_learn", "experiments", "eval", "global_fit", "best_family.json"))
    ap.add_argument("--limit_galaxies", type=int, default=16)
    ap.add_argument("--outdir", type=str, default=os.path.join("gravity_learn", "experiments", "eval", "global_fit"))
    args = ap.parse_args()

    fam, params = load_best(args.best_json)
    ds = load_sparc()
    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(args.outdir, f"montage_best_{fam}_{ts}.png")
    df = plot_grid(ds, fam, params, out_png, limit=args.limit_galaxies)
    df.to_csv(os.path.join(args.outdir, f"per_galaxy_metrics_best_{fam}_{ts}.csv"), index=False)
    print(f"[best-overlays] wrote {out_png}")


if __name__ == "__main__":
    main()