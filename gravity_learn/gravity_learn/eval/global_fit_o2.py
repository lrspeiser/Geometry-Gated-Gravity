from __future__ import annotations
import os, argparse, json, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rigor.rigor.data import load_sparc
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma

try:
    import scipy.optimize as opt
except Exception:
    opt = None


def build_dataset(ds):
    items = []
    for g in ds.galaxies:
        R = g.R_kpc; Vobs = g.Vobs_kms; Vbar = g.Vbar_kms
        if R is None or Vobs is None or Vbar is None:
            continue
        mask = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
        R = np.asarray(R)[mask]
        Vobs = np.asarray(Vobs)[mask]
        Vbar = np.asarray(Vbar)[mask]
        if R.size < 6:
            continue
        Sigma = g.Sigma_bar[mask] if (g.Sigma_bar is not None) else np.maximum(1e-3, np.exp(-R / np.maximum(1.0, np.nanmedian(R))))
        x = np.asarray(dimensionless_radius(R, Rd=(g.Rd_kpc or None)))
        Sh = np.asarray(sigma_hat(Sigma))
        dlnS = np.asarray(grad_log_sigma(R, Sigma))
        items.append({
            "name": g.name,
            "R": R, "Vobs": Vobs, "Vbar": Vbar,
            "x": x, "Sh": Sh, "dlnS": dlnS,
        })
    return items


def fX_ratio(params, x, Sh, dlnS):
    a, b = params
    denom = (a - b * Sh)
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
    return np.maximum(0.0, (x * x) / denom)


def fX_exp(params, x, Sh, dlnS):
    alpha, c = params
    alpha = np.maximum(alpha, 0.0)
    return np.maximum(0.0, alpha * (x * x) * (np.exp(Sh) + c))


FAMILIES = {
    "ratio": {"func": fX_ratio, "x0": np.array([0.5, 0.2]), "bounds": [(-1.0, 2.0), (0.0, 2.0)]},
    "exp":   {"func": fX_exp,   "x0": np.array([1.0, 0.5]),  "bounds": [(0.0, 10.0), (-0.9, 5.0)]},
}


def loss_global(params, family_key, dataset):
    f = FAMILIES[family_key]["func"]
    err = 0.0; npts = 0
    for it in dataset:
        Vbar = it["Vbar"]; Vobs = it["Vobs"]
        fX = f(params, it["x"], it["Sh"], it["dlnS"])  # shape (N,)
        Vmod = Vbar * np.sqrt(np.maximum(0.0, 1.0 + fX))
        diff = Vmod - Vobs
        err += float(np.sum(diff * diff))
        npts += Vobs.size
    return err / max(npts, 1)


def fit_family_global(family_key, dataset):
    x0 = FAMILIES[family_key]["x0"]
    bounds = FAMILIES[family_key]["bounds"]
    if opt is None:
        # coarse grid fallback
        if family_key == "ratio":
            a_grid = np.linspace(-0.5, 1.5, 41)
            b_grid = np.linspace(0.0, 1.0, 41)
            best = None
            for a in a_grid:
                for b in b_grid:
                    L = loss_global([a, b], family_key, dataset)
                    if (best is None) or (L < best[0]):
                        best = (L, a, b)
            return {"params": [float(best[1]), float(best[2])], "loss": float(best[0])}
        else:
            alpha_grid = np.linspace(0.0, 3.0, 31)
            c_grid = np.linspace(-0.5, 2.0, 26)
            best = None
            for a in alpha_grid:
                for c in c_grid:
                    L = loss_global([a, c], family_key, dataset)
                    if (best is None) or (L < best[0]):
                        best = (L, a, c)
            return {"params": [float(best[1]), float(best[2])], "loss": float(best[0])}
    else:
        res = opt.minimize(lambda p: loss_global(p, family_key, dataset), x0=x0, bounds=bounds, method="L-BFGS-B")
        return {"params": [float(v) for v in res.x], "loss": float(res.fun)}


def evaluate_metrics(family_key, params, dataset):
    f = FAMILIES[family_key]["func"]
    rows = []
    for it in dataset:
        name = it["name"]; R = it["R"]; Vobs = it["Vobs"]; Vbar = it["Vbar"]
        fX = f(params, it["x"], it["Sh"], it["dlnS"])  # (N,)
        Vmod = Vbar * np.sqrt(np.maximum(0.0, 1.0 + fX))
        rmse = float(np.sqrt(np.mean((Vmod - Vobs) ** 2)))
        mape = float(np.median(np.abs((Vmod - Vobs) / np.maximum(np.abs(Vobs), 1e-9))))
        rows.append({"Galaxy": name, "rmse": rmse, "median_ape": mape, "n_points": int(len(R))})
    df = pd.DataFrame(rows)
    summary = {
        "rmse_median": float(df["rmse"].median()),
        "rmse_iqr": [float(df["rmse"].quantile(0.25)), float(df["rmse"].quantile(0.75))],
        "mape_median": float(df["median_ape"].median()),
        "mape_iqr": [float(df["median_ape"].quantile(0.25)), float(df["median_ape"].quantile(0.75))],
    }
    return df, summary


def save_montage(family_key, params, dataset, out_png, limit=16):
    f = FAMILIES[family_key]["func"]
    N = min(limit, len(dataset))
    ncols = 4
    nrows = int(np.ceil(N / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for i in range(N):
        it = dataset[i]
        ax = axes[i]
        R = it["R"]; Vobs = it["Vobs"]; Vbar = it["Vbar"]
        fX = f(params, it["x"], it["Sh"], it["dlnS"])  # (N,)
        Vmod = Vbar * np.sqrt(np.maximum(0.0, 1.0 + fX))
        ax.plot(R, Vobs, 'k.', label='Observed')
        ax.plot(R, Vbar, color='#1f77b4', alpha=0.8, label='Baryons')
        ax.plot(R, Vmod, color='#d62728', alpha=0.9, label='Model')
        ax.set_title(f"{it['name']} [{family_key}]")
        ax.set_xlabel('R [kpc]'); ax.set_ylabel('V [km/s]'); ax.grid(True, alpha=0.3)
    for j in range(N, len(axes)):
        fig.delaxes(axes[j])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f"Global-fit O2: {family_key}")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_galaxies", type=int, default=-1, help="-1 to use all galaxies")
    ap.add_argument("--montage_limit", type=int, default=16)
    ap.add_argument("--outdir", type=str, default=os.path.join("gravity_learn", "experiments", "eval", "global_fit"))
    args = ap.parse_args()

    ds = load_sparc()
    data_all = build_dataset(ds)
    if args.limit_galaxies > 0:
        data_all = data_all[:args.limit_galaxies]

    os.makedirs(args.outdir, exist_ok=True)

    results = {}
    for fam in ("ratio", "exp"):
        fit = fit_family_global(fam, data_all)
        df, summary = evaluate_metrics(fam, fit["params"], data_all)
        results[fam] = {"params": fit["params"], "loss": fit["loss"], "summary": summary}
        # Save metrics and montage
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(os.path.join(args.outdir, f"per_galaxy_metrics_{fam}_{ts}.csv"), index=False)
        with open(os.path.join(args.outdir, f"summary_{fam}_{ts}.json"), "w", encoding="utf-8") as f:
            json.dump({"family": fam, **results[fam]}, f, indent=2)
        save_montage(fam, fit["params"], data_all, os.path.join(args.outdir, f"montage_{fam}_{ts}.png"), limit=args.montage_limit)

    # choose best family by loss
    best_fam = min(results.keys(), key=lambda k: results[k]["loss"])
    with open(os.path.join(args.outdir, "best_family.json"), "w", encoding="utf-8") as f:
        json.dump({"best_family": best_fam, **results[best_fam]}, f, indent=2)
    print(f"[global-fit] Best: {best_fam} with params {results[best_fam]['params']} and loss {results[best_fam]['loss']:.4f}")


if __name__ == "__main__":
    main()