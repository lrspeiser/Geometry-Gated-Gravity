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

try:
    from pysr import PySRRegressor
except Exception:
    PySRRegressor = None


def build_features(g):
    R = g.R_kpc; Vobs = g.Vobs_kms; Vbar = g.Vbar_kms
    if R is None or Vobs is None or Vbar is None:
        return None
    mask = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
    R = R[mask]; Vobs = Vobs[mask]; Vbar = Vbar[mask]
    if R.size < 6:
        return None
    Sigma = g.Sigma_bar[mask] if (g.Sigma_bar is not None) else np.maximum(1e-3, np.exp(-R / np.maximum(1.0, np.nanmedian(R))))
    x = np.asarray(dimensionless_radius(R, Rd=(g.Rd_kpc or None)))
    Sh = np.asarray(sigma_hat(Sigma))
    dlnS = np.asarray(grad_log_sigma(R, Sigma))
    eps = 1e-12
    xi_emp = (Vobs / np.maximum(Vbar, 1e-6)) ** 2
    fX_req = np.maximum(0.0, xi_emp - 1.0)
    return {
        "R": R, "Vobs": Vobs, "Vbar": Vbar,
        "x": x, "Sh": Sh, "dlnS": dlnS,
        "fX_req": fX_req,
    }


# Parametric families ---------------------------------------------------------

def fX_simple(params, x, Sh, dlnS):
    k = np.maximum(params[0], 0.0)
    return np.maximum(0.0, k * x)

def fX_ratio(params, x, Sh, dlnS):
    a = params[0]; b = np.maximum(params[1], 0.0)
    denom = (a - b * Sh)
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
    return np.maximum(0.0, (x * x) / denom)

def fX_exp(params, x, Sh, dlnS):
    alpha = np.maximum(params[0], 0.0)
    c = params[1]
    return np.maximum(0.0, alpha * (x * x) * (np.exp(Sh) + c))


FAMILIES = {
    "simple": (fX_simple, [0.5], [(0.0, 10.0)]),
    "ratio":  (fX_ratio,  [0.6, 0.2], [(-1.0, 2.0), (0.0, 2.0)]),
    "exp":    (fX_exp,    [1.0, 0.5], [(0.0, 10.0), (-0.9, 5.0)]),
}


def fit_family(name: str, feats: dict):
    func, x0, bnds = FAMILIES[name]
    R = feats["R"]; Vobs = feats["Vobs"]; Vbar = feats["Vbar"]
    x = feats["x"]; Sh = feats["Sh"]; dlnS = feats["dlnS"]

    def loss(p):
        fX = func(p, x, Sh, dlnS)
        Vmod = Vbar * np.sqrt(np.maximum(0.0, 1.0 + fX))
        return np.mean((Vmod - Vobs) ** 2)

    if opt is None:
        # Fallback grid if SciPy missing (coarse)
        if name == "simple":
            grid = np.linspace(0.0, 6.0, 61)
            errs = [loss([k]) for k in grid]
            k = grid[int(np.argmin(errs))]
            return {"params": [float(k)], "loss": float(np.min(errs))}
        elif name == "ratio":
            a_grid = np.linspace(-0.5, 1.5, 41)
            b_grid = np.linspace(0.0, 1.0, 41)
            best = None
            for a in a_grid:
                for b in b_grid:
                    L = loss([a, b])
                    if (best is None) or (L < best[0]):
                        best = (L, a, b)
            return {"params": [best[1], best[2]], "loss": float(best[0])}
        else:
            a_grid = np.linspace(0.1, 3.0, 30)
            c_grid = np.linspace(-0.5, 2.0, 26)
            best = None
            for a in a_grid:
                for c in c_grid:
                    L = loss([a, c])
                    if (best is None) or (L < best[0]):
                        best = (L, a, c)
            return {"params": [best[1], best[2]], "loss": float(best[0])}
    else:
        res = opt.minimize(lambda p: loss(p), x0=np.array(x0, float), bounds=bnds, method="L-BFGS-B")
        p = res.x
        return {"params": [float(v) for v in p], "loss": float(res.fun)}


def sr_fit(feats: dict, niterations: int = 60):
    if PySRRegressor is None:
        return None
    X = np.c_[feats["x"], feats["Sh"], feats["dlnS"]]
    y = feats["fX_req"]
    model = PySRRegressor(
        niterations=niterations,
        unary_operators=["log", "sqrt", "exp", "abs"],
        binary_operators=["+", "-", "*", "/"],
        model_selection="best",
        procs=0,
        progress=False,
        maxsize=20,
    )
    model.fit(X, y)
    eqs = model.equations_
    return eqs


def plot_overlay(ax, feats: dict, family_name: str, params: list, title=""):
    R = feats["R"]; Vobs = feats["Vobs"]; Vbar = feats["Vbar"]
    x = feats["x"]; Sh = feats["Sh"]; dlnS = feats["dlnS"]
    func = FAMILIES[family_name][0]
    fX = func(params, x, Sh, dlnS)
    Vmod = Vbar * np.sqrt(np.maximum(0.0, 1.0 + fX))

    ax.plot(R, Vobs, 'k.', label='Observed')
    ax.plot(R, Vbar, color='#1f77b4', alpha=0.8, label='Baryons')
    ax.plot(R, Vmod, color='#d62728', alpha=0.9, label=f'{family_name}')
    ax.set_xlabel('R [kpc]'); ax.set_ylabel('V [km/s]'); ax.grid(True, alpha=0.3)
    ax.set_title(title)

    rmse = float(np.sqrt(np.mean((Vmod - Vobs) ** 2)))
    mape = float(np.median(np.abs((Vmod - Vobs) / np.maximum(np.abs(Vobs), 1e-9))))
    return rmse, mape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_galaxies", type=int, default=16)
    ap.add_argument("--do_sr", action="store_true", help="Run per-galaxy SR for fX_req")
    ap.add_argument("--outdir", type=str, default=os.path.join("gravity_learn", "experiments", "pg_infer"))
    args = ap.parse_args()

    ds = load_sparc()
    galaxies = ds.galaxies[:args.limit_galaxies]

    ts = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_run = os.path.join(args.outdir, ts)
    os.makedirs(out_run, exist_ok=True)

    rows = []
    for g in galaxies:
        feats = build_features(g)
        if feats is None:
            continue
        results = {}
        for fam in ("simple", "ratio", "exp"):
            fit = fit_family(fam, feats)
            results[fam] = fit
        # optional SR
        eq_path = None
        if args.do_sr and (PySRRegressor is not None):
            eqs = sr_fit(feats, niterations=60)
            if eqs is not None:
                eq_path = os.path.join(out_run, f"SR_{g.name}.csv")
                eqs.to_csv(eq_path, index=False)
        # Make a small overlay panel (best family by loss)
        best_fam = min(results.keys(), key=lambda k: results[k]["loss"])
        fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.6))
        rmse, mape = plot_overlay(ax, feats, best_fam, results[best_fam]["params"], title=f"{g.name} [{best_fam}]")
        fig.savefig(os.path.join(out_run, f"overlay_{g.name}.png"), dpi=140); plt.close(fig)
        # Summarize
        rows.append({
            "Galaxy": g.name,
            "best_family": best_fam,
            "rmse": rmse,
            "median_ape": mape,
            **{f"{fam}_loss": results[fam]["loss"] for fam in results},
            **{f"{fam}_params": json.dumps(results[fam]["params"]) for fam in results},
            "sr_equations_csv": eq_path or "",
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_run, "per_galaxy_summary.csv"), index=False)
        print(f"[per-galaxy] wrote {os.path.join(out_run, 'per_galaxy_summary.csv')}")


if __name__ == "__main__":
    main()