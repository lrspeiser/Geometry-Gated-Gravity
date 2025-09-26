from __future__ import annotations
import os, argparse, datetime
import numpy as np
import pandas as pd

try:
    from pysr import PySRRegressor
except Exception as e:
    raise ImportError("pysr is required for extended SR. pip install pysr")

from rigor.rigor.data import load_sparc
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma
from gravity_learn.analysis.effective_mass import compute_effective_mass_arrays


OUT_ROOT = os.path.join("gravity_learn", "experiments", "sr", "extended")


def build_table(limit_galaxies=100, use_outer_only=True):
    ds = load_sparc()
    rows = []
    n = 0
    for g in ds.galaxies:
        if n >= limit_galaxies:
            break
        R = g.R_kpc
        Vobs = g.Vobs_kms
        Vbar = g.Vbar_kms
        if R is None or Vobs is None or Vbar is None:
            continue
        mask = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
        if use_outer_only:
            mask = mask & g.outer_mask
        R = R[mask]; Vobs = Vobs[mask]; Vbar = Vbar[mask]
        if R.size < 8:
            continue
        Sigma = None
        if g.Sigma_bar is not None:
            Sigma = g.Sigma_bar[mask]
        else:
            Sigma = np.maximum(1e-3, np.exp(-R / np.maximum(1.0, np.nanmedian(R))))
        x = dimensionless_radius(R, Rd=(g.Rd_kpc or None))
        Sh = sigma_hat(Sigma)
        dlnS = grad_log_sigma(R, Sigma)
        # Targets
        eps = 1e-12
        xi_emp = (Vobs / np.maximum(Vbar, 1e-6)) ** 2
        xi_excess = xi_emp - 1.0
        Mbar, MX, fX, gX = compute_effective_mass_arrays(R, Vobs, Vbar)
        df = pd.DataFrame({
            "Galaxy": g.name,
            "R_kpc": R,
            "x_dimless": np.asarray(x),
            "Sigma_hat": np.asarray(Sh),
            "grad_ln_Sigma": np.asarray(dlnS),
            "xi_excess": xi_excess,
            "fX_over_bar": fX,
            "gX": gX,
        })
        # Sanitize
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df) == 0:
            continue
        rows.append(df)
        n += 1
    if not rows:
        raise RuntimeError("No rows for extended SR.")
    return pd.concat(rows, ignore_index=True)


def run_pysr_fit(X, y, out_dir, label, niterations=300):
    os.makedirs(out_dir, exist_ok=True)
    model = PySRRegressor(
        niterations=niterations,
        unary_operators=["log", "sqrt", "exp", "abs"],
        binary_operators=["+", "-", "*", "/"],
        model_selection="best",
        procs=0,
        progress=True,
    )
    model.fit(X, y)
    eqs = model.equations_
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"{label}_equations_{ts}.csv")
    eqs.to_csv(out_csv, index=False)
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_galaxies", type=int, default=100)
    ap.add_argument("--niterations", type=int, default=300)
    args = ap.parse_args()

    df = build_table(limit_galaxies=args.limit_galaxies)
    # Features matrix
    X_cols = ["x_dimless", "Sigma_hat", "grad_ln_Sigma"]
    X = df[X_cols].to_numpy()

    # Targets: xi_excess and fX_over_bar (mass ratio)
    out_dir = os.path.join(OUT_ROOT, datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    xi_csv = run_pysr_fit(X, df["xi_excess"].to_numpy(), out_dir, label="xi_excess", niterations=args.niterations)
    fx_csv = run_pysr_fit(X, df["fX_over_bar"].to_numpy(), out_dir, label="fX_over_bar", niterations=args.niterations)
    print(f"[extended SR] wrote {xi_csv} and {fx_csv}")


if __name__ == "__main__":
    main()