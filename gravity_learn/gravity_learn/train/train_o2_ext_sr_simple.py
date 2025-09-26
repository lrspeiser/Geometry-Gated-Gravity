from __future__ import annotations
import os, argparse, json, datetime
import numpy as np
import pandas as pd

try:
    from pysr import PySRRegressor
except Exception:
    raise ImportError("pysr is required for SR. pip install pysr")

from gravity_learn.train.train_o2_ext_sr import build_table

OUT_ROOT = os.path.join("gravity_learn", "experiments", "sr", "extended")


def choose_simplest_within_tolerance(eqs: pd.DataFrame, rel_tol: float = 0.05) -> pd.Series:
    # Ensure required columns
    assert "loss" in eqs.columns and "complexity" in eqs.columns
    best_loss = float(eqs["loss"].min())
    allowed = eqs[eqs["loss"] <= (1.0 + rel_tol) * best_loss]
    # pick minimal complexity then minimal loss among that
    idx = allowed.sort_values(["complexity", "loss"], ascending=[True, True]).index[0]
    return allowed.loc[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_galaxies", type=int, default=120)
    ap.add_argument("--niterations", type=int, default=350)
    ap.add_argument("--maxsize", type=int, default=14)
    ap.add_argument("--rel_tol", type=float, default=0.05, help="Relative loss tolerance for selecting simplest model")
    args = ap.parse_args()

    df = build_table(limit_galaxies=args.limit_galaxies)
    X_cols = ["x_dimless", "Sigma_hat", "grad_ln_Sigma"]
    X = df[X_cols].to_numpy()
    y = df["fX_over_bar"].to_numpy()

    # Simplicity-biased operator set and small maxsize
    model = PySRRegressor(
        niterations=int(args.niterations),
        unary_operators=["log", "sqrt", "exp"],
        binary_operators=["+", "-", "*", "/"],
        maxsize=int(args.maxsize),
        model_selection="best",
        procs=0,
        progress=True,
    )
    model.fit(X, y)
    eqs = model.equations_

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"run_simple_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    eq_path = os.path.join(out_dir, "fX_over_bar_simple_equations.csv")
    eqs.to_csv(eq_path, index=False)

    chosen = choose_simplest_within_tolerance(eqs, rel_tol=float(args.rel_tol))

    # Evaluate metrics
    y_hat = model.predict(X)
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    mape = float(np.median(np.abs((y_hat - y) / np.maximum(np.abs(y), 1e-9))))

    metrics = {
        "rmse": rmse,
        "median_ape": mape,
        "best_loss": float(eqs["loss"].min()),
        "chosen_complexity": int(chosen["complexity"]),
        "chosen_loss": float(chosen["loss"]),
        "chosen_equation": str(chosen.get("equation", "")),
        "sympy_format": str(chosen.get("sympy_format", "")),
        "lambda_format": str(chosen.get("lambda_format", "")),
        "features": X_cols,
        "n_points": int(len(y)),
        "config": {
            "niterations": int(args.niterations),
            "maxsize": int(args.maxsize),
            "rel_tol": float(args.rel_tol),
        },
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[simple SR] chosen:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()