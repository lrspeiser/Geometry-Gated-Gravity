from __future__ import annotations
import os, glob, pandas as pd

try:
    from pysr import PySRRegressor
except Exception:
    PySRRegressor = None


def distill(run_dir: str, out_csv: str):
    if PySRRegressor is None:
        raise ImportError("pysr is required for distillation. See gravity_learn/README.md")
    # Expecting files like pred_*.csv with columns: R_kpc,g_tail
    paths = sorted(glob.glob(os.path.join(run_dir, "pred_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No prediction CSVs found under {run_dir}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    X = df[["R_kpc"]].to_numpy()
    y = df["g_tail"].to_numpy()
    model = PySRRegressor(niterations=50, unary_operators=["log", "sqrt", "exp"], binary_operators=["+", "-", "*", "/"])
    model.fit(X, y)
    eqs = model.equations_
    eqs.to_csv(out_csv, index=False)
    return out_csv
