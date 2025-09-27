from __future__ import annotations
import os, argparse, json, datetime
import numpy as np
import pandas as pd

from rigor.rigor.data import load_sparc
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma, kappa_rho

from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

try:
    from pysr import PySRRegressor
except Exception:
    PySRRegressor = None

G_KPC_KMS2_PER_MSUN = 4.30091e-6


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, kernel, mode='same')

def _gaussian_smooth(x: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return x.copy()
    # Build kernel with radius 3*sigma
    radius = max(1, int(3 * sigma_bins))
    xs = np.arange(-radius, radius + 1)
    kern = np.exp(-(xs ** 2) / (2.0 * sigma_bins * sigma_bins))
    kern /= kern.sum()
    return np.convolve(x, kern, mode='same')

def _align_len(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a)
    if a.shape[0] == n:
        return a
    if a.shape[0] > n:
        return a[:n]
    # pad by edge value if shorter
    if a.shape[0] == 0:
        return np.zeros(n, dtype=float)
    pad_width = n - a.shape[0]
    return np.pad(a, (0, pad_width), mode='edge')

def build_table(limit_galaxies: int = -1, use_outer_flag: bool = True) -> pd.DataFrame:
    ds = load_sparc()
    rows = []
    count = 0
    for g in ds.galaxies:
        if (limit_galaxies > 0) and (count >= limit_galaxies):
            break
        R = g.R_kpc; Vobs = g.Vobs_kms; Vbar = g.Vbar_kms
        if R is None or Vobs is None or Vbar is None:
            continue
        mask = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
        R = np.asarray(R)[mask]; Vobs = np.asarray(Vobs)[mask]; Vbar = np.asarray(Vbar)[mask]
        if R.size < 6:
            continue
        Sigma = g.Sigma_bar[mask] if (g.Sigma_bar is not None) else np.maximum(1e-3, np.exp(-R / np.maximum(1.0, np.nanmedian(R))))
        x = np.asarray(dimensionless_radius(R, Rd=(g.Rd_kpc or None)))
        Sh = np.asarray(sigma_hat(Sigma))
        dlnS = np.asarray(grad_log_sigma(R, Sigma))
        # Non-local and curvature features
        n = len(R)
        Sh_rm3 = _align_len(_rolling_mean(Sh, 3), n)
        Sh_rm5 = _align_len(_rolling_mean(Sh, 5), n)
        Sh_rm9 = _align_len(_rolling_mean(Sh, 9), n)
        Sh_g1 = _align_len(_gaussian_smooth(Sh, sigma_bins=1.0), n)
        Sh_g2 = _align_len(_gaussian_smooth(Sh, sigma_bins=2.0), n)
        kappa_sigma = _align_len(kappa_rho(R, Sigma), n)
        eps = 1e-12
        g_req = (Vobs * Vobs) / (R + eps)
        g_bar = (Vbar * Vbar) / (R + eps)
        gX = g_req - g_bar
        M_bar_eff = (Vbar * Vbar) * R / G_KPC_KMS2_PER_MSUN
        M_X_eff = gX * (R * R) / G_KPC_KMS2_PER_MSUN
        fX = M_X_eff / np.maximum(M_bar_eff, 1e-12)
        xi_emp = (Vobs / np.maximum(Vbar, 1e-6)) ** 2
        # auxiliary engineered terms
        lnR = np.log(np.maximum(R, 1e-6)); invR = 1.0 / np.maximum(R, 1e-6)
        x2 = x * x; x3 = x2 * x
        # outer indicator
        if use_outer_flag and hasattr(g, 'outer_mask') and g.outer_mask is not None:
            outer = g.outer_mask[mask].astype(int)
            outer = _align_len(outer, len(R)).astype(int)
        else:
            outer = (np.arange(len(R)) >= int(0.66 * len(R))).astype(int)
        df = pd.DataFrame({
            "Galaxy": g.name,
            "R_kpc": R,
            "Vobs": Vobs,
            "Vbar": Vbar,
            "g_bar": g_bar,
            "x": x,
            "x2": x2,
            "x3": x3,
            "lnR": lnR,
            "invR": invR,
            "Sigma": Sigma,
            "Sigma_hat": Sh,
            "Sigma_hat_rm3": Sh_rm3,
            "Sigma_hat_rm5": Sh_rm5,
            "Sigma_hat_rm9": Sh_rm9,
            "Sigma_hat_g1": Sh_g1,
            "Sigma_hat_g2": Sh_g2,
            "kappa_sigma": kappa_sigma,
            "grad_ln_Sigma": dlnS,
            "outer": outer,
            "xi_emp": xi_emp,
            "fX_over_bar": fX,
            "gX": gX,
        })
        rows.append(df)
        count += 1
    if not rows:
        raise RuntimeError("No SPARC rows for NN training.")
    return pd.concat(rows, ignore_index=True)


def pick_features(df: pd.DataFrame):
    feature_cols = [
        "x", "x2", "x3", "lnR", "invR", "g_bar",
        "Sigma", "Sigma_hat",
        "Sigma_hat_rm3", "Sigma_hat_rm5", "Sigma_hat_rm9",
        "Sigma_hat_g1", "Sigma_hat_g2",
        "kappa_sigma", "grad_ln_Sigma", "outer",
    ]
    X = df[feature_cols].to_numpy()
    groups = df["Galaxy"].to_numpy()
    return feature_cols, X, groups


def compute_target(df: pd.DataFrame, target: str):
    if target == "fX":
        y = df["fX_over_bar"].to_numpy()
    elif target == "xi":
        y = (df["xi_emp"].to_numpy() - 1.0)
    elif target == "gX":
        y = df["gX"].to_numpy()
    else:
        raise ValueError("target must be one of fX|xi|gX")
    return y


def train_nn(df: pd.DataFrame, target: str = "fX", n_splits: int = 5, max_iter: int = 400):
    feature_cols, X, groups = pick_features(df)
    y = compute_target(df, target)

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    preds = np.zeros_like(y, dtype=float)

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                         alpha=1e-4, learning_rate_init=1e-3, max_iter=max_iter,
                         early_stopping=True, n_iter_no_change=15, validation_fraction=0.1,
                         random_state=42)
        )
        model.fit(X[tr], y[tr])
        y_hat = model.predict(X[va])
        mse = float(mean_squared_error(y[va], y_hat))
        # median absolute percent error relative to 1+fX for stability
        denom = np.maximum(np.abs(y[va]), 1e-6)
        mape = float(np.median(np.abs((y_hat - y[va]) / denom)))
        fold_metrics.append({"fold": fold, "mse": mse, "median_ape": mape})
        preds[va] = y_hat

    # Fit on all data for permutation importance
    final_model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                     alpha=1e-4, learning_rate_init=1e-3, max_iter=max_iter,
                     early_stopping=True, n_iter_no_change=15, validation_fraction=0.1,
                     random_state=42)
    )
    final_model.fit(X, y)

    try:
        pim = permutation_importance(final_model, X, y, n_repeats=8, random_state=0)
        imp_df = pd.DataFrame({"feature": feature_cols, "importance_mean": pim.importances_mean, "importance_std": pim.importances_std})
        imp_df = imp_df.sort_values("importance_mean", ascending=False)
    except Exception:
        imp_df = pd.DataFrame({"feature": feature_cols, "importance_mean": np.nan, "importance_std": np.nan})

    return final_model, feature_cols, preds, np.array(y), fold_metrics, imp_df


def distill_with_pysr(feature_cols, X, y_pred, out_csv, niterations=300):
    if PySRRegressor is None:
        return None
    model = PySRRegressor(
        niterations=niterations,
        unary_operators=["log", "sqrt", "exp", "abs"],
        binary_operators=["+", "-", "*", "/"],
        model_selection="best",
        procs=0,
        progress=True,
        maxsize=30,
    )
    model.fit(X, y_pred)
    eqs = model.equations_
    eqs.to_csv(out_csv, index=False)
    return eqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="fX", choices=["fX", "xi", "gX"])  # predict fX_over_bar (default)
    ap.add_argument("--limit_galaxies", type=int, default=-1)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--max_iter", type=int, default=400)
    ap.add_argument("--do_sr", action="store_true")
    ap.add_argument("--outdir", type=str, default=os.path.join("gravity_learn", "experiments", "nn_o2"))
    args = ap.parse_args()

    df = build_table(limit_galaxies=args.limit_galaxies)
    feature_cols, X, groups = pick_features(df)
    y = compute_target(df, args.target)

    model, fcols, preds, y_true, fold_metrics, imp_df = train_nn(df, target=args.target, n_splits=args.splits, max_iter=args.max_iter)

    ts = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_run = os.path.join(args.outdir, ts)
    os.makedirs(out_run, exist_ok=True)

    # Save metrics and importances
    with open(os.path.join(out_run, "fold_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)
    imp_df.to_csv(os.path.join(out_run, "permutation_importance.csv"), index=False)

    # Save predictions joined with IDs
    out_df = df[["Galaxy", "R_kpc"]].copy()
    out_df["y_true"] = y_true
    out_df["y_pred"] = preds
    out_df.to_csv(os.path.join(out_run, "predictions_by_radius.csv"), index=False)

    # Distill with PySR if requested
    if args.do_sr and (PySRRegressor is not None):
        eqs = distill_with_pysr(fcols, X, preds, os.path.join(out_run, "pysr_distilled_equations.csv"), niterations=300)
        if eqs is not None:
            print("[nn_o2] distilled equations saved.")

    print(f"[nn_o2] wrote outputs to {out_run}")


if __name__ == "__main__":
    main()