from __future__ import annotations
import os, json, argparse, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rigor.rigor.data import load_sparc
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

try:
    import scipy.stats as stats
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _load_best_family(path_json: str):
    if os.path.exists(path_json):
        with open(path_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"best_family": "ratio", "params": [0.97, 0.0]}


def _family_predict(fam: str, params, x, Sh, dlnS):
    if fam == "ratio":
        a, b = params
        denom = (a - b * Sh)
        denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
        fX = np.maximum(0.0, (x * x) / denom)
    elif fam == "exp":
        alpha, c = params
        alpha = max(alpha, 0.0)
        fX = np.maximum(0.0, alpha * (x * x) * (np.exp(Sh) + c))
    else:
        # fallback simple
        k = params[0] if (params and len(params) > 0) else 2.56
        fX = np.maximum(0.0, k * x)
    return fX


def build_rows(ds, fam: str, params):
    rows = []
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

        fX = _family_predict(fam, params, x, Sh, dlnS)
        Vmod = Vbar * np.sqrt(np.maximum(0.0, 1.0 + fX))

        # define outer region: use provided mask if present else last third
        if hasattr(g, 'outer_mask') and g.outer_mask is not None:
            outer_mask = g.outer_mask[mask]
        else:
            outer_mask = np.zeros_like(R, dtype=bool)
            outer_mask[int(0.66 * len(R)):] = True

        # error metrics
        rmse = float(np.sqrt(np.mean((Vmod - Vobs) ** 2)))
        mape = float(np.median(np.abs((Vmod - Vobs) / np.maximum(np.abs(Vobs), 1e-9))))
        outer_mape = float(np.median(np.abs((Vmod[outer_mask] - Vobs[outer_mask]) / np.maximum(np.abs(Vobs[outer_mask]), 1e-9)))) if np.any(outer_mask) else np.nan

        # feature summaries
        feats = {
            "Galaxy": g.name,
            "n_points": int(len(R)),
            "Rd_kpc": float(g.Rd_kpc) if getattr(g, 'Rd_kpc', None) is not None else np.nan,
            "Mbar_Msun": float(g.Mbar_Msun) if getattr(g, 'Mbar_Msun', None) is not None else np.nan,
            "x_max": float(np.max(x)),
            "x_med": float(np.median(x)),
            "Sh_med": float(np.median(Sh)),
            "Sh_med_outer": float(np.median(Sh[outer_mask])) if np.any(outer_mask) else np.nan,
            "abs_dlnS_med": float(np.median(np.abs(dlnS))),
            "abs_dlnS_med_outer": float(np.median(np.abs(dlnS[outer_mask]))) if np.any(outer_mask) else np.nan,
            # slopes
            "slope_Vobs_outer": _log_slope(R, Vobs, outer_mask),
            "slope_Vbar_outer": _log_slope(R, Vbar, outer_mask),
            # errors
            "rmse": rmse,
            "mape": mape,
            "outer_mape": outer_mape,
        }
        rows.append(feats)
    return pd.DataFrame(rows)


def _log_slope(R, Y, mask):
    R = np.asarray(R); Y = np.asarray(Y)
    if mask is None or not np.any(mask):
        return np.nan
    Rm = R[mask]; Ym = Y[mask]
    if len(Rm) < 3:
        return np.nan
    eps = 1e-12
    lR = np.log(np.maximum(Rm, eps)); lY = np.log(np.maximum(np.abs(Ym), eps))
    # simple linear fit in log space
    A = np.vstack([lR, np.ones_like(lR)]).T
    coeff, *_ = np.linalg.lstsq(A, lY, rcond=None)
    return float(coeff[0])


def compute_correlations(df: pd.DataFrame, feature_cols, error_col="mape"):
    corrs = []
    for c in feature_cols:
        x = df[c].to_numpy()
        y = df[error_col].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            continue
        if _HAVE_SCIPY:
            rho, p = stats.spearmanr(x[mask], y[mask])
            corrs.append({"feature": c, "spearman": float(rho), "p_value": float(p)})
        else:
            xm = x[mask] - x[mask].mean(); ym = y[mask] - y[mask].mean()
            r = float((xm * ym).sum() / (np.sqrt((xm * xm).sum()) * np.sqrt((ym * ym).sum()) + 1e-12))
            corrs.append({"feature": c, "pearson": r})
    return pd.DataFrame(corrs)


def model_feature_importance(df: pd.DataFrame, feature_cols, error_col="mape", out_png=None):
    if not _HAVE_SKLEARN:
        return None
    # Label: good vs poor by quartiles of error
    y = df[error_col].to_numpy()
    thr = float(np.nanquantile(y, 0.75))  # worst quartile = 1, others = 0
    ylab = (y >= thr).astype(int)
    X = df[feature_cols].to_numpy()
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[mask]; ylab = ylab[mask]
    if len(ylab) < 20:
        return None

    # Random forest importance
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    rf.fit(X, ylab)
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)

    if out_png is not None:
        plt.figure(figsize=(6, 4))
        plt.barh(imp_df["feature"], imp_df["importance"], color="#2ca02c")
        plt.gca().invert_yaxis()
        plt.xlabel("RF importance"); plt.title("Feature importances (RandomForest)")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

    # Logistic regression coefficients (z-scored)
    logi = make_pipeline(StandardScaler(with_mean=True, with_std=True), LogisticRegression(max_iter=200, class_weight="balanced"))
    logi.fit(X, ylab)
    # extract coefs (after scaler)
    LR = logi.named_steps["logisticregression"]
    coef = LR.coef_[0]
    lr_df = pd.DataFrame({"feature": feature_cols, "coef": coef}).sort_values("coef", ascending=False)
    return {"rf": imp_df, "logit": lr_df, "threshold_mape": thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family_json", type=str, default=os.path.join("gravity_learn", "experiments", "eval", "global_fit", "best_family.json"))
    ap.add_argument("--outdir", type=str, default=os.path.join("gravity_learn", "experiments", "eval", "error_analysis"))
    ap.add_argument("--error_col", type=str, default="mape", choices=["mape", "outer_mape", "rmse"]) 
    args = ap.parse_args()

    best = _load_best_family(args.family_json)
    fam = best.get("best_family", "ratio")
    params = best.get("params", [0.97, 0.0])

    ds = load_sparc()
    df = build_rows(ds, fam, params)

    # Select features
    feature_cols = [
        "n_points", "Rd_kpc", "Mbar_Msun", "x_max", "x_med", "Sh_med", "Sh_med_outer",
        "abs_dlnS_med", "abs_dlnS_med_outer", "slope_Vobs_outer", "slope_Vbar_outer"
    ]

    ts = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_run = os.path.join(args.outdir, ts)
    os.makedirs(out_run, exist_ok=True)

    # Save raw table
    df.to_csv(os.path.join(out_run, "features_and_errors.csv"), index=False)

    # Correlations
    corr_df = compute_correlations(df, feature_cols, error_col=args.error_col)
    corr_df.to_csv(os.path.join(out_run, f"spearman_{args.error_col}.csv"), index=False)

    # Feature importances (if sklearn available)
    fi = model_feature_importance(df, feature_cols, error_col=args.error_col, out_png=os.path.join(out_run, "feature_importance_rf.png"))
    if fi is not None:
        fi["rf"].to_csv(os.path.join(out_run, "rf_importance.csv"), index=False)
        fi["logit"].to_csv(os.path.join(out_run, "logit_coef.csv"), index=False)
        with open(os.path.join(out_run, "labels_threshold.json"), "w", encoding="utf-8") as f:
            json.dump({"threshold_on": args.error_col, "threshold_value": fi["threshold_mape"]}, f, indent=2)

    # Simple insights
    insights = {
        "top_corr": corr_df.sort_values(by=[c for c in ["spearman", "pearson"] if c in corr_df.columns][0], ascending=True if "spearman" in corr_df.columns else False).head(8).to_dict(orient="records"),
        "note": "Negative correlation means larger feature associates with smaller error (better fits)."
    }
    with open(os.path.join(out_run, "insights.json"), "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    print(f"[error-analysis] wrote outputs to {out_run}")


if __name__ == "__main__":
    main()