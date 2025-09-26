from __future__ import annotations
import os, datetime, pandas as pd


def run_o2_sr(load_sparc_fn, prepare_training_table_fn, run_pysr_search_fn, out_root: str, niterations: int = 10,
               unary_ops=None, binary_ops=None):
    """Run a minimal SR pass for O2 using rigor-provided helpers.
    - load_sparc_fn: function(path_parquet, path_master, ...) -> dataset
    - prepare_training_table_fn: rigor.rigor.formula_search.prepare_training_table
    - run_pysr_search_fn: rigor.rigor.formula_search.run_pysr_search
    - out_root: directory for artifacts (created if needed)
    Returns (model, results_dataframe_path)
    """
    os.makedirs(out_root, exist_ok=True)
    # Minimal default paths; caller ensures existence
    ds = load_sparc_fn()
    train = prepare_training_table_fn(ds, use_outer_only=True)
    # Clean/Impute minimal NaNs for PySR compatibility
    if "Mbar_Msun" in train.columns:
        med = float(train["Mbar_Msun"].median(skipna=True)) if len(train) else 0.0
        if not (med == med):  # NaN check
            med = 0.0
        train["Mbar_Msun"] = train["Mbar_Msun"].fillna(med)
    if "Sigma_bar" in train.columns:
        medS = float(train["Sigma_bar"].median(skipna=True)) if len(train) else 1.0
        if not (medS == medS):
            medS = 1.0
        train["Sigma_bar"] = train["Sigma_bar"].fillna(medS)
    if "R_kpc" in train.columns:
        train = train.dropna(subset=["R_kpc"])  # must have radii
    train = train.dropna(subset=[c for c in train.columns if c in ("xi_emp",)])
    # Drop any residual NaNs in X columns that PySR will use
    X_cols = [c for c in ["R_kpc", "Sigma_bar", "Mbar_Msun"] if c in train.columns]
    if X_cols:
        train = train.dropna(subset=X_cols)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_root, f"equations_{ts}.csv")
    model, table = run_pysr_search_fn(train, out_csv=out_csv, niterations=niterations,
                                      unary_operators=unary_ops, binary_operators=binary_ops)
    return model, out_csv
