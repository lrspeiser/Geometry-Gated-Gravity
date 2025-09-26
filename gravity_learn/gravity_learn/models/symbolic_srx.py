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
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_root, f"equations_{ts}.csv")
    model, table = run_pysr_search_fn(train, out_csv=out_csv, niterations=niterations,
                                      unary_operators=unary_ops, binary_operators=binary_ops)
    return model, out_csv
