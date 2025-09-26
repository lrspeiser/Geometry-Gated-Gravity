#!/usr/bin/env python
from __future__ import annotations
import os
import argparse
import math
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pareto_front_min(df: pd.DataFrame, loss_col: str, complexity_col: str) -> pd.DataFrame:
    # Keep non-dominated points (minimize loss and complexity)
    pts = df[[loss_col, complexity_col]].to_numpy()
    keep = np.ones(len(df), dtype=bool)
    # Sort by loss then complexity for a simple sweep
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    best_complexity = math.inf
    # Walk increasing loss, track best complexity seen
    # A point is on Pareto front if its complexity is strictly less than all previous complexities at <= loss.
    keep[:] = False
    for idx in order:
        loss_i, comp_i = pts[idx]
        if comp_i < best_complexity - 1e-12:
            keep[idx] = True
            best_complexity = comp_i
    return df.loc[keep]


def select_top_k(front: pd.DataFrame, loss_col: str, complexity_col: str, k: int = 5) -> pd.DataFrame:
    # Normalize both axes and select by sum of normalized ranks
    f = front.copy()
    # Avoid divide by zero for degenerate cases
    l = f[loss_col].to_numpy()
    c = f[complexity_col].to_numpy()
    if len(f) == 0:
        return f
    lmin, lmax = float(np.min(l)), float(np.max(l))
    cmin, cmax = float(np.min(c)), float(np.max(c))
    ln = (l - lmin) / (lmax - lmin + 1e-12)
    cn = (c - cmin) / (cmax - cmin + 1e-12)
    f["simplicity_accuracy_score"] = ln + cn
    f = f.sort_values(["simplicity_accuracy_score", loss_col, complexity_col], ascending=[True, True, True])
    return f.head(k)


def plot_complexity_vs_loss(df: pd.DataFrame, front_idx: set[int], loss_col: str, complexity_col: str, out_png: str, title: str):
    plt.figure(figsize=(7, 5))
    is_front = df.index.to_series().isin(front_idx)
    plt.scatter(df[complexity_col][~is_front], df[loss_col][~is_front], s=18, alpha=0.5, label="candidates")
    plt.scatter(df[complexity_col][is_front], df[loss_col][is_front], s=30, color="#d62728", label="Pareto front")
    plt.xlabel("Complexity (lower = simpler)")
    plt.ylabel("Loss (lower = better)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def run_for_file(csv_path: str, outdir: str, top_k: int):
    df = pd.read_csv(csv_path)
    # Infer columns
    col_loss = "loss" if "loss" in df.columns else ("val_loss" if "val_loss" in df.columns else None)
    if col_loss is None:
        raise ValueError(f"No 'loss' or 'val_loss' column in {csv_path}")
    col_complexity = "complexity" if "complexity" in df.columns else None
    if col_complexity is None:
        # If not provided, approximate complexity from expression length
        # but in your CSVs, 'complexity' exists, so this is just a fallback
        expr_col = None
        for cand in ("equation", "sympy_format", "lambda_format"):
            if cand in df.columns:
                expr_col = cand
                break
        if expr_col is None:
            raise ValueError(f"No complexity or expression column in {csv_path}")
        df[col_complexity] = df[expr_col].astype(str).map(len)
        col_complexity = col_complexity

    front = pareto_front_min(df, loss_col=col_loss, complexity_col=col_complexity)
    top = select_top_k(front, loss_col=col_loss, complexity_col=col_complexity, k=top_k)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    front_csv = os.path.join(outdir, f"pareto_front_{base}.csv")
    top_csv = os.path.join(outdir, f"top{top_k}_{base}.csv")
    plot_png = os.path.join(outdir, f"scatter_{base}.png")

    front.to_csv(front_csv, index=False)
    top.to_csv(top_csv, index=False)

    # Plot
    plot_complexity_vs_loss(df, set(front.index.tolist()), loss_col=col_loss, complexity_col=col_complexity,
                            out_png=plot_png, title=f"Pareto: {base}")

    # Return small summary for markdown
    eq_col = "equation" if "equation" in df.columns else ("sympy_format" if "sympy_format" in df.columns else None)
    lines = []
    for i, row in top.iterrows():
        eq = str(row[eq_col]) if eq_col else "(equation not found)"
        lines.append(f"- loss={row[col_loss]:.6g}, complexity={int(row[col_complexity])}: {eq}")
    return base, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV paths of SR equations")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("gravity_learn", "experiments", "sr", "extended", f"pareto_{ts}")
    os.makedirs(outdir, exist_ok=True)

    summary_md = [f"# Pareto filtering summary ({ts})", ""]
    for p in args.inputs:
        base, lines = run_for_file(p, outdir, args.top_k)
        summary_md.append(f"## {base}")
        summary_md.extend(lines)
        summary_md.append("")

    with open(os.path.join(outdir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md))

    print(f"[pareto] wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
