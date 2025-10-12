
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit smooth B/T laws for many-path parameters from per-galaxy best fits
(mega_parallel_results.json).  Produces bt_law_params.json and a figure.
"""
import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bt_laws import morph_to_bt, fit_one_law, save_theta

def load_mega_results(path: Path) -> pd.DataFrame:
    d = json.loads(Path(path).read_text())
    rows = []
    for item in d["results"]:
        row = {
            "name": item.get("name"),
            "hubble_type": item.get("hubble_type"),
            "type_group": item.get("type_group"),
            "best_error": item.get("best_error"),
        }
        bp = item.get("best_params", {})
        # handle both lambda_hat and lambda_ring naming
        lam = bp.get("lambda_ring", bp.get("lambda_hat"))
        row.update({
            "eta": bp.get("eta"),
            "ring_amp": bp.get("ring_amp"),
            "M_max": bp.get("M_max"),
            "lambda_ring": lam,
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    # Estimate B/T from morphology
    df["B_T"] = [morph_to_bt(ht, tg) for ht, tg in zip(df["hubble_type"], df["type_group"])]
    # weights: emphasize low-error galaxies (downweight outliers)
    df["w"] = 1.0 / (1.0 + 0.02 * df["best_error"].values)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/mega_test/mega_parallel_results.json"))
    ap.add_argument("--out_params", type=Path, default=Path("many_path_model/bt_law/bt_law_params.json"))
    ap.add_argument("--out_fig", type=Path, default=Path("many_path_model/bt_law/bt_law_fits.png"))
    args = ap.parse_args()

    df = load_mega_results(args.results).dropna(subset=["eta","ring_amp","M_max","lambda_ring"])

    theta = {}
    # Fit each parameter with robust loss
    # Bounds informed by your clustering plots
    theta["eta"] = fit_one_law(
        df["B_T"].values, df["eta"].values,
        lo_bounds=(0.01, 0.3),  hi_bounds=(0.5, 2.0),
        gamma_bounds=(0.4, 4.0), n_trials=6000, weights=df["w"].values
    )
    theta["ring_amp"] = fit_one_law(
        df["B_T"].values, df["ring_amp"].values,
        lo_bounds=(0.0, 0.5),   hi_bounds=(3.0, 15.0),
        gamma_bounds=(0.4, 4.0), n_trials=6000, weights=df["w"].values
    )
    theta["M_max"] = fit_one_law(
        df["B_T"].values, df["M_max"].values,
        lo_bounds=(0.8, 1.2),   hi_bounds=(2.0, 5.0),
        gamma_bounds=(0.4, 4.0), n_trials=6000, weights=df["w"].values
    )
    theta["lambda_ring"] = fit_one_law(
        df["B_T"].values, df["lambda_ring"].values,
        lo_bounds=(6.0, 10.0),  hi_bounds=(25.0, 55.0),
        gamma_bounds=(0.4, 4.0), n_trials=6000, weights=df["w"].values
    )

    # Save
    save_theta(args.out_params, theta)

    # Diagnostic plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()
    params = ["eta", "ring_amp", "M_max", "lambda_ring"]
    labels = [r"$\eta$", "ring_amp", r"$M_{\max}$", r"$\lambda_{\rm ring}$"]
    B = np.linspace(0, 0.7, 200)

    def law(B, lo, hi, gamma):
        return lo + (hi - lo) * (1.0 - B) ** gamma

    for ax, p, lbl in zip(axs, params, labels):
        lo, hi, g = theta[p]["lo"], theta[p]["hi"], theta[p]["gamma"]
        curve = law(B, lo, hi, g)
        # scatter
        ax.scatter(df["B_T"], df[p], s=16, alpha=0.5, label="per-galaxy best")
        ax.plot(B, curve, lw=2.5, label=f"fit: lo={lo:.2f}, hi={hi:.2f}, gamma={g:.2f}")
        ax.set_xlabel("Bulge-to-total B/T")
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle("Fitted Continuous B/T Laws for Many-Path Parameters", y=0.98, fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=150)
    print(f"Saved laws to: {args.out_params}")
    print(f"Saved diagnostic plot to: {args.out_fig}")

if __name__ == "__main__":
    main()
