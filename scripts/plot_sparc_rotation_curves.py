#!/usr/bin/env python3
"""
Generate rotation curve plots for SPARC galaxies.

- X axis: radius (kpc)
- Y axis (left): observed rotation speed (km/s)
- Y2 axis (right): Required G per radius (from predictions-by-radius), plotted as a line. Also overlays GR baseline (G=1).
- One PNG per galaxy, saved under outputs/rotation_curves/
- Coverage summary saved under outputs/coverage/sparc_coverage_summary.csv

Data sources used (existing in this repo):
- data/sparc_rotmod_ltg.parquet (columns: galaxy, R_kpc, Vobs_kms, eVobs_kms, Vgas_kms, Vdisk_kms, Vbul_kms)
- data/sparc_predictions_by_radius.csv (columns: galaxy, R_kpc, G_required, G_ratio, ...)
- data/sparc_master_clean.parquet (galaxy list and metadata) [optional but preferred for coverage]

This script does not require any API keys or external services.
"""
from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_name(name: str) -> str:
    # Make a filesystem-safe filename from a galaxy name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_rotmod(rotmod_path: Path) -> pd.DataFrame:
    if not rotmod_path.exists():
        raise FileNotFoundError(f"Rotation data parquet not found: {rotmod_path}")
    df = pd.read_parquet(rotmod_path)
    expected = {"galaxy", "R_kpc", "Vobs_kms"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Parquet {rotmod_path} is missing required columns: {sorted(missing)}"
        )
    return df


def load_master(master_path: Path) -> Optional[pd.DataFrame]:
    if not master_path.exists():
        return None
    df = pd.read_parquet(master_path)
    # Try common name columns
    name_col = None
    for c in ("galaxy", "Galaxy"):
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        return None
    # Normalize to 'galaxy'
    if name_col != "galaxy":
        df = df.rename(columns={name_col: "galaxy"})
    # Drop na
    df = df[df["galaxy"].notna()].copy()
    df["galaxy"] = df["galaxy"].astype(str)
    return df


def load_predictions_by_radius(pred_path: Path) -> Optional[pd.DataFrame]:
    """Load predictions-by-radius data (supports G_required and G_ratio).

    Accepts CSV or Parquet. Requires at minimum: galaxy, R_kpc.
    Returns None if file is missing.
    """
    if not pred_path.exists():
        return None
    if pred_path.suffix.lower() in {".csv", ".tsv"}:
        if pred_path.suffix.lower() == ".tsv":
            df = pd.read_csv(pred_path, sep="\t")
        else:
            df = pd.read_csv(pred_path)
    else:
        df = pd.read_parquet(pred_path)
    required_min = {"galaxy", "R_kpc"}
    if not required_min.issubset(df.columns):
        return None
    df = df[df["galaxy"].notna() & df["R_kpc"].notna()].copy()
    df["galaxy"] = df["galaxy"].astype(str)
    return df


def plot_one(
    galaxy: str,
    gdf: pd.DataFrame,
    outdir: Path,
    overlay_components: bool = False,
    pred_gdf: Optional[pd.DataFrame] = None,
    y2_field: str = "G_required",
) -> Optional[Path]:
    # Sanitize
    gdf = gdf.copy()
    gdf = gdf[(gdf["R_kpc"].notna()) & (gdf["Vobs_kms"].notna())]
    if gdf.empty:
        return None
    gdf = gdf.sort_values("R_kpc")

    fig, ax = plt.subplots(figsize=(7, 5))

    # Left axis: observed rotation speed
    if "eVobs_kms" in gdf.columns and gdf["eVobs_kms"].notna().any():
        yerr = gdf["eVobs_kms"].to_numpy()
        ax.errorbar(
            gdf["R_kpc"], gdf["Vobs_kms"], yerr=yerr, fmt="o", ms=3, elinewidth=0.8, capsize=2, label="Observed"
        )
    else:
        ax.plot(gdf["R_kpc"], gdf["Vobs_kms"], "o-", ms=3, lw=1.2, label="Observed")

    # Optional overlays on left axis
    if overlay_components:
        if "Vgas_kms" in gdf.columns and gdf["Vgas_kms"].notna().any():
            ax.plot(gdf["R_kpc"], gdf["Vgas_kms"], "-", lw=0.9, alpha=0.7, label="Gas")
        if "Vdisk_kms" in gdf.columns and gdf["Vdisk_kms"].notna().any():
            ax.plot(gdf["R_kpc"], gdf["Vdisk_kms"], "-", lw=0.9, alpha=0.7, label="Disk")
        if "Vbul_kms" in gdf.columns and gdf["Vbul_kms"].notna().any():
            ax.plot(gdf["R_kpc"], gdf["Vbul_kms"], "-", lw=0.9, alpha=0.7, label="Bulge")
        if {"Vgas_kms", "Vdisk_kms", "Vbul_kms"}.issubset(gdf.columns):
            Vbar = np.sqrt(
                np.nan_to_num(gdf["Vgas_kms"].to_numpy(), nan=0.0) ** 2
                + np.nan_to_num(gdf["Vdisk_kms"].to_numpy(), nan=0.0) ** 2
                + np.nan_to_num(gdf["Vbul_kms"].to_numpy(), nan=0.0) ** 2
            )
            if np.isfinite(Vbar).any():
                ax.plot(gdf["R_kpc"], Vbar, "--", lw=1.0, alpha=0.8, label="Baryonic (quad sum)")

    # Right axis: required G line from predictions-by-radius, if provided
    ax2 = None
    if pred_gdf is not None and y2_field in pred_gdf.columns:
        pg = pred_gdf.copy()
        pg = pg[(pg["R_kpc"].notna()) & (pg[y2_field].notna())]
        if not pg.empty:
            pg = pg.sort_values("R_kpc")
            ax2 = ax.twinx()
            label = "Required G (to match Vobs)" if y2_field == "G_required" else ("G ratio (G_required/G_pred)" if y2_field == "G_ratio" else y2_field)
            ax2.plot(pg["R_kpc"], pg[y2_field], color="tab:red", lw=1.2, label=label)
            # Overlay GR baseline: G = 1
            ax2.axhline(1.0, color="gray", lw=1.0, ls=":", label="GR baseline (G=1)")
            ax2.set_ylabel(label, color="tab:red")
            ax2.tick_params(axis='y', labelcolor='tab:red')

    # Titles, labels, legend
    ax.set_title(f"{galaxy} rotation curve")
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Rotation speed (km/s)")
    ax.grid(True, ls=":", alpha=0.5)

    # Combined legend across both axes, if present
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if ax2 is not None else ([], []))
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)

    fig.tight_layout()

    outpath = outdir / f"{safe_name(galaxy)}_rotation_curve.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def build_coverage(rotmod: pd.DataFrame, master: Optional[pd.DataFrame]) -> pd.DataFrame:
    rotmod_galaxies = rotmod["galaxy"].dropna().astype(str).unique().tolist()
    rot_points = rotmod.groupby("galaxy")["R_kpc"].count()

    if master is None:
        cov = (
            pd.DataFrame({"galaxy": rotmod_galaxies})
            .assign(in_master=np.nan)
            .assign(in_rotmod=True)
            .assign(num_points=lambda d: d["galaxy"].map(rot_points).fillna(0).astype(int))
        )
        return cov

    master_galaxies = master["galaxy"].dropna().astype(str).unique().tolist()
    all_gal = sorted(set(master_galaxies) | set(rotmod_galaxies))
    cov = pd.DataFrame({"galaxy": all_gal})
    cov["in_master"] = cov["galaxy"].isin(master_galaxies)
    cov["in_rotmod"] = cov["galaxy"].isin(rotmod_galaxies)
    cov["num_points"] = cov["galaxy"].map(rot_points).fillna(0).astype(int)
    cov["has_points"] = cov["num_points"] > 0
    return cov


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Plot SPARC rotation curves and verify data coverage.")
    ap.add_argument("--rotmod-path", default="data/sparc_rotmod_ltg.parquet", help="Path to rotmod parquet with R_kpc and Vobs_kms.")
    ap.add_argument("--master-path", default="data/sparc_master_clean.parquet", help="Path to master galaxy list parquet (optional).")
    ap.add_argument("--pred-radius-path", default="data/sparc_predictions_by_radius.csv", help="Path to predictions-by-radius file with G_required/G_ratio (CSV or Parquet).")
    ap.add_argument("--outdir", default="outputs/rotation_curves", help="Directory to write per-galaxy plots.")
    ap.add_argument("--coverage-out", default="outputs/coverage/sparc_coverage_summary.csv", help="Path to write coverage CSV.")
    ap.add_argument("--overlay-components", action="store_true", help="Overlay gas/disk/bulge and baryonic quadrature sum if available.")

    args = ap.parse_args(argv)

    rotmod_path = Path(args.rotmod_path)
    master_path = Path(args.master_path)
    pred_radius_path = Path(args.pred_radius_path)
    outdir = Path(args.outdir)
    coverage_out = Path(args.coverage_out)

    ensure_dir(outdir)
    ensure_dir(coverage_out.parent)

    rotmod = load_rotmod(rotmod_path)
    # Normalize galaxy name to string
    rotmod["galaxy"] = rotmod["galaxy"].astype(str)

    master = load_master(master_path)
    pred_by_radius = load_predictions_by_radius(pred_radius_path)

    # Plot per galaxy
    produced = 0
    skipped = 0
    for galaxy, gdf in rotmod.groupby("galaxy", sort=False):
        pred_g = None
        if pred_by_radius is not None:
            pred_g = pred_by_radius[pred_by_radius["galaxy"] == galaxy]
            if pred_g.empty:
                pred_g = None
        outpath = plot_one(
            galaxy,
            gdf,
            outdir,
            overlay_components=args.overlay_components,
            pred_gdf=pred_g,
            y2_field="G_required",
        )
        if outpath is not None:
            produced += 1
        else:
            skipped += 1

    # Coverage summary
    coverage = build_coverage(rotmod, master)
    coverage.to_csv(coverage_out, index=False)

    # Print short summary to stdout
    total_in_master = int(coverage["in_master"].sum()) if "in_master" in coverage.columns else len(coverage)
    total_in_rotmod = int(coverage["in_rotmod"].sum()) if "in_rotmod" in coverage.columns else len(coverage)
    missing = coverage[(coverage.get("in_master", True)) & (~coverage["in_rotmod"])].shape[0] if "in_master" in coverage.columns else 0

    print(
        f"Plots produced: {produced}, skipped: {skipped}.\n"
        f"Galaxies in master: {total_in_master}. In rotmod: {total_in_rotmod}. Missing in rotmod: {missing}.\n"
        f"Coverage CSV: {coverage_out}\n"
        f"Plots dir: {outdir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
