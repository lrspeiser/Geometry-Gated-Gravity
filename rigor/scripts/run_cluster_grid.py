# -*- coding: utf-8 -*-
"""
rigor/scripts/run_cluster_grid.py

Run a small parameter grid for the cluster PDE solver on selected clusters,
collect metrics, and copy plots to tagged filenames for review.

Outputs:
- root-m/out/pde_clusters/grid_summary.csv (all runs)
- root-m/out/pde_clusters/<CL>_grid/*.json (metrics per run, tagged)
- root-m/out/pde_clusters/<CL>_grid/*.png  (plot per run, tagged)

Usage:
  py -u rigor\scripts\run_cluster_grid.py
  py -u rigor\scripts\run_cluster_grid.py --clusters ABELL_0426 ABELL_1689
  py -u rigor\scripts\run_cluster_grid.py --S0 7e-6 8.8e-6 1.1e-5 --gsat 2000 2500 3000 --lam 0 0.001 0.002
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess as sp
from pathlib import Path
from shutil import copy2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repo root
RUNNER = ROOT/"root-m"/"pde"/"run_cluster_pde.py"
OUTBASE = ROOT/"root-m"/"out"/"pde_clusters"
DATABASE = ROOT/"data"/"clusters"

DEFAULT_RMAX = {
    "ABELL_0426": (600.0, 600.0),
    "ABELL_1689": (900.0, 900.0),
}


def float_tag(x: float) -> str:
    # Compact and filesystem-safe
    return ("%.6g" % float(x)).replace(".", "p").replace("-", "m")


def run_one(cluster: str, S0: float, gsat: float, lam: float) -> dict:
    Rmax, Zmax = DEFAULT_RMAX.get(cluster, (800.0, 800.0))
    cldir = DATABASE/cluster
    clump_csv = cldir/"clump_profile.csv"
    stars_csv = cldir/"stars_profile.csv"

    cmd = [
        "py", "-u", str(RUNNER),
        "--cluster", cluster,
        "--S0", str(S0),
        "--rc_kpc", "20",
        "--g0_kms2_per_kpc", "1200",
        "--NR", "128", "--NZ", "128",
        "--Rmax", str(Rmax), "--Zmax", str(Zmax),
        "--clump_profile_csv", str(clump_csv),
        "--stars_csv", str(stars_csv),
        "--bc_robin_lambda", str(lam),
        "--use_saturating_mobility",
        "--gsat_kms2_per_kpc", str(gsat),
        "--n_sat", "2",
    ]
    print("[grid] RUN:", " ".join(cmd))
    cp = sp.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"runner failed for {cluster} S0={S0} gsat={gsat} lam={lam}")

    # Parse med_ratio from stdout
    med_ratio = None
    m = re.search(r"median g_phi/g_N on kT radii = ([0-9eE+\-.]+)", cp.stdout)
    if m:
        med_ratio = float(m.group(1))

    # Read metrics.json
    mpath = OUTBASE/cluster/"metrics.json"
    with open(mpath, "r") as f:
        metrics = json.load(f)
    temp_err = float(metrics.get("temp_median_frac_err", float("nan")))

    # Copy plot and metrics to tagged filenames to avoid overwrite
    tag = f"S0_{float_tag(S0)}__gsat_{float_tag(gsat)}__lam_{float_tag(lam)}"
    gdir = OUTBASE/f"{cluster}_grid"
    gdir.mkdir(parents=True, exist_ok=True)
    src_png = OUTBASE/cluster/"cluster_pde_results.png"
    if src_png.exists():
        copy2(src_png, gdir/f"plot_{tag}.png")
    copy2(mpath, gdir/f"metrics_{tag}.json")

    return {
        "cluster": cluster,
        "S0": S0,
        "gsat": gsat,
        "lam": lam,
        "temp_median_frac_err": temp_err,
        "med_ratio": med_ratio,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", nargs="*", default=["ABELL_0426", "ABELL_1689"])
    ap.add_argument("--S0", nargs="*", type=float, default=[7.5e-6, 8.8e-6, 1.05e-5])
    ap.add_argument("--gsat", nargs="*", type=float, default=[2000.0, 2500.0])
    ap.add_argument("--lam", nargs="*", type=float, default=[0.0, 1.0e-3, 2.0e-3])
    args = ap.parse_args()

    rows = []
    for cl in args.clusters:
        for S0 in args.S0:
            for g in args.gsat:
                for lam in args.lam:
                    try:
                        rows.append(run_one(cl, S0, g, lam))
                    except Exception as e:
                        print("[grid] ERROR:", e)

    if rows:
        df = pd.DataFrame(rows)
        df.sort_values(["cluster", "temp_median_frac_err"], inplace=True)
        OUTBASE.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTBASE/"grid_summary.csv", index=False)
        print("[grid] wrote", OUTBASE/"grid_summary.csv")
        print(df.groupby("cluster").head(5).to_string(index=False))

if __name__ == "__main__":
    main()
