# -*- coding: utf-8 -*-
"""
rigor/scripts/run_cluster_s0_sweep.py

Sweep a set of S0 values (single-law) for both clusters using clump/stars CSVs
and the simple PDE (no A1/A2/Robin). Choose the S0 that minimizes the average
median fractional temperature error across clusters.

Outputs:
- root-m/out/pde_clusters/s0_sweep_summary.csv
"""
from __future__ import annotations
import argparse
import json
import subprocess as sp
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT/"root-m"/"pde"/"run_cluster_pde.py"
OUTBASE = ROOT/"root-m"/"out"/"pde_clusters"
DATABASE = ROOT/"data"/"clusters"

DEFAULTS = {
    "ABELL_0426": dict(Rmax=600.0, Zmax=600.0),
    "ABELL_1689": dict(Rmax=900.0, Zmax=900.0),
}


def run_one(cluster: str, S0: float) -> float:
    d = DEFAULTS.get(cluster, dict(Rmax=800.0, Zmax=800.0))
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
        "--Rmax", str(d["Rmax"]), "--Zmax", str(d["Zmax"]),
        "--clump_profile_csv", str(clump_csv),
        "--stars_csv", str(stars_csv),
        "--bc_robin_lambda", "0.0",
    ]
    cp = sp.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        print(cp.stdout); print(cp.stderr)
        raise RuntimeError(f"run failed for {cluster} S0={S0}")
    mpath = OUTBASE/cluster/"metrics.json"
    with open(mpath, "r") as f:
        metrics = json.load(f)
    return float(metrics.get("temp_median_frac_err", float("nan")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", nargs="*", default=["ABELL_0426","ABELL_1689"]) 
    ap.add_argument("--S0", nargs="*", type=float, default=[8.0e-6, 8.8e-6, 9.6e-6])
    args = ap.parse_args()

    rows = []
    for s0 in args.S0:
        errs = []
        for cl in args.clusters:
            err = run_one(cl, s0)
            rows.append(dict(cluster=cl, S0=s0, temp_median_frac_err=err))
            errs.append(err)
        avg = sum(errs)/len(errs)
        rows.append(dict(cluster="__AVERAGE__", S0=s0, temp_median_frac_err=avg))
        print(f"S0={s0:.3e} avg temp median frac err = {avg:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTBASE/"s0_sweep_summary.csv", index=False)
    print("wrote", OUTBASE/"s0_sweep_summary.csv")

if __name__ == "__main__":
    main()
