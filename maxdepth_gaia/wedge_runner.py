#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from typing import List


def run_one(phi_bins: int, phi_index: int, base_args: List[str], outputs_dir: str) -> dict:
    # Set a unique saveplot per wedge to avoid clobbering figures
    saveplot = os.path.join(outputs_dir, f"mw_rotation_curve_maxdepth_phi{phi_bins}_{phi_index}.png")
    cmd = [sys.executable, "-m", "maxdepth_gaia.run_pipeline", "--use_source", "auto",
           "--phi_bins", str(phi_bins), "--phi_bin_index", str(phi_index),
           "--saveplot", saveplot]
    cmd.extend(base_args)
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)

    fit_path = os.path.join(outputs_dir, "fit_params.json")
    with open(fit_path, "r") as f:
        fit = json.load(f)

    # Extract a compact summary row
    b = fit.get("boundary", {})
    sw = fit.get("saturated_well", {})
    nfw = fit.get("nfw", {})
    bo = fit.get("baryons_only", {})

    row = dict(
        phi_bins=phi_bins,
        phi_index=phi_index,
        R_boundary=b.get("R_boundary"),
        R_unc_lo=b.get("R_unc_lo"),
        R_unc_hi=b.get("R_unc_hi"),
        sw_xi=(sw.get("params", {}) or {}).get("xi"),
        sw_R_s=(sw.get("params", {}) or {}).get("R_s"),
        sw_m=(sw.get("params", {}) or {}).get("m"),
        sw_gate_dR=(sw.get("params", {}) or {}).get("gate_width_kpc"),
        sw_v_flat=sw.get("v_flat"),
        sw_chi2=sw.get("chi2"), sw_bic=sw.get("bic"),
        nfw_V200=(nfw.get("params", {}) or {}).get("V200"),
        nfw_c=(nfw.get("params", {}) or {}).get("c"),
        nfw_chi2=nfw.get("chi2"), nfw_bic=nfw.get("bic"),
        bo_chi2=bo.get("chi2"), bo_bic=bo.get("bic"),
    )
    return row


def main():
    ap = argparse.ArgumentParser(description="Run run_pipeline across azimuthal wedges and summarize results.")
    ap.add_argument("--phi_bins", type=int, default=4)
    ap.add_argument("--outputs_dir", default=os.path.join("maxdepth_gaia", "outputs"))
    # Common run_pipeline args (pass-through)
    ap.add_argument("--baryon_model", default="mw_multi")
    ap.add_argument("--ad_correction", action="store_true")
    ap.add_argument("--rmin", type=float, default=3.0)
    ap.add_argument("--rmax", type=float, default=20.0)
    ap.add_argument("--nbins", type=int, default=24)
    ap.add_argument("--inner_fit_min", type=float, default=3.0)
    ap.add_argument("--inner_fit_max", type=float, default=8.0)
    ap.add_argument("--boundary_method", default="both")
    ap.add_argument("--gate_width_kpc", type=float, default=None)
    ap.add_argument("--fix_m", type=float, default=None)
    ap.add_argument("--eta_rs", type=float, default=None)
    # Filtering controls for wedges
    ap.add_argument("--zmax", type=float, default=0.5)
    ap.add_argument("--sigma_vmax", type=float, default=30.0)
    ap.add_argument("--vRmax", type=float, default=40.0)
    args = ap.parse_args()

    os.makedirs(args.outputs_dir, exist_ok=True)

    base_args: List[str] = ["--baryon_model", args.baryon_model,
                            "--rmin", str(args.rmin), "--rmax", str(args.rmax), "--nbins", str(args.nbins),
                            "--inner_fit_min", str(args.inner_fit_min), "--inner_fit_max", str(args.inner_fit_max),
                            "--boundary_method", args.boundary_method,
                            "--zmax", str(args.zmax), "--sigma_vmax", str(args.sigma_vmax), "--vRmax", str(args.vRmax)]
    if args.ad_correction:
        base_args.append("--ad_correction")
    if args.gate_width_kpc is not None:
        base_args.extend(["--gate_width_kpc", str(args.gate_width_kpc)])
    if args.fix_m is not None:
        base_args.extend(["--fix_m", str(args.fix_m)])
    if args.eta_rs is not None:
        base_args.extend(["--eta_rs", str(args.eta_rs)])

    rows = []
    for i in range(int(args.phi_bins)):
        try:
            rows.append(run_one(args.phi_bins, i, base_args, args.outputs_dir))
        except subprocess.CalledProcessError as e:
            # Record failure row with diagnostics
            rows.append(dict(phi_bins=args.phi_bins, phi_index=i, error=str(e)))
        time.sleep(0.1)

    # Write summary CSV and JSON
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outputs_dir, f"wedge_summary_phi{args.phi_bins}.csv")
    json_path = os.path.join(args.outputs_dir, f"wedge_summary_phi{args.phi_bins}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"Saved wedge summary to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
