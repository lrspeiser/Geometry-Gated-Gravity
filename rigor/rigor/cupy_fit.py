#!/usr/bin/env python3
from __future__ import annotations
import os, json, math, random, argparse
import numpy as np
from typing import Dict, Any, List, Tuple

from .data import load_sparc
from .metrics import summarize_outer
from .cupy_utils import xp, HAS_CUPY, to_xp, xi_shell_logistic_radius, xi_logistic_density, xi_combined_radius_density, gate_fixed, gate_learned, vpred_from_xi

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def eval_config(ds, xi_name: str, gating: str, params: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
    # Compute avg_mean_off across outer regions for given params
    total_mean = 0.0
    n_gal = 0
    total_outer_pts = 0
    for g in ds.galaxies:
        R = to_xp(g.R_kpc)
        Vbar = to_xp(g.Vbar_kms)
        Sigma = to_xp(g.Sigma_bar) if g.Sigma_bar is not None else None
        Mbar = g.Mbar_Msun or 1e10
        # xi core
        if xi_name == "logistic_density":
            xi = xi_logistic_density(R, Sigma, Mbar,
                                     xi_max=params.get("xi_max", 3.0),
                                     lnSigma_c=params.get("lnSigma_c", math.log(10.0)),
                                     width_sigma=params.get("width_sigma", 0.6),
                                     n_sigma=params.get("n_sigma", 1.0))
        elif xi_name == "combined_radius_density":
            xi = xi_combined_radius_density(R, Sigma, Mbar,
                                            xi_cap=params.get("xi_cap", 6.0),
                                            xi_max_r=params.get("xi_max_r", 3.0), lnR0_base=params.get("lnR0_base", math.log(3.0)), width=params.get("width", 0.6), alpha_M=params.get("alpha_M", -0.2),
                                            xi_max_d=params.get("xi_max_d", 3.0), lnSigma_c=params.get("lnSigma_c", math.log(10.0)), width_sigma=params.get("width_sigma", 0.6), n_sigma=params.get("n_sigma", 1.0))
        else:
            xi = xi_shell_logistic_radius(R, Mbar,
                                          xi_max=params.get("xi_max", 3.0),
                                          lnR0_base=params.get("lnR0_base", math.log(3.0)),
                                          width=params.get("width", 0.6),
                                          alpha_M=params.get("alpha_M", -0.2))
        # gating
        if gating == "fixed":
            xi = gate_fixed(xi, R, gate_R_kpc=params.get("gate_R_kpc", 3.0), gate_width=params.get("gate_width", 0.4))
        elif gating == "learned":
            xi = gate_learned(xi, R, Mbar,
                               lnR_gate_base=params.get("lnR_gate_base", math.log(3.0)),
                               width_gate=params.get("width_gate", 0.4),
                               alpha_gate_M=params.get("alpha_gate_M", -0.2))
        # prediction
        Vpred = vpred_from_xi(Vbar, xi)
        mask = np.asarray(g.outer_mask, dtype=bool)
        # move to cpu for summarize
        Vpred_cpu = np.asarray(Vpred.get() if HAS_CUPY else Vpred)
        res = summarize_outer(np.asarray(g.Vobs_kms), Vpred_cpu, np.asarray(g.eVobs_kms), mask)
        total_mean += float(res["mean_off"])  # per-galaxy avg percent off in outer
        total_outer_pts += int(mask.sum())
        n_gal += 1
    avg_mean_off = total_mean / max(n_gal, 1)
    return avg_mean_off, {"avg_mean_off": avg_mean_off, "galaxies": n_gal, "outer_points": total_outer_pts}


def random_params(xi_name: str, gating: str, rng: random.Random) -> Dict[str, float]:
    p = {}
    # xi core
    p["xi_max"] = rng.uniform(1.5, 8.0)
    if xi_name == "logistic_density":
        p["lnSigma_c"] = math.log(10.0) + rng.uniform(-2.0, 2.0)  # ~ [e^-2*10, e^2*10]
        p["width_sigma"] = rng.uniform(0.1, 1.5)
        p["n_sigma"] = rng.uniform(0.5, 5.0)
    elif xi_name == "combined_radius_density":
        # radius branch
        p["xi_max_r"] = rng.uniform(1.5, 8.0)
        p["lnR0_base"] = math.log(3.0) + rng.uniform(-1.2, 1.2)
        p["width"] = rng.uniform(0.1, 1.5)
        p["alpha_M"] = rng.uniform(-0.8, 0.3)
        # density branch
        p["xi_max_d"] = rng.uniform(1.2, 8.0)
        p["lnSigma_c"] = math.log(10.0) + rng.uniform(-2.0, 2.0)
        p["width_sigma"] = rng.uniform(0.1, 1.5)
        p["n_sigma"] = rng.uniform(0.5, 5.0)
        # cap
        p["xi_cap"] = rng.uniform(2.0, 10.0)
    else:
        p["lnR0_base"] = math.log(3.0) + rng.uniform(-1.2, 1.2)
        p["width"] = rng.uniform(0.1, 1.5)
        p["alpha_M"] = rng.uniform(-0.8, 0.3)
    # gating
    if gating == "fixed":
        p["gate_R_kpc"] = rng.uniform(1.0, 8.0)
        p["gate_width"] = rng.uniform(0.1, 1.0)
    elif gating == "learned":
        p["lnR_gate_base"] = math.log(3.0) + rng.uniform(-1.2, 1.2)
        p["width_gate"] = rng.uniform(0.1, 1.0)
        p["alpha_gate_M"] = rng.uniform(-0.8, 0.3)
    return p


def refine_around(best: Dict[str, float], scale: float, xi_name: str, gating: str, rng: random.Random) -> Dict[str, float]:
    p = {}
    for k, v in best.items():
        if isinstance(v, (int, float)):
            p[k] = float(v) + rng.uniform(-scale, scale)
    # Keep within reasonable bounds
    if "xi_max" in p: p["xi_max"] = float(np.clip(p["xi_max"], 1.2, 10.0))
    if xi_name == "logistic_density":
        if "width_sigma" in p: p["width_sigma"] = float(np.clip(p["width_sigma"], 0.05, 2.0))
        if "n_sigma" in p: p["n_sigma"] = float(np.clip(p["n_sigma"], 0.3, 8.0))
    elif xi_name == "combined_radius_density":
        if "width" in p: p["width"] = float(np.clip(p["width"], 0.05, 2.0))
        if "width_sigma" in p: p["width_sigma"] = float(np.clip(p["width_sigma"], 0.05, 2.0))
        if "n_sigma" in p: p["n_sigma"] = float(np.clip(p["n_sigma"], 0.3, 8.0))
        if "xi_cap" in p: p["xi_cap"] = float(np.clip(p["xi_cap"], 1.5, 12.0))
    else:
        if "width" in p: p["width"] = float(np.clip(p["width"], 0.05, 2.0))
    if gating == "fixed":
        if "gate_width" in p: p["gate_width"] = float(np.clip(p["gate_width"], 0.05, 2.0))
        if "gate_R_kpc" in p: p["gate_R_kpc"] = float(np.clip(p["gate_R_kpc"], 0.3, 15.0))
    elif gating == "learned":
        if "width_gate" in p: p["width_gate"] = float(np.clip(p["width_gate"], 0.05, 2.0))
        if "alpha_gate_M" in p: p["alpha_gate_M"] = float(np.clip(p["alpha_gate_M"], -1.5, 1.0))
    return p


def overlay_quick(ds, xi_name: str, gating: str, params: Dict[str, float], out_dir: str, max_n: int = 12):
    if not HAS_MPL:
        return
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for g in ds.galaxies[:max_n]:
        R = to_xp(g.R_kpc)
        Vbar = to_xp(g.Vbar_kms)
        Mbar = g.Mbar_Msun or 1e10
        if xi_name == "logistic_density":
            Xi = xi_logistic_density(R, g.Sigma_bar, Mbar,
                                     xi_max=params.get("xi_max", 3.0),
                                     lnSigma_c=params.get("lnSigma_c", math.log(10.0)),
                                     width_sigma=params.get("width_sigma", 0.6),
                                     n_sigma=params.get("n_sigma", 1.0))
        elif xi_name == "combined_radius_density":
            Xi = xi_combined_radius_density(R, g.Sigma_bar, Mbar,
                                            xi_cap=params.get("xi_cap", 6.0),
                                            xi_max_r=params.get("xi_max_r", 3.0), lnR0_base=params.get("lnR0_base", math.log(3.0)), width=params.get("width", 0.6), alpha_M=params.get("alpha_M", -0.2),
                                            xi_max_d=params.get("xi_max_d", 3.0), lnSigma_c=params.get("lnSigma_c", math.log(10.0)), width_sigma=params.get("width_sigma", 0.6), n_sigma=params.get("n_sigma", 1.0))
        else:
            Xi = xi_shell_logistic_radius(R, Mbar,
                                          xi_max=params.get("xi_max", 3.0),
                                          lnR0_base=params.get("lnR0_base", math.log(3.0)),
                                          width=params.get("width", 0.6),
                                          alpha_M=params.get("alpha_M", -0.2))
        if gating == "fixed":
            Xi = gate_fixed(Xi, R, gate_R_kpc=params.get("gate_R_kpc", 3.0), gate_width=params.get("gate_width", 0.4))
        elif gating == "learned":
            Xi = gate_learned(Xi, R, Mbar,
                               lnR_gate_base=params.get("lnR_gate_base", math.log(3.0)),
                               width_gate=params.get("width_gate", 0.4),
                               alpha_gate_M=params.get("alpha_gate_M", -0.2))
        Vpred = vpred_from_xi(Vbar, Xi)
        # to cpu for plotting
        Rn = np.asarray(R.get() if HAS_CUPY else R)
        Vobs = np.asarray(g.Vobs_kms)
        Vbarn = np.asarray(Vbar.get() if HAS_CUPY else Vbar)
        Vpredn = np.asarray(Vpred.get() if HAS_CUPY else Vpred)
        is_outer = np.asarray(g.outer_mask, dtype=bool)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5), tight_layout=True)
        ax.scatter(Rn[~is_outer], Vobs[~is_outer], s=18, color='tab:blue', alpha=0.6, label='Observed (inner)')
        ax.scatter(Rn[is_outer], Vobs[is_outer], s=22, color='tab:blue', edgecolor='k', linewidths=0.4, label='Observed (outer)')
        ax.plot(Rn, Vpredn, color='tab:red', linewidth=2.0, label='Model Vpred')
        ax.plot(Rn, Vbarn, color='tab:green', linestyle='--', linewidth=1.6, alpha=0.9, label='GR baseline')
        ax.set_xlabel('Radius R (kpc)'); ax.set_ylabel('Speed (km/s)'); ax.set_title(f"{g.name} — xi={xi_name}, gate={gating}")
        handles, labels = ax.get_legend_handles_labels(); uniq = dict(zip(labels, handles)); ax.legend(uniq.values(), uniq.keys(), fontsize=9, framealpha=0.9)
        fpath = os.path.join(out_dir, f"{g.name.replace(' ', '_')}.png")
        fig.savefig(fpath, dpi=140); plt.close(fig)
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/sparc_rotmod_ltg.parquet")
    ap.add_argument("--master", default="data/Rotmod_LTG/MasterSheet_SPARC.csv")
    ap.add_argument("--outdir", default="out/cupy_search")
    ap.add_argument("--xi", nargs="*", default=["shell_logistic_radius", "logistic_density", "combined_radius_density"])
    ap.add_argument("--gating", nargs="*", default=["fixed", "learned"])
    ap.add_argument("--random", type=int, default=60, help="Random samples per variant")
    ap.add_argument("--refine", type=int, default=30, help="Refinement samples around best")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not HAS_CUPY:
        print("[warn] CuPy not available; running with NumPy fallback on CPU. Install cupy-cuda12x for GPU.")

    ds = load_sparc(args.parquet, args.master)
    os.makedirs(args.outdir, exist_ok=True)

    rng = random.Random(args.seed)
    leaderboard: List[Dict[str, Any]] = []

    for xi_name in args.xi:
        for gating in args.gating:
            best_score = float("inf")
            best_params: Dict[str, float] = {}
            # random search
            for _ in range(args.random):
                p = random_params(xi_name, gating, rng)
                score, summary = eval_config(ds, xi_name, gating, p)
                if score < best_score:
                    best_score = score; best_params = p.copy(); best_summary = summary
            # refine around best
            for _ in range(args.refine):
                p = refine_around(best_params, scale=0.2, xi_name=xi_name, gating=gating, rng=rng)
                score, summary = eval_config(ds, xi_name, gating, p)
                if score < best_score:
                    best_score = score; best_params = p.copy(); best_summary = summary
            # write variant results
            tag = f"xi_{xi_name}__gate_{gating}"
            vdir = os.path.join(args.outdir, tag)
            os.makedirs(vdir, exist_ok=True)
            with open(os.path.join(vdir, "best_params.json"), "w") as f:
                json.dump({"params": best_params, "summary": best_summary}, f, indent=2)
            # overlays
            figs_dir = os.path.join(vdir, "figs")
            overlay_quick(ds, xi_name, gating, best_params, figs_dir, max_n=12)
            leaderboard.append({"xi": xi_name, "gating": gating, **best_summary})
            print(f"Variant {tag}: avg_mean_off={best_summary['avg_mean_off']:.3f} over {best_summary['galaxies']} galaxies")

    # rank & write leaderboard
    leaderboard_sorted = sorted(leaderboard, key=lambda r: r.get("avg_mean_off", float("inf")))
    with open(os.path.join(args.outdir, "leaderboard.json"), "w") as f:
        json.dump(leaderboard_sorted, f, indent=2)
    import csv
    with open(os.path.join(args.outdir, "leaderboard.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["xi","gating","avg_mean_off","outer_points","galaxies"])
        w.writeheader(); [w.writerow({k: row.get(k, "") for k in ["xi","gating","avg_mean_off","outer_points","galaxies"]}) for row in leaderboard_sorted]
    print("Wrote leaderboard to", os.path.join(args.outdir, "leaderboard.json"))

if __name__ == "__main__":
    main()