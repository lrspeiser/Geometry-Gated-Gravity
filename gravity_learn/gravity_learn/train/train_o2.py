from __future__ import annotations
import os, argparse, json, datetime

try:
    import yaml  # pyyaml
except Exception as e:  # pragma: no cover
    raise ImportError("pyyaml is required. pip install pyyaml")

from typing import Dict, Any

# Prefer reusing rigor utilities for data and SR prep
from rigor.rigor.data import load_sparc
try:
    from rigor.rigor.formula_search import prepare_training_table, run_pysr_search
except Exception:
    prepare_training_table = None
    run_pysr_search = None

from gravity_learn.models.pinn_o2 import O2PINN
from gravity_learn.models.symbolic_srx import run_o2_sr


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_includes(cfg: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    # Very small include mechanism: if cfg["includes"] is present, merge them first
    incs = cfg.get("includes", []) or []
    merged: Dict[str, Any] = {}
    for inc in incs:
        inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
        if os.path.exists(inc_path):
            merged.update(_read_yaml(inc_path))
    merged.update({k: v for k, v in cfg.items() if k != "includes"})
    return merged


def _build_gx_from_dataset(ds, limit_galaxies=8, limit_points_per_galaxy=128):
    import numpy as np
    R_all, gX_all = [], []
    count = 0
    for g in ds.galaxies:
        if count >= limit_galaxies:
            break
        mask = g.outer_mask & np.isfinite(g.R_kpc) & np.isfinite(g.Vobs_kms) & np.isfinite(g.Vbar_kms)
        R = g.R_kpc[mask]
        if R.size == 0:
            continue
        Vobs = g.Vobs_kms[mask]
        Vbar = g.Vbar_kms[mask]
        # g = V^2 / R (unit-consistent up to constants; fine for learning residual shape)
        eps = 1e-9
        g_req = (Vobs * Vobs) / (R + eps)
        g_bar = (Vbar * Vbar) / (R + eps)
        gX = g_req - g_bar
        if R.size > limit_points_per_galaxy:
            R = R[:limit_points_per_galaxy]
            gX = gX[:limit_points_per_galaxy]
        R_all.append(R)
        gX_all.append(gX)
        count += 1
    if not R_all:
        raise RuntimeError("No usable SPARC data points for O2 training.")
    R_all = np.concatenate(R_all, axis=0)
    gX_all = np.concatenate(gX_all, axis=0)
    return R_all, gX_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="Path to YAML config.")
    ap.add_argument("--mode", type=str, default=None, help="Override mode [pinn|sr]")
    args = ap.parse_args()

    cfg0 = _read_yaml(args.cfg)
    cfg = _merge_includes(cfg0, base_dir=os.path.dirname(args.cfg))
    mode = (args.mode or cfg.get("mode") or "pinn").lower()

    # Load SPARC once (paths come from included data_defaults.yaml or defaults in loader)
    ds = load_sparc(cfg.get("sparc_path", "data/sparc_rotmod_ltg.parquet"), cfg.get("master_path", "data/Rotmod_LTG/MasterSheet_SPARC.csv"))

    # Artifacts root
    root = os.path.join("gravity_learn", "experiments", mode)
    os.makedirs(root, exist_ok=True)

    if mode == "pinn":
        R, gX = _build_gx_from_dataset(ds, cfg.get("limit_galaxies", 8), cfg.get("limit_points_per_galaxy", 128))
        pinn = O2PINN(seed=int(cfg.get("seed", 0)), lr=float(cfg.get("training", {}).get("lr", 1e-2)))
        pinn.fit(R, gX,
                 epochs=int(cfg.get("training", {}).get("epochs", 60)),
                 weight_smooth=float(cfg.get("training", {}).get("weight_smooth", 1e-3)),
                 weight_positive=float(cfg.get("training", {}).get("weight_positive", 1e-3)))
        # Save artifacts
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(root, f"run_{ts}")
        pinn.save_artifacts(out_dir, R_eval=R)
        print(f"[O2 PINN] Saved artifacts to {out_dir}")
    elif mode == "sr":
        if (prepare_training_table is None) or (run_pysr_search is None):
            raise ImportError("rigor.rigor.formula_search is required for SR mode. Ensure rigor is importable and pysr installed.")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(root, f"run_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        model, out_csv = run_o2_sr(load_sparc, prepare_training_table, run_pysr_search, out_dir,
                                   niterations=int(cfg.get("sr", {}).get("niterations", 10)))
        print(f"[O2 SR] Results saved to {out_csv}")
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
