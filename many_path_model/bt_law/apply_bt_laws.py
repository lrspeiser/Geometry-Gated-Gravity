
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply the learned B/T laws to produce a parameter set for a given galaxy.
"""
import argparse, json
from pathlib import Path
from bt_laws import load_theta, eval_all_laws, morph_to_bt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=Path, default=Path("many_path_model/bt_law/bt_law_params.json"))
    ap.add_argument("--galaxy", type=str, default=None, help="Optional galaxy name (for logging only)")
    ap.add_argument("--B_T", type=float, default=None, help="Bulge-to-total (if known)")
    ap.add_argument("--hubble_type", type=str, default=None, help="Alternative: Hubble type (e.g., Scd, Sb)")
    ap.add_argument("--type_group", type=str, default=None, help="Optional: late/intermediate/early")
    ap.add_argument("--output", type=Path, default=None, help="Output path for params JSON")
    args = ap.parse_args()

    theta = load_theta(args.theta)

    if args.B_T is not None:
        B = float(args.B_T)
    else:
        B = morph_to_bt(args.hubble_type, args.type_group)

    params = eval_all_laws(B, theta)
    params["source"] = "bt_law"
    params["B_T_used"] = B
    if args.galaxy:
        params["galaxy"] = args.galaxy

    if args.output:
        out = args.output
    else:
        out = Path(f"results/bt_law_params/params_{args.galaxy or 'unknown'}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out, "w") as f:
        json.dump(params, f, indent=2)
    print(json.dumps(params, indent=2))
    print(f"\nSaved to: {out}")

if __name__ == "__main__":
    main()
