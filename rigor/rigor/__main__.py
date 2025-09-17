
from __future__ import annotations
import argparse, json
from .compare_baselines import compare_baselines as _compare
from .fit_sparc import main as _fit_sparc_main

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=False)

    # baselines
    ap_b = sub.add_parser("baselines")
    ap_b.add_argument("--parquet", default="data/sparc_rotmod_ltg.parquet")
    ap_b.add_argument("--master", default="data/Rotmod_LTG/MasterSheet_SPARC.csv")
    ap_b.add_argument("--out_json", default="out/baselines_summary.json")
    ap_b.add_argument("--outer", action="store_true", help="Use outer points only")

    # fit (proxy to fit_sparc)
    ap_f = sub.add_parser("fit")
    ap_f.add_argument("--passthrough", nargs=argparse.REMAINDER, help="Arguments passed to rigor.fit_sparc")

    args = ap.parse_args()
    if args.cmd == "baselines" or args.cmd is None:
        # Default to baselines if no subcommand
        out = _compare(parquet=getattr(args, 'parquet', 'data/sparc_rotmod_ltg.parquet'),
                       master=getattr(args, 'master', 'data/Rotmod_LTG/MasterSheet_SPARC.csv'),
                       out_json=getattr(args, 'out_json', 'out/baselines_summary.json'),
                       use_outer=getattr(args, 'outer', True))
        print(json.dumps(out, indent=2))
    elif args.cmd == "fit":
        import sys
        # Re-invoke fit_sparc main with remaining args
        sys.argv = ["python", "-m", "rigor.fit_sparc"] + (args.passthrough or [])
        _fit_sparc_main()

if __name__ == "__main__":
    main()
