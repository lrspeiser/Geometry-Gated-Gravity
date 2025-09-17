
from __future__ import annotations
import argparse, json
from .compare_baselines import compare_baselines as _compare

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/sparc_rotmod_ltg.parquet")
    ap.add_argument("--master", default="data/Rotmod_LTG/MasterSheet_SPARC.csv")
    ap.add_argument("--out_json", default="out/baselines_summary.json")
    ap.add_argument("--outer", action="store_true", help="Use outer points only")    
    args = ap.parse_args()
    out = _compare(parquet=args.parquet, master=args.master, out_json=args.out_json, use_outer=args.outer)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
