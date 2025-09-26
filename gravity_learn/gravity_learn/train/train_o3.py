from __future__ import annotations
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=False, help="Path to YAML config (future use)")
    args = ap.parse_args()
    print("O3 training is a TODO: add cluster ΔΣ ingestion and training here.")

if __name__ == "__main__":
    main()
