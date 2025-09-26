from __future__ import annotations
import os, json

# TODO: Implement evaluation across galaxy types (leave-one-type-out), reusing rigor loaders


def summarize_metrics(metrics: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
