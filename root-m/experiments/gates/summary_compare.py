# -*- coding: utf-8 -*-
"""
root-m/experiments/gates/summary_compare.py

Aggregate comparisons for clusters and SPARC between gated variants and prior bests.
Writes a consolidated JSON report.
"""
import json
from pathlib import Path
import argparse

def load_json(p):
    return json.loads(Path(p).read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='root-m/out/experiments/gates/summary_compare.json')
    args = ap.parse_args()

    report = {}

    # Previous bests (from earlier runs in repo)
    prev = {}
    # LogTail clusters (if present)
    try:
        prev['ABELL_0426_logtail'] = load_json('root-m/out/clusters/ABELL_0426/cluster_logtail_metrics.json')
    except Exception:
        pass
    try:
        prev['ABELL_1689_logtail'] = load_json('root-m/out/clusters/ABELL_1689/cluster_logtail_metrics.json')
    except Exception:
        pass
    # PDE clusters (unit-correct kernel best)
    try:
        prev['ABELL_0426_pde'] = load_json('root-m/out/pde_clusters/ABELL_0426/metrics.json')
    except Exception:
        pass
    try:
        prev['ABELL_1689_pde'] = load_json('root-m/out/pde_clusters/ABELL_1689/metrics.json')
    except Exception:
        pass
    report['previous'] = prev

    # New gated clusters (rho-aware)
    gated = {}
    try:
        gated['ABELL_0426_rhoaware'] = load_json('root-m/out/experiments/gates/rhoaware/ABELL_0426/metrics.json')
    except Exception:
        pass
    try:
        gated['ABELL_1689_rhoaware'] = load_json('root-m/out/experiments/gates/rhoaware/ABELL_1689/metrics.json')
    except Exception:
        pass
    report['gated'] = gated

    # SPARC baselines and gated
    sparc = {}
    try:
        sparc['rootm_soft_cv'] = load_json('root-m/out/sparc_soft_v2/summary_rootm_soft.json')
    except Exception:
        pass
    try:
        sparc['pde_cv_subset'] = load_json('root-m/out/pde_sparc_axisym_cv_v2/cv_summary.json')
    except Exception:
        pass
    try:
        sparc['gated_rhoaware'] = load_json('root-m/out/experiments/gates/sparc/rhoaware/summary_rootm_gate.json')
    except Exception:
        pass
    report['sparc'] = sparc

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
