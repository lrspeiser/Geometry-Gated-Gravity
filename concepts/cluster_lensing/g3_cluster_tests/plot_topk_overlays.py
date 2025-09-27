# -*- coding: utf-8 -*-
"""
Plot top-K O3 slip overlays per cluster from saved diagnostics.
Reads concepts/cluster_lensing/g3_cluster_tests/outputs/o3_slip_topk_diags.json and
saves per-cluster overlays to concepts/cluster_lensing/g3_cluster_tests/outputs/o3_slip_topk_<CLUSTER>.png.
Optionally filter clusters or limit to first N ranks.
"""
from __future__ import annotations
import json
from pathlib import Path
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diags', type=str, default='concepts/cluster_lensing/g3_cluster_tests/outputs/o3_slip_topk_diags.json')
    ap.add_argument('--outdir', type=str, default='concepts/cluster_lensing/g3_cluster_tests/outputs')
    ap.add_argument('--clusters', type=str, default='', help='Comma-separated list of cluster names to plot (default: all)')
    ap.add_argument('--topn', type=int, default=0, help='If >0, only plot top N ranks per cluster')
    args = ap.parse_args()

    diag_path = Path(args.diags)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(diag_path.read_text())
    clusters = list(data.get('clusters', {}).keys())
    if args.clusters:
        wanted = set([c.strip() for c in args.clusters.split(',') if c.strip()])
        clusters = [c for c in clusters if c in wanted]

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    for name in clusters:
        entries = data['clusters'][name]
        if args.topn > 0:
            entries = [e for e in entries if int(e.get('rank', 0)) <= int(args.topn)]
        # sort by rank
        entries = sorted(entries, key=lambda e: int(e.get('rank', 99999)))
        fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=120)
        for ent in entries:
            r = np.asarray(ent['r_kpc'], float)
            kb = np.asarray(ent['kbar'], float)
            label = f"#{ent.get('rank','?')} (score {ent.get('score', float('nan')):.2f})"
            ax.plot(r, kb, lw=1.0, label=label)
        ax.axhline(1.0, color='k', lw=0.8, ls='--')
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('k̄')
        ax.set_title(f'Top-{len(entries)} O3 slip configs: {name}')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"o3_slip_topk_{name}.png")
        plt.close(fig)


if __name__ == '__main__':
    main()
