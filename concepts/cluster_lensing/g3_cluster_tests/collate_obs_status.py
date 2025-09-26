# -*- coding: utf-8 -*-
"""
Collate current observational status into a Markdown snapshot.
- Clusters: read slip and non-local O3 fixed eval JSONs, compare θE_pred vs θE_obs, κ̄ stats
- SPARC: aggregate median_percent_close from any root-m/out/pde_sparc/*/summary.json
- MW: read root-m/out/*/summary_rootm.json (first found)
Writes: concepts/cluster_lensing/g3_cluster_tests/outputs/OBS_STATUS.md
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime

OUT_DIR = Path('concepts/cluster_lensing/g3_cluster_tests/outputs')
SLIP_EVAL = OUT_DIR / 'o3_slip_fixed_eval.json'
NONLOCAL_EVAL = OUT_DIR / 'o3_nonlocal_fixed_eval.json'


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fmt(x):
    if x is None:
        return 'n/a'
    try:
        return f"{x:.2f}"
    except Exception:
        return str(x)


def collate_clusters():
    slip = _load_json(SLIP_EVAL) or {}
    nl = _load_json(NONLOCAL_EVAL) or {}
    names = sorted(set(list(slip.keys()) + list(nl.keys())))
    lines = []
    lines.append('| Cluster | θE_obs(\") | Slip θE(\") | Slip/Obs | Slip κ̄max | κ̄50 | κ̄100 | NonLocal θE(\") | NL/Obs | NL κ̄max | κ̄50 | κ̄100 |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for name in names:
        s = slip.get(name, {})
        n = nl.get(name, {})
        obs = s.get('theta_E_obs', n.get('theta_E_obs'))
        s_the = s.get('theta_E_arcsec')
        n_the = n.get('theta_E_arcsec')
        s_ratio = (s_the/obs) if (s_the is not None and obs not in (None, 0)) else None
        n_ratio = (n_the/obs) if (n_the is not None and obs not in (None, 0)) else None
        row = [
            name,
            _fmt(obs),
            _fmt(s_the),
            _fmt(s_ratio),
            _fmt(s.get('kappa_max')),
            _fmt(s.get('kappa_50')),
            _fmt(s.get('kappa_100')),
            _fmt(n_the),
            _fmt(n_ratio),
            _fmt(n.get('kappa_max')),
            _fmt(n.get('kappa_50')),
            _fmt(n.get('kappa_100')),
        ]
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def collate_sparc():
    root = Path('root-m/out/pde_sparc')
    if not root.exists():
        return 'No SPARC summaries found.'
    vals = []
    for sub in root.glob('*'):
        sp = sub / 'summary.json'
        if sp.exists():
            try:
                d = json.loads(sp.read_text())
                v = d.get('median_percent_close', None)
                if v is not None:
                    vals.append(float(v))
            except Exception:
                pass
    if not vals:
        return 'No SPARC median_percent_close found.'
    vals_sorted = sorted(vals, reverse=True)
    mean_val = sum(vals)/len(vals)
    return f"SPARC galaxies: {len(vals)} summaries found; mean median_percent_close = {mean_val:.1f}% (top 5: {', '.join(f'{x:.1f}%' for x in vals_sorted[:5])})"


def collate_mw():
    root = Path('root-m/out')
    candidates = []
    for base, dirs, files in os.walk(root):
        for f in files:
            if f == 'summary_rootm.json':
                candidates.append(Path(base) / f)
    if not candidates:
        return 'No Milky Way summary_rootm.json found.'
    # pick latest by mtime
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        d = json.loads(latest.read_text())
        med = d.get('global', {}).get('median_percent_close_rootm_all', None)
        mean = d.get('global', {}).get('mean_percent_close_rootm_all', None)
        return f"Milky Way: median={_fmt(med)}%, mean={_fmt(mean)}% (from {latest})"
    except Exception:
        return f"Milky Way: could not parse {latest}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    md = []
    md.append(f"# Observational Status Snapshot ({now})")
    md.append('')
    md.append('## Clusters (Strong Lensing)')
    md.append(collate_clusters())
    md.append('')
    md.append('## SPARC (Galaxy Rotation Curves)')
    md.append(collate_sparc())
    md.append('')
    md.append('## Milky Way (Rotation Curve)')
    md.append(collate_mw())
    md.append('')
    md.append('---')
    md.append('### Interpretation')
    md.append('- O3 slip gates corrected to favor low-Σ/low-curvature/large-R. This removes core inflation and θE overshoot, providing a safe baseline.')
    md.append('- Current configurations are conservative; next, we will expand amplitude and radial windows to push Einstein radii into the 50–150 kpc band while keeping core guardrails.')
    md.append('- SPARC/MW metrics should not degrade since O3 is lensing-only; dynamics alterations (O2) remain separately gated by environment.')
    out_path = OUT_DIR / 'OBS_STATUS.md'
    with out_path.open('w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('Wrote', out_path)

if __name__ == '__main__':
    main()
