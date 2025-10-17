#!/usr/bin/env python3
import sys, os, json, hashlib, csv, time, subprocess
from pathlib import Path

SEARCH_DEFAULTS = [
    Path(__file__).resolve().parents[2] / 'output',                      # repo-level outputs
    Path(__file__).resolve().parents[1] / 'output'                       # SigmaGravity outputs
]

OUT_JSON = Path(__file__).parent / 'manifests_index.json'
OUT_CSV = Path(__file__).parent / 'manifests_index.csv'


def git_commit_hash_or_none(cwd: Path) -> str | None:
    try:
        sha = subprocess.check_output(['git', '--no-pager', 'rev-parse', 'HEAD'], cwd=str(cwd), stderr=subprocess.DEVNULL, text=True).strip()
        return sha
    except Exception:
        return None


def find_posterior_artifacts(search_dirs: list[Path]) -> list[Path]:
    results = []
    for root in search_dirs:
        if not root.exists():
            continue
        for p in root.rglob('*.npz'):
            results.append(p)
        for p in root.rglob('trace.nc'):
            results.append(p)
    return results


def load_manifest(npz_path: Path):
    try:
        import numpy as np
        data = np.load(npz_path, allow_pickle=True)
        if 'manifest' not in data:
            return None, None
        try:
            manifest = json.loads(str(data['manifest'].item()))
        except Exception:
            return None, None
        return manifest, data
    except Exception:
        return None, None


def validate_catalog_md5(manifest: dict) -> bool:
    try:
        cp = Path(manifest['catalog_path'])
        with open(cp, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        return (md5 == manifest.get('catalog_md5'))
    except Exception:
        return False


def main(argv: list[str]):
    # Inputs: optional list of directories to search
    search_dirs = [Path(s) for s in argv] if argv else SEARCH_DEFAULTS
    search_dirs = [d for d in search_dirs if d.exists()]

    repo_root = Path(__file__).resolve().parents[2]
    commit = git_commit_hash_or_none(repo_root)

    rows = []
for art in find_posterior_artifacts(search_dirs):
        manifest, data = (None, None)
        if art.suffix == '.npz':
            manifest, data = load_manifest(art)
        else:
            # try to load sibling manifest.json
            man = art.with_name('manifest.json')
            if man.exists():
                try:
                    manifest = json.loads(man.read_text())
                except Exception:
                    manifest = None
        if manifest is None:
            continue
        ok_md5 = validate_catalog_md5(manifest)
        rows.append({
            'posterior_path': str(art.resolve()),
            'mtime': int(art.stat().st_mtime),
            'run_id': manifest.get('run_id'),
            'train_clusters': ','.join(manifest.get('train_clusters', [])),
            'tiers': ','.join(str(t) for t in manifest.get('tiers', [])),
            'H0': manifest.get('cosmology', {}).get('H0'),
            'Om0': manifest.get('cosmology', {}).get('Om0'),
            'pzsource': manifest.get('physics', {}).get('pzsource'),
            'bcg': manifest.get('physics', {}).get('bcg'),
            'triaxial': manifest.get('physics', {}).get('triaxial'),
            'kernel_norm': manifest.get('kernel', {}).get('norm'),
            'mass_scaling': manifest.get('kernel', {}).get('mass_scaling'),
            'gamma_prior': manifest.get('kernel', {}).get('gamma_prior'),
            'catalog_md5': manifest.get('catalog_md5'),
            'catalog_path': manifest.get('catalog_path'),
            'catalog_md5_valid': bool(ok_md5),
            'code_commit': commit,
        })

    # Sort by mtime desc
    rows.sort(key=lambda r: r['mtime'], reverse=True)

    # Write JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({'generated_at': int(time.time()), 'count': len(rows), 'rows': rows}, f, indent=2)

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            w.writeheader()
            w.writerows(rows)

    print(f"Indexed {len(rows)} runs")
    print(f"  JSON: {OUT_JSON}")
    print(f"  CSV:  {OUT_CSV}")


if __name__ == '__main__':
    main(sys.argv[1:])
