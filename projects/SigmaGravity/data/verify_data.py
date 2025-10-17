#!/usr/bin/env python3
"""
Data verifier for SigmaGravity replication.
- Verifies presence of SPARC and cluster data
- Computes MD5 checksums for auditing
- Writes a small report JSON under projects/SigmaGravity/provenance/data_md5.json
"""
import sys, os, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_JSON = ROOT / 'projects' / 'SigmaGravity' / 'provenance' / 'data_md5.json'

SPARC_DIR = ROOT / 'external_data' / 'Rotmod_LTG'
SPARC_MASTER = SPARC_DIR / 'MasterSheet_SPARC.mrt'

CLUSTER_CATALOG = ROOT / 'data' / 'clusters' / 'master_catalog.csv'


def md5sum(p: Path) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    report = {
        'sparc': {
            'present': False,
            'master_path': str(SPARC_MASTER),
            'master_md5': None,
            'rotmod_count': 0
        },
        'clusters': {
            'catalog_path': str(CLUSTER_CATALOG),
            'catalog_present': False,
            'catalog_md5': None
        }
    }

    # SPARC checks
    if SPARC_MASTER.exists():
        report['sparc']['present'] = True
        report['sparc']['master_md5'] = md5sum(SPARC_MASTER)
        # Count rotmod files
        if SPARC_DIR.exists():
            report['sparc']['rotmod_count'] = sum(1 for _ in SPARC_DIR.rglob('*_rotmod.dat'))
    else:
        print(f"WARN: SPARC MasterSheet not found: {SPARC_MASTER}")

    # Cluster catalog
    if CLUSTER_CATALOG.exists():
        report['clusters']['catalog_present'] = True
        report['clusters']['catalog_md5'] = md5sum(CLUSTER_CATALOG)
    else:
        print(f"FATAL: Cluster catalog not found: {CLUSTER_CATALOG}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print("Data verification summary:")
    print(json.dumps(report, indent=2))

    # Exit non-zero if required inputs missing
    if not report['clusters']['catalog_present']:
        sys.exit(2)
    # SPARC optional for cluster pipeline, but warn if missing
    sys.exit(0)


if __name__ == '__main__':
    main()
