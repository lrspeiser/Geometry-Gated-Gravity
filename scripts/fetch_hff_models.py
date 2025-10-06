#!/usr/bin/env python3
from __future__ import annotations
"""
Fetch HFF lens models (deflections, kappa, shear, readmes) from MAST into data/frontier/hlsp/

- Clusters: macs0416, macs0717, macs1149 (default)
- Teams: cats, williams, bradac, caminha, glafic, sharon, keeton, merten, diego (choose subset)
- Versions: auto-discover (prefers v4.1, then v4, then latest) unless explicitly provided

Files downloaded per model directory (if present):
  * *_x-arcsec-deflect.fits, *_y-arcsec-deflect.fits
  * *_kappa.fits, *_gamma1.fits, *_gamma2.fits (and *_gamma.fits if provided)
  * *_psi.fits (if present)
  * *_z02-magnif.fits and optionally z01,z04,z09 (toggle)
  * *_readme.txt

Usage examples:
  # Dry-run show what will be fetched for the three clusters from CATS/Williams
  python scripts/fetch_hff_models.py --clusters macs0416,macs0717,macs1149 --teams cats,williams --dry-run

  # Actually download CATS v4.1 and Williams v4 for MACS0416
  python scripts/fetch_hff_models.py --clusters macs0416 --teams cats:v4.1,williams:v4

  # Download Caminha v4 + CATS v4.1 for MACS0416 and CATS auto for MACS0717
  python scripts/fetch_hff_models.py --clusters macs0416,macs0717 --teams caminha:v4,cats

Notes:
- We do not alter the data; we mirror from MAST.
- If a team directory has multiple versions and none specified, we pick the highest semantic version (v4.1 > v4 > v3 ...).
- If a team directory has files directly (no version subdir), we download matched files from there.
"""

import argparse
import os
import re
import sys
from pathlib import Path
import urllib.request
import urllib.error
from html.parser import HTMLParser
from typing import List, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = ROOT / 'data' / 'frontier' / 'hlsp'
BASE = 'https://archive.stsci.edu/pub/hlsp/frontier'

CLUSTERS_DEFAULT = ['macs0416', 'macs0717', 'macs1149']
TEAMS_ALL = ['caminha','cats','williams','bradac','glafic','sharon','keeton','merten','diego','zitrin-ltm','zitrin-ltm-gauss','zitrin-nfw']
PREFER_VERSIONS = ['v4.1','v4','v3.1','v3','v2','v1']

# File patterns we consider worth fetching
FILE_PATTERNS = [
    re.compile(r'.*_x-arcsec-deflect\.fits$', re.I),
    re.compile(r'.*_y-arcsec-deflect\.fits$', re.I),
    re.compile(r'.*_x-pixels-deflect\.fits$', re.I),
    re.compile(r'.*_y-pixels-deflect\.fits$', re.I),
    re.compile(r'.*_kappa\.fits$', re.I),
    re.compile(r'.*_gamma1\.fits$', re.I),
    re.compile(r'.*_gamma2\.fits$', re.I),
    re.compile(r'.*_gamma\.fits$', re.I),
    re.compile(r'.*_psi\.fits$', re.I),
    re.compile(r'.*_z0[1249]-magnif\.fits$', re.I),
    re.compile(r'.*_readme\.txt$', re.I),
]

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: List[str] = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)

def fetch_url(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()

def list_dir(url: str) -> List[str]:
    try:
        html = fetch_url(url).decode('utf-8', errors='ignore')
    except Exception as e:
        raise RuntimeError(f'Failed to list {url}: {e}')
    p = LinkParser(); p.feed(html)
    # Filter out parent links
    return [ln for ln in p.links if ln not in ('../','./')]

_semver_re = re.compile(r'^v(\d+)(?:\.(\d+))?/?$')

def pick_version(entries: List[str], prefer: List[str]) -> Optional[str]:
    # Direct hit on preferred versions
    entry_lc = {e.strip('/').lower(): e for e in entries}
    for v in prefer:
        if v.lower() in entry_lc:
            return entry_lc[v.lower()].strip('/')
    # Otherwise pick highest vX[.Y]
    cands: List[Tuple[int,int,str]] = []
    for e in entries:
        m = _semver_re.match(e.strip('/'))
        if m:
            major = int(m.group(1)); minor = int(m.group(2) or 0)
            cands.append((major, minor, e.strip('/')))
    if cands:
        cands.sort(reverse=True)
        return cands[0][2]
    return None

def should_fetch(name: str, include_magnif_all: bool) -> bool:
    for rx in FILE_PATTERNS:
        if rx.match(name):
            if name.lower().endswith(('-z01-magnif.fits','-z04-magnif.fits','-z09-magnif.fits')) and not include_magnif_all:
                return False
            return True
    return False

def download_file(url: str, dest: Path, force: bool=False, dry_run: bool=False) -> Tuple[bool,str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return False, f'skip (exists): {dest.name}'
    if dry_run:
        return True, f'would download: {url} -> {dest}'
    try:
        with urllib.request.urlopen(url) as resp, open(dest, 'wb') as f:
            CHUNK = 1024*1024
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
        return True, f'downloaded: {dest.name}'
    except urllib.error.HTTPError as e:
        return False, f'HTTP {e.code} for {url}'
    except Exception as e:
        return False, f'error for {url}: {e}'

def parse_teams(spec: str) -> Dict[str, Optional[str]]:
    # cats:v4.1,williams:v4,bradac
    out: Dict[str, Optional[str]] = {}
    for part in (spec or '').split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            t, v = part.split(':', 1)
            out[t.strip().lower()] = v.strip()
        else:
            out[part.strip().lower()] = None
    return out

def main():
    ap = argparse.ArgumentParser(description='Fetch HFF lens models into data/frontier/hlsp')
    ap.add_argument('--clusters', type=str, default=','.join(CLUSTERS_DEFAULT), help='Comma-separated cluster ids (e.g. macs0416,macs0717,macs1149)')
    ap.add_argument('--teams', type=str, default='cats,williams,caminha', help='Comma-separated teams; use team:version to pin (e.g. cats:v4.1)')
    ap.add_argument('--include-magnif-all', action='store_true', help='Also fetch z01,z04,z09 magnification maps (default false, only z02)')
    ap.add_argument('--dest', type=str, default=str(DEST_ROOT))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    clusters = [c.strip().lower() for c in args.clusters.split(',') if c.strip()]
    teams = parse_teams(args.teams)
    dest_root = Path(args.dest)

    for cid in clusters:
        base_cluster = f"{BASE}/{cid}/models/"
        print(f"\nCluster: {cid} -> {base_cluster}")
        for team, pinned_version in teams.items():
            team_url = base_cluster + f"{team}/"
            try:
                entries = list_dir(team_url)
            except Exception as e:
                print(f"  [{team}] list failed: {e}")
                continue
            # Discover version subdirs
            versions = [e for e in entries if e.endswith('/') and e not in ('./','../')]
            version = pinned_version
            if versions:
                if not version:
                    version = pick_version(versions, PREFER_VERSIONS)
                if not version:
                    # Fall back to first subdir
                    version = versions[0].strip('/')
                model_url = team_url + f"{version}/"
            else:
                model_url = team_url
                version = 'unversioned'
            print(f"  [{team}] using version: {version} -> {model_url}")
            # List files in chosen directory
            try:
                files = [f for f in list_dir(model_url) if not f.endswith('/')]
            except Exception as e:
                print(f"    list error: {e}")
                continue
            # Filter files
            selected: List[str] = []
            for f in files:
                f_l = f.lower()
                # Always include readme
                if f_l.endswith('readme.txt'):
                    selected.append(f)
                    continue
                if not args.include_magnif_all:
                    # keep only z02 magnif of the magnif set
                    if re.match(r'.*_z02-magnif\.fits$', f_l):
                        selected.append(f)
                        continue
                if should_fetch(f, include_magnif_all=args.include_magnif_all):
                    selected.append(f)
            if not selected:
                print("    no matching files found")
                continue
            # Download
            out_dir = dest_root / cid / team / version
            for fname in selected:
                url = model_url + fname
                ok, msg = download_file(url, out_dir / fname, force=args.force, dry_run=args.dry_run)
                print("    ", msg)
            # Write a minimal meta json
            meta_path = out_dir / 'model_meta.json'
            if not args.dry_run:
                try:
                    import json
                    meta = {
                        'team': team,
                        'version': version,
                        'cluster_id': cid,
                        'source_url': model_url,
                    }
                    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
                except Exception as e:
                    print(f"    meta write error: {e}")

if __name__ == '__main__':
    sys.exit(main())
