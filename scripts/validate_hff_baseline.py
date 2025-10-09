#!/usr/bin/env python3
from __future__ import annotations
"""
Validate HFF self-consistency: published θE vs θE computed from the team's own maps.

Reads:
- data/frontier/hff_published_theta_E.csv
- data/frontier/gold_standard/report_thetaE.csv (θE_crit from accepted maps)

Outputs a PASS/FAIL per (cluster,team,version) with 5% threshold.
"""
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'data' / 'frontier' / 'hff_published_theta_E.csv'
REP = ROOT / 'data' / 'frontier' / 'gold_standard' / 'report_thetaE.csv'

THRESH_PCT = 5.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pub', default=str(PUB))
    ap.add_argument('--rep', default=str(REP))
    args = ap.parse_args()

    if not Path(args.pub).exists():
        print(f'Missing published CSV: {args.pub}')
        sys.exit(1)
    if not Path(args.rep).exists():
        print(f'Missing report CSV: {args.rep}')
        sys.exit(1)

    df_pub = pd.read_csv(args.pub)
    df_rep = pd.read_csv(args.rep)
    for col in ('cluster','team','version'):
        if col not in df_pub.columns:
            print(f'published CSV missing column: {col}')
            sys.exit(1)
    df_pub['cluster'] = df_pub['cluster'].str.lower()
    df_rep['cluster_id'] = df_rep['cluster_id'].str.lower()

    # Left join published -> computed
    m = df_pub.merge(df_rep, left_on=['cluster','team','version'], right_on=['cluster_id','team','version'], how='left')

    rows = []
    for _, r in m.iterrows():
        cid = r['cluster']
        team = r['team']
        ver = r['version']
        te_pub = r.get('theta_E_arcsec', float('nan'))
        te_comp = r.get('thetaE_crit_arcsec', float('nan'))
        if pd.isna(te_pub) or pd.isna(te_comp):
            status = 'MISSING'
            err_pct = float('nan')
        else:
            err_pct = abs(te_comp - te_pub) / te_pub * 100.0
            status = 'PASS' if err_pct <= THRESH_PCT else 'FAIL'
        rows.append((cid, team, ver, te_pub, te_comp, err_pct, status))

    print('\nHFF self-consistency validation (threshold: ±5%):')
    print('cluster, team, version, thetaE_published, thetaE_computed, pct_error, status')
    for row in rows:
        cid, team, ver, te_pub, te_comp, err_pct, status = row
        print(f"{cid}, {team}, {ver}, {te_pub}, {te_comp}, {err_pct}, {status}")

if __name__ == '__main__':
    main()
