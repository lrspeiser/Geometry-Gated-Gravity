#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'data' / 'frontier' / 'gold_standard' / 'report_thetaE.csv'
OBS = ROOT / 'data' / 'clash' / 'einstein_radii_observed.csv'
OUT = ROOT / 'data' / 'frontier' / 'gold_standard' / 'report_accepted_vs_observed.csv'

GOLD = {'macs0416','macs0717','macs1149'}

def main():
    if not REP.exists():
        print(f'missing report: {REP}')
        sys.exit(1)
    df_rep = pd.read_csv(REP)
    df_obs = pd.read_csv(OBS)
    df_obs['cluster_id'] = df_obs['cluster_id'].str.lower()
    df_rep['cluster_id'] = df_rep['cluster_id'].str.lower()
    df_rep = df_rep[df_rep['cluster_id'].isin(GOLD)].copy()
    df = df_rep.merge(df_obs[['cluster_id','theta_E_observed_arcsec']], on='cluster_id', how='left')
    df['abs_err_arcsec'] = df['thetaE_crit_arcsec'] - df['theta_E_observed_arcsec']
    df['pct_err'] = 100.0 * df['abs_err_arcsec'] / df['theta_E_observed_arcsec']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df[['cluster_id','team','version','theta_E_observed_arcsec','thetaE_crit_arcsec','abs_err_arcsec','pct_err']].to_string(index=False))
    print(f'\nWrote {OUT}')

if __name__ == '__main__':
    main()
