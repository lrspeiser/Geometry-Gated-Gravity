#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

def main():
    repo = Path(__file__).resolve().parents[1]
    data = repo / 'data'
    all_pq = data / 'sparc_all_tables.parquet'
    clean_pq = data / 'sparc_master_clean.parquet'
    if not all_pq.exists() or not clean_pq.exists():
        print('Parquet outputs not found; run make build-parquet first')
        sys.exit(1)
    all_df = pd.read_parquet(all_pq)
    clean = pd.read_parquet(clean_pq)
    # Basic checks
    assert all_df.shape[0] > 50, f"Unexpected few rows in all tables: {all_df.shape}"
    for col in ['galaxy','galaxy_key','T','D']:
        assert col in all_df.columns, f"Missing column {col} in all_tables"
    assert clean.shape[0] > 2, f"Unexpected few galaxies in clean: {clean.shape}"
    for col in ['galaxy','galaxy_key','M_bary']:
        assert col in clean.columns, f"Missing column {col} in clean"
    print('Parquet build checks passed.')

if __name__ == '__main__':
    main()