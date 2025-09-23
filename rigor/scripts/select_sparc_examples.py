#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    in_csv = Path('data/sparc_predictions_by_radius.csv')
    out_json = Path('out/analysis/type_breakdown/sparc_example_galaxies.json')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(in_csv)
    agg = df.groupby('galaxy').agg(n=('R_kpc','size'), rmax=('R_kpc','max')).reset_index()
    # Top 6 by n desc then rmax desc
    ex = agg.sort_values(['n','rmax'], ascending=[False, False]).head(6)
    names = ex['galaxy'].tolist()
    # Map to last radius from the main table (more precise than groupby max if filtering needed later)
    r_last_map = {}
    for g in names:
        sub = df[df['galaxy'] == g]
        r_last_map[g] = float(np.nanmax(sub['R_kpc'].to_numpy(dtype=float)))
    out = {'galaxies': names, 'r_last_kpc': r_last_map}
    out_json.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(out_json.resolve())

if __name__ == '__main__':
    main()