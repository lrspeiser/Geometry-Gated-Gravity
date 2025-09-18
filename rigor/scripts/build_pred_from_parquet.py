#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def build_predictions(parquet_path: Path, out_csv: Path, frac_outer: float = 0.3) -> None:
    df = pd.read_parquet(parquet_path)
    # Expect columns: galaxy, R_kpc, Vobs_kms, Vgas_kms, Vdisk_kms, Vbul_kms
    required = ['galaxy','R_kpc','Vobs_kms','Vgas_kms','Vdisk_kms','Vbul_kms']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Parquet {parquet_path} is missing required columns: {missing}")
    Vbar2 = (pd.to_numeric(df['Vgas_kms'], errors='coerce')**2 +
             pd.to_numeric(df['Vdisk_kms'], errors='coerce')**2 +
             pd.to_numeric(df.get('Vbul_kms', 0.0), errors='coerce')**2)
    Vbar = np.sqrt(np.clip(Vbar2.to_numpy(), 0.0, None))
    out = pd.DataFrame({
        'galaxy': df['galaxy'].astype(str),
        'R_kpc': pd.to_numeric(df['R_kpc'], errors='coerce'),
        'Vobs_kms': pd.to_numeric(df['Vobs_kms'], errors='coerce'),
        'Vbar_kms': Vbar,
    })
    # Outer mask: top frac_outer of points per galaxy by radius
    mask = np.zeros(len(out), dtype=bool)
    for gid, sub in out.groupby('galaxy'):
        idx = sub.sort_values('R_kpc').index
        k = max(1, int(frac_outer*len(idx)))
        mask[idx[-k:]] = True
    out['is_outer'] = mask
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} rows={len(out)}")


def main():
    ap = argparse.ArgumentParser(description='Build base predictions CSV (galaxy,R_kpc,Vobs_kms,Vbar_kms,is_outer) from SPARC rotmod parquet.')
    ap.add_argument('--parquet', default=str(Path('data')/'sparc_rotmod_ltg.parquet'))
    ap.add_argument('--out', default=str(Path('out')/'analysis'/'type_breakdown'/'predictions_rotmod_base.csv'))
    ap.add_argument('--frac_outer', type=float, default=0.3)
    args = ap.parse_args()
    build_predictions(Path(args.parquet), Path(args.out), frac_outer=float(args.frac_outer))


if __name__ == '__main__':
    main()