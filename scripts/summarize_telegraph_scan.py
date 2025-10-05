#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import glob

def main():
    root = Path(__file__).resolve().parents[1]
    pat = str(root / 'data' / 'clash' / 'processed' / 'eval' / 'er_telegraph_metrics_*.json')
    rows = []
    for fp in glob.glob(pat):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            overall = obj.get('overall', {})
            mae = overall.get('mae_arcsec')
            mape = overall.get('mape_percent')
            rows.append((fp, mae, mape))
        except Exception:
            pass
    rows = [r for r in rows if r[1] is not None]
    rows.sort(key=lambda x: x[1])
    out = root / 'data' / 'clash' / 'processed' / 'telegraph' / 'scan_summary.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write('file,mae_arcsec,mape_percent\n')
        for fp, mae, mape in rows:
            f.write(f'{Path(fp).name},{mae},{mape}\n')
    print('Top 5 (lowest MAE):')
    for fp, mae, mape in rows[:5]:
        print(f'  {Path(fp).name}: MAE={mae:.3f} arcsec, MAPE={mape:.2f}%')

if __name__ == '__main__':
    main()
