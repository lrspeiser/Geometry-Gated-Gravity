#!/usr/bin/env python3
from __future__ import annotations
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys as _sys

# Make rigor importable
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigor.baselines import mond_simple

# ---- Helpers to normalize columns in predictions_by_radius

def _first(cols, *names):
    for n in names:
        if n in cols: return n
    return None

def normalize_pred_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    m = {
        'gal': _first(cols,'gal_id','galaxy','Galaxy'),
        'r':   _first(cols,'r_kpc','R_kpc','R'),
        'vbar':_first(cols,'vbar_kms','Vbar_kms','Vbar'),
        'vobs':_first(cols,'v_obs_kms','Vobs_kms','V_obs_kms','Vobs'),
        'outer':_first(cols,'is_outer','outer_flag'),
    }
    missing = [k for k in ['gal','r','vbar','vobs'] if m[k] is None]
    if missing:
        raise KeyError(f"Missing required columns: {missing}; have {cols}")
    out = pd.DataFrame({
        'gal_id': df[m['gal']],
        'r_kpc': pd.to_numeric(df[m['r']], errors='coerce'),
        'vbar_kms': pd.to_numeric(df[m['vbar']], errors='coerce'),
        'v_obs_kms': pd.to_numeric(df[m['vobs']], errors='coerce'),
    })
    if m['outer']:
        out['is_outer'] = df[m['outer']].astype(bool)
    return out

def infer_outer_mask(df: pd.DataFrame) -> np.ndarray:
    if 'is_outer' in df.columns:
        return df['is_outer'].astype(bool).to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    for gid, sub in df.groupby('gal_id'):
        idx = sub.sort_values('r_kpc').index
        k = max(1, int(0.3*len(idx)))
        mask[idx[-k:]] = True
    return mask

# ---- BTFR and RAR utilities

def v_flat_per_gal(df_pred: pd.DataFrame, vcol: str, frac_outer=0.3) -> pd.DataFrame:
    rows=[]
    for gid, g in df_pred.groupby('gal_id'):
        g = g.sort_values('r_kpc')
        k = max(1, int(frac_outer*len(g)))
        vflat = float(np.median(g[vcol].tail(k).values))
        rows.append((gid, vflat))
    return pd.DataFrame(rows, columns=['gal_id','vflat_kms'])

def rar_table(df_pred: pd.DataFrame, vcol: str) -> pd.DataFrame:
    r = df_pred['r_kpc'].to_numpy()
    vobs2 = df_pred['v_obs_kms'].to_numpy()**2
    vbar2 = df_pred['vbar_kms'].to_numpy()**2
    g_obs = vobs2 / np.maximum(r, 1e-9)
    g_bar = vbar2 / np.maximum(r, 1e-9)
    g_mod = df_pred[vcol].to_numpy()**2 / np.maximum(r, 1e-9)
    return pd.DataFrame({'g_bar': g_bar, 'g_obs': g_obs, 'g_mod': g_mod})

# ---- Plotting

def plot_bar_medians(medians: dict[str,float], out_png: Path):
    names = list(medians.keys())
    vals = [medians[k] for k in names]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(names, vals, color=['C3','C4','C0'])
    for i,v in enumerate(vals):
        ax.text(i, v+1.0, f"{v:.1f}%", ha='center', va='bottom')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Median % closeness (outer)')
    ax.set_title('Outer-region accuracy')
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_btfr(btfr_mond, btfr_lt, btfr_mp, mass_table, out_png: Path):
    # Join masses if available
    mass = mass_table[['galaxy','M_bary_Msun']].rename(columns={'galaxy':'gal_id'})
    j_m = btfr_mond.merge(mass, on='gal_id', how='left')
    j_l = btfr_lt.merge(mass, on='gal_id', how='left')
    j_p = btfr_mp.merge(mass, on='gal_id', how='left')
    # Drop NaNs
    j_m = j_m.dropna(subset=['M_bary_Msun']); j_l=j_l.dropna(subset=['M_bary_Msun']); j_p=j_p.dropna(subset=['M_bary_Msun'])
    fig, ax = plt.subplots(figsize=(6,5))
    for df,c,lbl in [(j_m,'C3','MOND'),(j_l,'C0','LogTail'),(j_p,'C4','MuPhi')]:
        if len(df)==0: continue
        x=np.log10(df['M_bary_Msun'].to_numpy()); y=np.log10(np.maximum(df['vflat_kms'].to_numpy(),1e-3))
        ax.scatter(x, y, s=12, alpha=0.5, label=lbl, color=c)
    ax.set_xlabel('log10 M_b (Msun)')
    ax.set_ylabel('log10 v_flat (km/s)')
    ax.legend()
    ax.set_title('BTFR comparison (no fit lines)')
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_rar(rar_mond, rar_lt, rar_mp, out_png: Path):
    # Bin by g_bar and plot median g_mod
    def binned(df, nb=20):
        gb = df['g_bar'].to_numpy(); gm = df['g_mod'].to_numpy();
        qs = np.quantile(gb[~np.isnan(gb)], np.linspace(0,1,nb+1))
        xs=[]; ys=[]
        for i in range(nb):
            mask = (gb>=qs[i]) & (gb<qs[i+1])
            if mask.sum()<5: continue
            xs.append(np.median(gb[mask])); ys.append(np.median(gm[mask]))
        return np.array(xs), np.array(ys)
    fig, ax = plt.subplots(figsize=(6,5))
    # Background: g_obs vs g_bar
    gb = rar_mond['g_bar'].to_numpy(); go = rar_mond['g_obs'].to_numpy()
    ax.scatter(gb, go, s=5, alpha=0.15, color='gray', label='Observed (per-radius)')
    for df, c, lbl in [(rar_mond,'C3','MOND'),(rar_lt,'C0','LogTail'),(rar_mp,'C4','MuPhi')]:
        x,y = binned(df)
        if len(x)==0: continue
        ax.plot(x, y, '-', color=c, lw=2, label=f'{lbl} (binned medians)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('g_bar (km^2/s^2/kpc)'); ax.set_ylabel('g (km^2/s^2/kpc)')
    ax.legend()
    ax.set_title('RAR comparison')
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out_png, dpi=160)
    plt.close(fig)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Plots: MOND vs LogTail vs MuPhi')
    ap.add_argument('--pred_csv', default=str(Path('out/analysis/type_breakdown/predictions_by_radius.csv')))
    ap.add_argument('--extended_json', default=str(Path('out/analysis/type_breakdown/accuracy_comparison_extended.json')))
    ap.add_argument('--summary_new', default=str(Path('out/analysis/type_breakdown/summary_logtail_muphi.json')))
    ap.add_argument('--btfr_lt', default=str(Path('out/analysis/type_breakdown/btfr_logtail.csv')))
    ap.add_argument('--btfr_mp', default=str(Path('out/analysis/type_breakdown/btfr_muphi.csv')))
    ap.add_argument('--rar_lt', default=str(Path('out/analysis/type_breakdown/rar_logtail.csv')))
    ap.add_argument('--rar_mp', default=str(Path('out/analysis/type_breakdown/rar_muphi.csv')))
    ap.add_argument('--mass_table', default=str(Path('data/sparc_predictions_by_galaxy.csv')))
    ap.add_argument('--out_dir', default=str(Path('out/analysis/type_breakdown')))
    ap.add_argument('--a0_json', default=str(Path('out/baselines_summary.json')))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # 1) Medians bar
    ext = json.loads(Path(args.extended_json).read_text())['models']
    summ = json.loads(Path(args.summary_new).read_text())
    med_mond = float(ext['MOND']['median']) if 'MOND' in ext else float('nan')
    med_lt = float(summ['LogTail']['median'])
    med_mp = float(summ['MuPhi']['median'])
    plot_bar_medians({'MOND':med_mond, 'LogTail':med_lt, 'MuPhi':med_mp}, out_dir/'compare_medians_mond_logtail_muphi.png')

    # 2) Compute MOND velocities from preds
    raw = pd.read_csv(args.pred_csv)
    dfp = normalize_pred_cols(raw)
    # a0
    a0 = 1.2e-10
    a0_path = Path(args.a0_json)
    if a0_path.exists():
        try:
            a0 = float(json.loads(a0_path.read_text())['MOND_simple']['a0_hat'])
        except Exception:
            pass
    v_mond = mond_simple(dfp['vbar_kms'].to_numpy(), dfp['r_kpc'].to_numpy(), a0=a0)
    dfp = dfp.copy(); dfp['v_MOND_kms'] = v_mond

    # 3) BTFR
    btfr_mond = v_flat_per_gal(dfp.rename(columns={'v_MOND_kms':'v_mond_kms'}), 'v_mond_kms')
    btfr_lt = pd.read_csv(args.btfr_lt)
    btfr_mp = pd.read_csv(args.btfr_mp)
    mass_table = pd.read_csv(args.mass_table)
    plot_btfr(btfr_mond, btfr_lt, btfr_mp, mass_table, out_dir/'btfr_compare_mond_logtail_muphi.png')

    # 4) RAR
    rar_mond = rar_table(dfp.rename(columns={'v_MOND_kms':'vcol'}).assign(vcol=dfp['v_MOND_kms']), 'v_MOND_kms')
    rar_lt = pd.read_csv(args.rar_lt)
    rar_mp = pd.read_csv(args.rar_mp)
    plot_rar(rar_mond, rar_lt, rar_mp, out_dir/'rar_compare_mond_logtail_muphi.png')

    print('Wrote plots:')
    print(out_dir/'compare_medians_mond_logtail_muphi.png')
    print(out_dir/'btfr_compare_mond_logtail_muphi.png')
    print(out_dir/'rar_compare_mond_logtail_muphi.png')