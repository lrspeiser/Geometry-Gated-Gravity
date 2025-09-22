# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Try to import plotting libs; fall back to minimal if seaborn missing
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:
    import seaborn as sns  # type: ignore
    sns.set_context("talk")
    sns.set_style("whitegrid")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "out" / "analysis" / "type_breakdown"
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Unit constants for MOND convenience
KPC_M = 3.085677581e19
A0_MOND = 1.2e-10  # m/s^2 (typical)


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def savefig(path: Path, tight=True):
    if tight:
        plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def pick_example_galaxies(df: pd.DataFrame, n=6) -> list[str]:
    # prefer galaxies with many points and broad radius coverage
    agg = df.groupby("galaxy").agg(n=("R_kpc","size"), rmax=("R_kpc","max")).reset_index()
    # sort by n descending then rmax
    agg = agg.sort_values(["n","rmax"], ascending=[False, False])
    # ensure variety by picking alternates across the list
    picks = []
    for name in agg["galaxy"].tolist():
        if name not in picks:
            picks.append(name)
        if len(picks) >= n:
            break
    return picks


def mond_velocity_simple(r_kpc: np.ndarray, vbar_kms: np.ndarray, a0=A0_MOND) -> np.ndarray:
    # g_N = vbar^2 / r (convert to SI)
    r_m = np.asarray(r_kpc) * KPC_M
    vbar_ms = np.asarray(vbar_kms) * 1e3
    gN = (vbar_ms**2) / np.clip(r_m, 1.0, None)
    # "simple" MOND: g = 0.5*(gN + sqrt(gN^2 + 4 gN a0))
    g = 0.5*(gN + np.sqrt(gN**2 + 4.0*gN*a0))
    v_ms = np.sqrt(g * r_m)
    return v_ms / 1e3


def plot_rotation_curve_overlays():
    path = OUT_DIR / "predictions_with_LogTail_MuPhi.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    need = ["galaxy","R_kpc","Vobs_kms","Vbar_kms","v_LogTail_kms"]
    if not ensure_cols(df, need):
        return None
    galaxies = pick_example_galaxies(df, n=6)

    # Optional alternate LogTail params for overlay
    alt_cfg = FIG_DIR / "logtail_alt.json"
    alt = None
    if alt_cfg.exists():
        try:
            alt = json.loads(alt_cfg.read_text())
            # expect keys: v0, rc, r0, delta
            _ = (float(alt.get('v0')), float(alt.get('rc')), float(alt.get('r0')), float(alt.get('delta')))
        except Exception:
            alt = None

    def _vlog(r, vbar, p):
        vbar2 = np.maximum(np.asarray(vbar), 0.0)**2
        rr = np.maximum(np.asarray(r), 1e-9)
        S = 0.5*(1.0 + np.tanh((rr - float(p['r0']))/max(float(p['delta']), 1e-6)))
        tail = (float(p['v0'])**2) * (rr/(rr + max(float(p['rc']), 1e-6))) * S
        return np.sqrt(np.maximum(vbar2 + np.maximum(tail, 0.0), 0.0))

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), sharex=False, sharey=False)
    axes = axes.ravel()

    # Proxies to ensure a complete legend regardless of per-panel presence
    legend_proxies = [
        Line2D([0], [0], marker='o', linestyle='None', color="#444444", label="Observed"),
        Line2D([0], [0], linestyle='--', color="#1f77b4", label="GR (baryons)"),
        Line2D([0], [0], linestyle='-.', color="#2ca02c", label="MOND (simple)"),
        Line2D([0], [0], linestyle=':', color="#9467bd", label="Isothermal plateau (outer median)"),
        Line2D([0], [0], linestyle='-', color="#d62728", label="G³ (global)"),
        Line2D([0], [0], linestyle='--', color="#d62728", label="G³ (alt)"),
    ]

    for ax, gal in zip(axes, galaxies):
        sub = df[df["galaxy"] == gal].sort_values("R_kpc")
        r = sub["R_kpc"].to_numpy()
        vobs = sub["Vobs_kms"].to_numpy()
        vbar = sub["Vbar_kms"].to_numpy()
        vlog = sub["v_LogTail_kms"].to_numpy()
        # MOND (simple mu) from baryons
        vmond = mond_velocity_simple(r, vbar)
        # DM-like isothermal: use outer 30% observed median as constant
        k = max(3, int(0.3*len(sub)))
        vhalo = float(np.median(sub.tail(k)["Vobs_kms"])) if len(sub) else np.nan

        ax.scatter(r, vobs, s=12, c="#444444", alpha=0.7, label="Observed")
        ax.plot(r, vbar, lw=2, ls="--", c="#1f77b4", label="GR (baryons)")
        ax.plot(r, vmond, lw=2, ls="-.", c="#2ca02c", label="MOND (simple)")
        if np.isfinite(vhalo):
            ax.plot([np.nanmin(r), np.nanmax(r)], [vhalo, vhalo], lw=2, ls=":", c="#9467bd", label="Isothermal plateau (outer median)")
        ax.plot(r, vlog, lw=2.5, c="#d62728", label="G³ (global)")
        if alt is not None:
            try:
                v2 = _vlog(r, vbar, alt)
                ax.plot(r, v2, lw=2.0, ls='--', c="#d62728", alpha=0.9, label="G³ (alt)")
            except Exception:
                pass
        ax.set_title(str(gal))
        ax.set_xlabel("R (kpc)")
        ax.set_ylabel("v (km/s)")
        # Start axes just above zero
        try:
            xmin = float(np.nanmin(r))
            ax.set_xlim(left=max(1e-3, xmin))
            ymin = float(np.nanmin([np.nanmin(v) for v in [vobs, vbar, vmond, vlog] if np.isfinite(np.nanmin(v))]))
            ax.set_ylim(bottom=max(1.0, 0.8*ymin))
        except Exception:
            pass

    # Shared legend outside the plot area to avoid overlap
    fig.subplots_adjust(bottom=0.18)
    fig.legend(handles=legend_proxies,
               labels=[h.get_label() for h in legend_proxies],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.06))
    fig.suptitle("Rotation curves: observed vs GR(baryons), MOND, Isothermal plateau, and G³ (disk surrogate)")
    savefig(FIG_DIR / "rc_overlays_examples_v2.png")


def plot_rar_obs_vs_model():
    path = OUT_DIR / "rar_logtail.csv"
    if not path.exists():
        return None
    rar = pd.read_csv(path)
    need = ["g_bar","g_obs","g_mod"]
    if not ensure_cols(rar, need):
        return None
    # Clean
    d = rar.replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()
    x = np.log10(np.clip(d["g_bar"].to_numpy(), 1e-15, None))
    y_obs = np.log10(np.clip(d["g_obs"].to_numpy(), 1e-15, None))
    y_mod = np.log10(np.clip(d["g_mod"].to_numpy(), 1e-15, None))

    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.hexbin(x, y_obs, gridsize=60, cmap="Greys", bins="log", mincnt=5)

    # Median curves
    nb = 25
    bins = np.linspace(x.min(), x.max(), nb+1)
    xc = 0.5*(bins[1:]+bins[:-1])
    yobs_med = np.full(nb, np.nan)
    ymod_med = np.full(nb, np.nan)
    for i in range(nb):
        m = (x>=bins[i]) & (x<bins[i+1])
        if np.any(m):
            yobs_med[i] = np.median(y_obs[m])
            ymod_med[i] = np.median(y_mod[m])
    ax.plot(xc, yobs_med, lw=2.5, c="#1f77b4", label="Observed median")
    ax.plot(xc, ymod_med, lw=2.5, c="#d62728", label="G³ median (disk surrogate)")

    lims = [min(x.min(), y_obs.min(), y_mod.min()), max(x.max(), y_obs.max(), y_mod.max())]
    ax.plot(lims, lims, ls=":", c="#888888", lw=1)
    ax.set_xlabel(r"log10 g_bar (kpc km^2 s^-2 kpc^-1)")
    ax.set_ylabel(r"log10 g (same units)")
    ax.set_title("RAR: observed vs G³ (disk surrogate)")
    ax.legend(frameon=False)
    savefig(FIG_DIR / "rar_obs_vs_model_v2.png")


def plot_btfr_two_panel():
    obs_path = OUT_DIR / "btfr_observed.csv"
    fit_obs = OUT_DIR / "btfr_observed_fit.json"
    mod_path = OUT_DIR / "btfr_logtail.csv"
    fit_mod = OUT_DIR / "btfr_logtail_fit.json"
    if not (obs_path.exists() and mod_path.exists()):
        return None
    obs = pd.read_csv(obs_path)
    mod = pd.read_csv(mod_path)

    def get_vcol(df: pd.DataFrame, pref: list[str]) -> str|None:
        for c in pref:
            if c in df.columns:
                return c
        return None

    v_obs_col = get_vcol(obs, ["Vflat_obs_kms","vflat_obs_kms","vflat_kms"])
    v_mod_col = get_vcol(mod, ["vflat_kms","Vflat_kms"]) or "vflat_kms"

    # panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    def do_panel(ax, df, vcol, fit_json):
        d = df.dropna(subset=[vcol, "M_bary_Msun"]).copy()
        d = d[(d[vcol] > 0) & (d["M_bary_Msun"] > 0)]
        x = np.log10(d[vcol]); y = np.log10(d["M_bary_Msun"])
        ax.scatter(x, y, s=16, alpha=0.6)
        # Fit line
        alpha = None; beta = None
        if fit_json and fit_json.exists():
            try:
                f = json.loads(fit_json.read_text())
                alpha = float(f.get("form_Mb_vs_v",{}).get("alpha", np.nan))
                beta = float(f.get("form_Mb_vs_v",{}).get("beta", np.nan))
            except Exception:
                pass
        if alpha is None or not math.isfinite(alpha):
            # OLS
            A = np.vstack([x, np.ones_like(x)]).T
            alpha, beta = np.linalg.lstsq(A, y, rcond=None)[0]
        xv = np.linspace(x.min(), x.max(), 50)
        yv = alpha*xv + beta
        ax.plot(xv, yv, c="#d62728", lw=2.5)
        return alpha

    a1 = do_panel(ax1, obs, v_obs_col or "vflat_kms", fit_obs)
    ax1.set_title(f"Observed BTFR (slope α≈{a1:.2f})")
    ax1.set_xlabel("log10 v_flat (km/s)")
    ax1.set_ylabel("log10 M_b (Msun)")

    a2 = do_panel(ax2, mod, v_mod_col, fit_mod)
    ax2.set_title(f"G³ BTFR (disk surrogate; slope α≈{a2:.2f})")
    ax2.set_xlabel("log10 v_flat (km/s)")

    savefig(FIG_DIR / "btfr_two_panel_v2.png")


def plot_lensing_shapes():
    shp = OUT_DIR / "lensing_logtail_shapes.csv"
    cmp = OUT_DIR / "lensing_logtail_comparison.json"
    if not shp.exists():
        return None
    d = pd.read_csv(shp)
    R = d.iloc[:,0].to_numpy(dtype=float)
    DS = d.iloc[:,1].to_numpy(dtype=float)
    # filter strictly positive to avoid log(0)
    msk = (R > 0) & (DS > 0)
    R = R[msk]; DS = DS[msk]
    fig, ax = plt.subplots(figsize=(7,6))
    ax.loglog(R, DS, lw=2.5, c="#d62728", label="G³ ΔΣ (disk surrogate)")
    # annotate slope
    x = np.log10(R); y = np.log10(DS)
    m, b = np.polyfit(x, y, 1)
    ax.text(0.05, 0.05, f"slope ≈ {m:.2f}", transform=ax.transAxes)
    # annotate 50/100 kpc if comparison present
    if cmp.exists():
        try:
            cj = json.loads(cmp.read_text())
            for Rq in (50, 100):
                key = f"pred_DeltaSigma_{Rq}kpc"
                if key in cj:
                    ax.scatter([Rq], [cj[key]], c="#1f77b4")
                    ax.annotate(f"{Rq} kpc", (Rq, cj[key]), textcoords="offset points", xytext=(5,5))
        except Exception:
            pass
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel(r"ΔΣ (Msun/kpc²)")
    ax.set_title("Galaxy–galaxy lensing shape (G³ disk surrogate)")
    ax.legend(frameon=False)
    savefig(FIG_DIR / "lensing_logtail_shape_v2.png")


def plot_cv_medians():
    path = OUT_DIR / "cv" / "cv_summary.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8,5.5))
    x = np.arange(len(d))
    ax.bar(x, d["LogTail_test_median"], width=0.6, label="G³ test (LogTail surrogate)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in d["fold"]])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Median closeness (%)")
    ax.set_title("5-fold CV: G³ (LogTail surrogate) test medians by fold")
    ax.legend(frameon=False)
    savefig(FIG_DIR / "cv_medians_bar_v2.png")


def plot_outer_slopes_hist():
    path = OUT_DIR / "outer_slopes_logtail.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if not ensure_cols(d, ["s_obs","s_model"]):
        return None
    fig, ax = plt.subplots(figsize=(7,5.5))
    ax.hist(d["s_obs"].dropna(), bins=30, alpha=0.6, label="Observed", color="#7f7f7f")
    ax.hist(d["s_model"].dropna(), bins=30, alpha=0.6, label="G³ (disk surrogate)", color="#d62728")
    ax.set_xlabel("Outer slope s = d ln v / d ln R")
    ax.set_ylabel("Count")
    ax.set_title("Outer slope distribution (G³ disk surrogate)")
    ax.legend(frameon=False)
    savefig(FIG_DIR / "outer_slopes_hist_v2.png")


def plot_shear_amp_vs_phiphi():
    # default location from earlier run
    shear = ROOT / "out" / "lensing" / "kids_b21" / "combined" / "shear_amp_summary.json"
    if not shear.exists():
        return None
    sj = json.loads(shear.read_text())
    # Try to get first chain's A_shear median and phi amplitude
    try:
        ch = sj.get("chains", [])[0]
        A_med = float(ch.get("A_shear_from_S8", {}).get("median"))
        A16 = float(ch.get("A_shear_from_S8", {}).get("p16"))
        A84 = float(ch.get("A_shear_from_S8", {}).get("p84"))
        alpha_phi = float(ch.get("phi_amp", {}).get("alpha_hat"))
        sigma_phi = float(ch.get("phi_amp", {}).get("sigma"))
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(7.5,5.5))
    # two points with error bars
    ax.errorbar([0], [A_med], yerr=[[A_med-A16],[A84-A_med]], fmt="o", capsize=4, label="Shear A_shear")
    ax.errorbar([1], [alpha_phi], yerr=[[2*sigma_phi],[2*sigma_phi]], fmt="o", capsize=4, label="Planck φφ (±2σ)")
    ax.set_xticks([0,1]); ax.set_xticklabels(["A_shear","α_φφ"])
    ax.set_ylim(0.6, 1.2)
    ax.set_ylabel("Amplitude")
    ax.set_title("Shear vs CMB lensing amplitude")
    # tension text
    try:
        tens = float(ch.get("A_over_phi", {}).get("tension_sigma"))
        ax.text(0.05, 0.9, f"tension ≈ {tens:.2f}σ", transform=ax.transAxes)
    except Exception:
        pass
    ax.legend(frameon=False)
    savefig(FIG_DIR / "shear_vs_phiphi_v2.png")

if __name__ == "__main__":
    # Generate all standard paper figures if inputs exist; each function is no-op if missing.
    try:
        plot_rotation_curve_overlays()
    except Exception:
        pass
    try:
        plot_rar_obs_vs_model()
    except Exception:
        pass
    try:
        plot_btfr_two_panel()
    except Exception:
        pass
    try:
        plot_lensing_shapes()
    except Exception:
        pass
    try:
        plot_cv_medians()
    except Exception:
        pass
    try:
        plot_outer_slopes_hist()
    except Exception:
        pass
    try:
        plot_shear_amp_vs_phiphi()
    except Exception:
        pass


def plot_cmb_envelope_summary():
    cmb = ROOT / "out" / "cmb_envelopes_tttee" / "cmb_envelope_lens.json"
    if not cmb.exists():
        return None
    cj = json.loads(cmb.read_text())
    sigma_A = float(cj.get("sigma_A", np.nan))
    A95 = float(cj.get("A95", np.nan))
    fig, ax = plt.subplots(figsize=(7,5))
    # Show a simple gaussian width depiction centered at 0 with sigma band
    xs = np.linspace(-3*sigma_A, 3*sigma_A, 200)
    ys = np.exp(-0.5*(xs/sigma_A)**2)
    ax.plot(xs, ys, c="#1f77b4")
    ax.axvspan(-2*sigma_A, 2*sigma_A, color="#1f77b4", alpha=0.1, label="≈95% band (2σ)")
    ax.axvline(A95, c="#d62728", ls=":", label=f"A95≈{A95:.4f}")
    ax.axvline(-A95, c="#d62728", ls=":")
    ax.set_xlabel("Envelope amplitude A (TT)")
    ax.set_yticks([])
    ax.set_title(f"TTTEEE envelope null: σ_A≈{sigma_A:.4f}, 95%≈±{A95:.4f}")
    ax.legend(frameon=False)
    savefig(FIG_DIR / "cmb_tttee_envelope_v2.png")


def write_logtail_only_artifacts():
    # Create LogTail-only summary and predictions for clean paper references
    try:
        s_path = OUT_DIR / "summary_logtail_muphi.json"
        if s_path.exists():
            sj = json.loads(s_path.read_text())
            lt = sj.get("LogTail", None)
            if lt is not None:
                (OUT_DIR / "summary_logtail.json").write_text(json.dumps(lt, indent=2))
    except Exception:
        pass
    try:
        p_path = OUT_DIR / "predictions_with_LogTail_MuPhi.csv"
        if p_path.exists():
            df = pd.read_csv(p_path)
            keep = [c for c in ["galaxy","R_kpc","Vobs_kms","Vbar_kms","is_outer","v_LogTail_kms"] if c in df.columns]
            if keep:
                df[keep].to_csv(OUT_DIR / "predictions_with_LogTail.csv", index=False)
    except Exception:
        pass


    """Overlay median |ΔT|/T and a small G³ tuple note onto cached cluster paper figures if present."""
    import json
    from pathlib import Path
    import numpy as np
    clusters = ["ABELL_0426", "ABELL_1689"]
    for cl in clusters:
        mpath = ROOT / "root-m" / "out" / "pde_clusters" / cl / "metrics.json"
        med = None
        if mpath.exists():
            try:
                mj = json.loads(mpath.read_text())
                med = float(mj.get("temp_median_frac_err", np.nan))
            except Exception:
                med = None
        # pick latest cached paper figure
        candidates = sorted(FIG_DIR.glob(f"cluster_{cl}_pde_results_*.png"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            continue
        ipath = candidates[-1]
        try:
            im = plt.imread(ipath)
            h, w = im.shape[:2]
            fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
            ax.imshow(im)
            ax.axis('off')
            # annotation positions in pixels (relative fractions)
            if med is not None and np.isfinite(med):
                ax.text(0.03*w, 0.06*h, f"median |ΔT|/T ≈ {med:.3f}", color="white", fontsize=14, weight="bold",
                        bbox=dict(facecolor="black", alpha=0.35, boxstyle="round,pad=0.2"))
            ax.text(0.03*w, 0.12*h, "G³: single global tuple", color="white", fontsize=12,
                    bbox=dict(facecolor="black", alpha=0.25, boxstyle="round,pad=0.2"))
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(ipath, dpi=100)
            plt.close(fig)
        except Exception:
            try:
                plt.close(fig)
            except Exception:
                pass


def main():
    write_logtail_only_artifacts()
    plot_rotation_curve_overlays()
    plot_rar_obs_vs_model()
    plot_btfr_two_panel()
    plot_lensing_shapes()
    plot_cv_medians()
    plot_outer_slopes_hist()
    plot_shear_amp_vs_phiphi()
    plot_cmb_envelope_summary()


if __name__ == "__main__":
    main()
