from __future__ import annotations
import os, argparse, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from rigor.rigor.data import load_sparc
from gravity_learn.features.geometry import dimensionless_radius, sigma_hat, grad_log_sigma


def fx_simple(x, Sigma_hat, grad_ln_Sigma):
    # fX_over_bar ≈ k * x  (from simplicity-biased SR)
    k = 2.5608
    return np.maximum(0.0, k * x)


def fx_ratio(x, Sigma_hat, grad_ln_Sigma):
    # fX_over_bar ≈ x^2 / (a - b * Sigma_hat)
    # constants inspired by compact candidate in the simple SR hall-of-fame
    a = 0.53605
    b = 0.23831
    denom = (a - b * Sigma_hat)
    denom = np.where(np.abs(denom) < 1e-6, np.sign(denom) * 1e-6, denom)
    return np.maximum(0.0, (x * x) / denom)


def fx_exp(x, Sigma_hat, grad_ln_Sigma):
    # fX_over_bar ≈ x^2 * (exp(Sigma_hat) + c)
    c = 0.67621
    return np.maximum(0.0, (x * x) * (np.exp(Sigma_hat) + c))


FORMULAS = {
    "simple": fx_simple,
    "ratio": fx_ratio,
    "exp": fx_exp,
}


def plot_one(ax, g, f_func, title_suffix=""):
    R = g.R_kpc
    Vobs = g.Vobs_kms
    Vbar = g.Vbar_kms
    if R is None or Vobs is None or Vbar is None:
        return None
    mask = np.isfinite(R) & np.isfinite(Vobs) & np.isfinite(Vbar)
    R = R[mask]
    Vobs = Vobs[mask]
    Vbar = Vbar[mask]
    if R.size < 4:
        return None
    Sigma = None
    if g.Sigma_bar is not None:
        Sigma = g.Sigma_bar[mask]
    else:
        Sigma = np.maximum(1e-3, np.exp(-R / np.maximum(1.0, np.nanmedian(R))))

    x = dimensionless_radius(R, Rd=(g.Rd_kpc or None))
    Sh = sigma_hat(Sigma)
    dlnS = grad_log_sigma(R, Sigma)

    fX = f_func(np.asarray(x), np.asarray(Sh), np.asarray(dlnS))
    Vmod = Vbar * np.sqrt(np.maximum(0.0, 1.0 + fX))

    eV = g.eVobs_kms if hasattr(g, 'eVobs_kms') and g.eVobs_kms is not None else None
    eV = eV[mask] if (eV is not None and np.size(eV) == np.size(mask)) else None

    ax.plot(R, Vobs, 'k.', label='Observed')
    if eV is not None and np.isfinite(eV).any():
        ax.fill_between(R, Vobs - eV, Vobs + eV, color='k', alpha=0.1, linewidth=0)
    ax.plot(R, Vbar, color='#1f77b4', alpha=0.8, label='Baryons')
    ax.plot(R, Vmod, color='#d62728', alpha=0.9, label='Model')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('V [km/s]')
    ax.set_title(f"{g.name} {title_suffix}")
    ax.grid(True, alpha=0.3)
    return {
        "Galaxy": g.name,
        "n_points": int(len(R)),
        "rmse": float(np.sqrt(np.mean((Vmod - Vobs) ** 2))),
        "median_ape": float(np.median(np.abs((Vmod - Vobs) / np.maximum(np.abs(Vobs), 1e-9)))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formula", type=str, default="simple", choices=list(FORMULAS.keys()))
    ap.add_argument("--limit_galaxies", type=int, default=16)
    ap.add_argument("--outdir", type=str, default=os.path.join("gravity_learn", "experiments", "eval", "plots"))
    args = ap.parse_args()

    ds = load_sparc()
    os.makedirs(args.outdir, exist_ok=True)
    f_func = FORMULAS[args.formula]

    # Plot a grid of galaxies
    galaxies = ds.galaxies[:args.limit_galaxies]
    n = len(galaxies)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    rows = []
    for i, g in enumerate(galaxies):
        res = plot_one(axes[i], g, f_func, title_suffix=f"[{args.formula}]")
        if res is not None:
            rows.append(res)
    for j in range(len(galaxies), len(axes)):
        fig.delaxes(axes[j])

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    # Save
    base = f"rc_overlays_{args.formula}_L{args.limit_galaxies}.png"
    out_png = os.path.join(args.outdir, base)
    fig.suptitle(f"Rotation Curves: Observed vs Model ({args.formula})", fontsize=14)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # Metrics CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.outdir, f"metrics_{args.formula}_L{args.limit_galaxies}.csv"), index=False)

    print(f"[plot] wrote {out_png}")


if __name__ == "__main__":
    main()