#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit Extended B/T Laws with Size and Compactness Scaling
========================================================

Regresses scaling coefficients (a0, a1, a4 for radial scales; Sigma_ref, Sigma_alpha for compactness)
from per-galaxy best fits while accounting for galaxy R_d and Sigma0.

Usage:
    python many_path_model/bt_law/fit_extended_bt_laws.py
"""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from bt_laws import morph_to_bt, fit_one_law, save_theta, load_theta


def load_mega_results_with_disk_params(results_path: Path, disk_params_path: Path) -> pd.DataFrame:
    """Load per-galaxy best fits and merge with disk parameters."""
    # Load per-galaxy results
    with open(results_path, 'r') as f:
        results_data = json.load(f)
    
    rows = []
    for item in results_data["results"]:
        name = item.get("name")
        bp = item.get("best_params", {})
        lam = bp.get("lambda_ring", bp.get("lambda_hat"))
        
        row = {
            "name": name,
            "hubble_type": item.get("hubble_type"),
            "type_group": item.get("type_group"),
            "best_error": item.get("best_error"),
            "eta": bp.get("eta"),
            "ring_amp": bp.get("ring_amp"),
            "M_max": bp.get("M_max"),
            "lambda_ring": lam,
            "R0": bp.get("R0", 5.0),  # Might not be in results if fixed
            "R1": bp.get("R1", 70.0),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df["B_T"] = [morph_to_bt(ht, tg) for ht, tg in zip(df["hubble_type"], df["type_group"])]
    
    # Load disk parameters
    with open(disk_params_path, 'r') as f:
        disk_params = json.load(f)
    
    # Merge
    df_disk = pd.DataFrame([
        {
            'name': p['name'],
            'R_d_kpc': p.get('R_d_kpc'),
            'Sigma0': p.get('Sigma0')
        }
        for p in disk_params.values()
    ])
    
    df = df.merge(df_disk, on='name', how='left')
    
    # Filter valid rows
    df = df.dropna(subset=["eta", "ring_amp", "M_max", "lambda_ring", "R_d_kpc", "Sigma0"])
    
    # Weights: emphasize low-error galaxies
    df["w"] = 1.0 / (1.0 + 0.02 * df["best_error"].values)
    
    return df


def fit_radial_scaling(df: pd.DataFrame, param_name: str, default_val: float = 5.0):
    """
    Fit R0 or R1 as a linear function of R_d: param = a * R_d
    
    Args:
        df: DataFrame with per-galaxy best params and R_d_kpc
        param_name: 'R0' or 'R1'
        default_val: Default value if param not in df
    
    Returns:
        Scaling coefficient a
    """
    if param_name not in df.columns or df[param_name].isna().all():
        # Param not optimized per-galaxy, use default scaling
        ratio = default_val / 2.5  # Assume MW R_d ~ 2.5 kpc
        print(f"  {param_name} not in results, using default ratio {ratio:.2f}")
        return ratio
    
    valid = df[[param_name, 'R_d_kpc', 'w']].dropna()
    if len(valid) < 10:
        ratio = default_val / 2.5
        print(f"  {param_name}: insufficient data, using default {ratio:.2f}")
        return ratio
    
    # Weighted linear regression through origin: param = a * R_d
    w = valid['w'].values
    x = valid['R_d_kpc'].values
    y = valid[param_name].values
    
    a = np.sum(w * x * y) / np.sum(w * x * x)
    
    residuals = y - a * x
    weighted_rmse = np.sqrt(np.average(residuals**2, weights=w))
    
    print(f"  {param_name} = {a:.3f} * R_d  (RMSE: {weighted_rmse:.2f})")
    return float(a)


def fit_compactness_params(df: pd.DataFrame):
    """
    Fit Sigma_ref and Sigma_alpha for compactness gating.
    
    Optimize to match the variation in eta and ring_amp with Sigma0.
    
    Returns:
        (Sigma_ref, Sigma_alpha) tuple
    """
    # Use eta and ring_amp variation with Sigma0 to infer compactness effect
    # For late-type galaxies (B/T < 0.2), expect stronger Sigma dependence
    
    late = df[df['B_T'] < 0.2].copy()
    if len(late) < 20:
        print("  Insufficient late-type data for compactness fit, using defaults")
        return 150.0, 0.8
    
    # Normalize eta by B/T baseline to isolate Sigma effect
    # Expect eta to scale with Sigma at fixed B/T
    
    # Simple robust estimate: use median Sigma0 as reference
    Sigma_ref = float(np.median(late['Sigma0']))
    
    # Estimate alpha from scatter: higher alpha = steeper transition
    # Use coefficient of variation as proxy
    cv = np.std(late['eta']) / np.mean(late['eta'])
    alpha = np.clip(0.5 / cv, 0.3, 1.5)  # Heuristic mapping
    
    print(f"  Compactness: Sigma_ref = {Sigma_ref:.1f} M_sun/pc^2, alpha = {alpha:.2f}")
    return float(Sigma_ref), float(alpha)


def main():
    parser = argparse.ArgumentParser(description='Fit Extended B/T Laws')
    parser.add_argument('--results', type=Path,
                       default=Path('results/mega_test/mega_parallel_results.json'),
                       help='Per-galaxy best fit results')
    parser.add_argument('--disk_params', type=Path,
                       default=Path('many_path_model/bt_law/sparc_disk_params.json'),
                       help='Disk parameters (R_d, Sigma0)')
    parser.add_argument('--out_params', type=Path,
                       default=Path('many_path_model/bt_law/bt_law_params_extended.json'),
                       help='Output extended parameter file')
    parser.add_argument('--out_fig', type=Path,
                       default=Path('many_path_model/bt_law/extended_bt_law_fits.png'),
                       help='Output diagnostic figure')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FITTING EXTENDED B/T LAWS WITH SIZE AND COMPACTNESS SCALING")
    print("="*80)
    
    # Load data
    print(f"\nLoading data...")
    df = load_mega_results_with_disk_params(args.results, args.disk_params)
    print(f"  Loaded {len(df)} galaxies with complete data")
    print(f"  R_d range: {df['R_d_kpc'].min():.2f} - {df['R_d_kpc'].max():.2f} kpc")
    print(f"  Sigma0 range: {df['Sigma0'].min():.1f} - {df['Sigma0'].max():.1f} M_sun/pc^2")
    
    # Fit base B/T laws (same as before)
    print("\n" + "-"*80)
    print("FITTING BASE B/T LAWS")
    print("-"*80)
    
    theta = {}
    
    print("\nFitting eta...")
    theta["eta"] = fit_one_law(
        df["B_T"].values, df["eta"].values,
        lo_bounds=(0.01, 0.3), hi_bounds=(0.5, 2.0),
        gamma_bounds=(0.4, 4.0), n_trials=6000, weights=df["w"].values
    )
    print(f"  lo={theta['eta']['lo']:.3f}, hi={theta['eta']['hi']:.3f}, gamma={theta['eta']['gamma']:.3f}")
    
    print("\nFitting ring_amp...")
    theta["ring_amp"] = fit_one_law(
        df["B_T"].values, df["ring_amp"].values,
        lo_bounds=(0.0, 0.5), hi_bounds=(3.0, 15.0),
        gamma_bounds=(0.4, 4.0), n_trials=6000, weights=df["w"].values
    )
    print(f"  lo={theta['ring_amp']['lo']:.3f}, hi={theta['ring_amp']['hi']:.3f}, gamma={theta['ring_amp']['gamma']:.3f}")
    
    print("\nFitting M_max...")
    theta["M_max"] = fit_one_law(
        df["B_T"].values, df["M_max"].values,
        lo_bounds=(0.8, 1.2), hi_bounds=(2.0, 5.0),
        gamma_bounds=(0.4, 4.0), n_trials=6000, weights=df["w"].values
    )
    print(f"  lo={theta['M_max']['lo']:.3f}, hi={theta['M_max']['hi']:.3f}, gamma={theta['M_max']['gamma']:.3f}")
    
    print("\nFitting lambda_ring (base)...")
    theta["lambda_ring"] = fit_one_law(
        df["B_T"].values, df["lambda_ring"].values,
        lo_bounds=(6.0, 10.0), hi_bounds=(25.0, 55.0),
        gamma_bounds=(0.4, 4.0), n_trials=6000, weights=df["w"].values
    )
    print(f"  lo={theta['lambda_ring']['lo']:.3f}, hi={theta['lambda_ring']['hi']:.3f}, gamma={theta['lambda_ring']['gamma']:.3f}")
    
    # Fit size scaling coefficients
    print("\n" + "-"*80)
    print("FITTING SIZE SCALING COEFFICIENTS")
    print("-"*80)
    
    print("\nFitting R0 scaling...")
    theta['a0'] = fit_radial_scaling(df, 'R0', default_val=5.0)
    
    print("\nFitting R1 scaling...")
    theta['a1'] = fit_radial_scaling(df, 'R1', default_val=70.0)
    
    # Lambda_ring vs R_d correlation (expect larger rings in larger galaxies)
    print("\nAnalyzing lambda_ring vs R_d correlation...")
    corr = df[['lambda_ring', 'R_d_kpc']].corr().iloc[0, 1]
    print(f"  Correlation: {corr:.3f}")
    
    # Fit weak linear slope: lambda ~ lambda_base(B/T) * (1 + slope * (R_d/2.5 - 1))
    # This makes lambda scale slowly with size around MW reference
    valid = df[['lambda_ring', 'R_d_kpc', 'B_T', 'w']].dropna()
    if len(valid) > 30 and abs(corr) > 0.1:
        # Residual after B/T law
        from bt_laws import law_value
        lambda_pred_bt = np.array([
            law_value(bt, theta["lambda_ring"]['lo'], theta["lambda_ring"]['hi'], theta["lambda_ring"]['gamma'])
            for bt in valid['B_T']
        ])
        residual = valid['lambda_ring'].values - lambda_pred_bt
        
        # Fit residual ~ slope * (R_d/2.5 - 1) * lambda_pred_bt
        X = (valid['R_d_kpc'].values / 2.5 - 1.0) * lambda_pred_bt
        w = valid['w'].values
        slope = np.sum(w * X * residual) / np.sum(w * X * X)
        slope = np.clip(slope, -0.5, 0.5)  # Reasonable bounds
        
        theta['lambda_Rd_slope'] = float(slope)
        print(f"  lambda_Rd_slope = {slope:.4f}")
    else:
        theta['lambda_Rd_slope'] = 0.0
        print(f"  lambda_Rd_slope = 0.0 (insufficient correlation)")
    
    # Fit compactness parameters
    print("\n" + "-"*80)
    print("FITTING COMPACTNESS PARAMETERS")
    print("-"*80)
    
    Sigma_ref, Sigma_alpha = fit_compactness_params(df)
    theta['Sigma_ref'] = Sigma_ref
    theta['Sigma_alpha'] = Sigma_alpha
    
    # Save extended parameters
    args.out_params.parent.mkdir(parents=True, exist_ok=True)
    save_theta(args.out_params, theta)
    print(f"\n✓ Saved extended parameters to: {args.out_params}")
    
    # Generate diagnostic plots
    print(f"\nGenerating diagnostic plots...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    from bt_laws import law_value
    B_plot = np.linspace(0, 0.7, 200)
    
    # Row 1: Base B/T laws
    for idx, (param, label) in enumerate([
        ("eta", r"$\eta$"),
        ("ring_amp", r"ring_amp"),
        ("M_max", r"$M_{\max}$")
    ]):
        ax = fig.add_subplot(gs[0, idx])
        law = theta[param]
        curve = law_value(B_plot, law['lo'], law['hi'], law['gamma'])
        
        ax.scatter(df["B_T"], df[param], s=12, alpha=0.4, c='gray', label='Per-galaxy')
        ax.plot(B_plot, curve, 'r-', lw=2.5, label=f'Law (γ={law["gamma"]:.2f})')
        ax.set_xlabel("B/T")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title(f"{label} vs B/T", fontweight='bold')
    
    # Row 2: Size scaling
    for idx, (param, label, a_key) in enumerate([
        ("R0", r"$R_0$ (kpc)", 'a0'),
        ("R1", r"$R_1$ (kpc)", 'a1'),
        ("lambda_ring", r"$\lambda_{\rm ring}$ (kpc)", None)
    ]):
        ax = fig.add_subplot(gs[1, idx])
        
        if a_key and a_key in theta:
            a = theta[a_key]
            R_d_plot = np.linspace(df['R_d_kpc'].min(), df['R_d_kpc'].max(), 100)
            ax.scatter(df['R_d_kpc'], df[param], s=12, alpha=0.4, c='gray')
            ax.plot(R_d_plot, a * R_d_plot, 'b-', lw=2.5, label=f'{param} = {a:.2f} × R_d')
            ax.set_xlabel(r"$R_d$ (kpc)")
            ax.set_ylabel(label)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_title(f"{label} vs $R_d$", fontweight='bold')
        else:
            # Lambda vs R_d (color by B/T)
            sc = ax.scatter(df['R_d_kpc'], df['lambda_ring'], s=12, c=df['B_T'],
                           cmap='viridis', alpha=0.6)
            ax.set_xlabel(r"$R_d$ (kpc)")
            ax.set_ylabel(label)
            plt.colorbar(sc, ax=ax, label='B/T')
            ax.grid(alpha=0.3)
            ax.set_title(f"{label} vs $R_d$ (color=B/T)", fontweight='bold')
    
    # Row 3: Compactness effects
    for idx, (param, label) in enumerate([
        ("eta", r"$\eta$"),
        ("ring_amp", r"ring_amp"),
        ("lambda_ring", r"$\lambda_{\rm ring}$")
    ]):
        ax = fig.add_subplot(gs[2, idx])
        
        # Color by Sigma0, size by B/T
        late_df = df[df['B_T'] < 0.25]  # Focus on disk-dominated
        sc = ax.scatter(late_df['Sigma0'], late_df[param], 
                       s=50*(1-late_df['B_T']), c=late_df['B_T'],
                       cmap='coolwarm', alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.set_xlabel(r"$\Sigma_0$ (M$_\odot$/pc$^2$)")
        ax.set_ylabel(label)
        ax.set_xscale('log')
        plt.colorbar(sc, ax=ax, label='B/T')
        ax.grid(alpha=0.3, which='both')
        ax.axvline(Sigma_ref, color='red', linestyle='--', alpha=0.7, label=f'Σ_ref={Sigma_ref:.0f}')
        ax.legend(fontsize=8)
        ax.set_title(f"{label} vs $\Sigma_0$ (late-type)", fontweight='bold')
    
    fig.suptitle("Extended B/T Laws: Size & Compactness Scaling", fontsize=16, fontweight='bold', y=0.995)
    
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150, bbox_inches='tight')
    print(f"✓ Saved diagnostic figure to: {args.out_fig}")
    
    print("\n" + "="*80)
    print("EXTENDED B/T LAW FITTING COMPLETE")
    print("="*80)
    
    print("\nSummary of scaling coefficients:")
    print(f"  Size scaling:")
    print(f"    R0 = {theta['a0']:.3f} × R_d")
    print(f"    R1 = {theta['a1']:.3f} × R_d")
    print(f"    λ slope = {theta['lambda_Rd_slope']:.4f}")
    print(f"  Compactness:")
    print(f"    Σ_ref = {theta['Sigma_ref']:.1f} M_sun/pc^2")
    print(f"    α = {theta['Sigma_alpha']:.2f}")


if __name__ == "__main__":
    main()
