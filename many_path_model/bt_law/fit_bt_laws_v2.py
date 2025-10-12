#!/usr/bin/env python3
"""
Fit V2 Extended B/T Laws with Multi-Predictor Gating
====================================================

Regresses gate parameters from per-galaxy best fits:
- Sigma_ref, gamma_Sigma (compactness gating)
- S0, n_shear (shear gating)
- eta_min_fraction, Mmax_min_fraction, ring_min_fraction
- kappa_min, kappa_max

Uses scipy optimization to find gate parameters that best explain
the variation in per-galaxy eta, ring_amp, M_max vs (B/T, Sigma0, shear).
"""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution

import sys
sys.path.insert(0, str(Path(__file__).parent))
from bt_laws import morph_to_bt, fit_one_law, law_value
from bt_laws_v2 import compactness_gate, shear_gate, save_theta


def load_all_data(results_path, disk_params_path, shear_preds_path):
    """Load and merge all data sources."""
    # Per-galaxy results
    with open(results_path, 'r') as f:
        results_data = json.load(f)
    
    rows = []
    for item in results_data["results"]:
        name = item.get("name")
        bp = item.get("best_params", {})
        lam = bp.get("lambda_ring", bp.get("lambda_hat"))
        
        rows.append({
            "name": name,
            "hubble_type": item.get("hubble_type"),
            "type_group": item.get("type_group"),
            "best_error": item.get("best_error"),
            "eta": bp.get("eta"),
            "ring_amp": bp.get("ring_amp"),
            "M_max": bp.get("M_max"),
            "lambda_ring": lam,
        })
    
    df = pd.DataFrame(rows)
    df["B_T"] = [morph_to_bt(ht, tg) for ht, tg in zip(df["hubble_type"], df["type_group"])]
    
    # Disk parameters
    with open(disk_params_path, 'r') as f:
        disk_params = json.load(f)
    
    df_disk = pd.DataFrame([
        {'name': p['name'], 'R_d_kpc': p.get('R_d_kpc'), 'Sigma0': p.get('Sigma0')}
        for p in disk_params.values()
    ])
    
    # Shear predictors
    with open(shear_preds_path, 'r') as f:
        shear_preds = json.load(f)
    
    df_shear = pd.DataFrame([
        {'name': p['name'], 'shear': p.get('shear_2p2Rd'), 'compactness': p.get('compactness')}
        for p in shear_preds.values()
    ])
    
    # Merge
    df = df.merge(df_disk, on='name', how='left')
    df = df.merge(df_shear, on='name', how='left')
    
    # Filter complete data
    df = df.dropna(subset=["eta", "ring_amp", "M_max", "lambda_ring", "R_d_kpc", "Sigma0", "shear"])
    
    # Weights
    df["w"] = 1.0 / (1.0 + 0.02 * df["best_error"].values)
    
    return df


def fit_base_bt_laws(df):
    """Fit base B/T laws (without gating)."""
    print("\n" + "="*80)
    print("FITTING BASE B/T LAWS")
    print("="*80)
    
    theta = {}
    
    for param, bounds_lo, bounds_hi, gamma_bounds in [
        ("eta", (0.01, 0.3), (0.5, 2.0), (0.4, 4.0)),
        ("ring_amp", (0.0, 0.5), (3.0, 15.0), (0.4, 4.0)),
        ("M_max", (0.8, 1.2), (2.0, 5.0), (0.4, 4.0)),
        ("lambda_ring", (6.0, 10.0), (25.0, 55.0), (0.4, 4.0)),
    ]:
        print(f"\nFitting {param}...")
        theta[param] = fit_one_law(
            df["B_T"].values, df[param].values,
            lo_bounds=bounds_lo, hi_bounds=bounds_hi,
            gamma_bounds=gamma_bounds, n_trials=6000, weights=df["w"].values
        )
        print(f"  lo={theta[param]['lo']:.3f}, hi={theta[param]['hi']:.3f}, "
              f"gamma={theta[param]['gamma']:.3f}")
    
    return theta


def fit_compactness_gates(df, theta):
    """
    Fit compactness gate parameters (Sigma_ref, gamma_Sigma, min_fractions)
    by minimizing residuals between predicted and actual eta, M_max.
    """
    print("\n" + "="*80)
    print("FITTING COMPACTNESS GATE PARAMETERS")
    print("="*80)
    
    # For each galaxy, compute base eta from B/T law
    eta_base = np.array([
        law_value(bt, theta["eta"]['lo'], theta["eta"]['hi'], theta["eta"]['gamma'])
        for bt in df['B_T']
    ])
    
    Mmax_base = np.array([
        law_value(bt, theta["M_max"]['lo'], theta["M_max"]['hi'], theta["M_max"]['gamma'])
        for bt in df['B_T']
    ])
    
    # Actual values from per-galaxy fits
    eta_actual = df['eta'].values
    Mmax_actual = df['M_max'].values
    Sigma0 = df['Sigma0'].values
    w = df['w'].values
    
    def objective(params):
        Sigma_ref, gamma_Sigma, eta_min_frac, Mmax_min_frac = params
        
        # Compute gated predictions
        comp_gates = np.array([
            compactness_gate(s, Sigma_ref, gamma_Sigma) for s in Sigma0
        ])
        
        eta_pred = eta_base * (eta_min_frac + (1.0 - eta_min_frac) * comp_gates)
        Mmax_pred = Mmax_base * (Mmax_min_frac + (1.0 - Mmax_min_frac) * comp_gates)
        
        # Weighted MSE
        eta_err = np.sum(w * (eta_pred - eta_actual)**2)
        mmax_err = np.sum(w * (Mmax_pred - Mmax_actual)**2)
        
        return eta_err + 0.5 * mmax_err  # Weight eta more (it's more important)
    
    # Optimize
    bounds = [
        (20.0, 300.0),   # Sigma_ref
        (0.3, 1.5),      # gamma_Sigma
        (0.1, 0.5),      # eta_min_fraction
        (0.3, 0.7),      # Mmax_min_fraction
    ]
    
    print("\nOptimizing compactness gate parameters...")
    result = differential_evolution(objective, bounds, seed=42, maxiter=300, workers=1)
    
    Sigma_ref, gamma_Sigma, eta_min_frac, Mmax_min_frac = result.x
    
    print(f"\nOptimal compactness gates:")
    print(f"  Sigma_ref = {Sigma_ref:.1f} M_sun/pc^2")
    print(f"  gamma_Sigma = {gamma_Sigma:.3f}")
    print(f"  eta_min_fraction = {eta_min_frac:.3f}")
    print(f"  Mmax_min_fraction = {Mmax_min_frac:.3f}")
    print(f"  Final loss = {result.fun:.4f}")
    
    return {
        'Sigma_ref': float(Sigma_ref),
        'gamma_Sigma': float(gamma_Sigma),
        'eta_min_fraction': float(eta_min_frac),
        'Mmax_min_fraction': float(Mmax_min_frac),
    }


def fit_shear_gates(df, theta):
    """
    Fit shear gate parameters (S0, n_shear, ring_min_fraction)
    by minimizing residuals for ring_amp and lambda_ring.
    """
    print("\n" + "="*80)
    print("FITTING SHEAR GATE PARAMETERS")
    print("="*80)
    
    ring_amp_base = np.array([
        law_value(bt, theta["ring_amp"]['lo'], theta["ring_amp"]['hi'], theta["ring_amp"]['gamma'])
        for bt in df['B_T']
    ])
    
    lambda_base = np.array([
        law_value(bt, theta["lambda_ring"]['lo'], theta["lambda_ring"]['hi'], theta["lambda_ring"]['gamma'])
        for bt in df['B_T']
    ])
    
    ring_amp_actual = df['ring_amp'].values
    lambda_actual = df['lambda_ring'].values
    shear = df['shear'].values
    w = df['w'].values
    
    def objective(params):
        S0, n_shear, ring_min_frac = params
        
        shear_gates = np.array([shear_gate(s, S0, n_shear) for s in shear])
        
        ring_pred = ring_amp_base * (ring_min_frac + (1.0 - ring_min_frac) * shear_gates)
        lambda_pred = lambda_base * (0.5 + 0.5 * shear_gates)
        
        ring_err = np.sum(w * (ring_pred - ring_amp_actual)**2)
        lambda_err = np.sum(w * (lambda_pred - lambda_actual)**2) * 0.01  # Less weight on lambda
        
        return ring_err + lambda_err
    
    bounds = [
        (0.5, 1.2),     # S0
        (1.0, 4.0),     # n_shear
        (0.1, 0.4),     # ring_min_fraction
    ]
    
    print("\nOptimizing shear gate parameters...")
    result = differential_evolution(objective, bounds, seed=43, maxiter=300, workers=1)
    
    S0, n_shear, ring_min_frac = result.x
    
    print(f"\nOptimal shear gates:")
    print(f"  S0 = {S0:.3f}")
    print(f"  n_shear = {n_shear:.3f}")
    print(f"  ring_min_fraction = {ring_min_frac:.3f}")
    print(f"  Final loss = {result.fun:.4f}")
    
    return {
        'S0': float(S0),
        'n_shear': float(n_shear),
        'ring_min_fraction': float(ring_min_frac),
    }


def fit_kappa_bounds(df, comp_gates, shear_gates):
    """
    Fit kappa_min and kappa_max based on typical ranges.
    Use simple heuristics since kappa affects the kernel directly.
    """
    print("\n" + "="*80)
    print("ESTIMATING KAPPA BOUNDS")
    print("="*80)
    
    # Kappa should be high for HSB + low-shear + disk-dominated
    # and low for LSB + high-shear + bulge-dominated
    
    # Use conservative estimates
    kappa_min = 0.3  # Allow significant decoherence
    kappa_max = 0.95  # Never fully coherent (real disks have some turbulence)
    
    print(f"\nUsing conservative kappa bounds:")
    print(f"  kappa_min = {kappa_min}")
    print(f"  kappa_max = {kappa_max}")
    
    return {'kappa_min': kappa_min, 'kappa_max': kappa_max}


def main():
    parser = argparse.ArgumentParser(description='Fit V2 Extended B/T Laws')
    parser.add_argument('--results', type=Path,
                       default=Path('results/mega_test/mega_parallel_results.json'))
    parser.add_argument('--disk_params', type=Path,
                       default=Path('many_path_model/bt_law/sparc_disk_params.json'))
    parser.add_argument('--shear_preds', type=Path,
                       default=Path('many_path_model/bt_law/sparc_shear_predictors.json'))
    parser.add_argument('--output', type=Path,
                       default=Path('many_path_model/bt_law/bt_law_params_v2.json'))
    parser.add_argument('--output_fig', type=Path,
                       default=Path('many_path_model/bt_law/bt_law_v2_fits.png'))
    
    args = parser.parse_args()
    
    print("="*80)
    print("FITTING V2 EXTENDED B/T LAWS WITH MULTI-PREDICTOR GATING")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = load_all_data(args.results, args.disk_params, args.shear_preds)
    print(f"  Loaded {len(df)} galaxies with complete data")
    print(f"  B/T range: {df['B_T'].min():.3f} - {df['B_T'].max():.3f}")
    print(f"  Sigma0 range: {df['Sigma0'].min():.1f} - {df['Sigma0'].max():.1f}")
    print(f"  Shear range: {df['shear'].min():.3f} - {df['shear'].max():.3f}")
    
    # Fit base B/T laws
    theta = fit_base_bt_laws(df)
    
    # Fit compactness gates
    comp_params = fit_compactness_gates(df, theta)
    theta.update(comp_params)
    
    # Fit shear gates
    shear_params = fit_shear_gates(df, theta)
    theta.update(shear_params)
    
    # Kappa bounds
    kappa_params = fit_kappa_bounds(df, comp_params, shear_params)
    theta.update(kappa_params)
    
    # Size scaling (use MW-centric defaults)
    theta['a0'] = 2.0
    theta['a1'] = 28.0
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_theta(args.output, theta)
    print(f"\n[OK] Saved v2 theta to: {args.output}")
    
    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Compute predictions with fitted gates
    from bt_laws_v2 import eval_all_laws_v2
    
    predictions = []
    for _, row in df.iterrows():
        pred = eval_all_laws_v2(row['B_T'], theta, 
                                Sigma0=row['Sigma0'], 
                                R_d=row['R_d_kpc'],
                                shear=row['shear'])
        predictions.append(pred)
    
    pred_df = pd.DataFrame(predictions)
    
    # Plot eta vs actual
    ax = axes[0, 0]
    ax.scatter(df['eta'], pred_df['eta'], alpha=0.5, s=20)
    ax.plot([0, df['eta'].max()], [0, df['eta'].max()], 'r--', alpha=0.5)
    ax.set_xlabel('Actual eta')
    ax.set_ylabel('Predicted eta (v2)')
    ax.set_title('eta: Compactness Gated')
    ax.grid(alpha=0.3)
    
    # Plot ring_amp vs actual
    ax = axes[0, 1]
    ax.scatter(df['ring_amp'], pred_df['ring_amp'], alpha=0.5, s=20, c=df['shear'], cmap='viridis')
    ax.plot([0, df['ring_amp'].max()], [0, df['ring_amp'].max()], 'r--', alpha=0.5)
    ax.set_xlabel('Actual ring_amp')
    ax.set_ylabel('Predicted ring_amp (v2)')
    ax.set_title('ring_amp: Shear Gated (color=shear)')
    ax.grid(alpha=0.3)
    
    # Plot M_max vs actual
    ax = axes[0, 2]
    ax.scatter(df['M_max'], pred_df['M_max'], alpha=0.5, s=20)
    ax.plot([0, df['M_max'].max()], [0, df['M_max'].max()], 'r--', alpha=0.5)
    ax.set_xlabel('Actual M_max')
    ax.set_ylabel('Predicted M_max (v2)')
    ax.set_title('M_max: Compactness Gated')
    ax.grid(alpha=0.3)
    
    # Eta vs Sigma0 (colored by B/T)
    ax = axes[1, 0]
    sc = ax.scatter(df['Sigma0'], df['eta'], c=df['B_T'], cmap='coolwarm', alpha=0.6, s=20)
    ax.set_xscale('log')
    ax.set_xlabel('Sigma0 (M_sun/pc^2)')
    ax.set_ylabel('eta')
    ax.set_title('eta vs Compactness (color=B/T)')
    plt.colorbar(sc, ax=ax, label='B/T')
    ax.grid(alpha=0.3, which='both')
    ax.axvline(theta['Sigma_ref'], color='red', linestyle='--', alpha=0.7)
    
    # Ring_amp vs Shear (colored by B/T)
    ax = axes[1, 1]
    sc = ax.scatter(df['shear'], df['ring_amp'], c=df['B_T'], cmap='coolwarm', alpha=0.6, s=20)
    ax.set_xlabel('Shear S')
    ax.set_ylabel('ring_amp')
    ax.set_title('ring_amp vs Shear (color=B/T)')
    plt.colorbar(sc, ax=ax, label='B/T')
    ax.grid(alpha=0.3)
    ax.axvline(theta['S0'], color='red', linestyle='--', alpha=0.7)
    
    # Kappa distribution
    ax = axes[1, 2]
    ax.hist(pred_df['kappa'], bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Predicted kappa')
    ax.set_ylabel('Count')
    ax.set_title(f'Coherence Factor Distribution\n(min={theta["kappa_min"]:.2f}, max={theta["kappa_max"]:.2f})')
    ax.axvline(theta['kappa_min'], color='red', linestyle='--', alpha=0.7, label='kappa_min')
    ax.axvline(theta['kappa_max'], color='red', linestyle='--', alpha=0.7, label='kappa_max')
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.suptitle('V2 Extended B/T Laws: Multi-Predictor Gating', fontsize=16, fontweight='bold')
    fig.tight_layout()
    
    args.output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_fig, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved diagnostic figure to: {args.output_fig}")
    
    # Print summary
    print("\n" + "="*80)
    print("V2 EXTENDED LAW FITTING COMPLETE")
    print("="*80)
    print("\nFitted gate parameters:")
    print(f"  Compactness:")
    print(f"    Sigma_ref = {theta['Sigma_ref']:.1f} M_sun/pc^2")
    print(f"    gamma_Sigma = {theta['gamma_Sigma']:.3f}")
    print(f"    eta_min_fraction = {theta['eta_min_fraction']:.3f}")
    print(f"    Mmax_min_fraction = {theta['Mmax_min_fraction']:.3f}")
    print(f"  Shear:")
    print(f"    S0 = {theta['S0']:.3f}")
    print(f"    n_shear = {theta['n_shear']:.3f}")
    print(f"    ring_min_fraction = {theta['ring_min_fraction']:.3f}")
    print(f"  Coherence:")
    print(f"    kappa_min = {theta['kappa_min']:.3f}")
    print(f"    kappa_max = {theta['kappa_max']:.3f}")
    
    # Compute R² for validation
    for param in ['eta', 'ring_amp', 'M_max']:
        residuals = df[param].values - pred_df[param].values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((df[param].values - df[param].mean())**2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"\n  {param}: R² = {r2:.3f}, RMSE = {rmse:.4f}")


if __name__ == "__main__":
    main()
