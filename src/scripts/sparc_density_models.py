#!/usr/bin/env python3
"""
SPARC Predictor with Density-Based Gravity Models

Key Discovery: G enhancement is NOT primarily mass-dependent but structure-dependent.
All galaxies need similar enhancement (~2-3×) at their characteristic density boundaries.

This script implements multiple models to find the best formula:
1. Pure Radial Model - Enhancement depends only on r/boundary
2. Density Gradient Model - Enhancement triggered by density drop
3. Phase Transition Model - Sharp transition at critical density
4. Smooth Shell Model - Gaussian shells at characteristic radii
5. Hybrid Model - Combines radial and density factors

Outputs (under --output-dir):
- model_comparison.csv — Metrics for all models
- best_model.json — Best model and parameters
- model_examples.png — Plot on a sample of galaxies (saved; no interactive window)

Data inputs:
- Prefers data/sparc_human_by_radius.csv (has M_bary)
- Falls back to data/sparc_predictions_by_radius.csv and attempts to join M_bary if present; otherwise aborts

This code avoids interactive windows (uses Agg backend) so it can run in CI/headless.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Use a non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from scipy.special import erf


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class GravityModel:
    """Base class for gravity enhancement models"""

    def __init__(self, name: str, param_names: list[str], param_bounds: list[tuple[float, float]]):
        self.name = name
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.best_params: np.ndarray | None = None
        self.best_score: float = float('inf')

    def compute_G(self, R_kpc: np.ndarray, boundary_kpc: np.ndarray, M_bary: np.ndarray, params: list[float] | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __str__(self):
        return f"{self.name} Model"


class PureRadialModel(GravityModel):
    """Enhancement depends only on normalized radius r/boundary.
    Physics: Galaxies create enhancement shells at fixed fractions of their size.
    """
    def __init__(self):
        super().__init__(
            name="PureRadial",
            param_names=['G_peak', 'r_peak', 'width', 'decay'],
            param_bounds=[(1.5, 4.0), (0.8, 1.5), (0.1, 0.8), (0.2, 3.0)],
        )

    def compute_G(self, R_kpc, boundary_kpc, M_bary, params):
        G_peak, r_peak, width, decay = params
        b = np.maximum(boundary_kpc, 1e-6)
        r_norm = np.asarray(R_kpc, dtype=float) / b
        # Smooth rise to ~0.5 boundary using error function
        rise = 0.5 * (1.0 + erf((r_norm - 0.5) / max(1e-6, width)))
        # Gaussian-like peak near r_peak
        peak = np.exp(-0.5 * ((r_norm - r_peak) / max(1e-6, width)) ** 2)
        # Exponential tail beyond peak
        decay_factor = np.where(r_norm > r_peak, np.exp(-decay * (r_norm - r_peak)), 1.0)
        G = 1.0 + (G_peak - 1.0) * rise * peak * decay_factor
        return G.astype(float)


class DensityGradientModel(GravityModel):
    """Enhancement triggered by local density falling below critical fraction.
    Physics: Low-density regions get gravitational boost to retain matter.
    """
    def __init__(self):
        super().__init__(
            name="DensityGradient",
            param_names=['G_max', 'rho_crit_frac', 'transition_width', 'r_core_frac'],
            param_bounds=[(1.5, 4.0), (0.05, 0.8), (0.05, 1.0), (0.05, 0.7)],
        )

    def compute_G(self, R_kpc, boundary_kpc, M_bary, params):
        G_max, rho_crit_frac, transition_width, r_core_frac = params
        b = np.maximum(boundary_kpc, 1e-6)
        r = np.asarray(R_kpc, dtype=float)
        r_core = np.maximum(r_core_frac * b, 1e-6)
        rho_central = np.maximum(M_bary, 1e-6) / np.maximum(b ** 3, 1e-6)
        rho_local = rho_central * np.exp(-r / b) * (1.0 + (r / r_core) ** 2) ** (-1.0)
        rho_boundary = rho_central * np.exp(-1.0) * (1.0 + (1.0 / np.maximum(r_core_frac, 1e-6)) ** 2) ** (-1.0)
        rho_crit = rho_crit_frac * rho_boundary
        # Smooth transition using tanh
        tw = np.maximum(transition_width * np.maximum(rho_crit, 1e-12), 1e-12)
        enhancement_factor = 0.5 * (1.0 + np.tanh((rho_crit - rho_local) / tw))
        G = 1.0 + (G_max - 1.0) * enhancement_factor
        return G.astype(float)


class PhaseTransitionModel(GravityModel):
    """Sharp transition at critical surface density (like a phase change).
    Physics: Gravity undergoes phase transition at characteristic surface density.
    """
    def __init__(self):
        super().__init__(
            name="PhaseTransition",
            param_names=['G_low', 'G_high', 'Sigma_crit', 'sharpness'],
            param_bounds=[(1.0, 1.5), (2.0, 4.0), (1.0, 1e4), (0.5, 15.0)],
        )

    def compute_G(self, R_kpc, boundary_kpc, M_bary, params):
        G_low, G_high, Sigma_crit, sharpness = params
        b = np.maximum(boundary_kpc, 1e-6)
        Rd = np.maximum(b / 3.0, 1e-6)
        Sigma0 = np.maximum(M_bary, 1e-6) / (2.0 * np.pi * Rd ** 2)
        Sigma_local = Sigma0 * np.exp(-np.asarray(R_kpc, dtype=float) / Rd)
        tr = 1.0 / (1.0 + (np.maximum(Sigma_local, 1e-30) / np.maximum(Sigma_crit, 1e-30)) ** np.maximum(sharpness, 1e-6))
        G = G_low + (G_high - G_low) * tr
        return G.astype(float)


class SmoothShellModel(GravityModel):
    """Multiple Gaussian shells at characteristic radii (electron-orbital analogy)."""
    def __init__(self):
        super().__init__(
            name="SmoothShell",
            param_names=['G_shell1', 'G_shell2', 'r1_frac', 'r2_frac', 'width1', 'width2'],
            param_bounds=[(1.2, 2.5), (1.5, 3.5), (0.2, 0.8), (0.8, 1.8), (0.05, 0.6), (0.05, 0.9)],
        )

    def compute_G(self, R_kpc, boundary_kpc, M_bary, params):
        G1, G2, r1, r2, w1, w2 = params
        b = np.maximum(boundary_kpc, 1e-6)
        r_norm = np.asarray(R_kpc, dtype=float) / b
        shell1 = np.exp(-0.5 * ((r_norm - r1) / max(1e-6, w1)) ** 2)
        shell2 = np.exp(-0.5 * ((r_norm - r2) / max(1e-6, w2)) ** 2)
        G = 1.0 + (G1 - 1.0) * shell1 + (G2 - 1.0) * shell2
        # Avoid exceeding the stronger shell cap unnecessarily
        G = np.minimum(G, np.maximum(G1, G2))
        return G.astype(float)


class HybridModel(GravityModel):
    """Combines radial profile with tiny mass correction (structure-dominant)."""
    def __init__(self):
        super().__init__(
            name="Hybrid",
            param_names=['G_base', 'radial_weight', 'mass_exp', 'r_trans', 'width'],
            param_bounds=[(2.0, 3.5), (0.3, 1.0), (-0.1, 0.1), (0.6, 1.4), (0.1, 0.9)],
        )

    def compute_G(self, R_kpc, boundary_kpc, M_bary, params):
        G_base, radial_weight, mass_exp, r_trans, width = params
        b = np.maximum(boundary_kpc, 1e-6)
        r_norm = np.asarray(R_kpc, dtype=float) / b
        radial_factor = 0.5 * (1.0 + np.tanh((r_norm - r_trans) / max(1e-6, width)))
        mass_factor = np.power(np.maximum(M_bary, 1e-12) / 1e10, mass_exp)
        G_max = G_base * mass_factor
        G = 1.0 + (G_max - 1.0) * radial_weight * radial_factor
        # Suppress inner region to respect Newtonian core
        inner_suppr = np.exp(-2.0 / (r_norm + 0.1))
        G = 1.0 + (G - 1.0) * inner_suppr
        return G.astype(float)


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'Galaxy': 'galaxy',
        'Radius_kpc': 'R_kpc',
        'Boundary_kpc': 'boundary_kpc',
        'Observed_Speed_km_s': 'Vobs_kms',
        'Baryonic_Speed_km_s': 'Vbar_kms',
        'Baryonic_Mass_Msun': 'M_bary',
    }
    out = df.copy()
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    return out


def load_sparc_data(data_dir: str | Path = 'data') -> pd.DataFrame:
    """Load SPARC by-radius data with baryonic mass available.

    Preference order:
    1) data/sparc_human_by_radius.csv
    2) <data_dir>/sparc_human_by_radius.csv
    3) data/sparc_predictions_by_radius.csv (only if contains/mergeable with M_bary)
    """
    data_dir = Path(data_dir)
    cand = [
        Path('data/sparc_human_by_radius.csv'),
        data_dir / 'sparc_human_by_radius.csv',
    ]
    for p in cand:
        if p.exists():
            print(f"Loading SPARC data from {p}")
            df = pd.read_csv(p)
            df = _standardize_columns(df)
            req = {'galaxy', 'R_kpc', 'boundary_kpc', 'Vobs_kms', 'Vbar_kms', 'M_bary'}
            missing = req - set(df.columns)
            if missing:
                raise SystemExit(f"Missing required columns in {p}: {sorted(missing)}")
            return df

    # Fallback to predictions_by_radius if absolutely necessary
    p = Path('data/sparc_predictions_by_radius.csv')
    if p.exists():
        print(f"Loading SPARC data from {p} (note: must have M_bary elsewhere)")
        df = pd.read_csv(p)
        df = _standardize_columns(df)
        if 'M_bary' not in df.columns:
            raise SystemExit("sparc_predictions_by_radius.csv lacks M_bary; please use sparc_human_by_radius.csv")
        return df

    raise FileNotFoundError("Could not locate SPARC data; expected data/sparc_human_by_radius.csv. Run sparc_predict.py first.")


def split_train_test(df: pd.DataFrame, test_fraction: float = 0.2, random_seed: int = 42):
    galaxies = df['galaxy'].astype(str).unique()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(galaxies)
    n_test = max(1, int(round(len(galaxies) * float(test_fraction))))
    test_galaxies = galaxies[:n_test]
    train_galaxies = galaxies[n_test:]
    train_df = df[df['galaxy'].isin(train_galaxies)].copy()
    test_df = df[df['galaxy'].isin(test_galaxies)].copy()
    return train_df, test_df, train_galaxies, test_galaxies


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model: GravityModel, params: list[float] | np.ndarray, df: pd.DataFrame) -> float:
    # Compute predictions
    G_pred = model.compute_G(df['R_kpc'].to_numpy(float), df['boundary_kpc'].to_numpy(float),
                             df['M_bary'].to_numpy(float), params)
    V_pred = np.sqrt(np.maximum(0.0, G_pred)) * df['Vbar_kms'].to_numpy(float)
    mask = (df['Vobs_kms'] > 0).to_numpy(bool) & np.isfinite(df['Vobs_kms'].to_numpy(float)) & (df['Vbar_kms'] > 0).to_numpy(bool)
    if not np.any(mask):
        return float('inf')
    V_obs = df.loc[mask, 'Vobs_kms'].to_numpy(float)
    Vp = V_pred[mask]
    rel_error = np.abs(Vp - V_obs) / np.maximum(1e-9, V_obs)
    mean_error = float(np.mean(rel_error))
    p90 = float(np.percentile(rel_error, 90))
    # Score penalizes high tails
    score = mean_error + 0.5 * p90
    return score


def optimize_model(model: GravityModel, train_df: pd.DataFrame, n_iter: int = 50) -> GravityModel:
    print(f"\nOptimizing {model.name} Model...")
    def objective(x):
        return evaluate_model(model, x, train_df)
    result = differential_evolution(
        objective,
        model.param_bounds,
        maxiter=int(n_iter),
        popsize=15,
        seed=42,
        tol=0.01,
        polish=True,
        disp=False,
    )
    model.best_params = result.x
    model.best_score = float(result.fun)
    print(f"  Best score: {model.best_score:.4f}")
    print(f"  Best params: {dict(zip(model.param_names, model.best_params))}")
    return model


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_results(models: list[GravityModel], test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in models:
        if m.best_params is None:
            continue
        test_score = evaluate_model(m, m.best_params, test_df)
        G_pred = m.compute_G(test_df['R_kpc'].to_numpy(float), test_df['boundary_kpc'].to_numpy(float),
                             test_df['M_bary'].to_numpy(float), m.best_params)
        V_pred = np.sqrt(np.maximum(0.0, G_pred)) * test_df['Vbar_kms'].to_numpy(float)
        mask = (test_df['Vobs_kms'] > 0).to_numpy(bool) & np.isfinite(test_df['Vobs_kms'].to_numpy(float))
        V_obs = test_df.loc[mask, 'Vobs_kms'].to_numpy(float)
        Vp = V_pred[mask]
        rel = np.abs(Vp - V_obs) / np.maximum(1e-9, V_obs)
        within_10 = float(np.mean(rel < 0.10) * 100.0)
        within_20 = float(np.mean(rel < 0.20) * 100.0)
        med_err = float(np.median(rel) * 100.0)
        rows.append({
            'Model': m.name,
            'Train Score': m.best_score,
            'Test Score': test_score,
            'Within 10%': within_10,
            'Within 20%': within_20,
            'Median Error %': med_err,
            'Parameters': dict(zip(m.param_names, [float(v) for v in m.best_params])),
        })
    return pd.DataFrame(rows)


def plot_example_galaxies(best_model: GravityModel, df: pd.DataFrame, n_examples: int = 6, save_path: Path | str = 'model_examples.png') -> None:
    galaxies = df['galaxy'].astype(str).unique()
    rng = np.random.default_rng(42)
    ex = rng.choice(galaxies, size=min(n_examples, len(galaxies)), replace=False)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    axes = axes.flatten()
    for i, gal in enumerate(ex):
        if i >= len(axes):
            break
        ax = axes[i]
        gdf = df[df['galaxy'] == gal].sort_values('R_kpc')
        G_pred = best_model.compute_G(gdf['R_kpc'].to_numpy(float), gdf['boundary_kpc'].to_numpy(float),
                                      gdf['M_bary'].to_numpy(float), best_model.best_params)
        V_pred = np.sqrt(np.maximum(0.0, G_pred)) * gdf['Vbar_kms'].to_numpy(float)
        ax.scatter(gdf['R_kpc'], gdf['Vobs_kms'], s=20, alpha=0.7, label='Observed')
        ax.plot(gdf['R_kpc'], gdf['Vbar_kms'], 'g--', alpha=0.7, label='Baryonic (G=1)')
        ax.plot(gdf['R_kpc'], V_pred, 'r-', lw=2.0, label=f'{best_model.name} Pred')
        # Boundary marker
        try:
            boundary = float(gdf['boundary_kpc'].iloc[0])
            ax.axvline(boundary, color='gray', linestyle=':', alpha=0.5, label='Boundary')
        except Exception:
            pass
        # G axis
        ax2 = ax.twinx()
        ax2.plot(gdf['R_kpc'], G_pred, 'b:', alpha=0.5)
        ax2.set_ylabel('G factor', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Velocity (km/s)')
        ax.set_title(f"{gal}")
        if i == 0:
            ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Best Model: {best_model.name}', fontsize=14)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved example plots to {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description='Density-based gravity model comparison on SPARC')
    ap.add_argument('--data-dir', type=str, default='data', help='Directory with SPARC CSVs (prefer sparc_human_by_radius.csv)')
    ap.add_argument('--output-dir', type=str, default='model_results', help='Directory to write results')
    ap.add_argument('--n-iter', type=int, default=50, help='Optimization iterations for each model')
    ap.add_argument('--test-fraction', type=float, default=0.2, help='Fraction of galaxies for testing')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPARC Density-Based Gravity Model Optimizer")
    print("=" * 70)

    # Load and split
    df = load_sparc_data(args.data_dir)
    print(f"Loaded {len(df)} rows across {df['galaxy'].nunique()} galaxies")
    train_df, test_df, train_gals, test_gals = split_train_test(df, args.test_fraction)
    print(f"Training on {len(train_gals)} galaxies; testing on {len(test_gals)} galaxies")

    # Initialize models
    models: list[GravityModel] = [
        PureRadialModel(),
        DensityGradientModel(),
        PhaseTransitionModel(),
        SmoothShellModel(),
        HybridModel(),
    ]

    # Optimize
    for m in models:
        optimize_model(m, train_df, args.n_iter)

    # Compare
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    comp = analyze_results(models, test_df)
    print('\n' + (comp.to_string(index=False) if not comp.empty else 'No results'))
    comp.to_csv(out_dir / 'model_comparison.csv', index=False)

    if comp.empty:
        print("No successful model fits; exiting.")
        return

    # Select best by Train Score (you can switch to Test Score)
    best_idx = int(np.argmin([m.best_score for m in models]))
    best_model = models[best_idx]

    best_payload = {
        'model_name': best_model.name,
        'parameters': dict(zip(best_model.param_names, [float(v) for v in best_model.best_params])),
        'param_bounds': best_model.param_bounds,
        'train_score': float(best_model.best_score),
        'test_score': float(evaluate_model(best_model, best_model.best_params, test_df)),
    }
    with (out_dir / 'best_model.json').open('w') as f:
        json.dump(best_payload, f, indent=2)
    print(f"\nBest Model: {best_model.name}")
    print(f"Saved best model to {out_dir / 'best_model.json'}")

    # Plots
    plot_example_galaxies(best_model, test_df, save_path=out_dir / 'model_examples.png')

    print("\n" + "=" * 70)
    print("Done. See:")
    print(f"- {out_dir / 'model_comparison.csv'}")
    print(f"- {out_dir / 'best_model.json'}")
    print(f"- {out_dir / 'model_examples.png'}")


if __name__ == '__main__':
    main()
