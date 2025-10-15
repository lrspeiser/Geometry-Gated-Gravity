#!/usr/bin/env python3
"""
Run Hierarchical Inference with Mass-Scaled Coherence Length
=============================================================

This script performs the complete inference pipeline:
1. Load cluster metadata (Tier-1 + Tier-2 for training)
2. Run hierarchical MCMC to constrain (ℓ₀,⋆, γ, μ_A, σ_A)
3. Perform posterior predictive checks on hold-out clusters (A1689, MACS1149)
4. Save results, plots, and summary statistics

Usage:
------
    python scripts/run_mass_scaled_hierarchical_inference.py

Output:
-------
    results/mass_scaled_inference/
        ├── trace.nc                    # ArviZ posterior samples
        ├── posterior_summary.csv       # Summary statistics
        ├── posterior_plots.png         # Corner plots, pair plots
        ├── holdout_predictions.csv     # A1689 + MACS1149 predictions
        ├── holdout_plots.png           # Observed vs predicted
        └── INFERENCE_REPORT.md         # Human-readable summary

Author: GravityCalculator - Mass-Scaling Inference
Date: 2025-01-19
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.hierarchical_cluster_model_mass_scaled import (
    HierarchicalClusterModelMassScaled,
    GeometryPredictors,
    HyperParameters
)

try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False
    print("WARNING: ArviZ not available - install with: pip install arviz")


def load_cluster_data(catalog_path: str):
    """Load cluster metadata and split into training/holdout sets."""
    print("="*70)
    print("Loading Cluster Data")
    print("="*70)
    
    df = pd.read_csv(catalog_path)
    print(f"Total clusters in catalog: {len(df)}")
    
    # Training set: Tier-1 + Tier-2, excluding hold-outs
    train_mask = (df['tier'].isin([1, 2])) & (~df['cluster_name'].isin(['A1689', 'MACS1149']))
    df_train = df[train_mask].copy()
    
    # Hold-out set: A1689 + MACS1149
    df_holdout = df[df['cluster_name'].isin(['A1689', 'MACS1149'])].copy()
    
    print(f"\nTraining clusters (Tier-1 + Tier-2): {len(df_train)}")
    print(df_train[['cluster_name', 'R_500_kpc', 'theta_E_obs_arcsec', 'tier']].to_string(index=False))
    
    print(f"\nHold-out clusters: {len(df_holdout)}")
    print(df_holdout[['cluster_name', 'R_500_kpc', 'theta_E_obs_arcsec']].to_string(index=False))
    
    return df_train, df_holdout


def build_predictors(df: pd.DataFrame):
    """Convert DataFrame to GeometryPredictors list."""
    predictors = []
    for _, row in df.iterrows():
        predictors.append(GeometryPredictors(
            R_500=row['R_500_kpc'],
            M_500=row['M_500_Msun'],
            z=row['z_lens'],
            cool_core=(row['dynamical_state'] == 'relaxed'),
            c_500=3.0,  # Default concentration (could be cluster-specific)
            T_X=row.get('TX_central_keV', 8.0)  # Use catalog value or default
        ))
    return predictors


def run_hierarchical_inference(
    df_train: pd.DataFrame,
    output_dir: Path,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 4
):
    """Run hierarchical MCMC inference with mass-scaling."""
    print("\n" + "="*70)
    print("Hierarchical Inference with Mass-Scaling")
    print("="*70)
    
    # Build predictors and observations
    predictors = build_predictors(df_train)
    observations = {
        'theta_E': df_train['theta_E_obs_arcsec'].values,
        'theta_E_err': df_train['theta_E_err_arcsec'].values
    }
    
    print(f"\nInitializing model...")
    print(f"  - {len(predictors)} training clusters")
    print(f"  - Mass-scaling: ℓ₀(M) = ℓ₀,⋆ × (R₅₀₀/1Mpc)^γ")
    print(f"  - Population parameters: (ℓ₀,⋆, γ, μ_A, σ_A)")
    
    # Initialize model
    model = HierarchicalClusterModelMassScaled(
        predictors=predictors,
        observations=observations,
        use_pymc=True,
        include_secondary_effects=False  # Simple mass-scaling only
    )
    
    # Run MCMC
    print(f"\nRunning MCMC sampling...")
    print(f"  - Draws: {n_samples} per chain × {n_chains} chains = {n_samples * n_chains} total")
    print(f"  - Tune: {n_tune} steps")
    print(f"  - Sampler: NUTS with target_accept=0.95")
    
    trace = model.fit_mcmc(
        n_samples=n_samples,
        n_tune=n_tune,
        n_chains=n_chains,
        target_accept=0.95
    )
    
    # Save trace
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.nc"
    print(f"\nSaving posterior samples to {trace_path}")
    trace.to_netcdf(trace_path)
    
    # Save summary statistics
    summary = az.summary(trace, var_names=['ell0_star_pop', 'gamma_pop', 'mu_A_pop', 'sigma_A_pop'])
    summary_path = output_dir / "posterior_summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved summary statistics to {summary_path}")
    
    # Generate diagnostic plots
    if HAS_ARVIZ:
        print(f"\nGenerating posterior diagnostic plots...")
        
        # Trace plots
        fig, axes = plt.subplots(4, 2, figsize=(12, 10))
        az.plot_trace(trace, var_names=['ell0_star_pop', 'gamma_pop', 'mu_A_pop', 'sigma_A_pop'], axes=axes)
        plt.tight_layout()
        plt.savefig(output_dir / "posterior_trace.png", dpi=150)
        plt.close()
        
        # Pair plot
        fig = plt.figure(figsize=(10, 10))
        az.plot_pair(
            trace,
            var_names=['ell0_star_pop', 'gamma_pop', 'mu_A_pop'],
            kind='kde',
            marginals=True
        )
        plt.savefig(output_dir / "posterior_pairs.png", dpi=150)
        plt.close()
        
        # Posterior predictive check
        fig, ax = plt.subplots(figsize=(10, 6))
        az.plot_ppc(trace, ax=ax)
        ax.set_xlabel("Einstein Radius (arcsec)")
        ax.set_title("Posterior Predictive Check: Training Clusters")
        plt.tight_layout()
        plt.savefig(output_dir / "posterior_predictive_check.png", dpi=150)
        plt.close()
        
        print(f"  - posterior_trace.png")
        print(f"  - posterior_pairs.png")
        print(f"  - posterior_predictive_check.png")
    
    return model, trace


def perform_holdout_validation(
    model: HierarchicalClusterModelMassScaled,
    df_holdout: pd.DataFrame,
    output_dir: Path
):
    """Perform posterior predictive checks on hold-out clusters."""
    print("\n" + "="*70)
    print("Hold-Out Validation: A1689 & MACS1149")
    print("="*70)
    
    results = []
    
    for _, row in df_holdout.iterrows():
        cluster_name = row['cluster_name']
        theta_E_obs = row['theta_E_obs_arcsec']
        theta_E_err = row['theta_E_err_arcsec']
        
        # Build predictor
        pred = GeometryPredictors(
            R_500=row['R_500_kpc'],
            M_500=row['M_500_Msun'],
            z=row['z_lens'],
            cool_core=(row['dynamical_state'] == 'relaxed')
        )
        
        # Predict
        theta_E_med, theta_E_lo, theta_E_hi = model.predict_holdout(pred, use_posterior=True, n_samples=2000)
        
        # Compute Z-score
        theta_E_err_pred = (theta_E_hi - theta_E_lo) / 2.0
        z_score = (theta_E_obs - theta_E_med) / np.sqrt(theta_E_err**2 + theta_E_err_pred**2)
        
        # Store results
        results.append({
            'cluster_name': cluster_name,
            'R_500_kpc': row['R_500_kpc'],
            'z_lens': row['z_lens'],
            'theta_E_obs_arcsec': theta_E_obs,
            'theta_E_obs_err_arcsec': theta_E_err,
            'theta_E_pred_median_arcsec': theta_E_med,
            'theta_E_pred_16pct_arcsec': theta_E_lo,
            'theta_E_pred_84pct_arcsec': theta_E_hi,
            'z_score': z_score,
            'within_1sigma': abs(z_score) < 1.0,
            'within_2sigma': abs(z_score) < 2.0
        })
        
        # Print results
        print(f"\n{cluster_name}:")
        print(f"  R₅₀₀: {row['R_500_kpc']:.0f} kpc")
        print(f"  Observed: θ_E = {theta_E_obs:.1f} ± {theta_E_err:.1f}\"")
        print(f"  Predicted: θ_E = {theta_E_med:.1f} [{theta_E_lo:.1f}, {theta_E_hi:.1f}]\"")
        print(f"  Z-score: {z_score:.2f} ({'' if abs(z_score) < 2 else 'OUT'}SIDE 2σ)")
    
    # Save results
    df_results = pd.DataFrame(results)
    results_path = output_dir / "holdout_predictions.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\nSaved hold-out predictions to {results_path}")
    
    # Plot observed vs predicted
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for res in results:
        color = 'green' if res['within_2sigma'] else 'red'
        ax.errorbar(
            res['theta_E_obs_arcsec'],
            res['theta_E_pred_median_arcsec'],
            xerr=res['theta_E_obs_err_arcsec'],
            yerr=[(res['theta_E_pred_median_arcsec'] - res['theta_E_pred_16pct_arcsec']),
                  (res['theta_E_pred_84pct_arcsec'] - res['theta_E_pred_median_arcsec'])],
            fmt='o',
            color=color,
            markersize=10,
            label=res['cluster_name']
        )
    
    # 1:1 line
    ax_min = min(df_results['theta_E_obs_arcsec'].min(), df_results['theta_E_pred_median_arcsec'].min()) * 0.9
    ax_max = max(df_results['theta_E_obs_arcsec'].max(), df_results['theta_E_pred_median_arcsec'].max()) * 1.1
    ax.plot([ax_min, ax_max], [ax_min, ax_max], 'k--', alpha=0.5, label='1:1')
    
    ax.set_xlabel("Observed θ_E (arcsec)", fontsize=12)
    ax.set_ylabel("Predicted θ_E (arcsec)", fontsize=12)
    ax.set_title("Hold-Out Validation: A1689 & MACS1149", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / "holdout_validation.png", dpi=150)
    plt.close()
    
    print(f"Saved hold-out validation plot to {output_dir / 'holdout_validation.png'}")
    
    return df_results


def generate_inference_report(
    trace,
    df_train: pd.DataFrame,
    df_holdout_results: pd.DataFrame,
    output_dir: Path
):
    """Generate human-readable inference report."""
    print("\n" + "="*70)
    print("Generating Inference Report")
    print("="*70)
    
    # Extract posterior statistics
    ell0_star_post = trace.posterior['ell0_star_pop'].values.flatten()
    gamma_post = trace.posterior['gamma_pop'].values.flatten()
    mu_A_post = trace.posterior['mu_A_pop'].values.flatten()
    sigma_A_post = trace.posterior['sigma_A_pop'].values.flatten()
    
    report = f"""# Mass-Scaled Hierarchical Inference Report

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model:** Hierarchical Bayesian with mass-scaled coherence length  
**Parameterization:** ℓ₀(M) = ℓ₀,⋆ × (R₅₀₀/1Mpc)^γ

---

## 1. Training Data

**Clusters Used:** {len(df_train)} (Tier-1 + Tier-2)

| Cluster | R₅₀₀ (kpc) | θ_E obs (") | Tier |
|---------|-----------|-------------|------|
"""
    
    for _, row in df_train.iterrows():
        report += f"| {row['cluster_name']} | {row['R_500_kpc']:.0f} | {row['theta_E_obs_arcsec']:.1f} ± {row['theta_E_err_arcsec']:.1f} | {row['tier']} |\n"
    
    report += f"""
---

## 2. Posterior Results

### Mass-Scaling Parameters

| Parameter | Median | 16th pct | 84th pct | Interpretation |
|-----------|--------|----------|----------|----------------|
| **ℓ₀,⋆** | {np.median(ell0_star_post):.1f} kpc | {np.percentile(ell0_star_post, 16):.1f} | {np.percentile(ell0_star_post, 84):.1f} | Coherence at 1 Mpc scale |
| **γ** | {np.median(gamma_post):.3f} | {np.percentile(gamma_post, 16):.3f} | {np.percentile(gamma_post, 84):.3f} | Mass-scaling exponent |

**Interpretation of γ:**
- γ = 0: Fixed coherence scale (mass-independent)
- γ ≈ 0.5: Sub-linear scaling (ℓ₀ ∝ √R₅₀₀)
- γ ≈ 1.0: Linear scaling (ℓ₀ ∝ R₅₀₀, self-similar)

**Result:** γ = {np.median(gamma_post):.3f} ± {np.std(gamma_post):.3f}
"""
    
    # Test if gamma is significantly > 0
    gamma_greater_than_zero = np.mean(gamma_post > 0) * 100
    report += f"\n**Evidence for mass-scaling:** {gamma_greater_than_zero:.1f}% of posterior has γ > 0\n"
    
    if gamma_greater_than_zero > 95:
        report += "\n✅ **Strong evidence for mass-dependent coherence** (>95% posterior support)\n"
    elif gamma_greater_than_zero > 68:
        report += "\n⚠️ **Moderate evidence for mass-dependent coherence** (68-95% posterior support)\n"
    else:
        report += "\n❌ **No evidence for mass-dependent coherence** (<68% posterior support)\n"
    
    report += f"""
### Amplitude Parameters

| Parameter | Median | 16th pct | 84th pct |
|-----------|--------|----------|----------|
| **μ_A** | {np.median(mu_A_post):.3f} | {np.percentile(mu_A_post, 16):.3f} | {np.percentile(mu_A_post, 84):.3f} |
| **σ_A** | {np.median(sigma_A_post):.3f} | {np.percentile(sigma_A_post, 16):.3f} | {np.percentile(sigma_A_post, 84):.3f} |

---

## 3. Hold-Out Validation

**Test Clusters:** A1689, MACS1149 (not used in training)

| Cluster | R₅₀₀ (kpc) | θ_E obs (") | θ_E pred (") | Z-score | Within 2σ? |
|---------|-----------|-------------|--------------|---------|------------|
"""
    
    for _, row in df_holdout_results.iterrows():
        within = "✅" if row['within_2sigma'] else "❌"
        report += f"| {row['cluster_name']} | {row['R_500_kpc']:.0f} | {row['theta_E_obs_arcsec']:.1f} ± {row['theta_E_obs_err_arcsec']:.1f} | {row['theta_E_pred_median_arcsec']:.1f} [{row['theta_E_pred_16pct_arcsec']:.1f}, {row['theta_E_pred_84pct_arcsec']:.1f}] | {row['z_score']:.2f} | {within} |\n"
    
    n_within_2sigma = df_holdout_results['within_2sigma'].sum()
    report += f"\n**Summary:** {n_within_2sigma}/{len(df_holdout_results)} hold-out clusters within 2σ\n"
    
    if n_within_2sigma == len(df_holdout_results):
        report += "\n✅ **Excellent hold-out performance** - all predictions within 2σ\n"
    elif n_within_2sigma >= len(df_holdout_results) * 0.68:
        report += "\n✅ **Good hold-out performance** - consistent with Gaussian errors\n"
    else:
        report += "\n⚠️ **Poor hold-out performance** - systematic errors present\n"
    
    report += f"""
---

## 4. Cross-Scale Predictions

Using the inferred (ℓ₀,⋆, γ) = ({np.median(ell0_star_post):.0f} kpc, {np.median(gamma_post):.2f}), we can predict ℓ₀ for any halo mass:

| System | M₅₀₀ (M☉) | R₅₀₀ (kpc) | Predicted ℓ₀ (kpc) |
|--------|----------|-----------|-------------------|
| Dwarf galaxy | 10⁹ | 10 | {np.median(ell0_star_post) * (10/1000)**np.median(gamma_post):.1f} |
| Milky Way | 10¹² | 200 | {np.median(ell0_star_post) * (200/1000)**np.median(gamma_post):.1f} |
| **Cluster (typical)** | **10¹⁴** | **1000** | **{np.median(ell0_star_post):.1f}** (pivot) |
| Massive cluster | 10¹⁵ | 1500 | {np.median(ell0_star_post) * (1500/1000)**np.median(gamma_post):.1f} |
| Supercluster | 10¹⁶ | 5000 | {np.median(ell0_star_post) * (5000/1000)**np.median(gamma_post):.1f} |

These predictions can be tested with:
- **Galaxy rotation curves** (dwarf/spiral galaxies)
- **Weak lensing** (massive clusters)
- **Cosmic web simulations** (superclusters)

---

## 5. Comparison with Modified Gravity Theories

| Theory | Expected γ | Our Result | Consistency |
|--------|-----------|-----------|-------------|
| Many-paths (fixed scale) | γ = 0 | γ = {np.median(gamma_post):.2f} ± {np.std(gamma_post):.2f} | {('✅ Consistent' if abs(np.median(gamma_post)) < 0.2 else '❌ Inconsistent')} |
| Many-paths (self-similar) | γ ≈ 1 | γ = {np.median(gamma_post):.2f} ± {np.std(gamma_post):.2f} | {('✅ Consistent' if abs(np.median(gamma_post) - 1.0) < 0.3 else '❌ Inconsistent')} |
| MOND | N/A | N/A | N/A (no coherence length) |
| f(R) gravity | γ = 0 | γ = {np.median(gamma_post):.2f} ± {np.std(gamma_post):.2f} | {('✅ Consistent' if abs(np.median(gamma_post)) < 0.2 else '❌ Inconsistent')} |
| Emergent gravity | γ ≈ 0.5-1 | γ = {np.median(gamma_post):.2f} ± {np.std(gamma_post):.2f} | {('✅ Consistent' if 0.3 < np.median(gamma_post) < 1.2 else '❌ Inconsistent')} |

---

## 6. Next Steps

### Scientific
1. **Test cross-scale predictions** with galaxy rotation curves
2. **Add weak-lensing profiles** (γ_t(R)) to likelihood
3. **Model complex mergers** (e.g., MACS0717) with multi-component Σ
4. **Constrain secondary effects** (temperature, morphology modulation of ℓ₀)

### Technical
1. Replace mock likelihood with real lensing forward model
2. Run longer chains (5000+ draws) for publication-quality posteriors
3. Perform model comparison (WAIC, LOO-CV) vs fixed-ℓ₀ model
4. Generate publication-ready figures

---

## 7. Files Generated

- `trace.nc` - Full posterior samples (ArviZ format)
- `posterior_summary.csv` - Summary statistics
- `posterior_trace.png` - MCMC trace diagnostics
- `posterior_pairs.png` - Parameter correlations
- `posterior_predictive_check.png` - Training cluster fit quality
- `holdout_predictions.csv` - Hold-out cluster predictions
- `holdout_validation.png` - Observed vs predicted plot
- `INFERENCE_REPORT.md` - This report

---

**Generated by:** `scripts/run_mass_scaled_hierarchical_inference.py`  
**Contact:** GravityCalculator - Mass-Scaling Extension
"""
    
    # Save report
    report_path = output_dir / "INFERENCE_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Saved inference report to {report_path}")
    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")


def main():
    """Main execution function."""
    # Paths
    base_dir = Path(__file__).parent.parent
    catalog_path = base_dir / "data" / "clusters" / "master_catalog.csv"
    output_dir = base_dir / "results" / "mass_scaled_inference"
    
    # Load data
    df_train, df_holdout = load_cluster_data(str(catalog_path))
    
    # Run hierarchical inference
    model, trace = run_hierarchical_inference(
        df_train,
        output_dir,
        n_samples=2000,
        n_tune=1000,
        n_chains=4
    )
    
    # Perform hold-out validation
    df_holdout_results = perform_holdout_validation(model, df_holdout, output_dir)
    
    # Generate report
    generate_inference_report(trace, df_train, df_holdout_results, output_dir)
    
    print("\n✅ All tasks completed successfully!")
    print(f"📁 Results: {output_dir}")


if __name__ == "__main__":
    main()
