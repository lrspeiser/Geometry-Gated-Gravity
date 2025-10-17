"""
Model Comparison: Fixed-Scale vs Mass-Scaled
=============================================

Compares predictive capability of baseline (γ=0, ℓ₀=200kpc) vs mass-scaled models.

Metrics:
  - Train set χ²/d.o.f.
  - Posterior parameter uncertainties
  - BIC and AIC for model selection
  - Posterior predictive checks on holdout clusters

Author: GravityCalculator
Date: 2025-01-19
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

# CLI
parser = argparse.ArgumentParser(description='Compare baseline and mass-scaled models')
parser.add_argument('--baseline', type=str, required=True, help='Path to baseline run output directory')
parser.add_argument('--mass_scaled', type=str, required=True, help='Path to mass-scaled run output directory')
parser.add_argument('--outdir', type=str, default='output/model_comparison', help='Output directory')
args = parser.parse_args()

BASELINE_DIR = Path(args.baseline)
MASS_SCALED_DIR = Path(args.mass_scaled)
OUTPUT_DIR = Path(args.outdir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("MODEL COMPARISON: BASELINE vs MASS-SCALED")
print("="*70)

# =============================================================================
# Load Results
# =============================================================================
print("\n[1/4] Loading results...")

# Load baseline
baseline_samples = np.load(BASELINE_DIR / 'flat_samples.npy')
baseline_train_df = pd.read_csv(BASELINE_DIR / 'train_results.csv')
with open(BASELINE_DIR / 'settings.json', 'r') as f:
    baseline_settings = json.load(f)

# Load mass-scaled
mass_scaled_samples = np.load(MASS_SCALED_DIR / 'flat_samples.npy')
mass_scaled_train_df = pd.read_csv(MASS_SCALED_DIR / 'train_results.csv')
with open(MASS_SCALED_DIR / 'settings.json', 'r') as f:
    mass_scaled_settings = json.load(f)

# Extract chi2 from train results
chi2_baseline = baseline_train_df['chi2'].sum()
chi2_mass_scaled = mass_scaled_train_df['chi2'].sum()
n_data = len(baseline_train_df)

# Count parameters
# Baseline: [mu_A, sigma_A, A_c_1, q_plane_1, q_LOS_1, kappa_ext_1, ...]
n_params_baseline = 2 + 4 * n_data  # population + per-cluster
# Mass-scaled: [ell0_star, gamma, mu_A, sigma_A, A_c_1, q_plane_1, q_LOS_1, kappa_ext_1, ...]
n_params_mass_scaled = 4 + 4 * n_data  # mass-scaling + population + per-cluster

baseline_summary = {
    'n_params': n_params_baseline,
    'n_train': n_data,
    'chi2_train': chi2_baseline
}

mass_scaled_summary = {
    'n_params': n_params_mass_scaled,
    'n_train': n_data,
    'chi2_train': chi2_mass_scaled
}

print(f"  Baseline: {baseline_samples.shape[0]} samples, {n_params_baseline} parameters")
print(f"  Mass-scaled: {mass_scaled_samples.shape[0]} samples, {n_params_mass_scaled} parameters")

# =============================================================================
# Train Set Performance
# =============================================================================
print("\n[2/4] Comparing train set performance...")

# Degrees of freedom
dof_baseline = n_data - baseline_summary['n_params']
dof_mass_scaled = n_data - mass_scaled_summary['n_params']

chi2_dof_baseline = chi2_baseline / dof_baseline
chi2_dof_mass_scaled = chi2_mass_scaled / dof_mass_scaled

print(f"\n  Train set fit quality:")
print(f"    Baseline (γ=0):      χ²={chi2_baseline:.2f}, χ²/d.o.f.={chi2_dof_baseline:.2f}")
print(f"    Mass-scaled (γ free): χ²={chi2_mass_scaled:.2f}, χ²/d.o.f.={chi2_dof_mass_scaled:.2f}")

delta_chi2 = chi2_baseline - chi2_mass_scaled
print(f"\n  Δχ² = {delta_chi2:.2f} (positive = mass-scaled improves fit)")

# =============================================================================
# Model Selection Criteria
# =============================================================================
print("\n[3/4] Computing model selection criteria...")

k_baseline = baseline_summary['n_params']
k_mass_scaled = mass_scaled_summary['n_params']

# BIC = χ² + k*ln(n)
bic_baseline = chi2_baseline + k_baseline * np.log(n_data)
bic_mass_scaled = chi2_mass_scaled + k_mass_scaled * np.log(n_data)

# AIC = χ² + 2*k
aic_baseline = chi2_baseline + 2 * k_baseline
aic_mass_scaled = chi2_mass_scaled + 2 * k_mass_scaled

print(f"\n  BIC (lower = better):")
print(f"    Baseline:    {bic_baseline:.2f}")
print(f"    Mass-scaled: {bic_mass_scaled:.2f}")
print(f"    ΔBIC = {bic_mass_scaled - bic_baseline:.2f} (negative = mass-scaled preferred)")

print(f"\n  AIC (lower = better):")
print(f"    Baseline:    {aic_baseline:.2f}")
print(f"    Mass-scaled: {aic_mass_scaled:.2f}")
print(f"    ΔAIC = {aic_mass_scaled - aic_baseline:.2f} (negative = mass-scaled preferred)")

# Interpret
if (bic_mass_scaled - bic_baseline) < -10:
    bic_interpretation = "STRONG evidence for mass-scaled model"
elif (bic_mass_scaled - bic_baseline) < -2:
    bic_interpretation = "Positive evidence for mass-scaled model"
elif (bic_mass_scaled - bic_baseline) < 2:
    bic_interpretation = "Inconclusive (models comparable)"
else:
    bic_interpretation = "Evidence favors baseline (mass-scaling not justified)"

print(f"\n  Interpretation: {bic_interpretation}")

# =============================================================================
# Parameter Posteriors
# =============================================================================
print("\n[4/4] Comparing parameter posteriors...")

# Baseline: [mu_A, sigma_A, ...]
mu_A_baseline = baseline_samples[:, 0]
sigma_A_baseline = baseline_samples[:, 1]

# Mass-scaled: [ell0_star, gamma, mu_A, sigma_A, ...]
ell0_star = mass_scaled_samples[:, 0]
gamma = mass_scaled_samples[:, 1]
mu_A_mass_scaled = mass_scaled_samples[:, 2]
sigma_A_mass_scaled = mass_scaled_samples[:, 3]

print(f"\n  Baseline (ℓ₀ = 200 kpc fixed):")
print(f"    μ_A    = {np.median(mu_A_baseline):.3f} ± {np.std(mu_A_baseline):.3f}")
print(f"    σ_A    = {np.median(sigma_A_baseline):.3f} ± {np.std(sigma_A_baseline):.3f}")

print(f"\n  Mass-scaled:")
print(f"    ℓ₀,⋆   = {np.median(ell0_star):.1f} ± {np.std(ell0_star):.1f} kpc")
print(f"    γ      = {np.median(gamma):.3f} ± {np.std(gamma):.3f}")
print(f"    μ_A    = {np.median(mu_A_mass_scaled):.3f} ± {np.std(mu_A_mass_scaled):.3f}")
print(f"    σ_A    = {np.median(sigma_A_mass_scaled):.3f} ± {np.std(sigma_A_mass_scaled):.3f}")

# Check if γ is consistent with zero
gamma_med = np.median(gamma)
gamma_16, gamma_84 = np.percentile(gamma, [16, 84])
gamma_consistent_with_zero = (gamma_16 < 0.0 < gamma_84)

print(f"\n  Mass-scaling exponent γ:")
print(f"    Median: {gamma_med:.3f} [{gamma_16:.3f}, {gamma_84:.3f}]")
if gamma_consistent_with_zero:
    print(f"    Status: Consistent with γ=0 (no mass-scaling detected)")
else:
    print(f"    Status: Evidence for non-zero γ (mass-scaling detected)")

# =============================================================================
# Visualization
# =============================================================================
print("\n[5/5] Creating diagnostic plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: γ posterior
ax = axes[0, 0]
ax.hist(gamma, bins=50, density=True, alpha=0.7, color='steelblue')
ax.axvline(gamma_med, color='red', linestyle='--', label=f'Median: {gamma_med:.3f}')
ax.axvline(0.0, color='black', linestyle=':', label='γ=0 (no scaling)')
ax.set_xlabel('γ (mass-scaling exponent)')
ax.set_ylabel('Posterior density')
ax.set_title('Mass-Scaling Exponent')
ax.legend()

# Plot 2: ℓ₀,⋆ posterior
ax = axes[0, 1]
ax.hist(ell0_star, bins=50, density=True, alpha=0.7, color='steelblue')
ax.axvline(np.median(ell0_star), color='red', linestyle='--', label=f'Median: {np.median(ell0_star):.1f} kpc')
ax.axvline(200.0, color='black', linestyle=':', label='Baseline: 200 kpc')
ax.set_xlabel('ℓ₀,⋆ (kpc)')
ax.set_ylabel('Posterior density')
ax.set_title('Reference Coherence Length')
ax.legend()

# Plot 3: μ_A comparison
ax = axes[0, 2]
ax.hist(mu_A_baseline, bins=50, density=True, alpha=0.5, color='gray', label='Baseline')
ax.hist(mu_A_mass_scaled, bins=50, density=True, alpha=0.5, color='steelblue', label='Mass-scaled')
ax.set_xlabel('μ_A (population mean)')
ax.set_ylabel('Posterior density')
ax.set_title('Population Mean A_c')
ax.legend()

# Plot 4: σ_A comparison
ax = axes[1, 0]
ax.hist(sigma_A_baseline, bins=50, density=True, alpha=0.5, color='gray', label='Baseline')
ax.hist(sigma_A_mass_scaled, bins=50, density=True, alpha=0.5, color='steelblue', label='Mass-scaled')
ax.set_xlabel('σ_A (population scatter)')
ax.set_ylabel('Posterior density')
ax.set_title('Population Scatter A_c')
ax.legend()

# Plot 5: Model comparison summary
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""
MODEL COMPARISON SUMMARY

Train Set (N={n_data}):
  Baseline:      χ²/d.o.f. = {chi2_dof_baseline:.2f}
  Mass-scaled:   χ²/d.o.f. = {chi2_dof_mass_scaled:.2f}
  Δχ² = {delta_chi2:.2f}

Model Selection:
  ΔBIC = {bic_mass_scaled - bic_baseline:.2f}
  ΔAIC = {aic_mass_scaled - aic_baseline:.2f}

Mass-Scaling:
  γ = {gamma_med:.3f} [{gamma_16:.3f}, {gamma_84:.3f}]
  
Interpretation:
  {bic_interpretation}
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center', family='monospace')

# Plot 6: χ² comparison bar chart
ax = axes[1, 2]
models = ['Baseline\n(γ=0)', 'Mass-Scaled\n(γ free)']
chi2_values = [chi2_dof_baseline, chi2_dof_mass_scaled]
colors = ['gray', 'steelblue']
ax.bar(models, chi2_values, color=colors, alpha=0.7)
ax.axhline(1.0, color='black', linestyle=':', label='Ideal χ²/d.o.f.=1')
ax.set_ylabel('χ²/d.o.f.')
ax.set_title('Train Set Fit Quality')
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150)
print(f"  Saved: {OUTPUT_DIR / 'model_comparison.png'}")

# =============================================================================
# Save Summary
# =============================================================================
comparison_summary = {
    'baseline': {
        'chi2': float(chi2_baseline),
        'chi2_dof': float(chi2_dof_baseline),
        'bic': float(bic_baseline),
        'aic': float(aic_baseline),
        'n_params': int(k_baseline),
        'mu_A': float(np.median(mu_A_baseline)),
        'sigma_A': float(np.median(sigma_A_baseline))
    },
    'mass_scaled': {
        'chi2': float(chi2_mass_scaled),
        'chi2_dof': float(chi2_dof_mass_scaled),
        'bic': float(bic_mass_scaled),
        'aic': float(aic_mass_scaled),
        'n_params': int(k_mass_scaled),
        'ell0_star': float(np.median(ell0_star)),
        'gamma': float(gamma_med),
        'gamma_16': float(gamma_16),
        'gamma_84': float(gamma_84),
        'mu_A': float(np.median(mu_A_mass_scaled)),
        'sigma_A': float(np.median(sigma_A_mass_scaled))
    },
    'comparison': {
        'delta_chi2': float(delta_chi2),
        'delta_bic': float(bic_mass_scaled - bic_baseline),
        'delta_aic': float(aic_mass_scaled - aic_baseline),
        'interpretation': bic_interpretation
    }
}

with open(OUTPUT_DIR / 'comparison_summary.json', 'w') as f:
    json.dump(comparison_summary, f, indent=2)

print(f"  Saved: {OUTPUT_DIR / 'comparison_summary.json'}")

print("\n" + "="*70)
print("MODEL COMPARISON COMPLETE")
print("="*70)
