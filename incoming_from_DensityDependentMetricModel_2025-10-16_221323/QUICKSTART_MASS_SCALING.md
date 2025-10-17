# Quick Start: Mass-Scaled Hierarchical Inference

**Created**: 2025-01-XX  
**Status**: ✅ Ready to run  
**Data**: `data/cluster_lensing_catalog.csv` (20 CLASH clusters)

---

## What You're About to Run

This analysis tests whether the **coherence length of Sigma-Gravity scales with halo mass**:

```
ℓ₀(M) = ℓ₀,⋆ × (R₅₀₀ / 1 Mpc)^γ
```

- **γ = 0**: Scale-invariant (same ℓ₀ for all masses)
- **γ > 0**: Coherence length grows with mass
- **γ < 0**: Coherence length shrinks with mass (unlikely)

We'll fit both models and compare via WAIC/LOOIC.

---

## Prerequisites

```bash
# Install required packages
pip install pymc arviz pandas numpy matplotlib astropy

# Verify installation
python -c "import pymc; print(f'PyMC {pymc.__version__}')"
# Should print: PyMC 5.x.x
```

**Hardware**: 
- Minimum: 8 GB RAM, 4 CPU cores
- Recommended: 16 GB RAM, 8+ cores (for faster sampling)
- Runtime: ~2-4 hours per model

---

## Step 1: Validate Catalog

```bash
cd C:\Users\henry\Documents\GitHub\DensityDependentMetricModel

python scripts/validate_cluster_catalog.py
```

**Expected output**:
```
✓ Schema validation passed
✓ Physics checks passed
...
✓ Validation complete!
  - N_analysis = 18 clusters
```

**Outputs**:
- `output/validation/catalog_validation.png` (4-panel diagnostic plot)

**What to check**:
- Mass-radius relation follows M ∝ R³ trend
- Einstein radii are 25-55 arcsec (typical for clusters)
- No major outliers

---

## Step 2: Run Mass-Scaled Model (γ free)

```bash
python scripts/run_mass_scaled_hierarchical_inference.py \
  --tiers 1,2 --exclude MACSJ0717.5+3745 \
  --use-triaxial 1 \
  --draws 4000 --tune 2000 --chains 4 --target_accept 0.9 \
  --out output/mass_scaled/
```

**What this does**:
1. Loads 18 clusters (tiers 1+2, exclude MACS0717 merger)
2. Fits hierarchical model with free γ
3. Samples posterior using NUTS (4 chains × 4000 draws)
4. Saves trace to `output/mass_scaled/trace.netcdf`

**Runtime**: ~2-4 hours

**Expected output** (end of log):
```
Sampling 4 chains: 100%|██████████| 24000/24000 [2:15:32<00:00,  2.95draws/s]
✓ Sampling complete!
  - r_hat < 1.01 for all parameters
  - ESS > 1000 for all parameters
...
Posterior summary:
  μ_A: 16.8 [16.2, 17.4]
  σ_A: 0.7 [0.5, 1.1]
  ℓ₀,⋆: 195 kpc [170, 230]
  γ: 0.32 [0.18, 0.48]  ← KEY RESULT
```

**Key checks**:
- `r_hat < 1.01` (good convergence)
- `ESS > 1000` (sufficient effective samples)
- `γ posterior` doesn't overlap zero → evidence for mass-scaling

---

## Step 3: Run Fixed-Scale Model (γ = 0)

```bash
python scripts/run_mass_scaled_hierarchical_inference.py \
  --tiers 1,2 --exclude MACSJ0717.5+3745 \
  --use-triaxial 1 \
  --fix-gamma 0 \
  --draws 4000 --tune 2000 --chains 4 --target_accept 0.9 \
  --out output/fixed_scale/
```

**What this does**:
- Same as Step 2, but forces γ = 0 (scale-invariant)
- Used for model comparison

**Runtime**: ~2-4 hours

---

## Step 4: Model Comparison

```python
import arviz as az
import numpy as np

# Load traces
trace_mass = az.from_netcdf("output/mass_scaled/trace.netcdf")
trace_fixed = az.from_netcdf("output/fixed_scale/trace.netcdf")

# WAIC comparison
waic_mass = az.waic(trace_mass, pointwise=True)
waic_fixed = az.waic(trace_fixed, pointwise=True)

delta_waic = waic_mass.elpd_waic - waic_fixed.elpd_waic
delta_se = np.sqrt(waic_mass.se**2 + waic_fixed.se**2)

print(f"ΔWAIC = {delta_waic:.1f} ± {delta_se:.1f}")

# Decision rule
if delta_waic > 4:
    print("✓ Strong evidence for mass-scaling (γ ≠ 0)")
elif delta_waic < -4:
    print("✓ Strong evidence for scale-invariance (γ = 0)")
else:
    print("⚠ Models are comparable (ΔWAIC < 4)")
    
# Check γ posterior
gamma_post = trace_mass.posterior['gamma'].values.flatten()
print(f"\nγ posterior: {np.median(gamma_post):.2f} [{np.percentile(gamma_post, 16):.2f}, {np.percentile(gamma_post, 84):.2f}]")

if np.percentile(gamma_post, 16) > 0:
    print("  → γ > 0 at 68% credibility (evidence for mass-scaling)")
elif np.percentile(gamma_post, 84) < 0:
    print("  → γ < 0 at 68% credibility (shrinking coherence)")
else:
    print("  → γ consistent with zero (scale-invariant)")
```

**Interpretation**:

| Outcome | Conclusion |
|---------|------------|
| ΔWAIC > 4 and γ₁₆ > 0 | **Mass-scaling confirmed** (coherence grows with mass) |
| ΔWAIC < 4 and \|γ\| < 0.1 | **Scale-invariant** (same ℓ₀ for all masses) |
| ΔWAIC < 4 and γ uncertain | **Inconclusive** (need more data or tighter priors) |

---

## Step 5: Blind Validation (Hold-Out Test)

```bash
python scripts/run_holdout_validation.py \
  --posterior output/mass_scaled/trace.netcdf \
  --holdout Abell2261,MACSJ1149.5+2223 \
  --use-mass-scaling 1
```

**What this does**:
1. Removes 2 clusters from fit
2. Predicts their θ_E using posterior from remaining 16 clusters
3. Compares predictions to observations

**Pass criteria**:
- Median \|Δθ_E\| / θ_E,obs < 20%
- ≥68% hold-outs inside 68% posterior predictive credible interval
- No systematic bias (all positive or all negative)

**Expected output**:
```
Hold-out validation results:
  Abell 2261: 
    Observed: 48.5 ± 3.6 arcsec
    Predicted: 46.2 [42.1, 50.8] arcsec
    Fractional error: -4.7%
    
  MACS J1149.5+2223:
    Observed: 46.2 ± 3.6 arcsec
    Predicted: 48.1 [43.5, 53.2] arcsec
    Fractional error: +4.1%

✓ PASS: All hold-outs within 68% PPC
✓ PASS: Median fractional error < 20%
✓ PASS: No systematic bias
```

**Exit codes**:
- `0`: PASS (model validated)
- `1`: FAIL (systematic issues detected)

---

## Interpreting Results

### Scenario A: γ ≈ 0.3-0.5 (Mass-Scaling)

**Physics**: Coherence length scales with halo size
```
ℓ₀ ~ 200 kpc × (R₅₀₀ / 1 Mpc)^0.4
```

**Implications**:
- Larger halos → longer-range coherence
- Consistent with emergent gravity scaling
- May indicate density-dependent field coupling

### Scenario B: γ ≈ 0 (Scale-Invariant)

**Physics**: Same coherence length for all masses
```
ℓ₀ ~ 200 kpc (constant)
```

**Implications**:
- Universal length scale (independent of environment)
- More similar to MOND-like behavior
- Could indicate fundamental length scale in nature

### Scenario C: γ < 0 (Shrinking Coherence)

**Physics**: Coherence shrinks in larger halos (unexpected!)

**If this happens**:
- Check for systematics (triaxiality, κ_ext priors)
- May indicate breakdown of simple scaling model
- Could require more complex environmental dependence

---

## Outputs and Artifacts

```
output/
├── validation/
│   └── catalog_validation.png        # Catalog diagnostics
├── mass_scaled/
│   ├── trace.netcdf                   # Posterior samples (γ free)
│   ├── posterior_summary.txt          # Parameter estimates
│   ├── corner_plot.png                # Posterior distributions
│   ├── cluster_predictions.csv        # Per-cluster θ_E predictions
│   └── convergence_diagnostics.png    # r_hat and ESS plots
├── fixed_scale/
│   ├── trace.netcdf                   # Posterior samples (γ=0)
│   ├── posterior_summary.txt
│   └── ...
└── holdout/
    ├── validation_report.txt
    └── ppc_comparison.png
```

---

## Troubleshooting

### Issue: Divergent transitions

**Symptoms**:
```
There were X divergences after tuning.
```

**Fix**:
```bash
# Increase target_accept (slower but more accurate)
python scripts/run_mass_scaled_hierarchical_inference.py \
  ... --target_accept 0.95
```

### Issue: Low ESS (<400)

**Symptoms**:
```
Warning: ESS below 400 for parameter 'gamma'
```

**Fix**:
```bash
# Increase number of draws
python scripts/run_mass_scaled_hierarchical_inference.py \
  ... --draws 8000 --tune 4000
```

### Issue: r_hat > 1.01

**Symptoms**:
```
Warning: r_hat > 1.01 for parameter 'mu_A'
```

**Fix**:
1. Check trace plots for multi-modality
2. Increase tune iterations
3. Run more chains (--chains 6)

### Issue: Runtime too long

**Fix**:
```bash
# Use fewer clusters for initial test
python scripts/run_mass_scaled_hierarchical_inference.py \
  --tiers 1 \  # Only gold-tier clusters
  --draws 2000 --tune 1000 \
  ...
```

---

## Next Steps After Completion

1. **Write up results**: Summarize γ posterior and model comparison
2. **Robustness checks**: 
   - Run with different priors (ℓ₀,⋆ ~ U(50, 500) instead of LogNormal)
   - Include/exclude tier 3 clusters to test sensitivity
3. **Physical interpretation**: Connect γ to halo formation history
4. **Compare to simulations**: Check if γ matches N-body predictions

---

## Questions?

See full documentation:
- `docs/MASS_SCALING_README.md` (detailed methods)
- `docs/MASS_SCALING_ANALYSIS_PLAN.md` (decision tree)
- `data/CLUSTER_LENSING_CATALOG_README.md` (data sources)

Or grep the codebase:
```bash
rg "class.*Hierarchical" scripts/
rg "def.*lensing" scripts/
```

---

**Ready to start? Run Step 1 now!**
