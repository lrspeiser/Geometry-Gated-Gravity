# Mass-Scaling Analysis Implementation

**Status**: ✅ Infrastructure complete, ready for execution  
**Next step**: Build cluster catalog (`data/cluster_lensing_catalog.csv`)

---

## What Was Built

You now have a complete, publication-ready statistical framework to answer:

> **Does the coherence length of Sigma-Gravity scale with halo mass?**

### Core Scripts (3 files created)

1. **`scripts/run_mass_scaled_hierarchical_inference.py`** (515 lines)
   - Hierarchical Bayesian model: ℓ₀(M) = ℓ₀,⋆ (R₅₀₀/1 Mpc)^γ
   - PyMC implementation with NUTS sampler
   - Sigma-Gravity kernel + triaxial geometry projection
   - Pre-registered priors and decision rules
   
2. **`scripts/run_holdout_validation.py`** (285 lines)
   - Blind validation on hold-out clusters
   - Posterior predictive checks
   - Pass/fail criteria enforcement
   - Exit code 0 (pass) or 1 (fail)
   
3. **`docs/MASS_SCALING_ANALYSIS_PLAN.md`** (327 lines)
   - Complete scientific workflow
   - Pre-registered decision tree
   - Failure mode guardrails
   - Expected ranges and success criteria

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install pymc arviz pandas numpy astropy matplotlib

# Verify installation
python -c "import pymc, arviz; print('PyMC ready')"
```

### Step 1: Build Catalog

**Required**: Create `data/cluster_lensing_catalog.csv` with schema:

```csv
cluster_name,z_lens,z_source,theta_E_obs,sigma_theta_E,M500_1e14Msun,R500_Mpc,tier,has_weak_lensing
ABELL1689,0.184,1.0,40.5,2.1,9.0,1.8,1,1
MACS0717,0.548,2.5,55.0,3.0,15.0,1.95,3,1
...
```

**Sources**:
- Einstein radii: CLASH, HFF strong lensing catalogs
- Masses/radii: X-ray (Chandra, XMM) or weak lensing
- Redshifts: Spectroscopic confirmation

**Tier assignments**:
- **1 (Gold)**: Clean strong lensing, no mergers, θ_E uncertainty < 10%
- **2 (Silver)**: Good lensing, mild systematics, θ_E uncertainty < 20%
- **3 (Complex)**: Mergers (MACS0717, MACS0416), substructure, θ_E uncertain

### Step 2: Run Mass-Scaled Inference

```bash
cd C:\Users\henry\Documents\GitHub\DensityDependentMetricModel

# Mass-scaled model (γ free)
python scripts/run_mass_scaled_hierarchical_inference.py \
  --tiers 1,2 --exclude MACS0717 --use-triaxial 1 \
  --draws 4000 --tune 2000 --chains 4 --target_accept 0.9 \
  --out output/mass_scaled/

# Fixed-scale comparison (γ=0)
python scripts/run_mass_scaled_hierarchical_inference.py \
  --tiers 1,2 --exclude MACS0717 --use-triaxial 1 \
  --fix-gamma 0 --draws 4000 --tune 2000 --chains 4 \
  --out output/fixed_scale/
```

**Expected runtime**: 2-4 hours per model (depends on N_clusters and hardware)

### Step 3: Model Comparison

```python
import arviz as az
import numpy as np

# Load traces
trace_mass = az.from_netcdf("output/mass_scaled/trace.netcdf")
trace_fixed = az.from_netcdf("output/fixed_scale/trace.netcdf")

# WAIC comparison
waic_mass = az.waic(trace_mass)
waic_fixed = az.waic(trace_fixed)

delta_waic = waic_mass.elpd_waic - waic_fixed.elpd_waic
delta_waic_se = np.sqrt(waic_mass.se**2 + waic_fixed.se**2)

print(f"ΔWAIC = {delta_waic:.1f} ± {delta_waic_se:.1f}")

if delta_waic > 4:
    print("✓ Evidence for mass scaling")
elif abs(delta_waic) < 4:
    # Check γ posterior
    gamma_post = trace_mass.posterior['gamma'].values.flatten()
    if np.std(gamma_post) < 0.1 and np.mean(gamma_post) < 0.1:
        print("✓ Scale-invariant coherence")
    else:
        print("⚠ Inconclusive")
```

### Step 4: Blind Validation

```bash
python scripts/run_holdout_validation.py \
  --posterior output/mass_scaled/trace.netcdf \
  --holdout A1689,MACS1149 \
  --use-mass-scaling 1

# Exit code: 0 = PASS, 1 = FAIL
echo $?  # (PowerShell: $LASTEXITCODE)
```

**Pass criteria**:
- Median |Δθ_E|/θ_E,obs < 20%
- ≥68% hold-outs inside 68% PPC
- No systematic bias (all positive or all negative)

---

## Model Details

### Population-Level Priors

| Parameter | Prior | Justification |
|-----------|-------|---------------|
| μ_A | N(16.5, 1.5²) | Centered on galaxy-scale amplitude |
| σ_A | HalfNormal(1.0) | Small astrophysical scatter |
| ℓ₀,⋆ | LogNormal(ln 200 kpc, 0.5²) | Pivot at 1 Mpc, ~150-300 kpc range |
| γ | Uniform(0, 1) | Fixed (γ=0) to self-similar (γ=1) |

### Per-Cluster Parameters

| Parameter | Prior | Range | Notes |
|-----------|-------|-------|-------|
| A_c,i | N(μ_A, σ_A²) | — | Hierarchical shrinkage |
| q_LOS,i | N(1, 0.15²) | [0.7, 1.4] | ~20-30% geometry variation |
| q_plane,i | N(1, 0.15²) | [0.7, 1.4] | Independent from LOS |
| κ_ext,i | N(0, 0.03²) | — | LSS convergence sheet |

### Physics Implementation

**Sigma-Gravity kernel**:
```
Σ_eff(R) = Σ_bar(R) × [1 + A_c exp(-R/ℓ₀)]
```

**Mass scaling**:
```
ℓ₀(M) = ℓ₀,⋆ (R₅₀₀ / 1 Mpc)^γ
```

**Triaxial projection**:
```
Σ_projected = q_LOS × q_plane × Σ_spherical
```

**Einstein radius**:
```
κ(θ_E) = 1, where κ = Σ_projected/Σ_crit + κ_ext
```

---

## Expected Results

### If mass-scaling is real (γ ≠ 0)

**Posterior** (example):
```
Parameter    Median   [16%, 84%]    r_hat
─────────────────────────────────────────
μ_A          16.8     [16.2, 17.4]  1.00
σ_A          0.7      [0.5, 1.1]    1.00
ℓ₀,⋆ [kpc]   195      [170, 230]    1.00
γ            0.32     [0.18, 0.48]  1.00
```

**Model comparison**:
```
ΔWAIC = +6.2 ± 2.1  (favors mass-scaled)
```

**Interpretation**: Coherence length grows as ℓ₀ ∝ R₅₀₀^0.32, suggesting the characteristic scale adapts to halo size.

### If scale-invariant (γ ≈ 0)

**Posterior**:
```
γ = 0.05 ± 0.08  (68% CI: [-0.03, 0.13])
```

**Model comparison**:
```
ΔWAIC = +1.4 ± 3.0  (inconclusive)
```

**Interpretation**: Coherence length is approximately constant (~200 kpc) across halo masses, suggesting a fundamental physical scale.

---

## Failure Modes & Fixes

### Hold-outs systematically low

**Diagnosis**: Underestimating Einstein radii

**Possible causes**:
1. Baryon profiles too smooth (missing substructure)
2. Geometry too restricted (need more flexible q)
3. Missing external convergence (LSS along LOS)

**Fixes** (in order):
```bash
# 1. Add hierarchical geometry with broader priors
python scripts/run_mass_scaled_hierarchical_inference.py \
  --use-hierarchical-geometry 1 --sigma-q 0.2 ...

# 2. Multi-component baryons for complex systems
# (requires updating compute_baryon_surface_density)

# 3. Increase κ_ext prior width
# (edit prior in build_hierarchical_model)
```

### γ posterior piles at boundary

**Diagnosis**: Prior too restrictive or data uninformative

**Fixes**:
```bash
# Extend γ prior
# Edit: gamma = pm.Uniform('gamma', lower=0.0, upper=1.5)

# Or try log-space
# gamma = pm.Lognormal('gamma', mu=np.log(0.3), sigma=0.5)
```

### High divergences (>1% of samples)

**Diagnosis**: Geometry of posterior challenging for NUTS

**Fixes**:
```bash
# Increase target_accept
--target_accept 0.95

# More tuning steps
--tune 3000

# Reparameterize (non-centered for hierarchical)
# (requires code modification)
```

---

## Outputs Explained

### Inference outputs (`output/mass_scaled/`)

```
trace.netcdf              # Full posterior samples (ArviZ format)
summary.csv               # Posterior summaries (mean, HDI, r_hat)
metrics.json              # WAIC, LOO, diagnostics
```

### Validation outputs (`output/holdout_validation/`)

```
holdout_validation.json   # Pass/fail summary
holdout_predictions.csv   # Per-cluster predictions
```

### Metrics JSON schema

```json
{
  "n_clusters": 15,
  "gamma_fixed": false,
  "waic": -123.4,
  "waic_se": 8.2,
  "loo": -124.1,
  "loo_se": 8.5,
  "n_divergences": 3,
  "mean_tree_depth": 4.8
}
```

---

## Next Steps

### Immediate (before first run)

1. ✅ Read `MASS_SCALING_ANALYSIS_PLAN.md`
2. ⏳ Build `data/cluster_lensing_catalog.csv`
3. ⏳ Test on 2-3 clusters (quick sanity check)
4. ⏳ Run full inference (tiers 1+2, ~15 clusters)

### After initial results

**If ΔWAIC ≥ 4 favoring mass-scaled**:
- ✍️ Write paper section on mass scaling
- 📊 Plot γ posterior with interpretation
- 📄 Report ℓ₀ at different halo masses

**If γ ≈ 0 (scale-invariant)**:
- ✍️ Emphasize fundamental scale (~200 kpc)
- 🔍 Compare to other physical scales (sound horizon, etc.)
- 📄 Discuss implications for gravity modification

**If hold-outs fail**:
- 🔧 Implement §D (geometry refinement)
- 🔧 Implement §E (baryon complexity)
- 🔁 Re-run blind validation

---

## Citation & Reproducibility

When using this framework:

```
@software{ddmm_mass_scaling,
  author = {Speiser, Leonard},
  title = {Mass-Scaled Coherence Length Analysis for Sigma-Gravity},
  year = {2025},
  url = {https://github.com/lspeiser/DensityDependentMetricModel},
  note = {Commit: [INSERT SHA AFTER COMMITTING]}
}
```

**Reproducibility checklist**:
- [ ] Catalog CSV committed (or documented sources)
- [ ] Inference script unchanged (or changes documented)
- [ ] Random seed set (`pm.sample(..., random_seed=42)`)
- [ ] Environment logged (`pip freeze > requirements_inference.txt`)
- [ ] Analysis plan committed **before** running (pre-registration)

---

## FAQ

**Q: Can I use fewer chains/draws for testing?**  
A: Yes, but use `--draws 500 --tune 500 --chains 2` minimum. Check r_hat < 1.05.

**Q: What if I don't have all catalog columns?**  
A: Minimum required: `cluster_name, z_lens, theta_E_obs, sigma_theta_E, R500_Mpc`. Set `tier=2` for all.

**Q: How do I visualize posteriors?**  
A: Use ArviZ:
```python
import arviz as az
trace = az.from_netcdf("output/mass_scaled/trace.netcdf")
az.plot_trace(trace, var_names=['gamma', 'mu_A', 'ell_0_star_kpc'])
az.plot_posterior(trace, var_names=['gamma'])
```

**Q: Can I add weak lensing γ_t(R)?**  
A: Not yet — that's §F in the plan. Current version uses θ_E only.

**Q: What about MACS0416?**  
A: Include in tier 2 initially. If it's an outlier, move to tier 3 and exclude (like MACS0717).

---

## Support

For questions or issues:

1. Check this README and `MASS_SCALING_ANALYSIS_PLAN.md`
2. Inspect `output/*/metrics.json` for numerical diagnostics
3. Run scripts with `--help` flag for usage
4. Verify PyMC/ArviZ versions (`pip show pymc arviz`)

**Pre-flight checklist**:
```bash
# Verify all dependencies
python scripts/run_mass_scaled_hierarchical_inference.py --help
python scripts/run_holdout_validation.py --help

# Test catalog loading
python -c "from scripts.run_mass_scaled_hierarchical_inference import load_cluster_catalog; print(load_cluster_catalog([1,2]))"

# Check computational resources
# (expect ~4GB RAM, ~2-4 hours CPU time for 15 clusters)
```

---

**Ready to execute.** Build the catalog and run the commands in §Quick Start.
