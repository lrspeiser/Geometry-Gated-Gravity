# Mass-Scaled Coherence Length Analysis Plan

**Scientific question**: Does the coherence length of Sigma-Gravity scale with halo mass?

**Status**: Ready to execute (infrastructure complete, catalog needed)

---

## Executive Summary

We test whether ℓ₀(M) = ℓ₀,⋆ (R₅₀₀/1 Mpc)^γ using hierarchical Bayesian inference on cluster strong lensing Einstein radii. Pre-registered decision rules guard against overfitting and selection bias.

**Decision rules**:
- ΔWAIC ≥ 4 favoring mass-scaled → evidence for scaling
- γ posterior piled near 0 → scale-invariant coherence

**Pass criteria for blind validation** (publishable):
- Median |Δθ_E|/θ_E,obs < 20%
- ≥68% of hold-outs inside 68% posterior predictive intervals
- No systematic residual bias

---

## Model Hierarchy

### Population level
```
μ_A ~ N(16.5, 1.5²)                      # Amplitude mean
σ_A ~ HalfNormal(1.0)                    # Amplitude scatter
ℓ₀,⋆ ~ LogNormal(ln 200 kpc, 0.5²)      # Pivot coherence at 1 Mpc
γ ~ Uniform(0, 1)                        # Mass-scaling exponent
```

### Per-cluster
```
A_c,i ~ N(μ_A, σ_A²)
q_LOS,i ~ N(1, 0.15²) truncated [0.7, 1.4]
q_plane,i ~ N(1, 0.15²) truncated [0.7, 1.4]
κ_ext,i ~ N(0, 0.03²)
```

### Observation model
```
θ_E,obs ~ N(θ_E,model, σ_obs²)
```

---

## Workflow

### 1. Sanity checks (30-45 min, once)

**Range check for ℓ₀(M)**:
```bash
python scripts/check_coherence_length_ranges.py \
  --catalog data/cluster_lensing_catalog.csv \
  --ell0-star 200 --gamma 0.3
```

Expected ranges:
- Groups (R₅₀₀≈0.7 Mpc): 120–220 kpc
- Massive clusters (R₅₀₀≈1.3 Mpc): 180–330 kpc

**Invariance tests**:
- Newtonian limit when ℓ₀,⋆→0 or A_c→0
- Triaxial geometry signal preserved (no averaging shortcuts)

---

### 2. Mass-scaled vs fixed-scale inference

**Mass-scaled model** (γ free):
```bash
python scripts/run_mass_scaled_hierarchical_inference.py \
  --tiers 1,2 --exclude MACS0717 --use-triaxial 1 \
  --draws 4000 --tune 2000 --chains 4 --target_accept 0.9 \
  --out output/mass_scaled/
```

**Fixed-scale comparison** (γ=0):
```bash
python scripts/run_mass_scaled_hierarchical_inference.py \
  --tiers 1,2 --exclude MACS0717 --use-triaxial 1 \
  --fix-gamma 0 --draws 4000 --tune 2000 --chains 4 \
  --out output/fixed_scale/
```

**Record**:
- Posterior medians + 68% CI for (μ_A, σ_A, ℓ₀,⋆, γ)
- WAIC/LOO for model comparison
- Posterior predictive χ²/d.o.f.

---

### 3. Blind validation (hold-out)

```bash
python scripts/run_holdout_validation.py \
  --posterior output/mass_scaled/trace.netcdf \
  --holdout A1689,MACS1149 \
  --use-mass-scaling 1
```

**Pass criteria**:
- Median fractional error < 20%
- Both hold-outs inside 68% PPC
- No systematic sign (both low or both high)

**If hold-outs fail**: Proceed to §D (geometry) and §E (baryon complexity)

---

### 4. Model comparison metrics

```python
import arviz as az

# Load traces
trace_mass = az.from_netcdf("output/mass_scaled/trace.netcdf")
trace_fixed = az.from_netcdf("output/fixed_scale/trace.netcdf")

# Compare
waic_mass = az.waic(trace_mass)
waic_fixed = az.waic(trace_fixed)

delta_waic = waic_mass.elpd_waic - waic_fixed.elpd_waic
delta_waic_se = np.sqrt(waic_mass.se**2 + waic_fixed.se**2)

print(f"ΔWAIC = {delta_waic:.1f} ± {delta_waic_se:.1f}")
print(f"Evidence: {'MASS-SCALED' if delta_waic > 4 else 'INCONCLUSIVE'}")
```

---

## Failure Modes & Guardrails

### Training bias
- **Guard**: Hold out MACS0717 (complex merger) and two others
- **Check**: Blind validation on hold-outs

### Geometry leakage
- **Guard**: Hierarchical q with shrinkage (μ_q ~ N(1, 0.1²))
- **Check**: Compare ΔWAIC with/without geometry

### Clumping
- **Guard**: Gas density divided by √C(r)
- **Check**: Document clumping correction in catalog

### Selection effects
- **Guard**: Tier system (1=Gold, 2=Silver, 3=Complex)
- **Check**: Report selection criteria in `sparc_selection.json` equivalent

---

## Expected Ranges

### Coherence length (with γ=0.3 example)
| Cluster | R₅₀₀ (Mpc) | ℓ₀ (kpc) |
|---------|-----------|----------|
| Group   | 0.7       | 160      |
| Typical | 1.0       | 200      |
| Massive | 1.3       | 230      |

### Amplitude
- Population mean μ_A: 16–18
- Population scatter σ_A: < 1 (small astrophysical scatter)

---

## What Success Looks Like

**Galaxy scale** (already done):
- RAR scatter ≈ 0.087 dex
- Newtonian limit satisfied

**Cluster scale** (this analysis):
- ΔWAIC ≥ 4 favoring mass-scaled **OR** tight γ ≈ 0
- Median |Δθ_E|/θ_E| < 20%
- ≥2/2 hold-outs inside 68% PPC
- Posterior μ_A ≈ 16–18, σ_A ≲ 1

---

## Next Steps After Initial Run

### If hold-outs pass
✅ **Done** — Write paper section with:
- Posterior plots (γ, μ_A, σ_A, ℓ₀,⋆)
- Model comparison table (ΔWAIC, ΔLOO)
- Hold-out validation table

### If hold-outs fail systematically low
→ **Geometry refinement** (§D):
```bash
# Add hierarchical geometry with orientation priors
python scripts/run_mass_scaled_hierarchical_inference.py \
  --tiers 1,2 --exclude MACS0717 \
  --use-hierarchical-geometry 1 \
  --out output/mass_scaled_geom/
```

→ **Baryon complexity** (§E):
- Multi-component Σ for MACS0717
- Subcluster gas halos + BCG/ICL
- LOS structure (weak κ_ext prior)

### If γ posterior piles at boundary
→ **Check boundary artifacts**:
- Extend γ prior to [0, 1.5]
- Verify prior sensitivity with simulated data

---

## Deliverables

### For repository
1. `scripts/run_mass_scaled_hierarchical_inference.py` ✅
2. `scripts/run_holdout_validation.py` ✅
3. `data/cluster_lensing_catalog.csv` (to be built)
4. Pre-registered analysis plan (this document) ✅

### For paper
1. **Methods addendum**: Model hierarchy, priors, mass-scaling formula
2. **Results**: Posterior plots, ΔWAIC table, hold-out validation
3. **Reproducibility box**: Commands, commit SHA, environment

---

## Pre-Registered Analysis Decision Tree

```
START
  ↓
Run mass-scaled inference (γ free)
Run fixed-scale inference (γ=0)
  ↓
Compute ΔWAIC
  ↓
  ├─→ ΔWAIC ≥ 4 favoring mass-scaled?
  │   ├─→ YES → "Evidence for mass scaling"
  │   └─→ NO → Check γ posterior
  │       ├─→ γ ≈ 0 (tight) → "Scale-invariant"
  │       └─→ γ uncertain → "Inconclusive"
  ↓
Blind validation on hold-outs
  ↓
  ├─→ PASS (error < 20%, coverage ≥68%, no bias)?
  │   ├─→ YES → PUBLISH ✅
  │   └─→ NO → Add geometry/baryons (§D, §E)
  ↓
Re-run blind validation
  ↓
  ├─→ PASS?
  │   ├─→ YES → PUBLISH with caveats
  │   └─→ NO → Report failure mode
```

---

## Code Structure

### Core inference
- `run_mass_scaled_hierarchical_inference.py`: PyMC hierarchical model
- `compute_theta_E_triaxial()`: Sigma-Gravity kernel + triaxial projection
- `sigma_gravity_kernel()`: Σ_eff = Σ_bar × (1 + A_c exp(-R/ℓ₀))

### Validation
- `run_holdout_validation.py`: Posterior predictive on hold-outs
- `predict_theta_E_from_posterior()`: Sample from population → per-cluster θ_E

### Utilities
- `load_cluster_catalog()`: CSV loader with tier filtering
- `compute_baryon_surface_density()`: gNFW gas + Hernquist BCG (placeholder)

---

## Dependencies

**Required**:
```bash
pip install pymc arviz pandas numpy astropy
```

**Optional** (for visualization):
```bash
pip install matplotlib seaborn
```

---

## Catalog Requirements

`data/cluster_lensing_catalog.csv` schema:
```
cluster_name,z_lens,z_source,theta_E_obs,sigma_theta_E,M500_1e14Msun,R500_Mpc,tier,has_weak_lensing
```

Tier definitions:
- **1 (Gold)**: High-quality strong lensing, minimal contamination
- **2 (Silver)**: Good lensing, some systematics
- **3 (Complex)**: Mergers (MACS0717, etc.)

Build with:
```bash
python scripts/build_cluster_catalog.py \
  --literature data/literature_clusters.bib \
  --out data/cluster_lensing_catalog.csv
```

---

## References

- Original Sigma-Gravity framework: `README.md`
- Galaxy-scale validation: `results/next_steps/btfr_fix_20250906/`
- Cluster RAR scatter: `results/cluster_rar/`

---

## Contact & Reproducibility

All analysis scripts are versioned in this repository. For questions:
1. Check `docs/` for detailed module documentation
2. Run validation scripts with `--help` flag
3. Inspect `output/*/metrics.json` for numerical summaries

**Commit this plan before running inference** to ensure pre-registration.
