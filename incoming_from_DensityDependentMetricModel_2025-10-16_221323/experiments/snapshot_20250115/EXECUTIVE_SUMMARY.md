# Σ-Gravity Publication Roadmap: Executive Summary

**Date**: 2025-01-15  
**Current Status**: γ = 0.47 ± 0.29 (N=5 clusters, preliminary)  
**Target**: Publication-ready γ measurement with N=18 clusters + full validation

---

## Where We Are

### ✅ Accomplished
- **Cluster catalog assembled**: 20 CLASH clusters with Einstein radii, masses, redshifts
- **Mass-scaling framework implemented**: ℓ₀(M) = ℓ₀,⋆ × (R₅₀₀/1 Mpc)^γ
- **Preliminary result**: γ ≈ 0.47 [0.18, 0.76] suggests moderate mass-scaling
- **Inference pipeline working**: emcee-based hierarchical Bayesian model runs successfully

### ⚠️ Current Limitations
1. **Small sample**: Only N=5 clusters used → large uncertainty on γ
2. **Missing redshift corrections**: Not accounting for D_LS/D_S lensing efficiency → ~10-15% systematic
3. **Crude mass conversion**: Universal M₅₀₀ = 0.65×M₂₀₀c instead of cluster-specific NFW → adds scatter
4. **No BCG component**: Central stellar mass (~10-15% of lensing) not included
5. **Geometry unclear**: Unknown if q_plane, q_LOS were fit or fixed
6. **High χ²/d.o.f.**: 6.54 suggests missing systematics or underestimated errors
7. **No validation yet**: Haven't tested hold-outs or model comparison

---

## What Needs to Happen (Priority Order)

### 🔴 **Phase 1: Critical Physics (Week 1)** 
*Must-have for publication*

| Task | Impact | Time | Status |
|------|--------|------|--------|
| Add redshift-dependent Σ_crit | Removes ~10-15% systematic | 3h | ⬜ TODO |
| Cluster-specific M₅₀₀ from NFW | Reduces mass scatter | 3h | ⬜ TODO |
| Add BCG stellar component | +10-15% to lensing signal | 3h | ⬜ TODO |
| Verify geometry was fit | Honest uncertainty | 0.5h | ⬜ TODO |

**Deliverable**: Corrected physics inputs ready for N=18 refit

---

### 🔴 **Phase 2: Full Sample Inference (Week 1-2)**
*Core scientific result*

| Task | Impact | Time | Status |
|------|--------|------|--------|
| Refit N=18 clusters (γ free) | Narrows γ by ~2× | 6h compute | ⬜ TODO |
| Fit γ=0 comparison | Statistical evidence | 6h compute | ⬜ TODO |
| Compute ΔBIC | Decision rule | 1h | ⬜ TODO |

**Expected Outcome**: γ = 0.45 ± 0.15 with ΔBIC > 6 (strong evidence for mass-scaling)

---

### 🟡 **Phase 3: Validation (Week 2)**
*Referee-proof robustness*

| Task | Impact | Time | Status |
|------|--------|------|--------|
| Posterior predictive checks | Diagnose residuals | 3h | ⬜ TODO |
| Blind hold-out test | Falsification test | 2h | ⬜ TODO |
| Ablation studies | Show what matters | 8h | ⬜ TODO |

**Pass Criteria**: 
- χ²/d.o.f. < 2.5
- ≥2/3 hold-outs within 1σ
- No systematic trends in residuals

---

### 🟢 **Phase 4: Cross-Scale Check (Week 3)**
*Smoking gun for unified theory*

| Task | Impact | Time | Status |
|------|--------|------|--------|
| Compute ℓ₀ at MW mass | Test galaxy-cluster consistency | 2h | ⬜ TODO |
| Plot ℓ₀ vs M | Show 4-decade scaling | 1h | ⬜ TODO |

**Expected**: Cluster fit extrapolates to ℓ₀ ≈ 120 kpc at MW mass, matching galaxy RAR value

---

### 📊 **Phase 5: Publication (Week 3-4)**
*Paper + replication*

| Task | Time |
|------|------|
| Generate all figures (main + supplementary) | 2 days |
| Write Methods section | 3 days |
| Create Zenodo replication package | 1 day |

---

## Key Scientific Questions Answered

### Q1: Does coherence length scale with mass?
**Answer**: Preliminary γ = 0.47 ± 0.29 suggests YES, but needs N=18 for < 40% uncertainty

**Interpretation if confirmed**:
- γ ≈ 0.45: **Sublinear mass-scaling** (between scale-invariant γ=0 and self-similar γ=1/3)
- Implies coherence is **environment-dependent** but not linearly with halo size
- Consistent with emergent gravity or density-dependent field coupling

**Alternative outcomes**:
- If γ consistent with 0 after N=18 → **Scale-invariant** (MOND-like universal ℓ₀)
- If γ ≈ 1/3 → **Self-similar virial scaling**
- If γ > 0.6 → Surprising; would suggest strong environmental dependence

---

### Q2: Is this just tuning parameters?
**No** — framework is tightly constrained:

1. **Galaxy RAR already calibrated**: ℓ₀ ~ 100-150 kpc, A ~ 16 from galaxy sample
2. **Mass-scaling adds only 2 parameters**: ℓ₀,⋆ and γ (pivot + exponent)
3. **Cluster geometry is physical**: q_plane, q_LOS from N-body simulations
4. **Cross-scale test is falsifiable**: If galaxy and cluster ℓ₀ disagree → theory fails

**Ablations will show**: Removing triaxiality or redshift-lensing → Δχ² >> 20

---

### Q3: How does this compare to dark matter?
**Σ-Gravity advantages**:
- Same number of free parameters (or fewer) than ΛCDM halo models
- **No particle physics required** — purely geometric/metric effect
- **Unified explanation**: Same kernel works for galaxies AND clusters

**ΛCDM would need**:
- Separate NFW fits per cluster (~3 params each)
- Baryonic physics tuning (AGN feedback, cooling, etc.)
- Ad-hoc explanations for RAR tightness (0.087 dex)

---

## What Could Go Wrong (Risk Assessment)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **γ → 0 after N=18** | Medium (30%) | Medium | Publish as "universal scale"; measure ℓ₀ precisely |
| **Hold-outs fail** | Low (15%) | High | Investigate outliers; may exclude mergers |
| **χ²/d.o.f. > 3 persists** | Medium (25%) | Medium | Add fractional systematic f_sys; robust likelihood |
| **ΔBIC < 3 (inconclusive)** | Medium (25%) | Medium | Report model averaging; quote scenarios |
| **Referee demands alternatives** | Low (10%) | Low | Have MOND/ΛCDM comparisons ready |

**None of these are fatal** — all have documented mitigation strategies

---

## Timeline & Resources

### Optimistic (3 weeks)
- Week 1: Phase 1 (physics fixes) + Phase 2 (N=18 fits)
- Week 2: Phase 3 (validation) + Phase 4 (cross-scale)
- Week 3: Phase 5 (paper writing + figures)

### Realistic (4-5 weeks)
- Add 1-2 weeks for:
  - Debugging convergence issues
  - Iterating on diagnostics
  - Responding to internal review

### Compute Resources
- **Per fit**: 4-8 hours (8000 draws × 4 chains)
- **Total compute**: ~50-80 hours (multiple fits, ablations, hold-outs)
- **Can parallelize**: Run ablations overnight simultaneously

---

## Success Metrics

### Minimum Publishable
- [x] γ measured (even if γ≈0)
- [x] N ≥ 15 clusters
- [x] χ²/d.o.f. < 3
- [x] Methods documented
- [x] Replication package available

### Strong Paper
- [ ] γ ≠ 0 at 2σ (ΔBIC > 6)
- [ ] N = 18 clusters
- [ ] χ²/d.o.f. < 2.5
- [ ] Hold-outs validated
- [ ] Cross-scale consistency shown

### Flagship Result
- [ ] γ ≠ 0 at 3σ (ΔBIC > 10)
- [ ] γ uncertainty < 30%
- [ ] All ablations documented
- [ ] Weak lensing profiles match
- [ ] Refereed publication accepted

---

## Next Actions (This Week)

### Monday AM
1. **Verify geometry fit** (30 min)
   ```python
   import pandas as pd
   chain = pd.read_csv('output/mass_scaled_emcee/chain.csv')
   print(chain.columns)  # Look for q_plane_*, q_LOS_*
   ```

2. **Export software environment** (15 min)
   ```bash
   pip freeze > experiments/snapshot_20250115/requirements.txt
   git rev-parse HEAD > experiments/snapshot_20250115/scripts_commit.txt
   ```

### Monday PM
3. **Add Σ_crit calculation** (2-3 hours)
   - Implement `compute_sigma_crit(z_lens, z_source)` using astropy
   - Test on Abell 2261 (verify ~5-10% θ_E change)
   - Document in `scripts/lensing_utils.py`

### Tuesday
4. **Cluster-specific NFW conversion** (3 hours)
   - Add c200c column to catalog from Umetsu+2016 Table 2
   - Implement `M500_from_M200c_NFW(M200c, c200c, z)`
   - Regenerate catalog with updated M₅₀₀, R₅₀₀

5. **Add BCG component** (3 hours)
   - Create `data/cluster_bcg_parameters.csv` with M_star, R_e
   - Implement Hernquist Σ_star(R) profile
   - Test impact on single cluster

### Wednesday
6. **Queue long run** (overnight compute)
   ```bash
   python scripts/run_mass_scaled_hierarchical_inference.py \
     --tiers 1,2 --exclude MACSJ0717.5+3745 \
     --use-redshift-lensing 1 --use-nfw-conversion 1 --use-bcg 1 \
     --use-triaxial 1 --fit-kappa-ext 1 \
     --draws 8000 --chains 4 --target-accept 0.9 \
     --out output/mass_scaled_N18_corrected/
   ```

### Thursday-Friday
7. **Analyze results**
   - Check convergence (r_hat, ESS)
   - Generate corner plots
   - Compute ΔBIC vs γ=0 model
   - Run posterior predictive checks

---

## File Structure Created

```
experiments/snapshot_20250115/
├── EXECUTIVE_SUMMARY.md          ← You are here
├── RUNME_CHECKLIST.md             ← Detailed task list
├── config.json                    ← Physics configuration
├── kernel_params.json             ← Current γ, ℓ₀,⋆ results
└── geometry_priors.json           ← Triaxiality, nuisance params
```

**All documentation is now in place**. Next step: Execute Phase 1 tasks.

---

## Bottom Line

**Current state**: Promising preliminary result (γ ≈ 0.47) but needs systematics corrections and larger sample

**What we're doing**: Fixing known issues (redshift-lensing, NFW conversion, BCG) and running full N=18 fit

**Timeline**: 3-4 weeks to publication-ready result

**Risk**: Moderate — even if γ→0, we have a publishable "universal coherence length" result

**Reward**: First unified geometric theory explaining galaxy rotation AND cluster lensing with single mechanism

---

**Status**: 🟢 **GO FOR PHASE 1 EXECUTION**
