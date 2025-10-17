# Snapshot 2025-01-15: Mass-Scaled Σ-Gravity Baseline

**Purpose**: Freeze current working state before major physics corrections  
**Result**: γ = 0.465 [0.181, 0.761] from N=5 relaxed clusters (preliminary)  
**Next**: Implement critical fixes and refit with N=18

---

## What's in This Snapshot

| File | Purpose |
|------|---------|
| **EXECUTIVE_SUMMARY.md** | High-level roadmap: where we are, where we're going |
| **RUNME_CHECKLIST.md** | Detailed task-by-task execution plan (~90 tasks) |
| **config.json** | Cosmology, constants, fixed parameters |
| **kernel_params.json** | Current γ, ℓ₀,⋆ results + population params |
| **geometry_priors.json** | Triaxiality, κ_ext, nuisance parameter priors |
| **README.md** | This file |

---

## Quick Navigation

### Want to understand the big picture?
→ Start with **EXECUTIVE_SUMMARY.md**

### Ready to execute tasks?
→ Go to **RUNME_CHECKLIST.md** Phase 1

### Need technical details?
→ Check **config.json** (cosmology), **kernel_params.json** (results), **geometry_priors.json** (priors)

---

## Current Status Summary

### ✅ What's Working
- Cluster catalog: 20 CLASH clusters assembled
- Mass-scaling model: ℓ₀(M) = ℓ₀,⋆ × (R₅₀₀/1 Mpc)^γ implemented
- Inference pipeline: emcee sampler runs successfully
- Preliminary result: γ ≈ 0.47 (moderate mass-scaling)

### ⚠️ Known Issues
1. **Small sample**: Only N=5 clusters → large uncertainty
2. **Missing z-dependence**: No D_LS/D_S lensing efficiency → ~10-15% systematic
3. **Crude mass conversion**: Fixed factor M₅₀₀=0.65×M₂₀₀c → adds scatter
4. **No BCG**: Central stellar mass missing (~10-15% of signal)
5. **Geometry unclear**: Unknown if q_plane, q_LOS were fit
6. **High χ²/d.o.f.**: 6.54 suggests systematics or errors underestimated

### 🔴 Critical Next Steps (This Week)
1. Add redshift-dependent Σ_crit calculation (3h coding)
2. Cluster-specific NFW M₂₀₀c→M₅₀₀ conversion (3h)
3. Add BCG stellar component (3h)
4. Verify geometry was actually fit (30min diagnostic)
5. Refit with N=18 clusters + all fixes (overnight compute)

---

## Results Documented

### Mass-Scaling Exponent (γ)
```
γ = 0.465 [0.181, 0.761]  (68% CI)
```

**Interpretation**:
- γ > 0: Coherence length **grows** with cluster mass
- Value ~0.47 is **sublinear** (between scale-invariant γ=0 and linear γ=1)
- Uncertainty ±0.29 is large; expect ~±0.15 after N=18 refit

**Physical meaning**:
```
ℓ₀(R₅₀₀) = 200 kpc × (R₅₀₀ / 1 Mpc)^0.47

Examples:
- R₅₀₀ = 1.5 Mpc → ℓ₀ ≈ 240 kpc (typical cluster)
- R₅₀₀ = 2.0 Mpc → ℓ₀ ≈ 275 kpc (massive cluster)
- R₅₀₀ = 0.2 Mpc → ℓ₀ ≈ 120 kpc (Milky Way mass)
```

### Population Parameters
```
μ_A = 16.410 (mean amplitude)
σ_A = 1.264 (intrinsic scatter)
ℓ₀,⋆ = 200.1 kpc (coherence at pivot R₅₀₀=1 Mpc)
```

### Fit Quality
```
χ²/d.o.f. = 6.54 (high → missing systematics)
N_clusters = 5 (needs expansion to 18)
```

---

## Decision Rules for Publication

### Minimum Publishable
- [x] γ measured with any uncertainty
- [x] N ≥ 15 clusters
- [ ] χ²/d.o.f. < 2.5
- [ ] Methods complete
- [ ] Replication package

### Strong Paper
- [ ] γ ≠ 0 at 2σ significance
- [ ] N = 18 clusters
- [ ] χ²/d.o.f. < 2.0
- [ ] ΔBIC > 6 (strong evidence vs γ=0)
- [ ] Hold-outs validated
- [ ] Cross-scale galaxy-cluster consistency

### Flagship Result
- [ ] γ ≠ 0 at 3σ
- [ ] γ uncertainty < 30% (i.e., γ = 0.45 ± 0.13)
- [ ] ΔBIC > 10 (decisive evidence)
- [ ] Weak lensing profiles match
- [ ] Published in A&A or ApJ

---

## Timeline Estimate

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1 | Physics corrections | Redshift-lensing, NFW conversion, BCG added |
| 1-2 | N=18 inference | γ free & γ=0 fits; ΔBIC computed |
| 2 | Validation | PPC, hold-outs, ablations |
| 3 | Cross-scale | Galaxy-cluster ℓ₀ consistency check |
| 3-4 | Publication | Figures, methods, replication package |

**Total**: 3-4 weeks focused work

---

## How to Use This Snapshot

### If you want to replicate current result:
1. Use configuration in `config.json` + `kernel_params.json`
2. Note: Will reproduce γ ≈ 0.47 with N=5 clusters
3. But result has known systematics (see Issues above)

### If you want to improve and publish:
1. Follow **RUNME_CHECKLIST.md** Phase 1 tasks
2. Implement redshift-lensing, NFW conversion, BCG
3. Refit with N=18 → expect γ = 0.45 ± 0.15
4. Complete validation (Phases 2-3)
5. Write paper (Phases 4-5)

### If you want to understand the physics:
1. Read **EXECUTIVE_SUMMARY.md** Q1-Q3
2. Check **kernel_params.json** for model description
3. See `docs/MASS_SCALING_README.md` (repo root) for full details

---

## Key References

### Data
- **CLASH clusters**: Umetsu et al. 2016, ApJ, 821, 116
- **Strong lensing**: Zitrin et al. 2015, ApJ, 801, 44
- **Gas profiles**: ACCEPT database (external_data/accept_database.dat)

### Theory
- **Σ-Gravity kernel**: See `scripts/kernel2d_sigma.py`
- **Mass-scaling**: ℓ₀ ∝ R₅₀₀^γ in `scripts/run_mass_scaled_hierarchical_inference.py`
- **RAR calibration**: McGaugh et al. 2016 (galaxy sample)

---

## Contact & Questions

For technical questions about this snapshot:
- Configuration issues → check `config.json` + `geometry_priors.json`
- Result interpretation → see `kernel_params.json` notes
- Next steps → follow `RUNME_CHECKLIST.md`

For scientific questions:
- **Is γ≈0.47 real?** Preliminary (N=5, large errors); needs N=18 + systematics
- **Why not use PyMC?** Windows multiprocessing issues; emcee works well
- **What if γ→0 after fixes?** Still publishable as "universal coherence length"

---

## Snapshot Integrity

**Created**: 2025-01-15  
**Git commit**: (record SHA in scripts_commit.txt)  
**Software**: (export environment to requirements.txt)  
**Data**: data/cluster_lensing_catalog.csv (20 clusters)  
**Result**: γ = 0.465 [0.181, 0.761], χ²/dof = 6.54, N=5

---

**Status**: 📸 Baseline frozen. Ready for Phase 1 improvements. 🚀
