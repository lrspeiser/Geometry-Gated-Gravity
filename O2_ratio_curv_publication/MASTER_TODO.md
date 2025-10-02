# O2 ratio_curv Publication Project - Master TODO List

**Project:** Geometry-Gated Gravity Publication and Research Extensions  
**Model:** O2 `ratio_curv` (3 parameters: a=0.669, b=0.140, d=0.087)  
**Status:** In Progress  
**Created:** October 2, 2025  

---

## 🎯 Project Overview

This master TODO tracks all research paths for the O2 `ratio_curv` geometry-gated gravity model, from core publication to advanced theoretical and observational extensions.

**Core Achievement:** 90% galaxy rotation curve accuracy (median APE 0.242) with 3 global parameters, competitive with MOND, vastly superior to GR baryons.

**Known Limitation:** Cluster strong lensing underprediction (40-140× too small) - systematic scale problem requiring future work.

---

## 📊 Priority Levels

- 🔴 **P0 - Critical:** Required for core publication
- 🟡 **P1 - High:** Strengthens paper significantly
- 🟢 **P2 - Medium:** Publishable follow-up work
- 🔵 **P3 - Low:** Long-term theoretical/exploratory

---

## 1️⃣ Core Publication (01_core_publication/) 🔴 P0

**Goal:** Complete and submit primary paper to ApJ/MNRAS

### Phase 1: Finalize Paper Content
- [x] ✅ Write complete manuscript (PAPER_O2_RATIO_CURV.md - 1227 lines)
- [x] ✅ Document model formulation and parameters
- [x] ✅ Include all results sections (SPARC, MW, cluster limitations)
- [x] ✅ Write appendices (feature computation, sensitivity, comparison)
- [x] ✅ Generate Figure 1: Rotation curve overlays (6 galaxies) - `Figure1_RC_Overlays.png`
- [x] ✅ Generate Figure 2: Milky Way rotation curve (Gaia bins ±1σ) - `Figure2_MW_RotationCurve.png`
- [x] ✅ Generate Figure 3: Residual diagnostics (4-panel) - `Figure3_Residual_Diagnostics.png/.pdf`
- [x] ✅ Generate Figure 4: Cluster lensing failure (3-panel) - `Figure4_Cluster_Lensing.png/.pdf`
- [x] ✅ Create supplementary Table S1: Per-galaxy metrics (120 galaxies CSV) - `TableS1_Per_Galaxy_Metrics.csv`
- [ ] 🔄 Create supplementary Table S2: Cross-validation detailed results (needs 5-fold CV script)
- [ ] 🔄 Create supplementary Table S3: Parameter sensitivity analysis (needs sensitivity script)

### Phase 2: Pre-submission Review
- [ ] Internal review pass (check for typos, clarity, consistency)
- [ ] Verify all figure references match actual figures
- [ ] Verify all code snippets are executable
- [ ] Test one-command reproduction on clean environment
- [ ] Generate arXiv-compatible PDF (Markdown → LaTeX → PDF)

### Phase 3: Submission
- [ ] Submit to arXiv (preprint)
- [ ] Get arXiv ID and update paper citations
- [ ] Send to 2-3 colleagues for pre-submission feedback
- [ ] Incorporate feedback
- [ ] Format for journal (ApJ or MNRAS LaTeX template)
- [ ] Submit to journal
- [ ] Track submission status

**Estimated Timeline:** 2-3 weeks  
**Deliverables:** 
- Main paper PDF
- 4 key figures (PNG/PDF)
- 3 supplementary tables (CSV)
- arXiv preprint link
- Journal submission confirmation

---

## 2️⃣ Uncertainty Quantification (02_uncertainty_quantification/) 🟡 P1

**Goal:** Quantify parameter uncertainties and prediction bands

### Tasks:
- [ ] Bootstrap parameter confidence intervals (1000 resamples)
  - Resample galaxies with replacement
  - Refit (a, b, d) on each bootstrap sample
  - Compute 68% and 95% CI for each parameter
  - **Output:** `bootstrap_params_ci.json`, `bootstrap_distribution.png`

- [ ] Per-galaxy prediction bands
  - Propagate parameter uncertainties through model
  - Compute ±1σ and ±2σ prediction bands for each galaxy
  - Account for Σ(R) measurement errors (if available)
  - **Output:** `per_galaxy_prediction_bands.csv`, `prediction_band_examples.png`

- [ ] Jackknife cross-validation
  - Leave-one-out or leave-10%-out resampling
  - Compare with bootstrap results for consistency
  - **Output:** `jackknife_params.json`

- [ ] Error propagation analysis
  - Trace uncertainty from Σ(R) → Σ̂ → fX → V_mod
  - Identify dominant error sources
  - **Output:** `error_budget.md`, `error_propagation.png`

**Estimated Timeline:** 1 week  
**Deliverables:** 
- Bootstrap CI JSON
- Prediction band CSV for all galaxies
- Error budget document
- 2-3 diagnostic figures

**Publication Impact:** Adds rigor, likely required by referees

---

## 3️⃣ Type-Specific Analysis (03_type_specific_analysis/) 🟡 P1

**Goal:** Break down performance by galaxy type and physical properties

### Tasks:
- [ ] Decompose SPARC into subsamples
  - Spiral galaxies (Sab, Sbc, Scd, Sd)
  - Dwarf irregulars (Im, BCD)
  - Low surface brightness (LSB)
  - **Output:** `galaxy_type_classification.csv`

- [ ] Per-type fit quality
  - Compute median APE, RMSE for each subsample
  - Test if (a, b, d) need type-dependent adjustments
  - **Output:** `per_type_metrics.csv`, `per_type_comparison.png`

- [ ] Systematic residuals analysis
  - Residuals vs. inclination angle
  - Residuals vs. metallicity (if available in SPARC)
  - Residuals vs. bar strength (if available)
  - Residuals vs. Rd (scale length)
  - **Output:** `systematic_residuals.csv`, `residuals_vs_properties.png` (6-panel)

- [ ] Edge cases study
  - Very low surface brightness (Σ̂ < -2)
  - Very high surface brightness (Σ̂ > 2)
  - Smallest galaxies (Rd < 0.5 kpc)
  - Largest galaxies (Rd > 8 kpc)
  - **Output:** `edge_cases_analysis.md`

**Estimated Timeline:** 1 week  
**Deliverables:**
- Type classification CSV
- Per-type metrics table
- Systematic residuals plots
- Edge cases report

**Publication Impact:** Strengthens paper, may go in appendix or supplementary

---

## 4️⃣ Extended Symbolic Regression (04_extended_symbolic_regression/) 🟢 P2

**Goal:** Search for alternative functional forms and cluster-specific features

### Tasks:
- [ ] Alternative functional forms (same features)
  - Test: (x² + c·x³) / (a - b·Σ̂ - d·|∇ln Σ|)
  - Test: x^α / (a - b·Σ̂ - d·|∇ln Σ|) with fitted α
  - Test: Mixed exponential-ratio forms
  - **Output:** `alternative_forms_comparison.csv`

- [ ] Cluster-specific feature search
  - Add ∇²Σ (Laplacian of surface density)
  - Add smoothed Σ at multiple scales (10, 50, 100, 200 kpc)
  - Add temperature-dependent terms (for clusters)
  - Run PySR on cluster lensing dataset
  - **Output:** `cluster_features_pareto.csv`, `cluster_sr_expressions.txt`

- [ ] Time-dependent terms exploration
  - Merger timescale features (for interacting galaxies)
  - Dynamical time corrections
  - **Output:** `time_dependent_features.md`

- [ ] Pareto frontier analysis
  - Plot complexity vs. accuracy for all candidates
  - Identify simplicity-performance sweet spots
  - **Output:** `pareto_frontier_extended.png`

**Estimated Timeline:** 2 weeks (computationally intensive)  
**Deliverables:**
- Alternative forms table
- Cluster SR results
- Pareto frontier plots
- Feature discovery report

**Publication Impact:** Separate follow-up paper or major revision

---

## 5️⃣ Cluster-Adapted Gating (05_cluster_adapted_gating/) 🟢 P2

**Goal:** Systematically test 6 parallel approaches to extend geometry gating to cluster scales

**Core Problem:** O2 ratio_curv underpredicts cluster Einstein radii by 40-140×. Need order-of-magnitude amplification boost while preserving galaxy fits (median APE < 0.30).

**Success Criterion (Universal):**
```python
cluster_pass = all(|θ_E_pred - θ_E_obs| / θ_E_obs < 0.3)  # Within 30%
galaxy_pass = median_APE < 0.30  # Max 6-point degradation
if cluster_pass and galaxy_pass: PUBLISH
elif cluster_pass only: TWO_REGIME_MODEL
else: NEXT_TEST
```

---

### **Phase 1: Quick Wins (1 week each)** ⭐ TOP PRIORITY

#### Test 1: Velocity Dispersion Gating
- [ ] **Hypothesis:** Deep potential wells (high σ) amplify tail. Clusters σ~1000 km/s vs galaxies σ~100 km/s → 10× natural factor
- [ ] **Formula:** `fX = x² / (a - b·Σ̂ - d·|∇ln Σ| - e·(σ/σ₀)^α)`
  - Add parameters: e (gate weight), α (power index)
  - σ₀ = 100 km/s reference
  - Prediction: α ≈ 1.5 → (1000/100)^1.5 = 31× cluster boost
- [ ] **Implementation:**
  - Code: `compute_velocity_dispersion_from_temperature(kT_keV)`
  - Code: `fX_ratio_curv_sigma(params, x, Sigma_hat, grad_ln_Sigma, sigma_kms)`
  - Script: `gravity_learn/eval/cluster_sigma_test.py`
- [ ] **Data:** Load A1689, A2029, A478 X-ray temperatures from ACCEPT
- [ ] **Fitting:**
  - Fix (a, b, d) = (0.6687, 0.1401, 0.0871) from SPARC
  - Fit (e, α) on 3 clusters to match θ_E
  - Test on SPARC galaxies with σ from virial mass
- [ ] **Critical Test:** Does galaxy median APE stay < 0.30?
- [ ] **Decision:**
  - ✅ PASS → 5-parameter model, publish immediately
  - ⚠️ PARTIAL → Proceed to two-regime model (Test 1.7)
  - ❌ FAIL → Move to Test 2
- [ ] **Output:** `velocity_dispersion_gating/best_params.json`, `sigma_test_results.csv`

**Estimated Time:** 1 week  
**Priority:** 🔴 Highest - Test first

---

#### Test 2: Hot Gas Fraction Gating
- [ ] **Hypothesis:** Hot X-ray gas (80% in clusters) gates differently than cold HI/stars (20% in galaxies). Baryon phase flip explains boost.
- [ ] **Formula:** `fX = [x² / (a - b·Σ̂ - d·|∇ln Σ|)] · (1 + k · f_gas)`
  - Add parameter: k (hot gas amplification factor)
  - f_gas = M_gas / (M_gas + M_stars)
  - Prediction: k ≈ 40 → galaxies 9× (f_gas=0.2), clusters 33× (f_gas=0.8)
- [ ] **Implementation:**
  - Code: `compute_gas_fraction(M_gas_Msun, M_stars_Msun)`
  - Code: `fX_ratio_curv_gas_fraction(params, x, Sigma_hat, grad_ln_Sigma, f_gas)`
  - Script: `gravity_learn/eval/cluster_gas_fraction_test.py`
- [ ] **Data:** 
  - Clusters: ACCEPT gas profiles + Halkola et al. (2006) stellar profiles
  - Galaxies: SPARC HI fraction (~0.2 typical)
- [ ] **Fitting:**
  - Fix (a, b, d) from SPARC
  - Fit k on clusters
  - Test on galaxies with f_gas ≈ 0.2
- [ ] **Critical Test:** Can single k fit clusters within 30% AND keep galaxy APE < 0.28?
- [ ] **Decision:**
  - ✅ PASS (k ~ 20-50) → 4-parameter model, publish
  - ⚠️ MARGINAL (k > 100 or breaks galaxies) → Two-regime
  - ❌ FAIL → Move to Test 3
- [ ] **Output:** `hot_gas_gating/best_k.json`, `gas_fraction_test_results.csv`

**Estimated Time:** 1 week  
**Priority:** 🔴 Second highest

---

### **Phase 2: Medium Complexity (2 weeks each)**

#### Test 3: Gravitational Potential Depth Gating
- [ ] **Hypothesis:** Deeper wells (larger |Φ|/c²) amplify tail. GR-motivated scale-free measure.
- [ ] **Formula:** `fX = [x² / (a - b·Σ̂ - d·|∇ln Σ|)] · exp(β · |Φ|/Φ₀)`
  - Φ(R) = -∫ g(r) dr (gravitational potential)
  - Φ₀ = 10⁴ km²/s² reference
  - β = amplification strength
  - Alternative: Power law `(|Φ|/Φ₀)^γ`
- [ ] **Implementation:**
  - Code: `compute_gravitational_potential(R_kpc, g_total_kms2)`
  - Code: `fX_ratio_curv_potential(params, x, Sigma_hat, grad_ln_Sigma, Phi_km2s2)`
  - Script: `gravity_learn/eval/cluster_potential_test.py`
- [ ] **Data:** Compute Φ(R) from observed g(R) for clusters and 10 SPARC galaxies
- [ ] **Diagnostic:** Plot |Φ_typical| vs. system type
  - Expect: Galaxies |Φ| ~ 10⁴ km²/s², clusters |Φ| ~ 10⁵-10⁶ km²/s²
  - If ratio < 3×, potential depth doesn't discriminate enough
- [ ] **Fitting:** Fit β on clusters, test on galaxies
- [ ] **Critical Test:** Clear factor-of-10 in |Φ| AND β ~ 0.5-2.0 (physical)?
- [ ] **Decision:**
  - ✅ PASS → 4-parameter potential-gated model
  - ❌ FAIL (β > 10 or no |Φ| difference) → Move to Test 4
- [ ] **Output:** `potential_gating/best_beta.json`, `potential_diagnostic.png`

**Estimated Time:** 2 weeks  
**Priority:** 🟡 Medium

---

#### Test 4: Multi-Scale Curvature (Laplacian)
- [ ] **Hypothesis:** Second derivative ∇²Σ captures "curvature of curvature" at cluster scales (100-1000 kpc vs 10 kpc)
- [ ] **Formula:** `fX = x² / (a - b·Σ̂ - d·|∇ln Σ| - f·|∇²ln Σ|)`
  - ∇²ln Σ = d²(ln Σ)/dR² (Laplacian of log surface density)
  - f = Laplacian weight parameter
- [ ] **Hierarchical variant:** Separate local (10 kpc) and global (100 kpc) scales
- [ ] **Implementation:**
  - Code: `compute_laplacian_log_sigma(R_kpc, Sigma_Msun_pc2, smoothing_window=3)`
  - Code: `fX_ratio_curv_laplacian(params, x, Sigma_hat, grad_ln_Sigma, laplacian_ln_Sigma)`
  - Script: `gravity_learn/eval/cluster_laplacian_test.py`
- [ ] **Diagnostic First:** Run `diagnose_laplacian_importance(sparc_data, cluster_data)`
  - Compute median |∇²ln Σ| for galaxies vs clusters
  - If ratio < 2×, Laplacian not discriminative → SKIP this test
  - If ratio > 5×, proceed with full fit
- [ ] **Fitting:** Fit f on clusters if diagnostic passes
- [ ] **Critical Test:** Cluster |∇²Σ| > 5× galaxy values AND f helps significantly?
- [ ] **Decision:**
  - ✅ PASS → 4-parameter Laplacian-gated model
  - ❌ FAIL (ratio < 2× or f < 0.01) → Move to Test 5
- [ ] **Output:** `laplacian_gating/diagnostic.csv`, `laplacian_test_results.json`

**Estimated Time:** 2 weeks  
**Priority:** 🟡 Medium

---

### **Phase 3: Complex (1 month each)**

#### Test 5: Dynamical State / Merger History
- [ ] **Hypothesis:** Disturbed/merging clusters (e.g., Bullet) have enhanced lensing due to non-equilibrium dynamics
- [ ] **Formula:** `fX = fX_base · (1 + h · w_substructure)`
  - w_substructure = morphological asymmetry metric
  - Examples: |BCG_offset| / R_500, Gini coefficient, X-ray asymmetry
- [ ] **Implementation:**
  - Code: `compute_substructure_metric(xray_centroid_kpc, BCG_position_kpc, R500_kpc)`
  - Code: `fX_ratio_curv_dynamical(params, x, Sigma_hat, grad_ln_Sigma, w_substructure)`
  - Script: `gravity_learn/eval/cluster_dynamical_test.py`
- [ ] **Data Requirements:**
  - X-ray centroid from Chandra/XMM maps
  - BCG position from optical imaging
  - R_500 from mass estimates
- [ ] **Critical Test:** Do disturbed clusters (w > 0.1) systematically have larger θ_E than relaxed (w < 0.05)?
- [ ] **Expected Outcome:** Explains scatter, not mean offset → Keep as covariate, not solution
- [ ] **Output:** `dynamical_state/substructure_metrics.csv`, `w_vs_einstein_radius.png`

**Estimated Time:** 1 month (data acquisition bottleneck)  
**Priority:** 🟢 Low - mostly explains scatter

---

#### Test 6: Environmental Density (Large-Scale Structure)
- [ ] **Hypothesis:** Clusters in overdense cosmic web nodes experience stronger modification
- [ ] **Formula:** `fX = fX_base · (1 + g · log(δ_LSS / δ₀))`
  - δ_LSS = (ρ_local - ρ_cosmic) / ρ_cosmic at R ~ 5-10 Mpc
- [ ] **Implementation:**
  - Code: `compute_environmental_overdensity(cluster_ra_dec, z, catalog)`
  - Code: `fX_ratio_curv_environmental(params, x, Sigma_hat, grad_ln_Sigma, delta_LSS)`
  - Script: `gravity_learn/eval/cluster_environment_test.py`
- [ ] **Data Requirements:**
  - Galaxy counts around clusters (SDSS, DES photometric redshifts)
  - Weak lensing convergence maps (optional)
  - Or: Cosmological simulation LSS density field
- [ ] **Critical Test:** Correlate θ_E with local overdensity δ measured at 5 Mpc scale
- [ ] **Expected Outcome:** Requires new large-scale structure data → **Future work**
- [ ] **Output:** `environmental_density/delta_LSS_catalog.csv`, `theta_E_vs_delta.png`

**Estimated Time:** 1 month (requires LSS data)  
**Priority:** 🟢 Low - data-intensive, future work

---

### **Master Decision Tree**

```
START (Week 1)
│
├─ Run Test 1 (Velocity Dispersion) [Week 1]
│  ├─ ✅ PASS → Publish 5-param model, DONE
│  ├─ ⚠️ PARTIAL → Run Test 2
│  └─ ❌ FAIL → Run Test 2
│
├─ Run Test 2 (Hot Gas Fraction) [Week 2]
│  ├─ ✅ PASS → Publish 4-param model, DONE
│  └─ ❌ FAIL → Run Test 3
│
├─ Run Test 3 (Potential Depth) [Weeks 3-4]
│  ├─ ✅ PASS + Test 1 PARTIAL → Combined velocity+potential model
│  └─ ❌ FAIL → Run Test 4
│
├─ Run Test 4 (Multi-Scale Curvature) [Weeks 5-6]
│  ├─ Diagnostic shows 5× difference → Fit full model
│  │  ├─ ✅ PASS → Publish Laplacian-gated model
│  │  └─ ❌ FAIL → Run Test 5
│  └─ Diagnostic shows <2× → Skip to Test 5
│
├─ Run Test 5 (Dynamical State) [Weeks 7-10]
│  └─ Explains scatter only → Document as covariate, proceed to two-regime
│
└─ Run Test 6 (Environmental Density) [Future Work]
   └─ Requires new LSS data → Defer to follow-up project
```

**If ALL tests fail:** Accept two-regime interpretation
- Galaxy scale (R < 30 kpc): Pure O2 ratio_curv with (a, b, d)
- Cluster scale (R > 100 kpc): Requires dark matter OR different physics
- Publish honest assessment: "Geometry gating solves galaxies, not clusters"

---

### **Immediate Action Plan (Next 2 Weeks)**

**Week 1: Velocity Dispersion Test**
```bash
# Day 1-2: Data preparation
python -m gravity_learn.data.prepare_cluster_velocity_dispersions \
    --clusters A1689,A2029,A478 \
    --output data/clusters/velocity_dispersions.csv

# Day 3-5: Fit model
python -m gravity_learn.eval.fit_sigma_model \
    --galaxy_params O2_ratio_curv_publication/results/best_fit/mape_median_20250926_2259/best_family.json \
    --cluster_data data/clusters/velocity_dispersions.csv \
    --output experiments/cluster_extensions/sigma_test_$(date +%Y%m%d)

# Day 6-7: Validation
python -m gravity_learn.eval.validate_sigma_on_galaxies \
    --sigma_params experiments/cluster_extensions/sigma_test_YYYYMMDD/best_params.json \
    --sparc_data data/SPARC_Lelli2016_MasterFile.mrt
```

**Week 2: Hot Gas Fraction Test** (if Week 1 fails or partial)
```bash
# Day 1-2: Compute gas fractions
python -m gravity_learn.data.compute_gas_fractions \
    --clusters data/clusters/ \
    --output data/clusters/gas_fractions.csv

# Day 3-5: Fit model
python -m gravity_learn.eval.fit_gas_model \
    --galaxy_params O2_ratio_curv_publication/results/best_fit/mape_median_20250926_2259/best_family.json \
    --cluster_data data/clusters/gas_fractions.csv \
    --output experiments/cluster_extensions/gas_test_$(date +%Y%m%d)

# Day 6-7: Validation + comparison with Test 1
python -m gravity_learn.eval.compare_cluster_models \
    --models sigma_test,gas_test \
    --output experiments/cluster_extensions/model_comparison.csv
```

---

### **Unified Success Evaluation Function**

```python
def evaluate_cluster_extension(model_name, cluster_results, galaxy_results):
    """
    Universal success criterion for all cluster extension tests.
    
    Parameters:
    -----------
    model_name : str
        Name of test (e.g., 'velocity_dispersion', 'hot_gas_fraction')
    cluster_results : list of dict
        Each dict: {'cluster', 'theta_E_obs', 'theta_E_pred', 'z', ...}
    galaxy_results : dict
        {'median_APE', 'median_RMSE', 'IQR_APE', ...}
    
    Returns:
    --------
    decision : str
        'PUBLISH' | 'TWO_REGIME_MODEL' | 'NEXT_TEST'
    """
    # Cluster criterion: All Einstein radii within 30%
    cluster_pass = all([
        abs(r['theta_E_pred'] - r['theta_E_obs']) / r['theta_E_obs'] < 0.3
        for r in cluster_results
    ])
    
    # Galaxy criterion: Median APE must not degrade beyond 0.30
    # (6-point degradation from baseline 0.242 is max acceptable)
    galaxy_pass = galaxy_results['median_APE'] < 0.30
    
    if cluster_pass and galaxy_pass:
        print(f"✅ {model_name} SUCCEEDS")
        print(f"   Cluster accuracy: {[r['theta_E_pred']/r['theta_E_obs'] for r in cluster_results]}")
        print(f"   Galaxy APE: {galaxy_results['median_APE']:.3f}")
        return "PUBLISH"
    
    elif cluster_pass and not galaxy_pass:
        print(f"⚠️ {model_name} PARTIAL - clusters work, galaxies break")
        print(f"   Consider two-regime model or combined approach")
        return "TWO_REGIME_MODEL"
    
    else:
        print(f"❌ {model_name} FAILS")
        print(f"   Cluster fit inadequate, move to next test")
        return "NEXT_TEST"
```

---

**Estimated Timeline:** 6-10 weeks (depends on how many tests needed)  
**Deliverables:**
- 6 test implementations (Python scripts + functions)
- Test results for each attempted approach (JSON/CSV)
- Decision tree outcome report
- Either: Extended parameter model OR two-regime interpretation
- Follow-up paper draft

**Publication Impact:** Major follow-up paper if any test succeeds, honest assessment paper if all fail

---

## 6️⃣ Lensing-Dynamics Decoupling (06_lensing_dynamics_decoupling/) 🟢 P2

**Goal:** Test if lensing and dynamics predict different amplitudes

### Tasks:
- [ ] Separate amplitude fitting
  - Fit (a_dyn, b_dyn, d_dyn) on rotation curves
  - Fit (a_lens, b_lens, d_lens) on weak lensing stacks
  - Test if a_dyn ≠ a_lens (k-mouflage predicts Σ_lens ~ 0.97)
  - **Output:** `lensing_dynamics_decoupling.json`

- [ ] Galaxy-galaxy weak lensing comparison
  - Use SDSS/DES/KiDS stacked shear profiles at 30-300 kpc
  - Predict ΔΣ(R) from O2 model
  - Compare observed vs. predicted lensing amplitude
  - **Output:** `gg_lensing_comparison.csv`, `gg_lensing_overlay.png`

- [ ] Lensing-to-dynamics ratio analysis
  - Compute Σ_lens / Σ_dyn as function of R
  - Expected: constant ~ 1 for GR, constant < 1 for some modified gravity
  - **Output:** `lensing_dynamics_ratio.csv`, `ratio_vs_radius.png`

- [ ] Strong lensing + dynamics joint fit
  - Use galaxies with both rotation curves AND lensing arcs/shear
  - Simultaneously fit dynamics and lensing with shared geometry
  - **Output:** `joint_lensing_dynamics_fit.json`

**Estimated Timeline:** 2 weeks (data acquisition may be bottleneck)  
**Deliverables:**
- Decoupling test results
- GG lensing comparison
- Lensing/dynamics ratio plots
- Joint fit report

**Publication Impact:** Discriminant for theory type (k-mouflage vs. others)

---

## 7️⃣ Cosmological Implementation (07_cosmological_implementation/) 🔵 P3

**Goal:** Implement geometry gating in N-body simulations

### Tasks:
- [ ] Field solver implementation
  - Write geometry-gated Poisson solver for RAMSES or AREPO
  - Compute Σ(R) on-the-fly from particle distribution
  - Apply gating: g_total = g_N · (1 + fX)
  - **Output:** `g3_field_solver.f90` or `.cpp`

- [ ] Zoom-in galaxy simulation
  - Run Milky Way-mass halo (M_200 ~ 10¹² M☉)
  - Track star formation, disk formation
  - Compare with dark matter run (same ICs)
  - **Output:** `zoom_mw_g3/` (snapshots, analysis)

- [ ] Structure formation impact
  - Run cosmological box (50-100 Mpc) with geometry gating
  - Measure: matter power spectrum, halo mass function, galaxy clustering
  - Compare with ΛCDM and MOND simulations
  - **Output:** `cosmo_box_g3/` (snapshots, power spectra)

- [ ] Disk stability test
  - Does geometry gating affect bar formation?
  - Does it affect spiral structure?
  - Compare disk kinematics in G³ vs. dark matter sims
  - **Output:** `disk_stability_analysis.md`

**Estimated Timeline:** 3-6 months (major computational project)  
**Deliverables:**
- Field solver code (GitHub repo)
- Zoom-in simulation outputs
- Cosmological box results
- Comparison with ΛCDM paper

**Publication Impact:** High-profile separate paper (Nature Astronomy tier if successful)

---

## 8️⃣ Field Theory Foundation (08_field_theory_foundation/) 🔵 P3

**Goal:** Derive O2 ratio_curv from fundamental scalar-tensor theory

### Tasks:
- [ ] K-mouflage derivation
  - Start with Lagrangian: L = Λ⁴ P(Y), Y = (∇φ)²/Λ⁴
  - Impose geometry-dependent mobility: μ(Σ, ∇Σ)
  - Derive field equation and quasi-static limit
  - Show ratio_curv emerges in thin disk approximation
  - **Output:** `kmouflage_derivation.pdf` (LaTeX)

- [ ] Solar system tests
  - Compute PPN parameters (γ, β) for k-mouflage + geometry gating
  - Check Cassini bound: |γ - 1| < 2.3 × 10⁻⁵
  - Verify perihelion precession (Mercury)
  - **Output:** `solar_system_tests.pdf`

- [ ] Gravitational wave constraints
  - Compute tensor sound speed c_T (must be 1 for GW170817)
  - Check if k-essence sector preserves c_T = 1
  - Scalar sound speed c_s² (should be subluminal, ~ 0.5)
  - **Output:** `gw_constraints.pdf`

- [ ] Stability analysis
  - Ghost-free conditions: P_Y > 0
  - Gradient stability: P_Y + 2Y P_YY > 0
  - Tachyon avoidance: c_s² > 0
  - **Output:** `stability_conditions.pdf`

- [ ] Vainshtein screening verification
  - Compute Vainshtein radius for Solar System
  - Show screening recovers GR in high-density regions
  - **Output:** `vainshtein_screening.pdf`

**Estimated Timeline:** 2-3 months (theoretical work)  
**Deliverables:**
- Complete field theory derivation (LaTeX)
- Solar system test calculations
- GW constraint verification
- Stability proof
- Theory paper draft

**Publication Impact:** High-profile theory paper (PRD, JCAP)

---

## 9️⃣ Quantum/Statistical Origin (09_quantum_statistical_origin/) 🔵 P3

**Goal:** Explore emergent gravity interpretation of geometry gating

### Tasks:
- [ ] Holographic screen hypothesis
  - Connection: Surface density Σ ↔ entanglement entropy S
  - Test: Does Σ-dependent gating emerge from holographic principle?
  - **Output:** `holographic_gating.pdf`

- [ ] Entropic force derivation
  - Verlinde-style emergent gravity with geometry dependence
  - Derive: F = -∇(TS) with S(Σ, ∇Σ)
  - Show if ratio_curv form can emerge
  - **Output:** `entropic_force_derivation.pdf`

- [ ] Quantum information perspective
  - Entanglement entropy between bulk and boundary
  - Area law violations near galaxy outskirts (low Σ)?
  - **Output:** `quantum_info_perspective.pdf`

- [ ] Statistical mechanics analogy
  - Treat galaxy as thermodynamic system
  - Surface density as entropy density
  - Gating as phase transition (screening ↔ unscreening)
  - **Output:** `stat_mech_analogy.pdf`

**Estimated Timeline:** 3-6 months (highly speculative)  
**Deliverables:**
- Theoretical exploration papers (4)
- Each explores different emergent gravity angle
- Assess viability of each approach

**Publication Impact:** Speculative theory papers (Foundations of Physics, PRD if rigorous)

---

## 🔟 Observational Tests (10_observational_tests/) 🟢 P2

**Goal:** Design and execute new observational tests of geometry gating

### Tasks:
- [ ] Extended rotation curve survey
  - Apply O2 model to THINGS + LITTLE THINGS (300+ galaxies)
  - Check if median APE stays at 0.24 or degrades
  - Test on SPARC-independent dataset
  - **Output:** `extended_survey_results.csv`, `extended_ape_distribution.png`

- [ ] High-resolution IFU test
  - Use MUSE, KCWI, or MaNGA 2D velocity fields
  - Test non-circular motions (bars, spiral arms)
  - Check if geometry gating predicts asymmetries
  - **Output:** `ifu_2d_test_results.md`, `2d_velocity_field_comparison.png`

- [ ] Dwarf spheroidals (MW satellites)
  - Use stellar kinematics (velocity dispersion σ)
  - Jeans modeling with geometry-gated potential
  - Compare with dark matter and MOND predictions
  - **Output:** `dwarf_spheroidals_test.csv`, `sigma_vs_radius_comparison.png`

- [ ] Lensing-dynamics comparison on same objects
  - Find galaxies with both rotation curves AND weak lensing shear
  - Measure Σ_lens / Σ_dyn ratio
  - Test k-mouflage prediction: Σ_lens ~ 0.97
  - **Output:** `lensing_dynamics_same_objects.csv`

- [ ] Low surface brightness (LSB) extreme test
  - Target ultra-diffuse galaxies (UDGs) and LSB dwarfs
  - These have Σ̂ < -2, strong gating regime
  - Predict: Model should work well (gate fully on)
  - **Output:** `lsb_extreme_test.csv`, `lsb_performance.png`

**Estimated Timeline:** 3-6 months (data acquisition + analysis)  
**Deliverables:**
- 5 observational test results
- Each tests different aspect/regime
- Comparison tables and diagnostic plots

**Publication Impact:** 1-2 observational papers, ApJ tier

---

## 1️⃣1️⃣ Hybrid Models (11_hybrid_models/) 🟢 P2

**Goal:** Test geometry gating + minimal dark matter combinations

### Tasks:
- [ ] 10% dark matter + geometry gating
  - Add NFW halo with M_200 = 10% of literature value
  - Keep geometry gating for galaxies
  - Test if this explains both galaxies AND clusters
  - **Output:** `hybrid_10pct_dm_results.json`

- [ ] 50% dark matter + geometry gating
  - More conservative: 50% dark matter reduction
  - Use geometry gating to reduce DM budget by half
  - **Output:** `hybrid_50pct_dm_results.json`

- [ ] Scale-dependent dark matter
  - Dark matter dominant at cluster scales (R > 100 kpc)
  - Geometry gating dominant at galaxy scales (R < 30 kpc)
  - Smooth transition between regimes
  - **Output:** `scale_dependent_dm_results.json`

- [ ] Unified fit: galaxies + clusters
  - Optimize hybrid model on SPARC + cluster lensing jointly
  - Find minimal dark matter budget that closes cluster gap
  - **Output:** `unified_hybrid_fit.json`, `dm_budget_analysis.md`

- [ ] Budget comparison
  - Compare: pure DM, pure G³, hybrid models
  - Metric: Total DM mass required per system
  - **Output:** `dm_budget_comparison.csv`, `budget_bar_chart.png`

**Estimated Timeline:** 2-3 weeks  
**Deliverables:**
- 4 hybrid model variants tested
- Unified fit results
- DM budget comparison
- Recommendation report

**Publication Impact:** Pragmatic middle-ground paper, high interest

---

## 📁 Project Organization

```
O2_ratio_curv_publication/
├── 01_core_publication/          # Main paper, figures, tables
├── 02_uncertainty_quantification/ # Bootstrap, prediction bands
├── 03_type_specific_analysis/     # Per-type breakdowns
├── 04_extended_symbolic_regression/ # Feature discovery
├── 05_cluster_adapted_gating/     # Scale-dependent models
├── 06_lensing_dynamics_decoupling/ # Lensing tests
├── 07_cosmological_implementation/ # N-body sims
├── 08_field_theory_foundation/    # Theoretical derivation
├── 09_quantum_statistical_origin/ # Emergent gravity
├── 10_observational_tests/        # New data analysis
├── 11_hybrid_models/              # DM + geometry gating
├── code/                          # Shared analysis code
├── data/                          # Datasets (symlinks to main data/)
├── figures/                       # All generated figures
├── results/                       # All JSON/CSV results
├── documentation/                 # READMEs, notes
└── MASTER_TODO.md                 # This file
```

---

## 📈 Progress Tracking

### Overall Completion: **15%** (Core paper written, figures pending)

| Research Path | Progress | Priority | Status |
|---------------|----------|----------|--------|
| 01. Core Publication | 60% | P0 🔴 | In Progress |
| 02. Uncertainty Quantification | 0% | P1 🟡 | Not Started |
| 03. Type-Specific Analysis | 0% | P1 🟡 | Not Started |
| 04. Extended Symbolic Regression | 0% | P2 🟢 | Not Started |
| 05. Cluster-Adapted Gating | 0% | P2 🟢 | Not Started |
| 06. Lensing-Dynamics Decoupling | 0% | P2 🟢 | Not Started |
| 07. Cosmological Implementation | 0% | P3 🔵 | Not Started |
| 08. Field Theory Foundation | 0% | P3 🔵 | Not Started |
| 09. Quantum/Statistical Origin | 0% | P3 🔵 | Not Started |
| 10. Observational Tests | 0% | P2 🟢 | Not Started |
| 11. Hybrid Models | 0% | P2 🟢 | Not Started |

---

## ⏱️ Timeline Estimates

**Phase 1 (Weeks 1-3):** Core publication finalization + submission  
**Phase 2 (Weeks 4-6):** Uncertainty quantification + Type-specific analysis  
**Phase 3 (Months 2-3):** Extended SR + observational tests  
**Phase 4 (Months 3-6):** Cluster gating + hybrid models  
**Phase 5 (Months 6-12):** Cosmological sims + field theory  
**Phase 6 (Months 12-24):** Quantum/statistical explorations (long-term)

---

## 🎓 Publication Strategy

**Primary Paper:** "Geometry-Gated Gravity: Surface Density and Curvature Determine Flat Galaxy Rotation Curves"  
- Target: ApJ or MNRAS  
- Status: Draft complete, figures pending  
- Timeline: Submit by Week 3

**Follow-up Papers (Planned):**
1. "Uncertainty Quantification and Type-Specific Analysis of Geometry-Gated Gravity" (Weeks 4-8)
2. "Cluster-Adapted Gating Extensions for Geometry-Based Modified Gravity" (Months 3-4)
3. "Observational Tests of Geometry Gating in Extended Galaxy Samples" (Months 4-6)
4. "Hybrid Models: Geometry Gating + Minimal Dark Matter" (Months 6-8)
5. "Field Theory Foundation of Geometry-Gated Gravity" (Months 8-12)

---

## 📞 Contact & Collaboration

**Lead:** Henry Speiser  
**Repository:** https://github.com/lrspeiser/GravityCalculator  
**Status Updates:** Track in this file + GitHub issues  

**Collaboration Opportunities:**
- Observational astronomers: Tests 10 (rotation curves, lensing, kinematics)
- Numerical simulators: Test 7 (N-body implementation)
- Theorists: Tests 8-9 (field theory, emergent gravity)
- Data scientists: Tests 2-4 (uncertainty, SR, analysis)

---

## 📝 Notes

**Philosophy:** Do one thing well, document thoroughly, be honest about limitations. Each research path is independent and publishable on its own merit.

**Reproducibility:** Every analysis generates:
1. JSON/CSV results file
2. Diagnostic plots (PNG/PDF)
3. README with exact commands
4. Code in `code/` subfolder

**Git Strategy:** Commit after each completed task, push to GitHub daily.

---

**Last Updated:** October 2, 2025  
**Next Review:** After core publication submission (Week 3)
