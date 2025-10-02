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
- [ ] 🔄 **IN PROGRESS** Generate Figure 1: Rotation curve overlays (6 galaxies)
- [ ] Generate Figure 2: Milky Way rotation curve (Gaia bins ±1σ)
- [ ] Generate Figure 3: Residual diagnostics (4-panel)
- [ ] Generate Figure 4: Cluster lensing failure (3-panel)
- [ ] Create supplementary Table S1: Per-galaxy metrics (120 galaxies CSV)
- [ ] Create supplementary Table S2: Cross-validation detailed results
- [ ] Create supplementary Table S3: Parameter sensitivity analysis

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

**Goal:** Develop scale-dependent gating to handle clusters

### Tasks:
- [ ] Laplacian gating hypothesis
  - Implement: fX_cluster = fX_galaxy × [1 + w_lap·∇²Σ]
  - Fit w_lap on cluster lensing data
  - Test if galaxy fits remain stable
  - **Output:** `laplacian_gating_results.json`

- [ ] Multi-scale smoothing
  - Smooth Σ over R_smooth ∈ [10, 50, 100, 200] kpc
  - Weight by scale: Σ_eff = Σ + Σ_smooth(R_smooth) weighted
  - Fit weights globally
  - **Output:** `multiscale_gating_results.json`

- [ ] Temperature-dependent screening
  - For clusters: gate depends on kT(r) (X-ray temperature)
  - Hypothesis: high-T regions amplify tail
  - Test on Perseus, A1689, A2029, A478
  - **Output:** `temperature_gating_results.json`

- [ ] Scale-dependent parameters
  - Test: b(R) = b₀ (R/R₀)^α, d(R) = d₀ (R/R₀)^β
  - Fit α, β to match clusters without breaking galaxies
  - **Output:** `scale_dependent_params.json`

- [ ] Unified galaxy+cluster fit
  - Combine SPARC + cluster lensing into single loss
  - Weight balance: 80% galaxies, 20% clusters
  - Optimize extended parameter set
  - **Output:** `unified_fit_results.json`, `unified_einstein_radii.csv`

**Estimated Timeline:** 3-4 weeks  
**Deliverables:**
- 4 gating extension variants tested
- Performance on galaxies + clusters
- Comparison table vs. baseline O2
- Recommendation report

**Publication Impact:** Major follow-up paper if successful

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
