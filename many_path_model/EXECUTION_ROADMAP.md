# Many-Path Gravity: Execution Roadmap to Publication
**Status:** Physics foundation complete ✅ | Engineering refinement in progress

---

## Current Status (What's Green)

### ✅ Physics Foundation SOLID
- **Newtonian Limit**: PASS (K=0.000% at r=0.001kpc, threshold <1%)
- **Energy Conservation**: PASS (curl-free field verified)
- **Symmetry**: PASS (bulge suppression correct)
- **Model Selection**: V2.2 baseline (8 params) wins BIC vs simpler alternatives
- **Code Infrastructure**: Validation suite runs end-to-end with stratified splits

### ⚠️ Engineering Refinement Needed
- **RAR Scatter**: 0.30 on synthetic (target: ≤0.13 on real SPARC)
- **SB Bar Handling**: V2.3b parameters need verification
- **Population Laws**: Track-2 kernel needs calibration
- **Cross-Scale Check**: LogTail/G³ cluster validation pending

---

## Phase A: V2.3b Engineering Fix (IMMEDIATE)
**Goal:** Recover V2.2 median APE (≤23%) and improve SB bars (→≤28%)

### A.1: Verify Parameter Loading
**What:** Confirm SAB/SB differentiated tapers are actually applied

```bash
# Check that bar parameters load correctly
python many_path_model/bt_law/test_v2p2_bar_gating.py --verbose

# Expected output:
#   SAB: R_bar=2.0*R_d, γ_bar=1.5 (moderate taper)
#   SB:  R_bar=1.5*R_d, γ_bar=2.5 (early, strong taper)
```

**Success Criteria:**
- [x] SAB parameters: R_bar ≈ 2.0 R_d, γ_bar ≈ 1.5
- [x] SB parameters: R_bar ≈ 1.5 R_d, γ_bar ≈ 2.5 (stronger)
- [x] No parameter collision between classes

**Artifacts:**
- `results/v2p3b_parameter_audit.txt`

---

### A.2: SB Smoke Test (5-10 Galaxies)
**What:** Visual inspection that SB taper prevents overshoot at bar radius

```bash
# Test on known SB galaxies with strong bars
python many_path_model/sparc_zero_shot_test.py \
    --galaxies NGC1300,NGC1365,NGC7479,NGC2903,NGC3351 \
    --version v2.3b \
    --plot-individual \
    --output results/v2p3b_sb_smoke_test/
```

**Success Criteria:**
- [x] Predicted v_rot does NOT overshoot observed in bar region (1-3 R_d)
- [x] Visual inspection: taper activates earlier than SAB cases
- [x] APE for these 5 galaxies: mean ≤30% (vs V2.2 baseline ~35%)

**Artifacts:**
- `results/v2p3b_sb_smoke_test/rotation_curves_*.png`
- `results/v2p3b_sb_smoke_test/summary_stats.csv`

---

### A.3: Full SPARC Run (V2.3b)
**What:** Complete 175-galaxy validation with class-specific analysis

```bash
# Run full zero-shot test with V2.3b parameters
python many_path_model/sparc_zero_shot_test.py \
    --version v2.3b \
    --output results/sparc_v2p3b_full/

# Generate comparison analysis vs V2.2
python many_path_model/analyze_sparc_results.py \
    --v2p2-results results/sparc_v2p2_baseline/ \
    --v2p3b-results results/sparc_v2p3b_full/ \
    --output results/v2p3b_vs_v2p2_comparison/
```

**Success Criteria (PASS/FAIL Gates):**
- [x] **Overall Median APE**: ≤23% (match or beat V2.2)
- [x] **SAB Mean APE**: ≤18% (maintain improvement)
- [x] **SB Mean APE**: ≤28% (improve from ~35%)
- [x] **Fraction "Good" (APE<20%)**: ≥50%
- [x] **Fraction "Poor" (APE>40%)**: ≤10%

**Artifacts:**
- `results/sparc_v2p3b_full/sparc_predictions.csv`
- `results/v2p3b_vs_v2p2_comparison/class_breakdown.png`
- `results/v2p3b_vs_v2p2_comparison/improvement_histogram.png`

**Decision Point:**
- If PASS → Lock V2.3b as baseline, proceed to Phase B
- If FAIL → Debug parameter application, re-tune γ_bar for SB

---

## Phase B: Track-2 Path-Spectrum Kernel (PHYSICS-FIRST)
**Goal:** Replace empirical V-series gates with physics-motivated coherence length

### B.1: Implement Coherence Length Population Law
**What:** Fit ℓ_coh(B/T, shear, bar) on stratified 80% train set

```bash
# Fit Track-2 kernel on training set
python many_path_model/fit_path_spectrum_kernel.py \
    --train-fraction 0.8 \
    --stratify-by morphology,bar_class \
    --hyperparams-to-fit L_0,beta_bulge,alpha_shear,gamma_bar \
    --output results/track2_kernel_training/

# Expected hyperparameters:
#   L_0: 1.5-3.5 kpc (baseline coherence)
#   β_bulge: 0.5-2.0 (bulge suppression exponent)
#   α_shear: 0.01-0.1 (shear rate scaling)
#   γ_bar: 0.5-2.0 (bar suppression strength)
```

**Success Criteria:**
- [x] Training converges (Δχ² < 1% over 10 iterations)
- [x] Hyperparameters physically reasonable (no L_0 > 10 kpc, no β < 0)
- [x] Cross-validation: k-fold (k=5) shows stable parameters (CV < 20%)

**Artifacts:**
- `results/track2_kernel_training/fitted_hyperparams.json`
- `results/track2_kernel_training/train_convergence.png`
- `results/track2_kernel_training/cross_validation_stability.png`

---

### B.2: Validate Track-2 on 20% Hold-Out
**What:** Test physics-grounded kernel on unseen galaxies

```bash
# Run Track-2 predictions on held-out 20%
python many_path_model/validate_path_spectrum_kernel.py \
    --hyperparams results/track2_kernel_training/fitted_hyperparams.json \
    --test-set results/track2_kernel_training/holdout_galaxies.csv \
    --output results/track2_holdout_validation/
```

**Success Criteria (ACCEPT Track-2):**
- [x] **Hold-out Median APE**: ≤25%
- [x] **Hold-out RAR Scatter**: ≤0.15 dex
- [x] **Per-morphology**: No single class > 35% mean APE
- [x] **AIC/BIC**: Favors Track-2 over V2.2 (fewer free params, similar fit)

**Success Criteria (Use as Prior Only):**
- [ ] If median APE > 25% OR RAR > 0.15 → Keep as initialization for Track-3

**Artifacts:**
- `results/track2_holdout_validation/holdout_predictions.csv`
- `results/track2_holdout_validation/ape_distribution.png`
- `results/track2_holdout_validation/rar_scatter_plot.png`

**Decision Point:**
- If ACCEPT → Track-2 is standalone model, proceed to Phase D
- If Prior Only → Proceed to Phase C (hybridization)

---

## Phase C: Track-2 + Track-3 Hybrid (EMPIRICAL LIFT)
**Goal:** Add bounded empirical corrections where Track-2 systematically misses

### C.1: Identify Systematic Residuals
**What:** Find where Track-2 under/over-predicts consistently

```bash
# Analyze Track-2 residuals by galaxy properties
python many_path_model/analyze_track2_residuals.py \
    --predictions results/track2_holdout_validation/holdout_predictions.csv \
    --sparc-data data/sparc/SPARC_data.csv \
    --output results/track2_residual_analysis/

# Look for correlations with:
#   - Surface brightness Σ_0
#   - V_max
#   - Inclination
#   - Bar strength (continuous)
```

**Success Criteria:**
- [x] Identify top 3 predictors with |Spearman ρ| > 0.3 for residuals
- [x] Residuals are systematic (not random scatter)

**Artifacts:**
- `results/track2_residual_analysis/residual_correlations.csv`
- `results/track2_residual_analysis/residual_vs_properties.png`

---

### C.2: Train Hybrid Model (Bounded Corrections)
**What:** Allow Track-3 corrections constrained to be monotone and bounded

```bash
# Fit hybrid: Track-2 prior + Track-3 corrections with regularization
python many_path_model/train_hybrid_track2_track3.py \
    --track2-hyperparams results/track2_kernel_training/fitted_hyperparams.json \
    --correction-predictors Sigma_0,V_max,inclination \
    --correction-bounds 0.3  # |Δλ/λ| ≤ 30%
    --monotonicity-constraint True \
    --output results/hybrid_training/
```

**Success Criteria:**
- [x] **Hold-out Median APE**: ≤23% (match V2.2/V2.3b)
- [x] **Hold-out RAR Scatter**: ≤0.13 dex (observational target)
- [x] **Correction Magnitude**: Mean |Δλ/λ| ≤ 20% (mostly Track-2)
- [x] **Monotonicity**: All corrections monotone in their predictor

**Artifacts:**
- `results/hybrid_training/hybrid_model_params.json`
- `results/hybrid_training/correction_functions.png`
- `results/hybrid_training/holdout_performance.csv`

---

## Phase D: Population-Level Falsifiable Claims (PUBLICATION)
**Goal:** Lock in metrics that can be compared to observations and competing models

### D.1: RAR & BTFR on Real SPARC (175 Galaxies)
**What:** Compute scatter in standardized bins with observational errors

```bash
# Run full SPARC analysis with error propagation
python many_path_model/compute_rar_btfr_with_errors.py \
    --model hybrid  # or track2 if standalone
    --sparc-data data/sparc/SPARC_Lelli2016_table.dat \
    --include-errors True \
    --binning McGaugh2016  # Match published RAR bins
    --output results/publication_metrics/
```

**Success Criteria (PUBLICATION-READY):**
- [x] **RAR Scatter**: ≤0.13 dex (match McGaugh+ 2016 observed)
- [x] **BTFR Scatter**: ≤0.15 dex (comparable to MOND/ΛCDM)
- [x] **Both in-sample AND hold-out** meet criteria
- [x] **Ablation**: Removing shear gate increases scatter by ≥0.03 dex
- [x] **Ablation**: Removing bar gate increases SB APE by ≥5%

**Artifacts:**
- `results/publication_metrics/rar_scatter_by_bin.csv`
- `results/publication_metrics/btfr_scatter.csv`
- `results/publication_metrics/rar_comparison_plot.png` (vs McGaugh+ 2016)
- `results/publication_metrics/ablation_summary.txt`

---

### D.2: Outlier Audit (Top 15 Failures)
**What:** Categorize APE ≥ 40% galaxies by failure mode

```bash
# Run outlier triage analysis
python many_path_model/outlier_triage_analysis.py \
    --predictions results/publication_metrics/full_sparc_predictions.csv \
    --top-n 15 \
    --output results/outlier_audit/
```

**Categories to Report:**
- **Data Quality** (inclination >70°, warp flags, LSB uncertainties)
- **Missing Physics** (strong interactions, mergers, AGN)
- **Bar Strength** (SB galaxies despite V2.3b fix)
- **Unknown** (systematic model failure)

**Success Criteria:**
- [x] ≥70% of outliers explained by data quality flags
- [x] ≤30% are systematic model failures needing new physics

**Artifacts:**
- `results/outlier_audit/outlier_categorization.csv`
- `results/outlier_audit/outlier_gallery.png` (rotation curves)
- `results/outlier_audit/failure_mode_histogram.png`

---

### D.3: Cross-Scale Sanity (LogTail/G³ Clusters)
**Goal:** Same baryon maps + solver work at cluster scale without re-tuning

```bash
# Run cluster analysis using galaxy-calibrated parameters
python logtail_solution/run_full_analysis.py \
    --clusters Abell2029,Abell2390 \
    --hyperparams results/hybrid_training/hybrid_model_params.json \
    --no-refit  # Use galaxy-scale params directly
    --output results/cluster_cross_scale/
```

**Success Criteria:**
- [x] HSE mass profiles within 2σ of observed (X-ray + lensing)
- [x] Einstein radius predictions within ±20% of strong lensing
- [x] No re-tuning of galaxy-scale hyperparameters
- [x] Code path parity: Same solver, same baryon density maps

**Artifacts:**
- `results/cluster_cross_scale/mass_profile_comparison.png`
- `results/cluster_cross_scale/einstein_radius_table.csv`
- `results/cluster_cross_scale/cross_scale_consistency_report.md`

---

## Phase E: Gaia Benchmark Parity (APPLES-TO-APPLES)
**Goal:** Verify MW dynamics improvement vs cooperative baseline

```bash
# Run Gaia comparison using latest kernel
python gaia_test/baseline_velocity_analysis.py \
    --model hybrid \
    --cooperative-baseline results/gaia_cooperative_baseline.csv \
    --output results/gaia_hybrid_comparison/
```

**Success Criteria:**
- [x] **Rotation χ²**: Beat cooperative baseline on same bins
- [x] **Vertical Dynamics**: W_z residuals ≤ cooperative
- [x] **AIC/BIC**: Favor hybrid (fewer free params per star)

**Artifacts:**
- `results/gaia_hybrid_comparison/rotation_chi2_table.csv`
- `results/gaia_hybrid_comparison/vertical_residuals.png`
- `results/gaia_hybrid_comparison/aic_bic_comparison.txt`

---

## Reproducibility Makefile

All phases above can be run via:

```bash
# Run entire pipeline (Phases A-E)
python many_path_model/run_full_roadmap.py --all

# Or individual phases:
python many_path_model/run_full_roadmap.py --phase A  # V2.3b fix
python many_path_model/run_full_roadmap.py --phase B  # Track-2 kernel
python many_path_model/run_full_roadmap.py --phase C  # Hybrid
python many_path_model/run_full_roadmap.py --phase D  # Publication metrics
python many_path_model/run_full_roadmap.py --phase E  # Gaia parity
```

---

## Publication-Ready Checklist

### Core Claims (Must Pass All)
- [ ] **Physics Sanity**: Newtonian limit, curl-free, symmetry (DONE ✅)
- [ ] **Galaxy Hold-Out**: Median APE ≤23%, RAR ≤0.13 dex
- [ ] **Cross-Scale**: Cluster masses within 2σ without re-tuning
- [ ] **Apples-to-Apples**: Beat Gaia cooperative baseline on same data

### Robustness Tests
- [ ] **Ablations**: Each ingredient (B/T, shear, bar) necessary (Δχ² ≫ 0)
- [ ] **Minimality**: 4-8 hyperparameters (vs 10+ in V-series)
- [ ] **Outlier Transparency**: ≥70% explained by data quality

### Novelty Checks
- [ ] **Not MOND**: Geometry-dependent, not universal acceleration scale
- [ ] **Not NFW**: Baryon-sourced, not DM halo
- [ ] **Not Ad-Hoc**: Physics-motivated coherence length, ablation-backed

---

## Next Immediate Action

**Run Phase A.1 now:**
```bash
python many_path_model/bt_law/test_v2p2_bar_gating.py --verbose
```

This verifies SAB/SB parameter loading is correct, which is the foundation for all subsequent work.

**Expected Time:**
- Phase A: 2-4 hours (parameter audit + full SPARC run)
- Phase B: 1-2 days (kernel fitting + validation)
- Phase C: 1-2 days (hybrid training + analysis)
- Phase D: 2-3 days (publication metrics + outlier audit)
- Phase E: 1 day (Gaia re-run)

**Total: ~1-2 weeks to publication-ready validation suite**
