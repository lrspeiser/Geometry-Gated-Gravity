# Roadmap to Publication-Grade Universal B/T Laws

## Current Status (V2.1 Baseline)

**Performance achieved:**
- Mean APE: 29.81% (vs 31.99% in V2)
- Median APE: 22.87% (vs 24.91% in V2)
- 71.4% within ±20% of per-galaxy best (vs 64.0% in V2)

**Key innovations implemented:**
1. ✅ Two-predictor lambda law: λ(B/T, S) suppresses coherence in high-shear systems
2. ✅ Radial ring concentration: Gaussian envelope prevents broad outer overshoot
3. ✅ Shear computation: S = -d(ln Ω)/d(ln R) at 2.2 R_d from baryonic curves
4. ✅ Bar classification: Extracted from SPARC Hubble types (30 SB, 10 SAB)

**Remaining gaps:**
- ~35% of galaxies still have APE ≥ 30% (outliers)
- Intermediate spirals scatter more than late/early types
- Gate parameters need robust, cross-validated refitting
- Bar suppression not yet integrated into laws

---

## Priority 1: Robust V2.1+ Gate Refitting (CRITICAL)

### Goal
Lock in S₀, γ_shear, (λ_min, λ_max, γ_b, γ_s), ring envelope (b₀, b₁, σ_ring), and **bar gates (g_SA, g_SAB, g_SB)** against per-galaxy best fits using robust regression with cross-validation.

### Implementation Steps

1. **Add bar gating to V2.1 laws** ✅ (data extracted, ready to integrate)
   - Multiply ring_amp by g_bar
   - Multiply lambda_ring by g_bar
   - Bars destroy azimuthal coherence → suppress ring term

2. **Robust regression with Huber/Tukey weighting**
   - Target: per-galaxy best (η̂, ring_amp̂, λ̂, M_max̂)
   - Predictors: B/T, Sigma0, shear S, bar g_bar, R_d (for size scaling only)
   - Loss: Huber (ρ=1.35) to handle outliers
   - Constraints: Enforce monotonicity in B/T, shear, bar_strength

3. **5-fold stratified cross-validation**
   - Stratify by morphology (late/intermediate/early)
   - Report mean/median APE on held-out folds
   - Ensure no one morphology class dominates fitting

4. **Sanity guardrails during fitting**
   - After each candidate law, check:
     - Milky Way vertical lag ≈ 15 ± 5 km/s at z=0.5 kpc
     - Milky Way outer slope |dv/dR| ≲ few km/s/kpc
     - Solar system gate: M ≈ 0 at R ≲ 0.5 kpc
   - These are already encoded in `parameter_optimizer.py`

### Target Performance
- **Median APE ≤ 20%**
- **≥ 80% within ±20% of per-galaxy best**
- **Mean APE ≤ 25%**

### Files to Create
- `fit_bt_laws_v2p2_robust.py` - Robust regression with CV
- `bt_laws_v2p2.py` - V2.1 + bar gating
- `evaluate_bt_laws_v2p2.py` - Full SPARC evaluation

---

## Priority 2: Outlier Triage and Surgical Fixes

### Goal
Characterize the worst 30-40 galaxies (APE ≥ 40%) and add **one** new predictor if a clear pattern emerges.

### Implementation Steps

1. **Create outlier gallery** (already have code for this)
   - Export worst Δ(BT-law - per-galaxy best) cases
   - Generate 3×3 or 4×4 panels with:
     - Rotation curve comparison
     - Parameter bar charts
     - Hubble type + bar class annotation

2. **Tag each outlier systematically**
   - Bar strength (SB/SAB/SA)
   - LSB vs HSB (Sigma0 < 50 vs > 500)
   - Warp indicators (if available from literature)
   - Interaction flags (if available)

3. **If barred systems dominate outliers:**
   - Already have bar gating ready to integrate
   - Expected improvement: ~5-8% median APE reduction in SB/SAB

4. **If super-LSB dwarfs dominate outliers:**
   - Add floor to compactness gate: `eta_floor = 0.1` for Sigma0 < 20
   - Prevent overshoot in tiniest disks

5. **If warped/thick disks dominate:**
   - Add vertical thickness proxy (if available):
     - Use outer velocity dispersion / v_c as proxy
     - Gate coherence κ with thickness factor
   - Or use simple type-based proxy (Sc/Sd assumed thicker)

### Target
- Reduce "poor" fraction from 35% to ≤ 25%
- No new predictor unless justified by clear systematic pattern

### Files to Create
- `analyze_outliers_v2p2.py` - Outlier characterization script
- `outlier_gallery_v2p2.png` - Visual diagnostic

---

## Priority 3: Conservative Field Implementation

### Goal
Convert acceleration multiplier to scalar potential to guarantee energy conservation and enable lensing predictions.

### Current Implementation
Multiplier sits directly on acceleration:
```
a_eff = a_Newton * (1 + M(r; geometry))
```

### Target Implementation
Scalar potential with geometry-aware kernel:
```
Φ(x) = ∫ ρ(x') * (1 + M(d; geometry)) / |x - x'| d³x'
a = -∇Φ
```

### Implementation Steps

1. **Grid-based potential evaluator**
   - 3D Cartesian grid (typical: 256³ cells, adaptive refinement near disk)
   - Compute Φ on grid using FFT convolution
   - Finite-difference gradient for acceleration

2. **Sanity checks at every step**
   - MW rotation curve χ²
   - Vertical lag at z=0.5 kpc
   - Outer slope penalty
   - Solar system gate (M ≈ 0 at R < 0.5 kpc)

3. **Lensing prediction**
   - Deflection angle from same potential
   - Compare to strong lensing observations (if available)

### Files to Modify/Create
- `conservative_field_kernel.py` - Potential-based implementation
- Integrate into existing `parameter_optimizer.py` sanity checks

---

## Priority 4: Publication Checklist

### Essential Validations

- [ ] **Cross-validated performance**
  - Report median APE, mean APE, σ_APE on 5-fold CV
  - Stratified by morphology
  - Compared to:
    - Per-galaxy best fits
    - Simple B/T-only law (V1)
    - MOND predictions (if available)

- [ ] **Milky Way constraints satisfied**
  - Rotation curve fit (χ² / dof ≲ 2)
  - Vertical lag: 15 ± 5 km/s at z=0.5 kpc
  - Outer slope: |dv/dR| ≲ 3 km/s/kpc beyond solar circle
  - Solar system: M(R < 0.5 kpc) ≈ 0 within errors

- [ ] **Physical interpretability**
  - All gate functions monotone and bounded
  - Parameters have clear physical meaning
  - No "epicycles" (arbitrarily complex functional forms)

- [ ] **Predictive power beyond fitting sample**
  - Apply to external sample (e.g., THINGS, LITTLE THINGS)
  - Zero free parameters per galaxy
  - Report performance vs per-galaxy tuning

### Figures for Paper

1. **Performance summary**
   - Violin plots of APE by morphology
   - V2.2 vs per-galaxy best scatter plot
   - Histogram of Δ(V2.2 - per-galaxy)

2. **Law visualization**
   - 2D heatmaps: η(B/T, Sigma0), λ(B/T, S), ring_amp(B/T, g_bar)
   - Show gate functions: f_Sigma(Sigma0), f_S(S), f_bar(bar_strength)

3. **Representative galaxies**
   - 4×4 gallery: early/intermediate/late × barred/unbarred
   - Each panel: rotation curve + parameter comparison

4. **Outlier analysis**
   - Identify failure modes
   - Show residual vs galaxy properties (Sigma0, S, g_bar, R_d)

5. **Milky Way validation**
   - Rotation curve fit
   - Vertical velocity distribution at z=0.5, 1.0 kpc
   - Comparison to Gaia DR3 data

---

## Hypothesis Being Tested

> **Many-path gravitational enhancement is governed by a small set of geometry/kinematics gates:**
> - **Sphericity** (B/T): Inner structure dominates path interference
> - **Compactness** (Sigma0): Surface density gates amplitude strength
> - **Shear** (S): Differential rotation suppresses coherence and lambda
> - **Bars** (g_bar): Non-axisymmetric torques destroy azimuthal loops
> - **Coherence** (κ): Turbulence/warps/thickness break idealized winding

> **If true**, a monotone, saturating law with these predictors should achieve:
> - **Median APE ≤ 20%**
> - **≥ 80% within ±20% of per-galaxy best**
> - **No per-galaxy tuning**

> **If false**, this falsifies the hypothesis cleanly and indicates either:
> 1. Need richer coherence predictor (pitch angle, arm strength, thickness)
> 2. Need refinement of kernel (stationary-phase weighting, anisotropy)

---

## Timeline (Aggressive but Achievable)

### Week 1: Robust Refitting + Bar Integration
- [ ] Implement V2.2 with bar gating
- [ ] Robust regression with Huber loss + CV
- [ ] Full SPARC evaluation
- [ ] **Target: Median APE ≤ 20%, ≥ 80% within ±20%**

### Week 2: Outlier Triage + Conservative Field
- [ ] Characterize worst 30-40 galaxies
- [ ] Add one surgical fix if justified
- [ ] Implement potential-based kernel
- [ ] Validate MW constraints hold

### Week 3: External Validation + Paper Figures
- [ ] Apply to THINGS/LITTLE THINGS (if data available)
- [ ] Generate all publication figures
- [ ] Draft methods section
- [ ] **Ready for submission**

---

## Code Architecture (Current State)

### Core Files
```
many_path_model/bt_law/
├── bt_laws.py                  # V1 base laws (B/T only)
├── bt_laws_v2.py               # V2 multi-predictor (B/T, Sigma0, S, κ)
├── bt_laws_v2p1.py             # V2.1 enhanced (two-predictor λ, ring concentration)
├── bt_laws_v2p2.py             # V2.2 to implement (+ bar gating)
├── fit_bt_laws_v2p2_robust.py  # To implement (robust CV regression)
├── evaluate_bt_laws_v2p2.py    # To implement (full SPARC eval with bars)
├── extract_bar_classification.py ✅ (done)
├── compute_shear_predictors.py ✅ (done)
├── parse_sparc_disk_params.py  ✅ (done)
└── sparc_*.json                # Data files
```

### Data Files
```
many_path_model/bt_law/
├── sparc_disk_params.json       ✅ (R_d, Sigma0, Hubble types)
├── sparc_shear_predictors.json  ✅ (shear S, compactness)
├── sparc_bar_classification.json ✅ (bar class, g_bar)
└── bt_law_params_v2p*.json      # Fitted gate parameters
```

### Results Files
```
results/
├── bt_law_evaluation_v2/        ✅ (V2 baseline)
├── bt_law_evaluation_v2p1/      ✅ (V2.1 enhanced)
└── bt_law_evaluation_v2p2/      # To generate (with bars)
```

---

## Key Insights from V2.1 Results

1. **Lambda gating works**: High-shear systems now get λ~8-9 kpc, low-shear get λ~10-14 kpc (adaptive!)

2. **Ring concentration matters**: Prevents broad outer overshoot by localizing boost near spiral arms

3. **Intermediate spirals need help**: They scatter more than late/early types → **bar gating is the answer**

4. **Compactness gating validated**: LSB dwarfs (low Sigma0) need amplitude suppression

5. **71.4% within ±20% is good but not enough**: Need to push to 80%+ for publication-grade universal law

---

## References for Methods Section

### Many-Path Framework
- Original toy model: `many_path_model/` (this repo)
- Per-galaxy optimization: `sparc_mega_parallel.py`, `results/mega_test/`

### SPARC Database
- Lelli et al. 2016 (SPARC master table)
- Rotation curves: `data/Rotmod_LTG/*.dat`

### Gaia Validation
- Gaia DR3 vertical kinematics
- Solar neighborhood constraints
- Already implemented in `parameter_optimizer.py`

### Statistical Methods
- Huber robust regression (scipy.optimize)
- Stratified k-fold CV (sklearn)
- Monotone constraints (scipy with bounds)

---

## Contact / Questions

See `V2_RESULTS_SUMMARY.md` for detailed V2 performance analysis.
See diagnostic plots in `results/bt_law_evaluation_v2p1/` for V2.1 improvements.

**Next immediate action**: Implement V2.2 with bar gating, then run robust CV refitting.
