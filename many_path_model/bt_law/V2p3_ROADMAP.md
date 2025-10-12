# V2.3 B/T Law Development Roadmap

## Overview

V2.3 extends V2.2 with **radius-dependent suppression** to address systematic overshoot in Sc/Sbc/SB galaxies. The key insight: bars and shear dephase long azimuthal paths *beyond specific radii*, not globally.

---

## V2.2 Performance Summary

**Full SPARC Sample (175 galaxies):**
- Mean APE: 28.74%
- Median APE: 23.07%
- Excellent/Good (< 20%): 41.1%
- Within ±20% of per-galaxy best: 68.0%

**By Morphology:**
| Type | Mean APE | Median APE | Count |
|------|----------|------------|-------|
| Early | 22.42% | 16.97% | 28 |
| Intermediate | 35.04% | 29.56% | 34 |
| Late | 28.42% | 22.44% | 113 |

**By Bar Class:**
| Bar Class | Mean APE | Median APE | Count |
|-----------|----------|------------|-------|
| SA (unbarred) | 27.52% | 25.76% | 3 |
| S (default) | 28.71% | 22.78% | 127 |
| SAB (weak bar) | **17.74%** | 12.92% | 10 |
| SB (strong bar) | 30.58% | 26.15% | 30 |

### What Worked in V2.2
✅ **Shear-dependent λ** - shortens azimuthal coherence in high-shear disks  
✅ **Radial ring concentration** - centers ring term at morphology-dependent R_ring  
✅ **Bar gating (global)** - SAB galaxies perform excellently (17.74% mean APE)  
✅ **Early types** - strong performance (median 17%) validates B/T scaling

### What Needs Improvement
❌ **Intermediate/late spirals** - overshoot beyond spiral region (6-12 kpc)  
❌ **SB galaxies** - global suppression insufficient (30.58% mean APE)  
❌ **Broad red plateaus** - too much ring power at intermediate radii  
❌ **High-shear disks** - need both shorter λ AND narrower σ_ring

---

## V2.3 Key Improvements

### 1. **Radius-Dependent Bar Taper** (Addresses SB/SAB Overshoot)

**Mathematical Form:**
```
bar_taper(R) = [1 + tanh((R_bar - R) / w_bar)]^γ_bar
```

**Behavior:**
- **Inside bar** (R << R_bar): taper ≈ 2^γ_bar (allow ring)
- **Outside bar** (R >> R_bar): taper → 0 (suppress ring)

**Parameters:**
- `R_bar_factor = 2.0` → R_bar = 2.0 × R_d (≈ bar corotation)
- `w_bar_factor = 0.3` → smooth transition width
- `gamma_bar_taper = 1.5` → suppression steepness

**Applied to:** SB and SAB galaxies only

**Physical Rationale:**  
Bars dephase azimuthal orbits beyond their corotation radius (~2 R_d). The global bar gate (g_bar) says *how much* suppression; the radial taper says *where* it happens.

---

### 2. **Radius-Dependent Shear Taper** (Addresses Sc/Sbc Overshoot)

**Mathematical Form:**
```
shear_taper(R) = [1 + tanh((R_shear - R) / w_shear)]^γ_shear
```

**Behavior:**
- **Inside spiral region** (R << R_shear): taper ≈ 2^γ_shear (allow ring)
- **Beyond spirals** (R >> R_shear): taper → 0 (suppress ring)

**Parameters:**
- `R_shear_factor = 2.5` → R_shear = 2.5 × R_d
- `w_shear_factor = 0.5` → transition width
- `gamma_shear_taper = 1.0` → suppression steepness
- `shear_threshold = 0.5` → only applied if shear > 0.5

**Applied to:** High-shear galaxies (shear > 0.5)

**Physical Rationale:**  
Strong spiral arms create azimuthal coherence in the spiral-dominated region (inner disk), but destroy it in the outer disk where shear-induced phase mixing dominates.

---

### 3. **Shear-Coupled σ_ring** (Narrows Ring Envelope)

**Mathematical Form:**
```
σ_ring(S) = σ_min + (σ_max - σ_min) × [1 - g_shear(S)]^γ_σ
```

**Parameters:**
- `sigma_ring_min = 0.3` R_d (narrow envelope for high shear)
- `sigma_ring_max = 0.8` R_d (broad envelope for low shear)
- `gamma_sigma_shear = 1.0`

**Physical Rationale:**  
High-shear disks (Sc/Sbc) have both **shorter** λ (from V2.2) AND **narrower** radial concentration. This prevents broad red plateaus at 6-12 kpc.

---

### 4. **Dynamic M_max(S, B/T)** (Prevents Red Roofs)

**Mathematical Form:**
```
M_max = M_min + (M_max_disk - M_min) × (1 - B/T)^γ_B × (1 - g_shear(S))^γ_S
```

**Parameters:**
- `M_min = 1.2` (floor for bulge-dominated systems)
- `M_max_disk = 3.0` (ceiling for pure disks with low shear)
- `gamma_bulge = 4.0` (B/T dependence)
- `gamma_shear = 2.0` (shear dependence - **NEW in V2.3**)

**Physical Rationale:**  
High-shear disks shouldn't reach the same M_max as quiescent disks. Chaotic velocity fields reduce the net many-path boost.

---

## Implementation Status

### ✅ Completed
- [x] V2.3 law functions (`bt_laws_v2p3.py`)
- [x] Radius-dependent bar/shear taper functions
- [x] Shear-coupled σ_ring law
- [x] Dynamic M_max(S, B/T) law
- [x] Initial parameter file (`bt_law_params_v2p3_initial.json`)
- [x] Test cases demonstrating taper behavior

### 🔄 Next Steps

#### 1. **Create Forward Model with Radial Tapers** (PRIORITY)
Modify the APE computation to apply `bar_taper(r)` and `shear_taper(r)` to the ring term:

```python
# In compute_galaxy_ape_v2p3():
ring_term_base = ring_amp * (ex / (1.0 - kappa * ex))
ring_term_base = ring_term_base * envelope  # Gaussian radial envelope

# NEW V2.3: Apply radial tapers
if R_bar is not None:
    bar_taper = bar_radial_taper(r, R_bar, w_bar, gamma_bar_taper)
    ring_term_base = ring_term_base * bar_taper

if R_shear is not None:
    shear_taper = shear_radial_taper(r, R_shear, w_shear, gamma_shear_taper)
    ring_term_base = ring_term_base * shear_taper

ring_term = ring_term_base * bulge_gate  # Bulge suppression
```

#### 2. **Full SPARC Evaluation Script** (`evaluate_bt_laws_v2p3.py`)
- Copy `evaluate_bt_laws_v2p2.py` → `evaluate_bt_laws_v2p3.py`
- Update imports to use `bt_laws_v2p3`
- Integrate radial tapers into APE computation
- Run on all 175 galaxies

#### 3. **Diagnostic Comparison**
Compare V2.3 vs. V2.2 performance:
- Overall: mean/median APE, quality distribution
- By morphology: early/intermediate/late
- By bar class: SA/SAB/SB/S
- **Target improvement zones:**
  - SB galaxies: 30.58% → target < 25%
  - Intermediate spirals: 35.04% → target < 30%
  - Maintain early type performance: 22.42% (or improve)

#### 4. **Failure Mode Analysis**
Identify 10-15 worst galaxies and diagnose:
- Is overshoot beyond R_bar/R_shear?
- Is σ_ring too broad?
- Is M_max too high?
- Does per-galaxy fit suggest different taper parameters?

#### 5. **Hyperparameter Refinement** (if needed)
If initial parameters don't improve enough, run targeted optimization:
- **Fixed:** Base B/T laws (η, ring_amp, λ, from V2.2)
- **Tune:** 
  - `R_bar_factor`, `w_bar_factor`, `gamma_bar_taper`
  - `R_shear_factor`, `w_shear_factor`, `gamma_shear_taper`
  - `sigma_ring_min`, `sigma_ring_max`, `gamma_sigma_shear`
  - `gamma_shear` in M_max law

Use:
- **Training:** 80% stratified by morphology + bar class
- **Validation:** 20% held-out
- **Loss:** Huber loss on APE (robust to outliers)
- **Constraints:** Monotone gates, physically reasonable ranges

---

## Expected Performance Gains

### Conservative Targets (V2.3 vs. V2.2)
- **Overall median APE:** 23.07% → **20-21%**
- **Excellent/Good fraction:** 41.1% → **50-55%**
- **SB galaxies:** 30.58% → **24-26%**
- **Intermediate spirals:** 35.04% → **28-30%**

### Optimistic Targets (with hyperparameter tuning)
- **Overall median APE:** 23.07% → **18-20%**
- **Excellent/Good fraction:** 41.1% → **60-65%**
- **SB galaxies:** 30.58% → **22-24%**
- **Within ±20% of per-galaxy best:** 68.0% → **75-80%**

---

## Physical Interpretation

### What V2.3 Models
1. **Azimuthal coherence exists in specific radial zones**
   - Inside bars: partial coherence
   - In spiral regions: strong coherence
   - Beyond spiral dominance: rapid decoherence

2. **Morphology determines WHERE coherence survives**
   - Bars → R < 2 R_d
   - Strong spirals → R < 2.5 R_d
   - Bulges → suppress everywhere

3. **Multiple dephasing agents interact**
   - Global: bar class (SA/SAB/SB)
   - Radial: bar/shear tapers
   - Shape: σ_ring width

### Connection to Many-Path Hypothesis
The ring term represents **long azimuthal path families** that bypass dark matter halos. V2.3 explicitly models where these families:
- **Form** (galactic scale, R > 0.5 kpc)
- **Survive** (spiral region, R < R_shear)
- **Decohere** (beyond bars, R > R_bar)

This is geometrically motivated: bars and shear are real dynamical agents that phase-mix orbits.

---

## Paper Updates Required

### Abstract/Introduction
- Note V2.2 performance (median 23.1%, 41% excellent/good)
- Highlight SAB success (17.74% mean) validates bar gating concept
- Motivate radius-dependent extension for SB/Sc failures

### Methods
- Add V2.3 equations (bar_taper, shear_taper, σ_ring(S), M_max(S, B/T))
- Explain physical motivation (bars/shear dephase beyond specific R)
- Show parameter table with V2.2 → V2.3 changes highlighted

### Results
- V2.2 vs. V2.3 comparison table
- Performance by morphology and bar class
- Ablation: show that each taper adds measurable value
  - V2.2 baseline
  - V2.2 + bar taper only
  - V2.2 + shear taper only
  - V2.2 + σ_ring coupling only
  - V2.3 full (all three)

### Discussion
- Connect to bar/spiral dynamics literature
- Explain why radial tapers improve interpretability
- Note that coherence zones depend on observable quantities (R_d, shear, bar class)

### Failure Modes Section (NEW)
- Include 3-4 example galaxies with V2.2 overshoot
- Show how V2.3 tapers fix them
- Discuss remaining outliers (warp candidates, data quality issues, etc.)

---

## Code Organization

```
many_path_model/bt_law/
├── bt_laws.py                          # V1.0 baseline (B/T only)
├── bt_laws_v2p1.py                     # V2.1 (ring concentration)
├── bt_laws_v2p2.py                     # V2.2 (shear + bar gates)
├── bt_laws_v2p3.py                     # V2.3 (radius-dependent tapers) ← NEW
├── bt_law_params_v2p3_initial.json     # Initial V2.3 parameters ← NEW
├── evaluate_bt_laws_v2p3.py            # Full SPARC evaluation ← TODO
├── V2p3_ROADMAP.md                     # This file ← NEW
└── results/
    ├── bt_law_evaluation_v2p2/         # V2.2 baseline results
    └── bt_law_evaluation_v2p3/         # V2.3 results ← TODO
```

---

## Timeline

**Phase 1: Initial Evaluation** (Today)
- [x] Implement V2.3 laws
- [ ] Create evaluation script
- [ ] Run on full SPARC sample
- [ ] Compare to V2.2 baseline

**Phase 2: Diagnostics** (This week)
- [ ] Identify improved/degraded galaxies
- [ ] Analyze failure modes
- [ ] Determine if hyperparameter tuning needed

**Phase 3: Optimization** (If needed)
- [ ] Set up training/validation split
- [ ] Run grid search or Bayesian optimization
- [ ] Validate on held-out set
- [ ] Document improvements

**Phase 4: Paper Integration** (Next week)
- [ ] Update equations and figures
- [ ] Add ablation studies
- [ ] Write failure modes section
- [ ] Finalize V2.3 as publishable universal law

---

## Questions to Answer with V2.3 Evaluation

1. **Does the radial bar taper improve SB galaxies?**
   - Target: 30.58% → < 25% mean APE
   - Check: overshoot location (should shift inward)

2. **Does the shear taper fix broad Sc/Sbc overshoot?**
   - Look at high-shear intermediate spirals
   - Should reduce APE in 6-12 kpc region

3. **Does shear-coupled σ_ring narrow broad plateaus?**
   - Compare σ_ring in V2.2 (fixed 0.5 R_d) vs. V2.3 (0.3-0.8 R_d)

4. **Does dynamic M_max prevent red roofs?**
   - Check high-shear + high-B/T galaxies
   - Should reduce excessive plateaus at large R

5. **Do all three changes work together coherently?**
   - Run ablation: test each change independently
   - Verify they don't interfere with each other

---

## Success Criteria

V2.3 is ready for publication if:
- ✅ Median APE < 21% (vs. 23.1% in V2.2)
- ✅ Excellent/Good fraction > 50% (vs. 41.1%)
- ✅ SB galaxies < 26% mean APE (vs. 30.58%)
- ✅ Ablation shows each component adds value
- ✅ Failure modes are well-understood (not random scatter)
- ✅ Physical interpretation is clear and testable

If not achieved with initial parameters, proceed to Phase 3 (optimization).

---

## Contact / Collaboration Notes

This roadmap reflects the user's analysis that V2.2 is moving in the right direction but needs radial selectivity to handle bars/spirals properly. The core insight—*global gates say HOW MUCH, radial tapers say WHERE*—is geometrically motivated and aligns with the many-path hypothesis.

Next step: create the evaluation script and run it on the full sample.
