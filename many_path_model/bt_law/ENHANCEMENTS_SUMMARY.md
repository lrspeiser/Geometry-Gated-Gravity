# B/T Law Framework - Enhancements Summary

## ✅ What's Been Implemented

Based on your excellent analysis of the B/T law evaluation results, I've extended the framework to address the key limitations you identified:

---

## 🔧 1. Extended B/T Laws with Disk Strength Predictors

**File:** `bt_laws.py` (updated)

**Enhancement:** `eval_all_laws()` now supports **two-predictor laws**:

```python
def eval_all_laws(B, theta, Sigma0=None, R_d=None):
    """
    Args:
        B: Bulge-to-total fraction [0, 1]
        Sigma0: Disk central surface density (M_sun/pc^2) - for ring_amp scaling
        R_d: Disk scale length (kpc) - for lambda_ring dependence
    """
```

### New Law Forms:

**ring_amp with Sigma0 scaling:**
```
ring_amp(B/T, Σ₀) = [base_law(B/T)] × (Σ₀/Σ_ref)^β
```
- **β > 0**: Higher surface density → stronger spiral winding
- **Fixes:** LSB galaxies at B/T≈0 that need **low** ring_amp

**lambda_ring with R_d dependence:**
```
lambda_ring(B/T, R_d) = base_law(B/T) + α × R_d
```
- **α**: Controls scale-length dependence
- **Fixes:** Coherence length scaling with disk size

### Why This Helps:

Your analysis showed **B/T alone is too coarse** - it can't distinguish:
- High-Σ₀ strong spirals vs. LSB flocculent disks (both B/T ≈ 0)
- Large vs. compact disks (different coherence scales)

The two-predictor laws address exactly these failure modes.

---

## 📊 2. SPARC Disk Parameter Parser

**File:** `parse_sparc_disk_params.py` (new)

**Purpose:** Extract disk structural parameters from SPARC master table

**Extracts:**
- `R_d` - Disk scale length (kpc)
- `Sigma0` - Central surface density (M_sun/pc^2, estimated from μ₀)
- `mu0` - Central surface brightness ([3.6] mag/arcsec²)
- `distance_mpc` - Distance for conversions

**Usage:**
```bash
python many_path_model/bt_law/parse_sparc_disk_params.py \
    --master_file data/SPARC_Lelli2016c.mrt \
    --output many_path_model/bt_law/sparc_disk_params.json
```

**Output:** JSON file mapping galaxy_name → {R_d, Sigma0, mu0, ...}

**Note:** Current Sigma0 estimation is rough (from surface brightness + assumed M/L). Can be refined with:
- Measured disk masses from SPARC
- Photometric decomposition results
- Direct surface density profiles

---

## 🔍 3. Outlier Diagnostic Tool

**File:** `analyze_outliers.py` (new)

**Purpose:** Deep-dive analysis of galaxies with largest ΔAPE

**Generates:**
1. **Rotation curve plots** (observed vs. per-galaxy vs. B/T law)
2. **Parameter comparison bar charts**
3. **Statistics tables**
4. **Summary CSV**

**Usage:**
```bash
python many_path_model/bt_law/analyze_outliers.py \
    --evaluation_results results/bt_law_evaluation/bt_law_evaluation_results.json \
    --output_dir results/bt_law_outliers \
    --top_n 15
```

**Output per galaxy:**
- `outlier_{galaxy}.png` - Comprehensive diagnostic plot
- Shows exactly **where and how** B/T law fails

**Key insights from these plots:**
- Over-predicted ring winding?
- Wrong coherence length?
- Saturation too weak/strong?

---

## 📋 What Still Needs to Be Done

### **Next: 5-Fold Cross-Validation & Robust Fitting**

**Goal:** Refit B/T laws with proper CV to avoid overfitting

**Implementation needed:**
1. Split 175 SPARC galaxies into 5 folds (stratified by morphology)
2. For each fold:
   - Fit laws on 80% (training)
   - Evaluate on 20% (validation)
   - Record median APE
3. Average laws across folds
4. Report per-fold and overall statistics

**Command (target):**
```bash
python many_path_model/bt_law/fit_bt_laws.py \
    --cv 5 \
    --robust 1 \
    --save results/bt_law_cv.json
```

---

### **Then: Fit Two-Predictor Laws**

**Goal:** Add Sigma0 and R_d predictors to reduce LSB outliers

**Implementation needed:**
1. Load disk parameters from `sparc_disk_params.json`
2. Extend fitter to optimize:
   - `ring_beta` (Sigma0 scaling exponent)
   - `lambda_alpha_Rd` (R_d dependence slope)
   - `Sigma_ref` (reference surface density)
3. Use same CV framework
4. Compare APE: **B/T-only vs. B/T+Sigma0+Rd**

**Command (target):**
```bash
python many_path_model/bt_law/fit_bt_laws.py \
    --cv 5 \
    --robust 1 \
    --use_sigma0 1 \
    --use_Rd 1 \
    --save results/bt_law_2predictor.json
```

---

### **Finally: Hold-Out Validation**

**Goal:** Gold-standard generalization test

**Steps:**
1. Randomly select 20 galaxies (stratified by morphology)
2. Refit laws on remaining 155
3. Evaluate on 20 hold-out (no tuning!)
4. Target: **median APE ≤ 20-25%** (vs. ~25% current, ~7.6% per-galaxy)

---

## 🎯 Expected Performance Improvements

### Current (B/T-only):
- **Median APE:** 25.0%
- **Mean APE:** 32.0%
- **Worst outliers:** 50-80% APE (mostly LSB late-types)

### With Sigma0 + R_d (target):
- **Median APE:** **18-22%** (8-12% improvement)
- **Mean APE:** **24-28%**
- **LSB outliers:** Reduce to **30-40% APE** (20-40% improvement)

### Why this should work:
- Your ablation studies show **ring-winding is critical** for flat curves
- LSB galaxies with B/T≈0 currently forced to high ring_amp
- Sigma0 scaling allows **LSB → weak ring_amp**, **HSB → strong ring_amp**
- R_d dependence fixes coherence length mismatches

---

## 📂 Files Updated/Created

### Enhanced Core
- ✅ `bt_laws.py` - Extended to support Sigma0, R_d predictors
- ✅ `parse_sparc_disk_params.py` - Extract disk params from SPARC
- ✅ `analyze_outliers.py` - Diagnostic tool for worst performers

### Still To Do
- ⏳ `fit_bt_laws.py` - Add CV mode and two-predictor fitting
- ⏳ Cross-validation runner
- ⏳ Hold-out validation script

---

## 🚀 Recommended Workflow

### Phase 1: Understand Current Failures (ready now!)
```bash
# 1. Generate outlier diagnostics
python many_path_model/bt_law/analyze_outliers.py --top_n 20

# 2. Parse disk parameters
python many_path_model/bt_law/parse_sparc_disk_params.py

# 3. Review plots in results/bt_law_outliers/
# Look for patterns: Are LSB galaxies consistently over-predicted?
```

### Phase 2: Fit Enhanced Laws (needs implementation)
```bash
# 4. CV refit with robust loss (B/T only, baseline)
python many_path_model/bt_law/fit_bt_laws.py --cv 5 --robust 1

# 5. Add Sigma0 + R_d predictors
python many_path_model/bt_law/fit_bt_laws.py --cv 5 --robust 1 \
    --use_sigma0 1 --use_Rd 1

# 6. Evaluate both on full SPARC
python many_path_model/bt_law/evaluate_bt_laws_sparc.py \
    --bt_params results/bt_law_2predictor.json \
    --output_dir results/bt_law_2pred_evaluation
```

### Phase 3: Validate
```bash
# 7. Hold-out test (20 galaxies)
python many_path_model/bt_law/holdout_validation.py \
    --n_holdout 20 \
    --seed 42
```

---

## 💡 Key Insights from Your Analysis

1. **Speed is expected** - Forward evaluation with laws is instant (no optimization)

2. **B/T alone is insufficient** - Can't distinguish strong vs. weak spirals at same B/T

3. **Late-type LSB galaxies are the problem** - Need disk strength predictor

4. **Early-types mostly OK** - Bulge-gating works as expected

5. **Ring winding is critical** - Getting ring_amp right is highest leverage

---

## 📊 Current vs. Target Performance

| Metric | Current (B/T only) | Target (B/T+Σ₀+R_d) | Per-Galaxy Best |
|--------|-------------------|---------------------|-----------------|
| **Median APE** | 25.0% | **18-22%** | 7.6% |
| **Mean APE** | 32.0% | **24-28%** | 12.5% |
| **Excellent (<10%)** | ~15% | **25-30%** | ~60% |
| **Poor (≥30%)** | ~40% | **<20%** | ~5% |

**Gap closure:** From 17.4% gap (25.0 - 7.6) to **10-14% gap** (18-22 vs. 7.6)

**Interpretation:** Two-predictor laws should close **40-50%** of the gap to per-galaxy performance while remaining **fully universal** (no galaxy-specific tuning).

---

## 🔬 Scientific Validation Path

1. ✅ **B/T-only laws** (done) → Establishes baseline, confirms bulge-gating

2. **B/T + Sigma0 + R_d** (next) → Tests if disk strength explains residual variance

3. **Cross-validation** → Confirms no overfitting, generalizes to new galaxies

4. **Hold-out test** → Gold-standard performance metric

5. **Residual analysis** → After step 2, check if remaining errors correlate with:
   - Bar strength
   - Asymmetry
   - Pitch angle
   - Environment

If step 5 shows **random residuals** → morphology + disk strength explains variance!

If **systematic trends remain** → need tertiary predictors or accept per-galaxy variance

---

## 📞 Quick Commands

**Run outlier analysis now:**
```bash
python many_path_model/bt_law/analyze_outliers.py --top_n 20
```

**Parse disk parameters:**
```bash
python many_path_model/bt_law/parse_sparc_disk_params.py
```

**Check what disk data is available:**
```bash
python many_path_model/bt_law/parse_sparc_disk_params.py | grep "R_d available"
```

---

## 🎓 Bottom Line

You correctly diagnosed the failure mode: **B/T alone can't distinguish disk strength**.

The framework is now **ready to test** whether adding Sigma0 and R_d closes the gap.

**Next concrete step:** Implement CV mode in `fit_bt_laws.py` and fit the two-predictor model.

**Expected outcome:** Median APE drops from **25%** to **~20%**, with LSB outliers improving most dramatically.

This will tell you if **morphology + disk structure** is sufficient or if you need per-galaxy tuning / secondary effects (bars, environment, etc.).

---

**Status:** Framework enhanced, diagnostic tools ready, awaiting CV implementation! 🚀
