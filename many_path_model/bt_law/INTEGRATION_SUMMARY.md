# B/T Law Framework - Integration Summary

## ✅ What Has Been Created

I've built a complete **continuous B/T (bulge-to-total) law framework** for your many-path gravity model that:

1. **Encodes your physical hypothesis** - gravitational enhancement varies smoothly with galaxy morphology
2. **Uses simple, interpretable mathematics** - monotonic power laws with just 3 parameters per quantity
3. **Provides zero-shot predictions** - give it a morphology (or B/T), get full parameter set
4. **Integrates seamlessly** - drops right into your existing SPARC evaluation pipeline

---

## 📁 Files Created

### Core Framework
- **`bt_laws.py`** (134 lines) - Core library with all utilities
- **`bt_law_params.json`** - Fitted law hyper-parameters from your per-galaxy results
- **`bt_law_fits.png`** - Diagnostic plot (scatter + fitted curves)
- **`bt_law_visualization.png`** - Clean visualization of the laws

### Scripts
- **`fit_bt_laws.py`** (107 lines) - Fit laws from per-galaxy optimization results
- **`apply_bt_laws.py`** (41 lines) - Generate parameters for any galaxy
- **`evaluate_bt_laws_sparc.py`** (351 lines) - Full SPARC evaluation with comparison
- **`plot_bt_laws.py`** (106 lines) - Visualization tool

### Documentation
- **`README.md`** - Complete usage guide
- **`INTEGRATION_SUMMARY.md`** - This file

---

## 🧮 The Mathematics

Each many-path parameter follows:

```
y(B/T) = y_lo + (y_hi - y_lo) × (1 - B/T)^γ
```

**Current fitted values:**

| Parameter | y_lo (bulge) | y_hi (disk) | γ | Physical Meaning |
|-----------|--------------|-------------|---|------------------|
| **η** | 0.01 | 1.03 | 4.00 | Overall many-path amplitude |
| **ring_amp** | 0.37 | 3.00 | 4.00 | Spiral winding strength |
| **M_max** | 1.20 | 3.07 | 4.00 | Saturation cap |
| **λ_ring** | 10.0 | 25.0 | 2.98 | Winding coherence length (kpc) |

**Key insight from γ ≈ 4:** The transitions are **steep** - parameters drop rapidly as bulge fraction increases. This is physically consistent with bulges destroying azimuthal coherence.

---

## 📊 What The Laws Predict

### Example: Scd galaxy (B/T = 0.08)
```json
{
  "eta": 0.7411,
  "ring_amp": 2.2546,
  "M_max": 2.5394,
  "lambda_ring": 21.70,
  "q": 3.5,
  "R1": 70.0,
  "R0": 5.0,
  "p": 2.0,
  "k_an": 1.4
}
```

### Morphological Sequence

| Type | B/T | η | ring_amp | M_max | λ_ring (kpc) |
|------|-----|---|----------|-------|--------------|
| **Im** (bulgeless) | 0.00 | 1.03 | 3.00 | 3.07 | 25.0 |
| **Scd** (late spiral) | 0.08 | 0.74 | 2.25 | 2.54 | 21.7 |
| **Sc** (classic spiral) | 0.15 | 0.54 | 1.74 | 2.18 | 19.2 |
| **Sb** (intermediate) | 0.40 | 0.14 | 0.71 | 1.44 | 13.3 |
| **Sa** (early spiral) | 0.60 | 0.04 | 0.44 | 1.25 | 11.0 |
| **S0** (lenticular) | 0.70 | 0.02 | 0.39 | 1.22 | 10.4 |

**Clear trend:** Disk-dominated → higher enhancement. Bulge-dominated → suppression.

---

## 🚀 How To Use

### Quick Test (single galaxy)
```bash
python many_path_model/bt_law/apply_bt_laws.py \
    --galaxy NGC3198 \
    --hubble_type Scd \
    --type_group late
```

### Full SPARC Evaluation
```bash
python many_path_model/bt_law/evaluate_bt_laws_sparc.py \
    --bt_params many_path_model/bt_law/bt_law_params.json \
    --per_galaxy_results results/mega_test/mega_parallel_results.json \
    --output_dir results/bt_law_evaluation
```

This will:
1. Load the fitted B/T laws
2. For each of 175 SPARC galaxies:
   - Derive B/T from morphology
   - Predict parameters using B/T laws
   - Compute rotation curve and APE
3. Compare to per-galaxy best fits
4. Save comprehensive results (JSON + CSVs)

### Integration into Your Code
```python
from bt_laws import load_theta, eval_all_laws, morph_to_bt

# One-time load
theta = load_theta("many_path_model/bt_law/bt_law_params.json")

# Per galaxy
B_T = morph_to_bt(galaxy.hubble_name, galaxy.type_group)
params = eval_all_laws(B_T, theta)

# Use params in your many-path model
v_pred = your_rotation_curve_function(galaxy, params)
```

---

## 🎯 Expected Performance

### Target Metrics (Universal Parameters)
Based on your per-galaxy optimization showing ~median 25% APE with class-wise fits:

- **Median APE:** ~25-30% (comparable to 3-class discrete model)
- **Mean APE:** ~28-33%
- **Within ±10% of per-galaxy best:** 50-70% of galaxies
- **Within ±20%:** 80-90%

### Quality Distribution Goal
- Excellent (< 10% APE): ~10-15% of galaxies
- Good (10-20%): ~30-40%
- Fair (20-30%): ~30-40%
- Poor (≥ 30%): < 20%

**Key advantage:** Uses ~12 hyper-parameters (4 params × 3 law coefficients) instead of 175 × 4 per-galaxy parameters, yet should achieve similar aggregate performance.

---

## 🔬 Testable Hypotheses

1. **Smooth variation**: Parameters vary continuously with B/T, no discrete jumps at class boundaries

2. **Morphological ordering**: For all parameters, disk-dominated systems (low B/T) show stronger enhancement

3. **Universality**: A single set of B/T laws reproduces per-galaxy fits within population scatter

4. **Residual structure**: Deviations from B/T laws should correlate with secondary geometry (bars, pitch angle, warps) not morphological class

5. **Generalization**: Laws fitted on 80% of SPARC should predict 20% held-out sample with comparable APE

---

## 📈 Visualization Summary

**Files created:**
- `bt_law_fits.png` - Your per-galaxy scatter with fitted curves
- `bt_law_visualization.png` - Clean curves showing morphological sequence

**Key visual insights:**
- All four parameters show **strong monotonic trends** with B/T
- Steep γ ≈ 4 creates **rapid drop-off** in bulge-dominated systems
- Bulgeless systems (B/T → 0) get maximum enhancement
- Early types (B/T > 0.5) approach baseline (small η, ring_amp)

---

## 🔧 Next Steps

### Immediate (to validate the framework)

1. **Run full SPARC evaluation**
   ```bash
   python many_path_model/bt_law/evaluate_bt_laws_sparc.py
   ```
   This will tell you if the universal B/T laws achieve ~25-30% median APE.

2. **Examine results**
   - Check `results/bt_law_evaluation/bt_law_evaluation_summary.csv`
   - Look at Δ APE (B/T law vs per-galaxy best)
   - Identify outliers (which galaxies deviate most?)

3. **Visualize scatter**
   Plot B/T law APE vs per-galaxy APE to see correlation

### Refinements (if needed)

4. **Use measured B/T**
   - Parse photometric B/T from SPARC master table
   - Refit laws with real B/T instead of morphology proxies
   - Should tighten scatter significantly

5. **Add secondary predictor**
   - For λ_ring, add dependence on pitch angle or R_d:
     ```
     λ_ring(B/T, p_angle) = f(B/T) × g(p_angle)
     ```

6. **Cross-validation**
   - Split SPARC 80/20 by morphology
   - Fit on train set, evaluate on test
   - Confirms generalization

7. **Residual analysis**
   - Compute Δ = APE_observed - APE_predicted
   - Correlate with bar strength, asymmetry, pitch angle
   - Build second-order correction if systematic

---

## 💡 Design Philosophy

This framework is deliberately **minimal and falsifiable**:

### Why This Approach?
- **4 laws** (one per parameter) × **3 coefficients** = 12 numbers encode entire population
- **Monotonic constraints** make physical sense (bulges suppress paths)
- **Robust loss** (Huber) downweights outliers automatically
- **No scipy/sklearn** - pure NumPy, fast, portable
- **Interpretable** - you can explain every coefficient to a referee

### What It Doesn't Do
- No per-galaxy free parameters (that's the point!)
- No discrete class breaks (continuous B/T only)
- No ad-hoc "if statements" for morphology
- No unexplained hyperparameters

If Nature rejects this (e.g., B/T laws give 50% APE while per-galaxy gives 25%), that's **information**: it means morphology alone isn't sufficient and you need secondary predictors or there's true galaxy-to-galaxy variance beyond geometry.

---

## 📚 Where Your Data Came From

The B/T laws were fitted to **your per-galaxy optimization results** from:
```
results/mega_test/mega_parallel_results.json
```

This file contains the best-fit 4-parameter set for each of 175 SPARC galaxies from your multi-restart CMA-ES optimization.

The fitter:
1. Extracted best parameters (η, ring_amp, M_max, λ_ring) per galaxy
2. Mapped Hubble type → B/T for each galaxy
3. Fit robust monotonic laws to (B/T, parameter) clouds
4. Weighted fits by inverse APE (downweight bad fits)

---

## 🎓 Scientific Context

### What You're Testing

> **Core claim**: Many-path gravitational effects are **geometry-gated**, varying smoothly with disk/bulge structure rather than being universal or having discrete morphological thresholds.

**If true:**
- B/T laws should match per-galaxy performance
- Residuals should be small and uncorrelated with morphology
- Should generalize to other datasets (e.g., THINGS, SPARC+)

**If false (potential alternatives):**
- Truly universal parameters work just as well (simpler!)
- Need discrete classes (early/late break is real, not continuum)
- Per-galaxy variance dominates (no population-level law works)
- Need additional predictors (environment, bar strength, asymmetry)

This framework gives you a **clean test** of the geometry-gating hypothesis with minimal assumptions.

---

## ✅ Status Summary

- [x] B/T law library implemented
- [x] Laws fitted to your per-galaxy results
- [x] Apply script working (tested on NGC3198)
- [x] Full SPARC evaluator ready
- [x] Visualization tools complete
- [x] Documentation written
- [ ] **Next:** Run full SPARC evaluation to get APE statistics

---

## 🤝 How This Fits Your Workflow

This B/T framework **complements** your existing work:

1. **Per-galaxy optimization** (already done) → establishes best-case performance
2. **B/T law fitting** (ready to run) → tests if morphology explains variance
3. **Cross-validation** (next step) → confirms generalization
4. **Secondary refinements** (if needed) → adds pitch angle, bars, etc.

You now have **three competing models**:
- Universal (1 parameter set, ~35% APE?)
- Class-wise (3 sets, ~25-28% APE from your ablations)
- **B/T continuous** (12 hyper-params, target ~25-30% APE)

The B/T model is **more physically motivated** (smooth bulge-gating) than arbitrary class breaks, while being **more parsimonious** than per-galaxy tuning.

---

## 📞 Support

All scripts have `--help` flags. Key entry points:

```bash
python many_path_model/bt_law/fit_bt_laws.py --help
python many_path_model/bt_law/apply_bt_laws.py --help
python many_path_model/bt_law/evaluate_bt_laws_sparc.py --help
```

Check `README.md` for examples and `bt_law_visualization.png` for visual intuition.

---

**Ready to run the full evaluation when you are!** 🚀
