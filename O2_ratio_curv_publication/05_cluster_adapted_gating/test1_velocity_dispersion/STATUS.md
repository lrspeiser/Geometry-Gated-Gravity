# Test 1: Velocity Dispersion Gating - STATUS

**Last Updated:** October 2, 2025, 18:30  
**Progress:** Step 1/4 Complete ✅ (25% done)

---

## 🎯 Overall Status

**Goal:** Test if velocity dispersion gating can close the cluster lensing gap while preserving galaxy fits.

**Progress:**
- ✅ **Step 1:** Data preparation (COMPLETE)
- ⏳ **Step 2:** Parameter fitting (NEXT)
- ⏳ **Step 3:** SPARC validation
- ⏳ **Step 4:** Diagnostic plots

---

## ✅ Step 1 Complete: Cluster Data Extraction

### Results (`cluster_sigma_data.csv`):

| Cluster | z | θ_E obs (") | σ median (km/s) | kT (keV) | Lensing? |
|---------|---|-------------|-----------------|----------|----------|
| ABELL_1689 | 0.183 | 47 | 1263 | 9.83 | ✅ |
| A2029 | 0.077 | 28 | 1132 | 7.90 | ✅ |
| A478 | 0.088 | 31 | 1047 | 6.75 | ✅ |
| ABELL_0426 | 0.018 | - | 884 | 4.82 | ❌ (no lensing) |

### Key Findings:

**✅ Perfect Separation:**
- **Clusters:** σ = 1047-1263 km/s (median 1132 km/s)
- **Galaxies:** σ = 29-268 km/s (median ~100 km/s)
- **Factor:** 8-10× difference

**This is exactly what we need!** The hypothesis predicts:
- With α ≈ 1.5: (1132/100)^1.5 = **38× amplification** for median cluster
- With α ≈ 1.5: (100/100)^1.5 = **1× amplification** for galaxies (no change)

### Galaxy Estimates for Validation:

| Type | M_200 (Msun) | σ (km/s) |
|------|--------------|----------|
| Dwarf | 1e10 | 29 |
| Small spiral | 5e10 | 52 |
| MW-like | 1e12 | 147 |
| Massive | 5e12 | 268 |

**Recommended for SPARC:** σ = 120 km/s (typical)  
**Sensitivity test:** σ = 80, 100, 150, 200 km/s

---

## ⏳ Step 2 Next: Parameter Fitting

### What to fit:

**Fixed parameters (from O2):**
- a = 0.6687
- b = 0.1401
- d = 0.0871

**NEW parameters to optimize:**
- **e:** Velocity dispersion gate weight
- **α:** Power law index

**Optimization target:**
Minimize |θ_E_predicted - θ_E_observed| for 3 lensing clusters

### Constraints:

1. **Stability:** e must keep denominator positive
   - denom = a - b·Σ̂ - d·|∇ln Σ| - e·(σ/σ₀)^α > 0
   - For σ = 1263 km/s, α = 1.5: e < ~0.05 to be safe

2. **Physical:** 0.5 < α < 3.0 (power law index)

3. **Galaxy preservation:** Median APE on SPARC must stay < 0.30

### Expected parameter ranges:

- **e:** 0.01 - 0.10 (gate weight)
- **α:** 1.0 - 2.0 (power index)

If e ≈ 0.05, α ≈ 1.5:
- Galaxy (σ=120): (120/100)^1.5 = 1.32× → **7% change** ✅ acceptable
- Cluster (σ=1132): (1132/100)^1.5 = 38× → **huge boost** ✅ what we need

---

## 📋 Next Steps (Detailed)

### Step 2a: Simple Einstein Radius Calculator (needed first)

Before fitting, need code to predict θ_E from model. This requires:

1. Load cluster gas/stars profiles
2. Compute geometry features (x, Σ̂, ∇ln Σ)
3. Compute fX with σ-gating
4. Project to 2D surface density
5. Compute convergence κ(R)
6. Find Einstein radius where κ̄(<R) = 1.0

**File to create:** `einstein_radius_calculator.py`  
**Time:** 1 hour

### Step 2b: Parameter Fitting

Optimize (e, α) using scipy.optimize.minimize:

```python
def loss(params_new):
    e, alpha = params_new
    full_params = (a_fixed, b_fixed, d_fixed, e, alpha)
    
    total_loss = 0
    for cluster in [ABELL_1689, A2029, A478]:
        theta_E_pred = predict_einstein_radius(cluster, full_params)
        theta_E_obs = cluster['theta_E_obs']
        loss = (theta_E_pred - theta_E_obs)**2
        total_loss += loss
    
    return total_loss
```

**File to create:** `fit_sigma_model.py`  
**Time:** 1-2 hours (includes optimization)

### Step 2c: Validation on SPARC

Test fitted (e, α) on 120 SPARC galaxies with σ = 120 km/s:

```python
# Compute rotation curves with sigma-gated model
for galaxy in sparc_galaxies:
    V_mod_sigma = compute_v_circ_with_sigma(
        galaxy, params=(a,b,d,e,alpha), sigma=120.0
    )
    ape = median_ape(V_mod_sigma, galaxy['V_obs'])
    
galaxy_median_ape = median(all_apes)
print(f"Galaxy APE with σ-gating: {galaxy_median_ape:.3f}")
print(f"Baseline APE: 0.242")
print(f"Degradation: {(galaxy_median_ape - 0.242)/0.242 * 100:.1f}%")
```

**Success:** galaxy_median_ape < 0.30 (max 24% degradation)  
**File to create:** `validate_on_sparc.py`  
**Time:** 1 hour

---

## 🎯 Decision Criteria

After Step 2-3 complete, evaluate results:

### ✅ PASS (Test 1 succeeds):
- All 3 clusters: |θ_E_pred - θ_E_obs| / θ_E_obs < 0.30
- SPARC galaxies: median APE < 0.30
- Parameters physical: 0.01 < e < 0.2, 0.5 < α < 3.0

→ **Publish 5-parameter model immediately!**  
→ **Major breakthrough - geometry gating works at all scales**

### ⚠️ PARTIAL (Two-regime model):
- Clusters fit well
- BUT galaxies degrade (APE > 0.30)

→ **Need separate parameters for galaxies vs clusters**  
→ **Still publishable as two-regime model**

### ❌ FAIL (Test 1 fails):
- Cannot fit clusters within factor 2
- OR requires unphysical parameters (e > 0.5, α > 5)

→ **Move to Test 2: Hot Gas Fraction Gating**  
→ **Document negative result**

---

## 📊 Files Created So Far

```
test1_velocity_dispersion/
├── velocity_dispersion_model.py  (269 lines) ✅
├── prepare_cluster_data.py       (259 lines) ✅
├── cluster_sigma_data.csv        (4 clusters) ✅
├── README.md                      (217 lines) ✅
├── STATUS.md                      (this file) ✅
├── einstein_radius_calculator.py              ⏳ NEXT
├── fit_sigma_model.py                         ⏳
├── validate_on_sparc.py                       ⏳
└── generate_diagnostics.py                    ⏳
```

---

## ⏱️ Time Estimate

**Completed:** 2 hours  
**Remaining:**
- Einstein radius calculator: 1 hour
- Parameter fitting: 1-2 hours
- SPARC validation: 1 hour
- Diagnostic plots: 1 hour

**Total remaining:** 4-5 hours

---

## 💡 Early Prediction

Based on the data:

**Optimistic scenario:**
- Clusters are 10× higher σ → Should provide massive amplification
- If α ≈ 1.5, e ≈ 0.03-0.05 might work
- Galaxy impact would be minimal (σ=120 vs σ=100 is small)

**Realistic concern:**
- Even with 38× amplification, current underprediction is 40-140×
- May still fall short by factor of 2-4
- But getting within factor 2 would be huge progress!

**Key test:** Does denominator stay positive for all systems?
- Need: a - b·Σ̂ - d·|∇ln Σ| - e·(σ/100)^α > 0
- Critical at cluster outskirts where Σ̂ is most negative

---

## 📞 Next Session Plan

1. Create `einstein_radius_calculator.py` (1 hour)
2. Create `fit_sigma_model.py` (1-2 hours)
3. Run optimization → get (e, α) values
4. If successful: validate on SPARC
5. If fails: document and move to Test 2

---

**Status:** On track, data looks promising! 🚀  
**Next:** Build Einstein radius calculator
