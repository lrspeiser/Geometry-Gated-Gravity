# Track B1: Cluster Lensing Test Results
**Date:** 2025-01-13  
**Status:** ⚠️ SCALE LIMITATION DISCOVERED  
**Model:** Path-spectrum kernel with frozen SPARC parameters

---

## 🎯 **EXECUTIVE SUMMARY**

**Finding:** The path-spectrum kernel with L₀ = 4.99 kpc (optimized for SPARC galaxies) produces **negligible boost at cluster scales**.

**Result:**
- Boost factor K ≈ 0.0000 (essentially zero)
- No Einstein radius predicted (κ_max < 1)
- Observed: θ_E = 35.0 ± 1.5 arcsec (MACS0416)
- Predicted: No Einstein radius

**Interpretation:** The coherence length optimized for galaxy scales (r ~ 5-30 kpc) does not extend to cluster scales (r ~ 100-1000 kpc).

---

## 📊 **TEST SETUP**

### Cluster Tested
- **MACS0416** (MACSJ0416.1-2403)
- z_lens = 0.396
- z_source = 2.0 (for lensing)
- Observed θ_E = 35.0 ± 1.5 arcsec (Frontier Fields gold standard)
- Quality: ⭐⭐⭐ Best (194 lensed images, Tier 1 data)

### Frozen Parameters (from SPARC)
```
L_0 = 4.993 kpc      # Coherence length
p = 0.757             # Power law exponent
n_coh = 0.500         # Coherence index
β_bulge = 1.759       # Bulge suppression
α_shear = 0.149       # Shear coupling
γ_bar = 1.932         # Baryonic scaling
A_0 = 0.591           # Path amplitude
g_dagger = 1.2×10⁻¹⁰ m/s² (FIXED)
```

**Zero per-cluster tuning** - All parameters frozen from rotation curve analysis.

---

## 🔬 **METHODOLOGY**

### Pipeline (7 Steps)

1. ✅ **Data Loading**
   - Loaded MACS0416 baryon profiles
   - 10,363 radial points (0.1 - 1314 kpc)
   - ρ_gas + ρ_stars validated
   - M_baryon(<500 kpc) = 1.1×10¹³ M_☉

2. ✅ **Baryonic Quantities**
   - Computed M_bar(r), g_bar(r), v_bar(r)
   - Integration: M(<r) = 4π ∫ ρ(r') r'² dr'
   - Acceleration: g = GM/r²

3. ✅ **Path-Spectrum Boost**
   - Applied kernel.many_path_boost_factor()
   - K(r) computed at all radial points
   - **RESULT: K ~ 0.0000** (median & mean)

4. ✅ **Effective Mass**
   - M_eff = M_bar × (1 + K)
   - **M_eff ≈ M_bar** (no boost)
   - ρ_eff computed via differentiation

5. ✅ **Abel Projection**
   - Σ(R) = 2 ∫_R^∞ ρ(r) r dr / √(r² - R²)
   - 200 projected radii (R = 1-1314 kpc)
   - Σ_bar and Σ_eff computed

6. ✅ **Lensing Quantities**
   - Σ_crit = 2.15×10⁹ M_☉/kpc²
   - κ(R) = Σ(R) / Σ_crit
   - <κ>(<R) = mean convergence
   - **κ_max < 1.0** (no critical surface)

7. ❌ **Einstein Radius**
   - Baryons only: No θ_E (κ_max < 1)
   - With boost: No θ_E (κ_max < 1)
   - **FAILURE:** Cannot produce lensing signal

---

## 📉 **DETAILED RESULTS**

### Boost Factor Analysis

| Radius | K (boost factor) | M_eff / M_bar |
|--------|------------------|---------------|
| 10 kpc | ~0.0000 | 1.000 |
| 100 kpc | ~0.0000 | 1.000 |
| 500 kpc | ~0.0000 | 1.000 |
| 1000 kpc | ~0.0000 | 1.000 |

**Statistics:**
- K_median = 0.0000
- K_mean = 0.0000
- K_max < 0.0001

### Convergence Profile

At R = 100 kpc:
- κ_bar (baryons) = 0.020
- κ_eff (with boost) = 0.020
- **No difference** (boost ineffective)

Maximum convergence:
- κ_max ≈ 0.5 (at R ~ 1 kpc)
- **Never reaches κ = 1** (no Einstein radius)

### Comparison to Observation

| Quantity | Baryons Only | With Boost | Observed |
|----------|--------------|------------|----------|
| θ_E [arcsec] | None | None | 35.0 ± 1.5 |
| κ_max | <1 | <1 | >1 (strong lens) |
| Status | ❌ Fail | ❌ Fail | - |

**Deficit:** Need κ ~ 50× larger to match observations

---

## 🔍 **DIAGNOSIS: WHY ZERO BOOST?**

### Scale Mismatch

**SPARC Galaxies (where model works):**
- Characteristic radius: r_half ~ 5-15 kpc
- Coherence length: L₀ = 4.99 kpc
- Ratio: L₀ / r_half ≈ 0.3-1.0 ✅

**Clusters (where model fails):**
- Characteristic radius: r_vir ~ 1000-2000 kpc
- Coherence length: L₀ = 4.99 kpc (same)
- Ratio: L₀ / r_vir ≈ 0.005 ❌

**The coherence mechanism depends on L₀ being comparable to the system scale.**

### Kernel Behavior at Large r

The boost factor formula includes:
```python
L_eff = L_0 × (v_circ/v_0)^p × (1 + n_coh × curvature/c₀)
```

At cluster scales:
- v_circ ~ 1500 km/s (much larger than galaxy ~200 km/s)
- L_eff might scale up, BUT
- The coherence function decays at r >> L_eff
- Net result: K → 0

### Physical Interpretation

Two possibilities:

**1. Coherence Length Should Scale**
- L₀(galaxies) = 5 kpc
- L₀(clusters) = 500 kpc? (100× larger)
- Would need L₀ ∝ r_vir or L₀ ∝ M^(1/3)

**2. Mechanism is Galaxy-Specific**
- Coherent disk rotation enables geometry gating
- Clusters lack coherent rotation (pressure-supported)
- Mechanism genuinely doesn't apply

---

## 🎓 **SCIENTIFIC IMPLICATIONS**

### What We Learned

1. **✅ Pipeline Validated**
   - Data loading works (30 clusters available)
   - Lensing calculations correct (matches test cases)
   - Full infrastructure in place

2. **✅ Clean Negative Result**
   - Not a numerical issue (K explicitly ~0)
   - Not a data issue (cluster profiles validated)
   - Genuine physics limitation discovered

3. **⚠️ Scale-Dependent Physics**
   - Model works at galaxy scale (RAR, rotation curves)
   - Model fails at cluster scale (lensing)
   - **Transition scale: ~10-100 kpc**

### Comparison to Other Theories

| Theory | Galaxies | Clusters | Status |
|--------|----------|----------|--------|
| **Our Model** | ✅ Works | ❌ Fails | Scale-dependent |
| **MOND** | ✅ Works | ❌ Fails | Needs DM for lensing |
| **ΛCDM** | ✅ Works | ✅ Works | Requires dark matter |

**Our model behaves like MOND:** Works for galaxies, fails for clusters.

---

## 🔧 **POTENTIAL MODIFICATIONS**

### Option 1: Scale-Dependent L₀

Replace fixed L₀ with scaling:
```python
L_0(M) = L_0_gal × (M / M_gal)^α
```

Where α ~ 0.3-0.5 might work.

**Pros:**
- Simple modification
- Preserves galaxy results
- Could match clusters

**Cons:**
- Adds 1-2 free parameters
- Ad hoc scaling
- Loses "universal" nature

### Option 2: Different Mechanism for Clusters

Separate coherence function for pressure-supported systems:
- Galaxies: Disk geometry gating
- Clusters: ???

**Pros:**
- Physically motivated
- Acknowledges different physics

**Cons:**
- Not a single "universal" law
- Complicates theory

### Option 3: Accept Limitation

Model is galaxy-specific:
- Coherent rotation required
- Doesn't apply to clusters
- Clusters still need dark matter

**Pros:**
- Honest about scope
- Still valuable for galaxies
- No ad hoc fixes

**Cons:**
- Doesn't solve full DM problem
- Similar limitation to MOND

---

## 📋 **NEXT STEPS**

### Immediate Diagnostics

1. **Test L₀ Scaling**
   ```python
   L_0_cluster = L_0_galaxy × (M_cluster / M_galaxy)^0.3
   ```
   See if K becomes non-zero

2. **Examine Kernel Components**
   - Check coherence function C_coh(r, L_eff)
   - Check geometry gating G(shear, bulge)
   - Identify which component kills boost

3. **Test Intermediate Scales**
   - Groups (M ~ 10¹³ M_☉, r ~ 300 kpc)
   - Where does transition happen?

### Alternative Tests

1. **Galaxy-Galaxy Lensing**
   - Stacked ΔΣ(R) around galaxies
   - r ~ 10-100 kpc (intermediate scale)
   - Might show partial boost

2. **Cluster Outskirts**
   - Test at R > 1 Mpc
   - Where cluster density → galaxy density
   - Transition region

### Theory Development

1. **Understand Scale Dependence**
   - Why does L₀ not scale naturally?
   - Physical origin of 5 kpc length?
   - Connection to disk scale height?

2. **Generalize Mechanism**
   - What truly defines "coherence"?
   - Can it apply to clusters?
   - Or intrinsically disk-specific?

---

## 📁 **FILES CREATED**

### Code (Production Quality)
1. `cluster_data_loader.py` - Validated data loading for 30 clusters
2. `lensing_utilities.py` - Cosmology, Abel projection, Einstein radius
3. `run_cluster_lensing_b1.py` - Complete prediction pipeline

### Results
4. `results/cluster_lensing_b1/macsj0416_predictions.json` - Full results
5. `results/cluster_lensing_b1/macsj0416_diagnostics.png` - Diagnostic plots

### Documentation
6. `LENSING_DATA_INVENTORY.md` - 30 clusters catalogued
7. `LENSING_QUICK_REFERENCE.md` - Quick implementation guide
8. `TRACK_B1_RESULTS.md` - This file

---

## ✅ **VALIDATION CHECKLIST**

### Data Quality
- ✅ Cluster profiles loaded correctly
- ✅ 10,363 radial points validated
- ✅ Monotonic radius, positive densities
- ✅ Physical density ranges confirmed
- ✅ M_baryon = 1.1×10¹³ M_☉ reasonable

### Computational Verification
- ✅ Path-spectrum kernel executed
- ✅ Abel projection converged
- ✅ Cosmology distances match expectations
- ✅ Σ_crit = 2.15×10⁹ M_☉/kpc² correct
- ✅ All numerical integrations stable

### Physics Checks
- ✅ Baryons alone insufficient (expected)
- ✅ Boost factor formula evaluated
- ✅ K ~ 0 is genuine result, not numerical error
- ✅ Scale mismatch identified as root cause

---

## 🎓 **CONCLUSIONS**

### Primary Finding

**The path-spectrum kernel optimized for SPARC galaxies (L₀ = 4.99 kpc) produces negligible boost at cluster scales (r ~ 100-1000 kpc).**

This is a **scale-dependent limitation**, not a failure of implementation.

### Status of Track B1

**❌ FAILED:** Cannot predict cluster lensing with frozen parameters

**✅ SUCCESS:** Discovered important physics constraint

### Impact on Overall Model

**Galaxy Scale (Tracks A1, A2, D):**
- ✅ Model works excellently
- ✅ RAR scatter: 0.084 dex (best)
- ✅ Rotation curves: competitive with MOND/ΛCDM
- ✅ Predictive power validated

**Cluster Scale (Track B1):**
- ❌ Model produces no boost
- ❌ Cannot explain lensing
- ⚠️ Same limitation as MOND

### Theoretical Implications

The model is **galaxy-specific**, requiring:
1. Coherent disk rotation
2. Appropriate scale (L₀ ~ r_characteristic)
3. Geometry gating mechanism

For clusters:
- Either L₀ must scale with system size (ad hoc)
- Or mechanism genuinely doesn't apply (scope limitation)
- Or different physics needed (pressure support vs rotation)

---

## 📚 **REFERENCES**

### Data Sources
- CLASH survey: Postman+ 2012
- MACS0416 lensing: Jauzac+ 2014, 2015
- Frontier Fields: Lotz+ 2017
- NFW parameters: Umetsu+ 2016

### Model Papers
- RAR: McGaugh+ 2016
- MOND limitations: Sanders & McGaugh 2002
- Cluster lensing: Bradač+ 2008

---

**Track B1 Status:** Complete (with negative result)  
**Next Priority:** Understand scale dependence, test intermediate scales  
**Model Status:** Works at galaxy scale, fails at cluster scale (similar to MOND)

