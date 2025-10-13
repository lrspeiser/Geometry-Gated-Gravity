# Cluster Lensing Data Inventory - Track B1 Preparation
**Date:** 2025-01-13  
**Purpose:** Document all existing cluster analysis work to integrate with new universal geometry-gated model  
**Context:** Track B1 requires lensing pipeline using frozen 7 parameters from rotation curve analysis

---

## 🎯 **EXECUTIVE SUMMARY**

You have **EXTENSIVE** existing cluster lensing infrastructure:

- ✅ **30 clusters** with complete baryon profiles (gas + stars + temperature)
- ✅ **20 clusters** with NFW parameters from Umetsu+ 2016
- ✅ **6 Frontier Fields** with published Einstein radii (gold standard)
- ✅ **Existing analysis pipeline** with hundreds of previous runs
- ✅ **Multiple analysis frameworks** already developed

**Gap**: Need to adapt previous G³ approaches to new path-spectrum kernel with frozen parameters from SPARC.

---

## 📊 **CLUSTER DATASETS**

### 1. Baryon Profiles (30 Clusters - COMPLETE)

**Location:** `C:\Users\henry\dev\GravityCalculator\data\clusters\`

Each cluster has 4 CSV files:
- `gas_profile.csv` - Electron density n_e(r) [cm⁻³]
- `stars_profile.csv` - Stellar density ρ_⋆(r) [M_☉/kpc³]
- `temp_profile.csv` - X-ray temperature kT(r) [keV]
- `clump_profile.csv` - Clumping factor C(r)

**Full Cluster List (30 clusters):**

| Cluster | z_lens | Notes |
|---------|--------|-------|
| A1795 | 0.0625 | Cool-core |
| A2029 | 0.0773 | Massive relaxed |
| A478 | 0.0881 | Intermediate-mass |
| ABELL_0209 | 0.206 | CLASH |
| ABELL_0383 | 0.187 | CLASH |
| ABELL_0426 | 0.0179 | Perseus |
| ABELL_0611 | 0.288 | CLASH |
| ABELL_1423 | - | CLASH |
| ABELL_1689 | 0.183 | Strong lens benchmark |
| ABELL_2261 | 0.224 | CLASH |
| CLJ1226 | 0.890 | High-z |
| MACSJ0329 | 0.450 | CLASH |
| **MACSJ0416** | **0.396** | **✅ HFF Priority** |
| MACSJ0429 | 0.399 | CLASH |
| MACSJ0647 | 0.584 | CLASH |
| **MACSJ0717** | **0.548** | **✅ HFF Priority** |
| MACSJ0744 | 0.686 | CLASH |
| MACSJ1115 | 0.352 | CLASH |
| **MACSJ1149** | **0.544** | **✅ HFF Priority** |
| MACSJ1206 | 0.440 | CLASH |
| MACSJ1311 | 0.494 | CLASH |
| MACSJ1423 | 0.545 | CLASH |
| MACSJ1720 | 0.391 | CLASH |
| MACSJ1931 | 0.352 | CLASH |
| MACSJ2129 | 0.570 | CLASH |
| MS2137 | 0.313 | CLASH |
| RXJ1347 | 0.451 | Brightest X-ray |
| RXJ1532 | 0.345 | CLASH |
| RXJ2129 | 0.234 | CLASH |
| RXJ2248 | 0.348 | CLASH |

---

### 2. NFW Dark Matter Parameters (20 Clusters)

**Source:** Umetsu et al. 2016 (ApJ 821, 116)  
**File:** `C:\Users\henry\dev\GravityCalculator\data\literature\nfw_params.json`

**Key Clusters for Track B1:**

| Cluster | z_lens | M_200c [10¹⁵ M_☉] | c_200c | r_s [kpc] | Notes |
|---------|--------|-------------------|--------|-----------|-------|
| **MACSJ0416** | **0.396** | **1.074±0.26** | **2.9±0.7** | **650±180** | ✅ HFF Tier 1 |
| **MACSJ0717** | **0.548** | **2.677±0.54** | **1.8±0.4** | **1310±310** | ✅ HFF Tier 2 |
| **MACSJ1149** | **0.544** | **2.502±0.55** | **2.1±0.6** | **1120±350** | ✅ HFF Tier 1 |
| ABELL_1689 | 0.183 | - | - | - | Classic benchmark |
| ABELL_0383 | 0.187 | 0.798±0.27 | 5.9±1.8 | 310±130 | CLASH |
| ABELL_0209 | 0.206 | 1.54±0.34 | 2.7±0.6 | 840±220 | CLASH |
| ABELL_2261 | 0.224 | 2.31±0.52 | 3.7±0.9 | 690±200 | CLASH |
| ABELL_0611 | 0.288 | 1.576±0.45 | 3.9±1.2 | 570±210 | CLASH |
| MS2137 | 0.313 | 1.356±0.53 | 2.7±1.3 | 800±450 | CLASH |
| RXJ1347 | 0.451 | 3.425±0.88 | 3.2±0.9 | 850±290 | CLASH |
| MACSJ0744 | 0.686 | 1.803±0.50 | 3.5±1.2 | 580±230 | CLASH |
| MACSJ0647 | 0.584 | 1.390±0.42 | 4.1±1.5 | 480±210 | CLASH |

**Coverage:** 20/30 clusters (67%) have NFW parameters for comparison

---

### 3. Frontier Fields Einstein Radii (6 Clusters - Gold Standard)

**Source:** Multiple published strong lensing models  
**File:** `C:\Users\henry\dev\GravityCalculator\data\frontier\gold_standard\gold_standard_clusters.json`

| Cluster | z_lens | z_source | θ_E [arcsec] | σ_θE | Quality |
|---------|--------|----------|--------------|------|---------|
| **MACS0416** | **0.396** | **2.0** | **35.0** | **1.5** | ✅ Tier 1 (194 images) |
| **MACS0717** | **0.548** | **2.5** | **55.0** | **3.0** | ✅ Tier 2 (Largest) |
| Abell 370 | 0.375 | 2.0 | 38.0 | 2.0 | Tier 1 (114+ images) |
| Abell 2744 | 0.308 | 2.0 | 26.0 | 2.0 | Tier 1 (Complex merger) |
| RXJ1347 | 0.451 | 2.0 | 32.0 | 2.0 | Tier 2 (Brightest X-ray) |
| **Abell 1689** | **0.183** | **2.0** | **47.0** | **3.0** | ✅ Classic benchmark |

**Priority Targets:** MACS0416, MACS0717, Abell 1689 (all have both baryons + NFW + θ_E)

---

## 💻 **EXISTING ANALYSIS CODE**

### Main Analysis Scripts

**Location:** `C:\Users\henry\dev\GravityCalculator\concepts\cluster_lensing\`

1. **`cluster_lensing_analysis.py`**
   - Full lensing pipeline: Σ(R), κ(R), γ_t(R), θ_E
   - Universal G³ model implementation
   - Abel transform for projection
   - Cosmology calculations (flat ΛCDM)
   - **Needs adaptation:** Uses old G³ parameters

2. **`cluster_lensing_diagnostic.py`**
   - Compare predictions vs observations
   - NFW comparisons
   - Einstein radius diagnostics

3. **`cluster_baryon_lensing_comparison.py`**
   - Compare baryonic vs effective lensing
   - ΔΣ(R) profiles

4. **`benchmark_vs_nfw.py`**
   - Head-to-head NFW comparison

5. **`analyze_deflections.py`**
   - Deflection angle analysis

### G³ Test Framework (Previous Approaches)

**Location:** `C:\Users\henry\dev\GravityCalculator\concepts\cluster_lensing\g3_cluster_tests\`

Multiple previous test approaches:
- `branch_a_late_saturation/` - Late saturation models
- `branch_b_photon_boost/` - Photon boost models
- `o3_lensing.py`, `o3_slip.py` - O³ approaches
- Grid searches for parameter tuning

**Status:** These used per-cluster tuning. Track B1 requires NO tuning (frozen 7 params).

---

## 📈 **PREVIOUS RESULTS**

### Extensive Output Directory

**Location:** `C:\Users\henry\dev\GravityCalculator\out\`

Hundreds of previous runs organized by approach:
- `cluster_lensing_real/` - Real baryon profiles
- `cluster_lensing_real_amp/` - Amplitude variations
- `cluster_lensing_real_beta_phi0/` - Beta-phi variations
- `cluster_lensing_real_beta_phi0_gsat/` - Saturation tests
- `cluster_lensing_real_sigma_gate/` - Sigma gating tests
- `cluster_lensing_sigma_gate_random/` - Random parameter searches

**File Format:**
Each run has `summary_realSigma.json` with:
- Einstein radius predictions
- Baryon-only θ_E
- Model-enhanced θ_E
- Cluster metadata

### Previous O² Lensing Results

**Location:** `C:\Users\henry\dev\GravityCalculator\O2_ratio_curv_publication\results\best_fit\`

- Abell 1689, Bullet, Coma profiles
- κ(R), γ(R), ΔΣ(R) plots
- Multiple cluster summaries

---

## 🔧 **KEY INFRASTRUCTURE UTILITIES**

### Cosmology (`lensing_cosmology.py`)
- Angular diameter distances D_A(z)
- Critical surface density Σ_crit(z_d, z_s)
- Flat ΛCDM with H₀=70, Ω_m=0.3

### Data Loading
- CSV parsers for gas/stars/temp profiles
- JSON loaders for NFW parameters
- Gold standard Einstein radii

### Projection Tools
- Abel transform: ρ(r) → Σ(R)
- M_enc(r) → ρ(r) differentiation
- Spherical mass profiles

---

## 🎯 **TRACK B1 IMPLEMENTATION PLAN**

### What We Have (Ready to Use)

✅ **Data:**
- 30 clusters with baryon profiles
- 20 clusters with NFW comparison parameters
- 6 clusters with gold-standard Einstein radii
- 3 priority targets with all data (MACS0416, MACS0717, Abell 1689)

✅ **Code:**
- Full lensing pipeline framework
- Cosmology utilities
- Abel projection tools
- Plotting and diagnostics

✅ **Infrastructure:**
- Organized data directories
- JSON metadata formats
- Output organization system

### What Needs Adaptation

⚠️ **Critical Changes:**

1. **Replace G³ model with Path-Spectrum Kernel**
   - Current: `UniversalG3Model` with 12 parameters
   - New: `PathSpectrumKernel` with 7 frozen parameters from SPARC
   - Source: `many_path_model/path_spectrum_kernel_track2.py`

2. **Load Frozen Parameters from SPARC**
   - File: `splits/sparc_split_v1.json`
   - Parameters: L_0, p, n_coh, β_bulge, α_shear, γ_bar, A_0, g_dagger

3. **Adapt to Cluster Scale**
   - SPARC: r ~ 1-30 kpc (galaxy scale)
   - Clusters: r ~ 10-1000 kpc (100× larger)
   - May need scale parameter adjustment

4. **Compute Lensing Quantities**
   - g_total(r) from boost factor K(r)
   - M_eff(r) from g_total
   - ρ_eff(r) via differentiation
   - Σ_eff(R) via Abel transform
   - κ(R), γ_t(R), θ_E from Σ_eff

---

## 📋 **RECOMMENDED TRACK B1 WORKFLOW**

### Phase 1: Single Cluster Test (This Week)

**Target:** MACS0416 (best quality data)

1. Load frozen hyperparameters from SPARC split
2. Load MACS0416 baryon profiles
3. Compute baryonic g_bar(r) from gas + stars
4. Apply path-spectrum kernel → g_total(r) = g_bar × (1 + K)
5. Compute effective Σ_eff(R) via Abel transform
6. Predict Einstein radius θ_E
7. Compare to observed θ_E = 35.0 ± 1.5 arcsec

**Success Criterion:** θ_E within factor of 2 of observation (no tuning!)

### Phase 2: 3-Cluster Validation (Next Week)

**Targets:** MACS0416, MACS0717, Abell 1689

1. Run pipeline on all 3 clusters
2. Compute θ_E predictions (zero free parameters)
3. Compare to gold standard observations
4. Compute median APE across 3 clusters
5. Compare to NFW predictions (uses 2-3 params per cluster)

**Success Criterion:** Comparable accuracy to NFW despite 0 params/cluster

### Phase 3: Extended Validation (Week After)

**Targets:** All 6 Frontier Fields + 10 additional CLASH clusters

1. Run on 16 total clusters with strong lensing
2. Compute Einstein radius predictions
3. ΔΣ(R) profiles where available
4. κ(R), γ_t(R) radial profiles
5. Comparison plots vs NFW

**Success Criterion:** No systematic bias; competitive scatter

### Phase 4: Galaxy-Galaxy Lensing (If Time)

Use stacked ΔΣ(R) from SDSS/DECaLS for lower-mass systems

---

## 📁 **FILE STRUCTURE FOR TRACK B1**

```
many_path_model/
├── run_cluster_lensing_b1.py         # Main Track B1 script
├── cluster_lensing_kernel.py         # Adapt path-spectrum to clusters
├── cluster_data_loader.py            # Load baryon profiles
├── lensing_predictions.py            # Compute θ_E, ΔΣ, etc.
└── results/
    └── cluster_lensing_b1/
        ├── macs0416_predictions.json
        ├── macs0416_profiles.csv
        ├── macs0416_plots.png
        ├── summary_3clusters.json
        └── comparison_vs_nfw.png
```

---

## 🔑 **KEY TECHNICAL CONSIDERATIONS**

### 1. Scale Transition: Galaxies → Clusters

**SPARC Galaxies:**
- r_half ~ 5-15 kpc
- L_0 = 4.99 kpc (coherence length)
- M_bary ~ 10⁹-10¹¹ M_☉

**Clusters:**
- r_vir ~ 1000-2000 kpc
- Core radius ~ 100-300 kpc
- M_bary ~ 10¹³-10¹⁴ M_☉

**Scaling Strategy:**
- Option A: Use L_0 directly (assume universal)
- Option B: Scale L_0 ∝ r_vir (cluster-dependent)
- Option C: Test both and see which works

### 2. Baryonic Profile Construction

For clusters, need to combine:
- **Gas:** ρ_gas(r) from n_e(r) × m_p × μ
- **Stars:** ρ_⋆(r) directly from profile
- **Temperature:** For pressure support (if needed)

Total baryonic acceleration:
```
g_bar(r) = G × M_bar(<r) / r²
M_bar(<r) = 4π ∫₀ʳ [ρ_gas(r') + ρ_⋆(r')] r'² dr'
```

### 3. Boost Factor Application

Two possible implementations:

**Option A: Direct g_total**
```python
K = kernel.many_path_boost_factor(r, v_circ, g_bar)
g_total = g_bar × (1 + K)
v_eff = sqrt(g_total × r)
```

**Option B: Effective mass**
```python
M_eff(<r) = M_bar(<r) × (1 + K_avg(<r))
g_total = G × M_eff(<r) / r²
```

### 4. Projection to ΔΣ(R)

Two approaches:

**Approach 1: Via κ(R)**
```python
Σ_eff(R) = Abel_project[ρ_eff(r)]
κ(R) = Σ_eff(R) / Σ_crit
ΔΣ(R) = Σ_eff(<R) - Σ_eff(R)
```

**Approach 2: Direct from M_eff**
```python
ΔΣ(R) = [M_eff(<R) / (π R²) - Σ_eff(R)]
```

---

## 🎓 **SCIENTIFIC CONTEXT**

### Why This Matters (Track B1)

**The Critical Test:**
- Rotation curves: Circular motion (1D velocity)
- Lensing: Gravitational potential (2D projection)
- **Different observable!** Tests potential directly

**If Track B1 works:**
- ✅ Validates geometry-gated boost at potential level
- ✅ Proves mechanism affects light paths (not just dynamics)
- ✅ 0 parameters/cluster (unlike NFW: 2-3 params/cluster)
- ✅ **Game-changing result**

**Historical Precedent:**
- MOND works for rotation curves
- **But fails for clusters** (needs dark matter for lensing)
- If we match lensing with NO dark matter...

---

## 📚 **REFERENCES & DATA SOURCES**

### Papers
1. **Umetsu et al. 2016** (ApJ 821, 116) - CLASH NFW masses
2. **Postman et al. 2012** - CLASH survey overview
3. **Zitrin et al. 2015** (ApJ 801, 44) - CLASH strong lensing
4. **Lotz et al. 2017** - Frontier Fields overview

### Data Archives
- CLASH: https://archive.stsci.edu/prepds/clash/
- Frontier Fields: https://frontierfields.org/
- Chandra: https://cxc.harvard.edu/cda/

---

## 🚀 **IMMEDIATE NEXT STEPS**

1. **Read frozen SPARC parameters** from `splits/sparc_split_v1.json`
2. **Load MACS0416 baryon profiles**
3. **Adapt path-spectrum kernel** to cluster scale
4. **Compute first lensing prediction**
5. **Compare to θ_E = 35 arcsec**

**Timeline:** First prediction within 1-2 days, full Track B1 within 1-2 weeks.

---

**Status:** Ready to begin Track B1 implementation  
**Confidence:** HIGH - Excellent data coverage, mature infrastructure  
**Risk:** Scale transition galaxies→clusters may require parameter adjustment

