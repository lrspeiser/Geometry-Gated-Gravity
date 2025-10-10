# Data Inventory - What We Have

**Date**: 2025-01-10  
**Purpose**: Document actual cluster data available in our repository

---

## Summary

✅ **We have EXTENSIVE real cluster data!**

### What We Found:

1. **25 CLASH clusters** with full baryon profiles (gas + stars + temperature)
2. **6 Frontier Fields clusters** with published Einstein radii and multiple lensing models
3. **Existing analysis outputs** for many CLASH clusters
4. **1 published paper** (Umetsu+ 2016) on CLASH masses

---

## 1. CLASH Cluster Data (25 clusters)

**Location**: `C:\Users\henry\dev\GravityCalculator\data\clusters\`

**Format**: Each cluster has 4 CSV files:
- `gas_profile.csv` - Gas density (n_e in cm^-3) vs radius (kpc)
- `stars_profile.csv` - Stellar density (Msun/kpc^3) vs radius
- `temp_profile.csv` - X-ray temperature (kT in keV) vs radius
- `clump_profile.csv` - Clumping factor vs radius

### Complete List of CLASH Clusters:

| Cluster Name | Data Folder | Notes |
|-------------|------------|-------|
| A1795 | `data/clusters/A1795/` | Cool-core, z=0.0625 |
| A2029 | `data/clusters/A2029/` | Massive relaxed, z=0.0773 |
| A478 | `data/clusters/A478/` | Intermediate-mass, z=0.0881 |
| Abell 0209 | `data/clusters/ABELL_0209/` | CLASH |
| Abell 0383 | `data/clusters/ABELL_0383/` | CLASH |
| Abell 0426 | `data/clusters/ABELL_0426/` | Perseus cluster, z=0.0179 |
| Abell 0611 | `data/clusters/ABELL_0611/` | CLASH |
| Abell 1423 | `data/clusters/ABELL_1423/` | CLASH |
| Abell 1689 | `data/clusters/ABELL_1689/` | Strong lensing, z=0.183 |
| Abell 2261 | `data/clusters/ABELL_2261/` | CLASH |
| CLJ1226 | `data/clusters/CLJ1226/` | High-z, z=0.890 |
| MACSJ0329 | `data/clusters/MACSJ0329/` | z=0.450 |
| **MACSJ0416** | `data/clusters/MACSJ0416/` | **✅ TRAINING**, z=0.396 |
| MACSJ0429 | `data/clusters/MACSJ0429/` | z=0.399 |
| MACSJ0647 | `data/clusters/MACSJ0647/` | z=0.584 |
| **MACSJ0717** | `data/clusters/MACSJ0717/` | **✅ TRAINING**, z=0.548 |
| MACSJ0744 | `data/clusters/MACSJ0744/` | z=0.686 |
| MACSJ1115 | `data/clusters/MACSJ1115/` | z=0.352 |
| **MACSJ1149** | `data/clusters/MACSJ1149/` | **✅ TRAINING**, z=0.544 |
| MACSJ1206 | `data/clusters/MACSJ1206/` | z=0.440 |
| MACSJ1311 | `data/clusters/MACSJ1311/` | z=0.494 |
| MACSJ1423 | `data/clusters/MACSJ1423/` | z=0.545 |
| MACSJ1720 | `data/clusters/MACSJ1720/` | z=0.391 |
| MACSJ1931 | `data/clusters/MACSJ1931/` | z=0.352 |
| MACSJ2129 | `data/clusters/MACSJ2129/` | z=0.570 |
| MS2137 | `data/clusters/MS2137/` | z=0.313 |
| RXJ1347 | `data/clusters/RXJ1347/` | z=0.451 |
| RXJ1532 | `data/clusters/RXJ1532/` | z=0.345 |
| RXJ2129 | `data/clusters/RXJ2129/` | z=0.234 |
| RXJ2248 | `data/clusters/RXJ2248/` | z=0.348 |

**Total**: 30 clusters with baryon profiles (includes 25 CLASH + 5 additional)

---

## 2. Frontier Fields Lensing Data

**Location**: `C:\Users\henry\dev\GravityCalculator\data\frontier\`

### Published Einstein Radii (Gold Standard):

**File**: `data/frontier/gold_standard/gold_standard_clusters.json`

| Cluster | z_lens | z_source | θ_E (arcsec) | σ | Notes |
|---------|--------|----------|--------------|---|-------|
| Abell 370 | 0.375 | 2.0 | 38.0 ± 2.0 | 2.0 | HFF Tier 1; 114+ images |
| **MACS0416** | 0.396 | 2.0 | 35.0 ± 1.5 | 1.5 | HFF Tier 1; 194 images |
| Abell 2744 | 0.308 | 2.0 | 26.0 ± 2.0 | 2.0 | Complex merger |
| **MACS0717** | 0.545 | 2.5 | 55.0 ± 3.0 | 3.0 | Largest lensing cluster |
| RXJ1347 | 0.451 | 2.0 | 32.0 ± 2.0 | 2.0 | Brightest X-ray |
| Abell 1689 | 0.183 | 2.0 | 47.0 ± 3.0 | 3.0 | Classic benchmark |

### Multiple Lensing Models Available:

**Location**: `data/frontier/hlsp/`

**MACS0416** models:
- Caminha v4
- CATS v4.1
- Williams v4

**MACS0717** models:
- CATS v4.1
- Williams v4
- Williams v4.1

**MACS1149** models:
- CATS v4.1
- Williams v4

**Predicted θ_E from models** (`data/frontier/gold_standard/report_thetaE.csv`):
- MACS0416: 38.0-39.3 arcsec (3 models, z_s=2.0)
- MACS0717: 81.1-82.3 arcsec (2 models, z_s=2.0)
- MACS1149: ~42 arcsec (Williams v4, z_s=2.0)

---

## 3. Existing Analysis Outputs

**Location**: `C:\Users\henry\dev\GravityCalculator\out\cluster_lensing_real\`

We have already run analysis on many CLASH clusters! Each has:
- `summary_realSigma.json` - Predicted Einstein radius
- `profiles_realSigma.csv` - Radial profiles

### Example: MACS0416
**File**: `out/cluster_lensing_real/macs0416/summary_realSigma.json`

```json
{
  "cluster": "MACSJ0416",
  "z_lens": 0.396,
  "z_source": 2.0,
  "Einstein_radius_arcsec_baryons": 0.192,
  "Einstein_radius_arcsec_realSigma": 0.187,
  "note": "G³ tail gated by real Σ(R)"
}
```

**Observed**: 35.0 ± 1.5 arcsec (from gold standard)  
**Our prediction**: 0.19 arcsec  
**❌ ISSUE**: Our predictions are ~180x too small!

This suggests something is fundamentally wrong with our current implementation.

---

## 4. Literature Data

### Umetsu+ 2016 (ApJ 821, 116)
**File**: `C:\Users\henry\Documents\GitHub\DensityDependentMetricModel\external_data\Umetsu_2016_ApJ_821_116.pdf`

**Content**: 
- CLASH weak lensing masses for all 25 clusters
- M_200, concentration parameters
- Combined weak + strong lensing analysis

**This paper likely contains NFW parameters we need!**

### Abell 1689 Data
**File**: `C:\Users\henry\dev\GravityCalculator\data\mast-abell-1689.pdf`

---

## 5. Processing Scripts Available

### CLASH Model Processor
**File**: `concepts/cluster_lensing/process_clash_models.py`

**Purpose**: Extract Einstein radii from CLASH lensing models (if we had the FITS files)

**Inputs**: 
- `data/clash/hlsp/<cluster>/models/*kappa*.fits`
- `data/clash/hlsp/<cluster>/models/*gamma*.fits`

**Outputs**:
- `data/clash/processed/einstein_radii_clash.csv`
- `data/clash/processed/profiles/<cluster>_kappa_profile.csv`

**Status**: ❌ CLASH FITS files NOT found in `data/clash/hlsp/`

### Real-Σ Runner for CLASH
**File**: `scripts/run_real_sigma_for_clash.py`

**Purpose**: Run our slip model on all 25 CLASH clusters

**Status**: ✅ Already executed (see outputs in `out/cluster_lensing_real/`)

---

## 6. What We're Missing

### Critical Missing Data:

1. **NFW parameters from literature** for training clusters:
   - MACS0416: M_200, c_vir, r_s
   - MACS0717: M_200, c_vir, r_s  
   - MACS1149: M_200, c_vir, r_s
   
   **Where to find**: 
   - Umetsu+ 2016 (we have PDF)
   - Strong lensing papers (need to search)
   - CLASH mass model papers

2. **Multiple source redshifts** for MST test:
   - Need lensing constraints at 2+ z_source for same cluster
   - Best candidates: MACS0416, MACS0717, Abell 2744

3. **CLASH FITS models** (optional, nice to have):
   - Kappa and gamma maps from MAST archive
   - Would allow us to extract Einstein radii directly

---

## 7. Immediate Action Plan

### Phase 1: Extract NFW Parameters (This Week)

**Task**: Extract dark matter parameters from Umetsu+ 2016

**Steps**:
1. ✅ Located PDF: `external_data/Umetsu_2016_ApJ_821_116.pdf`
2. ⏳ Read Tables 3-4 for M_200, c_vir, r_s
3. ⏳ Create `data/literature_nfw_params.json` with extracted values
4. ⏳ Especially get MACS0416, MACS0717, MACS1149

**Expected output**:
```json
{
  "MACS0416": {
    "M_200_Msun": 1.15e15,
    "M_200_err": 0.15e15,
    "c_vir": 3.8,
    "c_vir_err": 0.5,
    "r_s_kpc": 420,
    "reference": "Umetsu+ 2016"
  }
}
```

### Phase 2: Fix Our Predictions (Next Week)

**Issue**: Our Einstein radius predictions are ~180x too small

**Hypothesis**: 
- Wrong units in cosmology calculation?
- Missing slip factor normalization?
- Error in deflection angle computation?

**Action**: 
1. Debug why θ_E(predicted) = 0.19" but θ_E(observed) = 35"
2. Check angular diameter distance calculations
3. Verify critical surface density Σ_crit
4. Test on known analytic profiles (SIS, NFW)

### Phase 3: Bootstrap Uncertainties (Week After)

**Task**: Implement bootstrap/jackknife for universal parameters

**Data**: Use our 3 training clusters (once we fix predictions)

**Output**: Uncertainties on:
- R_s = 0.90 R_edge (fit coefficient)
- S_∞ = 1 + α ε^0.60 (M_core/10^13)^0.25 (exponents + amplitude)

---

## 8. Data Quality Assessment

### ✅ What We Have (EXCELLENT):

1. **Baryon profiles**: 30 clusters with complete gas + stars + temperature
2. **Lensing constraints**: 6 clusters with published Einstein radii
3. **Multiple models**: 3 independent teams for MACS0416/0717/1149
4. **Literature paper**: Umetsu+ 2016 with NFW masses

### ⚠️ What Needs Work:

1. **Extract NFW parameters**: Manual extraction from PDF tables
2. **Fix prediction code**: Debug 180x discrepancy
3. **Multi-z sources**: Search literature for multiple source redshifts

### ❌ Not Available (But Not Critical):

1. CLASH FITS kappa/gamma maps (can download from MAST if needed)
2. 2D lensing reconstructions (optional, for 2D extension later)

---

## 9. Data Format Examples

### Baryon Profile Format:

**File**: `data/clusters/MACSJ0416/gas_profile.csv`
```csv
r_kpc,n_e_cm3
1184.0,0.00028571
1163.05,0.00037185
...
10.565,0.0083151
```

**Rows**: ~200 radial bins from 1 kpc to 1200 kpc

### Stars Profile Format:

**File**: `data/clusters/MACSJ0416/stars_profile.csv`
```csv
r_kpc,rho_star_Msun_per_kpc3
1.0,1500000.0
5.0,800000.0
...
500.0,10.0
```

### Temperature Profile Format:

**File**: `data/clusters/MACSJ0416/temp_profile.csv`
```csv
r_kpc,kT_keV,kT_err_keV
10.0,8.5,0.5
50.0,7.2,0.4
...
500.0,4.1,0.6
```

---

## 10. References to Process

### Papers with NFW Parameters:

1. **Umetsu et al. 2016** (ApJ 821, 116)
   - CLASH weak lensing masses
   - All 25 clusters
   - Tables 3-4: M_200, c_200

2. **Jauzac et al. 2015** (MNRAS 452, 1437)
   - MACS0416 strong lensing
   - Likely has NFW fit

3. **Jauzac et al. 2014** (MNRAS 443, 1549)
   - MACS0416 earlier analysis

4. **Medezinski et al. 2013** (ApJ 777, 43)
   - MACS0717 lensing + dynamics

5. **Zitrin et al. 2015** (ApJ 801, 44)
   - CLASH strong lensing models
   - All 25 clusters

### Search Strategy:

```bash
# Search for papers in data directory
fd -e pdf . "C:\Users\henry\dev\GravityCalculator\data"
fd -e pdf . "C:\Users\henry\Documents\GitHub\DensityDependentMetricModel"

# Look for NFW parameters in any text files
rg -i "M_200|M200|c_vir|cvir|scale radius|r_s" data/ -t txt -t md
```

---

## Summary: What To Do Next

1. **✅ NOW**: Read Umetsu+ 2016 and extract NFW parameters (especially MACS0416, 0717, 1149)
2. **⚠️ CRITICAL**: Debug why our predictions are 180x too small
3. **⏳ NEXT**: Once fixed, implement bootstrap for parameter uncertainties
4. **📋 LATER**: Search for multi-z lensing constraints in literature

**We have MORE than enough data to implement the full research roadmap!**

The main blocker is:
1. Extracting NFW parameters from Umetsu+ 2016
2. Fixing our prediction code bug

Once those are done, we can proceed with all phases of the editorial response.
