# Cluster Data Master Documentation

**Last Updated**: 2025-01-10  
**Purpose**: Comprehensive documentation of all cluster data sources, formats, and processing status  
**Maintainer**: Keep this updated as new data is added!

---

## Quick Reference

| Data Type | # Clusters | Location | Status |
|-----------|-----------|----------|--------|
| Baryon Profiles (gas+stars) | 30 | `data/clusters/` | ✅ Complete |
| Published Einstein Radii | 6 | `data/frontier/gold_standard/` | ✅ Complete |
| NFW Dark Matter Parameters | 0/25 | `data/literature/nfw_params.json` | ❌ TODO |
| Multiple Lensing Models | 3 | `data/frontier/hlsp/` | ✅ Partial |
| Multi-z Lensing Constraints | 0 | N/A | ❌ TODO |

---

## Table of Contents

1. [Baryon Profile Data](#1-baryon-profile-data)
2. [Strong Lensing Data](#2-strong-lensing-data)
3. [Literature NFW Parameters](#3-literature-nfw-parameters)
4. [Data Validation](#4-data-validation)
5. [Processing Scripts](#5-processing-scripts)
6. [Known Issues](#6-known-issues)
7. [Data Changelog](#7-data-changelog)

---

## 1. Baryon Profile Data

### Location
```
C:\Users\henry\dev\GravityCalculator\data\clusters\
```

### Format
Each cluster has a subdirectory with 4 CSV files:

#### `gas_profile.csv`
- **Columns**: `r_kpc`, `n_e_cm3`
- **Units**: 
  - `r_kpc`: Radius in kiloparsecs (physical, not comoving)
  - `n_e_cm3`: Electron number density in cm^-3
- **Range**: Typically 1-1200 kpc, ~200 radial bins
- **Source**: Chandra/XMM X-ray observations
- **Notes**: 3D deprojected density, not projected

#### `stars_profile.csv`
- **Columns**: `r_kpc`, `rho_star_Msun_per_kpc3`
- **Units**:
  - `r_kpc`: Radius in kpc
  - `rho_star_Msun_per_kpc3`: 3D stellar mass density in M☉/kpc³
- **Components**: BCG (Brightest Cluster Galaxy) + ICL (Intracluster Light)
- **Source**: HST/ground-based optical/NIR photometry, deprojected
- **Notes**: Combined BCG+ICL profile

#### `temp_profile.csv`
- **Columns**: `r_kpc`, `kT_keV`, `kT_err_keV`
- **Units**:
  - `kT_keV`: X-ray spectroscopic temperature in keV
  - `kT_err_keV`: 1σ uncertainty
- **Source**: Chandra/XMM spectral fitting
- **Notes**: Used to compute gas pressure and hydrostatic equilibrium

#### `clump_profile.csv`
- **Columns**: `r_kpc`, `C`
- **Units**: `C` is dimensionless clumping factor
- **Definition**: C = sqrt(<ρ²>/<ρ>²), where C=1 means no clumping
- **Range**: Typically C = 1.0-2.5, higher in cores
- **Source**: Literature (Urban+2014, Eckert+2013) or conservative estimates
- **Notes**: Accounts for gas density fluctuations

### Complete Cluster List

#### CLASH Survey Clusters (25 total)

| Cluster | z_lens | Data Folder | Strong Lensing? | Notes |
|---------|--------|-------------|-----------------|-------|
| Abell 209 | 0.206 | `ABELL_0209/` | Yes | CLASH |
| Abell 383 | 0.187 | `ABELL_0383/` | Yes | CLASH |
| Abell 611 | 0.288 | `ABELL_0611/` | Yes | CLASH |
| Abell 1423 | 0.213 | `ABELL_1423/` | No | CLASH |
| Abell 2261 | 0.224 | `ABELL_2261/` | Yes | CLASH |
| CLJ1226+3332 | 0.890 | `CLJ1226/` | Yes | CLASH, high-z |
| MACSJ0329-02 | 0.450 | `MACSJ0329/` | Yes | CLASH |
| **MACSJ0416-24** | **0.396** | `MACSJ0416/` | **Yes** | **TRAINING**, HFF |
| MACSJ0429-02 | 0.399 | `MACSJ0429/` | Yes | CLASH |
| MACSJ0647+70 | 0.584 | `MACSJ0647/` | Yes | CLASH |
| **MACSJ0717+37** | **0.548** | `MACSJ0717/` | **Yes** | **TRAINING**, HFF |
| MACSJ0744+39 | 0.686 | `MACSJ0744/` | Yes | CLASH |
| MACSJ1115+01 | 0.352 | `MACSJ1115/` | Yes | CLASH |
| **MACSJ1149+22** | **0.544** | `MACSJ1149/` | **Yes** | **TRAINING**, HFF |
| MACSJ1206-08 | 0.440 | `MACSJ1206/` | Yes | CLASH |
| MACSJ1311-03 | 0.494 | `MACSJ1311/` | Yes | CLASH |
| MACSJ1423+24 | 0.545 | `MACSJ1423/` | Yes | CLASH |
| MACSJ1720+35 | 0.391 | `MACSJ1720/` | Yes | CLASH |
| MACSJ1931-26 | 0.352 | `MACSJ1931/` | Yes | CLASH |
| MACSJ2129-07 | 0.570 | `MACSJ2129/` | Yes | CLASH |
| MS2137-2353 | 0.313 | `MS2137/` | Yes | CLASH |
| RXJ1347-1145 | 0.451 | `RXJ1347/` | Yes | CLASH, HFF |
| RXJ1532+3021 | 0.345 | `RXJ1532/` | Yes | CLASH |
| RXJ2129+0005 | 0.234 | `RXJ2129/` | No | CLASH |
| RXJ2248-4431 | 0.348 | `RXJ2248/` | Yes | CLASH |

#### Additional Non-CLASH Clusters (5 total)

| Cluster | z_lens | Data Folder | Notes |
|---------|--------|-------------|-------|
| A1795 | 0.0625 | `A1795/` | Cool-core, relaxed |
| A2029 | 0.0773 | `A2029/` | Massive relaxed |
| A478 | 0.0881 | `A478/` | Intermediate-mass |
| Abell 426 (Perseus) | 0.0179 | `ABELL_0426/` | Cool-core, AGN feedback |
| Abell 1689 | 0.183 | `ABELL_1689/` | Classic strong lens |

**Total**: 30 clusters with complete baryon profiles

### Data Sources

**CLASH (Cluster Lensing And Supernova survey with Hubble)**
- PI: Marc Postman (STScI)
- Years: 2010-2013
- Clusters: 25 massive (M > 5×10^14 M☉)
- Data: HST 16-band imaging + Chandra X-ray + ground-based spectroscopy
- Papers:
  - Postman+ 2012 (ApJS 199, 25) - Survey design
  - Umetsu+ 2016 (ApJ 821, 116) - Weak lensing masses ⭐
  - Merten+ 2015 (ApJ 806, 4) - Weak lensing analysis
  - Zitrin+ 2015 (ApJ 801, 44) - Strong lensing models

**Non-CLASH Sources**
- A1795, A2029, A478: ACCEPT catalog (Cavagnolo+ 2009)
- Perseus: Chandra deep observations (Urban+ 2014)
- A1689: Multiple papers (Eckert+ 2013, Limousin+ 2007)

### Data Processing Notes

**Gas Profiles**:
- Derived from X-ray surface brightness via Abel deprojection
- Assumes spherical symmetry (appropriate for relaxed clusters)
- Electron density n_e converted to gas mass density via:
  ```
  ρ_gas = (μ_e m_p / X) × n_e
  ```
  where μ_e ≈ 1.17 (mean molecular weight per electron), X ≈ 0.71 (H mass fraction)

**Stellar Profiles**:
- BCG: Sérsic profile fit to optical/NIR photometry, deprojected
- ICL: Hernquist or exponential profile fit to extended light
- Combined 3D density profile for total stellar component
- M/L ratios from stellar population synthesis or dynamics

**Clumping Factors**:
- Direct measurements from X-ray surface brightness fluctuations (where available)
- Conservative estimates (C_0 ≈ 1.5-2.0) for clusters without direct measurements
- Important for accurate gas mass: M_gas ∝ 1/√C

---

## 2. Strong Lensing Data

### Published Einstein Radii (Gold Standard)

**Location**: 
```
C:\Users\henry\dev\GravityCalculator\data\frontier\gold_standard\gold_standard_clusters.json
```

**Format**:
```json
{
  "cluster_id": {
    "name": "Full cluster name",
    "z_lens": 0.396,
    "accepted": {
      "zs": 2.0,
      "theta_E_arcsec": 35.0,
      "sigma": 1.5
    },
    "notes": "Additional info"
  }
}
```

**Clusters with Published θ_E**:

| Cluster | z_lens | z_source | θ_E (arcsec) | σ (arcsec) | N_images | Reference |
|---------|--------|----------|--------------|------------|----------|-----------|
| **MACS0416** | 0.396 | 2.0 | **35.0** | 1.5 | 194 | HFF Tier 1 |
| **MACS0717** | 0.545 | 2.5 | **55.0** | 3.0 | ~150 | HFF Tier 2 |
| Abell 370 | 0.375 | 2.0 | 38.0 | 2.0 | 114+ | HFF Tier 1 |
| Abell 2744 | 0.308 | 2.0 | 26.0 | 2.0 | ~100 | HFF Tier 1 |
| Abell 1689 | 0.183 | 2.0 | 47.0 | 3.0 | ~135 | Classic |
| RXJ1347 | 0.451 | 2.0 | 32.0 | 2.0 | ~50 | HFF Tier 2 |

**Sources**:
- Hubble Frontier Fields (HFF): Lotz+ 2017 (ApJ 837, 97)
- Individual team models: Caminha, CATS, Williams, Zitrin, Sharon, etc.
- Consensus values from model comparison (Priewe+ 2017, Meneghetti+ 2017)

### Multiple Lensing Models

**Location**:
```
C:\Users\henry\dev\GravityCalculator\data\frontier\hlsp\
```

**Structure**:
```
data/frontier/hlsp/
├── macs0416/
│   ├── caminha/v4/
│   ├── cats/v4.1/
│   └── williams/v4/
├── macs0717/
│   ├── cats/v4.1/
│   └── williams/v4/
└── macs1149/
    ├── cats/v4.1/
    └── williams/v4/
```

**Model Predictions** (`data/frontier/gold_standard/report_thetaE.csv`):

| Cluster | Team | Version | z_s | θ_E (arcsec) |
|---------|------|---------|-----|--------------|
| MACS0416 | Caminha | v4 | 2.0 | 38.0 |
| MACS0416 | CATS | v4.1 | 2.0 | 39.0 |
| MACS0416 | Williams | v4 | 2.0 | 39.3 |
| MACS0717 | CATS | v4.1 | 2.0 | 81.1 |
| MACS0717 | Williams | v4 | 2.0 | 82.3 |
| MACS1149 | CATS | v4.1 | 2.0 | (low, error) |
| MACS1149 | Williams | v4 | 2.0 | 41.9 |

**Note**: Model spread indicates systematic uncertainties in mass modeling.

### Data Access

**MAST Archive**:
- URL: https://archive.stsci.edu/prepds/frontier/
- Content: Lensing catalogs, model outputs, mass maps
- Format: FITS files, ASCII catalogs, README files

**We Currently Have**:
- ✅ Model metadata (JSON files with URLs)
- ✅ README files describing models
- ❌ FITS kappa/gamma maps (can download if needed)

---

## 3. Literature NFW Parameters

### Status: ❌ NOT YET EXTRACTED

**Target Location** (to be created):
```
C:\Users\henry\dev\GravityCalculator\data\literature\nfw_params.json
```

**Required Format**:
```json
{
  "MACSJ0416": {
    "M_200_Msun": 1.15e15,
    "M_200_err_lower": 0.12e15,
    "M_200_err_upper": 0.18e15,
    "c_200": 3.8,
    "c_200_err": 0.5,
    "r_200_kpc": 1850,
    "r_s_kpc": 487,
    "method": "weak_lensing",
    "reference": "Umetsu+ 2016, ApJ 821, 116",
    "table": "Table 3",
    "notes": "Combined HST+Subaru weak lensing"
  }
}
```

### Primary Source: Umetsu+ 2016 ⭐

**Paper**: "CLASH: Joint Analysis of Strong-Lensing, Weak-Lensing Shear, and Magnification Data for 20 Galaxy Clusters"  
**Authors**: Umetsu et al.  
**Journal**: ApJ 821, 116 (2016)  
**ADS**: 2016ApJ...821..116U

**Location**: 
```
C:\Users\henry\Documents\GitHub\DensityDependentMetricModel\external_data\Umetsu_2016_ApJ_821_116.pdf
```

**What to Extract**:
- **Table 3**: Weak lensing masses (M_200, c_200) for 20 CLASH clusters
- **Table 4**: Combined strong+weak lensing masses (if available)
- **Redshifts**: z_lens for each cluster
- **Method**: Weak lensing vs combined lensing
- **Cosmology**: H_0, Ω_m, Ω_Λ used in paper (to convert if needed)

**Priority Clusters** (our 3 training clusters):
1. MACSJ0416-24
2. MACSJ0717+37
3. MACSJ1149+22

### Secondary Sources

**For cross-validation and additional clusters**:

1. **Merten+ 2015** (ApJ 806, 4)
   - CLASH weak lensing analysis
   - All 25 clusters
   - May have different mass estimates than Umetsu+2016

2. **Zitrin+ 2015** (ApJ 801, 44)
   - CLASH strong lensing models
   - NFW fits for cluster cores
   - 20+ clusters

3. **Individual cluster papers**:
   - MACS0416: Jauzac+ 2014, 2015 (MNRAS)
   - MACS0717: Medezinski+ 2013 (ApJ 777, 43)
   - MACS1149: Smith+ 2009, Ebeling+ 2007

### Extraction TODO List

- [ ] Read Umetsu+ 2016 Table 3
- [ ] Extract M_200, c_200 for all 20 clusters
- [ ] Compute r_s = r_200 / c_200
- [ ] Create `data/literature/nfw_params.json`
- [ ] Create `data/literature/nfw_params_sources.md` with references
- [ ] Cross-check with Merten+ 2015 for consistency
- [ ] Document any discrepancies between papers

---

## 4. Data Validation

### Validation Scripts

**Location**: `rigor/scripts/`

#### Check Cluster Data
```bash
py -u rigor/scripts/check_cluster_data.py --cluster MACSJ0416
```

**Checks**:
- File existence (4 required CSV files)
- Column names and format
- Radius monotonicity (must be increasing)
- Profile positivity (densities > 0)
- Mass integral convergence (at large radii)
- Unit consistency

**Output**: Pass/Fail + specific error messages

#### Compute Derived Quantities
```bash
py -u rigor/scripts/compute_cluster_summary.py --cluster MACSJ0416 --out data/summaries/
```

**Computes**:
- Total gas mass M_gas(<R) at various radii
- Total stellar mass M_stars(<R)
- Total baryon mass M_baryon = M_gas + M_stars
- Edge radius R_edge (where dΣ/dR is steepest)
- Edge sharpness ε (normalized gradient)
- Core baryon mass M_core = M_baryon(<R_edge)

**Output**: JSON summary file per cluster

### Quality Flags

Each cluster should have a quality assessment:

| Cluster | Gas Data | Stellar Data | Temp Data | Clumping | Overall | Notes |
|---------|----------|--------------|-----------|----------|---------|-------|
| MACSJ0416 | A | B | A | B | A | High S/N X-ray, good photometry |
| MACSJ0717 | A | B | A | C | A | Merging, complex morphology |
| MACSJ1149 | A | B | A | B | A | Well-studied, multiple arcs |

**Grades**:
- **A**: High S/N, well-measured, radial coverage >500 kpc
- **B**: Good S/N, coverage >300 kpc, some gaps acceptable
- **C**: Moderate S/N, limited coverage, or significant uncertainties
- **D**: Poor quality or incomplete data

### Known Data Issues

**MACSJ1149**:
- Missing `profiles_realSigma.csv` in `out/cluster_lensing_real/macs1149/`
- May indicate processing failure - needs investigation

**CLJ1226**:
- Very high redshift (z=0.890)
- Lower S/N X-ray data due to distance
- Stellar mass more uncertain

**Perseus (ABELL_0426)**:
- Very nearby (z=0.0179)
- AGN feedback creates cavities in gas
- Non-spherical, but best-studied cool core

---

## 5. Processing Scripts

### Baryon Surface Density Σ(R)

**Script**: `concepts/cluster_lensing/compute_sigma_from_profiles.py`

**Usage**:
```bash
py -u concepts/cluster_lensing/compute_sigma_from_profiles.py \
  --cluster MACSJ0416 \
  --cluster_dir data/clusters/MACSJ0416 \
  --z_lens 0.396 \
  --out data/processed/MACSJ0416_sigma.csv
```

**Process**:
1. Load gas_profile.csv (n_e vs r)
2. Convert n_e → ρ_gas using μ_e and clumping factor
3. Load stars_profile.csv (ρ_stars vs r)
4. Sum: ρ_baryon(r) = ρ_gas(r) + ρ_stars(r)
5. Abel projection: Σ(R) = 2 ∫_R^∞ ρ(r) r dr / √(r²-R²)
6. Output: R_kpc, Sigma_Msun_pc2

**Output**: Projected surface density Σ(R) in M☉/pc²

### Einstein Radius Prediction

**Script**: `concepts/cluster_lensing/cluster_lensing_analysis_real_sigma.py`

**Usage** (single cluster):
```bash
py -u concepts/cluster_lensing/cluster_lensing_analysis_real_sigma.py \
  --cluster MACSJ0416 \
  --z_lens 0.396 \
  --z_source 2.0 \
  --out out/cluster_lensing_real/macs0416/
```

**Usage** (all CLASH clusters):
```bash
py -u scripts/run_real_sigma_for_clash.py
```

**Process**:
1. Load Σ(R) from baryon profiles
2. Compute GR deflection angle α_GR(R)
3. Apply slip factor S(R) with universal parameters
4. Compute effective convergence κ_eff = S(R) × κ_GR
5. Find Einstein radius where κ_eff = 1 (or critical curve area)
6. Output: θ_E in arcsec

**Status**: ⚠️ Current predictions are ~180x too small - BUG TO FIX

### CLASH Model Processing (FITS → Einstein Radii)

**Script**: `concepts/cluster_lensing/process_clash_models.py`

**Usage**:
```bash
py -u concepts/cluster_lensing/process_clash_models.py
```

**Requires**:
- FITS files in `data/clash/hlsp/<cluster>/models/*kappa*.fits`
- FITS files in `data/clash/hlsp/<cluster>/models/*gamma*.fits`

**Status**: ❌ FITS files not downloaded yet (would need to fetch from MAST)

---

## 6. Known Issues

### Critical Issues (Blocking Progress)

1. **❌ Einstein Radius Predictions 180x Too Small**
   - **Symptom**: θ_E(predicted) = 0.19" vs θ_E(observed) = 35"
   - **Impact**: Cannot validate model until fixed
   - **Hypothesis**: Unit error in cosmology or deflection angle calculation
   - **Action**: Debug `cluster_lensing_analysis_real_sigma.py`
   - **Priority**: P0 - CRITICAL

2. **❌ NFW Parameters Not Extracted**
   - **Symptom**: No literature dark matter parameters available
   - **Impact**: Cannot compare our model vs traditional dark matter
   - **Action**: Extract from Umetsu+ 2016 Table 3
   - **Priority**: P0 - CRITICAL

### Medium Priority Issues

3. **⚠️ Missing Multi-z Lensing Constraints**
   - **Symptom**: Only have single z_source per cluster
   - **Impact**: Cannot perform MST degeneracy test (Editor Concern B)
   - **Action**: Search literature for multiple source redshifts
   - **Priority**: P1 - HIGH (needed for full response)

4. **⚠️ MACS1149 Processing Failure**
   - **Symptom**: Missing `profiles_realSigma.csv`
   - **Impact**: Only 2 training clusters available
   - **Action**: Re-run processing with verbose logging
   - **Priority**: P2 - MEDIUM

### Low Priority Issues

5. **⚠️ Clumping Factors Conservative**
   - **Symptom**: Many clusters use C≈1.5 estimate, not direct measurement
   - **Impact**: ~20% uncertainty in gas mass
   - **Action**: Search for direct clumping measurements in literature
   - **Priority**: P3 - LOW (acceptable for initial analysis)

---

## 7. Data Changelog

### 2025-01-10: Initial Documentation
- Created comprehensive README for all cluster data
- Documented 30 clusters with baryon profiles
- Documented 6 clusters with published Einstein radii
- Identified Umetsu+ 2016 as primary NFW parameter source
- Listed known issues and TODOs

### Previous (Undocumented)
- Downloaded CLASH cluster baryon profiles
- Downloaded Frontier Fields lensing constraints
- Downloaded Umetsu+ 2016 paper
- Ran initial predictions (with bug) on CLASH sample

---

## Quick Start Guide

### For New Users

**To add a new cluster**:

1. Create directory: `data/clusters/NEW_CLUSTER/`
2. Add 4 required CSV files (see Section 1 format specs)
3. Validate: `py -u rigor/scripts/check_cluster_data.py --cluster NEW_CLUSTER`
4. Add entry to this README's cluster table
5. Update git: `git add data/clusters/NEW_CLUSTER/ && git commit -m "Add NEW_CLUSTER data"`

**To run predictions**:

1. Fix the 180x bug first (see Issue #1)
2. Run: `py -u scripts/run_real_sigma_for_clash.py`
3. Check output: `out/cluster_lensing_real/*/summary_realSigma.json`
4. Compare to gold standard: `data/frontier/gold_standard/gold_standard_clusters.json`

**To extract NFW parameters**:

1. Open: `C:\Users\henry\Documents\GitHub\DensityDependentMetricModel\external_data\Umetsu_2016_ApJ_821_116.pdf`
2. Find Table 3 (weak lensing masses)
3. Extract M_200, c_200, z_lens for each cluster
4. Create: `data/literature/nfw_params.json` (see Section 3 format)
5. Update this README's changelog

---

## References

### Survey Papers

- **CLASH Overview**: Postman+ 2012, ApJS 199, 25
- **CLASH Weak Lensing**: Umetsu+ 2016, ApJ 821, 116 ⭐
- **CLASH Strong Lensing**: Zitrin+ 2015, ApJ 801, 44
- **Frontier Fields**: Lotz+ 2017, ApJ 837, 97

### X-ray & Baryon Data

- **ACCEPT Catalog**: Cavagnolo+ 2009, ApJS 182, 12
- **Perseus Clumping**: Urban+ 2014, A&A 558, A33
- **A1689 Clumping**: Eckert+ 2013, A&A 551, A22

### Individual Clusters

- **MACS0416**: Jauzac+ 2014 (MNRAS 443, 1549), Jauzac+ 2015 (MNRAS 452, 1437)
- **MACS0717**: Medezinski+ 2013 (ApJ 777, 43), Ma+ 2009 (ApJ 693, L56)
- **MACS1149**: Smith+ 2009 (ApJ 707, L163), Ebeling+ 2007 (ApJ 661, L33)
- **A1689**: Limousin+ 2007 (ApJ 668, 643), Broadhurst+ 2005 (ApJ 621, 53)

### Archive URLs

- **MAST CLASH**: https://archive.stsci.edu/prepds/clash/
- **MAST Frontier Fields**: https://archive.stsci.edu/prepds/frontier/
- **Chandra Data Archive**: https://cxc.harvard.edu/cda/

---

## Contact & Maintenance

**Questions?** See `concepts/cluster_lensing/DATA_INVENTORY.md` for detailed status report.

**Updates?** Always update:
1. This README's changelog (Section 7)
2. Cluster tables when adding/modifying data
3. Known issues list when bugs are found/fixed

**Git Commits**: 
- Use descriptive messages: "Add MACS0416 NFW parameters from Umetsu+2016"
- Reference this README in commit if updating data: "See data/clusters/CLUSTER_DATA_README.md"
