# Data Specification for Research Roadmap Implementation

**Purpose**: This document specifies exactly what data we need to implement the editor's roadmap for major revisions.

**Target**: 20-30 clusters total (3 train, 10 validation, 17 test)

---

## Quick Summary - What We Need

For each cluster, we need:

### TIER 1 (Essential - Can implement roadmap with this alone):
1. **X-ray surface brightness profile** Σ(R) from Chandra/XMM
2. **Strong lensing constraints**: Einstein radius OR multiple image positions
3. **Redshift information**: z_lens and z_source

### TIER 2 (Highly Desired - Enables full validation):
4. **Measured baryon masses**: M_gas, M_stars (or we can derive from Σ)
5. **Multiple source redshifts** (for MST falsification test)
6. **Einstein radius measurements** from literature

### TIER 3 (Nice to have - For cross-probes):
7. Weak lensing shear profiles
8. X-ray hydrostatic mass profiles
9. Galaxy velocity dispersions

---

## Detailed Specifications

### 1. Cluster Sample Selection

**Criteria**:
- z_lens = 0.2 - 0.8 (strong lensing regime)
- M_200 > 5×10¹³ M_☉ (massive enough for strong lensing)
- HST imaging available (for strong lensing)
- Chandra/XMM X-ray data available (for baryons)

**Suggested samples**:
- **CLASH**: 25 massive clusters, all with HST + Chandra
- **RELICS**: 41 clusters with strong lensing
- **Frontier Fields**: 6 clusters with ultra-deep data

**Minimum needed**:
- **Training**: 3 clusters (MACS0416, MACS0717, MACS1149 - already have)
- **Validation**: 10 clusters (tune universal laws)
- **Test**: 17 clusters (final evaluation)

---

### 2. X-ray Data (TIER 1 - Essential)

#### What We Need:

**Format Option A: Surface Brightness Profile**
```
# File: cluster_name_xray_profile.dat
# Columns: R_kpc  Sigma_Msun_pc2  Sigma_err_Msun_pc2
10.0      1500.0     50.0
20.0      1200.0     40.0
30.0      950.0      35.0
...
500.0     15.0       5.0
```

**Format Option B: Gas Parameters (we'll compute Σ)**
```json
{
  "cluster_name": "MACS0416",
  "z": 0.396,
  "gas_model": "beta",
  "params": {
    "rho0": 0.01,      // central density [M_☉/pc³]
    "r_core": 100,     // core radius [kpc]
    "beta": 0.67       // beta parameter
  },
  "stellar_mass": {
    "M_stars_tot": 2e13,  // total stellar mass [M_☉]
    "r_eff": 30           // effective radius [kpc]
  }
}
```

**Where to get it**:
- Chandra Data Archive: https://cxc.harvard.edu/cda/
- Published papers (CLASH, RELICS, Frontier Fields collaborations)
- Pre-computed profiles from strong lensing papers

**Acceptable alternatives**:
- We can derive Σ(R) from published gas density ρ_gas(r) via Abel projection
- We can use published NFW or beta-model fits if profile data unavailable

#### Minimum Quality Requirements:
- Radial coverage: 10 kpc < R < 500 kpc
- At least 20 data points
- S/N > 3 per radial bin

---

### 3. Strong Lensing Data (TIER 1 - Essential)

#### What We Need (Pick ONE):

**Option A: Einstein Radius (Simplest)**
```json
{
  "cluster_name": "MACS0416",
  "z_lens": 0.396,
  "z_source": 2.09,
  "einstein_radius_arcsec": 28.5,
  "einstein_radius_err_arcsec": 2.0,
  "reference": "Jauzac+ 2014"
}
```

**Option B: Multiple Image Positions**
```
# File: cluster_name_images.dat
# Columns: image_id  RA_deg  Dec_deg  z_source
1a        64.0381  -24.0672  2.09
1b        64.0389  -24.0654  2.09
1c        64.0375  -24.0661  2.09
2a        64.0421  -24.0701  1.95
2b        64.0435  -24.0689  1.95
...
```

**Option C: Deflection Angle Profile (Most useful)**
```
# File: cluster_name_deflection.dat
# Columns: theta_arcsec  alpha_arcsec  alpha_err_arcsec  z_source
20.0      15.2     1.5    2.09
40.0      22.8     1.8    2.09
60.0      28.3     2.0    2.09
...
```

**Where to get it**:
- Strong lensing papers (CLASH, RELICS, HFF)
- SExtractor catalogs from HST imaging
- LENSTOOL/LENSTRONOMY models from literature

**Acceptable quality**:
- At least ONE of: θ_E, multiple images, or deflection profile
- Error bars / uncertainties
- Source redshifts (spectroscopic preferred, photometric OK)

---

### 4. Cluster Properties (TIER 1 - Essential)

#### Minimum Info Per Cluster:

```json
{
  "cluster_name": "MACS0416",
  "aliases": ["MACSJ0416.1-2403", "MACS J0416"],
  
  "redshifts": {
    "z_lens": 0.396,
    "z_source_primary": 2.09,  // Main lensed source
    "z_sources_additional": [1.95, 2.51]  // Other sources (TIER 2)
  },
  
  "coordinates": {
    "RA_deg": 64.0383,
    "Dec_deg": -24.0679
  },
  
  "morphology": {
    "type": "relaxed",  // or "merger", "minor_merger"
    "n_components": 1,
    "centroid_shift": 0.02  // if merger
  },
  
  "literature_masses": {
    "M_200_Msun": 1.2e15,  // from literature
    "M_200_err_Msun": 0.2e15,
    "method": "weak lensing",  // or "X-ray", "strong lensing"
    "reference": "Umetsu+ 2016"
  }
}
```

---

### 5. Multiple Source Redshifts (TIER 2 - For MST Test)

**Purpose**: Break MST degeneracy by testing consistency across source planes

**What We Need**:
For at least 5 clusters, we need lensing constraints at 2+ source redshifts:

```json
{
  "cluster_name": "MACS0416",
  "sources": [
    {
      "z_source": 2.09,
      "theta_E_arcsec": 28.5,
      "images": ["1a", "1b", "1c"]
    },
    {
      "z_source": 1.95,
      "theta_E_arcsec": 25.2,
      "images": ["2a", "2b"]
    },
    {
      "z_source": 2.51,
      "theta_E_arcsec": 32.1,
      "images": ["3a", "3b", "3c", "3d"]
    }
  ]
}
```

**Clusters with known multi-z**:
- MACS0416 (multiple arcs at z=1.9, 2.1, 2.6)
- Abell 2744 (z=1.2, 2.5, 6.2)
- MACS0717 (3 distinct source planes)
- RXJ1347 (z=1.8, 4.9)
- Abell 370 (multiple)

---

### 6. Baryon Mass Components (TIER 2)

**What We Need** (if available):

```json
{
  "cluster_name": "MACS0416",
  "baryon_masses": {
    "M_gas_Msun": 1.0e14,
    "M_gas_err_Msun": 0.1e14,
    "M_stars_Msun": 2.0e13,
    "M_stars_err_Msun": 0.3e13,
    "M_baryon_tot_Msun": 1.2e14,
    "R_measurement_kpc": 500  // radius where measured
  }
}
```

**Where to get it**:
- X-ray papers (M_gas from Chandra)
- Optical/NIR papers (M_stars from HST, JWST)
- Or we compute from Σ(R) profiles

---

### 7. Literature Dark Matter Models (For Comparison)

**What We Need**:
NFW halo parameters from published strong lensing models:

```json
{
  "cluster_name": "MACS0416",
  "dm_model": "NFW",
  "params": {
    "M_200_Msun": 1.15e15,
    "M_200_err_Msun": 0.15e15,
    "c_vir": 3.8,
    "c_vir_err": 0.5,
    "r_s_kpc": 420,
    "r_s_err_kpc": 50
  },
  "fit_quality": {
    "chi2": 15.2,
    "dof": 42,
    "rms_arcsec": 0.18
  },
  "reference": "Jauzac+ 2015"
}
```

**Where to get it**:
- LENSTOOL/LENSTRONOMY papers
- CLASH/RELICS mass model papers
- Frontier Fields official models

---

## File Structure Suggestion

```
data/
├── clusters/
│   ├── MACS0416/
│   │   ├── cluster_properties.json
│   │   ├── xray_profile.dat
│   │   ├── lensing_constraints.json
│   │   ├── multiple_images.dat (if available)
│   │   └── literature_dm_model.json (if available)
│   │
│   ├── MACS0717/
│   │   ├── cluster_properties.json
│   │   ├── xray_profile.dat
│   │   └── ...
│   │
│   └── [18 more clusters]/
│
└── cluster_catalog.json  // Master list
```

---

## Minimum Viable Dataset

**To implement Phase 1 (Out-of-sample validation)**:

### Already Have (Train):
1. ✅ MACS0416
2. ✅ MACS0717
3. ✅ MACS1149

### Need (Validation - 10 clusters):
Pick 10 from CLASH/RELICS with:
- Published X-ray profiles
- Published Einstein radii
- Clean, single-component (relaxed)

**Suggested**:
- Abell 2744
- RXJ1347
- MACS1206
- MACS0329
- MACS1149
- MACS2129
- Abell 370
- MACS0429
- MACS1423
- RXJ2248

### Need (Test - 17 clusters):
Remaining CLASH clusters + RELICS additions

---

## Data Sources & Where to Find Them

### 1. CLASH (Cluster Lensing And Supernova survey with Hubble)
**Paper**: Postman+ 2012
**Data**: https://archive.stsci.edu/prepds/clash/

**Available**:
- 25 massive clusters
- HST 16-band imaging
- Chandra X-ray
- Strong + weak lensing
- Published mass models

**Key Papers**:
- Merten+ 2015 (weak lensing masses)
- Umetsu+ 2016 (lensing + X-ray)
- Zitrin+ 2015 (strong lensing models)

### 2. RELICS (Reionization Lensing Cluster Survey)
**Paper**: Coe+ 2019
**Data**: https://relics.stsci.edu/

**Available**:
- 41 massive clusters
- HST WFC3/IR imaging
- Strong lensing catalogs
- Published models

### 3. Frontier Fields
**Paper**: Lotz+ 2017
**Data**: https://frontierfields.org/

**Available**:
- 6 clusters with ultra-deep data
- Multiple independent mass models
- Extensive multi-z lensing
- Best for MST test

### 4. Chandra Data Archive
**URL**: https://cxc.harvard.edu/cda/

**Search for**:
- Cluster name
- Filter: Exposure time > 20 ks
- Download: Event files OR published profiles

### 5. Literature Compilations
- Ettori+ 2019 (Hydro masses)
- Applegate+ 2014 (Weak lensing compilation)
- CCCP (Canadian Cluster Comparison Project)

---

## What I Can Do vs What You Need to Provide

### I Can Generate/Compute:
✅ Σ(R) from gas density ρ_gas(r) via Abel projection
✅ M_enc(R) from Σ(R) integration
✅ R_edge, edge_sharp, M_core from Σ(R)
✅ GR deflection α_GR from Σ(R)
✅ Synthetic test cases for validation

### You Need to Provide:
❌ Raw X-ray data OR published Σ(R) profiles
❌ Lensing constraints (θ_E, images, or α_obs)
❌ Cluster redshifts (z_lens, z_source)
❌ Published NFW parameters (for DM comparison)

---

## Data Format Templates

### Template 1: Minimal CSV Format

**xray_profile.csv**:
```csv
R_kpc,Sigma_Msun_pc2,Sigma_err
10,1500,50
20,1200,40
30,950,35
...
```

**lensing_data.csv**:
```csv
theta_arcsec,alpha_arcsec,alpha_err,z_source
20,15.2,1.5,2.09
40,22.8,1.8,2.09
60,28.3,2.0,2.09
```

**cluster_info.csv**:
```csv
cluster_name,z_lens,z_source,theta_E_arcsec,theta_E_err
MACS0416,0.396,2.09,28.5,2.0
MACS0717,0.548,2.83,45.2,3.5
...
```

### Template 2: JSON Format (Preferred)

```json
{
  "cluster_name": "MACS0416",
  "z_lens": 0.396,
  "z_source": 2.09,
  
  "xray_profile": {
    "R_kpc": [10, 20, 30, ..., 500],
    "Sigma_Msun_pc2": [1500, 1200, 950, ..., 15],
    "Sigma_err": [50, 40, 35, ..., 5]
  },
  
  "lensing": {
    "einstein_radius": {"value": 28.5, "error": 2.0, "unit": "arcsec"},
    "deflection_profile": {
      "theta_arcsec": [20, 40, 60, 80, 100],
      "alpha_arcsec": [15.2, 22.8, 28.3, 31.5, 33.2],
      "alpha_err": [1.5, 1.8, 2.0, 2.2, 2.5]
    }
  },
  
  "literature_dm": {
    "M_200_Msun": 1.15e15,
    "c_vir": 3.8,
    "r_s_kpc": 420,
    "reference": "Jauzac+ 2015"
  }
}
```

---

## Priority Order

### Phase 1 (Weeks 1-2): MINIMAL DATA
**Goal**: Bootstrap parameter uncertainties + DM comparison framework

**Need**:
- Our existing 3 clusters (already have)
- Their published NFW parameters (literature)

### Phase 2 (Weeks 3-4): VALIDATION DATA
**Goal**: Out-of-sample validation on 10 clusters

**Need**:
- 10 validation clusters with Σ(R) + θ_E
- Can be simple (just Einstein radius is OK)

### Phase 3 (Weeks 5-8): TEST DATA
**Goal**: Final evaluation on 17 test clusters

**Need**:
- 17 test clusters with full constraints
- At least 5 with multi-z for MST test

---

## How to Send Data to Me

### Option 1: File Upload
- ZIP file with folder structure above
- I'll parse and integrate into pipeline

### Option 2: Data Links
- URLs to MAST/Chandra archives
- I can download and process

### Option 3: Literature References
- "Use CLASH sample, Merten+ 2015 Table 3"
- I'll extract from published tables

### Option 4: Generate from Models
- "Use MACS0416 from Jauzac+ 2014, beta model with..."
- I'll generate synthetic data

---

## Questions to Answer

Before gathering data, please clarify:

1. **Do you have access to any specific datasets already?**
   - CLASH data?
   - RELICS models?
   - Frontier Fields?

2. **Can you point me to specific papers for the 3 training clusters?**
   - MACS0416 reference?
   - MACS0717 reference?
   - MACS1149 reference?

3. **What level of access do you have?**
   - Can download Chandra data?
   - Can access HST archives?
   - Or rely on published tables/figures?

4. **Timeline preference?**
   - Get minimal data first, implement iteratively?
   - Wait until full dataset assembled?

---

## Summary: What To Get First

**Immediate (Next Step)**:
1. Published NFW parameters for MACS0416, MACS0717, MACS1149
   - Need M_200, c_vir, r_s from strong lensing papers
   - For dark matter comparison

2. Validation sample list
   - Pick 10 clusters from CLASH/RELICS
   - Need Einstein radii (can get from papers)

**This Week**:
3. X-ray profiles for 10 validation clusters
   - Can be from published figures/tables
   - Will digitize if needed

**Next Week**:
4. Multi-z lensing for 5 clusters (MST test)
   - MACS0416, Abell 2744, MACS0717, RXJ1347, Abell 370

---

**Ready to proceed?** Let me know what data you have access to and I'll adapt the implementation accordingly!
