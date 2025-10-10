# Data Acquisition Task Tracker

**Last Updated**: 2025-01-10  
**Purpose**: Track progress on filling data gaps identified in gap analysis  
**Reference**: See `CLUSTER_DATA_README.md` for schemas and templates

---

## Priority 0 (CRITICAL BLOCKERS)

### ❌ Task 1: Fix 180× Einstein Radius Scale Bug
**Status**: NOT STARTED  
**Blocking**: All validation and comparison work  
**Assigned to**: Debug cluster_lensing_analysis_real_sigma.py

**Issue**: 
- Predicted θ_E = 0.19 arcsec
- Observed θ_E = 35.0 arcsec  
- Factor: ~180× too small

**Debug Checklist**:
- [ ] Verify critical surface density Σ_crit calculation
  - Formula: Σ_crit = (c²/4πG) × (D_s / D_d D_ds)
  - Check: D_d, D_s, D_ds angular diameter distances
  - Cosmology: H_0 = 70, Ω_m = 0.3, Ω_Λ = 0.7 (match literature)
- [ ] Check convergence κ = Σ / Σ_crit units
  - Σ in M☉/pc² 
  - Σ_crit should be in same units
- [ ] Test on analytic profiles (SIS, NFW)
  - SIS: θ_E = 4π (σ_v/c)² (D_ds/D_s)
  - NFW: Compare to known cluster (e.g. A1689)
- [ ] Verify slip factor S(R) normalization
  - Check if S_∞ needs amplitude adjustment
  - Current: S_∞ ~ 1 + small correction, may need S_∞ >> 1?
- [ ] Check arcsec ↔ kpc conversion
  - θ = R / D_d, where D_d in proper units
  - 1 arcsec at z=0.4 ≈ 5.3 kpc

**Validation Test**:
```python
# Quick test with MACS0416
# Known: θ_E = 35" at z_s=2.0, z_d=0.396
# If we get ~0.19", there's a ~180x error somewhere
```

**Files to check**:
- `concepts/cluster_lensing/cluster_lensing_analysis_real_sigma.py`
- Lines computing: Σ_crit, κ, deflection angle α, Einstein radius

**Expected fix**: Unit conversion error or missing normalization constant

---

### ❌ Task 2: Extract NFW Parameters from Umetsu+ 2016
**Status**: NOT STARTED  
**Blocking**: Dark matter comparison framework  
**Priority**: P0 - CRITICAL

**Input**: 
```
C:\Users\henry\Documents\GitHub\DensityDependentMetricModel\external_data\Umetsu_2016_ApJ_821_116.pdf
```

**Output**:
```
C:\Users\henry\dev\GravityCalculator\data\literature\nfw_params.json
```

**What to Extract** (Table 3 or 4):

For each of 20-25 CLASH clusters:
- cluster_name (match our naming convention)
- z_lens
- M_200 (in M☉)
- M_200_err_lower, M_200_err_upper (or symmetric σ)
- c_200 (concentration)
- c_200_err
- r_200 (virial radius in kpc, if listed)
- method (e.g., "weak_lensing", "strong+weak")

**Priority Clusters** (training set):
1. ✅ MACS0416 (MACSJ0416.1-2403)
2. ✅ MACS0717 (MACSJ0717.5+3745)
3. ✅ MACS1149 (MACSJ1149.5+2223)

**Schema** (from spec):
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

**Compute r_s**:
```
r_s = r_200 / c_200
```

If r_200 not listed, compute from M_200:
```
r_200 = (3 M_200 / 4π × 200 × ρ_crit)^(1/3)
```
where ρ_crit = ρ_crit(z) from cosmology.

**Steps**:
- [ ] Open PDF, locate Table 3
- [ ] Extract data for all 20-25 clusters
- [ ] Format as JSON
- [ ] Validate against any cross-checks (Merten+2015)
- [ ] Create `data/literature/nfw_params.json`
- [ ] Create `data/literature/nfw_params_sources.md` with references
- [ ] Update `CLUSTER_DATA_README.md` changelog

---

## Priority 1 (HIGH - Needed for Full Response)

### ⏳ Task 3: Assemble Tier-1 Data for 10 Validation Clusters
**Status**: PARTIALLY COMPLETE  
**Depends on**: Task 1 (bug fix) for validation  
**Target**: 10 CLASH clusters beyond our 3 training clusters

**Tier-1 Requirements per Cluster**:
- [x] Σ(R) baryon profile (we have for 30 clusters)
- [ ] θ_E at stated z_source (need to consolidate from multiple sources)
- [x] z_lens, z_source (we have most)

**Candidate Validation Clusters** (from CLASH sample):

| Cluster | z_lens | Baryon Data | θ_E Source | Status |
|---------|--------|-------------|------------|--------|
| Abell 209 | 0.206 | ✅ | ⏳ Zitrin+2015 | Need θ_E |
| Abell 383 | 0.187 | ✅ | ⏳ Zitrin+2015 | Need θ_E |
| Abell 611 | 0.288 | ✅ | ⏳ Zitrin+2015 | Need θ_E |
| MACSJ0329 | 0.450 | ✅ | ⏳ Zitrin+2015 | Need θ_E |
| MACSJ0429 | 0.399 | ✅ | ⏳ Zitrin+2015 | Need θ_E |
| MACSJ1206 | 0.440 | ✅ | ⏳ Zitrin+2015 | Need θ_E |
| MACSJ1311 | 0.494 | ✅ | ⏳ Zitrin+2015 | Need θ_E |
| MACSJ1423 | 0.545 | ✅ | ⏳ Zitrin+2015 | Need θ_E |
| RXJ1532 | 0.345 | ✅ | ⏳ Zitrin+2015 | Need θ_E |
| RXJ2248 | 0.348 | ✅ | ⏳ Zitrin+2015 | Need θ_E |

**Primary Source**: Zitrin+ 2015 (ApJ 801, 44)
- Title: "CLASH: The Concentration-Mass Relation of Galaxy Clusters"
- Should have strong lensing Einstein radii for all CLASH clusters
- Check if we have PDF or can access via ADS

**Actions**:
- [ ] Locate Zitrin+ 2015 paper (check data/, docs/, or download)
- [ ] Extract θ_E for validation clusters (Table with Einstein radii)
- [ ] Create per-cluster `lensing_constraints.json` files
- [ ] Or consolidate into `data/clash/einstein_radii_zitrin2015.csv`
- [ ] Verify z_source used in Zitrin models (likely z_s=2.0 nominal)

**Format** (simple CSV):
```csv
cluster,z_lens,z_source,einstein_radius_arcsec,einstein_radius_err_arcsec,reference
Abell_209,0.206,2.0,25.0,2.0,Zitrin+2015
Abell_383,0.187,2.0,30.0,2.0,Zitrin+2015
...
```

---

### ⏳ Task 4: Multi-z Lensing Constraints (MST Test)
**Status**: NOT STARTED  
**Priority**: P1 - HIGH (needed for Editor Concern B)  
**Target**: 5 clusters with ≥2 source redshifts each

**Clusters** (from spec):
1. **MACS0416** - HFF Tier 1, many sources
2. **Abell 2744** - HFF Tier 1, complex merger, multi-z known
3. **MACS0717** - HFF Tier 2, largest lens, merger
4. **RXJ1347** - HFF Tier 2, X-ray bright
5. **Abell 370** - HFF Tier 1, well-studied

**What to Find**:

For each cluster, search literature for:
- Multiple background sources at different redshifts
- Einstein radii or multiple image positions for each source
- Spectroscopic or photo-z for each arc/source

**Primary Sources**:
- HFF team papers (Caminha, CATS, Williams models)
- Frontier Fields data releases (MAST)
- Individual cluster papers

**Format** (per cluster):
```json
{
  "cluster_name": "MACS0416",
  "z_lens": 0.396,
  "sources": [
    {
      "source_id": "arc_1",
      "z_source": 2.09,
      "z_source_err": 0.05,
      "z_method": "spectroscopic",
      "theta_E_arcsec": 35.0,
      "theta_E_err": 1.5,
      "images": ["1a", "1b", "1c"],
      "reference": "Caminha+ 2017"
    },
    {
      "source_id": "arc_2",
      "z_source": 1.95,
      "z_source_err": 0.1,
      "z_method": "photometric",
      "theta_E_arcsec": 32.0,
      "theta_E_err": 2.0,
      "images": ["2a", "2b"],
      "reference": "Caminha+ 2017"
    }
  ]
}
```

**Output Location**:
```
data/frontier/multi_z_constraints/
├── macs0416_multi_z.json
├── abell2744_multi_z.json
├── macs0717_multi_z.json
├── rxj1347_multi_z.json
└── abell370_multi_z.json
```

**Search Strategy**:
- [ ] Check HFF MAST archive for multi-z catalogs
- [ ] Read Caminha+ 2017 (if available) for MACS0416
- [ ] Read Jauzac+ papers for Abell 2744, MACS0717
- [ ] Check if `data/frontier/hlsp/` README files mention multi-z
- [ ] Use ADS to find "multiple sources" + cluster name

---

## Priority 2 (MEDIUM)

### ⏳ Task 5: Fix MACS1149 Processing Failure
**Status**: NOT STARTED  
**Issue**: Missing `profiles_realSigma.csv` in output

**Debug Steps**:
- [ ] Check if baryon profiles exist: `data/clusters/MACSJ1149/`
- [ ] Re-run with verbose logging:
  ```bash
  py -u concepts/cluster_lensing/cluster_lensing_analysis_real_sigma.py \
    --cluster MACSJ1149 --z_lens 0.544 --z_source 2.0 \
    --out out/cluster_lensing_real/macs1149/ --verbose
  ```
- [ ] Check for error messages in stdout/stderr
- [ ] Compare with successful MACS0416 run
- [ ] Fix any code issues
- [ ] Verify output: `summary_realSigma.json` and `profiles_realSigma.csv`

---

### ⏳ Task 6: Populate Baryon Mass Summaries
**Status**: NOT STARTED  
**Nice-to-have**: Derived quantities for quick reference

**Compute for Each Cluster**:
- M_gas(<500 kpc)
- M_stars(<500 kpc)  
- M_baryon = M_gas + M_stars
- R_edge (steepest gradient in Σ)
- ε (edge sharpness)
- M_core = M_baryon(<R_edge)

**Output**:
```json
{
  "cluster_name": "MACSJ0416",
  "z_lens": 0.396,
  "baryon_masses": {
    "M_gas_500kpc_Msun": 1.0e14,
    "M_gas_500kpc_err": 0.1e14,
    "M_stars_500kpc_Msun": 2.0e13,
    "M_stars_500kpc_err": 0.3e13,
    "M_baryon_500kpc_Msun": 1.2e14,
    "R_edge_kpc": 150,
    "edge_sharpness": 0.45,
    "M_core_Msun": 5.0e13
  }
}
```

**Script to Create**:
```bash
py -u concepts/cluster_lensing/compute_baryon_summary.py \
  --cluster MACSJ0416 \
  --out data/summaries/baryon_masses/MACSJ0416.json
```

---

## Priority 3 (LOW - Optional)

### ⏳ Task 7: Weak Lensing Shear Profiles
**Status**: NOT STARTED  
**Source**: Umetsu+ 2016, Merten+ 2015

**What to Extract**:
- Tangential shear γ_t(R) vs radius
- Radial range: typically 0.3-3 Mpc
- Use for cross-probe consistency check

**Format**: CSV per cluster
```csv
R_Mpc,gamma_t,gamma_t_err
0.3,0.15,0.02
0.5,0.12,0.015
1.0,0.08,0.01
...
```

**Priority**: LOW (can skip for initial analysis)

---

### ⏳ Task 8: Hydrostatic Mass Profiles
**Status**: NOT STARTED  
**Source**: ACCEPT catalog, individual papers

**What to Extract**:
- M_hydro(r) from X-ray temperature + gas density
- Compare to our baryon-only + slip model
- Check for hydrostatic bias

**Priority**: LOW (optional cross-check)

---

## Progress Tracker

### Summary Table

| Priority | Task | Status | Blocker | Target Date |
|----------|------|--------|---------|-------------|
| P0 | Fix 180× bug | ❌ NOT STARTED | None | ASAP |
| P0 | Extract NFW params | ❌ NOT STARTED | None | This week |
| P1 | Tier-1 for 10 clusters | ⏳ 30% (have baryons) | Task 1 | Week 2 |
| P1 | Multi-z constraints | ❌ NOT STARTED | Task 1 | Week 3 |
| P2 | Fix MACS1149 | ❌ NOT STARTED | Task 1 | Week 2 |
| P2 | Baryon mass summaries | ❌ NOT STARTED | None | Week 3 |
| P3 | Weak lensing profiles | ❌ NOT STARTED | None | Later |
| P3 | Hydrostatic masses | ❌ NOT STARTED | None | Later |

### Completion Metrics

**Tier-1 Minimum (for 20 clusters)**:
- [ ] 0/20 clusters with Σ(R) ✅ (actually 30/30 done!)
- [ ] 0/20 clusters with θ_E at stated z_source
- [ ] 0/20 clusters with z_lens, z_source ✅ (mostly done)
- **Overall**: ~30% (have baryons, need lensing)

**Tier-2 Desired**:
- [ ] 0/5 clusters with multi-z constraints (MST test)
- [ ] 0/20 clusters with NFW comparison parameters
- [ ] 0/20 clusters with baryon mass summaries
- **Overall**: ~0%

**Quality Gates**:
- [ ] θ_E scale bug fixed (180× error)
- [ ] At least 10 validation clusters ready
- [ ] At least 5 clusters with multi-z for MST test
- [ ] NFW parameters for all CLASH clusters

---

## Quick Actions (This Week)

### Monday-Tuesday:
1. ✅ Fix θ_E scale bug (P0)
   - Debug Σ_crit, angular diameter distances, κ calculation
   - Test on analytic profiles (SIS, NFW)
   - Validate against MACS0416 θ_E = 35"

### Wednesday:
2. ✅ Extract NFW parameters from Umetsu+ 2016 (P0)
   - Read Table 3
   - Extract M_200, c_200 for 20-25 CLASH clusters
   - Create `data/literature/nfw_params.json`

### Thursday-Friday:
3. ✅ Locate Zitrin+ 2015 paper
   - Extract θ_E for 10 validation clusters
   - Create consolidated Einstein radii CSV

4. ✅ Start multi-z search for MACS0416, Abell 2744
   - Check HFF papers (Caminha, CATS)
   - Create multi-z JSON files

---

## References

**Data Schemas**: See `CLUSTER_DATA_README.md` Section 1-3  
**File Templates**: See `DATA_SPECIFICATION.md` Templates  
**Current Inventory**: See `DATA_INVENTORY.md`  
**Gap Analysis**: This document (Task 1-8)

---

## Update Log

### 2025-01-10: Initial Task List
- Created from gap analysis
- Identified 8 tasks with priorities P0-P3
- Set P0 blockers: θ_E bug fix + NFW extraction
- Defined Tier-1 minimum for 20 clusters
- Created quick action plan for this week
