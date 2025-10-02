# Test 1: Velocity Dispersion Gating

**Status:** ✅ Core implementation complete, ready for fitting  
**Date:** October 2, 2025  
**Hypothesis:** Deep potential wells (high σ) amplify geometry-gated tail  

---

## Model Formula

```
fX = x² / (a - b·Σ̂ - d·|∇ln Σ| - e·(σ/σ₀)^α)
```

**Parameters:**
- **Fixed from O2:** a = 0.6687, b = 0.1401, d = 0.0871
- **NEW to fit:** e (gate weight), α (power index)
- **Reference:** σ₀ = 100 km/s (typical galaxy)

**Expected amplification:**
- Galaxies (σ ~ 100 km/s): (100/100)^1.5 = 1× (no change)
- Clusters (σ ~ 1000 km/s): (1000/100)^1.5 = 31.6× (large boost)

---

## Implementation Status

### ✅ Completed:
1. **velocity_dispersion_model.py** - Core functions
   - `compute_velocity_dispersion_from_temperature(kT_keV)` - Convert kT → σ
   - `compute_velocity_dispersion_from_virial(M, R)` - For SPARC galaxies
   - `fX_ratio_curv_sigma(params, ...)` - 5-parameter model
   - `load_cluster_temperature_profile(cluster_name)` - Data loading
   - Unit tests: ✅ All passing

### ⏳ Next Steps:
2. **Prepare cluster data** - Extract median σ for 4 clusters
3. **Fit (e, α) parameters** - Optimize on cluster lensing
4. **Validate on SPARC** - Check if galaxy APE stays < 0.30
5. **Generate results** - CSV tables + diagnostic plots

---

## Test Results (Unit Tests)

### Velocity Dispersion Conversions

**X-ray Temperature → σ:**
```
kT =  1.0 keV  →  σ =   403 km/s
kT =  5.0 keV  →  σ =   901 km/s
kT = 10.0 keV  →  σ =  1274 km/s  ← Typical massive cluster
kT = 15.0 keV  →  σ =  1560 km/s
```

**Virial Mass → σ (at R=100 kpc):**
```
M_200 = 1e10 Msun  →  σ =   21 km/s  (dwarf)
M_200 = 1e11 Msun  →  σ =   66 km/s  (small galaxy)
M_200 = 1e12 Msun  →  σ =  207 km/s  (Milky Way-like)
M_200 = 1e13 Msun  →  σ =  656 km/s  (group)
```

### Amplification Test

**Conditions:** x=5.0, Σ̂=-1.0, |∇ln Σ|=0.5  
**Parameters:** e=0.05, α=1.5

```
σ (km/s)    fX (baseline)    fX (σ-gated)    Amplification
   50           32.7             33.4             1.02×
  100           32.7             35.0             1.07×
  200           32.7             40.1             1.23×
  500           32.7            121.2             3.71×
```

⚠️ **Note:** At σ > 1000 km/s, model becomes unstable (denominator → negative).
This is expected and will be handled by parameter fitting to keep e small enough.

---

## Data Available

### Clusters (4):
1. **ABELL_1689** - Massive lensing cluster
   - z = 0.183
   - Temperature profile: 61 points, 8-909 kpc
   - Observed θ_E = 47" (strong lensing)

2. **A2029** - Massive relaxed cluster  
   - z = 0.077
   - Temperature profile: available
   - Observed θ_E = 28"

3. **A478** - Intermediate-mass cluster
   - z = 0.088
   - Temperature profile: available
   - Observed θ_E = 31"

4. **ABELL_0426** (Perseus) - Cool-core cluster
   - z = 0.018
   - Temperature profile: available
   - No strong lensing arcs (too nearby/disturbed)

### SPARC Galaxies (120):
- Rotation curves available
- Virial masses (estimated) or use typical σ ~ 100-150 km/s
- Baseline O2 performance: median APE = 0.242

---

## Success Criteria

**✅ PASS:** 
- Cluster Einstein radii within 30% (all 3)
- Galaxy median APE < 0.30 (max 6-point degradation)
- Parameters physically reasonable: e < 0.2, 0.5 < α < 3.0

**⚠️ PARTIAL:**
- Clusters fit, but galaxies degrade (APE > 0.30)
- Suggests two-regime model needed

**❌ FAIL:**
- Cannot fit clusters within factor 2 even with free (e, α)
- Move to Test 2 (Hot Gas Fraction)

---

## Next Commands

### 1. Prepare cluster data
```bash
python prepare_cluster_data.py
# Output: cluster_sigma_data.csv
```

### 2. Fit (e, α) parameters
```bash
python fit_sigma_model.py
# Output: best_params.json
```

### 3. Validate on SPARC
```bash
python validate_on_sparc.py
# Output: galaxy_validation_results.csv
```

### 4. Generate diagnostic plots
```bash
python generate_diagnostics.py
# Output: figures/*.png
```

---

## Files in this folder

- `velocity_dispersion_model.py` - ✅ Core implementation (269 lines)
- `README.md` - This file
- `prepare_cluster_data.py` - ⏳ To be created
- `fit_sigma_model.py` - ⏳ To be created
- `validate_on_sparc.py` - ⏳ To be created
- `generate_diagnostics.py` - ⏳ To be created

---

## Expected Timeline

**Total:** 4-6 hours

1. Data preparation: 30 min
2. Parameter fitting: 1-2 hours (optimization)
3. SPARC validation: 1 hour
4. Diagnostic plots: 1 hour
5. Analysis + writeup: 1-2 hours

---

## Physics Notes

### Why velocity dispersion?

1. **Natural scale:** σ measures depth of potential well
   - Directly related to virial temperature: kT ∝ μ m_p σ²
   - Single-valued for relaxed systems

2. **Clear galaxy-cluster separation:**
   - Galaxies: σ ~ 50-200 km/s
   - Groups: σ ~ 300-500 km/s
   - Clusters: σ ~ 800-1500 km/s
   - Factor of 5-10 difference

3. **Theoretical motivation:**
   - σ ∝ sqrt(GM/R) relates to gravitational binding
   - Deeper wells → stronger geometry-gating response
   - Analogous to temperature-dependent phase transitions

### Potential issues:

1. **Stability:** Large σ can make denominator negative
   - Solution: Constrain e to keep denom > 0 for all systems
   - Typical constraint: e < 0.1 should be safe

2. **Galaxy σ estimates:** SPARC doesn't have direct σ measurements
   - Solution: Use typical σ ~ 100-150 km/s for all
   - Sensitivity: Test with σ = 80, 100, 150, 200 km/s

3. **Radial dependence:** σ may vary with radius in clusters
   - Solution: Use median σ from 10-500 kpc range
   - Avoids central AGN and outer accretion regions

---

## Contact

For questions or issues, see main project README or GitHub issues.
