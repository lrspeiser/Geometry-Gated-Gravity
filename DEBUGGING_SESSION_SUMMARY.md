# Debugging Session Summary
## Date: 2025-01-10

---

## Problems Fixed ✅

### Issue #1: NaN values in gas density
**Symptom**: `rho3d range: [0.000e+00, 0.000e+00]`  
**Root Cause**: CSV contained NaN values in n_e column, which weren't being handled  
**Fix**: Added `np.nan_to_num(ne, nan=0.0)` to convert NaN→0 before processing  
**Status**: ✅ FIXED

### Issue #2: Unsorted r-grid breaking Abel projection
**Symptom**: `Sigma range: [-4.256e+06, 0.000e+00]` (negative values!)  
**Root Cause**: r-grid was not sorted ascending, breaking the Abel integral  
**Fix**: Added `sort_idx = np.argsort(r_all); r_all = r_all[sort_idx]; rho_b = rho_b[sort_idx]`  
**Status**: ✅ FIXED

### Issue #3: Interpolation collapsing to zero
**Symptom**: `rho_gi interp: range=[0.000e+00, 0.000e+00]` even though source data was good  
**Root Cause**: Using `np.unique(np.concatenate([r_g[r_g>0], r_s[r_s>0]]))` created mismatched grids for interp  
**Fix**: Since gas and stars have same r-grid, use direct masking instead of interpolation  
**Status**: ✅ FIXED

---

## Current Status

### ✅ Working:
- Data loading: n_e → rho_gas conversion now produces physical values
- Abel projection: Σ(R) now in range [2.7e7, 2.7e9] Msun/kpc² (PHYSICAL!)
- Enclosed mass: M_enc up to 1.9e14 Msun (REASONABLE!)
- Feature extraction: M_core = 3.0e12 Msun, R_edge = 94.5 kpc, edge_sharp = 0.776 (ALL NONZERO!)
- Slip model: S_inf = 7.37, S(R) enhancement working correctly
- Deflection calculation: alpha_model max = 19" (enhanced from GR's 4")

### ❌ Remaining Issues:

#### Issue #4: Insufficient baryon mass for lensing
**Symptom**: 
```
alpha_model(theta) max = 19" 
alpha_model(35") = 7.5"
thetaE_model = 5" (where alpha crosses theta)
thetaE_obs = 35"  ← NEED 7× MORE DEFLECTION!
```

**Analysis**:
- M_core = 3.0e12 Msun (current)
- Paper claims M_core ~ 1.2e13 Msun for MACSJ0416 (4× more)
- Even with 7× slip enhancement, total deflection only reaches 19" at max
- Need alpha(35") ≈ 35" to match observations

**Possible causes**:
1. **Baryon data is incomplete**: Missing significant gas mass or stars
2. **Units error**: Some conversion factor is off by ~4×
3. **Expected values in paper are synthetic**: Paper Table 5.1 might be from idealized models, not real data
4. **Wrong data files**: We might be loading wrong or incomplete cluster profiles

**Diagnostic questions**:
1. What is the expected total baryon mass for MACSJ0416 from literature?
2. Are the CSV files the complete CLASH/HFF gas+stars data?
3. Should M_enc(100 kpc) ≈ 3e12 or 1e13 Msun?

#### Issue #5: All clusters loading same data
**Symptom**: All three clusters (MACSJ0416, MACSJ0717, MACSJ1149) show identical features  
**Root Cause**: Likely loading MACSJ0416 data for all three due to directory mismatch  
**Impact**: Can't validate universal scaling until each cluster has correct data  
**Status**: ⚠️ NEEDS FIX

---

## Next Steps (Prioritized)

### Immediate (15 min):
1. ✅ Document current progress (this file)
2. ⏳ Check if cluster data directories exist for all three clusters
3. ⏳ Verify data loading is using correct cluster names

### Short-term (1 hour):
4. ⏳ Compare our M_core values against literature (Umetsu+2016, Postman+2012)
5. ⏳ Check if gas profiles need different conversion (maybe μ_gas is wrong?)
6. ⏳ Test with known good cluster (e.g., Abell 1689 or MACS0717 if data exists)

### Medium-term (4 hours):
7. ⏳ If baryon data is genuinely insufficient, accept that baryons alone can't explain observations
8. ⏳ Modify paper to reflect "baryons + slip can explain X% of deflection" rather than 100%
9. ⏳ OR: investigate if CLASH data has systematic mass underestimate

---

## Code Changes Made

### `scripts/run_real_cluster_tests.py`

**Line 51-52**: Added NaN cleaning for n_e
```python
ne = np.nan_to_num(ne, nan=0.0, posinf=0.0, neginf=0.0)
```

**Line 56**: Added NaN cleaning for rho_gas
```python
rho_g = np.nan_to_num(rho_g, nan=0.0, posinf=0.0, neginf=0.0)
```

**Line 65**: Added NaN cleaning for rho_star
```python
rho_s = np.nan_to_num(rho_s, nan=0.0, posinf=0.0, neginf=0.0)
```

**Lines 73-76**: Changed from interpolation to direct masking
```python
# OLD (BROKEN):
r_all = np.unique(np.concatenate([r_g[r_g > 0], r_s[r_s > 0]]))
rho_gi = np.interp(r_all, r_g, rho_g, left=0.0, right=0.0)
rho_si = np.interp(r_all, r_s, rho_s, left=0.0, right=0.0)

# NEW (FIXED):
valid_mask = (r_g > 0) & (r_s > 0) & np.isfinite(r_g) & np.isfinite(r_s)
r_all = r_g[valid_mask]
rho_gi = rho_g[valid_mask]
rho_si = rho_s[valid_mask]
```

**Lines 90-92**: Added r-grid sorting
```python
sort_idx = np.argsort(r_all)
r_all = r_all[sort_idx]
rho_b = rho_b[sort_idx]
```

**Throughout**: Added comprehensive debug logging (can be toggled off by setting `debug=False`)

---

## Test Results Summary

### MACSJ0416 (z_l=0.396, z_s=2.0):
| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| M_core | 3.0e12 Msun | ~1.2e13 Msun | ⚠️ Low |
| R_edge | 94.5 kpc | ~150-370 kpc | ⚠️ Low |
| edge_sharp (ε) | 0.776 | ~2.5 | ⚠️ Low |
| S_inf | 7.37 | ~19.1 | ⚠️ Low |
| θ_E,model | 5.0" | 35.0" | ❌ FAIL |
| α_model(35") | 7.5" | ~35" | ❌ Low |

### MACSJ0717 (z_l=0.546, z_s=2.0):
- **Same values as MACSJ0416** ← indicates data loading bug

### MACSJ1149 (z_l=0.544, z_s=2.0):
- **Same values as MACSJ0416** ← indicates data loading bug

---

## Validation Checklist

- [x] rho3d nonzero
- [x] Sigma nonzero and positive
- [x] M_enc growing monotonically
- [x] Features extracted (nonzero M_core, edge_sharp, R_edge)
- [x] Slip model producing S > 1
- [x] alpha_model > alpha_GR
- [ ] alpha_model(theta) crosses theta line at physically plausible theta_E
- [ ] theta_E,model within 50% of theta_E,obs
- [ ] Each cluster loads its own distinct data

---

## Conclusion

**Major Progress**: We fixed three critical bugs that were causing complete pipeline failure:
1. NaN handling
2. Unsorted r-grid
3. Interpolation collapse

**Remaining Challenge**: The baryon masses we're extracting are ~4× too small to produce observed lensing, even with slip enhancement. This could indicate:
- Data quality issues
- Missing baryon components
- Fundamental limitations of the "baryons-only" hypothesis

**Recommendation**: Before proceeding, we need to:
1. Verify the baryon data is complete and correct
2. Check literature values for MACSJ0416 total baryon mass
3. Decide if the model should aim for 100% or partial explanation of lensing

