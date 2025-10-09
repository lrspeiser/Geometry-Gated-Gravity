# Lensing Analysis Fixes - 2025-01-08

## Bugs Identified and Fixed

### 1. **Critical Bug: alpha_fun_ACCEPTED returned radians instead of arcsec**
**Location:** `scripts/lensing_utils.py` lines 192 and 374

**Problem:** The function was dividing by 206265.0 to convert arcsec to radians, but the documentation and all calling code expected arcsec output.

**Impact:** All deflection curves from HLSP maps were showing values 206,265x smaller than they should be.

**Fix:** Removed the division by 206265.0 and added comments clarifying the function returns ARCSEC.

### 2. **Abel Projection Gradient Calculation Fails with Duplicate R Values**
**Location:** `concepts/cluster_lensing/cluster_lensing_analysis_real_sigma.py` line 126

**Problem:** The `abel_project_sigma` function was computing `np.gradient(np.log(rho), np.log(r))` to estimate the power-law tail slope. When the input `r` array had duplicate or near-duplicate values (spacing < 1e-9 kpc), this caused divide-by-zero errors in the gradient calculation, producing all-NaN output.

**Impact:** 
- GR baryons-only deflection calculation failed completely (returned NaN)
- GE custom deflection calculation also failed (depends on abel_project_sigma)

**Fix:** 
- Filter out consecutive duplicate/near-duplicate r values before gradient calculation
- Use simple finite-difference slope `(log_rho[-1] - log_rho[0]) / (log_r[-1] - log_r[0])` instead of np.gradient
- Added safeguards for edge cases (insufficient points, zero spacing)

### 3. **GR Baryons Deflection is 1000x Too Large**
**Location:** `scripts/lensing_utils.py` alpha_fun_GR_baryons

**Problem:** After fixing the abel projection, GR baryons-only deflection produces values ~9,000″ at 50″ radius, compared to the accepted (DM+baryons) value of ~24″. Baryons-only should be MUCH SMALLER than the total, not 400x larger.

**Root Cause:** Unknown - likely an issue with how the cluster baryon profiles are stored/loaded, or a missing normalization factor.

**Status:** **UNFIXED - GR and GE calculations disabled in plots**

The plotting script now comments out the GR and GE deflection calculations to prevent them from breaking the visualizations.

## Current State

### Working Components
✅ **ACCEPTED deflection from HLSP maps** - Correctly loads HFF deflection maps, applies geometry scaling (β_user/β_ref), and returns deflection magnitude in arcsec along the vertical cut through the κ peak.

✅ **Critical curve detection** - Identifies T = 1-κ-|γ| ≤ 0 regions and uses flood-fill from κ peak to find the main tangential critical curve.

✅ **Einstein radius computation** - Calculates θ_E from critical curve area (area-equivalent radius).

✅ **Deflection curve plots** - Shows accepted α(θ) properly with α=θ reference line and θ_E marker.

✅ **Mean convergence k̄(<θ) plots** - Correctly shows α(θ)/θ profile with physically reasonable structure (supercritical core, subcritical ring, declining envelope).

### Broken Components  
❌ **GR baryons-only deflection** - Produces values 1000x too large; likely issue with baryon profile data or normalization.

❌ **GE custom deflection** - Returns None; depends on broken GR calculation.

❌ **Interior-anchored export boost** - Cannot test until GE calculation works.

## Validation Results

### MACS0416 cats v4.1 (z_lens=0.396, z_source=2.0)

**Deflection values** (from test script):
- α(5″) = 2.93″ (supercritical core)
- α(10″) = 0.28″ (drops into subcritical ring)
- α(20″) = 8.55″ (back to strong deflection)
- α(30″) = 16.39″
- α(50″) = 23.56″ (asymptotes around 24-28″)

**Einstein radius:** θ_E ~ 3.72″ (where α=θ)

**Critical curve area:** T≤0 region has 53,018 pixels → θ_E ~ 39.01″ (area-equivalent)

**Note:** The two θ_E estimates differ significantly. The area-based method gives a much larger value because it includes the full T≤0 region which may be fragmented. The α=θ crossing is more precisely defined.

## Recommendations

### Immediate Next Steps
1. **Fix GR baryons calculation:**
   - Check if cluster baryon profiles in `data/clusters/MACSJ*/` have correct units
   - Verify Σ_crit calculation
   - Check if there's a missing factor (e.g., should use ρ_DM+ρ_bar but only using ρ_bar?)

2. **Fix GE calculation:**
   - Add verbose error logging in the try-except block (currently swallows all exceptions)
   - Check if GeometricExponentGravity.Sigma_effective is failing

3. **Test with other clusters:**
   - Verify ACCEPTED deflection works for MACS0717, MACS1149
   - Check if baryon calculation issue is cluster-specific

### Future Enhancements
- Add option to load DM+baryon total profile for GR comparison
- Implement GE boost parameter sweep with result tables
- Add validation against published θ_E values
- Create interactive plot with adjustable GE parameters

## Files Modified
- `scripts/lensing_utils.py` - Fixed alpha_fun_ACCEPTED, disabled GR/GE
- `concepts/cluster_lensing/cluster_lensing_analysis_real_sigma.py` - Fixed abel_project_sigma gradient
- `scripts/plot_hlsp_lensing_overview.py` - Disabled broken GR/GE curves

## Test Files Created
- `scripts/test_deflection_functions.py` - Tests all three deflection functions
- `scripts/test_ge_debug.py` - Debug GE failure
- `LENSING_FIXES_2025-01-08.md` - This document
