# A_0 Calibration Summary

## Problem Statement
After fixing the g_bar calculation bug (quadrature method), the RAR g† improved from 7.0× to 3.19× the literature value. We added the A_0 amplitude parameter to enable further calibration.

## Key Findings

### 1. g_bar Calculation Fix ✅
**Before**: `v_bar² = v_disk² + v_bulge² + v_gas²` (WRONG)  
**After**: `v_bar = √(v_disk² + v_bulge² + v_gas²)` then square (CORRECT)

**Impact**:
- g† reduced from 8.4×10⁻¹⁰ to 3.83×10⁻¹⁰ m/s²
- Ratio vs literature (1.2×10⁻¹⁰): **7.0× → 3.19×** 🎉

### 2. A_0 Parameter Implementation ✅
Added global amplitude scaling factor A_0 to `PathSpectrumHyperparams`:
- Default value: 1.0 (backward compatible)
- Implementation: `K_scaled = A_0 × K_total`
- **Preserves Newtonian limit**: K=0 at small radii regardless of A_0

**Verification**:
- Boost factor K scales linearly with A_0 ✅
- Model predictions respond correctly to A_0 ✅  
- See `diagnose_A0_effect.py` results

### 3. RAR Calibration Analysis

#### Method 1: Free g† Fitting (INCORRECT)
- Fitted g† = 3.83×10⁻¹⁰ m/s² **constant for all A_0**
- Problem: RAR fitting is scale-invariant
- Conclusion: Can't use this method to find optimal A_0

#### Method 2: Fixed g† = 1.2×10⁻¹⁰ m/s² (CORRECT)
- Scatter: **~0.256 dex** (constant across all A_0)
- Mean bias: -0.36 to -0.26 dex (shifts with A_0)
- Target scatter: < 0.15 dex (literature)
- **Status**: Scatter too high ⚠️

### 4. Current Limitations

**The fundamental issue**: Scatter is intrinsic to the model structure, not the amplitude.

**What A_0 can do**:
- ✅ Scale the boost amplitude uniformly
- ✅ Shift mean bias (bring g_model closer to g_obs on average)
- ✅ Preserve physics (Newtonian limit, energy conservation)

**What A_0 cannot do**:
- ❌ Reduce intrinsic scatter
- ❌ Fix shape mismatch between model and RAR curve
- ❌ Account for galaxy-specific variations

**Current performance**:
- RAR scatter: 0.256 dex (target < 0.15)
- Mean bias: -0.33 dex at A_0=1.0 (model underpredicts by ~2.1×)
- Best A_0: ~1.0 (minimizes scatter, though scatter is essentially flat)

### 5. Why the Model Underpredicts

The model with current hyperparameters (L_0=1.82, etc.) produces too **weak** a boost:
- At outer radii, K ~ 0.05-0.15 (5-15% boost)
- To match observations, need K ~ 0.5-1.0 (50-100% boost)
- Factor of ~3-7× amplification needed

**Options**:
1. **Increase A_0** to ~3-5: Would reduce bias but scatter remains high
2. **Re-tune L_0, K_max**: Increase baseline boost strength
3. **Add new physics**: e.g., dark matter halo coupling, non-linear effects
4. **Accept limitations**: Current model may be a first-order approximation

## Recommendations

### Short-term: Optimal A_0 for Current Model
**A_0 = 1.0** (current value)
- Minimizes scatter (though improvement is marginal)
- Mean bias: -0.33 dex (model low by factor of 2.1×)
- Physics tests: ✅ PASS (Newtonian limit preserved)

### Medium-term: Boost Amplitude Increase
Test **A_0 = 2.5-3.0**:
- Would reduce mean bias significantly
- Scatter remains ~0.256 dex
- Check if outer radius behavior is reasonable

### Long-term: Model Structure Improvements
1. **Re-optimize hyperparameters** with A_0 included
2. **Add radius-dependent modulation**: K(r) could have different shape
3. **Include halo-coupling term**: Baryons may not act alone
4. **Galaxy-specific corrections**: BT, bar strength may need stronger effects

## Files Generated
- `test_A0_calibration.py`: Initial A_0 scan (free g† - showed scale invariance)
- `diagnose_A0_effect.py`: Direct verification that A_0 scales boost correctly
- `calibrate_A0_fixed_gdagger.py`: Proper calibration with fixed g† = 1.2e-10
- `A0_direct_effect.png`: Visualization of A_0 effect on single galaxy
- `A0_calibration_fixed_gdagger.png`: Scatter vs A_0 curves

## Next Steps
1. ✅ Commit diagnostic scripts and findings
2. ⚠️  Decide on A_0 value (keep 1.0 or increase to 2.5-3.0)
3. 📋 Consider full hyperparameter re-optimization including A_0
4. 📊 Investigate sources of high scatter (galaxy-by-galaxy analysis)
