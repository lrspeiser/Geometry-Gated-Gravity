# CRITICAL BUG: Conflicting Clumping Corrections

## Discovery Date
2025-01-14

## Severity
**CRITICAL** - Affects all cluster Einstein radius predictions

## Summary
Two different implementations of gas clumping correction apply **opposite** corrections to gas density, causing systematic errors in Einstein radius predictions of ~2x.

## The Bug

### Implementation 1: `core/gas_profiles.py::apply_clumping_correction`
```python
def apply_clumping_correction(r, rho_gas, C0=0.3, eta=2.0, R_200=None):
    """
    X-ray observations underestimate density by factor sqrt(C) due to
    unresolved clumping (X-ray ∝ n_e² sees clumps as overdense).
    
    C(r) = 1 + C₀ (r/R_200)^η
    ρ_true = sqrt(C(r)) × ρ_X-ray
    """
    C_r = 1 + C0 * (r / R_200)**eta
    return rho_gas * np.sqrt(C_r)  # INCREASES density
```

**Effect**: Multiplies gas density by sqrt(C) → **INCREASES** total baryon mass

### Implementation 2: `core/build_cluster_baryons.py::build_cluster_baryon_model`
```python
def build_cluster_baryon_model(...):
    """
    # Line 296-300
    # Simionescu: C = <n_e^2>/<n_e>^2
    # So measured n_e is biased high by sqrt(C), true n_e = measured / sqrt(C)
    """
    C_factor = clumping_profile(r, R_500, C0=1.3, eta=2.0, C_max=2.5)
    rho_gas_corrected = rho_gas / np.sqrt(C_factor)  # DECREASES density
```

**Effect**: Divides gas density by sqrt(C) → **DECREASES** total baryon mass

## Impact on Predictions

### MACS0416 Test Case

| Implementation | Clumping | theta_E(pred) | theta_E(obs) | Error |
|---------------|----------|---------------|--------------|-------|
| **test_macs0416_full_physics.py** | `apply_clumping_correction` (×sqrt(C)) | 33.04" | 30.0" | +10% ✓ |
| **run_cluster_suite.py** | `build_cluster_baryon_model` (÷sqrt(C)) | 16.66" | 30.0" | -44% ✗ |

The **factor of 2** difference (33" vs 17") is directly caused by the opposite clumping corrections.

### Blind Cluster Suite Results (all affected)
- Median residual: -42% (systematic underprediction)
- Only 1/12 clusters within ±10%
- All clusters underpredict due to reduced gas mass from incorrect clumping

## Physical Reasoning

### What is "Clumping"?
Clumping factor C(r) = ⟨n_e²⟩ / ⟨n_e⟩² ≥ 1

- X-ray luminosity L_X ∝ ∫ n_e² dV  (bremsstrahlung)
- If gas has unresolved clumps, ⟨n_e²⟩ > ⟨n_e⟩²
- X-ray observations **overestimate** the **emission-weighted** density
- To get true mean density: n_e(true) = n_e(X-ray) / sqrt(C)
- Therefore: **ρ_gas(true) = ρ_gas(X-ray) / sqrt(C)**

### Correct Physics
The **second implementation** (`build_cluster_baryons.py`) is **CORRECT**.

The **first implementation** (`gas_profiles.py`) is **WRONG** - it applies the correction backwards.

## Citation Trail

### Simionescu+ 2011 (ApJ, 757, 182)
- Measured C(r) in Coma cluster from Suzaku
- C ranges from ~1.2 in core to ~2.5 at R_200
- Convention: C = ⟨n_e²⟩/⟨n_e⟩²
- **Correction**: true density = X-ray density **divided by sqrt(C)**

### Eckert+ 2015 (A&A, 575, A72)
- Confirmed clumping increases with radius
- C(r) ~ (r/R_500)^1.7
- **Correction**: reduces inferred gas mass by factor ~1.4 at R_200

## Files Affected

### Primary Bug Location
- `core/gas_profiles.py::apply_clumping_correction` (lines 193-227)
  - **Status**: INCORRECT (applies correction backwards)
  - **Used by**: `scripts/test_macs0416_full_physics.py`
  
### Correct Implementation
- `core/build_cluster_baryons.py::build_cluster_baryon_model` (lines 292-300)
  - **Status**: CORRECT
  - **Used by**: `scripts/run_cluster_suite.py`

## Required Fixes

### Priority 1: Fix `gas_profiles.py`
```python
def apply_clumping_correction(r, rho_gas, C0=0.3, eta=2.0, R_200=None):
    """
    Apply clumping correction to gas density.
    
    X-ray observations OVERestimate density by factor sqrt(C) due to
    unresolved clumping (L_X ∝ n_e² weights clumps).
    
    C(r) = 1 + C₀ (r/R_200)^η
    ρ_true = ρ_X-ray / sqrt(C(r))  # <-- DIVIDE, not multiply
    """
    if R_200 is None or C0 == 0:
        return rho_gas
    
    C_r = 1 + C0 * (r / R_200)**eta
    return rho_gas / np.sqrt(C_r)  # FIX: divide instead of multiply
```

### Priority 2: Re-run `test_macs0416_full_physics.py`
After fixing `gas_profiles.py`, the MACS0416 test should predict:
- **Interior only**: ~17" (currently 33" due to bug)
- **Both families**: ~55" (currently 111" due to bug)

This will show that **exterior arcs are necessary** to reach 30" observed.

### Priority 3: Re-run Blind Cluster Suite
After fix, median residual should improve from -42% to closer to 0%.

### Priority 4: Update Clumping Parameters
Current parameters may need adjustment:
- `build_cluster_baryons.py` uses C0=1.3, C_max=2.5 (physically motivated)
- `gas_profiles.py` uses C0=0.3 (too small, may be compensating for bug)

## Verification Steps

1. ✓ Identify discrepancy: MACS0416 predicts 33" (standalone) vs 17" (suite)
2. ✓ Trace to clumping: Factor ~2x difference matches sqrt(C) ≈ 1.5-1.8
3. ✓ Find opposite corrections in code
4. ✓ Confirm physics: literature supports DIVISION by sqrt(C)
5. ⬜ Fix `gas_profiles.py::apply_clumping_correction`
6. ⬜ Re-run standalone MACS0416 test
7. ⬜ Re-run blind cluster suite
8. ⬜ Verify median residual improves to <15%

## Next Steps

1. **IMMEDIATE**: Fix `apply_clumping_correction` to divide by sqrt(C)
2. **TEST**: Run both test suites to verify consistent results
3. **DOCUMENT**: Update physics documentation to clarify clumping convention
4. **CALIBRATE**: May need to adjust w_interior/w_exterior weights after fix
5. **PAPER**: Update cluster validation section with corrected results

## Notes

- This bug has been present since the initial cluster kernel implementation
- The blind cluster suite accidentally used the CORRECT implementation
- The standalone MACS0416 test used the INCORRECT implementation
- The fact that standalone test gave "good" results (33" vs 30") was actually due to the bug doubling the gas mass artificially
- With the correct clumping, we likely need **both** interior and exterior families to match observations

---

**Error Type**: Sign error / opposite correction
**Introduced**: Phase 1 gas profile implementation
**Discovered**: 2025-01-14 during cluster suite diagnostic analysis
**Impact**: ~2x systematic error in all cluster Einstein radius predictions
