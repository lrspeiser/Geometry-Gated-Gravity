# Population Law Fitting System - Status Report

## Date: 2025-10-12

## Summary
Created a complete population law fitting system that replaces discrete morphological classes with smooth, continuous functions of galaxy properties (B/T ratio, stellar mass, scale length).

## Completed Tasks

### 1. ✅ Added L2 Regularization to Fitting
**File:** `many_path_model/fit_population_laws.py`

- Implemented ridge regression (L2 penalty) to prevent parameter blow-up
- Configurable via `--l2_lambda` parameter (default: 0.01)
- Applied during training but not validation (proper generalization testing)

**Key Changes:**
- `evaluate_population_laws()` now includes `l2_lambda` parameter
- L2 penalty: `l2_lambda * np.sum(params**2)` added to median APE
- Validation sets evaluated without L2 penalty for fair comparison

### 2. ✅ Created Zero-Shot Test Script
**File:** `many_path_model/sparc_zero_shot_population.py`

Complete testing system for population laws without per-galaxy tuning:
- Loads fitted population laws from JSON
- Predicts rotation curves using only B/T, M_*, and R_d
- Compares against class-wise universal parameters
- Generates per-galaxy plots and CSV exports
- Computes stratified performance metrics

**Features:**
- GPU acceleration support
- Optional class-wise parameter comparison
- Saves plots for each galaxy
- Exports results to JSON and CSV
- Performance binned by morphology (pure disk, mixed, bulge-dominated)

### 3. ⚠️ Population Law Fitting - In Progress
**Status:** Initial 5-fold CV attempt interrupted; switched to 3-fold with reduced iterations

**Configuration:**
- Reduced from `maxiter=100, popsize=15` to `maxiter=50, popsize=10` for faster completion
- Using differential evolution optimizer with L2 regularization
- 3-fold cross-validation for faster testing

**Issue:** Process interrupted during JSON writing. File created but incomplete (~2.4KB).

## Population Laws Designed

The system fits smooth functions to replace discrete class parameters:

### 1. Eta (Kernel Softness)
```
eta(B/T, M_*) = a0 + a1*log(M_*) + a2*B/T + a3*(B/T)^2
```
- **a0**: Baseline eta
- **a1**: Mass dependence (how eta scales with stellar mass)
- **a2**: Linear bulge fraction effect
- **a3**: Quadratic bulge fraction effect (captures non-linear trends)

**Physics:** Captures how disk-dominated galaxies (low B/T) have different kernel behavior than bulge-dominated systems.

### 2. Ring Amplitude
```
ring_amp(B/T) = b0 * exp(-b1 * B/T)
```
- **b0**: Maximum ring amplitude (pure disk limit)
- **b1**: Exponential decay rate with increasing bulge fraction

**Physics:** Ring structures are prominent in pure disks, decay exponentially as bulges grow.

### 3. Maximum Kernel Mass
```
M_max(M_*, R_d) = c0 + c1*log(M_*) + c2*log(R_d)
```
- **c0**: Baseline M_max
- **c1**: Stellar mass scaling
- **c2**: Disk size scaling

**Physics:** Total gravitational "capacity" scales with both mass and spatial extent.

### 4. Fixed Parameters
- **lambda_hat**: 20.0 kpc (kept fixed, anchored to Milky Way)
- **bulge_gate_power**: 32.9 (MW-anchored, controls bulge kernel suppression)

## Next Steps

### Immediate Actions
1. **Re-run population law fitting with robust configuration:**
   ```powershell
   python many_path_model/fit_population_laws.py --sparc_dir data/Rotmod_LTG --output_dir results/pop_laws_v2 --cv_folds 3 --l2_lambda 0.01
   ```

2. **Once fitting completes, run zero-shot tests:**
   ```powershell
   python many_path_model/sparc_zero_shot_population.py --laws results/pop_laws_v2/population_laws.json --output_dir results/zero_shot_pop --verbose
   ```

3. **Compare against class-wise params:**
   ```powershell
   python many_path_model/sparc_zero_shot_population.py --laws results/pop_laws_v2/population_laws.json --class_params results/mega_parallel/class_params_for_zero_shot.json --output_dir results/zero_shot_comparison --verbose
   ```

### Expected Outcomes
- **Train APE:** ~30-40% (based on class-wise medians: late=35.9%, early=34.2%, intermediate=39.6%)
- **Val APE:** ~35-45% (expect small generalization gap with L2 regularization)
- **Zero-shot comparison:** Population laws should match or slightly outperform class-wise params due to smooth interpolation

### Success Criteria
1. **Convergence:** All 3 CV folds complete without errors
2. **Generalization gap < 10%:** Difference between train and val APE
3. **Physical parameter values:**
   - `eta_a0` ∈ [0.01, 2.0]
   - `ring_amp_b0` ∈ [1.0, 3.0]
   - `M_max_c0` ∈ [2.0, 3.5]
4. **Zero-shot APE ~35%:** Similar to class-wise universal performance

## Physics Validation Tests (After Successful Fitting)

### 1. Parameter Trends
- Verify `eta` decreases with increasing B/T (disk → bulge transition)
- Check `ring_amp` exponential decay with B/T
- Confirm `M_max` scales positively with both M_* and R_d

### 2. Outlier Analysis
- Identify galaxies with APE > 50%
- Check for systematic biases in specific morphology bins
- Investigate dwarf galaxies (M_* < 10^9) and giants (M_* > 10^11)

### 3. Continuous vs. Discrete Comparison
- Plot APE distributions for both methods
- Check boundary regions (e.g., B/T ~ 0.15, 0.5)
- Verify smooth transitions eliminate class-wise discontinuities

## File Manifest

### Core Files
- `many_path_model/fit_population_laws.py` - Main fitting system with CV and L2 regularization
- `many_path_model/sparc_zero_shot_population.py` - Zero-shot testing framework
- `results/pop_laws/population_laws.json` - (Incomplete) Fitted parameters
- `POPULATION_LAWS_STATUS.md` - This status report

### Dependencies
- `sparc_stratified_test.py` - Galaxy loading, prediction, metrics
- Requires: `numpy`, `scipy`, `pandas`, `matplotlib`, `cupy` (optional GPU)

## Technical Notes

### GPU Acceleration
- CuPy detected and enabled
- Processing rate: 5-11M particles/s (typical)
- Slower rates (0.5-2M particles/s) observed during intensive computation phases

### Optimization Details
- **Algorithm:** Differential Evolution (global optimizer)
- **Population size:** 10 (reduced from 15)
- **Max iterations:** 50 per fold (reduced from 100)
- **Workers:** 1 (avoid nested parallelism)
- **Polishing:** Enabled (local refinement after DE)

### L2 Regularization Impact
- **Purpose:** Prevent overfitting to training data
- **Implementation:** Adds `lambda * ||params||^2` to objective
- **Tuning:** Start with 0.01, increase if val >> train APE

## Known Issues

1. **JSON Writing Incomplete:** Previous run interrupted during file write
   - **Solution:** Re-run with increased stability (fewer folds/iterations)
   
2. **Long Runtime:** 3-fold CV ~30-60 minutes
   - **Mitigation:** Reduced iterations (50) and population (10)
   - Consider 2-fold for quick tests

3. **Parameter Bounds:** Current bounds are conservative
   - May need expansion if optimizer hits bounds frequently
   - Check convergence warnings in output

## Computational Resources

**Estimated Requirements:**
- **Time:** 30-60 min (3-fold CV)
- **Memory:** ~8-16GB RAM (GPU mode)
- **Storage:** <100MB for results
- **GPU:** NVIDIA GPU with CuPy support (optional but recommended)

## References

**Related Files:**
- `results/mega_parallel/class_params_for_zero_shot.json` - Class-wise comparison baseline
- `NEXT_STEPS_COMMANDS.md` - Original workflow plan
- `many_path_model/sparc_stratified_test.py` - Testing framework

**Key Commits:**
- Added L2 regularization to population law fitting
- Created zero-shot testing system for population laws
- Reduced optimization iterations for faster completion

---

**Status:** System complete, awaiting successful fitting run for validation.
**Next Action:** Re-run `fit_population_laws.py` with current configuration.
