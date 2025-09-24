# G³ Gravity Model - ACTUAL Performance Report

## Executive Summary

**The "99.3% accuracy" reported by the optimizer is misleading.** The actual median prediction error is **12.3%**, with only **36.1%** of stars predicted within 10% accuracy.

## Key Performance Metrics

### Overall Accuracy
- **Median relative error**: 12.3%
- **Mean relative error**: 12.6%
- **Median absolute error**: 33.1 km/s
- **95th percentile error**: 72.7 km/s

### Distribution of Errors
- Stars with **< 5% error**: 12.8%
- Stars with **< 10% error**: 36.1%
- Stars with **< 20% error**: 87.5%
- Stars with **< 30% error**: 98.8%
- Stars with **< 50% error**: 99.9%

### Comparison with Pure Newtonian
- **Newtonian median error**: 37.9%
- **Our model median error**: 12.3%
- **Improvement factor**: 3.1x better than Newtonian

## Systematic Issues Identified

### 1. Radial Dependence
The model performs best in the inner galaxy and degrades with radius:
- **Inner zone (4-6 kpc)**: 6.1% median error, 69.2% within 10%
- **Solar zone (6-8 kpc)**: 8.7% median error, 58.5% within 10%
- **Mid zone (8-10 kpc)**: 13.8% median error, 24.8% within 10%
- **Outer zone (10-12 kpc)**: 21.4% median error, 2.7% within 10%
- **Far zone (12-15 kpc)**: 30.7% median error, 2.0% within 10%

### 2. Systematic Under-prediction
- **107,360 stars** (74.6%) are under-predicted by >20 km/s
- Mean under-prediction: -43.3 km/s
- This suggests the dark matter contribution is systematically underestimated

### 3. Vertical Structure Effects
Performance degrades with height above the disk:
- **Thin disk (|z| < 0.3 kpc)**: 12.3% median error
- **Mid disk (0.3-0.6 kpc)**: 16.3% median error  
- **Thick disk (|z| > 0.6 kpc)**: 21.9% median error

## Why the Optimizer Reported 99.3% Accuracy

The optimizer minimized a different metric:
```
Optimizer loss = median(|v_pred - v_obs| / v_obs)
```

This gave a loss value of 0.1234, which the optimizer interpreted as "87.7% accuracy" (1 - 0.1234). The reported "99.3%" appears to come from an even different calculation or a bug in the reporting.

The actual relative error that matters for physical predictions:
```
Actual error = |v_pred - v_obs| / v_obs * 100%
```

## Model Components Analysis

The thickness_gate model uses:
1. **Rational term**: R/(R + rc_eff) - provides basic profile
2. **Logistic gate**: 1/(1 + exp(-(R-r0)/delta)) - transition function
3. **Density screening**: 1/(1 + (Σ0/Σ_loc)^α) - local density effects
4. **Vertical modification**: rc_eff = rc * (1 + (|z|/hz)^m)

## Recommendations for Improvement

1. **Address systematic under-prediction**
   - The model consistently underestimates velocities at R > 8 kpc
   - Consider adjusting the logistic gate parameters or profile shape

2. **Improve outer galaxy modeling**
   - Performance degrades severely beyond 10 kpc
   - May need different functional form for outer regions

3. **Refine vertical structure**
   - The thickness gate may need adjustment
   - Consider more sophisticated z-dependence

4. **Re-optimize with proper metric**
   - Use actual percentage error as the loss function
   - Weight stars equally regardless of velocity magnitude

5. **Consider additional physics**
   - Non-axisymmetric features (spiral arms, bar)
   - Velocity dispersion effects
   - More sophisticated density screening

## Files Generated

- `actual_predictions.png` - Comprehensive visualization of actual vs predicted velocities
- `star_diagnostics.png` - Detailed 9-panel diagnostic of systematic errors
- `worst_predictions.csv` - 1000 stars with largest prediction errors for inspection

## Conclusion

While the model achieves a 3x improvement over pure Newtonian gravity, the actual performance falls short of the claimed accuracy. The median 12.3% error and systematic under-prediction indicate that further refinement is needed, particularly for the outer galaxy and vertical structure modeling.

The model successfully captures the need for additional (dark matter) acceleration beyond Newtonian, but the specific functional form and parameters require adjustment to better match the Gaia observations.