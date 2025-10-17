# Blind Hold-Out Validation Results

## Calibration

- Population mean: μ_A = 16.468
- Population scatter: σ_A = 0.812
- Training sample: 5 relaxed clusters (MACS0717 excluded)
- Training χ²/d.o.f. = 2.21

## Hold-Out Predictions

| Cluster | Observed θ_E | Predicted θ_E | Residual | Z-score | Status |
|---------|--------------|---------------|----------|---------|--------|
| A1689 | 47.0±3.0" | 31.9 (+4.9, -4.9)" | -15.09" | -5.03σ | ⚠ TENSION |
| MACS1149 | 42.0±2.0" | 22.5 (+4.7, -0.0)" | -19.50" | -9.75σ | ⚠ TENSION |

## Interpretation

- **|Z| < 1.5σ**: Excellent agreement
- **1.5σ < |Z| < 2.0σ**: Acceptable (within ~95% CI)
- **|Z| > 2.0σ**: Significant tension (requires investigation)
