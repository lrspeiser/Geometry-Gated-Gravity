# Best Variant per Hubble Type

Baseline (global best): xi=shell_logistic_radius, gate=fixed; ranking metric: rmse

Per type: best variant vs baseline (Δ = best − baseline):

| Type | Label | Best xi | Best gate | RMSE | ΔRMSE | |mean_off| Δ | Under% | ΔUnder% |
|------|-------|---------|-----------|------|-------|-----------|--------|----------|
| 0 | S0 | combined_radius_density_gamma | fixed | 45.1 | -0.4 | 25.4 | -0.9 | 68.2% | -4.5% |
| 1 | Sa | combined_radius_density_gamma | learned_compact | 37.6 | -0.2 | 12.2 | -0.4 | 40.6% | +0.0% |
| 2 | Sab | logistic_density | fixed | 66.1 | -8.5 | 50.7 | +14.4 | 82.5% | +47.9% |
| 3 | Sb | logistic_density | fixed | 71.7 | -0.8 | 52.4 | +23.8 | 88.8% | +53.3% |
| 4 | Sbc | logistic_density | fixed | 48.7 | -28.2 | 30.0 | -20.3 | 77.9% | +63.1% |
| 5 | Sc | combined_radius_density_gamma | learned_compact | 39.0 | -0.1 | 4.9 | -0.3 | 59.4% | +1.0% |
| 6 | Scd | combined_radius_density_gamma | learned_compact | 44.3 | -0.2 | 0.5 | +0.3 | 63.9% | +0.0% |
| 7 | Sd | combined_radius_density_gamma | fixed | 23.6 | -0.1 | 11.6 | -0.3 | 78.7% | +0.0% |
| 8 | Sdm | combined_radius_density_gamma | fixed | 32.8 | -0.0 | 10.7 | -0.3 | 71.7% | -2.2% |
| 9 | Sm | combined_radius_density_gamma | fixed | 22.6 | -0.2 | 18.7 | -0.2 | 95.9% | -0.8% |
| 10 | Im | combined_radius_density_gamma | learned | 20.2 | -0.1 | 10.6 | -0.1 | 82.7% | -0.8% |
| 11 | BCD | combined_radius_density_gamma | fixed | 41.4 | -0.1 | 41.1 | -0.1 | 100.0% | +0.0% |

Winners by variant:
- combined_radius_density_gamma / fixed: 5 types
- combined_radius_density_gamma / learned_compact: 3 types
- logistic_density / fixed: 3 types
- combined_radius_density_gamma / learned: 1 types

Plain-language summary (see also assistant write-up):
- Which model style tends to win across types and where it differs.
- Whether improvements are large (e.g., >5 km/s RMSE) or minor.
- Whether late types (Sd/Im) show different bias than earlier spirals (Sa/Sb).