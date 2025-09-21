# Under/Over by Hubble Type

Top variant: xi=shell_logistic_radius, gate=fixed

Overall (all types included):
- Total outer points: 1167
- Under: 679 (58.18%)  Over: 488 (41.82%)
- Mean offset: 8.92 km/s  RMSE: 52.36 km/s

Per type:

| Type | Label | n_gal | outer_pts | under% | over% | mean_off | rmse | flags |
|------|-------|-------|-----------|--------|-------|----------|------|-------|
| 0 | S0 | 2 | 22 | 72.7% | 27.3% | -26.3 | 45.6 | under/over imbalance ≥ 65%, |mean offset| ≥ 10 km/s, small-sample (n_gal < 3) |
| 1 | Sa | 3 | 32 | 40.6% | 59.4% | 12.6 | 37.8 | |mean offset| ≥ 10 km/s |
| 2 | Sab | 10 | 194 | 34.5% | 65.5% | 36.3 | 74.6 | under/over imbalance ≥ 65%, |mean offset| ≥ 10 km/s, rmse > 65.5 |
| 3 | Sb | 12 | 107 | 35.5% | 64.5% | 28.6 | 72.5 | |mean offset| ≥ 10 km/s, rmse > 65.5 |
| 4 | Sbc | 15 | 122 | 14.8% | 85.2% | 50.3 | 76.9 | under/over imbalance ≥ 65%, |mean offset| ≥ 10 km/s, rmse > 65.5 |
| 5 | Sc | 15 | 101 | 58.4% | 41.6% | 5.2 | 39.2 |  |
| 6 | Scd | 16 | 169 | 63.9% | 36.1% | -0.2 | 44.5 |  |
| 7 | Sd | 15 | 108 | 78.7% | 21.3% | -11.9 | 23.7 | under/over imbalance ≥ 65%, |mean offset| ≥ 10 km/s |
| 8 | Sdm | 9 | 46 | 73.9% | 26.1% | -11.0 | 32.8 | under/over imbalance ≥ 65%, |mean offset| ≥ 10 km/s |
| 9 | Sm | 20 | 123 | 96.7% | 3.3% | -18.9 | 22.7 | under/over imbalance ≥ 65%, |mean offset| ≥ 10 km/s |
| 10 | Im | 24 | 127 | 83.5% | 16.5% | -10.8 | 20.3 | under/over imbalance ≥ 65%, |mean offset| ≥ 10 km/s |
| 11 | BCD | 2 | 16 | 100.0% | 0.0% | -41.3 | 41.6 | under/over imbalance ≥ 65%, |mean offset| ≥ 10 km/s, small-sample (n_gal < 3) |

If we exclude flagged types:
- Total outer points: 270
- Under: 167 (61.85%)  Over: 103 (38.15%)
- Mean offset: 1.83 km/s  RMSE: 42.58 km/s

Top outlier galaxies by RMSE (z-score):

| Galaxy | Type | outer_pts | RMSE | z |
|--------|------|-----------|------|---|
| NGC5005 | Sbc | 7 | 169.3 | 4.20 |
| UGC11914 | Sab | 23 | 147.9 | 3.54 |
| NGC6195 | Sb | 8 | 144.4 | 3.43 |
| NGC3521 | Sbc | 14 | 129.6 | 2.96 |
| NGC3953 | Sbc | 3 | 121.1 | 2.70 |
| NGC3877 | Sc | 5 | 120.3 | 2.68 |
| UGC11455 | Scd | 13 | 119.9 | 2.66 |
| NGC2955 | Sb | 9 | 117.5 | 2.59 |
| NGC5371 | Sbc | 7 | 111.0 | 2.39 |
| NGC0891 | Sb | 7 | 105.6 | 2.22 |

Top outlier galaxies by |mean offset| (z-score):

| Galaxy | Type | outer_pts | mean_off | z |
|--------|------|-----------|----------|---|
| NGC5005 | Sbc | 7 | 168.3 | 3.71 |
| UGC11914 | Sab | 23 | 146.5 | 3.22 |
| NGC6195 | Sb | 8 | 140.0 | 3.07 |
| NGC3953 | Sbc | 3 | 120.7 | 2.63 |
| NGC3877 | Sc | 5 | 120.2 | 2.62 |
| NGC3521 | Sbc | 14 | 119.1 | 2.59 |
| NGC2955 | Sb | 9 | 115.1 | 2.50 |
| UGC11455 | Scd | 13 | 111.2 | 2.42 |
| NGC5371 | Sbc | 7 | 107.0 | 2.32 |
| NGC0891 | Sb | 7 | 104.6 | 2.26 |

Closest galaxy pairs by metric similarity (z-space on rmse, mean_off, under_pct):

| Galaxy A | Type A | Galaxy B | Type B | distance |
|----------|--------|----------|--------|----------|
| UGC07608 | Im | UGC08286 | Scd | 0.00 |
| NGC1003 | Scd | NGC2915 | BCD | 0.00 |
| DDO064 | Im | NGC2366 | Im | 0.00 |
| D631-7 | Im | UGC12732 | Sm | 0.00 |
| F583-1 | Sm | NGC4214 | Im | 0.00 |
| UGC04278 | Sd | UGC06983 | Scd | 0.00 |
| F568-3 | Sd | NGC0100 | Scd | 0.00 |
| UGC07603 | Sd | UGC12506 | Scd | 0.00 |
| DDO154 | Im | UGC07603 | Sd | 0.00 |
| IC2574 | Sm | UGC00191 | Sm | 0.01 |

Data sources: C:\Users\henry\dev\GravityCalculator\out\cupy_nocap\leaderboard.csv and C:\Users\henry\dev\GravityCalculator\out\cupy_nocap\xi_shell_logistic_radius__gate_fixed\best_params.json ; types from C:\Users\henry\dev\GravityCalculator\data\Rotmod_LTG\MasterSheet_SPARC.csv