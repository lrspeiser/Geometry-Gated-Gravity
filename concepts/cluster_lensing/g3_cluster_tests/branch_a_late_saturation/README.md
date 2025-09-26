# Branch A: Late-Saturation G³

## Overview
This branch implements a **late-saturation** modification to G³ that allows the tail acceleration to continue growing at cluster scales while remaining gentle in galaxies. This addresses the cluster lensing deficit without violating galaxy or Solar System constraints.

## Physics

### The Problem
Standard G³ suffers from early saturation at `g_sat ~ 1200 (km/s)²/kpc`, which is calibrated for galaxies but prevents sufficient mass buildup for cluster strong lensing. The convergence κ̄ saturates at ~0.17 when we need ~1.0 for Einstein rings.

### The Solution
We introduce a **cluster-scale booster** that activates beyond a characteristic radius `r_boost`:

```
g_tail(r) = (v₀²/r) × (r/(r+rc))^p × S(Σ) × [1 + (r/r_boost)^q]^η
```

Additionally, we make the saturation cap **adaptive** to the system's mean density:

```
g_sat_eff = g_sat₀ × (Σ_bar/Σ₀)^(-ζ)
```

### Key Features
- **Preserves galaxy dynamics**: For r << r_boost, the booster ≈ 1 (no change)
- **Enhances cluster lensing**: For r > r_boost, the tail grows by factor (r/r_boost)^(qη)
- **Single universal law**: Same equation, just different regime activation
- **No EEP violation**: All matter types experience the same field

## Parameters

### Standard G³ Parameters (unchanged)
- `v0 = 150.0 km/s` - Tail velocity scale
- `rc0 = 25.0 kpc` - Tail turnover radius
- `gamma = 0.2` - Geometry scaling for rc
- `beta = 0.1` - Amplitude tilt with surface density
- `Sigma_star = 20.0 M☉/pc²` - Screening threshold
- `alpha = 1.5` - Screening sharpness

### New Late-Saturation Parameters
- `r_boost = 300 kpc` - Radius where boost activates (default: ∞ = off)
- `q_boost = 2.0` - Power law for booster
- `eta_boost = 0.3` - Boost strength (0 = no boost)
- `zeta_gsat = 0.2` - Adaptive cap exponent (0 = fixed cap)

## Usage

### Basic Test
```python
from g3_late_saturation import G3LateSaturation, test_late_saturation_on_cluster

# Run with moderate boost
metrics = test_late_saturation_on_cluster('A1689', eta_boost=0.3, r_boost_kpc=300)
```

### Custom Analysis
```python
import numpy as np
from g3_late_saturation import G3LateSaturation

# Create solver with late-saturation
g3 = G3LateSaturation(
    v0=150.0, rc0=25.0,           # Standard parameters
    r_boost=300.0, eta_boost=0.3,  # Late-saturation
    zeta_gsat=0.2                  # Adaptive cap
)

# Mock cluster density
r = np.logspace(0, 3.5, 100)  # 1-3000 kpc
rho = 1e7 / ((r/300) * (1 + r/300)**2)  # NFW-like

# Solve field
result = g3.solve_field(rho, r, geometry='spherical')

# Get lensing convergence
kappa, kappa_bar = g3.compute_lensing_convergence(result)
```

### Parameter Sweep
```python
# Test different boost strengths
for eta in [0.0, 0.2, 0.3, 0.5]:
    metrics = test_late_saturation_on_cluster(f'test_eta_{eta}', eta_boost=eta)
    print(f"η={eta}: κ_max={metrics['late_saturation']['kappa_bar_max']:.3f}")
```

## Expected Results

### Without boost (η=0)
- κ̄_max ≈ 0.17 (insufficient for strong lensing)
- Works well for galaxies
- Cluster lensing severely underpredicted

### With moderate boost (η=0.3, r_boost=300 kpc)
- κ̄_max ≈ 0.4-0.5 (improved but still below unity)
- Galaxy dynamics preserved (r < 50 kpc unaffected)
- Partial improvement for clusters

### With strong boost (η=0.5, r_boost=200 kpc)
- κ̄_max ≈ 0.6-0.8 (approaching requirements)
- May start affecting extended galaxies
- Better cluster match but needs validation

## Validation Tests

### 1. Milky Way Check
```bash
python test_mw_late_saturation.py --eta_boost 0.3 --r_boost_kpc 300
```
Expected: < 5% change in MW rotation curve

### 2. SPARC Galaxies
```bash
python test_sparc_late_saturation.py --eta_boost 0.3 --r_boost_kpc 300
```
Expected: < 2% degradation in median accuracy

### 3. Solar System
The booster only activates at r > r_boost ~ 300 kpc, so Solar System (r ~ 30 AU ~ 10^-7 kpc) is completely unaffected.

## Limitations

1. **Ad hoc booster**: The functional form [1+(r/r_boost)^q]^η is phenomenological
2. **Parameter tuning**: Need to find optimal (r_boost, η) that works for all clusters
3. **Physical motivation**: Unclear why gravity would "know" about cluster scales
4. **Still may not reach κ=1**: Even with boost, may fall short of full Einstein rings

## Output Files

- `outputs/{cluster}_test.png` - Diagnostic plots showing acceleration, convergence, boost factor
- `outputs/{cluster}_metrics.json` - Quantitative metrics and improvement factors

## Comparison with Other Branches

| Aspect | Branch A (Late-sat) | Branch B (Photon) | Branch C (Mass-dep) |
|--------|---------------------|-------------------|---------------------|
| Physics | Enhanced tail at large r | Photons feel stronger field | Light tracers pulled more |
| EEP | Preserved ✓ | Preserved ✓ | Violated ✗ |
| Solar Safety | Automatic (r >> r_boost) | Via screening (S→0) | Requires tuning |
| MW/Galaxies | Preserved if r_boost > 100 kpc | Fully preserved | May break |
| Implementation | Modify tail formula | Separate dynamics/lensing | Post-process g_eff |