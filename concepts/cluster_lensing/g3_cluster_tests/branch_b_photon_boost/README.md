# Branch B: Photon Boost (Disformal Coupling) G³

## Overview
This branch implements a **photon-specific enhancement** where photons (null geodesics) experience a stronger G³ potential than massive particles. This is physically motivated by disformal gravity theories where the scalar field couples differently to the photon and matter sectors.

## Physics

### The Problem
Standard G³ produces the same field for both dynamics (massive particles) and lensing (photons). The field strength needed for cluster lensing (κ̄ ~ 1) would overpredict dynamics (temperature), while fitting dynamics underpredicts lensing.

### The Solution
We introduce a **disformal coupling** that enhances the lensing potential while preserving dynamics:

```
Φ_lens = Φ_dyn × [1 + ξ_γ × S(Σ)^β_γ]
```

Where:
- `Φ_dyn` is the standard G³ potential used for massive particles
- `Φ_lens` is the enhanced potential seen by photons
- `S(Σ)` is the screening function (→0 in high density, →1 in low density)
- `ξ_γ` controls the photon coupling strength
- `β_γ` controls how screening affects photons

### Key Features
- **Preserves equivalence principle for massive particles**: All matter sees the same field
- **Solar System safe**: S(Σ) → 0 in high-density regions, so boost vanishes
- **No effect on galaxies/MW dynamics**: Temperature, rotation curves unchanged
- **Single universal parameter**: One ξ_γ for all clusters

## Physical Motivation

In many scalar-tensor theories, the metric for photons differs from that for matter:

```
g_μν^(matter) = A²(φ) η_μν
g_μν^(photon) = A²(φ) [1 + f(φ,∇φ)] η_μν
```

This **disformal** structure naturally arises in:
- k-essence theories
- Horndeski gravity
- Generalized Proca theories
- DBI-like actions

The screening function ensures the modification vanishes in high-density regions (Solar System, stellar interiors) while being active in low-density cluster outskirts.

## Parameters

### Standard G³ Parameters (unchanged)
- `v0 = 150.0 km/s` - Tail velocity scale
- `rc0 = 25.0 kpc` - Tail turnover radius
- `gamma = 0.2` - Geometry scaling
- `beta = 0.1` - Amplitude tilt
- `Sigma_star = 20.0 M☉/pc²` - Screening threshold
- `alpha = 1.5` - Screening sharpness

### New Photon Boost Parameters
- `xi_gamma = 0.3` - Photon coupling strength (0 = no boost)
- `beta_gamma = 1.0` - Screening power for photons

## Usage

### Basic Test
```python
from g3_photon_boost import G3PhotonBoost, test_photon_boost_on_cluster

# Run with moderate boost
metrics = test_photon_boost_on_cluster('A1689', xi_gamma=0.3, beta_gamma=1.0)
```

### Custom Analysis
```python
import numpy as np
from g3_photon_boost import G3PhotonBoost

# Create solver with photon boost
g3 = G3PhotonBoost(
    v0=150.0, rc0=25.0,        # Standard parameters
    xi_gamma=0.3, beta_gamma=1.0  # Photon boost
)

# Mock cluster
r = np.logspace(0, 3.5, 100)
rho = 1e7 / ((r/300) * (1 + r/300)**2)

# Solve field (gets both dynamics and lensing)
result = g3.solve_field(rho, r)

# Dynamics uses g_tot_dyn
g_dynamics = result['g_tot_dyn']

# Lensing uses g_tot_lens (boosted)
g_lensing = result['g_tot_lens']

# Convergence with boost
kappa, kappa_bar = g3.compute_lensing_convergence(result, use_boost=True)
```

### Solar System Safety Check
```python
from g3_photon_boost import test_solar_system_safety

# Verify boost vanishes at high density
test_solar_system_safety()
```

## Expected Results

### Without boost (ξ_γ=0)
- κ̄_max ≈ 0.17 (standard G³ deficit)
- Temperature predictions match observations
- Consistent dynamics but poor lensing

### With moderate boost (ξ_γ=0.3)
- κ̄_max ≈ 0.35-0.45 (2-2.5× improvement)
- Temperature unchanged (< 0.1% difference)
- Partial lensing improvement

### With strong boost (ξ_γ=0.5)
- κ̄_max ≈ 0.5-0.6 (3× improvement)
- Temperature still preserved
- Significant lensing enhancement

## Validation Tests

### 1. Temperature Preservation
The cluster temperature profile should be identical with and without photon boost:
```python
kT_standard = g3_no_boost.predict_cluster_temperature(rho_gas, rho_stars, r, n_e)
kT_boosted = g3_with_boost.predict_cluster_temperature(rho_gas, rho_stars, r, n_e)
assert np.max(np.abs(kT_boosted - kT_standard) / kT_standard) < 0.001
```

### 2. Solar System Safety
At Solar System densities (Σ ~ 10³ M☉/pc²):
```python
boost_solar = g3.photon_boost_factor(1000.0)  # High density
assert abs(boost_solar - 1.0) < 0.01  # Boost vanishes
```

### 3. Galaxy Rotation Curves
MW and SPARC galaxies unchanged since they test massive particle dynamics:
```python
v_circ_standard = np.sqrt(R * g_tot_dyn)  # Uses dynamics field
# Identical with or without photon boost
```

## Advantages Over Other Branches

1. **Physically motivated**: Disformal coupling is a known feature of scalar-tensor theories
2. **Automatic safety**: Screening makes it Solar System safe without tuning
3. **Preserves all dynamics**: Temperature, rotation curves completely unchanged
4. **Single parameter**: One ξ_γ might work for all clusters

## Limitations

1. **Still may not reach κ=1**: Even ξ_γ=0.5 gives κ̄_max ~ 0.6, below unity
2. **Requires relativistic theory**: Need full action to justify photon-matter difference
3. **Testable prediction**: Dynamics vs lensing mass discrepancy is observable
4. **Parameter constraint**: ξ_γ limited by weak lensing and CMB constraints

## Output Files

- `outputs/{cluster}_test.png` - Six-panel diagnostic plot
- `outputs/{cluster}_metrics.json` - Quantitative metrics
- `outputs/solar_safety.png` - Verification of Solar System safety

## Comparison Summary

| Test | Branch A Result | Branch B Result |
|------|-----------------|-----------------|
| Cluster κ̄_max | 0.4-0.8 (with boost) | 0.35-0.6 (with ξ_γ=0.3-0.5) |
| Galaxy impact | Small if r_boost>100kpc | None (dynamics unchanged) |
| Solar System | Safe (r<<r_boost) | Safe (screening) |
| Physical basis | Phenomenological | Disformal gravity |
| Parameters added | 4 (r_boost, q, η, ζ) | 2 (ξ_γ, β_γ) |
| EEP status | Preserved | Preserved |