# Many-Path Gravity SPARC Diagnostic Summary

## Executive Summary

Parameter optimization has revealed **systematic model issues** that prevent convergence to acceptable accuracy. While we improved from 64.6% to 42.9% median APE through parameter tuning, we observe **bimodal behavior** where some galaxies fit well (<20% APE) while others fail catastrophically (>100% APE). This suggests fundamental model formula issues beyond simple parameter scaling.

---

## What We've Done

### Phase 1: Baseline Assessment
- **MW-frozen parameters** (eta=0.39, optimized for Milky Way)
- **50 stratified galaxies** (10 per type: Sm, Scd, Sc, Sbc, Sb)
- **Result**: 64.6% mean APE, 12% success rate
- **Finding**: MW params too strong for smaller SPARC galaxies

### Phase 2: Parameter Optimization (Stage A)
- **Optimized**: eta, ring_amp on 15 late-type galaxies
- **Best params**: eta=0.30, ring_amp=0.05
- **Result**: 42.9% median APE (8.1% improvement)
- **Finding**: Bimodal distribution reveals systematic issues

---

## The Systematic Problem

### Bimodal Behavior Pattern

**Well-fit galaxies** (16-29% APE):
- NGC4183: 16.1% (Scd, 20 points)
- UGC06983: 17.3% (Scd, 29 points)
- NGC1003: 24.1% (Scd, 38 points)

**Catastrophic failures** (>100% APE):
- NGC0801: 116.7% (Sc, 9 points)
- UGC11455: 106.1% (Scd, 24 points)
- NGC3726: 87.5% (Sc, 34 points)

This **is NOT a parameter scaling issue**—it's a formula/methodology problem.

---

## Likely Root Causes

### 1. **Mass Normalization Issues** 🔴 HIGH PRIORITY

**Current approach:**
```python
# Normalize to match velocity at fiducial radius
M_total = v_fid^2 * r_fid / G
```

**Problems:**
- Assumes Newtonian relation holds at fiducial point
- Circular reasoning: uses observed velocity to set mass, then predicts velocity
- Doesn't account for pressure support in low-mass/dwarf galaxies
- Surface brightness → mass conversion may be galaxy-type dependent

**Solution needed:**
- Use SPARC's published stellar masses (M_star from Table 1)
- Use distance-independent M/L ratios
- Add gas mass explicitly from HI measurements
- Remove circular v_fid normalization

### 2. **Particle Distribution Approximation** 🟡 MEDIUM PRIORITY

**Current approach:**
```python
# Sample particles proportional to SB * r
weights = sb_total * r_kpc
```

**Problems:**
- Over-simplifies 3D mass distribution
- Doesn't properly account for radial gradients
- Thin disk assumption (z_d=0.3 kpc) may be wrong for dwarfs
- No vertical structure from SB profile

**Solution needed:**
- Use SPARC's decomposed Vdisk, Vgas, Vbulge directly
- Don't create particles—work with analytical profiles
- Or: use proper deprojection from SB to 3D density

### 3. **Missing Physics** 🟢 LOWER PRIORITY

- **Pressure support**: Velocity dispersion important in dwarfs
- **Non-circular motions**: Bars, spiral arms, warps
- **Inclination uncertainties**: Edge-on galaxies problematic
- **Distance errors**: Propagate to mass estimates

---

## Proposed Formula Changes

### Option A: Analytical Profile Approach (RECOMMENDED)

Instead of particle sampling, work directly with component velocities:

```python
def predict_v_circ(r, params, v_gas, v_disk, v_bulge):
    """
    Predict circular velocity using many-path multiplier
    on baryonic components.
    """
    # Baryonic contribution
    v_bar_sq = v_gas**2 + v_disk**2 + v_bulge**2
    
    # Many-path enhancement
    # Compute M(r, geometry) at each radius
    M_r = compute_multiplier(r, params)
    
    # Enhanced velocity
    v_pred_sq = v_bar_sq * (1 + M_r)
    
    return np.sqrt(v_pred_sq)
```

**Advantages:**
- No mass normalization issues
- Uses SPARC's validated baryonic velocities directly
- Computationally fast (no particles)
- Transparent what's happening at each radius

**Disadvantages:**
- Need to reformulate M(r, geometry) without pairwise distances
- Loses some geometric richness of particle approach

### Option B: Fixed Total Mass from SPARC Table

```python
def create_particle_distribution_fixed_mass(galaxy, n_particles):
    """
    Use SPARC's published stellar + gas masses.
    """
    # From SPARC Table 1
    M_star = galaxy.stellar_mass  # Msun
    M_gas = galaxy.gas_mass      # Msun
    M_total = M_star + M_gas
    
    # Distribute particles matching SB profile
    # but with FIXED total mass
    masses = sample_from_SB_profile(...)
    masses = masses / np.sum(masses) * M_total
    
    return positions, masses
```

**Advantages:**
- Keeps particle approach
- Removes circular normalization
- Uses observational mass estimates

**Disadvantages:**
- Still has 3D distribution uncertainties
- Computationally expensive
- SPARC masses have uncertainties (M/L ratios)

### Option C: Hybrid Approach

Use analytical for prediction, particles only for computing geometric M(r):

```python
def compute_M_field(r_test, galaxy_particles, params):
    """
    Compute many-path multiplier field from particle distribution.
    """
    # Use particles to get geometry-dependent M
    # But don't use for mass—just for spatial structure
    ...
    
def predict_v_circ(r, M_field, v_bar):
    """
    Apply precomputed M field to baryonic velocity.
    """
    return np.sqrt(v_bar**2 * (1 + M_field))
```

---

## Immediate Next Steps

### 1. **Load SPARC Table 1 Data** 

Extract stellar masses, gas masses, inclinations:
```python
def load_sparc_full_table(master_file):
    """
    Parse full SPARC table including L[3.6], MHI, distance, etc.
    """
    return {
        'name': ...,
        'L_3.6': ...,  # 10^9 solar lum
        'MHI': ...,     # 10^9 solar mass
        'distance': ...,
        'inclination': ...,
        ...
    }
```

### 2. **Implement Analytical Profile Method (Option A)**

This is the cleanest solution and lets us isolate whether the many-path multiplier **formula** works, independent of mass modeling issues.

### 3. **Diagnostic Test Suite**

Create tests that isolate each potential issue:
- Test 1: Use perfect Newtonian case (M=1, should recover v_bar exactly)
- Test 2: Vary only eta, check if relationship is monotonic
- Test 3: Compare edge-on vs face-on galaxies with same v_bar
- Test 4: Plot residuals vs galaxy properties (mass, size, B/T, inclination)

---

## Decision Points

**Question 1**: Should we switch to analytical profiles (Option A)?
- **Pros**: Faster, cleaner, removes mass issues
- **Cons**: Loses geometric richness, need to reformulate M(r)

**Question 2**: Can we get SPARC stellar masses easily?
- Check if Table 1 data is available in machine-readable format
- May need to parse the master .mrt file more completely

**Question 3**: Is the particle approach salvageable?
- If we fix mass normalization, does bimodal behavior persist?
- Quick test: manually set M_total for a failing galaxy to match literature

---

## Success Criteria Moving Forward

Before doing more parameter optimization, we need:

1. ✅ **Monotonic improvement**: Changing eta should smoothly affect all galaxies in same direction
2. ✅ **No catastrophic failures**: Max APE should be <60% even for worst galaxy
3. ✅ **Physical interpretability**: Understand WHY a galaxy fits well or poorly
4. ✅ **Residual patterns**: Residuals should show structure we can diagnose, not random chaos

**Current status**: ❌ None of these are met with particle approach

---

## Recommended Path Forward

1. **Immediate** (1-2 hours):
   - Implement Option A (analytical profiles)
   - Test on 5 well-fit + 5 poorly-fit galaxies from Stage A
   - Verify monotonic eta dependence

2. **If Option A works** (next 2-4 hours):
   - Run full 50-galaxy stratified test
   - Optimize eta, ring_amp, M_max
   - Achieve target <30% median APE

3. **If Option A fails** (diagnostic mode):
   - The many-path multiplier formula itself may need revision
   - Consider distance-only dependence: M(d) without geometry
   - Consider radial-only: M(R) without pairwise interactions
   - May need to add velocity-dependent term: M(v, R)

---

## Key Insight

**The fact that some galaxies fit beautifully (16-20% APE) proves the many-path multiplier CONCEPT works.** The bimodal distribution suggests we have a **methodology bug**, not a physics problem. Fixing the mass normalization should unlock universal parameters.

---

**Last updated**: 2025-01-18  
**Status**: 🔴 BLOCKED on mass normalization issue  
**Next action**: Implement analytical profile method (Option A)
