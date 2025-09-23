# Geometric Enhancement Theory

## Core Principle

Instead of requiring per-galaxy tuning, we introduce a **geometric enhancement factor** that naturally amplifies the gravitational response where baryonic density thins out rapidly. This is motivated by the observation that gravity appears "too weak" precisely where baryons become sparse.

## Mathematical Framework

### 1. Density Gradient Enhancement

The enhancement factor κ(r) responds to the local gradient of baryon density:

```
κ(r) = 1 + λ * |∇log(ρ_b)|^α / (1 + |∇log(ρ_b)|^α / β)
```

Where:
- λ: maximum enhancement amplitude (global)
- α: gradient sensitivity exponent (global) 
- β: saturation scale (global)

This naturally provides:
- κ ≈ 1 in dense regions (small gradients)
- κ > 1 in transition zones (large gradients)
- κ saturates at (1 + λ) in very sparse regions

### 2. Geometric Scale Coupling

The enhancement couples to the system's characteristic scale:

```
λ_eff = λ_0 * (r_half / r_ref)^γ
```

Where r_half is the half-mass radius, ensuring:
- Galaxies get appropriate enhancement based on size
- Clusters get scaled enhancement based on their extent
- No per-object tuning required

### 3. Modified Field Equation

The PDE becomes:

```
∇·[μ(|∇φ|) ∇φ] = S_0 * κ(r) * ρ_b
```

Or equivalently for the total acceleration:

```
g_tot = g_N + κ(r) * g_φ
```

### 4. Key Properties

1. **Universal**: Same parameters for all systems
2. **Geometric**: Responds to baryon distribution shape
3. **Smooth**: No discontinuities or sharp transitions
4. **Physical**: Enhancement where baryons thin = where binding weakens
5. **Conservative**: Returns to GR in high-density regions

### 5. Implementation Strategy

For clusters:
- Compute ∇log(ρ) from total baryon profile
- Apply enhancement in regions of rapid density fall-off
- This naturally boosts gravity at cluster periphery

For galaxies:
- Disk edges and halos get enhanced binding
- Central regions remain near-GR
- Flat rotation curves emerge naturally

### 6. Parameter Expectations

Based on the geometry-gated philosophy:
- λ_0 ≈ 0.3-1.0 (modest enhancement)
- α ≈ 1.5-2.0 (quadratic-like response)
- β ≈ 1.0 (natural units)
- γ ≈ 0.5 (square-root scaling with size)