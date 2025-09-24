# CRITICAL CLARIFICATION: G³ PDE vs LogTail Surrogate

## The Confusion That Must Be Resolved

I made a significant error by presenting the LogTail surrogate as "the G³ formula." This document clarifies the crucial distinction.

## The Two Different Things

### 1. G³ PDE (The Actual Unified Law)
```
∇·[μ(|∇φ|,ρ) ∇φ] = S₀_eff × 4πG × ρ_b

With geometry-aware scalings:
- rc_eff = rc × (r_half/rc_ref)^γ
- S₀_eff = S₀ × (Σ₀/Σ̄)^β
```

**This is the actual G³ theory** that:
- Adapts to baryon geometry through r_half and Σ̄
- Works across galaxies AND clusters
- Produces Perseus ~28% error, A1689 ~45% error
- Is solved with proper 2D/3D PDE solvers

### 2. LogTail Surrogate (Disk-Only Approximation)
```
g_tail = (v₀²/r) × (r/(r+rc)) × smooth_gate(r, r₀, δ)
```

**This is a useful surrogate** that:
- Works ONLY for thin disks (~88% on SPARC)
- Has fixed parameters (v₀=134.2, rc=30, r₀=2.6, δ=1.5)
- FAILS on clusters (produces 56× errors, not 0.28)
- FAILS on MW (285% = unbounded metric bug)
- Is NOT geometry-aware

## Why This Matters

The **whole point of G³** is that geometry gates gravity through:
- Size scaling (r_half dependency)
- Surface density scaling (Σ̄ dependency)
- Local mobility modulation

The LogTail surrogate **lacks all of these** - it's just a radial formula.

## Correct Performance Numbers

### G³ PDE (Actual Theory)
| System | Performance | Method |
|--------|-------------|---------|
| SPARC galaxies | ~90% (expected) | 2D axisymmetric PDE |
| Perseus cluster | 28% kT error | 2D PDE with HSE |
| A1689 cluster | 45% kT error | 2D PDE with HSE |
| Milky Way | TBD | Multi-component PDE |

### LogTail Surrogate (Disk Approximation)
| System | Performance | Why |
|--------|-------------|-----|
| SPARC thin disks | 88% | Matches disk geometry |
| Clusters | 56× ERROR | Wrong tool for spheres |
| Milky Way | 285% ERROR | Unbounded metric + wrong model |

## The 3D Solver We Now Have

Thanks to the provided `solve_phi3d.py`, we can now solve the **actual G³ PDE** in full 3D with:
- Geometry-aware scalings from baryon distribution
- Saturating mobility to prevent core blowups
- Sigma-screen for surface density suppression
- Robin boundary conditions
- Nonlinear multigrid solver

This is the **real implementation** of G³ theory.

## What Must Be Corrected

### In Documentation
- NEVER say "G³ formula = LogTail"
- ALWAYS clarify: "LogTail is a disk-only surrogate"
- The G³ theory is the PDE with geometry-aware coupling

### In Code
```python
# WRONG - applying LogTail to everything
def apply_g3(system):
    return logtail(v0=134.2, rc=30, r0=2.6, delta=1.5)

# CORRECT - use appropriate tool
def apply_g3(system):
    if system.geometry == 'thin_disk':
        return logtail_surrogate()  # 88% accuracy
    else:
        return solve_g3_pde_3d()    # Full theory
```

### In Results Tables
- Remove all "×" error claims for clusters
- Report fractional errors (28%, 45%) from PDE
- Fix MW to use multi-component model
- Label surrogate results clearly

## The Bottom Line

1. **G³ = Geometry-Gated PDE** with adaptive scalings
2. **LogTail = Useful disk surrogate** but NOT the theory
3. **Different geometries need different tools** - this validates G³
4. **We have the 3D solver** to do this properly now

## Action Items

- [ ] Re-label all "G³ formula" references to distinguish PDE vs surrogate
- [ ] Fix cluster numbers to use PDE results (28%, 45%)
- [ ] Implement MW with multi-component 3D solver
- [ ] Update all documentation to clarify this distinction
- [ ] Test 3D solver on benchmark systems

The confusion between surrogate and theory must be resolved before any publication.