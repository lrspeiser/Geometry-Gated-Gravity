# Squared Gravity (Geometric Exponent) Subproject

Purpose
- Explore scale-dependent gravity amplification where the effective boost exponent β grows with radius and/or decreasing surface density, without introducing dark matter.
- Keep all code and outputs here, separate from existing pipelines.

Core idea
- Replace a fixed excess with a geometric exponent that varies with Σ and scale R:
  - v_tot^2 = v_bar^2 × (1 + fX)^{β(Σ, R)}
  - For lensing (first-order proxy): Σ_eff(R) ≈ Σ_bar(R) × (1 + fX)^{β(Σ, R)}

β(Σ, R)
- β = 1 + γ1 · |Σ̂| + γ2 · log10(R/R_scale)
  - Σ̂ = log10(Σ_pc2 / 100)
  - R_scale ≈ 1 kpc (galaxies) or 100 kpc (clusters)
  - γ1, γ2 ≥ 0

Base excess fX (prototype)
- From ratio/curvature-style denominator:
  - fX = (x^2) / (a − b·Σ̂ − d·|d ln Σ/d ln r|) with x = R/Rd
  - denom clamped to > 0

Why this could help
- β grows at large R/low Σ → naturally increases effective mass profile at cluster radii without over-boosting galaxy cores.
- Smooth, continuous transition across scales; no piecewise switching.

Caveats and approximations (this subproject)
- For lensing, we start with a multiplicative proxy Σ_eff = Σ_bar × (1 + fX)^β. The correct treatment would recompute the 3D-to-2D mapping under the modified force law; this proxy is for exploration only.
- We use existing real baryon profiles (gas + stars) from data/clusters/<name> to build Σ_bar and its gradient.

What’s included
- geometric_exponent.py: Implements GeometricExponentGravity with β(Σ,R) and fX.
- cluster_runner.py: Uses cluster baryon profiles to compute Σ_bar(R) and predicts Einstein radii under the exponent model.
- scripts/fit_sg_clusters.py: Grid search over (γ1, γ2, a, b, d) on CLASH train/test; outputs predictions and metrics under data/clash/processed/squared_gravity/.

Usage (examples)
- Fit and evaluate CLASH:
  - py -u scripts/fit_sg_clusters.py --zs 2.0 --rscale 100 --rd 1000 \
        --gamma1 0.0:1.0:11 --gamma2 0.0:0.6:7 --a 0.3:1.5:7 --b 0.0:0.6:7 --d 0.0:0.6:7

Outputs
- data/clash/processed/squared_gravity/
  - einstein_radii_sg.csv (per cluster predictions)
  - eval_sg.json (best params and metrics)

Next steps
- Verify against SPARC galaxies with Rd-scale choices and check cross-scale consistency.
- If cluster θ_E deficit narrows (factor ~2–3), assess profile shapes (κ̄ vs R) and adjust β form.
