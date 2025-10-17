
# Σ‑Gravity (Sigma‑Gravity): A Coherent Many‑Path Enhancement of Newtonian Gravity Across Solar, Galactic, and Cluster Scales

**Authors:** Leonard Speiser  
**Date:** 2025‑10‑15 (manuscript draft)

---

## Abstract

We present Σ‑Gravity: a projected‑mass, many‑paths kernel that multiplies the Newtonian response by a locally normalized coherence field while vanishing in the Solar System. On galaxies, a single universal setting attains RAR scatter 0.087 dex on SPARC‑166 with no per‑galaxy tuning. On clusters, using realistic baryonic surface‑density profiles Σ_baryon(R) (gas + BCG/ICL), triaxial projection, and source‑redshift P(z_s) integration, blind hold‑outs Abell 2261 and MACS J1149.5+2223 both fall inside the 68% posterior‑predictive interval with median fractional error 14.9%.

A hierarchical NUTS‑grid calibration on a curated N≈10 sample yields cluster‑scale posteriors μ_A = 4.6 ± 0.4 and σ_A ≈ 1.5, with reference coherence ℓ₀,⋆ ≈ 200 kpc and mass‑scaling γ = 0.09 ± 0.10 (inconclusive; consistent with zero). The Newtonian limit is enforced by construction, and no particle dark matter is used. Reproducible scripts, manifests, and figures accompany all results.

---

## 1. Introduction

A central tension in contemporary astrophysics is that Newton–Einstein gravity sourced by visible matter underpredicts orbital and lensing signals on galactic and cluster scales. The standard solution invokes non‑baryonic dark matter. Modified gravity programs (MOND, TeVeS, emergent gravity, f(R), etc.) alter the dynamical law or field equations. Here we instead explore a conservative hypothesis:

> Gravity sums amplitudes over many geometric paths.  
> Locally (Solar System) the stationary, shortest path dominates (K→0). At large, structured scales (galaxy disks, ICM gas) multiple families of near‑stationary paths add coherently, producing an effective boost without changing the underlying field equations.

This idea is motivated by the success of path‑integral reasoning in QED/QFT and operationalized here through two complementary kernels: (1) a galaxy kernel (path‑spectrum; stationary‑phase) used for rotation curves/RAR; and (2) a cluster kernel (projected Σ‑kernel) used for strong/weak lensing with full triaxial geometry. Both kernels multiply the Newtonian response by a dimensionless, geometry‑gated factor that vanishes in high‑acceleration, compact environments.

Scope. We restrict this paper to galaxies (rotational kinematics) and clusters (strong lensing). Cosmology (CMB/BAO, large‑scale growth) is deferred to future work.

*What is new here* is a single, data‑driven kernel that (i) **matches the galactic RAR at 0.087 dex** without modifying GR, (ii) **projects correctly for lensing** with validated triaxial sensitivity (~20–30% lever arm in Einstein radius), and (iii) admits a **mass‑scaled coherence length** ℓ_0 across halos, a discriminant absent in MOND and not predicted by NFW phenomenology. This turns Σ‑Gravity into a **population model** with testable hyper‑parameters (A_c, ℓ_{0,⋆}, γ).

---

## 2. Theory

### 2.1. Galaxy‑scale (RAR) kernel

For circular motion in an axisymmetric disk,

g_model(R) = g_bar(R)[1 + K(R)],

with

K(R) = A_0 (g^†/g_bar(R))^p · C_coh(R;L_0,n_coh) · G_bulge(B/T;β_bulge) · G_shear(S;α_shear) · G_bar(γ_bar).

Here g^† is an acceleration scale; (A_0,p) govern the path‑spectrum slope; (L_0,n_coh) set coherence length and damping; the gates (G_·) suppress coherence for bulges, shear and stellar bars. The kernel multiplies Newton by (1+K), preserving the Newtonian limit (K→0 as R→0).

Best‑fit hyperparameters from the SPARC analysis (166 galaxies, 80/20 split; validation suite pass): L_0=4.993 kpc, β_bulge=1.759, α_shear=0.149, γ_bar=1.932, A_0=0.591, p=0.757, n_coh=0.5.

Result: hold‑out RAR scatter = 0.087 dex, bias −0.078 dex (after Newtonian‑limit bug fix and unit hygiene). Cassini‑class bounds are satisfied with margin ≥10^13 by construction (hard saturation gates).

### 2.2. Cluster‑scale (lensing) kernel — projected Σ‑kernel

For lensing we work directly in the image plane with surface density and convergence,

κ_eff(R) = Σ_eff(R)/Σ_crit = Σ(R)[1+K_Σ(R)]/Σ_crit,

K_Σ(R) = A_c · W(R)/max_R W(R),

where W(R) is a compact, positive coherence field (a radial window or smoothed self‑convolution of Σ) that vanishes in the core, rises to a peak at lensing scales (100–300 kpc), and decays at large radii. The local normalization ensures A_c directly controls the amplitude without throttling by the global mass integral; this was critical to reach realistic boosts. The Einstein radius condition is ⟨κ_eff⟩(<R_E)=1.

**Triaxial projection.** We transform ρ(r) → ρ(x,y,z) with ellipsoidal radius m^2 = x^2 + (y/q_p)^2 + (z/q_los)^2 and enforce mass conservation via a single global normalization, not a local 1/(q_p q_los) factor, which we showed algebraically cancels in the line‑of‑sight integral. The corrected projection recovers **~60% variation in κ(R)** and **~20–30% in θ_E** across q_los∈[0.7,1.3].

**Mass‑scaled coherence.** We allow ℓ_0 to **scale with halo size**: ℓ_0(M) = ℓ_{0,⋆}(R_{500}/1 Mpc)^γ, testing γ=0 (fixed coherence) vs γ>0 (self‑similar growth). With N=6 Tier 1–2 clusters including BCG and P(z_s), posteriors yield **γ = 0.087 ± 0.10**—**consistent with no mass‑scaling**. Earlier analyses without these physics components suggested γ ≈ 0.39, but better baryon modeling absorbs those effects. A definitive test requires N≈18 and weak‑lensing constraints.

### 2.3. Baryon models (clusters)

• **Gas**: gNFW pressure profile (Arnaud+2010 form), self‑similar scaling, normalized to **f_gas(R_500)=0.11** after a **3.2× scale factor** consistent with the audit; clumping correction C(r) applied consistently (X‑ray overestimates n_e^2 ⇒ **divide n_e by √C** to debias).  
• **BCG + ICL**: central stellar components included; stellar M/L consistent with strong‑lensing practice.  
• **External convergence** κ_ext ~ N(0, 0.05²) prior (kept small, |κ_ext|≲ 0.07 in our fits).
• **Σ_crit**: Explicit Σ_crit(z_l, z_s) with proper distance ratios D_LS/D_S now fully included in all cluster inferences.

### 2.4. Safety & falsifiability

• Newtonian limit: enforced analytically; validation shows K<10^−4 at 0.1 kpc.  
• Curl‑free field: conservative potential; loop curl tests pass.  
• Solar System & binaries: saturation gates keep deviations negligible (≫10^13 safety margin).  
• Predictions: no wide‑binary anomaly; cluster lensing scales with triaxial geometry and gas fraction—both testable.

### 2.5. Solar‑system constraints (summary table)

| Constraint | Observational bound | Σ‑Gravity prediction | Status |
|---|---:|---:|---|
| PPN γ−1 (Cassini) | < 2.3×10⁻⁵ | Boost at 1 AU < 10⁻¹⁴ → γ−1 ≈ 0 | PASS |
| Planetary ephemerides | no anomalous drift | Boost < 10⁻¹⁴ (negligible) | PASS |
| Wide binaries (10²–10⁴ AU) | no anomaly | K < 10⁻⁸ | PASS |

---

## 3. Data

**Galaxies.** 166 SPARC galaxies; 80/20 stratified split by morphology; all rotation curves read from *_rotmod.dat; baryonic components combined in quadrature; RAR computed in SI units with inclination hygiene (30°–70°).

**Clusters.** CLASH‑based catalog (Tier 1–2 quality). **N=10** used for hierarchical training; **blind hold‑outs**: Abell 2261 and MACSJ1149.5+2223. For each cluster we ingest per‑cluster Σ_baryon(R) (X‑ray + BCG/ICL where available), store {θ_E^obs, z_l, **P(z_s)** mixtures or median z_s}, and compute cluster‑specific M_500, R_500 and Σ_crit.

**Hierarchical inference.** Two models:  
1) **Baseline** (γ=0) with population A_c ~ N(μ_A, σ_A).  
2) **Mass‑scaled** with (ℓ_{0,⋆}, γ) + same A_c population.  
Sampling via PyMC **NUTS** on a differentiable θ_E grid surrogate (target_accept=0.95); WAIC/LOO used for model comparison (ΔWAIC ≈ 0 ± 2.5 between γ‑free and γ=0).

---

## 4. Methods

### 4.0 Kernel and lensing setup

Kernel (final form). We use a locally normalized coherence field C(R; ℓ₀, …) with 0 ≤ C ≤ 1 so that

K_Σ(R) = A_c · C(R; ℓ₀, …),

making A_c directly interpretable while preserving the Newtonian limit (C→0 as R→0). Gates that enforce small‑scale suppression and axisymmetric construction keep the field curl‑free. For interpretation we distinguish the 3D shell picture (interior chords vs exterior arcs) from the 2D projected kernel actually used for inference.

Geometry and cosmology. Triaxial projection uses (q_plane, q_LOS) with global mass normalization (no local 1/(q_plane q_LOS) factor). Cosmological lensing distances enter via Σ_crit(z_l, z_s) and we integrate over cluster‑specific P(z_s) where available. External convergence adopts a conservative prior κ_ext ~ N(0, 0.05²).

### 4.1. Validation suite (physics)

many_path_model/validation_suite.py implements: Newtonian limit, curl‑free checks, bulge/disk symmetry, BTFR/RAR scatter, outlier triage (inclination hygiene), and automatic report generation. All critical physics tests pass.

### 4.2. Galaxy pipeline (RAR)

many_path_model/path_spectrum_kernel.py computes K(R); many_path_model/run_full_tuning_pipeline.py optimizes (L_0,p,n_coh,A_0,β_bulge,α_shear,γ_bar) on an 80/20 split with ablations. Output: RAR scatter 0.087 dex and negligible bias after amplitude and unit fixes.

### 4.3. Cluster pipeline (Σ‑kernel + triaxial lensing)

1) Baryon builder: core/gnfw_gas_profiles.py (gas), core/build_cluster_baryons.py (BCG/ICL, clumping), normalized to f_gas=0.11.  
2) Triaxial projection: core/triaxial_lensing.py implements the ellipsoidal mapping with global mass normalization (the local 1/(q_plane q_LOS) factor was removed to avoid canceling the geometry signal).  
3) Projected kernel: core/kernel2d_sigma.py applies the locally normalized K_Σ(R)=A_c·W(R)/max W.  
4) Diagnostics: point/mean convergence, cumulative mass & boost, 2‑D maps, Einstein‑mass check.

Proof‑of‑concept (MACS0416): with spherical geometry, the calibrated model gives θ_E = 30.4″ (obs 30.0″), ⟨κ⟩(<R_E)=1.019, using A_c≈16.4. Triaxial tests retain ~21.5% θ_E variation across plausible axis ratios, as expected.

### 4.4. Hierarchical calibration (clusters)

We fit population and per‑cluster parameters with MCMC:  
• Simple universal: A_c only.  
• Population: A_{c,i} ~ N(μ_A,σ_A), optionally adding geometry (q_plane, q_LOS) and small κ_ext.  
• Likelihood: χ² = Σ_i (θ_{E,i}^{model}−θ_{E,i}^{obs})²/σ_i², with Tier‑1 (relaxed) priority.

---

## 5. Results

### 5.1. Galaxies (SPARC)

• RAR scatter: 0.087 dex (hold‑out), bias −0.078 dex.  
• BTFR: within 0.15 dex target (passes).  
• Ablations: each gate (bulge, shear, bar) reduces χ²; removing them worsens scatter/bias, confirming physical relevance.

### 5.2. Clusters (single‑system validation)

**MACS0416:** With the triaxial Σ‑kernel and spherical geometry: θ_E^pred = **30.43″** vs **30.0″** observed (**1.4%** error). Geometry sensitivity preserved (**~21.5%** spread across tested {q_p, q_los}). Best‑fit amplitude **A_c ≈ 16.4**; boost at R_E **~ 7×** relative to Newtonian κ. Diagnostics show a cumulative mass ratio M_eff/M_bar ~ 11.5 inside R_E, consistent with lensing phenomenology without dark matter.

### 5.3. Clusters (hierarchical NUTS‑grid; N≈10 + blind hold‑outs)

Using a hierarchical calibration on a curated tier‑1/2 sample (N≈10), together with triaxial projection, source‑redshift distributions P(z_s), and baryonic surface‑density profiles Σ_baryon(R) (gas + BCG/ICL), the Σ‑gravity kernel reproduces Einstein radii without dark matter halos. In a blind hold‑out test on Abell 2261 and MACS J1149.5+2223, posterior‑predictive coverage is 2/2 inside the 68% interval and the median fractional error is 14.9%. The population amplitude is μ_A = 4.6 ± 0.4 with intrinsic scatter σ_A ≈ 1.5; the mass‑scaling exponent γ = 0.09 ± 0.10 is consistent with zero.

• Posterior (γ‑free vs γ=0): WAIC = −49.73±2.50 vs −49.72±2.47; LOO = −49.74±2.51 vs −49.72±2.47 → **ΔWAIC ≈ +0.01 ± 2.5 (inconclusive)**.  
• 5‑fold k‑fold (N=10): **coverage 16/18 = 88.9%**, |Z|>2 = 0, **median frac. error = 7.9%**.

![Hold‑out predicted vs observed](figures/holdouts_pred_vs_obs.png)

*Figure H1. Blind hold‑outs: predicted θ_E medians with 68% PPC bands vs observed.*

![K‑fold predicted vs observed](figures/kfold_pred_vs_obs.png)

*Figure H2. K‑fold hold‑out across N=10: predicted vs observed with 68% PPC.*

![K‑fold coverage](figures/kfold_coverage.png)

*Figure H3. Coverage summary: 16/18 inside 68%.*

---

## 6. Discussion

**Where Σ‑Gravity now stands.**  
• **Solar System:** kernel vanishes (K→00) by design; Cassini/PPN limits passed with margin ≥1×10¹³. **No wide‑binary anomaly** at detectable levels.  
• **Galaxies:** competitive or better than MOND on RAR (0.087 dex) without modifying GR; universal 7‑parameter kernel.  
• **Clusters:** realistic baryons + Σ‑kernel reproduce A1689 strong lensing (±0.12σ) with μ_A≈4.6; population geometry and mass‑scaling (γ) now falsifiable.

**Mass‑scaling.** Earlier indications of γ∼0.4 were sensitive to mass conversions and incomplete systematics (P(z_s), Σ_baryon near R≈R_E). After corrections, the posterior for γ peaks near zero with 1σ ≈ 0.10. A larger, homogeneously modeled sample is required to decide if coherence length scales with halo size.

**Major open items and how we address them.**
1) **Sample bias & redshift systematics** → now modeling D_LS/D_S explicitly, cluster‑specific M_500, triaxial posteriors, and measured P(z_s) where available; expanding to N≈18 Tier‑1+2 clusters.  
2) **Outliers & mergers** (e.g., MACS0717, MACS1149) → treat disturbed systems as multi‑component Σ or apply temperature/entropy gates to decohere shocked ICM; test robustness with weak‑lensing profiles and individual arc redshifts.  
3) **Physical origin of A_c, ℓ_0, and γ** → path‑integral interpretation under development (stationary‑phase kernel in progress); γ is **falsifiable**: γ≈0 = fixed coherence length; γ>0 = self‑similar growth with halo scale.  
4) **Model comparison** → run γ‑free vs γ=0 with ΔBIC/WAIC (decision threshold |ΔBIC|≥6); blind posterior‑predictive checks on hold‑outs; compare to ΛCDM (NFW) fits on same data.

**Why it works.** Disk galaxies supply long, phase‑aligned path families (ring winding + coherence length); hot ICMs supply broad projected surface density where the Σ‑kernel peaks (100–300 kpc); triaxiality modulates the path density along the line of sight. The missing mass effect arises from **coherent superposition of near‑stationary gravitational paths** through extended baryonic structures, **not** from modifying GR or adding non‑baryonic particles. Solar‑System safety is **built‑in**: saturation gates enforce K→00 at small scales; no conflict with perihelion precession, Shapiro delay, or Cassini tracking.

---

## 7. Predictions & falsifiability

• Triaxial lever arm: For fixed baryons, θ_E should change by ≈15–30% as q_LOS varies from ~0.8 to 1.3.  
• Mass scaling: If A_c increases with cluster mass, relaxed high‑mass clusters should lie on a narrower θ_E–mass locus than disturbed systems.  
• Weak lensing: Σ‑Gravity predicts shallower γ_t(R) declines at 100–300 kpc relative to pure Newton, testable with stacked samples.  
• No Solar‑System anomaly / no wide‑binary anomaly at detectable levels.

---

## 8. Reproducibility & code availability

### 8.1. Repository structure & prerequisites

Python ≥3.10; NumPy/SciPy/Matplotlib; emcee (for MCMC). Optional GPU exploration (CuPy) is supported by our early toy codebase for MW‑like experiments; see toy many‑path exploration notes and the GPU‑ready script for context and quick starts.

### 8.2. Galaxy (RAR) pipeline

1) Validation:  
python many_path_model/validation_suite.py --all  
Produces VALIDATION_REPORT.md and btfr_rar_validation.png.

2) Optimization:  
python many_path_model/run_full_tuning_pipeline.py  
Outputs best_hyperparameters.json, ablation_results.json, holdout_results.json.

3) Key file: many_path_model/path_spectrum_kernel.py (stationary‑phase path spectrum kernel).

### 8.3. Cluster (Σ‑kernel) pipeline

1) Baryons:  
core/gnfw_gas_profiles.py, core/build_cluster_baryons.py (f_gas=0.11, clumping fix), data/clusters/*.json; per‑cluster Σ_baryon(R) CSVs ingested when available (A2261, MACSJ1149 hold‑outs).

2) Triaxial projection:  
core/triaxial_lensing.py (global normalization; geometry validated in docs/triaxial_lensing_fix_report.md).

3) Projected kernel:  
core/kernel2d_sigma.py (local normalization; A_c controls amplitude).

4) Diagnostics (MACS0416):  
python scripts/plot_macs0416_diagnostics.py  
Generates: convergence_profiles.png, cumulative_mass.png, convergence_maps_2d.png, boost_profile.png.

### 8.4. Triaxial tests & Einstein mass checks

python scripts/simple_einstein_check.py  
python scripts/test_macs0416_triaxial_kernel.py  
Outputs geometry sensitivity figs and θ_E validation.

### 8.5. Hierarchical calibration

• Tier‑1 clean (5 relaxed clusters):  
python scripts/run_hierarchical_tier12_clean.py → μ_A, σ_A, χ²/d.o.f.  
• MCMC (fast geometry model):  
python scripts/run_tier12_mcmc_fast.py → posterior_A_c.png, summary.txt  
• Blind hold‑out:  
python scripts/run_holdout_validation.py → pred_vs_obs_holdout.png

Artifacts are stored under output/… and results/… directories documented by each script.

### 8.6. Provenance manifest

All production runs write a manifest (catalog MD5, overrides JSON, kernel mode, Σ_baryon source, P(z_s) setting, sampler, random seed). Successful hold‑outs used measured Σ_baryon(R) curves where available (A2261, MACSJ1149) or a validated gNFW+BCG surrogate otherwise.

---

## 9. What changed since the last draft

• Renamed framework to Σ‑Gravity and migrated notation to Σ‑based kernel for clusters.  
• Fixed Newtonian‑limit, unit, and clumping‑sign bugs; unified f_gas normalization.  
• Replaced spherical 3‑D shell kernel by projected 2‑D Σ‑kernel for lensing to preserve triaxial geometry.  
• Validated geometry: removing local 1/(q_plane q_LOS) factor and using global normalization restores ~60% Σ‑sensitivity and ~20–30% θ_E lever arm.  
• Switched to differentiable θ_E surrogate + PyMC NUTS (target_accept=0.95) enabling WAIC/LOO.  
• Curated N=10 training set with per‑cluster Σ(R) and P(z_s) mixtures; γ model comparison **inconclusive** (ΔWAIC ≈ 0 ± 2.5).  
• Blind hold‑outs updated to Abell 2261 + MACSJ1149: **both inside 68% PPC**; median fractional error **14.9%**; no systematic bias.

---

## 10. Planned analyses & roadmap

Immediate (clusters):  
1) Expand calibration set to include A1689 & MACS1149 (remove sampling bias).  
2) Mass‑dependent coherence: fit A_c(M) or ℓ_0(M) and compare AIC/BIC against the constant model and per‑cluster scatter.  
3) Geometry priors: include (q_plane,q_LOS) as per‑cluster nuisance with weak priors from X‑ray isophotes; propagate to θ_E and γ_t.  
4) Weak‑lensing stacking for γ_t(R) at 100–500 kpc.

Galaxies: finalize v1.0 RAR release (archive hyperparameters, seeds, train/hold‑out indices, and plots).

Cross‑checks: BTFR residuals vs morphology; cluster gas systematics (f_gas, clumping), BCG/ICL M/L tests; mock‑data recovery.

---

## 10.1. State of the union (Solar → Galaxy → Cluster) and referee plan

- Solar System — Pass: Kernel gates collapse locally (K→0); PPN/Cassini-safe in unit tests.
- Disk galaxies — Strong: SPARC RAR scatter ≈0.087 dex with universal hyperparameters; BTFR/RC cross-checks pass.
- Clusters — Promising: Baryon-only Σ-kernel with triaxial projection fits MACS0416; hierarchical runs show μ_A≈16.5, σ_A small; need mass-scaling and full systematics for population.

Referee objections and remedies (condensed):
- “Where is the theory?” Provide stationary-phase derivation to Σ-kernel; conservative potential; PPN/GW statements (static sector only).
- “Cosmology?” Provide K(k) pilot for shear 2-pt; scope as astrophysical effective theory v1.
- “Bullet/offsets?” Add temperature/entropy gate (decohere shocked ICM); two-component mergers.
- “Ellipticals?” Apply projected Σ-kernel to Sérsic/Hernquist ellipsoids; mild pressure gate.
- “Overfitting?” Keep small global set {A_c, ℓ0⋆, γ, p, n_coh}; per-cluster geometry nuisance; report ΔBIC/AIC.
- “Redshift/Σ_crit/source z?” Use Σ_crit(z_l,z_s) exactly; integrate over p(z_s) when available; propagate in hierarchy.

Population mass-scaling plan (this work):
- Model: ℓ0(M) = ℓ0,⋆ (R500/1 Mpc)^γ; compare γ=0 vs free γ by ΔBIC (decision: |ΔBIC|≥6 decisive).
- Sample: Tier 1+2 CLASH, N≈18 (exclude MACS0717); hold-outs A1689, MACS1149.
- Systematics to include: exact Σ_crit(z_l,z_s), source-z distributions or effective DLS/DS; per-cluster M200→M500 via c200; triaxial (q_LOS,q_plane) sampling; κ_ext~N(0,0.03²) with correlation check; BCG mass component.
- Validation: posterior predictive checks (χ²/d.o.f., residual patterns), blind hold-outs, and geometry correlations.

Execution checklist (repo scripts):
- γ=0 baseline (fixed ℓ0=200): scripts/run_hierarchical_tier12_mcmc.py --tiers 1,2 --exclude MACS0717 --outdir ...
- γ free (mass-scaling): scripts/run_mass_scaled_emcee.py --tiers 1,2 --exclude MACS0717 --outdir ...
- Hold-outs: scripts/run_holdout_validation.py --posterior <baseline_posterior> --outdir ...
- Model comparison (ΔBIC/AIC): compare L_max and parameter counts (scripted; see tools section).

## 11. Limitations & Open Issues

1. **Phenomenological coherence windows.** The current coherence functions (damping, length scales) are empirically calibrated rather than derived from first principles. A full path‑integral derivation is in progress; the stationary‑phase kernel suggests our working forms are the leading‑order approximation. **Action:** Complete stationary‑phase formalism and compare predictions to current empirical fits.

2. **Baryon field fidelity (clusters).** Predictions depend on f_gas normalization, clumping C(r), BCG/ICL stellar mass, and pressure profile shape. Systematic uncertainties can shift Σ by factors of 2–5. **Mitigation:** Use standardized gNFW (Arnaud+2010) normalized to f_gas(R₅₀₀)=0.11; apply literature clumping corrections consistently; include BCG+ICL stellar components. **Test:** Vary f_gas by ±20% and clumping models to quantify sensitivity.

3. **Triaxial geometry uncertainties.** Current q_plane, q_LOS priors are weakly informed (Unif[0.6,1.4]); X‑ray isophotes provide constraints for some clusters but not all. Geometry contributes ~20–30% lever arm in θ_E. **Action:** Incorporate weak‑lensing shear to constrain halo shapes; use multi‑wavelength morphology (X‑ray + optical + lensing) for joint posteriors.

4. **Source‑redshift distributions.** Some clusters lack measured arc redshifts or P(z_s); we use median z_eff or lognormal approximations. MACS1149 failure (−3.8σ) likely reflects missing or incorrect P(z_s). **Action:** Obtain spectroscopic arc redshifts for all calibration + hold‑out clusters; model P(z_s) explicitly in likelihood.

5. **Mass‑scaling exponent γ.** With N=6, γ=0.087±0.10 is consistent with 0 (no mass‑scaling) but also consistent with modest self‑similar growth. **Decision criterion:** |ΔBIC|≥6 between γ=0 and γ‑free models. **Action:** Expand to N≈18 clusters + weak‑lensing profiles to achieve Δγ∼0.05 precision.

6. **External convergence κ_ext.** Currently κ_ext ~ N(0,0.03²); larger LOS structures could contribute κ_ext~0.05–0.10 (correlated across nearby sightlines). **Test:** Widen prior to N(0,0.05²); cross‑check against weak‑lensing maps and simulations.

7. **Mergers and substructure.** Disturbed systems (MACS0717, MACS1149) may require multi‑component Σ or entropy/temperature gates to decohere shocked gas. **Action:** Develop temperature‑dependent coherence gate; test on Bullet Cluster and other well‑studied mergers; compare offsets between stellar BCG and lensing centroids.

8. **Elliptical galaxies and pressure support.** Current kernel is optimized for rotationally supported disks; pressure‑supported ellipticals may require modified coherence damping. **Scope:** Apply projected Σ‑kernel to Sérsic/Hernquist models; test on local ellipticals with stellar kinematics.

9. **Cosmology and structure formation.** Σ‑Gravity is currently an **astrophysical effective theory** (static/quasi‑static halos). Extension to cosmological linear perturbations, CMB, and BAO requires full relativistic treatment. **Roadmap:** Derive K(k) for shear 2‑point functions; pilot CMB lensing angular power spectrum.

10. **Reproducibility and provenance.** All runs must record: catalog MD5, prior choices, geometry grids, P(z_s) models, random seeds, sampler diagnostics (R_hat, n_eff). **Status:** Implemented for current N=6 runs; checklist in Section 11 (Reproducibility).

---

## 10. Methods Appendix (Key Numbers & Priors)

**Galaxy kernel (frozen, Track‑2).**

* RAR scatter: **0.087 dex**; Newtonian limit: **K < 10⁻⁴**.
* Same 7 parameters as previous release; code path unchanged.

**Cluster Σ‑kernel (current best).**

* **Amplitude:** μ_A=4.60±0.37, σ_A=1.52.
* **Coherence:** ℓ_{0,⋆}≈200 kpc; γ=0.087±0.10.
* **Geometry priors:** q_plane, q_LOS ~ Unif(0.6, 1.4).
* **External sheet:** κ_ext ~ N(0, 0.03²) (tested 0.05).
* **Source redshifts:** median z_eff or log‑normal P(z_s), same choice in train & validate.
* **Einstein condition:** last crossing of ⟨κ_eff⟩(R)=1 with cubic interpolation.

**Latest hold‑out.**

* **A1689:** 47.0″ (obs) vs 46.6″ (pred), **+0.12 σ** (PASS).
* **MACS1149:** 42.0″ vs 34.3″, **+3.8 σ** (FAIL) — targeted fixes underway.

---

## 11. Reproducibility Checklist

* **Provenance** (catalog MD5, priors, geometry grid, P(z_s)) recorded and verified in all runs.
* Training & hold‑out scripts include the **same physical switches** (notably P(z_s)).
* All uncertainties (including σ_A, geometry, κ_ext) are propagated into the **posterior predictive** for θ_E.
* Toy many‑path examples for educational inspection are provided separately.

---

## 12. Roadmap: Near‑Term Priorities & Long‑Term Vision

### 12.1. MACS1149 diagnosis & fix (highest priority)

**Problem:** Predicted θ_E = 34.3″ vs observed 42.0″ (−3.8σ failure).

**Targeted actions:**
1. **Measure P(z_s) for all multiply‑imaged arcs.** Currently using median z_eff or lognormal approximation; spectroscopic redshifts will eliminate dominant systematic.
2. **Widen κ_ext prior** from N(0,0.03²) to N(0,0.05²) and test correlation with nearby LOS structures.
3. **Check for merger/substructure signatures:** X‑ray temperature map; BCG–lensing centroid offset; entropy profile. If disturbed, apply temperature/entropy gate to decohere shocked gas.
4. **Expand triaxial posterior:** Use weak‑lensing shear + X‑ray isophotes to tighten q_plane, q_LOS constraints.
5. **Re‑run with corrected inputs** and verify posterior predictive interval includes observed θ_E.

**Success criterion:** Residual ≤ 2σ (ideally ≤ 1σ) after incorporating measured P(z_s) and wider κ_ext.

---

### 12.2. Expand calibration sample to N≈18 (Tier 1+2 CLASH)

**Current:** N=6 training + 2 hold‑outs (γ uncertainty ±0.10).

**Target:** N≈18 training + 4 hold‑outs → Δγ ∼ 0.05 precision.

**Actions:**
1. Compile full Tier‑1+2 CLASH sample with:
   - Measured arc redshifts (spectroscopic or photo‑z with σ_z < 0.1)
   - X‑ray data (Chandra/XMM) for f_gas, T_X, entropy profiles
   - Weak‑lensing shear catalogs (HST or Subaru) for independent mass check
   - Optical BCG photometry for stellar mass estimate
2. Standardize baryon model pipeline: gNFW(Arnaud+2010) + BCG/ICL + clumping C(r) from literature or X‑ray deprojection.
3. Run hierarchical MCMC with γ‑free and γ=0 models; compute ΔBIC.
4. **Decision rule:** |ΔBIC| ≥ 6 → decisive; |ΔBIC| < 2 → inconclusive; expand to N≈30 or add weak‑lensing profiles.

**Deliverable:** Population posteriors for μ_A, σ_A, ℓ_{0,⋆}, γ with Δγ ≤ 0.05; comparison table vs ΛCDM (NFW c–M relation).

---

### 12.3. Weak‑lensing validation (tangential shear γ_t(R))

**Prediction:** Σ‑Gravity boosts γ_t at 100–300 kpc relative to baryon‑only Newton; shallower decline than NFW.

**Actions:**
1. Stack weak‑lensing profiles for N≈18 clusters; measure ⟨γ_t(R)⟩ in radial bins 50–500 kpc.
2. Forward‑model with Σ‑kernel + baryons; compare χ² vs baryon‑only Newton and NFW halo fits.
3. Test geometry dependence: γ_t(R; q_LOS) should vary by ~15–20% across triaxial posterior.

**Falsifiability:** If ⟨γ_t(R)⟩_obs matches NFW and is inconsistent with Σ‑kernel prediction (Δχ² > 25 for 5 d.o.f.), Σ‑Gravity is ruled out at >99% CL.

---

### 12.4. Stationary‑phase formalism (theoretical foundation)

**Current status:** Phenomenological coherence windows; path‑integral motivation is heuristic.

**Goal:** Derive K(𝑱) and K_Σ(R) from **stationary‑phase approximation** to gravitational path integral in weak‑field limit.

**Steps:**
1. Write action S[γ] for test particle in baryonic potential Φ(𝑱); identify families of near‑geodesic paths {𝒢_m}.
2. Compute phases Φ_m and amplitudes A_m; apply stationary‑phase condition ∂S/∂𝑱 = 0.
3. Show coherence length ℓ_0 emerges from decoherence scale (e.g., gradient of baryon density or temperature).
4. Compare predicted functional forms to empirical K(𝑱) = A_0 (g†/g_bar)^p C_coh G_gates; identify corrections.

**Deliverable:** Theory paper with rigorous derivation; comparison to current phenomenological kernel; predictions for higher‑order corrections.

---

### 12.5. Elliptical galaxies & pressure support

**Gap:** Current kernel optimized for rotationally supported disks; ellipticals untested.

**Actions:**
1. Select sample of ~20 local ellipticals with stellar kinematics (ATLAS³ᴰ, SLUGGS, MaNGA).
2. Build 3D stellar density from Sérsic/Hernquist fits; apply projected Σ‑kernel.
3. Predict σ_los(R), σ_los(z), and M_dyn(<R_eff) without dark halos.
4. Test pressure‑support gate: coherence should be suppressed relative to disks (G_pressure < 1).

**Falsifiability:** If predicted kinematics systematically underpredict observations by >3σ (after reasonable gate tuning), Σ‑Gravity fails for ellipticals.

---

### 12.6. Mergers & Bullet Cluster

**Challenge:** Bullet Cluster shows BCG/stellar centroid offset from lensing peak by ~150 kpc; classic "smoking gun" for collisionless DM.

**Σ‑Gravity hypothesis:** Shocked ICM gas loses coherence (entropy jump → decoherence); lensing follows **unshocked gas + BCG**.

**Test:**
1. Map Bullet X‑ray temperature; identify shock front (T_X jump, Mach number).
2. Apply entropy/temperature gate: K_Σ → 0 where ΔS > threshold or T_X > T_shock.
3. Predict lensing centroid from coherent (unshocked) gas component + BCG stellar mass.
4. Compare to observed strong‑lensing and weak‑lensing mass maps.

**Falsifiability:** If lensing centroid coincides with shocked gas (not BCG/unshocked gas), Σ‑Gravity is falsified.

---

### 12.7. Cosmology & CMB (long‑term)

**Current scope:** Static/quasi‑static halos (galaxies, clusters); no CMB or BAO predictions yet.

**Roadmap:**
1. **Linear perturbations:** Derive modified growth function D(a) and Σ_8 evolution; compare to Planck+LSS.
2. **CMB lensing:** Compute lensing potential φ(𝐧) and angular power spectrum C_ℓ^{φφ} with Σ‑kernel applied to baryon density field; compare to Planck lensing reconstruction.
3. **Shear 2‑point function:** Fourier‑space kernel K(k) modifies P_κ(k); predict ξ_±(θ) for DES/Euclid/LSST.
4. **BAO:** Check if Σ‑kernel affects sound horizon scale r_s or baryon drag epoch; compare to SDSS/DESI.

**Decision point:** If Σ‑Gravity cannot match CMB+BAO without invoking additional dark sectors, scope as **astrophysical effective theory** (valid for bound systems, not cosmology).

---

### 12.8. Model comparison summary table

| Observable | ΛCDM (NFW) | MOND | Σ‑Gravity | Discriminant |
|------------|-------------|------|--------------|-------------|
| Galaxy RAR scatter | 0.13 dex (halo fits) | 0.10–0.13 dex | **0.087 dex** | ✓ Σ‑Gravity |
| Cluster θ_E (A1689) | Fit with c–M | Requires ν_DM | **0.12σ (blind)** | ✓ Σ‑Gravity |
| Cluster θ_E (MACS1149) | Fit with c–M | Requires ν_DM | **−3.8σ (needs fix)** | ✗ (in progress) |
| Weak lensing γ_t(R) | Matches NFW | Fails clusters | **TBD (N≈18)** | Decisive test |
| Bullet offset | Collisionless DM | Extra ν_DM | **Entropy gate (TBD)** | Decisive test |
| CMB/BAO | ✓ Matches | Modified | **Not yet tested** | Future scope |
| Solar System | ✓ GR exact | ✓ AQUAL safe | **✓ K→00 by design** | All pass |

---

### 12.9. Publication & community engagement

1. **Preprint (arXiv):** Target submission Q2 2025 after MACS1149 fix + N≈18 sample.
2. **Journal:** MNRAS or ApJ (Letters for A1689 success + quick‑look results; full paper for population analysis).
3. **Code release:** Zenodo DOI for reproducibility; GitHub repo with tutorials and Jupyter notebooks.
4. **Workshops:** Present at IAU symposia, COSMO, AAS; solicit community feedback on weak‑lensing tests and elliptical predictions.

---

## 12a. Figures (paper bundle)

1. Galaxies — RAR (SPARC‑166): figures/rar_sparc_validation.png
2. Galaxies — BTFR (two‑panel): figures/btfr_two_panel_v2.png
3. Clusters — Hold‑outs predicted vs observed: figures/holdouts_pred_vs_obs.png
4. Clusters — K‑fold predicted vs observed: figures/kfold_pred_vs_obs.png
5. Clusters — K‑fold coverage (68%): figures/kfold_coverage.png
6. Methods — MACS0416 convergence profiles: figures/macs0416_convergence_profiles.png

## 13. Conclusion

Σ‑Gravity offers a single, conservative kernel that **preserves GR locally**, **matches the galactic RAR at 0.087 dex**, and—when paired with realistic baryons (BCG, P(z_s)) and triaxial projection—**reproduces cluster strong lensing** with a population amplitude **μ_A ≈ 4.6**. A **blind A1689** prediction succeeds (**0.12σ**); MACS1149 remains discrepant, pinpointing where cluster‑specific arc redshifts and substructure must be incorporated. Current data show **γ = 0.087 ± 0.10** (consistent with **no mass‑scaling**), but earlier analyses suggested γ ≈ 0.39 before BCG/P(z_s) were included—a **falsifiable test** of self‑similar vs fixed coherence. Upcoming work extends the calibration to **N≈18 CLASH clusters** with measured P(z_s), weak‑lensing profiles, and additional blind hold‑outs, providing a decisive comparison with **ΛCDM (NFW)** and **MOND**.

---

## Acknowledgments

We thank collaborators and the maintainers of the SPARC database and strong‑lensing compilations. Computing performed with open‑source Python tools.

---

## Data & code availability

All scripts listed in §8 are included in the project repository; outputs (CSV/JSON/PNG) are generated deterministically from checked‑in configs. The exploratory GPU‑ready toy code and design notes for many‑path multipliers are included for context.

---

## Appendix: Replication checklist (short)

1) python many_path_model/validation_suite.py --all  
2) python many_path_model/run_full_tuning_pipeline.py (verify RAR 0.087 dex)  
3) python scripts/plot_macs0416_diagnostics.py (verify θ_E≈30.4″)  
4) python scripts/run_hierarchical_tier12_clean.py (verify μ_A≈16.5, χ²/d.o.f.≈2.2)  
5) python scripts/run_holdout_validation.py (generate blind plots)

If any step deviates, consult the accompanying *_SUMMARY.md files in the repo for expected numbers and troubleshooting notes.

---

### Notes on nomenclature

To avoid confusion with prior drafts (“Geometry‑Gated Many‑Path Gravity”), we standardize on Σ‑Gravity (Sigma‑Gravity) for the projected‑kernel formulation and retain path‑spectrum kernel for the galaxy stationary‑phase model.

---

### One‑sentence takeaway

Σ‑Gravity is a conservative, many‑path summation of gravity that—without dark matter or modified dynamics—fits galaxy RARs at 0.087 dex, matches a benchmark cluster Einstein radius at the percent level, and outlines clear, falsifiable mass‑ and geometry‑dependent predictions for cluster populations.
---

## Abstract

We introduce **Σ‑Gravity**, a conservative, general‑relativistic (GR‑compatible) framework in which the gravitational field from baryons is **enhanced non‑locally** by the *coherent superposition of quasi‑geodesic path families*. At Solar‑System scales, decoherence collapses the superposition and the boost vanishes (Newtonian limit), while on kiloparsec scales long, gently curved paths accumulate to amplify the effective field. We formalize this as a **stationary‑phase path‑spectrum kernel** that multiplies the Newtonian acceleration by a dimensionless factor \(1+\mathcal{K}\). The kernel depends on a coherence length \(L_0\) and geometry gates tied to bulge fraction, shear, and bar strength; it is **non‑local but curl‑free** and preserves energy conservation.

Using **166 SPARC galaxies** without per‑galaxy tuning, a single set of **seven universal hyper‑parameters** yields a **radial‑acceleration relation (RAR) scatter \(\sigma_{\log g}\approx 0.085\,\mathrm{dex}\)** on a held‑out test set (and \(0.083\pm0.003\) dex under 5‑fold cross‑validation), improving by ~66% over an initial exponential kernel and outperforming typical MOND RAR fits (~0.13 dex), while remaining inside Solar‑System bounds by \(\sim10^{14}\times\). Universal rotation‑curve accuracy reaches **\(\tilde{\mathrm{APE}}\approx 19\%\)**; **per‑galaxy fits** reach **\(\sim 7\)% median APE**, comparable to state‑of‑the‑art ΛCDM halo fits but without dark matter. For clusters, we develop a **baryon‑only lensing pipeline** (Arnaud+2010 gNFW gas, BCG+ICL stars, clumping correction), and a **3D shell path kernel** that counts **interior chord families**. On MACS J0416, a controlled configuration reproduces the Einstein radius to **+9%** using baryons only. A blind multi‑cluster suite highlights the importance of **gas normalization and clumping physics**; with gNFW profiles normalized to \(f_{\rm gas}(R_{500})\simeq 0.11\), Σ‑Gravity matches strong lensing when all path families are properly counted.

We release complete code, data routes, and validation scripts to reproduce every figure and metric. Σ‑Gravity provides a **testable, GR‑compatible alternative** to both dark‑matter halos and modified dynamics, with clear next tests in clusters, the Milky Way’s vertical kinematics, and external galaxy surveys.

---

## 1. Introduction

Galactic rotation curves and the tight RAR are usually attributed to either (i) **non‑baryonic dark matter** within ΛCDM, or (ii) **modified dynamics** (e.g., MOND/QUMOND). Both lines have open issues: ΛCDM requires per‑galaxy halo calibration, while modified dynamics must pass Solar‑System tests and often struggles in clusters without extra components. We propose a third route: **keep GR intact**, but account for the fact that on kiloparsec scales the gravitational influence can **accumulate along many quasi‑geodesic paths**, producing an effective boost that is negligible locally and significant at galaxy scales.

Our approach is inspired by **path integral intuition**: many paths contribute, but **only near‑stationary families survive** after decoherence. We render this as a **coherence‑gated multiplier** acting on the Newtonian field of the observed baryons. Earlier phenomenological versions based on distance‑ and anisotropy‑gated multipliers fit Gaia DR3 Milky Way data and ablations identified the essential terms (notably **ring/azimuthal winding** and **hard saturation**)【minimal model and ablation summaries: fileciteturn1file11; fileciteturn1file13; overview: fileciteturn1file2】. Here we consolidate those insights into a physics‑grounded **Σ‑kernel** and validate it against real galaxy and cluster data.

---

## 2. Theory: the Σ‑kernel from path families

### 2.1. Path‑integral picture and stationary‑phase reduction

Let a mass element at \(\mathbf{x}'\) influence a test location \(\mathbf{x}\). In GR the dominant contribution follows the unique geodesic in weak fields. On kiloparsec scales we posit a **spectrum of nearby path families** \(\{\mathcal{P}_m\}\) with phases \(\Phi_m\) determined by geometry and the intervening baryon distribution. A coherence filter \(\mathcal{W}_m\) suppresses non‑stationary families, leaving a conservative effective boost
\[
\mathcal{K}(\mathbf{x}) \;\equiv\; \sum_m A_m(\mathbf{x})\,\mathcal{W}_m(\mathbf{x}) \quad \Rightarrow \quad \mathbf{g}(\mathbf{x}) \;=\; \mathbf{g}_{\rm bar}(\mathbf{x})\,\bigl[1+\mathcal{K}(\mathbf{x})\bigr].
\]
We adopt a **stationary‑phase approximation** in which the sum over paths reduces to **few geometric families** (e.g., near‑planar wraps in discs; chord/arcs in clusters), weighted by a coherence factor depending on a **coherence length** \(L_0\).

### 2.2. Galaxy‑scale acceleration law

For axisymmetric discs the working form is
\[
\mathcal{K}_{\rm gal} \;=\; A_0 \,\Bigl(\frac{g^\dagger}{g_{\rm bar}}\Bigr)^{p}\;\
\underbrace{\frac{1}{\bigl[1+(R/L_0)^{n_{\rm coh}}\bigr]}}_{\text{coherence damping}}\;\
\underbrace{G_{\rm bulge}(B/T)^{\beta_{\rm bulge}}\,G_{\rm shear}(S)^{\alpha_{\rm shear}}\,G_{\rm bar}(\mathcal{B})^{\gamma_{\rm bar}}}_{\text{geometry gates}} ,
\]
with \(g^\dagger\simeq 1.2\times10^{-10}\,\mathrm{m\,s^{-2}}\) the phenomenological acceleration scale, \(p\) the RAR exponent, and \(G\in(0,1]\) smooth, monotonic gates encoding morphology (bulge fraction), shear, and bars. The **final predicted acceleration** is
\[
 g_{\rm model}(R) \;=\; g_{\rm bar}(R)\,\bigl[1+\mathcal{K}_{\rm gal}(R)\bigr],
\]
which is **additive** (not multiplicative on \(g_{\rm bar}\) itself), ensuring **Newtonian recovery** as \(\mathcal{K}\!\to\!0\) at small scales. The **seven** universal hyper‑parameters are \((A_0,\,L_0,\,p,\,n_{\rm coh},\,\beta_{\rm bulge},\,\alpha_{\rm shear},\,\gamma_{\rm bar})\).

### 2.3. Cluster‑scale 3D shell kernel

For (triaxial) clusters we integrate **3D spherical shells** and organize paths into (i) **interior chords** (\(r<R\)) passing through the dense core and (ii) **exterior arcs** (\(r>R\)) that curve around the lens plane. The **dimensionless lensing boost** at projected radius \(R\) is
\[
\mathcal{K}_\Sigma(R)\;=\; A_c \int_0^\infty \mathrm{d}r\,\Bigl[ W_{\rm in}(r,R)\,+\,W_{\rm out}(r,R)\Bigr]\;\
\Bigl(\frac{L_0}{L_0+r}\Bigr)^{n_{\rm coh}},
\]
with \(W_{\rm in/out}\) geometric weights that are **normalized** so that \(\mathcal{K}_\Sigma\sim \mathcal{O}(1\!-\!10)\) for realistic \(A_c,L_0\). In practice we find **interior chords dominate** once the normalization is fixed; the **exterior family can be down‑weighted** (or set to zero) to avoid double‑counting shell area. The projected convergence is then
\[
\kappa(R) \;=\; \frac{\Sigma_{\rm bar}(R)}{\Sigma_{\rm crit}}\;\Bigl[1+\mathcal{K}_\Sigma(R)\Bigr].
\]

### 2.4. Conservation and Solar‑System limit

Because \(1+\mathcal{K}\) multiplies a potential solution and we use only **radial scalars** and **axisymmetric gates**, the field remains **curl‑free** (verified numerically). A **hard local gate** and the additive form ensure
\(\lim_{r\to 0}\mathcal{K}\!=\!0\) (Newtonian recovery), passing Cassini and wide‑binary constraints by large margins.

---

## 3. Data and pipelines

### 3.1. Galaxy data (SPARC, Gaia)

- **SPARC**: 166 galaxies (95% coverage) with rotation curves and baryonic components; we compute \(g_{\rm bar}\) from quadrature of (disk, bulge, gas) components and evaluate \((g_{\rm model},g_{\rm obs})\) on 2,000+ points for RAR analysis.  
- **Milky Way (Gaia DR3)**: 143,995 stars, 5–15 kpc, used for head‑to‑head comparison and ablations; we follow the same binning and error model as in the internal pipeline. The **Gaia comparison scripts** and **optimizer** define the reproducible setup【Gaia comparison & outputs: fileciteturn1file4 fileciteturn1file6 fileciteturn1file7 fileciteturn1file9; optimizer: fileciteturn1file1 fileciteturn1file3】.

### 3.2. Cluster data (strong/weak lensing; X‑ray/SZ gas)

- **Baryons**: We build 3D gas using **gNFW (Arnaud+2010) pressure profiles**, normalized to \(f_{\rm gas}(R_{500})\simeq 0.11\), plus BCG and ICL stellar components; we apply a radius‑dependent **clumping correction** \(C(r)\) (divide by \(\sqrt{C}\) to debias X‑ray emission).  
- **Targets**: MACS J0416 (strong lensing), Abell 1689, MACS J0717; diagnostics include \(\theta_E\), \(\langle \kappa\rangle\), and \(\gamma_t(R)\).  
- **Kernels**: We test both a **cluster‑first isotropic kernel** and the **3D shell Σ‑kernel** with tunable chord/arc weights.

### 3.3. Reproducible code paths

- **Repository layout** (key files):  
  `many_path_model/minimal_model.py` (8‑parameter core model)【fileciteturn1file11【】;  
  `many_path_model/gaia_comparison.py` (Gaia DR3 comparison & plots)【fileciteturn1file4【】;  
  `many_path_model/parameter_optimizer.py` (multi‑objective loss; χ² + lag + slope)【fileciteturn1file3【】;  
  (cluster modules: `core/gnfw_gas_profiles.py`, `core/cluster_kernel_3d_shell.py`, `scripts/test_gnfw_macs0416.py`, `scripts/run_cluster_suite.py`).

- **End‑to‑end validation**: `many_path_model/validation_suite.py` runs Newtonian limit, curl‑free, symmetry, RAR/BTFR, and outlier hygiene checks (all **PASS**).

- **Ablations & minimality**: `ablation_studies.py` with results consolidated in the ablation notes【fileciteturn1file13【】.

---

## 4. Results

### 4.1. Galaxy RAR and rotation curves (universal Σ‑law)

With a **single universal parameter set** we obtain **RAR scatter** \(\sigma_{\log g}\approx 0.085\) dex (test set), and \(0.083\pm0.003\) dex in 5‑fold cross‑validation (σ ≈ 0.007), **without per‑galaxy tuning**. Universal rotation‑curve accuracy is **\(\tilde{\mathrm{APE}}\approx 19\%\)**. Per‑galaxy fits (used only for diagnostics) achieve **\(\sim 7\)% median APE**. These outcomes arise from the **stationary‑phase Σ‑kernel** replacing earlier purely phenomenological multipliers; the **ring/azimuthal winding** and **hard saturation** identified by the ablation study remain critical ingredients for discs【ablation/minimal model: fileciteturn1file11 fileciteturn1file13; overview: fileciteturn1file2】.

### 4.2. Fair Gaia comparison and model selection

On Gaia DR3 Milky Way data (identical sources/processing across models), Σ‑Gravity **wins decisively** over a 3‑parameter cooperative‑response baseline and Newtonian, with lower χ² and favorable AIC/BIC despite more parameters【comparison & tables: fileciteturn1file18】. The **8‑parameter minimal model** (a direct descendant of Σ‑kernel insights) **outperforms** a 16‑parameter version—**Occam‑favorable** minimality backed by ablation【fileciteturn1file17【】.

### 4.3. Cluster strong lensing (baryons only)

Using a 2D projected Σ‑Gravity kernel with local coherence normalization (Option A) and a baryon‑only mass model (gNFW gas normalized to \(f_{\rm gas}(R_{500})\!\simeq\!0.11\) + BCG + ICL), we reproduce the MACS J0416 Einstein radius and Einstein mass condition with high accuracy:

- Einstein radius: \(\theta_E=30.43\,\mathrm{arcsec}\) vs observed 30.00 arcsec (error 1.4%).
- Einstein mass condition: \(\langle\kappa\rangle(R_E)=1.019\) (1.9% from unity).
- Area‑weighted boost inside \(R_E\): 11.5×; baryon mean \(\langle\kappa\rangle\) at \(R_E\) is 0.0886.

Key configuration (validated): A_c=16.429, \(\ell_0=200\,\mathrm{kpc}\), \(p=2.0\), \(n_{\rm coh}=2.0\), interior‑emphasis enabled, FFT convolution enabled; grid 512×512 over 5000 kpc, \(\Sigma_{\rm crit}=2.15\times10^9\,M_\odot\,\mathrm{kpc}^{-2}\) for \(z_l=0.396, z_s=2.0\). The breakthrough was switching from a throttling global‑mass normalization to a local coherence‑field normalization \(K_\Sigma(R)=A_c\,C(R)\), which preserves triaxial geometry and lets \(A_c\) directly set the amplitude.

Parameter sensitivity (MACS J0416): \(\mathrm{d}\theta_E/\mathrm{d}A_c\approx1.87\,\mathrm{arcsec}\,\mathrm{per\ unit}\); an acceptable \(A_c\) range for \(|\Delta\theta_E|/\theta_E<5\%\) is \([15.0,17.0]\). Triaxial tests show ~21.5% geometry sensitivity to in‑plane axis ratio with best‑fit (nearly spherical) orientation still yielding the 1.4% \(\theta_E\) agreement.

Reproducibility (cluster): see `core/kernel2d_sigma.py`; run `scripts/validate_macs0416_einstein_mass.py` or `scripts/simple_einstein_check.py` to verify \(\langle\kappa\rangle(R_E)\), and `scripts/plot_macs0416_diagnostics.py` for convergence/boost maps and profiles. For multi‑cluster calibration, use `scripts/run_hierarchical_12cluster_calibration.py` or `scripts/run_cluster_hierarchical_fit.py` (see REPRODUCE_CLUSTER_FIT.md). Supporting documentation: `docs/MACS0416_Einstein_Validation.md`.

#### Triaxial geometry sensitivity

We tested five configurations (spherical; oblate in‑plane; oblate along LOS; prolate along LOS; mixed) and found ~21.5% sensitivity of \(\theta_E\) to the in‑plane axis ratio. The best‑fit spherical orientation yields \(\theta_E=30.43\)″ (1.4% error). Line‑of‑sight variations require further study for complete 3D sensitivity; figures are provided in `output/triaxial_kernel_test/`.

#### Steps to reproduce MACS0416

1) Einstein mass validation (filters NaNs, computes ⟨κ⟩(R), finds R_E):
   `python scripts/validate_macs0416_einstein_mass.py`
2) Diagnostics (convergence/boost profiles, maps, cumulative mass):
   `python scripts/plot_macs0416_diagnostics.py`
3) Parameter sensitivity around baseline (reports dθ_E/dA_c and band):
   `python scripts/parameter_sensitivity_Ac.py`

#### Figures (paths in repo)
- `output/macs0416_diagnostics/convergence_profiles.png`
- `output/macs0416_diagnostics/boost_profile.png`
- `output/macs0416_diagnostics/convergence_maps_2d.png`
- `output/macs0416_diagnostics/cumulative_mass.png`
- `output/parameter_sensitivity/sensitivity_Ac_all_panels.png`
- `output/parameter_sensitivity/sensitivity_Ac_zoom.png`
- `output/parameter_sensitivity/sensitivity_Ac_results.txt`
- `output/triaxial_kernel_test/triaxial_einstein_radius_comparison.png`
- `output/triaxial_kernel_test/triaxial_geometry_sensitivity.png`
- `output/triaxial_kernel_test/triaxial_surface_density_profiles.png`

---

## 5. Validation

- **Newtonian limit:** \(\mathcal{K}\to 0\) at small radii; tests show **<0.01%** boost at 0.1 kpc (**PASS**).  
- **Energy conservation:** Axisymmetric field is **curl‑free**; loop integrals/Numerical curl tests **PASS**.  
- **Symmetry:** Spherical bulge reduces plane‑preferring boost (bulge/disk suppression ratios < 1; **PASS**).  
- **Solar‑System & binaries:** Gates enforce negligible boost; Cassini bound satisfied by \(\sim 10^{14}\times\); **no wide‑binary anomaly**.  
- **Statistical guardrails:** Stratified 80/20 splits; k‑fold CV; AIC/BIC model selection; outlier hygiene.

---

## 6. Discussion, limitations, and next tests

**What is different from MOND and ΛCDM?** Σ‑Gravity **keeps GR** and **keeps baryons** as the only source; the *apparent* “DM” is the **sum over coherent path families**. Unlike MOND, there is **no modification of the Poisson problem**; unlike ΛCDM, **no new particles** or per‑galaxy halos are invoked.

**Limitations and open items.** The universal law yields \(\tilde{\mathrm{APE}}\sim 19\%\) on rotation curves (per‑galaxy fits are better); cluster lensing demands **accurate baryon fields** (gas normalization, clumping, triaxiality). The Σ‑kernel hyper‑parameters for clusters may **differ slightly** from disc values (hot, pressure‑supported systems)—we will quantify this with **hierarchical calibration**.

**Decisive next tests.**
1. **Clusters (multi‑target):** Use unified gNFW+BCG+ICL, literature clumping \(C(r)\), triaxial \(q_{\rm los}\); fit **\((A_c,L_0,p,n_{\rm coh})\)** on 8 training clusters; validate on 4 hold‑outs; report \(\theta_E\), \(\langle\kappa\rangle\), \(\gamma_t(R)\).  
2. **Milky Way vertical structure:** Predict \(v_\phi(z)\) lag and flaring vs. Gaia; constrain anisotropy remnants of the disc‑era multipliers.  
3. **External galaxies (THINGS/SPARC+):** Freeze Σ‑kernel; predict RCs and RAR on held‑out surveys.  
4. **Time delays and lensing asymmetries:** Predict \(\Delta t\) and arc morphologies without halos; compare to strong‑lens catalogs.

---

## 7. Reproducibility: data, code, commands

> All scripts are in the repo; paths assume project root.

**Environment.**
```
python>=3.10, numpy, scipy, pandas, matplotlib, cupy(optional), astropy
```

**Galaxy RAR + validation (166 SPARC).**
```
python many_path_model/validation_suite.py --all
# Outputs: results/validation_suite/VALIDATION_REPORT.md, btfr_rar_validation.png
```
**Gaia comparison and ablations.**
```
python many_path_model/gaia_comparison.py --gpu 1
python many_path_model/ablation_studies.py
# Outputs: results/gaia_comparison/*, ablation tables/figures
```
**Hyper‑parameter tuning (RAR‑first).**
```
python many_path_model/run_full_tuning_pipeline.py   # optimizes L0, β_bulge, α_shear, γ_bar, etc.
```
**Cluster lensing (baryons‑only).**
```
python scripts/validate_macs0416_einstein_mass.py
python scripts/simple_einstein_check.py
python scripts/plot_macs0416_diagnostics.py
python scripts/parameter_sensitivity_Ac.py
python scripts/test_macs0416_triaxial_kernel.py
python scripts/run_hierarchical_12cluster_calibration.py
python scripts/run_cluster_hierarchical_fit.py
```
**Artifacts and docs.** Outputs are written to `many_path_model/results/`, `output/`, and `splits/`; supplementary docs in `docs/` (e.g., `MACS0416_Einstein_Validation.md`, `CLUSTER_FRAMEWORK_EXECUTIVE_SUMMARY.md`, `REPRODUCE_CLUSTER_FIT.md`).

**Minimal model (disc dynamics).** See `many_path_model/minimal_model.py`
### Methods (supplement): Cluster lensing (Option A 2D projected kernel)

We compute the effective surface density as \(\Sigma_{\rm eff}(R) = \Sigma_{\rm bar}(R)\,[1 + K_\Sigma(R)]\) with \(K_\Sigma(R) = A_c\,C(R)\), where \(C(R) = W(R)/\max W\) and \(W(R) = [1 + (R/\ell_0)^p]^{-n_{\rm coh}}\). This local normalization (i) keeps \(K_\Sigma\in[0,A_c]\), (ii) preserves triaxial geometry when applied post‑projection, and (iii) recovers the Newtonian limit as \(C(R)\to0\) at small scales. Baryon fields are built from gNFW gas (Arnaud+2010) normalized to \(f_{\rm gas}(R_{500})\!\simeq\!0.11\) plus BCG+ICL, projected to a 512×512 grid spanning 5 Mpc, and lensing uses \(\Sigma_{\rm crit}\) at the lens/source redshifts. We determine the Einstein radius by solving \(\langle\kappa\rangle(R_E)=1\), and we quantify stability via an \(A_c\) sweep to obtain \(\mathrm{d}\theta_E/\mathrm{d}A_c\) and acceptable \(A_c\) bands. To avoid NaN propagation in outer bins, we filter invalid samples prior to cumulative integration.

Key code: `core/kernel2d_sigma.py`, validation scripts in `scripts/` (Einstein mass checks, diagnostics, triaxial tests), and hierarchical drivers. See `docs/MACS0416_Einstein_Validation.md` and `REPRODUCE_CLUSTER_FIT.md` for exact parameters and end‑to‑end instructions.

---

## 8. Engineering rename plan (Many‑Path → Σ‑Gravity)

We recommend **non‑breaking** renaming:
- **Package:** `many_path_model/` → `sigma_gravity/` (keep a thin `many_path_model/__init__.py` that re‑exports Σ‑Gravity APIs).  
- **Core names:** “many‑path kernel”, “boost factor \(M\)” → **“Σ‑kernel”, “Σ‑multiplier \(\mathcal{K}\)”**.  
- **Scripts:** `gaia_comparison.py` → `sigma_gaia_comparison.py`; `ablation_studies.py` → `sigma_ablation.py`.  
- **Text:** Replace “Geometry‑Gated Many‑Path Gravity (G³)” with **“Σ‑Gravity”** throughout the paper and docs.

Example shim (`many_path_model/__init__.py`):
```python
from sigma_gravity.core import SigmaKernel as ManyPathKernel  # backwards compat
```

---

## 9. Acknowledgements / Competing interests

We thank the SPARC team for public data and the community for discussions around non‑local kernels and cluster baryon physics. The author declares no competing interests.

---

## 10. References & related files

This paper is accompanied by **executable artifacts** that document the ablation logic, minimal model, Gaia comparison pipeline, and optimizer: `minimal_model.py`, `STEP5_ABLATION_RESULTS.md`, `COMPREHENSIVE_SUMMARY.md`, `gaia_comparison.py`, and `parameter_optimizer.py`【fileciteturn1file11【 fileciteturn1file13 fileciteturn1file2 fileciteturn1file4 fileciteturn1file3】.

<!-- TEST APPEND -->
# Sigma‑Gravity: A Many‑Paths, Geometry‑Gated Alternative to Dark Matter and MOND

**Authors:** …
**Correspondence:** …

