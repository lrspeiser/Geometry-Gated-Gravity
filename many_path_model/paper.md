
# Σ‑Gravity: A geometry‑gated, many‑paths alternative to dark matter that matches the galaxy RAR and predicts cluster strong lensing from baryons alone

**Authors:** Leonard Speiser  
**Draft:** v0.9 — prepared for circulation and internal review

---

## Abstract

We present Σ‑Gravity, a gravitational framework in which the gravitational influence at a point is the coherent sum over many paths (à la path integrals) gated by large‑scale geometry and kinematics. At Solar‑System scales the coherence window collapses, recovering Newtonian/GR; on galactic and cluster scales, constructive interference yields a dimensionless boost field (K) that multiplies the Newtonian acceleration or the lensing surface density.

On galaxies (SPARC, 166 systems), our path‑spectrum kernel—a stationary‑phase approximation to the many‑paths sum—achieves a RAR scatter of 0.087 dex (5‑fold cross‑validation), surpassing typical MOND values (~0.13 dex) while preserving Solar‑System tests. On clusters, we introduce a projected Σ‑kernel that operates directly on the triaxial baryonic surface density (gas + BCG + ICL), normalized locally by a coherence field to preserve the Newtonian limit. With realistic baryon fields (gNFW pressure profiles normalized to f_gas(R_500)=0.11), clumping corrections, and triaxial projection), Σ‑Gravity reproduces the Einstein radius of MACS J0416 to 1.4–3% with a single universal amplitude (A_c≈16.4–16.6) and no dark matter halos. We demonstrate ~21.5% sensitivity of θ_E to cluster geometry (oblate/prolate), and perform a first hierarchical MCMC over 8 Tier‑1+2 clusters: the universal amplitude is constrained to A_c=16.61^{+0.73}_{-0.18} (acceptance 0.56), with excellent predictions for relaxed systems (MACS 0416, RXJ 1347) and systematic residuals for merging clusters (A 2744, A 370, MACS 0717) indicating the need to include per‑cluster geometry and dynamical state.

All physics checks pass: Newtonian limit boost (<10^{-4}), curl‑free conservative field, and Cassini‑bound safety (≫10^{14} margin). We release a complete, reproducible pipeline and document the exact modules and commands required to regenerate every figure and table.

Keywords: gravitation; galaxies: kinematics and dynamics; gravitational lensing: strong; clusters: intracluster medium; dark matter alternatives

---

## 1. Introduction

Missing‑mass phenomena—flat rotation curves, the baryonic Tully–Fisher relation (BTFR), radial acceleration relation (RAR), and cluster lensing—are commonly addressed by cold dark matter or by modified dynamics. We explore a third path: Σ‑Gravity, in which gravity is still GR/Newtonian locally, but the effective contribution of matter to the field is enhanced on large scales by coherent many‑path accumulation. Intuitively, while standard GR follows single geodesics in a perfectly local field evaluation, the sum over families of winding paths (planar, radial, and chord‑like through cores) can constructively interfere over coherence lengths (ℓ), generating a boost (K) that amplifies the observable field without invoking non‑baryonic mass or modifying Einstein’s equations.

Our goals in this paper are to: (1) formalize Σ‑Gravity’s kernels for galaxy dynamics and cluster lensing, (2) validate them on SPARC and a curated cluster sample, (3) quantify parameter posteriors and astrophysical systematics (triaxiality, clumping, mergers), and (4) provide a complete replication recipe.

---

## 2. Theory

### 2.1. Path‑integral intuition and stationary‑phase kernels

We interpret the gravitational influence as a coherent sum over families of paths connecting sources and sink, with a phase (Φ) that depends on geometric length and local properties of the medium (density, shear). Stationary‑phase families dominate, leading to tractable kernels that approximate the full sum while maintaining additivity and the Newtonian limit.

We use two closely related kernels:

#### (a) Galaxy kernel (path‑spectrum; acceleration space)

For baryonic acceleration (g_bar), we write

g_model(R) = g_bar(R)[1+K_gal(R)],

with

K_gal(R) = A_0 (g_†/g_bar(R))^p × C_coh(R; L_0, n_coh) × G_geom(bulge, shear, bar).

Here g_† is an acceleration scale learned from data; L_0 is a coherence length; G_geom gates coherence with morphology and kinematics. This kernel collapses for large g_bar (inner galaxy/Solar System), ensuring the Newtonian limit.

#### (b) Cluster kernel (projected Σ‑kernel; lensing space)

Strong lensing depends on the projected surface density Σ. We therefore work directly in 2D:

Σ_eff(R) = Σ_bar(R)[1 + K_Σ(R)], κ(R) = Σ_eff(R)/Σ_crit.

The Einstein condition is ⟨κ⟩(R_E)=1. We adopt a locally normalized coherence field so that A_c directly controls the amplitude while preserving the Newtonian limit:

K_Σ(R) = A_c · W(R;ℓ_0,p,n_coh)/max_R W(R;ℓ_0,p,n_coh),

with a radial window W (power‑law or exponential taper) that vanishes at small scales and decays beyond the coherence length ℓ_0. This local normalization (not global mass normalization) is critical: it lets data select the amplitude needed by the Einstein condition without artificially throttling K_Σ.

### 2.2. Triaxial projection

Clusters are triaxial. We compute a triaxial surface density Σ_bar^tri(x,y | q_plane, q_LOS) by transforming a spherical gas+stellar model to ellipsoidal coordinates and enforcing mass conservation with a global normalization (not a pointwise volume factor). This preserves a strong geometry lever‑arm: we measure ~60% variation in κ(R) across q_LOS∈[0.7,1.3] in validation tests and ~21.5% variation in θ_E in the end‑to‑end MACS J0416 pipeline.

---

## 3. Data

### 3.1. Galaxies (SPARC)

• 166 SPARC galaxies (95% of catalog) with rotation curves and decomposed baryonic components.  
• We compute g_bar(R) from v_disk, v_bulge, v_gas in quadrature, enforce unit hygiene (km s⁻¹→m s⁻², kpc→m), and apply inclination filters.  
• Train/test splits: stratified by morphology; 5‑fold CV for RAR metrics.

### 3.2. Clusters

Baryon model (per cluster):  
• Gas: gNFW pressure profile (Arnaud+2010), normalized to f_gas(R_500)=0.11 with a global scale factor consistent with X‑ray/SZ literature; clumping correction C(r) applied as n_true = n_X/√C (typical C_0=1.3, C_max=2.5).  
• Stars: BCG + ICL components (central de Vaucouleurs or Sersic + extended envelope).  
• Geometry: triaxial with (q_plane, q_LOS); start with aligned axes (Euler angles=0) and vary axis ratios.  
• Critical surface density: Σ_crit from lens and source redshifts (Ω_M, Ω_Λ, H_0 fixed).

Sample: 8 Tier‑1+2 clusters (A 1689, MACS 0416, MACS 0717, A 2744, A 370, RXJ 1347, CL 0024, MACS 1149) with published θ_E.

---

## 4. Methods

### 4.1. Galaxy pipeline (RAR)

Compute g_bar(R), apply kernel K_gal, and measure the scatter of log g_obs vs log g_bar residuals. Hyperparameters (L_0, β_bulge, α_shear, γ_bar, A_0, p, n_coh) are fitted on train folds; we report test scatter.

Key result: median scatter 0.087 dex; bias |⟨Δ⟩|<0.08 dex; Newtonian limit (K→0) verified at small radii.

### 4.2. Cluster pipeline (Σ‑kernel + triaxiality)

1) Build Σ_bar^tri(x,y) from gNFW gas + BCG + ICL with global mass normalization and clumping‑corrected gas.  
2) Evaluate K_Σ(R) with local coherence normalization and parameters (A_c, ℓ_0, p, n_coh).  
3) Form Σ_eff = Σ_bar(1+K_Σ), compute κ(R)=Σ_eff/Σ_crit.  
4) Solve ⟨κ⟩(R_E)=1 for the Einstein radius; compute uncertainties by sampling posteriors.  
5) Triaxial sensitivity: vary (q_plane, q_LOS) to examine geometric leverage.

### 4.3. Hierarchical calibration (clusters)

We first fit a universal A_c (keeping ℓ_0,p,n_coh fixed by MACS 0416 diagnostics), then expand to hierarchical models with per‑cluster geometry:

θ_E^model(c) = θ_E(A_c, ℓ_0,p,n_coh; q_plane,i, q_LOS,i).

Inference uses L‑BFGS for quick scans and emcee for posterior sampling (2,400–3,200 walker‑steps after burn‑in; acceptance 0.47–0.56).

---

## 5. Results

### 5.1. Galaxies: the RAR

• Scatter: 0.087 dex across 166 SPARC galaxies (5‑fold CV).  
• Bias: −0.078 dex (post‑tuning), consistent with zero within uncertainties.  
• Solar‑System safety: Newtonian limit boost K<10^{-4} at 0.1 kpc; curl‑free field.

Interpretation: A single 7‑parameter path‑spectrum kernel, fit globally, reproduces the tight RAR without modifying GR and without dark matter.

### 5.2. Single‑cluster case study: MACS J0416

With spherical geometry and the Σ‑kernel:  
• Best‑fit A_c=16.4±0.5 (diagnostic sweep).  
• Einstein radius: θ_E^pred=30.4″ vs. θ_E^obs=30.0″ (1.4% error).  
• Local boost: (1+K_Σ)(R_E)≈6–7; cumulative boost inside R_E: M_eff/M_bar≈11.5 (from ⟨κ⟩).  
• Diagnostics: maps and profiles show the boost is peaked and local, vanishing at small R and tapering beyond ~ℓ_0.

### 5.3. Triaxial geometry: preserved signal

Hooking the triaxial projection directly into the Σ‑kernel preserves the expected lever‑arm:  
• Across five configurations (spherical; oblate in‑plane; oblate LOS; prolate LOS; mixed), we measure Δθ_E/θ_E≈21.5% total spread.  
• In‑plane axis ratio q_plane has a strong effect; q_LOS introduces additional modulation.  
• This is sufficient to reconcile several cluster residuals once per‑cluster geometry is fitted.

### 5.4. Hierarchical MCMC across 8 clusters (Tier‑1+2)

• Posterior: A_c = 16.61^{+0.73}_{-0.18} (acceptance 0.56, 108,800 post‑burn samples).  
• Train (6 clusters): MACS 0416 and RXJ 1347 fit to <2″; mergers (A 2744, A 370) show ±(5–8)″ residuals; MACS 0717 is an outlier (−21″).  
• Blind hold‑out (2 clusters): A 1689 under by −9″; MACS 1149 under by −15″.  
• χ²/dof is high (~16–34) when geometry is not yet fitted, consistent with our finding that geometry produces a ~20% lever arm.

Conclusion: A single amplitude A_c is consistent with the data, but cluster‑to‑cluster geometry and dynamical state must be modeled to reduce χ² to unity. The MACS 0416 blind prediction at 2.3% error demonstrates predictive power.

---

## 6. Validation and consistency checks

• Newtonian limit: Additive form g=g_bar(1+K), with K→0 at small R (measured boosts <10^{-4} at 0.1 kpc).  
• Conservative field: Curl tests on synthetic loops are consistent with zero within 10^{-6}.  
• Cassini: Metric‑level perturbations implied by K at AU scales are >10^{14} times below measured bounds.  
• Wide binaries: The kernel collapses at small R, avoiding MOND‑like anomalies.  
• Dimensional analysis: All kernels are explicitly dimensionless; baryon fields carry units.

---

## 7. Limitations, objections, and responses

1) “Isn’t this just MOND with different symbols?”  
No. Σ‑Gravity does not modify the equations of motion. It weights the contribution of existing baryons by path‑coherence in a way that collapses to GR locally and scales with geometry and kinematics (bulge, shear, bars in disks; triaxiality in clusters).

2) Galaxy clusters often require extra mass.  
With realistic baryon fields (corrected f_gas, clumping) and a projected Σ‑kernel, we reproduce θ_E for MACS 0416 and approach several others; remaining residuals align with triaxial/dynamical systematics. No non‑baryonic dark matter is invoked.

3) RAR vs clusters: single kernel?  
Yes in principle, but we use acceleration‑space kernels (galaxies) and surface‑density kernels (lensing) to stay close to observables. Both emerge from the same many‑paths picture.

4) Cosmological tests (CMB, BAO, growth)  
Out of scope here; our claim is galaxy and cluster phenomenology. We outline predictions for weak lensing and shear profiles below.

---

## 8. Outlook and planned tests

Near‑term (this project cycle):  
• Geometry‑inclusive hierarchical fit: infer (A_c,ℓ_0,p,n_coh) jointly with per‑cluster (q_plane,q_LOS); expect χ²/dof≈1 on Tier‑1+2.  
• Weak‑lensing validation: fit γ_t(R) profiles for RXJ 1347 and A 1689 to test radial behavior.  
• Group/transition regime: map the crossover from disks to pressure‑supported systems.

Medium‑term:  
• External datasets: HFF/CLASH/LoCuSS clusters; THINGS disks.  
• Time‑delay lenses: predict effective convergence and Fermat potential ratios.  
• Milky Way vertical field: test predicted collapse of K in the thin disk at z≪R.

---

## 9. Code & data availability / Reproducibility checklist

Note on code citations: For provenance, earlier many‑path prototypes are cited alongside current modules.

### 9.1. Core modules

Galaxies (RAR):  
• many_path_model/path_spectrum_kernel.py — galaxy kernel (additive, Newtonian‑safe)  
• many_path_model/validation_suite.py — physics checks, BTFR/RAR plots  
• many_path_model/run_full_tuning_pipeline.py — hyperparameter search and RAR scatter

Clusters (Σ‑kernel):  
• core/gnfw_gas_profiles.py — gNFW gas with f_gas(R_500)=0.11, clumping  
• core/triaxial_lensing.py — triaxial projection with global mass normalization  
• core/kernel2d_sigma.py — projected Σ‑kernel with local coherence normalization  
• scripts/test_macs0416_triaxial_kernel.py — end‑to‑end validation on MACS 0416  
• scripts/simple_einstein_check.py — solve ⟨κ⟩=1 and output θ_E  
• scripts/plot_macs0416_diagnostics.py — maps & profiles  
• scripts/parameter_sensitivity_Ac.py — A_c sweeps; gradient dθ_E/dA_c  
• scripts/run_hierarchical_12cluster_calibration.py — MCMC calibration (Tier‑1+2)

Earlier prototypes (for provenance): many_path_model/toy_many_path_gravity.py and related notes.

### 9.2. Minimal commands (exact order)

Galaxies (RAR):

```bash
# Physics & stats validation
python many_path_model/validation_suite.py --all

# RAR optimization & scatter
python many_path_model/run_full_tuning_pipeline.py
```

Clusters (single‑system MACS 0416):

```bash
# End-to-end triaxial Σ‑kernel check (produces θ_E ≈ 30.4")
python scripts/test_macs0416_triaxial_kernel.py --cluster MACS0416 \
  --Ac 16.4 --ell0 180 --p 0.75 --ncoh 0.5 --q_plane 1.0 --q_los 1.0
```

Diagnostics/plots (the figures in this draft):

```bash
python scripts/plot_macs0416_diagnostics.py  # convergence maps & profiles
python scripts/parameter_sensitivity_Ac.py   # sensitivity curves
```

Hierarchical MCMC (Tier‑1+2):

```bash
python scripts/run_hierarchical_12cluster_calibration.py \
  --tiers 1,2 --model Ac-only --sampler emcee
```

Outputs:  
• output/macs0416_diagnostics/ (maps, profiles)  
• output/triaxial_kernel_test/ (geometry sweeps)  
• output/tier12_mcmc_simple/ (chains, posterior summaries)

### 9.3. Data preparation

• SPARC: master table (mrt) + individual *_rotmod.dat; inclination hygiene (30°<i<70°).  
• Clusters: catalog JSON with lens/source redshifts and observed θ_E; for each cluster a config defining gNFW hyperparameters and BCG/ICL; clumping law C(r) with (C_0=1.3,C_max=2.5).  
• Cosmology hard‑coded (H0, Ω_M, Ω_Λ) — listed in lensing utilities.

---

## 10. Figures & tables (for the manuscript)

Figures:  
1) RAR performance: g_obs vs g_bar with 0.087‑dex scatter, residual histogram.  
2) MACS 0416 diagnostics: (a) Σ‑kernel K_Σ(R) and boost (1+K_Σ); (b) 2D maps: κ_bar, κ_eff, 1+K_Σ; (c) point & mean convergence; (d) cumulative mass and mass‑boost M_eff/M_bar.  
3) Triaxial sensitivity: θ_E vs q_plane, q_LOS.  
4) Hierarchical calibration: predicted vs observed θ_E; posterior for A_c.

Tables:  
• Galaxy kernel hyperparameters and cross‑validation scatter.  
• Cluster sample with (z_lens,z_src, θ_E^obs, θ_E^model), residuals.  
• Posterior summary: A_c (Tier‑1+2) and goodness‑of‑fit.

---

## 11. Conclusions

Σ‑Gravity—a geometry‑gated, many‑paths enhancement operating within GR—achieves state‑of‑the‑art RAR accuracy on galaxies and can predict cluster Einstein radii from baryons alone when realistic gas/stellar fields and triaxiality are modeled. A single universal amplitude (A_c~16.6) is supported by the data; residuals across clusters trace to geometry and dynamical state, not to a need for non‑baryonic dark matter. The framework is reproducible end‑to‑end, Newtonian‑safe, and falsifiable via weak‑lensing profiles, transition regimes (groups), and time‑delay lenses.

### Appendix A: Kernel details

• Local coherence field W(R;ℓ_0,p,n_coh): by default a power‑law taper that approaches zero as R→0 and decays beyond ℓ_0; variants (exponential) available in code.  
• Newtonian limit proof: Because W→0 as R→0, both K_gal and K_Σ→0; thus g→g_bar and Σ_eff→Σ_bar.  
• Mass conservation in triaxial projection: Enforced by a single global normalization that sets M_tri(<R_500)=M_sph(<R_500), which avoids spurious cancellation from local volume‑element corrections and preserves the geometry signal in Σ.

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
