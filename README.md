# A gated logarithmic tail for galactic rotation curves (LogTail)

## Abstract

Observed galaxy rotation curves (RCs) remain approximately flat out to tens of kiloparsecs, deviating from the $v^2 \!\propto\! r^{-1}$ falloff expected from baryons under Newtonian gravity. The standard remedy posits massive dark‑matter halos; modified‑gravity approaches (e.g., MOND‑class relations tied to a universal acceleration scale $a_0$) instead alter the force law at low accelerations. Here we propose and test **LogTail**, a phenomenological model in which the gravitational potential acquires an **additive logarithmic component** that turns on smoothly beyond a galaxy‑scale radius. With a single set of global parameters applied to all galaxies in our sample, LogTail achieves rotation‑curve accuracy competitive with MOND on outer points (median closeness \approx 90%), while remaining Solar‑System–safe by construction via an inner gate. We summarize the hypothesis, provide the working formula, and report first results relative to GR, MOND, and other phenomenological baselines.&#x20;

---

## 1. Background and problem statement

In baryons‑only Newtonian gravity, a circular speed profile declines as $v^2(r)\!\sim\! GM(<r)/r$. Instead, **most disks show approximately flat outer RCs**, implying either additional unseen mass (dark halos) or a breakdown of the Newtonian/GR description on galactic scales.

Two broad strategies have dominated:

* **ΛCDM halos (e.g., NFW):** Add a quasi‑spherical collisionless component that dominates at large radii, reproducing flat curves and lensing signals. Accurate fits typically use **per‑galaxy** halo parameters (mass $M_{200}$, concentration $c$), which complicates parameter‑economy comparisons to global modified‑gravity ansätze.

* **MOND‑class dynamics:** Replace the Newtonian relation by $a\,\mu(a/a_0)=a_N$ with a universal $a_0$, naturally yielding flat outer speeds and the baryonic Tully–Fisher relation (BTFR), $v_\infty^4 \propto M_b a_0$. Predictivity is strong in disks but consistency at cluster and cosmological scales requires additional structure (e.g., relativistic completions).

Despite these successes, it is scientifically valuable to map the **space between** “extra mass” and “RAR‑based dynamics”—seeking minimal, inner‑safe phenomenology that captures the **observed outer flattening** with **global (population‑level) parameters**.

---

## 2. The LogTail hypothesis

We posit that the **effective gravitational potential** felt by stars and gas includes, beyond a few kiloparsecs, a **logarithmic tail** whose contribution grows smoothly with radius and then saturates, **without** altering inner‑galaxy or Solar‑System dynamics.

### 2.1. Working equation

Let $v_{\rm bar}(r)$ be the circular speed from baryons under Newtonian gravity. We model the total circular speed as

$$
\boxed{\,v^2(r)=v_{\rm bar}^2(r)\;+\;v_0^2\,\frac{r}{r+r_c}\,S(r;r_0,\Delta)\,}
$$

where

$$
S(r;r_0,\Delta)=\tfrac{1}{2}\Big[1+\tanh\!\Big(\frac{r-r_0}{\Delta}\Big)\Big]
$$

is a smooth gate. Parameters:

* $v_0$ sets the **asymptotic amplitude** (flat‑tail plateau),
* $r_c$ controls the **turn‑on scale** of the logarithmic term,
* $r_0$ and $\Delta$ gate the effect to protect small radii.

Equivalently, the added potential is $\Phi_{\rm add}(r)\!\propto\!\ln(1+r/r_c)\,S(r)$; in spherical symmetry this mimics an **isothermal‑like** contribution (effective $\rho\!\propto\! r^{-2}$) once the gate is on. By default we keep lensing GR‑like (unit slip, $\Sigma\!=\!1$); if the tail is interpreted as a true potential term, lensing from the tail should also be included (future work).

### 2.2. What LogTail *is not*

* It is **not MOND**: there is no universal acceleration scale $a_0$ and no RAR built in.
* It is **not** a per‑galaxy dark halo fit: we use **one global parameter set** for the full sample, trading per‑galaxy flexibility for population‑level parsimony.

### 2.3. Testable implications

* **Outer shape:** tends to $v(r)\!\to\!\sqrt{v_{\rm bar}^2+v_0^2}$ (flat) for $r\!\gg\! r_c$ and inside the gate.
* **Inner safety:** for $r\!\ll\! r_0-$a few $\Delta$, $S\!\approx\!0$ and $v\!\approx\!v_{\rm bar}$.
* **Population scaling:** LogTail does **not** enforce the BTFR; if observations demand it, $v_0$ may need a weak mass‑coupling $v_0(M_b)$ (to be tested).

---

## 3. Methods (summary)

We evaluate LogTail against a compiled rotation‑curve table with **$N\!=\!1167$** outer points (outermost fraction per galaxy), using the project’s **closeness** metric (higher is better). We perform a **global grid search** over $(v_0,r_c,r_0,\Delta)$ and compare to baselines: GR (baryons only), MOND (single $a_0$), and several GR‑adjacent phenomenologies. For context, an NFW reference is also fit on a **subset** with per‑galaxy parameters (not a like‑for‑like comparison).&#x20;

---

## 4. Results to date

### 4.1. Rotation‑curve accuracy (outer)

* **Latest refined run (global parameters):**
  **LogTail** median closeness **90.01%**, with best‑fit $\{v_0=140\ \mathrm{km\,s^{-1}},\ r_c=15\ \mathrm{kpc},\ r_0=3\ \mathrm{kpc},\ \Delta=4\ \mathrm{kpc}\}$.
  **MuPhi** median **86.51%** ($\{\epsilon=2.0,\ v_c=140\ \mathrm{km\,s^{-1}},\ p=3\}$).&#x20;

* **Coarser earlier run (same dataset, $N\!=\!1167$):**
  **LogTail** mean/median **85.25%/89.47%** (best $\{v_0\approx120,\ r_c\approx10,\ r_0=3,\ \Delta=3\}$).
  **MOND** mean/median **87.11%/89.81%**.
  **GR** median **63.86%**; **Shell** median **79.19%**.
  **NFW** (subset, per‑galaxy) median **97.69%** (not apples‑to‑apples).&#x20;

**Takeaway:** With **one global parameter set**, LogTail is **RC‑competitive with MOND** on outer points in this sample.

### 4.2. Qualitative diagnostics (brief)

* **Outer shape:** LogTail produces truly **flat** outer curves once the gate is on, unlike pure force multipliers that merely lessen the decline.
* **Inner behavior:** The gate keeps inner regions GR‑like by design.
* **Lensing:** If interpreted as a physical potential, LogTail’s tail implies an SIS‑like $\Delta\Sigma(R)\!\propto\!1/R$ shape inside a truncation scale; in the present RC‑only tests we left lensing unmodified (to be addressed next).

---

## 5. Limitations and planned tests

* **BTFR & RAR:** Because LogTail does not encode a universal $a_0$, the BTFR and RAR are **not automatic** and must be **tested explicitly** (add baryonic masses $M_b$ per galaxy; fit slope and intrinsic scatter; apply out‑of‑sample validation).
* **Lensing and clusters:** A credible replacement for DM/MOND must match **galaxy–galaxy lensing** and (eventually) **cluster** probes. We will compute $\Delta\Sigma(R)$ from the LogTail potential and confront public stacks; for MuPhi, we will specify a slip $\Sigma(R)$ or acknowledge a baryons‑only lensing shortfall.
* **Cosmological embedding:** The logarithmic tail requires a **large‑scale regulator** (e.g., truncation or environment‑dependent saturation) and a relativistic completion if interpreted as fundamental.

---

## 6. Summary

LogTail—an **inner‑gated, logarithmic potential tail**—achieves **$\sim$90%** median RC closeness with a **single global parameter set**, **comparable to MOND** on the same outer‑point sample. It differs conceptually from both MOND (no universal $a_0$, additive tail rather than an $a/a_0$ law) and from halo fits (no per‑galaxy mass/concentration). The immediate priorities are to (i) verify **BTFR/RAR** performance with galaxy baryonic masses, (ii) compute and compare **lensing** predictions, and (iii) test **out‑of‑sample generalization** across heterogeneous RC surveys under a fair parameter budget.

---

## 7. Latest findings (2025-09-19)

This section summarizes the current state across rotation curves, RAR, lensing toy checks, BTFR status, and new CMB “clean-slate” envelope constraints. It links to artifacts and embeds plots generated by the repository scripts.

### 7.1 Galaxy-scale performance (RCs, RAR, lensing toy)

- Rotation curves (outer points):
  - LogTail global params deliver ~90% median pointwise closeness on outer RCs with {v0≈140 km/s, rc≈15 kpc, r0≈3 kpc, Δ≈4 kpc}.
  - MuPhi trails LogTail with ~86% median.
  - Plots:
    - Overall accuracy comparison (higher is better):
      
      ![RC accuracy comparison](out/analysis/type_breakdown/accuracy_comparison.png)
      
    - MOND vs LogTail vs MuPhi median comparison:
      
      ![Median comparison](out/analysis/type_breakdown/compare_medians_mond_logtail_muphi.png)
      
    - Outer G-ratio overlay (diagnostic):
      
      ![G-ratio overlay](out/analysis/type_breakdown/g_ratio_overlay.png)

- RAR (Radial Acceleration Relation):
  - Observed curved-RAR orthogonal scatter ≈ 0.172 dex (baseline).
  - LogTail model curve (tied to baryons) scatter ≈ 0.069 dex — tighter than observed, as expected for a deterministic mapping from g_bar.
  - Artifacts: out/analysis/type_breakdown/rar_obs_curved_stats.json, out/analysis/type_breakdown/rar_logtail_curved_stats.json.

- Lensing toy (shape-only sanity):
  - LogTail’s additive tail implies ΔΣ(R) ∝ 1/R at 50–100 kpc (SIS-like slope ≈ −1) when the gate is on; a simple toy comparison matches the expected slope.
  - Artifact: out/analysis/type_breakdown/lensing_logtail_comparison.json.

### 7.2 BTFR status (work in progress)

- The SPARC MRT parser patch shipped successfully; the dataframe has expected columns.
- A quick BTFR diagnostic run produced empty observed tables (n=0) and NaN correlations, indicating a join/attachment issue downstream of the parser:
  - out/analysis/type_breakdown/btfr_observed.csv — headers only, no rows.
  - out/analysis/type_breakdown/btfr_qc.txt — NaN Pearson (log v_flat_obs, log M_b).
  - out/analysis/type_breakdown/*_fit.json — "n": 0 entries.
- Planned fix: derive M_bary directly from L[3.6] and MHI with Υ⋆≈0.5 and helium factor 1.33×MHI during join, so BTFR can proceed even if external mass joins are missing; then re-run observed-vs-model BTFR and report slope/scatter with QC.

### 7.3 CMB “clean-slate” envelope constraints (Planck 2018)

We bound additive late-time imprints on Planck plik_lite bandpowers without assuming ΛCDM. Templates: lensing-like peak smoothing (lens), low-ℓ gated kernel (gk), and high-ℓ void-envelope (vea).

- TT-only envelopes (binned TT, covariance fallback to diagonal if full SVD fails):
  - Lens (peak-smoothing): σ_A ≈ 0.0029 ⇒ |A| ≲ 0.0057 (95% CL)
  - GK (low-ℓ reweight): σ_A ≈ 0.019 ⇒ |A| ≲ 0.037 (95% CL)
  - VEA (high-ℓ amp): σ_A ≈ 0.00156 ⇒ |A| ≲ 0.00305 (95% CL)
  - Plots (TT envelopes):
    
    ![Lens envelope](out/cmb_envelopes/cmb_envelope_lens.png)
    
    ![VEA envelope](out/cmb_envelopes/cmb_envelope_vea.png)
    
    ![GK envelope](out/cmb_envelopes/cmb_envelope_gk.png)

- Lensing reconstruction amplitude (φφ):
  - Using Planck lensing clik_lensing (CMB-marginalized) vectors and a dependency-light reader with proper L^4/(2π) normalization, we find
    
    α_φ ≈ 1.007 ± 0.027 (1σ)
    
  - Artifact: out/cmb_envelopes/cmb_lensing_amp.json (includes normalization metadata).

- Interpretation:
  - Late-time gravitational processing in a static/unexpanded picture can only “sculpt” peaks at sub-percent levels in the acoustic/damping ranges; the peaks must largely be primordial (or from very high redshift).
  - Our galaxy-scale LogTail/MuPhi gates live at kpc–tens of kpc and, if not promoted to Mpc scales, are naturally safe under these envelopes. Any void/web-scale extension must respect the sub-percent TT bounds and φφ amplitude.

- Next steps:
  - TTTEEE joint envelopes via optional clik loader; fit band-limited A(ℓ) across TT, TE, EE; expect tighter constraints.
  - Map A to physical knobs (e.g., extra lensing amplitude, void-envelope strength) in any proposed cosmology-level completion (μ(k,z), Σ(k,z)).

### 7.4 Reproduce the CMB envelope and φφ results

- TT envelopes (TT plik_lite “_external”):

```bash path=null start=null
python -u rigor/scripts/cmb_static_expts.py \
  --mode lens \
  --plik_lite_dir data/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik/clik/lkl_0/_external \
  --out_dir out/cmb_envelopes
```

- Piecewise bands (default bands: 2-50,50-250,250-800,800-1500,1500-2500):

```bash path=null start=null
python -u rigor/scripts/cmb_static_expts.py \
  --piecewise --mode lens \
  --plik_lite_dir data/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik/clik/lkl_0/_external \
  --out_dir out/cmb_envelopes
```

- Lensing reconstruction amplitude (normalized):

```bash path=null start=null
python -u rigor/scripts/cmb_static_expts.py \
  --phiamp \
  --lensing_dir data/baseline/plc_3.0/lensing/smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8_CMBmarged.clik_lensing/clik_lensing \
  --out_dir out/cmb_envelopes
```

---

### Data and reproducibility

All numbers above come from the project’s exported summaries of the latest runs: the refined LogTail/MuPhi fit (`summary_logtail_muphi.json`) and the extended comparison (`accuracy_comparison_extended.json`). Paths and full parameter listings are preserved in those files.

---

#### (Optional next section you can add immediately)

* **Methods in detail:** dataset composition, exact definition of the “closeness” metric, grid ranges, convergence criteria.
* **Predictions beyond RCs:** closed‑form $\Delta\Sigma(R)$ for the tail; outer‑slope histograms; hierarchical variants where $v_0$ is tied weakly to $M_b$.
