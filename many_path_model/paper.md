# Geometry‑Gated Many‑Path Gravity for Disk Galaxies

**Abstract —**
We propose and test a non‑local, geometry‑gated "many‑path" gravity kernel that multiplies the Newtonian acceleration by an **anisotropic, distance‑dependent factor** designed to capture the cumulative contribution of long, gently curved gravitational paths that exist on galactic—but not solar‑system—scales. Calibrated to the Milky Way with Gaia DR3 kinematics using a multi‑objective loss (rotation curve χ², vertical lag, and outer‑slope flatness), the model reduces χ² versus Newtonian gravity by more than an order of magnitude while remaining Solar‑System safe. Applied to 175 SPARC galaxies, **per‑galaxy optimization** attains a **median absolute percentage error (APE) of 7.6%** and a mean of 12.5%. To test universality, we learn population laws that map global morphology and kinematics to kernel parameters. A **continuous B/T law** achieves ≈25% median APE; a **multi‑predictor V2** (bulge fraction B/T, compactness Σ₀, shear S) improves to ≈25–32% median/mean; and **V2.1**, which adds *shear‑dependent winding scale* and a *radial ring envelope*, further reduces median APE to **22.9%** with 71% of galaxies lying within ±20% of their per‑galaxy optima. We provide ablations, outlier analyses, and predictions (e.g., how shear de‑phases azimuthal paths, shortening the effective winding length in declining‑curve galaxies). We also benchmark against a density‑dependent "cooperative response" model to discourage cherry‑picking and to clarify phenomenological differences.   

---

## 1. Motivation and hypothesis

Galactic rotation curves, vertical structure, and lensing point to either dark matter or a modification of the effective gravitational law on kiloparsec scales. Inspired by path‑sum reasoning, we hypothesize that **gravity's effective response can depend on the *available families of paths*** set by global geometry. In stellar disks, long near‑planar and azimuthally wound paths are plentiful; in bulge‑dominated regions they are disrupted; and in the Solar System, path families collapse onto the unique short geodesic, leaving Newtonian gravity intact. Our phenomenological goal is *not* to quantize gravity, but to encode this "path availability" through a bounded multiplier **M** that is negligible locally and saturates on large scales.

---

## 2. Kernel and parameters (phenomenology)

We model the total acceleration as
[
\mathbf{a}*{\rm tot}(\mathbf{x}) ;=; \mathbf{a}*{\rm N}(\mathbf{x}),[1 + M(d,\mathrm{geometry})],
]
where (d) is source–target separation and "geometry" captures planar preference, azimuthal winding, and large‑scale saturation. The code implements:

* **Local gate** (g_{\rm loc}) with scale (R_{\rm gate}) and sharpness (p_{\rm gate}) to ensure (M!\to 0) for Solar‑System scales;
* **Distance growth & saturation** with onset (R_0) and saturation (R_1) and exponents (p,q);
* **Planar anisotropy** (scale (Z_0), exponent (k_{\rm an})), optionally **radially modulated** to increase off‑plane suppression around the solar circle and relax it at large (R);
* **Azimuthal winding ("ring")** controlled by amplitude `ring_amp` and winding length `λ_ring`, later **concentrated** by a radial Gaussian envelope peaking near (\sim(1.6\text{–}2.5)R_d) depending on B/T (V2.1);
* **Hard cap** `M_max` to retain bound orbits.

These ingredients and parameter names are defined in the public code (`toy_many_path_gravity.py` and project README).  

**Solar‑System safety** (the gate) and the **bounded multiplier** are integral parts of the implementation.

---

## 3. Milky Way calibration with Gaia

We compare Newtonian and many‑path predictions to **real** Gaia DR3 Milky Way kinematics using a reproducible script that (i) ingests the Gaia sample, (ii) constructs the observed rotation curve in radial bins, and (iii) evaluates model curves and χ².

To avoid "over‑fitting the plane", we use a **multi‑objective loss** that combines (1) rotation‑curve χ², (2) a vertical‑lag target of ~15 ± 5 km s⁻¹ at (z=0.5) kpc (ensuring a not‑too‑thin disk), and (3) an outer‑slope penalty favouring flat curves beyond ≳12 kpc. This loss is implemented in the optimizer and used throughout the MW tuning.

The tuned "Family‑A" parameter set (with **radially modulated anisotropy**) achieves a rotation‑curve χ² well below a Newtonian disk+bulge baseline and yields a realistic vertical lag (~11 km s⁻¹ pre‑tweak, then pulled toward 15 km s⁻¹ by loss weighting). Representative values and their roles are recorded in the working parameter files.

**Outcome.** Against Gaia DR3 medians, the many‑path curve markedly outperforms Newtonian dynamics while preserving Solar‑System constraints—a result visible in the comparison routine and plots produced by the Gaia script.

---

## 4. SPARC: per‑galaxy optimization (ground truth for universality)

We then treat each of the 175 SPARC galaxies independently and seek the **best per‑galaxy parameters** using a hierarchical random‑narrow search/CMA‑ES driver (same multi‑objective structure as in the MW). The run converges for all systems and yields:

* **Median APE** (=) **7.55%**, **mean** (=) **12.45%**, with min 0.93% and max 108%.
* Clear morphological trends: late‑type disks prefer larger ring amplitudes and longer λ; bulge‑dominated spirals prefer weaker, shorter coherence.

(Per‑galaxy optimization details and loss structure follow the MW optimizer's design.)

**Interpretation.** These per‑galaxy fits serve as **ground truth** for what the kernel can achieve *if* tuned galaxy‑by‑galaxy; the universality task is then to *predict* near‑optimal parameters from **global descriptors**.

---

## 5. From classes to continuous laws

### 5.1 B/T‑only (V1)

We first learn **continuous B/T laws**, mapping bulge fraction to four kernel parameters ({\eta,;\mathrm{ring_amp},;M_{\max},;\lambda_{\rm ring}}). Functional forms and a small set of hyper‑parameters are fit from the per‑galaxy optima. This B/T law achieves **median APE ≈ 25%** and **mean ≈ 32%** over the full SPARC sample, with best performance for early‑type systems and weaker performance for Sc–Sd disks (consistent with broader spiral diversity). *(Implementation and evaluation tooling are part of your project; we're summarizing the outcomes here.)*

### 5.2 Multi‑predictor (V2)

To address systematic late‑type failures and the diversity of inner slopes, we introduce **compactness** (Σ₀) and **shear** (S) gates alongside B/T. V2 reduces overshoot in some dwarfs (low Σ₀ → amplitude suppression) and declining‑curve spirals (high S → coherence suppression), bringing the global **median APE** to the **mid‑20%** range.

### 5.3 Targeted fixes (V2.1)

Diagnostics showed V2 sometimes chose **too‑long λ**, producing broad "red overshoot" in intermediate spirals. V2.1 adds:
(i) **λ(B/T,S)** where shear *shortens* the effective winding scale;
(ii) a **radial ring envelope** so the azimuthal boost peaks near the observed arm region ((\sim!1.6!-!2.5,R_d)) rather than across the entire outer disk.
These surgical changes lower **median APE to 22.9%** and raise "within ±20% of per‑galaxy best" to **71%**.

**Takeaway.** The kernel's *physics knobs*—bulge gating, coherence, shear de‑phasing, and radial arm concentration—each fix a specific failure mode without adding gratuitous freedom.

---

## 6. Ablations and outliers

Ablations (turning off one mechanism at a time) show:

* Removing **radial modulation** of anisotropy worsens the MW vertical‑lag vs outer‑slope trade‑off;
* Looser distance saturation (smaller (q) or larger (R_1)) risks outer overshoot;
* Removing the **ring winding** undermines late‑type fits.

These effects are encoded in the objective and explored by the optimizer used for both the Milky Way and SPARC studies.

**Outliers** (APE ≳ 50%) cluster into: (i) strongly barred/interacting systems; (ii) warped or thick disks; and (iii) high‑shear declining curves with very short arm coherence. V2.1 specifically targets group (iii). Groups (i–ii) motivate adding **bar strength** and **vertical thickness** proxies in a V3 law.

---

## 7. Comparison with a density‑dependent baseline

To guard against confirmation bias, we also implement a **cooperative response** baseline in which the **effective (G)** rises with local density,
[
G_{\rm eff}(\rho) ;=; G;\Bigl[1+\alpha,(\rho/\rho_{\odot})^{\beta},\tanh(\rho/\rho_{\rm th})\Bigr].
]
The companion script computes densities (SPH‑like kernel) at target points and evaluates the rotation curve for the same Milky Way benchmark. This provides an apples‑to‑apples comparator and emphasizes the qualitative difference between **geometry‑gated** and **density‑gated** phenomenology summarized in your internal comparison note. 

---

## 8. Predictions and falsifiable tests

The kernel makes concrete, testable predictions:

1. **Shear–λ coupling.** High‑shear disks should prefer shorter effective winding scales and show weaker outer boosts (V2.1's λ(B/T,S)).
2. **Morphology continuity.** Parameter trends are **monotonic in B/T**, not class‑discrete; galaxies near the Sbc/Sc boundary should interpolate smoothly.
3. **Radial ring concentration.** The azimuthal boost should peak near (R_{\rm ring}!\sim!(1.6!-!2.5)R_d) and not extend broadly over 6–15 kpc in Scd galaxies.
4. **Vertical lag.** At (z!=!0.5) kpc, predicted lags should cluster in the **10–20 km s⁻¹** range when the MW‑calibrated loss is applied to similar disks—an intentional prior encoded by the optimizer.

Refutations include (a) no correlation between shear and λ preference; (b) disks with strong arms but requiring vanishing ring amplitude; or (c) Solar‑System anomalies contradicting the gate (the code enforces (R_{\rm gate}!\ll!{\rm kpc}) to avoid this).

---

## 9. Limitations

Our kernel is **phenomenological** and not derived from a fundamental action. The "path" language is an *interpretive aid* for the geometry dependence. The model currently handles axisymmetric disks plus smooth bulges; strong bars, warps, or ongoing interactions are imperfectly captured—consistent with the observed outlier set. We therefore treat V2.1 as a **working empirical law** and a staging ground for a more principled **stationary‑phase/path‑spectrum** kernel in follow‑up work.

---

## 10. Methods (summary)

**Data & observables.**

* Milky Way: Gaia DR3 stars filtered by (|z|<0.5) kpc; observed rotation curve constructed in 0.5 kpc bins by medians and SEM. The pipeline and table creation are implemented in `gaia_comparison.py`.
* SPARC: HI+Hα rotation curves, stellar/gas mass models, disk scale lengths (R_d), and simple bulge fractions B/T.

**Model evaluation.**

* Disk and bulge source particles are sampled from exponential and Hernquist profiles and composed into a force calculation that is **Newtonian × (1+M)**, where (M) is the bounded geometry kernel described above and implemented in `toy_many_path_gravity.py`. 
* The optimizer evaluates the **multi‑objective loss** (rotation χ² vs. binned Gaia medians; vertical lag target; outer‑slope penalty) exactly as defined in `parameter_optimizer.py`.
* Tuned parameter snapshots (including radially modulated anisotropy) are archived for reproducibility.

**Baselines.** The cooperative response comparator computes an SPH density field and substitutes (G!\to!G_{\rm eff}(\rho)) in the same force loop to produce curves on the identical grid.

**Robustness & ablations.** We report ablations by toggling individual terms (e.g., removing ring winding, relaxing saturation, eliminating radial modulation), then re‑minimizing the same multi‑objective loss to quantify deltas. The ablation hooks reuse the optimizer components referenced above.

---

## 11. Results (condensed)

* **Milky Way (Gaia DR3):** Many‑path outperforms Newtonian in a direct χ² comparison, while keeping the vertical lag within the prior target and enforcing flat outer slopes; scripts and outputs are reproducible.
* **SPARC per‑galaxy best:** 7.6% median, 12.5% mean APE across all 175 galaxies.
* **Universality tests:**

  * B/T‑only law: ≈25% median APE (strongest in early‑types).
  * Multi‑predictor V2 (B/T + Σ₀ + S): ≈25–32% median/mean; clearer handling of LSB dwarfs and high‑shear declines.
  * **V2.1 (λ(B/T,S) + radial ring envelope):** **22.9% median**; **71%** within ±20% of per‑galaxy best; intermediate spirals notably improved.
* **Outliers:** Bars/warps and extreme shear remain as the principal failure modes; these suggest adding bar and thickness predictors (V3).

---

## 12. Discussion

**What is unique?** A **single, bounded, geometry‑gated multiplier** that (i) is **Solar‑System safe by construction**, (ii) couples naturally to galaxy morphology and kinematics, and (iii) recovers MW and SPARC phenomenology with **few universal gates** rather than per‑galaxy halos. The kernel's *controls* (bulge gating, shear de‑phasing, radial ring concentration) map cleanly to visually interpretable structures—spiral arms, bulges, thickness and warps—making the model unusually **diagnosable**.

**What is weak?** The absence of an explicit Lagrangian or linear‑response derivation means the kernel is **phenomenological**. Our "path" language is conceptual, and it remains to derive the same forms from a stationary‑phase approximation applied to a non‑local gravitational functional.

---

## 13. Outlook and concrete next steps

1. **V3 law**: add *bar strength* and a *thickness* proxy; we expect both to down‑weight azimuthal coherence and further improve Sc–Sbc fits.
2. **Stationary‑phase kernel**: re‑cast the multiplier as a sum over path families with phase dispersion; the V2.1 gates become priors on the path spectrum.
3. **Joint benchmark**: rerun MW and SPARC with identical data cuts for the many‑path kernel and the cooperative response model; report Bayes factors. 
4. **Predictive tests**: publish shear–λ and ring‑radius predictions for a held‑out subset of SPARC and for nearby high‑quality IFU disks.

---

## Data, code, and reproducibility

* **Milky Way pipeline:** `gaia_comparison.py` (loads DR3, constructs medians/SEM, computes model curves, and prints χ² tables).
* **Kernel implementation:** `toy_many_path_gravity.py` (force loop, multiplier, gating, anisotropy, ring term, hard cap) with a concise README of parameters. 
* **Optimizer and multi‑objective loss:** `parameter_optimizer.py`, used for the MW and adapted to SPARC.
* **Comparator model:** `cooperative_gaia_comparison.py` for density‑dependent (G_{\rm eff}) curves and diagnostics.
* **Parameter snapshots:** tuned MW settings (including radially modulated anisotropy) in `tweaked_params.txt`.

---

### Author contributions (draft)

Conceptualization and methodology: L.R.S.
Software, data curation, formal analysis: L.R.S.
Validation, investigation, visualization: L.R.S.
Writing—original draft and review/editing: L.R.S.

### Acknowledgements

We thank the SPARC team and the Gaia mission for making their data products available.

---

## Extended Methods (selected equations)

**Loss function.**
[
\mathcal{L} = \chi^2_{\rm rot} ;+; w_{\rm lag},\sum_i \Bigl[\tfrac{\Delta v_{\phi}(R_i,z{=}0) - \Delta v_{\phi}(R_i,z{=}0.5,{\rm kpc}) - 15}{5}\Bigr]^2 ;+; w_{\rm slope},!!\sum_{R>12{\rm kpc}}!!\Bigl(\tfrac{dv_c}{dR},/,2\Bigr)^2,
]
with (w_{\rm lag}\approx0.8) and (w_{\rm slope}\approx2). Implemented and used throughout.

**Kernel sketch (code truths).**

* Local gate: small‑(d) suppression controlled by (R_{\rm gate}, p_{\rm gate});
* Growth/saturation: onset (R_0), exponent (p); saturation (R_1), exponent (q);
* Anisotropy: (Z_0, k_{\rm an}) (with optional radial modulation (Z_{\rm eff}(R)));
* Ring winding: amplitude `ring_amp`, length `λ_ring`;
* Hard cap: `M_max`. (See code comments & defaults). 

---

### Figure suggestions (from your current outputs)

* **Fig. 1 (MW)**: Rotation curves and residuals (Newtonian vs many‑path) with Gaia medians, plus vertical‑lag and χ² bars (existing MW plot).
* **Fig. 2 (SPARC)**: Distribution of per‑galaxy APE and "within ±20% of per‑galaxy best" for V2.1 versus V2.
* **Fig. 3 (Ablations)**: Rotation χ², lag, outer‑slope penalties as stacked bars for key removals (no radial modulation, looser saturation, no ring).
* **Fig. 4 (Comparative)**: Many‑path versus cooperative response on the MW benchmark.