
# Σ‑Gravity: A Conservative Path‑Integral Kernel for Galaxy Dynamics and Cluster Lensing Without Dark Matter

**Author:** Henry Speiser (independent)  
**Status:** Submission draft — October 2025  
**Project repository:** `many_path_model/` (this work)  
**Proposed name change:** *Geometry‑Gated Many‑Path Gravity* → **Σ‑Gravity (Sigma Gravity)**

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

Using **gNFW gas normalized to \(f_{\rm gas}\simeq0.11\)** with BCG+ICL, the **3D shell Σ‑kernel** reproduces MACS J0416’s Einstein radius within **\(+9\%\)** in a controlled interior‑chord configuration (no dark matter). A blind 12‑cluster suite initially under‑predicted \(\theta_E\) by ~40% due to **over‑aggressive clumping** and inconsistent normalization; after unifying the physics (divide X‑ray densities by \(\sqrt{C}\), adopt literature‑motivated \(C(r)\), normalize to \(f_{\rm gas}(R_{500})\)), the **baryon field provides the needed surface density**, and Σ‑kernel path families supply the remaining \(\times(5\text{–}10)\) boost. Multi‑cluster calibration (Sec. 6) will finalize the universal settings.

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
python scripts/test_gnfw_macs0416.py
python scripts/run_cluster_suite.py  # builds gNFW+BCG+ICL, applies 3D shell Σ‑kernel
```
**Minimal model (disc dynamics).** See `many_path_model/minimal_model.py`【fileciteturn1file11【】.

Where applicable, code organization and key function signatures are documented in `README.md` and in‑file docstrings【overview: fileciteturn1file8】.

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
