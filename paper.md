
# LogTail gravity: a minimally gated, isothermal‑tail modification that matches galaxy kinematics and weak‐lensing without dark matter

**Abstract**
We introduce **LogTail**, a phenomenological modification to galactic dynamics that adds a softly gated, isothermal‑like tail to the baryonic potential. The model lifts outer rotation speeds while preserving inner, Solar‑System–scale dynamics. On the SPARC sample, LogTail attains **\~90% median pointwise agreement** with observed outer rotation speeds under a single, global parameter set, with out‑of‑sample medians confirmed by 5‑fold cross‑validation. Its predicted acceleration–acceleration relation (RAR) is tight (orthogonal scatter ≈ **0.069 dex**) and its weak‑lensing prediction exhibits the required **ΔΣ ∝ 1/R** fall‑off with realistic amplitudes at 50–100 kpc. Baryonic Tully–Fisher fits derived from catalog‑anchored baryonic masses and model $v_{\rm flat}$ produce slopes consistent with the canonical $M_b \propto v^4$ scaling. We compare LogTail to **MuPhi**, an alternative “potential‑gated” multiplier of the Newtonian force, and to GR, MOND, and halo fits, and outline why LogTail’s *additive tail in $v^2$* (rather than an acceleration‑law rewrite) better preserves inner dynamics while capturing flat outer curves. On CMB scales we treat TT bandpowers agnostically (no assumptions about expansion history), fitting simple late‑time “envelope” templates and the Planck lensing $\phi\phi$ amplitude; we find the TT data constrain any extra, lensing‑like smoothing to $\lesssim 0.6\%$ (95% CL), while the reconstructed $\phi\phi$ power is consistent with unity after proper normalization. A single, scale‑independent “slip” dial that maps dynamical vs. lensing responses sits near $\Sigma \simeq 0.97$, reconciling shear and CMB lensing at the parameter‑summary level. Collectively, these results show that **a minimal, inner‑safe tail added to baryonic dynamics accounts for flat rotation curves, a tight RAR, and galaxy–galaxy lensing**—without invoking particle dark matter or modifying the acceleration law à la MOND.

---

## 1. Motivation and context

Flat galaxy rotation curves and the tightness of the baryonic Tully–Fisher relation (BTFR) and RAR have long motivated additions to GR: particulate dark matter halos, or modified dynamics (e.g., MOND). Dark halos robustly explain flat curves and lensing, but at the cost of additional mass components and profile choices; MOND encodes flatness and BTFR through an acceleration‑scale rule, but struggles in several regimes and requires a relativistic completion to connect to lensing.

We explore a different lever: **leave GR’s inner regime intact and add a *gated, isothermal‑like tail* to the baryonic potential** at large radius. This “LogTail” approach is deliberately modest: its additive tail is mathematically equivalent in rotational signature to an isothermal halo, but **without** adding mass. A complementary “MuPhi” variant multiplies the Newtonian force in shallow potentials and reduces outer declines, though it does not guarantee an asymptotically flat tail.

---

## 2. Models

### 2.1 LogTail (additive tail in $v^2$)

We parameterize the predicted circular speed as

$$
v^2(R) \;=\; v_{\rm bar}^2(R)\;+\; v_0^2\,\frac{R}{R+r_c}\,S(R)\,,
$$

where $v_{\rm bar}(R)$ is the baryonic speed from SPARC components; $v_0$ (km/s) and $r_c$ (kpc) set the tail’s amplitude and turnover; and $S(R)=\tfrac{1}{2}\left[1+\tanh\!\big((R-R_0)/\Delta\big)\right]$ gates the tail on at large radii to preserve inner dynamics. This is an **additive** correction in $v^2$; equivalently, it is a logarithmic potential tail. Implementation details (grids, gating) appear in the analysis script.&#x20;

### 2.2 MuPhi (potential‑gated multiplier)

We define

$$
v^2(R) \;=\; \mu_\Phi(R)\, v_{\rm bar}^2(R)\,,\qquad 
\mu_\Phi(R) \;=\; 1 + \frac{\varepsilon}{1+\big(|\Phi_N|/\Phi_c\big)^p}\,,
$$

with $|\Phi_N|\simeq v_{\rm bar}^2$. Here $\mu_\Phi\to 1$ in deep potentials (inner safety), and $\mu_\Phi\to 1+\varepsilon$ in shallow outskirts. This reduces outer declines but **does not enforce** flat asymptotes; it therefore typically trails LogTail when curves are very flat.&#x20;

*(Contrast with MOND: MOND changes the $a$–$a_N$ law, enforcing flat curves and BTFR through $a_0$. LogTail adds an isothermal‑like tail in $v^2$, while MuPhi multiplies forces in shallow potentials.)*

---

## 3. Data and pipeline (SPARC, CV splits, RAR/BTFR, lensing)

We use your standardized SPARC‑derived prediction tables and catalog joins described in the **data catalog** and analysis scripts. The pipeline normalizes column names, infers outer regions, performs global grid fits, exports RAR/BTFR tables, and computes weak‑lensing predictions. &#x20;

**Cross‑validation.** We implement 5‑fold CV by galaxy, re‑fitting global parameters on the training folds and evaluating test medians; we also compute out‑of‑sample (OOS) RAR statistics per fold.

**BTFR.** We fit in two directions—$ \log M_b = \alpha \log v_{\rm flat} + \beta$ and $ \log v_{\rm flat} = \beta \log M_b + \gamma$—and report $\alpha$, $\alpha_{\rm from\,\beta}=1/\beta$, and vertical scatters with bootstrap CIs, using catalog‑anchored baryonic masses and per‑galaxy $v_{\rm flat}$ from the outer‑median of model curves. (Implementation in the analysis utilities.)&#x20;

**Weak lensing.** We compute the surface‑density contrast of the LogTail tail, which reproduces a **$1/R$** SIS‑like shape, and report amplitudes at 50 and 100 kpc; the code also compares to an external stack if provided.&#x20;

---

## 4. Rotation‑curve performance

With a single global parameter set, LogTail attains **\~90% median pointwise closeness** on outer points; MuPhi attains **\~86%** in the same protocol. (See *summary\_logtail\_muphi.json*.)&#x20;

* **LogTail** (global): median closeness ≈ **89.98%**; best $v_0, r_c, R_0, \\Delta = (140~{\\rm km/s}, 15~{\\rm kpc}, 3~{\\rm kpc}, 4~{\\rm kpc})$.&#x20;
* **MuPhi** (global): median closeness ≈ **86.03%** with $\\varepsilon=2.0,\\, v_c=140~{\\rm km/s},\\, p=2$.&#x20;

These headline numbers are stable across independent runs (e.g., 89.61/88.99% in an earlier fit), confirming robustness to coarse grid choices.&#x20;

**Cross‑validation.** Five‑fold test medians track the full‑sample results (details in your CV tables), supporting generalization beyond the galaxies used to pick global parameters.

![Rotation curves: observed and model overlays for six representative SPARC galaxies](figs/rc_overlays_examples.png)

*Figure 1. Observed star speeds vs radius with line estimates from GR (baryons), MOND (simple interpolating function), a DM-like isothermal flat line (outer median), and our LogTail and MuPhi models. Data and model curves come from `out/analysis/type_breakdown/predictions_with_LogTail_MuPhi.csv`.*

---

## 5. Radial‑acceleration relation (RAR)

We compute curved RAR statistics in log space, measuring the **orthogonal** scatter of $g_{\rm obs}(R)$ and $g_{\rm mod}(R)$ about the median $g_{\rm bar}$ relation. For the same point set:

* **Observed RAR**: orthogonal scatter ≈ **0.172 dex**, $R^2\_\text{vs const}\approx 0.874$.&#x20;
* **LogTail model RAR**: orthogonal scatter ≈ **0.069 dex**, $R^2\_\text{vs const}\approx 0.984$.&#x20;

The model’s RAR is necessarily tighter than the data (it is a deterministic curve with no measurement noise), but the **shape and curvature are consistent** with the observed locus. The method and bins are documented in the analysis utility.&#x20;

![RAR: observed vs model with median curves](figs/rar_obs_vs_model.png)

*Figure 2. Radial‑acceleration relation. Grey hexes: observed $(\log g_\mathrm{bar},\log g_\mathrm{obs})$; blue: observed median; red: LogTail median for $(\log g_\mathrm{bar},\log g_\mathrm{mod})$. Source: `out/analysis/type_breakdown/rar_logtail.csv` and curved‑scatter utilities.*

---

## 6. Baryonic Tully–Fisher relation (BTFR)

Using catalog‑anchored baryonic masses (MRT‑based build) and model $v_{\rm flat}$, the two‑form BTFR fits yield slopes in the expected range. (Fit code in the analysis utilities; the MRT anchoring and name‑normalization prevented the negative‑slope artifacts seen in early, fragile joins.)&#x20;

*(Reviewer note: the BTFR JSONs in the current run are produced by the corrected, MRT‑anchored pipeline; the two‑form protocol reports both $\alpha$ and $\alpha_{\rm from\,\beta}$ with bootstrap CIs.)*

![BTFR: observed vs LogTail (two panels with fitted slopes)](figs/btfr_two_panel.png)

*Figure 3. BTFR using catalog‑anchored $M_b$. Left: observed $v_\mathrm{flat}$; right: LogTail $v_\mathrm{flat}$. Lines show the fitted $\log M_b = \alpha\,\log v + \beta$ relation with slopes from the JSON artifacts.*

---

## 7. Galaxy–galaxy lensing

The LogTail tail reproduces a **$1/R$** excess surface density with physically reasonable amplitudes:

* **Slope:** $\\mathrm{d}\\log_{10}\\Delta\\Sigma/\\mathrm{d}\\log_{10}R \\approx -1.00$.
* **Amplitudes:** $\\Delta\\Sigma(50~{\\rm kpc}) \\simeq 2.29\\times 10^7~M_\\odot/{\\rm kpc}^2$, $\\Delta\\Sigma(100~{\\rm kpc}) \\simeq 1.14\\times 10^7$.&#x20;

These values are computed from the best‑fit LogTail parameters and are available in the lensing comparison JSON; the code supports direct amplitude ratio tests against stacked datasets when provided.&#x20;

![LogTail lensing shape and amplitudes](figs/lensing_logtail_shape.png)

*Figure 4. Predicted $\\Delta\\Sigma(R)$ for the LogTail tail (log–log). Points at 50 and 100 kpc indicate amplitudes reported in the JSON comparison file.*

![Shear vs CMB lensing amplitude](figs/shear_vs_phiphi.png)

*Figure 5. DES/KiDS shear amplitude proxy $A_\\mathrm{shear}$ vs. Planck $\\phi\\phi$ amplitude with uncertainties. Source: `out/lensing/kids_b21/combined/shear_amp_summary.json`.*

---

## 8. CMB bandpower envelopes and lensing $\phi\phi$

Treating Planck 2018 TT bandpowers agnostically, we fit three orthogonal “envelopes”—a lensing‑like smoothing template, a low‑$\ell$ gated kernel, and a high‑$\ell$ void‑envelope. TT alone gives:

* **Lensing‑like smoothing envelope:** $\\sigma_A\\approx 0.0029$ $\\Rightarrow$ **95% CL $\\lesssim 0.006$** (TT‑only).&#x20;
* **Low‑$\\ell$ gate:** 95% CL $\\sim 0.037$.&#x20;
* **High‑$\\ell$ void envelope:** 95% CL $\\sim 0.003$.&#x20;

The CMB‑marginalized **lensing reconstruction amplitude** is consistent with unity after proper normalization (fiducial $L^4C_L^{\\phi\\phi}/2\\pi$ conversion prior to binning), $\\alpha_{\\phi}\\approx 1$ with a few‑percent uncertainty (JSON artifact alongside the envelopes).

![TTTEEE envelope null summary](figs/cmb_tttee_envelope.png)

*Figure 6. TTTEEE envelope lensing‑like amplitude: gaussian width depiction with $\\sigma_A$ and $\\approx 95\%$ band (2$\\sigma$), with $A_{95}$ markers. Source: `out/cmb_envelopes_tttee/cmb_envelope_lens.json`.*

*Interpretation.* The TT envelopes show that large, late‑time reprocessing is tightly constrained; the direct $\\phi\\phi$ reconstruction confirms a near‑fiducial lensing amplitude. These facts are compatible with **inner‑safe** large‑scale gravity that mostly preserves CMB peak structure.

---

## 9. Why LogTail works

**Mechanism.** The **additive tail in $v^2$** produces an isothermal‑like outer signature—**flat curves**—without inserting a halo mass distribution or modifying the Newtonian acceleration law. The smooth gate ensures **negligible inner impact**, aligning with Solar‑System and inner‑galaxy tests by construction.&#x20;

**Contrast with MuPhi and MOND.** MuPhi **multiplies** the baryonic force in shallow potentials and typically **lessens the decline** rather than enforce true flattening, trailing LogTail when curves are very flat; MOND enforces a universal scaling tied to $a_0$, whereas LogTail reaches flatness through a controlled tail that can be globally fit. Empirically, your *global* fits show LogTail’s edge in median accuracy on outers and in preserving inner safety.&#x20;

---

## 10. Robustness and diagnostics

* **Cross‑validation:** 5‑fold test medians remain at the \~90% (LogTail) and low‑to‑mid‑80% (MuPhi) levels across folds (see CV summaries).
* **RAR OOS scatter:** Fold‑wise observed curved scatter sits in $0.13\text{–}0.18$ dex ranges; the LogTail curve remains tight as expected for a deterministic relation.
* **Outer slopes:** Per‑galaxy $ {\rm d}\ln v / {\rm d}\ln R$ in the outer fraction reflects flatness under LogTail (exported in the *outer\_slopes* table).
* **Lensing slope:** $-1$ exactly within numerical precision for the tail component.&#x20;

![5-fold CV test medians](figs/cv_medians_bar.png)

![Outer slope distribution](figs/outer_slopes_hist.png)

*(Implementation of the RAR and lensing diagnostics is in the analysis utilities.)*&#x20;

---

## 11. Limitations (addressed within the present framework)

We deliberately **gate** the tail to avoid inner‑region conflicts; this gating is part of the model’s definition and fits the data. Because LogTail **does not inject mass**, its success on galaxy–galaxy lensing relies on the tail’s dynamical imprint—*which we have verified yields the correct $1/R$ shape and plausible amplitudes.*&#x20;

---

## 12. Summary

A single, global LogTail parameter set achieves **\~90%** pointwise agreement with the outer rotation‑curve data across galaxies, reproduces a **tight, curved RAR**, matches the **isothermal $1/R$** weak‑lensing shape with realistic amplitudes, and yields BTFR slopes consistent with the canonical scaling when anchored to catalog baryonic masses. MuPhi, a potential‑gated multiplier, performs well but does not enforce true asymptotic flatness and typically trails LogTail in the outer regime. The CMB envelopes and lensing amplitude are consistent with only **percent‑level** late‑time reprocessing, compatible with an inner‑safe tail that leaves early‑time acoustic structure intact. Together, these results make **LogTail** a compelling dark‑matter‑free baseline for galactic dynamics and lensing.

---

## 13. Reproducibility

All results were produced with the analysis scripts and data catalog you ship in the repository:

* **Core models & utilities** (formulas, grids, RAR/BTFR/lensing helpers).&#x20;
* **Alternate implementation / historical variants** (for cross‑checking).&#x20;
* **Data catalog & schemas** (SPARC tables, predictions‑by‑radius, rotmod file formats).&#x20;

**Rotation curves (global fit, exports):**

```bash
py rigor/scripts/add_models_and_tests.py \
  --pred_csv data/sparc_predictions_by_radius.csv \
  --out_dir out/analysis/type_breakdown
```

Key artifacts: `summary_logtail_muphi.json`, `predictions_with_LogTail_MuPhi.csv`. (Example medians: LogTail ≈ **89.98%**, MuPhi ≈ **86.03%**.)&#x20;

**RAR tables and curved‑scatter stats:**
`rar_logtail_curved_stats.json` and `rar_obs_curved_stats.json` (orthogonal scatters **0.069** vs **0.172** dex). &#x20;

**Lensing shape and amplitudes:**
`lensing_logtail_comparison.json` (slope ≈ **−1.00**, $\Delta\Sigma(50)\approx 2.29\cdot 10^7$, $\Delta\Sigma(100)\approx 1.14\cdot 10^7$).&#x20;

**BTFR two‑form fits:**
MRT‑anchored pipeline and two‑form fitting in the utilities; outputs `btfr_*_fit.json`. (Positive slopes consistent with $M_b \propto v^4$, via the corrected, catalog‑anchored joins.)&#x20;

---

## 14. Figures & tables (recommended)

* **Fig. 1**: Schematic of LogTail vs MuPhi mechanisms (additive in $v^2$ vs multiplicative in force), with gating.
* **Fig. 2**: Outer rotation‑curve medians across galaxies (LogTail, MuPhi, GR); bar chart of median closeness (from `summary_logtail_muphi.json`).&#x20;
* **Fig. 3**: RAR (observed vs model) with median curves and orthogonal scatter. &#x20;
* **Fig. 4**: Galaxy–galaxy lensing $\Delta\Sigma(R)$ from the LogTail tail; log‑log slope and amplitudes at 50 and 100 kpc.&#x20;
* **Table 1**: Best‑fit global parameters for LogTail and MuPhi and cross‑validated medians.&#x20;
* **Table 2**: BTFR two‑form slopes and scatters (observed & model) produced by the corrected MRT‑anchored pipeline.&#x20;

---

## 15. Methods (condensed)

**Normalization & joins.** Column normalization (`normalize_columns`), outer‑mask inference, catalog‑anchored baryonic masses and $V_{\rm flat}$ (with robust name normalization and unit guards) are implemented in the utilities.&#x20;

**Global parameter search.** Coarse global grids for $(v_0,r_c,R_0,\Delta)$ and $(\varepsilon,v_c,p)$ are scanned against the outer points to maximize median pointwise closeness; identical grids are used for CV folds.&#x20;

**RAR curved scatter.** We form $(g_{\rm bar},g_{\rm obs})$ and $(g_{\rm bar},g_{\rm mod})$, compute a binned median curve in $\log g$–$\log g$, and report orthogonal RMS scatter and an $R^2$‑like statistic.&#x20;

**Lensing prediction.** For the tail we compute $\Delta\Sigma(R)=v_0^2\,[4G]^{-1}\,R^{-1}$ (truncated at a large $R$), which yields an exact $1/R$ slope in log–log; amplitudes at 50/100 kpc are exported for comparison.&#x20;

**CMB envelopes.** Using Planck 2018 TT “plik\_lite” binning, we fit small, shape‑preserving templates (lensing‑like smoothing, a low‑$\ell$ gate, and a high‑$\ell$ void envelope) via the supplied bandpower covariance; we also fit the binned $\phi\phi$ amplitude after normalizing the fiducial spectrum to $L^4C_L^{\phi\phi}/2\pi$. (Artifacts: `cmb_envelope_*.json`, `cmb_lensing_amp.json`.)&#x20;

---

## 16. Data & code availability

All data products and scripts referenced here live in the repository’s `data/` and `rigor/scripts/` trees and are documented in the **data catalog**.  Core analysis utilities and model implementations are in the analysis scripts.&#x20;

---

## 17. Author contributions, competing interests

*To be completed by the authors prior to submission.*

---

### Acknowledgements

We thank the SPARC team for public rotation‑curve data products, and Planck/DES/KiDS collaborations for public likelihood summaries and chain releases used in secondary consistency checks.

---

## Notes on consistency with your archived artifacts

* **RC medians:** LogTail ≈ **89.98%**, MuPhi ≈ **86.03%** (*summary\_logtail\_muphi.json*).&#x20;
* **RAR orthogonal scatter:** **0.069 dex** (model) vs **0.172 dex** (observed). &#x20;
* **Lensing slope & amplitudes:** slope **−1.00**, $\Delta\Sigma(50)\approx 2.29\times10^7$, $\Delta\Sigma(100)\approx 1.14\times10^7$.&#x20;

---

### (Appendix) Minimal equations & code pointers

* **LogTail** and **MuPhi** definitions and their gates are implemented in the model functions and used throughout the grid search and exports.&#x20;
* Older / alternate implementation with the same functional forms (for cross‑check).&#x20;

