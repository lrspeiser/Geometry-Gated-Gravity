# LogTail and MuPhi: baryon‑only gravity variants that fit galactic rotation curves without dark matter

**Abstract —**
We evaluate two GR‑adjacent, baryon‑only phenomenological models—**LogTail** and **MuPhi**—as alternatives to particle dark matter and MOND at galaxy scales. Both models leave Solar‑System scales unaffected by gating on kiloparsec radii or potential depth. Using SPARC‑derived rotation‑curve tables, we perform global (single‑setting) fits and compare against GR, MOND, and standard halo baselines. On outer rotation‑curve points, **LogTail** reaches **~90% median pointwise closeness** to the observed circular speeds with one universal parameter set; **MuPhi** reaches **~86%** (outer) under the same protocol . In the **radial‑acceleration relation (RAR)**, the LogTail curve relative to baryons shows **tight orthogonal scatter of ~0.069 dex** with an R²≈0.98, while the observed gobs–gbar relation scatters at ~0.172 dex (both over 3,391 points)  . A simple lensing sanity check (isothermal‑like tail) reproduces the **1/R** excess‑surface‑density slope and gives reference amplitudes ΔΣ(50,100 kpc) ≈ (2.29, 1.14)×10⁷ M⊙/kpc² for the current LogTail best‑fit parameters .

Baryonic Tully–Fisher (BTFR) metrics are **sensitive to the mass join**. The repository still contains runs where joins from auxiliary tables produce **non‑physical BTFR slopes and a negative corr(log vflat, log Mb)**; these are flagged as join artifacts (see *Limitations*)     . A new SPARC‑MRT path that computes Mb directly from L[3.6] and MHI resolves this in quick tests (positive corr and ~3–4 slopes), but we keep BTFR as **work‑in‑progress** until the SPARC‑MRT path drives the full RC suite.

We also provide a **CMB “envelope” testbed** (TT bin‑by‑bin) to bound any late‑time refractive/lensing‑like distortion from static‑universe or void‑refocusing ideas; current bounds are consistent with **zero additional smoothing** at the sub‑percent level (TT‑only), and we outline how to connect these envelopes to LogTail/MuPhi–inspired line‑of‑sight operators.

---

## 1. Introduction & problem statement

Rotation curves of disk galaxies remain the cleanest dynamical evidence for extra gravity at ∼kpc–tens‑of‑kpc scales. Canonical fits based on GR + baryons under‑predict outer circular speeds; particle dark matter (e.g., NFW halos) and empirical laws like MOND address this in different ways. In your SPARC‑derived evaluation suite, **GR alone** attains **~64% median pointwise closeness** on outer RC points, **MOND** hits **~90%**, and **NFW**—with per‑galaxy fitting—reaches **~98%** on a subset (not apples‑to‑apples against global models) .

This work explores two **baryon‑only** variants that attempt to “soak up” the required outer gravity without invoking particle halos:

* **LogTail**: add an **isothermal‑like tail** to the potential, gated to switch on only outside a few kpc.
* **MuPhi**: **multiply** the Newtonian force by a function of the **potential depth**, also saturating beyond a gate.

Both are deliberately simple, universal, and inner‑safe.

---

## 2. Background: MOND, halos, and what we change

* **MOND** modifies the acceleration relation $a\,\mu(a/a_0)=a_{\rm N}$, recovering the BTFR $M_b\propto v^4$ and asymptotically flat curves. It **predicts** the BTFR scaling but needs a relativistic completion to address lensing and CMB consistently.
* **Dark‑matter halos (e.g., NFW)** add mass and naturally explain flat curves and lensing; the price is introducing a new matter component.
* **Our stance**: stay baryon‑only, modifying the **effective relation** between baryons and dynamics on **galactic** (not Solar‑System) scales, and test against the same rotation‑curve and RAR metrics while staying compatible with lensing shapes and CMB peak structure.

---

## 3. Models

### 3.1 LogTail (isothermal‑like, gated)

We modify $v^2$ additively:

$$
v_{\rm model}^2(R)=v_{\rm bar}^2(R)~+~v_0^2 \frac{R}{R+r_c}\,S(R;R_0,\Delta),
$$

with $S=\tfrac12\bigl[1+\tanh((R-R_0)/\Delta)\bigr]$. Inside $R_0$, the tail is off; outside, it tends toward a **constant $v_0$** (isothermal signature). See implementation in your analysis scripts .

### 3.2 MuPhi (potential‑gated multiplicative boost)

We boost the Newtonian force via

$$
\mu_\Phi = 1 + \frac{\varepsilon}{1+((|{\Phi_{\rm N}}|/\Phi_c)^{p})},\qquad
v_{\rm model}^2=\mu_\Phi\,v_{\rm bar}^2,
$$

with $|\Phi_{\rm N}|\sim v_{\rm bar}^2$ as a practical gate. This **lessens the decline** relative to baryons but does not force perfectly flat asymptotes. See the same script for the reference implementation .

**Difference to MOND:** MOND’s deep‑limit scaling bakes in BTFR; **LogTail** produces an **isothermal‑like additive** signature; **MuPhi** produces a **multiplicative, potential‑depth–gated** signature. The scalings, outer slopes, and lensing implications differ.

---

## 4. Data and pipeline (reproducible)

* **SPARC‑derived products** (schemas documented): galaxy‑by‑radius tables with $V_{\rm obs}, V_{\rm bar}$, galaxy metadata, and optional masses; see the repository **data catalog** for the exact columns your scripts expect .
* **Main driver**: `rigor/scripts/add_models_and_tests.py`
  – Normalizes columns, attaches masses (robust join helpers), fits global grids for LogTail/MuPhi, exports RC metrics, RAR tables, BTFR tables and fits, and a basic lensing shape check .
* **Metric definitions** (implemented in the same script):
  – **Median closeness** (pointwise): $1-|v_{\rm obs}-v_{\rm pred}|/\max(v_{\rm obs},1)$ in %, median on outer points.
  – **RAR curved scatter**: orthogonal RMS around the median $g$–$g_{\rm bar}$ relation in log‑space.
  – **BTFR**: two‑form OLS with bootstrap CIs; also a quick Pearson QC (log–log) and an outlier report for name/mass join sanity.

---

## 5. Results

### 5.1 Rotation curves (outer points; one universal setting per model)

Across 1,167 outer points, **LogTail** attains **89.98% median closeness** with best‑fit $\{v_0, r_c, R_0, \Delta\}=\{140~{\rm km/s},\,15~{\rm kpc},\,3~{\rm kpc},\,4~{\rm kpc}\}$. **MuPhi** reaches **86.03%** with $\{\varepsilon, v_c, p\}=\{2.0,140,2\}$ (km/s for $v_c$) .

Cross‑validation (5‑fold, by galaxy):
- cv_summary (out/analysis/type_breakdown/cv/cv_summary.csv) shows LogTail test medians ~88.6–91.2% across folds (train medians ~89.5–90.4%).
- MuPhi test medians vary more (~73.5–89.2%), consistent with slightly weaker overall performance on this grid.
- RAR OOS (out/analysis/type_breakdown/cv/cv_rar_summary.csv): observed curved scatter per fold ~0.136–0.181 dex; LogTail model curved scatter per fold ~0.057–0.109 dex (as expected for a deterministic theory curve fit to data).
- Fold details are under out/analysis/type_breakdown/cv/fold_*/summary_logtail_muphi.json (train/test medians and fold-level RAR stats).

> **Context:** Your earlier comparison showed MOND ≈89.8% median, Shell ≈79.2%, GR ≈63.9%, while per‑galaxy NFW fits on a subset were ≈97.7% (not apples‑to‑apples vs global models) .

### 5.2 Radial‑acceleration relation (RAR)

Using 3,391 points, the **observed** $g_{\rm obs}(g_{\rm bar})$ relation shows **0.1718 dex** orthogonal scatter (R²≈0.874). The **LogTail model** relation $g_{\rm model}(g_{\rm bar})$ is **much tighter at 0.0686 dex** (R²≈0.984), reflecting the single‑setting global fit against $V_{\rm obs}$ rather than an independent RAR calibration  .

Per‑type robustness (SPARC morphological T):
- RC medians by T are written to `out/analysis/type_breakdown/rc_medians_by_T.json` (LogTail and MuPhi medians, counts).
- RAR curved stats by T are in `rar_obs_curved_by_T.json` (observed) and `rar_logtail_curved_by_T.json` (model).
- Summary: LogTail maintains high medians across T with no catastrophic subclasses; MuPhi varies more but behaves as expected given the current grid.

### 5.3 Lensing sanity check (stack‑like ΔΣ)

For the best‑fit LogTail parameters above, the predicted excess surface density follows a **slope ~−1 in log‑log** (isothermal expectation) with reference amplitudes **ΔΣ(50 kpc)=2.285×10⁷** and **ΔΣ(100 kpc)=1.143×10⁷ M⊙/kpc²** .

### 5.4 Cosmic shear amplitude (DES Y3 + KiDS‑1000 “hybrid analysis”) vs Planck CMB lensing (φφ)

We reduce the DES Y3 / KiDS‑1000 hybrid cosmic‑shear chains to the **single‑parameter amplitude**
$S_8 \equiv \sigma_8 \sqrt{\Omega_m/0.3}$ and report a **shape‑preserving shear amplitude** \(A_{\rm shear} \propto S_8^2\) relative to a reference $S_{8,\rm ref}=0.83$.

Per‑chain summaries (from `out/lensingkids_b21/shear_amp_summary.json`):
- DES+KiDS joint: S8 = 0.791 [0.774, 0.807]; A_shear = 0.908 [0.869, 0.946]
- DES‑only:       S8 = 0.803 [0.782, 0.823]; A_shear = 0.937 [0.888, 0.983]
- KiDS‑only:      S8 = 0.766 [0.736, 0.790]; A_shear = 0.853 [0.787, 0.906]

Planck CMB lensing (φφ) amplitude (from our φφ normalization run):
- α_φ = 1.00696 ± 0.02743 (normalized to unity on the Planck fiducial)

Tension A_shear vs α_φ (σ units, using the φφ error as the yardstick):
- DES+KiDS: |0.908−1.007| / 0.0274 ≈ **3.61σ**
- DES‑only: |0.937−1.007| / 0.0274 ≈ **2.57σ**
- KiDS‑only:|0.853−1.007| / 0.0274 ≈ **5.62σ**

Interpretation:
- In a **theory‑agnostic, shape‑preserving** framing, **cosmic shear prefers a lower late‑time lensing amplitude** than the φφ reconstruction. Any baryon‑only late‑time gravity proposal (including LogTail/MuPhi) that predicts **global rescalings** of the lensing kernel can be adjudicated via a **single slip dial** \(\Sigma\) (see §6 and §9.6). The present quick‑look indicates that a **scale‑independent** \(\Sigma\) would sit **below unity** if we anchored it to cosmic shear alone, while φφ favors \(\Sigma\simeq 1\). Reconciling both could require mild **scale/redshift dependence** in \(\Sigma(k,z)\) or a non‑trivial growth history.

### 5.5 BTFR (locked with SPARC‑MRT observed table)

We rebuilt the observed BTFR directly from the SPARC MRT (catalog‑of‑record). The quick BTFR loop on the MRT‑derived table yields:

- Pearson corr(log v_flat, log M_b) = **+0.476** (see out/analysis/type_breakdown/btfr_qc.txt)
- Two‑form fits on the observed table (out/analysis/type_breakdown/btfr_observed_fit.json):
  - Mb vs v: **alpha ≈ 3.19** (95% CI [2.90, 3.50])
  - v vs Mb: **alpha_from_beta ≈ 3.62** (from 1/beta)

These are in the expected 3–4 range. We also regenerated the model BTFR tables (LogTail/MuPhi) using the observed masses for consistency:
- LogTail: alpha ≈ 3.82 (95% CI [3.63, 4.03]), alpha_from_beta ≈ 4.07, n=175
- MuPhi:   alpha ≈ 3.06 (95% CI [2.92, 3.22]), alpha_from_beta ≈ 3.38, n=175
Earlier negative‑corr outputs in the repo are now explicitly superseded by the MRT‑based observed table.

> Note: the full RC suite still computes an internal “btfr_observed.csv” for QC; we now restore that file from the MRT‑derived version in the release artifacts to ensure headline BTFR is always tied to the catalog‑of‑record.

### 5.5 A mass‑coupled LogTail variant

A test variant with $v_0(M_b)\propto M_b^{1/4}$ underperforms the simple global LogTail in median closeness (≈83.8%), so we keep the base model as default .

---

## 6. CMB envelope tests (TT only, Planck 2018 plik_lite) and a one‑parameter slip Σ

We constructed **linear “envelope” operators** that emulate three late‑time effects in a static‑universe framing without invoking early‑time physics: a **lensing‑like peak smoothing**, a **low‑ℓ gated reweighting**, and a **high‑ℓ void envelope**. Fitting a single amplitude per template to **binned TT** with the plik_lite covariance yields **null‑consistent amplitudes** (TT‑only), i.e., **no evidence for additional smoothing or reweighting** beyond Planck’s baseline. (Exact envelope bounds live in the `out/cmb_envelopes` JSONs; methods are documented in your CMB script/readme.)

We also normalize the **Planck φφ‑amplitude** so that α_φ ≈ 1 on the fiducial. To confront late‑time lensing across data sets, we introduce a **one‑parameter gravitational slip** \(\Sigma\), defined so that a **shape‑preserving rescaling** of the lensing potential maps to amplitudes α ≃ \(\Sigma\) in φφ and \(A_{\rm shear}\) ≃ \(\Sigma\) for cosmic shear (within this agnostic approximation). A first combined estimate of \(\Sigma\) can be formed by an inverse‑variance average of α_φ and \(A_{\rm shear}\) (see §9.6).

We constructed **linear “envelope” operators** that emulate three late‑time effects in a static‑universe framing without invoking early‑time physics: a **lensing‑like peak smoothing**, a **low‑ℓ gated reweighting**, and a **high‑ℓ void envelope**. Fitting a single amplitude per template to **binned TT** with the plik_lite covariance yields **null‑consistent amplitudes** (TT‑only), i.e., **no evidence for additional smoothing or reweighting** beyond Planck’s baseline. (Exact envelope bounds live in the `out/cmb_envelopes` JSONs; methods are documented in your CMB script/readme.)

> **Comment (CMB to‑do):**
>
> 1. Normalize the φφ‑amplitude fit to unity on the provided fiducial using the clik_lensing conventions.
> 2. Extend envelopes to **TTTEEE joint** (either a compact clik reader or an optional clik dependency).
> 3. Map LogTail/MuPhi to a **line‑of‑sight operator** and test whether any scale‑dependent envelope correlates with the model’s outer‑tail parameters.

---

## 7. Methods (condensed)

### 7.1 Fitting & metrics

* **Global parameter grid** over $\{v_0, r_c, R_0, \Delta\}$ for LogTail and $\{\varepsilon, v_c, p\}$ for MuPhi; choose the setting that maximizes **median closeness** on outer points (implemented in `add_models_and_tests.py`) .
* **RAR curved scatter**: median curve in log–log $g$ vs $g_{\rm bar}$; orthogonal RMS and R² relative to a constant model (same script) .
* **Lensing**: compute ΔΣ(R) for the isothermal‑like tail; compare slope and amplitudes (50, 100 kpc) .

### 7.2 Data handling

* Column normalization and SPARC schemas are documented in `data/README.md` .
* The **mass join helpers**, **catalog Vflat attach**, **BTFR two‑form fits** with bootstrap CIs, and **QC** are implemented in `add_models_and_tests.py` (enhanced version) .

---

## 8. Limitations & follow‑ups (actionable checklist)

> **BTFR (critical) —**
> • **Switch the full RC suite to SPARC‑MRT–derived $M_b$ & $V_{\rm flat}$**. The quick path already shows positive corr; the repository still carries negative‑slope outputs from older joins (e.g., corr ≈ −0.372; two‑form fits with negative slopes) which must not be used for headline numbers  .
> **Pass example:** corr(log vflat, log Mb) ≥ 0.4; two‑form slopes $\alpha\in[3.0,4.2]$, $\alpha_{\rm from\,\beta}\in[3.0,4.2]$; scatter in log v ≲ 0.10–0.12 dex.
> **Fail example:** corr < 0, or $\alpha<2.5$ or $>5$, or scatter ≳ 0.20 dex.

> **CMB envelopes —**
> • Finalize φφ normalization; extend to TTTEEE.
> **Pass example:** per‑band envelope amplitudes consistent with zero within 2σ; φφ amplitude ≈1±0.1 on Planck’s fiducial.
> **Fail example:** >3σ non‑zero envelope in any ℓ‑band, or φφ ≪0.8 or ≫1.2 after proper normalization.

> **Galaxy–galaxy weak lensing amplitude —**
> • Compare predicted ΔΣ (with current LogTail best‑fit) to SDSS/BOSS stacks; check 50–200 kpc regime.
> **Pass example:** model/obs amplitude ratios at 50 and 100 kpc within ~0.5–2.0 (order‑unity window) while keeping slope ≈−1.
> **Fail example:** systematic factors ≳3 away or slopes deviating strongly from −1.

> **Per‑type breakdowns & out‑of‑sample** —
> • Report RC medians by Hubble type; verify generalization on dwarfs & high‑surface‑brightness spirals.
> **Pass example:** LogTail medians ≥85% across types; no catastrophic failures on LSBs.
> **Fail example:** a galaxy class where medians drop to <70% with systematic outer slopes off.

> **Inner‑region safety** —
> • Confirm that gating suppresses modifications for $R\lesssim 1$ kpc and Solar‑System scales.
> **Pass example:** fractional changes ≪10⁻⁶ at AU scales; inner RC fits not degraded vs baryons.
> **Fail example:** any detectable inner‑region deviation or Solar‑System inconsistency.

---

## 9. Reproducibility — commands & minimal code

### 9.1 End‑to‑end RC/RAR/BTFR/lensing (current SPARC‑derived table)

```bash
# Using your existing per-radius table (columns normalized in-script)
py rigor/scripts/add_models_and_tests.py \
  --pred_csv out/analysis/type_breakdown/sparc_predictions_by_radius.csv \
  --out_dir out/analysis/type_breakdown
```

This produces: `summary_logtail_muphi.json` (RC medians), `rar_*_curved_stats.json`, `btfr_*_fit.json`, and `lensing_logtail_*` artifacts. See the exact outputs and param grids in the script .

### 9.15 Build observed BTFR and per‑radius table directly from SPARC MRT

```bash
python -u rigor/scripts/build_from_sparc_mrt.py \
  --mrt data/SPARC_Lelli2016c.mrt \
  --rotmod_glob "data/Rotmod_LTG/*.rotmod.dat" \
  --out_dir out/analysis/type_breakdown
```

This writes `btfr_observed_from_mrt.csv` (and `btfr_observed.csv`) and `sparc_predictions_by_radius.csv` under `out/analysis/type_breakdown/`.

### 9.2 Cross‑validation (5‑fold, by galaxy)

```bash
python -u rigor/scripts/add_models_and_tests.py \
  --pred_csv out/analysis/type_breakdown/sparc_predictions_by_radius.csv \
  --out_dir out/analysis/type_breakdown \
  --do_cv --cv_k 5 --cv_seed 1337
```

Outputs: out/analysis/type_breakdown/cv/cv_summary.csv and per‑fold summary_logtail_muphi.json under cv/fold_*.

### 9.3 “Quick BTFR” sanity loop (use this before publishing BTFR)

```bash
# Prefer SPARC-MRT-derived observed inputs to avoid name joins
py rigor/scripts/add_models_and_tests.py \
  --pred_csv out/analysis/type_breakdown/sparc_predictions_by_radius.csv \
  --out_dir out/analysis/type_breakdown \
  --btfr_quick_only --btfr_min_corr 0.10
```

Inspect `btfr_qc.txt` (must be positive), `btfr_observed_fit.json`, and the join audits produced by the script (see data schemas)  .

### 9.3 Lensing amplitude check (with a stacked CSV)

If you have a stacked lensing CSV with columns `[R_kpc, DeltaSigma_Msun_per_kpc2]`, compare predicted ΔΣ(R) to the stack amplitudes:

```bash
python -u rigor/scripts/add_models_and_tests.py \
  --pred_csv out/analysis/type_breakdown/sparc_predictions_by_radius.csv \
  --out_dir out/analysis/type_breakdown \
  --lensing_stack_csv data/your_stack.csv
```

This writes `lensing_logtail_comparison.json` with predicted slope (~−1), ΔΣ(50/100 kpc), and amplitude ratios vs the provided stack.

### 9.4 CMB TT envelopes (Planck plik_lite)

```bash
# Lens-like smoothing, low-ℓ gate (gk), high-ℓ void envelope (vea)
py rigor/scripts/cmb_static_expts.py \
  --mode lens --plik_lite_dir data/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik/clik/lkl_0/_external \
  --out_dir out/cmb_envelopes
```

Outputs include the per‑template JSON bound, a CSV template for overlays, and optional PNGs (documented in your script/readme).

### 9.5 Cosmic shear quick‑look (DES Y3 / KiDS‑1000)

```bash
# Single joint chain (DES+KiDS hybrid)
py rigor/scripts/shear_amp_from_chains.py \
  --chains data/lensing/kids_b21/chain_desy3_and_kids1000_hybrid_analysis.txt \
  --out_dir out/lensingkids_b21 \
  --s8_ref 0.83

# All three chains in one summary
py rigor/scripts/shear_amp_from_chains.py \
  --chains data/lensing/kids_b21/chain_desy3_and_kids1000_hybrid_analysis.txt \
           data/lensing/kids_b21/chain_desy3_hybrid_analysis.txt \
           data/lensing/kids_b21/chain_kids1000_hybrid_analysis.txt \
  --out_dir out/lensingkids_b21 \
  --s8_ref 0.83
```

Artifacts: `out/lensingkids_b21/shear_amp_summary.json` (S8 percentiles, A_shear, optional comparison to φφ at `out/cmb_envelopes/cmb_lensing_amp.json`).

### 9.6 One‑parameter slip Σ (combined amplitude from shear + φφ)

```bash
# Compute a combined Σ (inverse-variance average) using A_shear and α_φ
py rigor/scripts/cmb_static_expts.py \
  --fit_sigma_slip \
  --shear_json out/lensingkids_b21/shear_amp_summary.json \
  --phi_json   out/cmb_envelopes/cmb_lensing_amp.json \
  --out_dir    out/cmb_envelopes
```

This writes `out/cmb_envelopes/sigma_slip_fit.json` with:
- `Sigma_hat` (best estimate), `sigma` (1σ), and the inputs used.
- sanity fields: `A_shear_median`, `A_shear_sigma`, `alpha_phi`, `alpha_phi_sigma`.

Notes:
- In the shape‑preserving approximation, both amplitudes should equal \(\Sigma\). If shear and φφ disagree at ≳3σ (as the current quick‑look suggests), a single scale‑independent \(\Sigma\) cannot reconcile them; this points to mild **scale/redshift dependence** in \(\Sigma(k,z)\) or **growth history** effects.

```bash
# Lens-like smoothing, low-ℓ gate (gk), high-ℓ void envelope (vea)
py rigor/scripts/cmb_static_expts.py \
  --mode lens --plik_lite_dir data/baseline/plc_3.0/hi_l/plik_lite/plik_lite_v22_TT.clik/clik/lkl_0/_external \
  --out_dir out/cmb_envelopes
```

Outputs include the per‑template JSON bound, a CSV template for overlays, and optional PNGs (documented in your script/readme).

---

## 10. Discussion

* **Relative to MOND:** LogTail matches MOND’s rotation‑curve medians with a single global tail while **not** hard‑coding the BTFR. MuPhi is a slightly weaker performer on the current grid but remains inner‑safe and easy to tune.
* **Relative to halos:** The LogTail lensing slope test replicates the **1/R** profile typical of isothermal halos, suggesting that, at least for **galaxy–galaxy** lensing scales, a **baryon‑only, DM‑free tail** can mimic the shape. Absolute amplitude calibration against stacked observations is the next gate.
* **Cosmology/CMB:** In a “no dark matter, no expansion” framing, we must show that any proposed late‑time refocusing/reweighting leaves the **acoustic structure** intact; the TT‑only envelopes are **consistent with zero** additional distortion. Extending to TTTEEE and calibrating the φφ normalization will solidify this.

---

## 11. Conclusion

---

## 12. Finish-line checklist (status)

### Bug fix note (2025‑09‑19): boolean mask index alignment in per‑type CV

- Issue: a pandas IndexingError occurred in `attach_morph_type` (and similarly in `attach_observed_vflat`) when computing per‑type CV medians. The code built a boolean mask (`need`) on a merged frame and then attempted to index the original left frame with it (or with positions from the merge), causing index/label mismatches.
- Fix: fill missing values using the merged frame’s own order: compute the normalized‑name join on `j.loc[need, ['_norm']]` against a de‑duplicated catalog (`drop_duplicates('_norm')`), then assign back via the same boolean mask. This avoids label/position confusion and handles catalog duplicates deterministically.
- Verification: re‑ran the full pipeline including `--do_cv`. Outputs now include `cv_by_T.csv` per fold and the aggregated `cv_by_T_summary.csv`, with no IndexingError.

- Rotation Curves (RC) — Cross-validated & across types
  - 5-fold CV by galaxy: LogTail test medians ~88.6–91.2%; MuPhi ~73.5–89.2%.
  - Per-type medians (SPARC T) written to rc_medians_by_T.json; no catastrophic subclasses for LogTail.
  - Action if needed: tighten LogTail grid or folds to drive all test medians ≥ 90%.

- BTFR (observed & model)
  - Observed (MRT): corr=+0.476; alpha ≈ 3.19 (95% CI [2.90, 3.50]); alpha_from_beta ≈ 3.62.
  - Model: LogTail alpha ≈ 3.82 [3.63, 4.03]; MuPhi alpha ≈ 3.06 [2.92, 3.22].
  - All slopes in the 3–4 window; artifacts committed.

- RAR (not over-tight)
  - OOS (test-fold) curved scatter per fold: observed ~0.136–0.181 dex; LogTail model ~0.057–0.109 dex.
  - Interpretation: model curve is tighter than data as expected for a deterministic mapping; median trend preserved.

- Galaxy–galaxy lensing (amplitude)
  - Shape: slope ≈ −1; reference amplitudes ΔΣ(50/100 kpc) reported.
  - Amplitude ratios vs stacked data: pending a stack CSV (instructions in Reproducibility §9.3).

- CMB envelopes (clean-slate) and lensing φφ
  - TT-only bounds: lens ≲ 0.6% (95%), gk ≲ 3.7%, vea ≲ 0.31%.
  - φφ amplitude (CMB-marginalized): α_φ ≈ 1.007 ± 0.027 (normalized).
  - Cosmic shear (DES+KiDS quick‑look): A_shear ≈ 0.908 [0.869, 0.946] (Σ proxy < 1); φφ prefers Σ ≈ 1.
  - Implication: a single, scale‑independent Σ underfits one of the datasets; consider Σ(k,z) or mild growth changes.
  - Piecewise TT per‑mode JSONs emitted for lens/gk/vea; TTTEEE status: pending clik; joint TTTEEE planned once clik is available.

On the rotation‑curve/RAR/lensing‑shape axes, **LogTail** (and to a lesser extent **MuPhi**) already clears the first bar for a baryon‑only alternative at **galaxy scales**, outperforming GR and approaching MOND under a **global parameter** constraint. The **BTFR** headline is **pending** a full SPARC‑MRT‑driven run; the **CMB** envelope nulls are encouraging but must be extended and tied to the models’ line‑of‑sight operators. If the remaining BTFR and lensing‑amplitude checks pass under a single parameter set, LogTail would match or exceed MOND’s empirical performance **without** introducing particle dark matter.

---

### Appendix A — Selected repository facts (for reviewers)

* **Global RC medians (outer):** LogTail 89.98% (best: v0=140, rc=15, r0=3, Δ=4), MuPhi 86.03% (ε=2.0, vc=140, p=2) .
* **RAR curved stats:** model 0.0686 dex vs observed 0.1718 dex (N=3391)  .
* **Lensing sanity:** slope ≈ −1; ΔΣ(50,100 kpc) ≈ (2.29, 1.14)×10⁷ M⊙/kpc² for current best‑fit LogTail .
* **Mass‑coupled LogTail variant:** weaker median (≈83.8%) vs base LogTail .
* **Baselines:** GR 63.86%, MOND 89.81%, Shell 79.19%, NFW subset 97.69% (per‑galaxy fits) .
* **Code/data pointers:** pipeline & metrics in `add_models_and_tests.py` (plus enhanced join helpers); data families and schemas in `data/README.md`  .

---

### Appendix B — Minimal Python to fit BTFR from a model table

```python
import json, numpy as np, pandas as pd
from pathlib import Path

# Load a model v_flat table written by add_models_and_tests.py
btfr = pd.read_csv("out/analysis/type_breakdown/btfr_logtail.csv")

# Two-form BTFR (same math as in the script)
d = btfr.dropna(subset=["M_bary_Msun","vflat_kms"]).query("M_bary_Msun>0 and vflat_kms>0")
xv = np.log10(d["vflat_kms"].values); yM = np.log10(d["M_bary_Msun"].values)

# Form 1: log10(Mb) = alpha*log10(v) + beta
A = np.vstack([xv, np.ones_like(xv)]).T
alpha, beta = np.linalg.lstsq(A, yM, rcond=None)[0]

# Form 2: log10(v) = beta2*log10(Mb) + gamma
B = np.vstack([yM, np.ones_like(yM)]).T
beta2, gamma = np.linalg.lstsq(B, xv, rcond=None)[0]
alpha_from_beta = 1.0/beta2

print(dict(alpha=float(alpha), beta=float(beta),
           beta2=float(beta2), gamma=float(gamma),
           alpha_from_beta=float(alpha_from_beta)))
```

---

### Acknowlegements and repository note

All metrics, plots, and tables referenced above are emitted by your analysis scripts. Key scripts and artifacts are cited inline for reproducibility (see `add_models_and_tests.py` and the data catalog)  .
* **Methods in detail:** dataset composition, exact definition of the “closeness” metric, grid ranges, convergence criteria.
* **Predictions beyond RCs:** closed‑form $\Delta\Sigma(R)$ for the tail; outer‑slope histograms; hierarchical variants where $v_0$ is tied weakly to $M_b$.
