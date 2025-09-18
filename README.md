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

### Data and reproducibility

All numbers above come from the project’s exported summaries of the latest runs: the refined LogTail/MuPhi fit (`summary_logtail_muphi.json`) and the extended comparison (`accuracy_comparison_extended.json`). Paths and full parameter listings are preserved in those files.

---

#### (Optional next section you can add immediately)

* **Methods in detail:** dataset composition, exact definition of the “closeness” metric, grid ranges, convergence criteria.
* **Predictions beyond RCs:** closed‑form $\Delta\Sigma(R)$ for the tail; outer‑slope histograms; hierarchical variants where $v_0$ is tied weakly to $M_b$.
