# gravity_learn — data-driven equation discovery for geometry-gated gravity (O2/O3)

Status: experimental but functional. This folder contains a self-contained pipeline to learn, test, and distill closed-form laws that map baryonic geometry to the extra gravitational response observed in galaxies (O2) and, next, lensing in clusters (O3). It never edits paper.md or code outside gravity_learn/.

Highlights so far
- Reusable scaffold (features, physics, models, train/eval, tests) under gravity_learn/.
- Physics and tests: an Abel projector (with midpoint rule fix) + NFW checks; unit tests pass.
- Symbolic discovery (PySR): multiple sweeps produce compact O2 gates centered on scaled radius x and surface density Σ̂, with curvature/non-local improvements.
- Overlay plots: observed vs baryons vs model velocities across a galaxy grid.
- Per-galaxy inference: exact-match fX_req and best family selection per system with optional per-galaxy SR.
- Global compact fits: ratio/exp families fit across SPARC with per-galaxy metrics and montages.
- Statistical error analysis: emergent patterns of where the law works vs. struggles (outer slopes matter; more points help).
- Neural net + distillation: feature-rich MLP learns fX/xi/gX and PySR distills back to readable formulas, confirming x, Σ̂ (and its smooths), curvature, and mild g_bar sensitivity.

Data & environment
- SPARC galaxies (rotation curves & decompositions): ../data/sparc_rotmod_ltg.parquet (+ MasterSheet if available).
- Other data (MW, clusters, lensing): will be wired for O3; place under ../data/…
- GPU (RTX 5090): optional CuPy acceleration; JAX optional for PINN (CPU fallback available).

Quickstart
1) Environment
   - conda create -n gravlearn python=3.11 -y && conda activate gravlearn
   - pip install -r gravity_learn/requirements.txt
   - (GPU) pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   - (GPU) pip install cupy-cuda12x
2) Sanity tests
   - pytest -q gravity_learn/tests
3) O2 SR smoke test
   - python -m gravity_learn.train.train_o2 --cfg gravity_learn/configs/o2_default.yaml --mode sr
4) O2 PINN smoke test (CPU ok)
   - python -m gravity_learn.train.train_o2 --cfg gravity_learn/configs/o2_default.yaml
5) Overlay plots (observed vs baryons vs formula)
   - python -m gravity_learn.eval.plot_rc_overlays --formula ratio --limit_galaxies 16

Directory map (major pieces)
- features/geometry.py
  - x=R/Rd (or fallback), Σ̂, ∂lnΣ/∂lnR, curvature kappa_rho; optional CuPy
- physics/poisson_abel.py
  - M(<r), g(r), Abel projection Σ(R), ΔΣ(R); midpoint fix near r→R; NFW analytic check
- models/pinn_o2.py, models/symbolic_srx.py
  - Minimal JAX gating (O2) + PySR wrapper (can reuse rigor.rigor.formula_search)
- train/train_o2.py
  - CLI for O2 using PINN or SR; artifacts under experiments/
- eval/
  - plot_rc_overlays.py — overlay grids + metrics
  - infer_per_galaxy.py — fit per-galaxy families & optional per-galaxy SR
  - global_fit_o2.py — fit compact families globally, dump per-galaxy metrics & montages
  - error_analysis.py — emergent patterns of good vs poor fits (correlation, ML importances)
- train/train_nn_o2.py
  - Feature-rich MLP across galaxies/radii (x, x^2,x^3, lnR, 1/R, g_bar, Σ, Σ̂, ∂lnΣ, non-local Σ̂ smooths, curvature), GroupKFold CV; permutation importance; PySR distillation back to formulas
- configs/
  - o2_default.yaml, o2_sr_sweep.yaml
- experiments/
  - sr/, pinn/, eval/, pg_infer/, nn_o2/ — artifacts tracked via local LFS rules

What we’ve accomplished (O2: galaxies)
1) Physics & tests
   - Implemented Abel projection with midpoint rule to control the r≈R singular kernel; NFW Σ(R) numeric vs analytic within tolerance; tests pass (2/2).
2) Symbolic regression (global sweeps)
   - Compact families consistently emerge:
     - fX ≈ k · x  (minimal, simple)
     - fX ≈ x^2 / (a − b Σ̂)  (ratio gate)
     - fX ≈ α x^2 [exp(Σ̂) + c]  (exp gate)
   - Correlations support these: xi_excess and gX grow with x, fall with Σ̂, and increase with |∇lnΣ|.
3) Overlay plots
   - Generated rc_overlays_{simple,ratio,exp}_L16.png and per-run metrics; exp/ratio reduce large outliers relative to the simple law.
4) Per-galaxy inference
   - infer_per_galaxy fits simple/ratio/exp families to each system + optional per-galaxy SR on fX_req; writes per-galaxy overlays and a summary CSV.
5) Global compact fit
   - global_fit_o2 minimizes global MSE across galaxies for ratio/exp families; current best under MSE is ratio with params ~ (a≈0.97, b≈0), i.e., fX≈x^2/a.
   - Note: MSE downweights Σ̂ damping; robust/stratified objectives likely surface stronger Σ̂ & curvature roles (see error analysis).
6) Statistical error analysis
   - error_analysis computes feature summaries and errors (mape/outer_mape/rmse) per galaxy, with correlations and optional ML importances.
   - First pass with global-MSE params shows error grows with outer slopes (slope_Vobs_outer, slope_Vbar_outer) and falls with n_points; Σ̂, |∇lnΣ| appear modest globally but stronger in subsets/NN.
7) NN + distillation
   - train_nn_o2 fits a feature-rich MLP on fX (and can target xi/gX), then distills back to compact formulas via PySR.
   - Distilled forms consistently highlight:
     - x (scaled radius) gating
     - Σ̂ (often through exp(Σ̂) or smoothed Σ̂ terms) as a damping gate
     - a mild g_bar dependence (sqrt(g_bar) / exp(c·sqrt(g_bar)))
     - gains from non-local Σ̂ smooths and curvature kappa_sigma

Artifacts to inspect
- Overlay plots & metrics
  - gravity_learn/experiments/eval/plots/rc_overlays_{simple,ratio,exp}_L16.png
  - gravity_learn/experiments/eval/plots/metrics_{simple,ratio,exp}_L16.csv
- Per-galaxy inference
  - gravity_learn/experiments/pg_infer/run_*/per_galaxy_summary.csv and overlay_*.png
- Global compact fit
  - gravity_learn/experiments/eval/global_fit/best_family.json
  - gravity_learn/experiments/eval/global_fit/montage_{ratio,exp}_*.png
  - gravity_learn/experiments/eval/global_fit/per_galaxy_metrics_{ratio,exp}_*.csv
- Error analysis
  - gravity_learn/experiments/eval/error_analysis/run_*/{features_and_errors.csv, spearman_*.csv, insights.json}
- NN + distillation
  - gravity_learn/experiments/nn_o2/run_*/{permutation_importance.csv, fold_metrics.json, predictions_by_radius.csv, pysr_distilled_equations.csv}

Emergent principle (O2)
- A geometry-gated effective mass fraction law fits the data:
  - fX increases with scaled radius x = R/Rd
  - fX decreases with surface density Σ̂ (strong damping in dense regions)
  - fX increases with outer steepness/curvature (|∇lnΣ|, kappa_sigma)
  - small local tie-down by g_bar improves difficult cases
- Practical compact families:
  - fX ≈ x^2 / (a − b Σ̂ − d |∇lnΣ|) − c
  - fX ≈ α x^2 [exp(Σ̂_smooth) + c + d |∇lnΣ| + ε √g_bar]
- We will finalize a ≤6-parameter law under a robust objective and publish it with per-type medians & overlays.

Latest robust global fit (updated)
- Objective: median APE (mape_median) across all points.
- Best family: ratio_curv (fX ≈ x^2 / (a − b Σ̂ − d |∇lnΣ|)).
- Params: a ≈ 0.669, b ≈ 0.140, d ≈ 0.087 (see gravity_learn/experiments/eval/global_fit/best_family.json).
- Summary (per-galaxy):
  - rmse_median ≈ 24.40 km/s (IQR ≈ 14.69–39.36)
  - mape_median ≈ 0.242 (IQR ≈ 0.135–0.358)
- Note: adding the curvature/outer-slope term improved the robust objective vs the Σ̂-only exp family; this matches NN/distillation signals that |∇lnΣ| (or curvature) helps adjust outer behavior automatically.

What’s next (prioritized)
1) Robust global fit & stratified validation
   - Fit (a,b,c,d,ε,…) for the ratio/exp templates with median APE or Huber loss; stratify by type/outer slope to prevent dominance by easy subsets.
   - Re-generate overlays, per-type medians, and summarize acceptance metrics.
2) Cross-target triangulation
   - Repeat NN+SR for xi_excess and gX; prefer a single compact family that is consistent across targets.
3) Lens ing (O3) pipeline
   - Ingest ΔΣ/κ profiles (clusters), include non-local Σ̂ smooths on larger scales (30–600 kpc), and run the same NN→SR path to discover the O3 gate.
   - Produce ΔΣ overlays and a compact O3 law; confirm neutrality in dense/strong-g regimes.
4) Universal law proposal (O1 fixed, O2+O3)
   - Freeze O2+O3 parameter sets; export a single function set with ≤10 global parameters.
   - Provide a small CLI to apply the law zero-shot to new systems.
5) Reporting & reproducibility
   - Collect tables & figures (overlays, medians/IQRs, Pareto complexity vs error, ablations) with exact commands and seeds.

Repro recipes (selected)
- Overlay grids
  - python -m gravity_learn.eval.plot_rc_overlays --formula ratio --limit_galaxies 16
- Per-galaxy inference (families + optional SR per galaxy)
  - python -m gravity_learn.eval.infer_per_galaxy --limit_galaxies 16 --do_sr
- Global compact fit (ratio/exp) + montages + metrics
  - python -m gravity_learn.eval.global_fit_o2 --limit_galaxies -1 --montage_limit 16
- Error analysis (emergent patterns)
  - python -m gravity_learn.eval.error_analysis --error_col mape
- NN discovery + distillation (fX target)
  - python -m gravity_learn.train.train_nn_o2 --target fX --limit_galaxies -1 --splits 5 --max_iter 350 --do_sr

Notes & policies
- LFS scope: gravity_learn/experiments/**/* is tracked by local LFS rules; keep heavy artifacts there.
- Safety: no edits outside gravity_learn; no paper.md changes.
- Windows tips: prefer python -m, avoid smart quotes & trailing backslashes; use fd/rg if installed.

Troubleshooting
- PySR/Julia bootstrap: first run may download Julia & packages; let it finish.
- CuPy/JAX wheels: match your CUDA 12 install; see their wheel indexes for Windows.
- Missing columns: loaders are defensive, but per-galaxy SR/NN can be sensitive to NaNs; median-impute or drop rows in-stage (we already impute/drop in SR wrappers).

Contact
- No API keys or web services needed. If we add any, we’ll document setup right here.
