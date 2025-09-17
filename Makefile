# Convenience Makefile to regenerate SPARC outputs

PY=python3
SCRIPT=src/shell_model/sparc_predict.py
DATA_DIR=data
PARQUET=$(DATA_DIR)/sparc_rotmod_ltg.parquet
MRT=$(DATA_DIR)/SPARC_Lelli2016c.mrt
MASTER=$(DATA_DIR)/Rotmod_LTG/MasterSheet_SPARC.csv

.PHONY: all run run-master run-mrt run-shell optimize-shell run-density-models build-parquet plot-overlays

all: run-master

run:
	$(PY) $(SCRIPT) --parquet $(PARQUET) --mass-csv $(DATA_DIR)/masses.csv --output-dir $(DATA_DIR)

run-master:
	$(PY) $(SCRIPT) --parquet $(PARQUET) --sparc-master-csv $(MASTER) --output-dir $(DATA_DIR)

run-mrt:
	$(PY) $(SCRIPT) --parquet $(PARQUET) --sparc-mrt $(MRT) --output-dir $(DATA_DIR)

run-shell:
	$(PY) $(SCRIPT) --parquet $(PARQUET) --sparc-master-csv $(MASTER) --model shell --shell-mass-exp -0.35 --shell-middle 2.0 --shell-max 5.0 --output-dir $(DATA_DIR)

optimize-shell:
	$(PY) src/shell_model/optimize_shell.py --parquet $(PARQUET) --master-csv $(MASTER) --out-base $(DATA_DIR)/opt_shell --refine

# Generate overlay figures from a model output directory
# Example:
#   make plot-overlays IN_DIR=$(DATA_DIR)/opt_shell/refine__me_-0.400__mid_2.50__max_4.00 \
#                      OUT_DIR=reports/figures/opt_shell_refine_-0.40_2.50_4.00
plot-overlays:
	$(PY) src/shell_model/plot_overlays.py --in-dir "$(IN_DIR)" --out-dir "$(OUT_DIR)"

# Run density-based model optimizer (no mass power-law)
run-density-models:
	$(PY) src/legacy/sparc_density_models.py --data-dir $(DATA_DIR) --output-dir $(DATA_DIR)/model_results --n-iter 60 --test-fraction 0.2

build-parquet:
	$(PY) src/legacy/build_sparc_parquet.py

# Install vendored rigor package (Option B)
install-rigor:
	$(PY) -m pip install -e rigor

# Hierarchical Bayesian fit on SPARC (outer-only)
fit-bayes:
	$(PY) -m rigor.fit_sparc \
	  --parquet data/sparc_rotmod_ltg.parquet \
	  --master  data/Rotmod_LTG/MasterSheet_SPARC.csv \
	  --xi shell_logistic_radius \
	  --outer sigma --sigma_th 10 \
	  --use_outer_only \
	  --outdir out/xi_shell_logistic_radius \
	  --platform cpu \
	  --warmup 1500 --samples 1500 --chains 4

# Joint SPARC + Milky Way (binned) fit
fit-bayes-joint:
	$(PY) -m rigor.fit_joint_mw \
	  --parquet data/sparc_rotmod_ltg.parquet \
	  --master  data/Rotmod_LTG/MasterSheet_SPARC.csv \
	  --mw_csv path/to/gaia_stars.csv \
	  --xi shell_logistic_radius \
	  --outer sigma --sigma_th 10 \
	  --use_outer_only \
	  --outdir out/joint_mw_sparc \
	  --platform cpu \
	  --warmup 1500 --samples 1500 --chains 4

# Baseline comparison (GR/MOND/Burkert)
compare-baselines:
	$(PY) -m rigor --parquet data/sparc_rotmod_ltg.parquet --master data/Rotmod_LTG/MasterSheet_SPARC.csv --out_json out/baselines_summary.json --outer

# Posterior overlays (68% bands) after a completed run
bayes-overlays:
	$(PY) -m rigor.plotting \
	  --post  out/xi_shell_logistic_radius/posterior_samples.npz \
	  --figdir out/xi_shell_logistic_radius/figs \
	  --parquet data/sparc_rotmod_ltg.parquet \
	  --master  data/Rotmod_LTG/MasterSheet_SPARC.csv \
	  --outer sigma --sigma_th 10
