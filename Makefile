# Convenience Makefile to regenerate SPARC outputs

PY=python3
SCRIPT=src/scripts/sparc_predict.py
DATA_DIR=data
PARQUET=$(DATA_DIR)/sparc_rotmod_ltg.parquet
MRT=$(DATA_DIR)/SPARC_Lelli2016c.mrt
MASTER=$(DATA_DIR)/Rotmod_LTG/MasterSheet_SPARC.csv

.PHONY: all run run-master run-mrt

all: run-master

run:
	$(PY) $(SCRIPT) --parquet $(PARQUET) --mass-csv $(DATA_DIR)/masses.csv --output-dir $(DATA_DIR)

run-master:
	$(PY) $(SCRIPT) --parquet $(PARQUET) --sparc-master-csv $(MASTER) --output-dir $(DATA_DIR)

run-mrt:
	$(PY) $(SCRIPT) --parquet $(PARQUET) --sparc-mrt $(MRT) --output-dir $(DATA_DIR)
