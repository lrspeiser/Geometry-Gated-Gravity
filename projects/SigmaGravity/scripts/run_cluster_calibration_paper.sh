#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
CATALOG="$ROOT/data/clusters/master_catalog.csv"
python "$ROOT/scripts/run_mass_scaled_emcee.py" --catalog "$CATALOG" --tiers 1,2 --exclude 'MACS0717' --holdout 'A1689,MACS1149' --pzs lognormal --outdir "$ROOT/output/mass_scaled_emcee_paper"
