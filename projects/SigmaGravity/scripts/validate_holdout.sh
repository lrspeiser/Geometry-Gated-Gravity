#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
POST="$ROOT/output/mass_scaled_emcee_paper/flat_samples.npz"
CATALOG="$ROOT/data/clusters/master_catalog.csv"
python "$ROOT/scripts/validate_holdout_mass_scaled.py" --posterior "$POST" --catalog "$CATALOG" --clusters 'A1689,MACS1149' --pzs lognormal
