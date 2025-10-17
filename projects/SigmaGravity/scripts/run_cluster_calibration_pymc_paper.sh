#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
python "$ROOT/projects/SigmaGravity/scripts/run_cluster_calibration_pymc_paper.py" "$@"
