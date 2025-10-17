#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
python "$ROOT/scripts/plot_macs0416_diagnostics.py"
