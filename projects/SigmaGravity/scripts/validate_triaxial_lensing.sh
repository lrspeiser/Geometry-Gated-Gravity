#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
python "$ROOT/scripts/validate_triaxial_lensing.py" "$@"