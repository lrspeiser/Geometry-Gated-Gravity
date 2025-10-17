#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
python "$ROOT/projects/SigmaGravity/scripts/validate_holdout_paper.py" "$@"
