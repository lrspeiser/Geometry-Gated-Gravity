#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../.. && pwd)"
python "$ROOT/scripts/simple_einstein_check.py" "$@"