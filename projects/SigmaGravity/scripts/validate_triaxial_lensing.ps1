#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ROOT = Resolve-Path (Join-Path $ScriptPath "../../..")
python "$ROOT/scripts/validate_triaxial_lensing.py" $args