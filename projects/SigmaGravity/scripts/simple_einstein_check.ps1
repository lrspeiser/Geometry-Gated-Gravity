#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ROOT = Resolve-Path (Join-Path $ScriptPath "../../..")
python "$ROOT/scripts/simple_einstein_check.py" $args