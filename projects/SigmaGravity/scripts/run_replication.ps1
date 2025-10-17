#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ROOT = Resolve-Path (Join-Path $ScriptPath "../../..")
python "$ROOT/projects/SigmaGravity/scripts/run_replication.py" $args