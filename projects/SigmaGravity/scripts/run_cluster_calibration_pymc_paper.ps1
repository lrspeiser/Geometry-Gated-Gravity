#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ROOT = Resolve-Path (Join-Path $ScriptPath "../../..")
python "$ROOT/projects/SigmaGravity/scripts/run_cluster_calibration_pymc_paper.py" $args