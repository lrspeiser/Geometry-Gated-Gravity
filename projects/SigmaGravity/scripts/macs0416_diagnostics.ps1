Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
python "$Root\scripts\plot_macs0416_diagnostics.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
