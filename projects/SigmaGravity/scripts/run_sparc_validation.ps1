Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
python "$Root\many_path_model\validation_suite.py" --all
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
