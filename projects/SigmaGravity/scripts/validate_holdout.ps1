Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
$Posterior = Join-Path $Root 'output\mass_scaled_emcee_paper\flat_samples.npz'
$Catalog = Join-Path $Root 'data\clusters\master_catalog.csv'
python "$Root\scripts\validate_holdout_mass_scaled.py" --posterior "$Posterior" --catalog "$Catalog" --clusters 'A1689,MACS1149' --pzs lognormal
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
