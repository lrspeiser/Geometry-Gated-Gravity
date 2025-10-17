Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..\..')
$Catalog = Join-Path $Root 'data\clusters\master_catalog.csv'
python "$Root\scripts\run_mass_scaled_emcee.py" --catalog "$Catalog" --tiers 1,2 --exclude 'MACS0717' --holdout 'A1689,MACS1149' --pzs lognormal --outdir "$Root\output\mass_scaled_emcee_paper"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
