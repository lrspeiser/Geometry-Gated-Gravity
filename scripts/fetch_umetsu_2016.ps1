param(
    [string]$BaseUrl = 'https://cdsarc.cds.unistra.fr/ftp/J/ApJ/821/116/',
    [string]$OutDir  = 'C:\Users\henry\dev\GravityCalculator\data\clash\umetsu821_116'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# List directory and filter for table*.dat and ReadMe
try {
  $resp = Invoke-WebRequest -Uri $BaseUrl -UseBasicParsing -TimeoutSec 60
} catch {
  Write-Host "Failed to list $BaseUrl: $_"; exit 1
}

$links = @($resp.Links | Where-Object href | ForEach-Object { $_.href })
$files = foreach ($h in $links) {
  if ($h -match '^https?://') { $abs = $h } else { $abs = [System.Uri]::new([System.Uri]$BaseUrl, $h).AbsoluteUri }
  $name = Split-Path $abs -Leaf
  if ($name -match '^table.*\.dat$' -or $name -eq 'ReadMe') { $abs }
}
$files = $files | Sort-Object -Unique

foreach ($url in $files) {
  $name = Split-Path $url -Leaf
  $out = Join-Path $OutDir $name
  try {
    Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing -TimeoutSec 600
    Write-Host "Fetched $name"
  } catch {
    Write-Host "Failed $name: $_"
  }
}