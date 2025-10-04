param(
    [string]$RootUrl = 'https://cdsarc.cds.unistra.fr/ftp/J/ApJ/821/',
    [string]$OutDir  = 'C:\Users\henry\dev\GravityCalculator\data\clash\umetsu821_116'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function List-Links([string]$url) {
  try {
    $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 60
    return @($r.Links | Where-Object href | ForEach-Object { $_.href })
  } catch {
    return @()
  }
}

# Discover candidate subdirectories under ApJ/821
$subdirs = @()
$top = List-Links -url $RootUrl
foreach ($h in $top) {
  if ($h -match '/$') {
    $abs = [System.Uri]::new([System.Uri]$RootUrl, $h).AbsoluteUri
    $subdirs += $abs
  }
}

$targets = @()
foreach ($dir in $subdirs) {
  $readmeUrl = [System.Uri]::new([System.Uri]$dir, 'ReadMe').AbsoluteUri
  try {
    $rm = Invoke-WebRequest -Uri $readmeUrl -UseBasicParsing -TimeoutSec 30
    $txt = $rm.Content
    if ($txt -match '(?i)Umetsu' -or $txt -match '(?i)CLASH') {
      $targets += $dir
    }
  } catch {}
}

if ($targets.Count -eq 0) {
  Write-Host "No Umetsu/CLASH tables found under $RootUrl"
  exit 0
}

foreach ($base in $targets | Sort-Object -Unique) {
  $links = List-Links -url $base
  $files = foreach ($h in $links) {
    $abs = if ($h -match '^https?://') { $h } else { [System.Uri]::new([System.Uri]$base, $h).AbsoluteUri }
    $name = Split-Path $abs -Leaf
    if ($name -match '^table.*\.dat$' -or $name -eq 'ReadMe') { $abs }
  }
  $files = $files | Sort-Object -Unique
  $outSub = Join-Path $OutDir ([System.Uri]$base).Segments[-2].Trim('/')
  New-Item -ItemType Directory -Force -Path $outSub | Out-Null
  foreach ($url in $files) {
    $name = Split-Path $url -Leaf
    $out = Join-Path $outSub $name
    try {
      Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing -TimeoutSec 600
      Write-Host ("Fetched {0} -> {1}" -f $name, $outSub)
    } catch {
      Write-Host ("Failed {0}: {1}" -f $name, $_)
    }
  }
}
