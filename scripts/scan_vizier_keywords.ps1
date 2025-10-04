param(
    [string[]]$Journals = @('J/ApJ/','J/ApJS/','J/MNRAS/','J/A+A/'),
    [string[]]$Keywords = @('Umetsu','CLASH','ACCEPT','weak-lensing','strong lensing','Chandra','X-ray'),
    [string]$OutDir = 'C:\Users\henry\dev\GravityCalculator\data\clash\vizier_scan'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$base = 'https://cdsarc.cds.unistra.fr/ftp/'

function List-Links([string]$url) {
  try {
    $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 60
    return @($r.Links | Where-Object href | ForEach-Object { $_.href })
  } catch {
    return @()
  }
}

$result = @()

foreach ($j in $Journals) {
  $root = [System.Uri]::new([System.Uri]$base, $j).AbsoluteUri
  $vols = List-Links -url $root
  foreach ($v in $vols) {
    if ($v -notmatch '/$') { continue }
    $volUrl = [System.Uri]::new([System.Uri]$root, $v).AbsoluteUri
    # Check volume ReadMe
    $volReadme = [System.Uri]::new([System.Uri]$volUrl, 'ReadMe').AbsoluteUri
    $hitVol = $false
    try {
      $rm = Invoke-WebRequest -Uri $volReadme -UseBasicParsing -TimeoutSec 20
      $txt = $rm.Content
      foreach ($kw in $Keywords) { if ($txt -match [regex]::Escape($kw)) { $hitVol = $true; break } }
      if ($hitVol) {
        $result += [pscustomobject]@{ level='volume'; path=$volUrl; readme=$volReadme }
      }
    } catch {}
    # Check subpages
    $pages = List-Links -url $volUrl
    foreach ($p in $pages) {
      if ($p -notmatch '/$') { continue }
      $pageUrl = [System.Uri]::new([System.Uri]$volUrl, $p).AbsoluteUri
      $pageReadme = [System.Uri]::new([System.Uri]$pageUrl, 'ReadMe').AbsoluteUri
      $hit = $false
      try {
        $pm = Invoke-WebRequest -Uri $pageReadme -UseBasicParsing -TimeoutSec 20
        $ptxt = $pm.Content
        foreach ($kw in $Keywords) { if ($ptxt -match [regex]::Escape($kw)) { $hit = $true; break } }
        if ($hit) {
          # list candidate data files
          $plinks = List-Links -url $pageUrl
          $files = @()
          foreach ($h in $plinks) {
            $abs = if ($h -match '^https?://') { $h } else { [System.Uri]::new([System.Uri]$pageUrl, $h).AbsoluteUri }
            $name = Split-Path $abs -Leaf
            if ($name -match '^table.*\.dat$' -or $name -eq 'ReadMe') { $files += $abs }
          }
          $result += [pscustomobject]@{ level='page'; path=$pageUrl; readme=$pageReadme; files=$files -join ';' }
        }
      } catch {}
    }
  }
}

# Write results JSON
$outJson = Join-Path $OutDir 'scan_results.json'
$result | ConvertTo-Json -Depth 4 | Out-File -FilePath $outJson -Encoding utf8
Write-Host ("Scan complete. Results: {0}" -f $outJson)