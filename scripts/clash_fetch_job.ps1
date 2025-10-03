param(
    [Parameter(Mandatory=$true)][string]$Cluster,
    [string]$BaseUrl = 'https://archive.stsci.edu/missions/hlsp/clash',
    [string]$Root    = 'C:\Users\henry\dev\GravityCalculator\data\clash'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$hlsp   = Join-Path $Root 'hlsp'
$logs   = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $hlsp, $logs | Out-Null
$log    = Join-Path $logs ("$Cluster.log")
New-Item -ItemType File -Force -Path $log | Out-Null

function Log { param([string]$m) $ts = Get-Date -Format s; Add-Content -Path $log -Value "[$ts] $m" }

function Download-One {
    param([string]$url, [string]$outPath)
    try {
        $head = Invoke-WebRequest -Method Head -Uri $url -UseBasicParsing -TimeoutSec 30
    } catch {
        Log ("MISS (HEAD) " + $url); return $false
    }
    $dir = Split-Path -Parent $outPath
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $skip = $false
    if (Test-Path $outPath) {
        try {
            $len = $head.Headers['Content-Length']
            $loc = (Get-Item $outPath).Length
            if ($len -and [int64]$len -eq $loc) { Log ("SKIP (match) " + $url); $skip = $true }
        } catch {}
    }
    if (-not $skip) {
        Log ("GET " + $url)
        Invoke-WebRequest -Uri $url -OutFile $outPath -UseBasicParsing -TimeoutSec 3600
    }
    return $true
}

Log "BEGIN $Cluster"
$baseDir = "$BaseUrl/$Cluster/models/zitrin/nfw"
$variants = @('v2','v3','v1')

$okK = $false; $okG = $false; $okG1 = $false; $okG2 = $false
foreach ($v in $variants) {
    if (-not $okK) {
        $k1 = "$baseDir/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_v2_kappa.fits"
        $k2 = "$baseDir/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_${v}_kappa.fits"
        $out1 = Join-Path $hlsp "$Cluster/models/zitrin/nfw/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_v2_kappa.fits"
        $out2 = Join-Path $hlsp "$Cluster/models/zitrin/nfw/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_${v}_kappa.fits"
        $okK = (Download-One -url $k1 -outPath $out1) -or (Download-One -url $k2 -outPath $out2)
    }
    if (-not ($okG -or ($okG1 -and $okG2))) {
        $g1 = "$baseDir/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_v2_gamma.fits"
        $g2 = "$baseDir/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_${v}_gamma.fits"
        $o1 = Join-Path $hlsp "$Cluster/models/zitrin/nfw/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_v2_gamma.fits"
        $o2 = Join-Path $hlsp "$Cluster/models/zitrin/nfw/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_${v}_gamma.fits"
        $okG = (Download-One -url $g1 -outPath $o1) -or (Download-One -url $g2 -outPath $o2)
        if (-not $okG) {
            $g1c = "$baseDir/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_${v}_gamma1.fits"
            $g2c = "$baseDir/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_${v}_gamma2.fits"
            $o1c = Join-Path $hlsp "$Cluster/models/zitrin/nfw/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_${v}_gamma1.fits"
            $o2c = Join-Path $hlsp "$Cluster/models/zitrin/nfw/$v/hlsp_clash_model_${Cluster}_zitrin-nfw_${v}_gamma2.fits"
            $okG1 = Download-One -url $g1c -outPath $o1c
            $okG2 = Download-One -url $g2c -outPath $o2c
        }
    }
    if ($okK -and ($okG -or ($okG1 -and $okG2))) { break }
}

if ($okK -and ($okG -or ($okG1 -and $okG2))) { Log "DONE" } else { Log "DONE WITH MISSING FILES" }