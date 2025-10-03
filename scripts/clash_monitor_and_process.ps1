param(
    [string[]]$Clusters = @('a1423','a209','a2261','a383','a611','clj1226','macs0329','macs0416','macs0429','macs0647','macs0717','macs0744','macs1115','macs1149','macs1206','macs1311','macs1423','macs1720','macs1931','macs2129','ms2137','rxj1347','rxj1532','rxj2129','rxj2248'),
    [int]$TimeoutMinutes = 360,
    [int]$PollSeconds = 30,
    [string]$Root = 'C:\Users\henry\dev\GravityCalculator\data\clash'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$logsDir = Join-Path $Root 'logs'
$hlsp    = Join-Path $Root 'hlsp'
$procDir = Join-Path $Root 'processed'
New-Item -ItemType Directory -Force -Path $logsDir, $hlsp, $procDir | Out-Null

$status = @{}
foreach ($c in $Clusters) { $status[$c] = 'pending' }

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)

function Is-Done([string]$cluster) {
    $log = Join-Path $logsDir ("$cluster.log")
    if (-not (Test-Path $log)) { return $false }
    try {
        $tail = Get-Content -Path $log -Tail 200 -ErrorAction Stop
    } catch { return $false }
    if ($tail -match 'DONE WITH MISSING FILES') { $script:status[$cluster] = 'done_missing'; return $true }
    if ($tail -match 'DONE') { $script:status[$cluster] = 'done'; return $true }
    return $false
}

"[clash-monitor] Start: $((Get-Date).ToString('s'))" | Out-File -FilePath (Join-Path $Root 'status_report.txt') -Encoding utf8

while ($true) {
    $completed = 0
    foreach ($c in $Clusters) {
        if ($status[$c] -like 'done*') { $completed++ ; continue }
        if (Is-Done $c) { $completed++ }
    }
    $line = "[status] $completed / $($Clusters.Count) complete @ $((Get-Date).ToString('s'))"
    Add-Content -Path (Join-Path $Root 'status_report.txt') -Value $line
    if ($completed -ge $Clusters.Count) { break }
    if (Get-Date > $deadline) { Add-Content -Path (Join-Path $Root 'status_report.txt') -Value '[clash-monitor] TIMEOUT reached'; break }
    Start-Sleep -Seconds $PollSeconds
}

# Run processing script
$scriptPy = 'C:\Users\henry\dev\GravityCalculator\concepts\cluster_lensing\process_clash_models.py'
$python = 'python'
try {
    $env:PYTHONUNBUFFERED = '1'
    & $python $scriptPy
    Add-Content -Path (Join-Path $Root 'status_report.txt') -Value '[clash-monitor] Processing script completed'
} catch {
    Add-Content -Path (Join-Path $Root 'status_report.txt') -Value ("[clash-monitor] Processing FAILED: " + $_)
}

# Create deterministic train/test split for clusters with maps
$withMaps = @()
if (Test-Path $hlsp) {
    $dirs = Get-ChildItem -Path $hlsp -Directory -ErrorAction SilentlyContinue
    foreach ($d in $dirs) {
        $models = Join-Path $d.FullName 'models'
        if (Test-Path $models) {
            $k = Get-ChildItem -Path $models -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'kappa\.fits$' }
            if ($k) { $withMaps += $d.Name }
        }
    }
}
$withMaps = $withMaps | Sort-Object -Unique
$splitPath = Join-Path $Root 'train_test_split.csv'
"cluster_id,set" | Out-File -FilePath $splitPath -Encoding utf8
$trainCount = [math]::Ceiling($withMaps.Count / 2)
$train = $withMaps | Select-Object -First $trainCount
$test  = $withMaps | Select-Object -Skip $trainCount
foreach ($c in $train) { "$c,train" | Out-File -FilePath $splitPath -Append -Encoding utf8 }
foreach ($c in $test)  { "$c,test"  | Out-File -FilePath $splitPath -Append -Encoding utf8 }
Add-Content -Path (Join-Path $Root 'status_report.txt') -Value ("[clash-monitor] Wrote split for " + $withMaps.Count + " clusters")

# Append per-cluster statuses
Add-Content -Path (Join-Path $Root 'status_report.txt') -Value '[clash-monitor] Cluster statuses:'
foreach ($c in $Clusters) { Add-Content -Path (Join-Path $Root 'status_report.txt') -Value ("  - " + $c + ": " + $status[$c]) }