param(
    [string[]]$Clusters = @('a1423','a209','a2261','a383','a611','clj1226','macs0329','macs0416','macs0429','macs0647','macs0717','macs0744','macs1115','macs1149','macs1206','macs1311','macs1423','macs1720','macs1931','macs2129','ms2137','rxj1347','rxj1532','rxj2129','rxj2248'),
    [string]$Root     = 'C:\Users\henry\dev\GravityCalculator\data\clash'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$scriptPath = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'clash_fetch_job.ps1'
if (-not (Test-Path $scriptPath)) { throw "Missing clash_fetch_job.ps1 at $scriptPath" }

# Start a detached powershell.exe per cluster (non-blocking)
foreach ($c in $Clusters) {
    $args = "-NoLogo -NoProfile -File `"$scriptPath`" -Cluster $c"
    Start-Process -FilePath powershell.exe -ArgumentList $args -WindowStyle Hidden | Out-Null
}

Write-Host "Started background fetchers for $($Clusters.Count) clusters. Logs: $Root\logs"