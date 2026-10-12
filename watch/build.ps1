# Buduje pole danych dla wybranego modelu Fenixa 8.
# Użycie: .\watch\build.ps1 [-Device fenix847mm]
param([string]$Device = "fenix847mm")

$ErrorActionPreference = "Stop"
$sdk = (Get-Content "$env:APPDATA\Garmin\ConnectIQ\current-sdk.cfg" -Raw).Trim()
$key = Join-Path $PSScriptRoot "developer_key"
if (-not (Test-Path $key)) { throw "Brak $key - zobacz docs/GARMIN.md, krok 4." }

New-Item -ItemType Directory -Force (Join-Path $PSScriptRoot "bin") | Out-Null
& "$sdk\bin\monkeyc.bat" -d $Device -f (Join-Path $PSScriptRoot "monkey.jungle") `
    -o (Join-Path $PSScriptRoot "bin\HrForecast.prg") -y $key -w
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "OK: watch\bin\HrForecast.prg ($Device)"
