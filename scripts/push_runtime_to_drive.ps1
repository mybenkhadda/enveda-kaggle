<#
.SYNOPSIS
    Upload the frozen CASMI runtime package to Google Drive through the Drive API (OAuth) -- wrapper for
    scripts\push_runtime_to_drive.py running in the casmi2026 conda environment.

.DESCRIPTION
    Uploads bundle\ (recursive), data\test.parquet and data\sample_submission.csv to My Drive\EnvedaCASMI\
    {bundle, competition}; creates results\. The OAuth client JSON is passed by PATH only -- this wrapper never reads
    or prints it. The token is cached outside the repository (%USERPROFILE%\.enveda\google_drive_token.json).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\push_runtime_to_drive.ps1 -Credentials "C:\Users\myben\secrets\google_drive_client_secret.json" -DryRun
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot = "C:\Users\myben\OneDrive\Documents\Enveda",
    [Parameter(Mandatory=$true)]
    [string]$Credentials,
    [switch]$DryRun,
    [switch]$Force,
    [string]$Token = (Join-Path $env:USERPROFILE '.enveda\google_drive_token.json'),
    [string]$DriveFolder = 'EnvedaCASMI',
    [int]$ChunkSizeMB = 32,
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'

function Fail([string]$Message) {
    Write-Host ''
    Write-Host "UPLOAD STOPPED: $Message" -ForegroundColor Red
    exit 1
}

# --- casmi2026 interpreter -------------------------------------------------------------------------------------------
if (-not $Python) {
    $candidates = @()
    if ($env:CONDA_PREFIX -and ((Split-Path -Leaf $env:CONDA_PREFIX) -eq 'casmi2026')) { $candidates += (Join-Path $env:CONDA_PREFIX 'python.exe') }
    $candidates += @(
        (Join-Path $env:USERPROFILE '.conda\envs\casmi2026\python.exe'),
        (Join-Path $env:USERPROFILE 'anaconda3\envs\casmi2026\python.exe'),
        (Join-Path $env:USERPROFILE 'miniconda3\envs\casmi2026\python.exe'),
        'C:\ProgramData\anaconda3\envs\casmi2026\python.exe',
        'C:\ProgramData\miniconda3\envs\casmi2026\python.exe'
    )
    $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $Python) { Fail ("casmi2026 python.exe not found. Tried:`n  " + ($candidates -join "`n  ") + "`nPass -Python <path>.") }
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { Fail "Python not found: $Python" }

# --- paths only; the credential contents are never read here ---------------------------------------------------------
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$Credentials = [System.IO.Path]::GetFullPath($Credentials)
if (-not (Test-Path -LiteralPath $Credentials -PathType Leaf)) { Fail "OAuth client file not found: $Credentials" }
$script = Join-Path $ProjectRoot 'scripts\push_runtime_to_drive.py'
if (-not (Test-Path -LiteralPath $script -PathType Leaf)) { Fail "uploader not found: $script" }

Write-Host "python:      $Python"
Write-Host "project:     $ProjectRoot"
Write-Host "credentials: $Credentials   (path only)"

$pyArgs = @($script, '--project-root', $ProjectRoot, '--credentials', $Credentials, '--token', $Token,
            '--drive-folder', $DriveFolder, '--chunk-size-mb', "$ChunkSizeMB")
if ($DryRun) { $pyArgs += '--dry-run' }
if ($Force) { $pyArgs += '--force' }

& $Python @pyArgs
exit $LASTEXITCODE
