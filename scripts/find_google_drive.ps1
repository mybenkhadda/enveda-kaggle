<#
.SYNOPSIS
    Locate the Google Drive for desktop "My Drive" root on this Windows machine. Read-only.

.DESCRIPTION
    Checks, for every mounted file-system drive letter and the user profile:
        <L>:\My Drive  (+ localized names: Mon Drive, Mi unidad, Meine Ablage, Il mio Drive, Meu Drive)
        %USERPROFILE%\My Drive, %USERPROFILE%\Google Drive, %USERPROFILE%\Google Drive\My Drive
    Never guesses silently:
        exactly one root  -> printed (exit 0)
        several roots     -> all printed; re-run the next scripts with an explicit -DriveRoot (exit 2)
        none              -> instructions (exit 1)
    With -DriveRoot it only validates that path.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ".\scripts\find_google_drive.ps1"
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ".\scripts\find_google_drive.ps1" -DriveRoot "G:\My Drive"
#>
[CmdletBinding()]
param(
    [string]$DriveRoot = ''
)
$ErrorActionPreference = 'Stop'
$ProjectFolder = 'EnvedaCASMI'

if ($DriveRoot) {
    if (Test-Path -LiteralPath $DriveRoot -PathType Container) {
        $full = [System.IO.Path]::GetFullPath($DriveRoot)
        $hasProject = Test-Path -LiteralPath (Join-Path $full $ProjectFolder)
        Write-Host "OK: $full (EnvedaCASMI folder present: $hasProject)" -ForegroundColor Green
        Write-Output $full
        exit 0
    }
    Write-Host "NOT FOUND: $DriveRoot is not an existing folder." -ForegroundColor Red
    exit 1
}

$names = @('My Drive', 'Mon Drive', 'Mi unidad', 'Meine Ablage', 'Il mio Drive', 'Meu Drive')
$candidates = New-Object System.Collections.ArrayList
foreach ($d in (Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue)) {
    if (-not $d.Root) { continue }
    foreach ($n in $names) { [void]$candidates.Add((Join-Path $d.Root $n)) }
}
$prof = $env:USERPROFILE
if ($prof) {
    foreach ($rel in @('My Drive', 'Mon Drive', 'Google Drive', 'Google Drive\My Drive', 'Google Drive\Mon Drive')) {
        [void]$candidates.Add((Join-Path $prof $rel))
    }
}

$found = New-Object System.Collections.ArrayList
foreach ($c in ($candidates | Select-Object -Unique)) {
    try {
        if (Test-Path -LiteralPath $c -PathType Container) {
            $full = [System.IO.Path]::GetFullPath($c)
            # "Google Drive" (profile folder) is only the root itself when it does not contain a "My Drive" child
            if ((Split-Path -Leaf $full) -eq 'Google Drive' -and ((Test-Path -LiteralPath (Join-Path $full 'My Drive')) -or (Test-Path -LiteralPath (Join-Path $full 'Mon Drive')))) { continue }
            if (-not ($found -contains $full)) { [void]$found.Add($full) }
        }
    } catch { }
}

Write-Host ''
if ($found.Count -eq 1) {
    $r = $found[0]
    Write-Host "Google Drive root found: $r" -ForegroundColor Green
    Write-Host ("EnvedaCASMI folder present: {0}" -f (Test-Path -LiteralPath (Join-Path $r $ProjectFolder)))
    Write-Host ''
    Write-Host 'Next:' -ForegroundColor Cyan
    Write-Host ("  powershell -ExecutionPolicy Bypass -File "".\scripts\create_colab_drive_architecture.ps1"" -DriveRoot ""{0}""" -f $r)
    Write-Output $r
    exit 0
}
if ($found.Count -gt 1) {
    Write-Host 'Several Google Drive roots were found -- choose one explicitly with -DriveRoot:' -ForegroundColor Yellow
    foreach ($r in $found) { Write-Host "  $r" }
    exit 2
}
Write-Host 'No Google Drive for desktop root was found.' -ForegroundColor Red
Write-Host '  1. Install / start Google Drive for desktop and sign in (it mounts a drive letter, often G:).'
Write-Host '  2. Or pass the synced folder explicitly to the next scripts:  -DriveRoot "X:\My Drive"'
exit 1
