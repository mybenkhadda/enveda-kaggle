<#
.SYNOPSIS
    Create the EnvedaCASMI persistent folder tree inside the Google Drive for desktop root. Directories only.

.DESCRIPTION
    Reads the `directories` list of configs/drive_asset_map.yaml and creates
        <DriveRoot>\EnvedaCASMI\<each directory>
    with New-Item -ItemType Directory -Force (which never deletes or overwrites anything; existing folders and
    files are left untouched). Idempotent and safe to re-run. Supports -WhatIf (prints, creates nothing).
    The Git repository is NOT placed on Drive (Colab clones it to /content/Enveda).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ".\scripts\create_colab_drive_architecture.ps1" -DriveRoot "G:\My Drive"
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ".\scripts\create_colab_drive_architecture.ps1" -DriveRoot "G:\My Drive" -WhatIf
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$DriveRoot,
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$MapPath = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\DriveAssetMap.ps1')

if (-not $MapPath) { $MapPath = Join-Path $RepoRoot 'configs\drive_asset_map.yaml' }
if (-not (Test-Path -LiteralPath $DriveRoot -PathType Container)) {
    Write-Host "STOPPED: DriveRoot '$DriveRoot' does not exist. Run scripts\find_google_drive.ps1 first." -ForegroundColor Red
    exit 1
}
$map = Read-AssetMap $MapPath
$target = Resolve-EnvedaTarget $DriveRoot $map['drive_root_name']
Write-Host "Drive project root: $target" -ForegroundColor Cyan
Write-Host "Directory list    : $MapPath ($($map['directories'].Count) folders)"
Write-Host ''

$created = 0; $existing = 0; $planned = 0
foreach ($rel in (@('') + @($map['directories']))) {
    $path = if ($rel) { Join-Path $target (ConvertTo-WinRel $rel) } else { $target }
    $label = if ($rel) { $rel } else { '(project root)' }
    if (Test-Path -LiteralPath $path -PathType Container) {
        Write-Host ("  EXISTS        {0}" -f $label) -ForegroundColor DarkGray
        $existing++
        continue
    }
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        Write-Host ("  CONFLICT      {0}  (a FILE with this name exists -- left untouched)" -f $label) -ForegroundColor Red
        continue
    }
    if ($PSCmdlet.ShouldProcess($path, 'Create directory')) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
        Write-Host ("  CREATED       {0}" -f $label) -ForegroundColor Green
        $created++
    } else {
        Write-Host ("  WOULD CREATE  {0}" -f $label) -ForegroundColor Yellow
        $planned++
    }
}
Write-Host ''
Write-Host ("Done. created={0} existing={1} would_create={2}. Nothing was deleted or overwritten." -f $created, $existing, $planned) -ForegroundColor Cyan
