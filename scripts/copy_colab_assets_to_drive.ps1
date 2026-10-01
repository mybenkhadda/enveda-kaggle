<#
.SYNOPSIS
    COPY (never move) the local research assets listed in configs/drive_asset_map.yaml into the Google Drive
    for desktop folder <DriveRoot>\EnvedaCASMI, verify them by relative name + size, write a copy manifest.

.DESCRIPTION
    1. resolve RepoRoot / DriveRoot           2. load configs/drive_asset_map.yaml
    3. verify every REQUIRED local source (stop before copying anything if one is missing)
    4. frozen bundle identity (v2-A7 / 60174e39a2a3c6b4 / V1_TL_1K_TESTSIM_STRICT) at the source and, if a bundle
       already exists on Drive, at the destination (a DIFFERENT bundle there is never mixed / overwritten)
    5. print the planned operations           6. -WhatIf: stop here (nothing is created or copied)
    7. ask for confirmation unless -Yes       8. create destination folders (directories only)
    9. copy: directories with robocopy /E /Z /R:2 /W:5 (+ /XC /XN /XO unless -Overwrite, i.e. existing
       destination files are never replaced), files with Copy-Item; NEVER /MIR, /PURGE, /MOV, /MOVE, never a
       source deletion
   10. verify every copied file by relative path + size (no hashing of multi-GB files)
   11. write <DriveRoot>\EnvedaCASMI\exports\manifests\local_to_drive_copy_manifest.json (+ a timestamped copy)

    Statuses: COPIED, ALREADY_EXISTS, SKIPPED, MISSING_OPTIONAL, ERROR.
    Categories: bundle (-SkipBundle), external (-SkipExternal; missing = NOT PROVIDED YET, never an error),
                raw (only with -IncludeRaw), processed, metadata.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ".\scripts\copy_colab_assets_to_drive.ps1" -RepoRoot "." -DriveRoot "G:\My Drive" -WhatIf
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File ".\scripts\copy_colab_assets_to_drive.ps1" -RepoRoot "." -DriveRoot "G:\My Drive" -Yes
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$RepoRoot = '.',
    [Parameter(Mandatory = $true)][string]$DriveRoot,
    [string]$MapPath = '',
    [switch]$Yes,
    [switch]$SkipBundle,
    [switch]$SkipExternal,
    [switch]$IncludeRaw,
    [switch]$Overwrite
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\DriveAssetMap.ps1')

$EXPECTED_BUNDLE = [ordered]@{ bundle_version = 'v2-A7'; CONFIG_HASH = '60174e39a2a3c6b4'; model_id = 'V1_TL_1K_TESTSIM_STRICT' }
$PLAN_ONLY = [bool]$WhatIfPreference

function Stop-Migration([string]$Message) {
    Write-Host ''
    Write-Host "MIGRATION STOPPED: $Message" -ForegroundColor Red
    Write-Host 'Nothing was copied.' -ForegroundColor Red
    exit 1
}
function Step([string]$Message) { Write-Host ''; Write-Host "== $Message" -ForegroundColor Cyan }

function Test-Excluded([string]$Rel, $ExcludeDirs, $ExcludeFiles) {
    $parts = $Rel.Split('\')
    for ($i = 0; $i -lt $parts.Length - 1; $i++) { if ($ExcludeDirs -contains $parts[$i]) { return $true } }
    foreach ($pat in $ExcludeFiles) { if ($parts[-1] -like $pat) { return $true } }
    return $false
}

function Get-DirectoryInventory([string]$Root, $ExcludeDirs, $ExcludeFiles) {
    # relative path -> size, transient content excluded (never hashed)
    $inv = [ordered]@{}
    $base = $Root.TrimEnd('\').Length + 1
    foreach ($f in (Get-ChildItem -LiteralPath $Root -Recurse -File -Force -ErrorAction Stop)) {
        $rel = $f.FullName.Substring($base)
        if (-not (Test-Excluded $rel $ExcludeDirs $ExcludeFiles)) { $inv[$rel] = [int64]$f.Length }
    }
    return $inv
}

function Read-BundleIdentity([string]$BundleDir) {
    $cfg = Join-Path $BundleDir 'config.json'
    $mi = Join-Path $BundleDir 'models\model_info.json'
    if (-not (Test-Path -LiteralPath $cfg) -or -not (Test-Path -LiteralPath $mi)) { return $null }
    $c = Get-Content -Raw -LiteralPath $cfg -Encoding UTF8 | ConvertFrom-Json
    $m = Get-Content -Raw -LiteralPath $mi -Encoding UTF8 | ConvertFrom-Json
    return [ordered]@{ bundle_version = $c.bundle_version; CONFIG_HASH = $c.CONFIG_HASH; model_id = $c.model_id; model_info_model_id = $m.model_id }
}

function Test-BundleIdentity($Id) {
    if ($null -eq $Id) { return 'config.json or models\model_info.json missing' }
    $bad = @()
    foreach ($k in $EXPECTED_BUNDLE.Keys) { if ($Id[$k] -ne $EXPECTED_BUNDLE[$k]) { $bad += "$k=$($Id[$k]) (expected $($EXPECTED_BUNDLE[$k]))" } }
    if ($Id['model_info_model_id'] -ne $EXPECTED_BUNDLE['model_id']) { $bad += "models\model_info.json model_id=$($Id['model_info_model_id'])" }
    if ($bad.Count) { return ($bad -join '; ') }
    return ''
}

# -----------------------------------------------------------------------------------------------
Step 'Resolve paths'
try { $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).ProviderPath } catch { Stop-Migration "RepoRoot '$RepoRoot' not found" }
if (-not (Test-Path -LiteralPath $DriveRoot -PathType Container)) { Stop-Migration "DriveRoot '$DriveRoot' does not exist (run scripts\find_google_drive.ps1)" }
if (-not $MapPath) { $MapPath = Join-Path $RepoRoot 'configs\drive_asset_map.yaml' }
$map = Read-AssetMap $MapPath
$Target = Resolve-EnvedaTarget $DriveRoot $map['drive_root_name']
if ($Target.StartsWith($RepoRoot.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) { Stop-Migration "the Drive target $Target lies inside the repository" }
Write-Host "RepoRoot : $RepoRoot"
Write-Host "Target   : $Target"
Write-Host "Asset map: $MapPath"
Write-Host ("Options  : WhatIf={0} Yes={1} SkipBundle={2} SkipExternal={3} IncludeRaw={4} Overwrite={5}" -f $PLAN_ONLY, [bool]$Yes, [bool]$SkipBundle, [bool]$SkipExternal, [bool]$IncludeRaw, [bool]$Overwrite)
if (-not (Test-Path -LiteralPath $Target -PathType Container)) {
    Write-Host "NOTE: $Target does not exist yet -- run scripts\create_colab_drive_architecture.ps1 first (folders are also created on demand)." -ForegroundColor Yellow
}

# -----------------------------------------------------------------------------------------------
Step 'Inspect sources and destinations (no copying)'
$plan = New-Object System.Collections.ArrayList
$missingRequired = @()
foreach ($name in $map['assets'].Keys) {
    $a = $map['assets'][$name]
    $src = Join-Path $RepoRoot (ConvertTo-WinRel $a['source'])
    $dst = Join-Path $Target (ConvertTo-WinRel $a['destination'])
    $exD = @(if ($a.Contains('exclude_dirs') -and $a['exclude_dirs']) { $a['exclude_dirs'] })
    $exF = @(if ($a.Contains('exclude_files') -and $a['exclude_files']) { $a['exclude_files'] })
    $row = [pscustomobject][ordered]@{ name = $name; category = $a['category']; type = $a['mode']; required = [bool]$a['required']; source = $src; destination = $dst
                       source_size = $null; destination_size = $null; n_files = $null; status = 'PLANNED'; message = ''; exclude_dirs = $exD; exclude_files = $exF; inventory = $null }
    $skip = ''
    if ($a['category'] -eq 'bundle' -and $SkipBundle) { $skip = 'skipped by -SkipBundle' }
    elseif ($a['category'] -eq 'external' -and $SkipExternal) { $skip = 'skipped by -SkipExternal' }
    elseif ($a['category'] -eq 'raw' -and -not $IncludeRaw) { $skip = 'raw competition file: copied only with -IncludeRaw' }
    $exists = if ($a['mode'] -eq 'file') { Test-Path -LiteralPath $src -PathType Leaf } else { Test-Path -LiteralPath $src -PathType Container }
    if ($skip) { $row.status = 'SKIPPED'; $row.message = $skip }
    elseif (-not $exists) {
        if ($a['required']) { $missingRequired += "$name -> $src"; $row.status = 'ERROR'; $row.message = 'required source missing' }
        else { $row.status = 'MISSING_OPTIONAL'; $row.message = $(if ($a['category'] -eq 'external') { 'NOT PROVIDED YET' } else { 'optional source not present' }) }
    }
    elseif ($a['mode'] -eq 'file') {
        $row.source_size = [int64](Get-Item -LiteralPath $src).Length
        $row.n_files = 1
        if (Test-Path -LiteralPath $dst -PathType Leaf) { $row.destination_size = [int64](Get-Item -LiteralPath $dst).Length }
    }
    else {
        $row.inventory = Get-DirectoryInventory $src $exD $exF
        $row.n_files = $row.inventory.Count
        $row.source_size = [int64](($row.inventory.Values | Measure-Object -Sum).Sum)
        if (Test-Path -LiteralPath $dst -PathType Container) {
            $sum = 0L
            foreach ($rel in $row.inventory.Keys) { $p = Join-Path $dst $rel; if (Test-Path -LiteralPath $p -PathType Leaf) { $sum += (Get-Item -LiteralPath $p).Length } }
            $row.destination_size = $sum
        }
    }
    [void]$plan.Add($row)
}
if ($missingRequired.Count) { Stop-Migration ("required local sources missing:`n  " + ($missingRequired -join "`n  ")) }

# frozen bundle identity
$bundleRow = $plan | Where-Object { $_.category -eq 'bundle' -and $_.status -eq 'PLANNED' } | Select-Object -First 1
if ($bundleRow) {
    $why = Test-BundleIdentity (Read-BundleIdentity $bundleRow.source)
    if ($why) { Stop-Migration "local bundle is not the frozen v2-A7 bundle: $why" }
    Write-Host '  local bundle identity OK (v2-A7 / 60174e39a2a3c6b4 / V1_TL_1K_TESTSIM_STRICT)' -ForegroundColor Green
    if (Test-Path -LiteralPath (Join-Path $bundleRow.destination 'config.json')) {
        $whyDst = Test-BundleIdentity (Read-BundleIdentity $bundleRow.destination)
        if ($whyDst) {
            $bundleRow.status = 'ERROR'
            $bundleRow.message = "a DIFFERENT bundle already exists on Drive ($whyDst); rename/remove it manually -- bundles are never mixed"
        } else { Write-Host '  Drive already holds the same frozen bundle identity (missing files will be completed)' -ForegroundColor Green }
    } elseif ((Test-Path -LiteralPath $bundleRow.destination -PathType Container) -and (Get-ChildItem -LiteralPath $bundleRow.destination -Force | Select-Object -First 1)) {
        $bundleRow.status = 'ERROR'
        $bundleRow.message = 'non-empty Drive bundle folder without config.json; inspect it manually -- nothing copied into it'
    }
}

# -----------------------------------------------------------------------------------------------
Step 'Planned operations'
foreach ($r in $plan) {
    if ($r.status -ne 'PLANNED') { $state = $r.status }
    elseif ($null -eq $r.destination_size) { $state = 'COPY (absent on Drive)' }
    elseif ($r.destination_size -eq $r.source_size) { $state = 'present on Drive, same size' }
    elseif ($Overwrite) { $state = 'COPY (-Overwrite: replace differing files)' }
    else { $state = 'COPY MISSING FILES ONLY (existing files kept)' }
    $size = if ($null -ne $r.source_size) { Format-Bytes $r.source_size } else { '-' }
    $nf = if ($r.n_files) { "$($r.n_files) files" } else { '' }
    $col = switch ($r.status) { 'ERROR' { 'Red' } 'MISSING_OPTIONAL' { 'Yellow' } 'SKIPPED' { 'DarkGray' } default { 'White' } }
    Write-Host ("  {0,-30} {1,-9} {2,-10} {3,-12} {4}" -f $r.name, $r.category, $size, $nf, $state) -ForegroundColor $col
    if ($r.message) { Write-Host ("      {0}" -f $r.message) -ForegroundColor $col }
    if ($r.status -eq 'PLANNED') { Write-Host ("      {0}  ->  {1}" -f $r.source, $r.destination) -ForegroundColor DarkGray }
}
$todo = @($plan | Where-Object { $_.status -eq 'PLANNED' })
$bytes = ($todo | ForEach-Object { if ($_.source_size) { $_.source_size } else { 0 } } | Measure-Object -Sum).Sum
Write-Host ''
Write-Host ("{0} assets to process, up to {1} (existing identical files are not re-copied)." -f $todo.Count, (Format-Bytes $bytes))

if ($PLAN_ONLY) {
    Write-Host ''
    Write-Host 'WhatIf: plan only. Nothing was created, copied or written.' -ForegroundColor Yellow
    exit 0
}
if (-not $Yes) {
    $answer = Read-Host 'Type YES to COPY these assets to Google Drive (sources are never deleted)'
    if ($answer -ne 'YES') { Write-Host 'Cancelled. Nothing was copied.' -ForegroundColor Yellow; exit 0 }
}

# -----------------------------------------------------------------------------------------------
Step 'Copy'
foreach ($r in $todo) {
    try {
        $parent = if ($r.type -eq 'file') { Split-Path -Parent $r.destination } else { $r.destination }
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        if ($r.type -eq 'file') {
            if ($null -ne $r.destination_size -and -not $Overwrite) {
                if ($r.destination_size -eq $r.source_size) { $r.status = 'ALREADY_EXISTS'; $r.message = 'same size on Drive' }
                else { $r.status = 'SKIPPED'; $r.message = "Drive file differs in size ($($r.destination_size) vs $($r.source_size)); re-run with -Overwrite to replace it" }
            } else {
                Copy-Item -LiteralPath $r.source -Destination $r.destination -Force
                $r.destination_size = [int64](Get-Item -LiteralPath $r.destination).Length
                if ($r.destination_size -eq $r.source_size) { $r.status = 'COPIED' } else { $r.status = 'ERROR'; $r.message = "size mismatch after copy ($($r.destination_size) vs $($r.source_size))" }
            }
        } else {
            # /E /Z /R:2 /W:5 only -- never /MIR, /PURGE, /MOV or /MOVE (nothing is ever deleted on either side)
            $rcArgs = @($r.source, $r.destination, '/E', '/Z', '/R:2', '/W:5', '/NP', '/NDL', '/NFL')
            if (-not $Overwrite) { $rcArgs += @('/XC', '/XN', '/XO') }         # never replace an existing destination file
            if ($r.exclude_dirs.Count) { $rcArgs += '/XD'; $rcArgs += $r.exclude_dirs }
            if ($r.exclude_files.Count) { $rcArgs += '/XF'; $rcArgs += $r.exclude_files }
            Write-Host ("  robocopy {0}" -f ($rcArgs -join ' ')) -ForegroundColor DarkGray
            & robocopy.exe @rcArgs | Out-Host
            $rc = $LASTEXITCODE
            if ($rc -ge 8) { throw "robocopy failed with exit code $rc" }
            # verification: every source file present on Drive with the same size
            $bad = @(); $sum = 0L
            foreach ($rel in $r.inventory.Keys) {
                $p = Join-Path $r.destination $rel
                if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { $bad += "missing: $rel"; continue }
                $len = [int64](Get-Item -LiteralPath $p).Length
                $sum += $len
                if ($len -ne $r.inventory[$rel]) { $bad += "size: $rel ($len vs $($r.inventory[$rel]))" }
            }
            $r.destination_size = $sum
            if ($bad.Count) {
                $r.status = 'ERROR'
                $hint = if (-not $Overwrite) { ' (existing differing files are kept without -Overwrite)' } else { '' }
                $r.message = "$($bad.Count) files failed verification$hint, e.g. " + (($bad | Select-Object -First 5) -join '; ')
            } elseif (($rc -band 1) -eq 1) { $r.status = 'COPIED' } else { $r.status = 'ALREADY_EXISTS'; $r.message = 'all files already on Drive with the same size' }
            if ($r.category -eq 'bundle' -and $r.status -ne 'ERROR') {
                $why = Test-BundleIdentity (Read-BundleIdentity $r.destination)
                if ($why) { $r.status = 'ERROR'; $r.message = "bundle identity on Drive: $why" }
            }
        }
    } catch {
        $r.status = 'ERROR'; $r.message = $_.Exception.Message
    }
    $col = if ($r.status -eq 'ERROR') { 'Red' } else { 'Green' }
    Write-Host ("  {0,-30} {1}  {2}" -f $r.name, $r.status, $r.message) -ForegroundColor $col
}

# -----------------------------------------------------------------------------------------------
Step 'Manifest'
$manifestDir = Join-Path $Target 'exports\manifests'
New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
$records = foreach ($r in $plan) {
    [ordered]@{ asset = $r.name; category = $r.category; type = $r.type; required = $r.required; source_path = $r.source; destination_path = $r.destination
                source_size = $r.source_size; destination_size = $r.destination_size; n_files = $r.n_files; status = $r.status; message = $r.message }
}
$manifest = [ordered]@{
    migration_time = (Get-Date).ToUniversalTime().ToString('o'); repo_root = $RepoRoot; drive_root = $Target; asset_map = $MapPath
    options = [ordered]@{ skip_bundle = [bool]$SkipBundle; skip_external = [bool]$SkipExternal; include_raw = [bool]$IncludeRaw; overwrite = [bool]$Overwrite }
    verification = 'relative path + file size (no hashing)'; sources_deleted = $false; assets = @($records)
}
$json = $manifest | ConvertTo-Json -Depth 6
$utf8 = New-Object System.Text.UTF8Encoding $false
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
[System.IO.File]::WriteAllText((Join-Path $manifestDir 'local_to_drive_copy_manifest.json'), $json, $utf8)
[System.IO.File]::WriteAllText((Join-Path $manifestDir "local_to_drive_copy_manifest_$stamp.json"), $json, $utf8)
Write-Host "  written: $(Join-Path $manifestDir 'local_to_drive_copy_manifest.json')"

# -----------------------------------------------------------------------------------------------
Step 'Summary'
$plan | Group-Object status | ForEach-Object { Write-Host ("  {0,-17} {1}" -f $_.Name, $_.Count) }
Write-Host ''
Write-Host 'No source file was deleted or moved. Wait for Google Drive for desktop to finish syncing before using Colab.' -ForegroundColor Cyan
if (@($plan | Where-Object { $_.status -eq 'ERROR' }).Count) { exit 1 }
exit 0
