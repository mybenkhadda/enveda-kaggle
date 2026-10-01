<#
.SYNOPSIS
    Validate the frozen CASMI runtime package and copy it to Google Drive (Drive for desktop) for Colab.

.DESCRIPTION
    Copies ONLY:  <ProjectRoot>\bundle\  (the ENTIRE directory, mirrored, unchanged)
                  <ProjectRoot>\data\test.parquet, <ProjectRoot>\data\sample_submission.csv
    to            <DriveRoot>\bundle\, <DriveRoot>\competition\   (and creates <DriveRoot>\results\, never touched).

    Never copied: src\, tests\, scripts\, notebooks\, outputs\, bundle_archive\, train.parquet, MOL_DEV / HOST
    artifacts, QCR caches, the 11_00 / 11_01 / 11_02 notebooks. Nothing in the local bundle is modified.

    Sequence: local inputs -> frozen identity (config.json) -> manifest completeness (presence; the multi-GB files are
    not hashed -- the Colab/Kaggle runtime hash-verifies every file) -> Drive target -> existing-bundle identity ->
    directories -> robocopy /MIR of the whole bundle -> competition files -> destination verification (every manifest
    file present with the same size; SHA256 of the small identity files) -> summary.

.PARAMETER ProjectRoot
    Training repo root (default C:\Users\myben\OneDrive\Documents\Enveda).

.PARAMETER DriveRoot
    Target folder inside Google Drive for desktop (default "G:\My Drive\EnvedaCASMI").

.PARAMETER Force
    Required to REPLACE a different bundle already on Drive (different CONFIG_HASH / bundle_version, or a non-empty
    bundle folder without config.json). Re-syncing the SAME bundle never needs -Force. results\ is never touched.

.PARAMETER DryRun
    Validate everything and show what robocopy / Copy-Item WOULD do (robocopy /L); Drive is not modified.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\export_runtime_to_drive.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\export_runtime_to_drive.ps1 -DriveRoot "H:\My Drive\EnvedaCASMI"
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\export_runtime_to_drive.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$ProjectRoot = 'C:\Users\myben\OneDrive\Documents\Enveda',
    [string]$DriveRoot = 'G:\My Drive\EnvedaCASMI',
    [switch]$Force,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'

$EXPECTED = [ordered]@{ bundle_version = 'v2-A7'; model_id = 'V1_TL_1K_TESTSIM_STRICT'; aggregator_name = 'MOST_CONFIDENT_SPECTRUM' }

function Fail([string]$Message) {
    Write-Host ''
    Write-Host "EXPORT STOPPED: $Message" -ForegroundColor Red
    exit 1
}
function Step([string]$Message) { Write-Host ''; Write-Host "== $Message" -ForegroundColor Cyan }
function Read-Json([string]$Path) { return (Get-Content -Raw -LiteralPath $Path -Encoding UTF8 | ConvertFrom-Json) }
function Win([string]$Rel) { return $Rel.Replace('/', '\') }

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$DriveRoot = [System.IO.Path]::GetFullPath($DriveRoot)
$SrcBundle = Join-Path $ProjectRoot 'bundle'
$SrcTest = Join-Path $ProjectRoot 'data\test.parquet'
$SrcSample = Join-Path $ProjectRoot 'data\sample_submission.csv'
$DstBundle = Join-Path $DriveRoot 'bundle'
$DstComp = Join-Path $DriveRoot 'competition'
$DstResults = Join-Path $DriveRoot 'results'
if ($DryRun) { Write-Host 'DRY RUN -- validation only; Drive will NOT be modified.' -ForegroundColor Yellow }

# ---------------------------------------------------------------------------------------------------------------
Step '1. local inputs'
$required = @((Join-Path $SrcBundle 'config.json'), (Join-Path $SrcBundle 'manifest.json'), $SrcTest, $SrcSample)
$missingInputs = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missingInputs.Count) { Fail ("missing local input(s):`n  " + ($missingInputs -join "`n  ")) }
$required | ForEach-Object { Write-Host "  OK  $_" }

# ---------------------------------------------------------------------------------------------------------------
Step '2. frozen identity (bundle\config.json + manifest.json; nothing is modified)'
$cfg = Read-Json (Join-Path $SrcBundle 'config.json')
$man = Read-Json (Join-Path $SrcBundle 'manifest.json')
$aggBlockName = if ($cfg.aggregator) { $cfg.aggregator.name } else { $null }
Write-Host ("  bundle_version          {0}" -f $cfg.bundle_version)
Write-Host ("  manifest bundle_version {0}" -f $man.bundle_version)
Write-Host ("  model_id                {0}" -f $cfg.model_id)
Write-Host ("  aggregator_name         {0}   (aggregator.name: {1})" -f $cfg.aggregator_name, $aggBlockName)
Write-Host ("  CONFIG_HASH             {0}   (manifest: {1})" -f $cfg.CONFIG_HASH, $man.CONFIG_HASH)
Write-Host ("  calibration_temperature {0}" -f $cfg.calibration_temperature)
$idProblems = @()
foreach ($k in $EXPECTED.Keys) {
    if ($cfg.$k -ne $EXPECTED[$k]) { $idProblems += "config.$k = '$($cfg.$k)', expected '$($EXPECTED[$k])'" }
}
if ($aggBlockName -ne $cfg.aggregator_name) { $idProblems += "aggregator.name '$aggBlockName' != aggregator_name '$($cfg.aggregator_name)'" }
if (-not $cfg.CONFIG_HASH) { $idProblems += 'config.json has no CONFIG_HASH' }
if ($man.CONFIG_HASH -ne $cfg.CONFIG_HASH) { $idProblems += "manifest CONFIG_HASH '$($man.CONFIG_HASH)' != config CONFIG_HASH '$($cfg.CONFIG_HASH)'" }
if (-not ([string]$man.bundle_version).StartsWith("$($EXPECTED.bundle_version)-")) { $idProblems += "manifest bundle_version '$($man.bundle_version)' is not a $($EXPECTED.bundle_version) export" }
if ($null -eq $cfg.calibration_temperature) { $idProblems += 'config.json has no calibration_temperature (A7 needs the frozen calibration)' }
if ($idProblems.Count) { Fail ("the local bundle is not the frozen runtime bundle:`n  " + ($idProblems -join "`n  ")) }
Write-Host '  frozen identity OK' -ForegroundColor Green

# ---------------------------------------------------------------------------------------------------------------
Step '3. manifest completeness (local bundle; presence only -- the runtime hash-verifies every file)'
if (-not $man.files) { Fail 'manifest.json has no "files" map' }
$manFiles = @($man.files.PSObject.Properties.Name)
$missingLocal = @($manFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $SrcBundle (Win $_)) -PathType Leaf) })
Write-Host ("  manifest files present: {0}/{1}" -f ($manFiles.Count - $missingLocal.Count), $manFiles.Count)
if ($missingLocal.Count) { Fail ("local bundle is incomplete; first missing files:`n  " + (($missingLocal | Select-Object -First 20) -join "`n  ")) }
$srcSizes = @{}
foreach ($f in $manFiles) { $srcSizes[$f] = (Get-Item -LiteralPath (Join-Path $SrcBundle (Win $f))).Length }
$bundleBytes = ($srcSizes.Values | Measure-Object -Sum).Sum
Write-Host ("  bundle size (manifest files): {0:N2} GB" -f ($bundleBytes / 1GB))

# ---------------------------------------------------------------------------------------------------------------
Step '4. Google Drive target'
$qualifier = Split-Path -Qualifier $DriveRoot
$driveParent = Split-Path -Parent $DriveRoot
if (-not (Test-Path -LiteralPath "$qualifier\") -or -not (Test-Path -LiteralPath $driveParent -PathType Container)) {
    Write-Host "  '$driveParent' does not exist. Available filesystem drives:" -ForegroundColor Yellow
    Get-PSDrive -PSProvider FileSystem | Format-Table Name, Root, Description -AutoSize | Out-String | Write-Host
    Fail "Google Drive path not found.`nPass -DriveRoot with the correct Google Drive for Desktop path."
}
Write-Host "  Drive parent OK: $driveParent"
Write-Host "  target:          $DriveRoot"

# ---------------------------------------------------------------------------------------------------------------
Step '5. bundle already on Drive?'
$dstCfgPath = Join-Path $DstBundle 'config.json'
if (Test-Path -LiteralPath $dstCfgPath -PathType Leaf) {
    $dstCfg = Read-Json $dstCfgPath
    Write-Host ("  destination: bundle_version {0} | CONFIG_HASH {1} | model {2}" -f $dstCfg.bundle_version, $dstCfg.CONFIG_HASH, $dstCfg.model_id)
    Write-Host ("  source:      bundle_version {0} | CONFIG_HASH {1} | model {2}" -f $cfg.bundle_version, $cfg.CONFIG_HASH, $cfg.model_id)
    if ($dstCfg.CONFIG_HASH -ne $cfg.CONFIG_HASH -or $dstCfg.bundle_version -ne $cfg.bundle_version) {
        if (-not $Force -and -not $DryRun) {
            Fail "Drive holds a DIFFERENT bundle (CONFIG_HASH $($dstCfg.CONFIG_HASH)). Re-run with -Force to replace it (robocopy /MIR; results\ is untouched)."
        }
        Write-Host '  a DIFFERENT bundle will be replaced (/MIR) -- results\ is not touched' -ForegroundColor Yellow
    } else {
        Write-Host '  same bundle identity -> re-sync (/MIR copies only what differs)'
    }
} elseif ((Test-Path -LiteralPath $DstBundle -PathType Container) -and @(Get-ChildItem -LiteralPath $DstBundle -Force).Count -gt 0) {
    if (-not $Force -and -not $DryRun) { Fail "$DstBundle is not empty but has no config.json. Re-run with -Force to mirror the bundle over it." }
    Write-Host "  $DstBundle is non-empty without config.json -- it will be mirrored over" -ForegroundColor Yellow
} else {
    Write-Host '  no bundle on Drive yet'
}

# ---------------------------------------------------------------------------------------------------------------
Step '6. target directories'
foreach ($d in @($DriveRoot, $DstBundle, $DstComp, $DstResults)) {
    if ($DryRun) { Write-Host "  [dry-run] would ensure $d" }
    else { New-Item -ItemType Directory -Force -Path $d | Out-Null; Write-Host "  OK  $d" }
}

# ---------------------------------------------------------------------------------------------------------------
Step '7. copy the ENTIRE bundle (robocopy /MIR)'
$roboArgs = @($SrcBundle, $DstBundle, '/MIR', '/XD', '__pycache__', '.ipynb_checkpoints', '/XF', '*.pyc', '/R:3', '/W:5', '/NFL', '/NDL', '/NP')
if ($DryRun) { $roboArgs += '/L' }
Write-Host ("  robocopy " + ($roboArgs -join ' '))
& robocopy @roboArgs
$rc = $LASTEXITCODE
if ($rc -ge 8) { Fail "robocopy failed with exit code $rc (>= 8)." }
Write-Host "  robocopy exit code $rc (0-7 = success / non-fatal)" -ForegroundColor Green

# ---------------------------------------------------------------------------------------------------------------
Step '8. competition files'
$compCopies = @(@{ Src = $SrcTest; Dst = (Join-Path $DstComp 'test.parquet') }, @{ Src = $SrcSample; Dst = (Join-Path $DstComp 'sample_submission.csv') })
foreach ($c in $compCopies) {
    if ($DryRun) { Write-Host "  [dry-run] would copy $($c.Src) -> $($c.Dst)" }
    else { Copy-Item -LiteralPath $c.Src -Destination $c.Dst -Force; Write-Host "  copied $($c.Src) -> $($c.Dst)" }
}

if ($DryRun) {
    Write-Host ''
    Write-Host 'DRY RUN COMPLETE -- nothing copied.' -ForegroundColor Yellow
    Write-Host ("Would export bundle {0} | model {1} | aggregator {2} | CONFIG_HASH {3} | {4} manifest files ({5:N2} GB) -> {6}" -f `
        $cfg.bundle_version, $cfg.model_id, $cfg.aggregator_name, $cfg.CONFIG_HASH, $manFiles.Count, ($bundleBytes / 1GB), $DriveRoot)
    exit 0
}

# ---------------------------------------------------------------------------------------------------------------
Step '9. verify the destination'
$dstRequired = @((Join-Path $DstBundle 'config.json'), (Join-Path $DstBundle 'manifest.json'), (Join-Path $DstComp 'test.parquet'), (Join-Path $DstComp 'sample_submission.csv'))
$missingDst = @($dstRequired | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missingDst.Count) { Fail ("missing on Drive after the copy:`n  " + ($missingDst -join "`n  ")) }
$missingDrive = @(); $sizeMismatch = @()
foreach ($f in $manFiles) {
    $p = Join-Path $DstBundle (Win $f)
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { $missingDrive += $f; continue }
    if ((Get-Item -LiteralPath $p).Length -ne $srcSizes[$f]) { $sizeMismatch += $f }
}
Write-Host ("  Drive manifest files present: {0}/{1}" -f ($manFiles.Count - $missingDrive.Count), $manFiles.Count)
if ($missingDrive.Count) { Fail ("the Drive bundle is incomplete; first missing files:`n  " + (($missingDrive | Select-Object -First 20) -join "`n  ")) }
if ($sizeMismatch.Count) { Fail ("size differs on Drive for:`n  " + (($sizeMismatch | Select-Object -First 20) -join "`n  ")) }
Write-Host '  every manifest file present on Drive with the source size' -ForegroundColor Green

# ---------------------------------------------------------------------------------------------------------------
Step '10. source / destination consistency (sizes; SHA256 of the small identity files)'
$pairs = @(
    @{ Name = 'bundle\config.json';                Src = (Join-Path $SrcBundle 'config.json');   Dst = (Join-Path $DstBundle 'config.json');         Hash = $true },
    @{ Name = 'bundle\manifest.json';              Src = (Join-Path $SrcBundle 'manifest.json'); Dst = (Join-Path $DstBundle 'manifest.json');       Hash = $true },
    @{ Name = 'competition\test.parquet';          Src = $SrcTest;                               Dst = (Join-Path $DstComp 'test.parquet');          Hash = $false },
    @{ Name = 'competition\sample_submission.csv'; Src = $SrcSample;                             Dst = (Join-Path $DstComp 'sample_submission.csv'); Hash = $true }
)
$rows = @(); $bad = @()
foreach ($p in $pairs) {
    $s = (Get-Item -LiteralPath $p.Src).Length
    $d = (Get-Item -LiteralPath $p.Dst).Length
    $hashOk = 'n/a'
    if ($p.Hash) { $hashOk = ((Get-FileHash -LiteralPath $p.Src -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $p.Dst -Algorithm SHA256).Hash) }
    $ok = ($s -eq $d) -and ($hashOk -ne $false)
    if (-not $ok) { $bad += $p.Name }
    $rows += [pscustomobject]@{ File = $p.Name; SourceBytes = $s; DestinationBytes = $d; SizeMatch = ($s -eq $d); SHA256Match = $hashOk }
}
$rows | Format-Table -AutoSize | Out-String | Write-Host
if ($bad.Count) { Fail ("source / destination mismatch: " + ($bad -join ', ')) }

# ---------------------------------------------------------------------------------------------------------------
$line = '=' * 52
Write-Host ''
Write-Host $line -ForegroundColor Green
Write-Host 'ENVEDA CASMI RUNTIME EXPORT COMPLETE' -ForegroundColor Green
Write-Host $line -ForegroundColor Green
Write-Host ''
Write-Host "Project:`n$ProjectRoot`n"
Write-Host "Drive:`n$DriveRoot`n"
Write-Host "Bundle:`n$($cfg.bundle_version)   ($($man.bundle_version))`n"
Write-Host "Model:`n$($cfg.model_id)`n"
Write-Host "Aggregator:`n$($cfg.aggregator_name)   (T = $($cfg.calibration_temperature))`n"
Write-Host "CONFIG_HASH:`n$($cfg.CONFIG_HASH)`n"
Write-Host "Bundle manifest:`n$($manFiles.Count)/$($manFiles.Count) files present locally`n$($manFiles.Count)/$($manFiles.Count) files present on Drive (sizes match)`n"
Write-Host "Competition:`ntest.parquet                  OK`nsample_submission.csv         OK`n"
Write-Host "results\ (untouched): $DstResults`n"
Write-Host 'Wait until Google Drive for desktop shows the upload as complete, then:'
Write-Host 'NEXT:'
Write-Host 'Open notebooks/00_colab_setup.ipynb in Colab, then notebooks/01_colab_inference.ipynb.'
Write-Host ''
Write-Host 'Quick check (any time):'
Write-Host ('  Get-Item "{0}", "{1}", "{2}", "{3}" | Select-Object FullName, Length, LastWriteTime' -f `
    (Join-Path $DstBundle 'config.json'), (Join-Path $DstBundle 'manifest.json'), (Join-Path $DstComp 'test.parquet'), (Join-Path $DstComp 'sample_submission.csv'))
exit 0
